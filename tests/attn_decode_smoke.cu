// attn_decode smoke: GPU chunked online-softmax attention vs CPU reference.
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>
#include "kernels/ops/attn_decode_kernel.h"

#define CK(call) do { cudaError_t e=(call); if(e){printf("cuda err %s @%d\n",cudaGetErrorString(e),__LINE__);return 1;} } while(0)

int main() {
    const int nQ = 48, nKV = 8, D = 128, T = 517;  // odd T to test chunk tail
    const float scale = 1.0f / sqrtf((float)D);
    std::mt19937 rng(5);
    std::normal_distribution<float> nd(0.f, 1.f);

    std::vector<float> q(nQ * D), k(T * nKV * D), v(T * nKV * D);
    for (auto& x : q) x = nd(rng);
    for (auto& x : k) x = nd(rng);
    for (auto& x : v) x = nd(rng);

    // CPU reference: per Q head, full softmax attention over KV (fp32).
    std::vector<float> ref(nQ * D, 0.f);
    for (int h = 0; h < nQ; ++h) {
        const int hk = (int)((long long)h * nKV / nQ);
        float mx = -1e30f;
        std::vector<float> d(T);
        for (int t = 0; t < T; ++t) {
            float s = 0.f;
            for (int i = 0; i < D; ++i)
                s += q[h * D + i] * k[((size_t)t * nKV + hk) * D + i];
            d[t] = s * scale;
            mx = fmaxf(mx, d[t]);
        }
        float l = 0.f;
        for (int t = 0; t < T; ++t) { d[t] = expf(d[t] - mx); l += d[t]; }
        for (int i = 0; i < D; ++i) {
            float a = 0.f;
            for (int t = 0; t < T; ++t)
                a += d[t] * v[((size_t)t * nKV + hk) * D + i];
            ref[h * D + i] = a / l;
        }
    }

    __nv_bfloat16 *dq, *dk, *dv, *do_;
    CK(cudaMalloc(&dq, q.size() * 2));
    CK(cudaMalloc(&dk, k.size() * 2));
    CK(cudaMalloc(&dv, v.size() * 2));
    CK(cudaMalloc(&do_, q.size() * 2));
    auto to_bf = [](const std::vector<float>& src, __nv_bfloat16* dst) {
        std::vector<__nv_bfloat16> tmp(src.size());
        for (size_t i = 0; i < src.size(); ++i) tmp[i] = __float2bfloat16(src[i]);
        cudaMemcpy(dst, tmp.data(), tmp.size() * 2, cudaMemcpyHostToDevice);
    };
    to_bf(q, dq); to_bf(k, dk); to_bf(v, dv);

    CK(viper::ops::attn_decode_bf16(dq, dk, dv, do_, nQ, nKV, D, T, scale, 0));
    CK(cudaDeviceSynchronize());

    std::vector<__nv_bfloat16> ho(q.size());
    CK(cudaMemcpy(ho.data(), do_, ho.size() * 2, cudaMemcpyDeviceToHost));

    float max_rel = 0.f;
    for (size_t i = 0; i < ref.size(); ++i) {
        float g = __bfloat162float(ho[i]);
        float rel = fabsf(g - ref[i]) / fmaxf(fabsf(ref[i]), 1.0f);
        max_rel = fmaxf(max_rel, rel);
    }
    cudaFree(dq); cudaFree(dk); cudaFree(dv); cudaFree(do_);
    printf("attn_decode smoke: max_rel=%.4e\n", max_rel);
    if (max_rel < 5e-2f) { printf("[PASS] attn_decode\n"); return 0; }
    printf("[FAIL] attn_decode\n");
    return 1;
}
