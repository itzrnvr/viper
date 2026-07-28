/*
 * viper engine smoke v2 — exercises 8 op kernels + sampling in one .exe.
 *
 *   rmsnorm, rope, embedding, swiglu, residual, linear (Q4_G64),
 *   sdpa, sampling (greedy).
 *
 * BUILD:   tests\engine_smoke.bat
 * RUN:     .\build\engine_smoke.exe
 */
#include "kernels/ops/rmsnorm_kernel.h"
#include "kernels/ops/rope_kernel.h"
#include "kernels/ops/embedding_kernel.h"
#include "kernels/ops/swiglu_kernel.h"
#include "kernels/ops/residual_kernel.h"
#include "kernels/ops/linear_kernel.h"
#include "kernels/ops/sdpa_kernel.h"
#include "kernels/ops/sampling_kernel.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#define VIPER_CHECK(call) do {                                                  \
    cudaError_t _e = (call);                                                    \
    if (_e != cudaSuccess) {                                                    \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,        \
                cudaGetErrorString(_e));                                       \
        return 1;                                                               \
    }                                                                           \
} while (0)

int main() {
    int dev = 0;
    cudaDeviceProp prop{};
    VIPER_CHECK(cudaGetDevice(&dev));
    VIPER_CHECK(cudaGetDeviceProperties(&prop, dev));
    printf("Device %d: %s (sm_%d%d, %d SMs, %.1f GB)\n",
           dev, prop.name, prop.major, prop.minor, prop.multiProcessorCount,
           prop.totalGlobalMem / 1e9);

    constexpr int H = 3072, I = 10752, V = 4096, B = 1, T = 4;
    constexpr int HEAD_DIM = 128, N_HEADS = 4, N_KV_HEADS = 2;
    constexpr float EPS = 1e-5f, THETA = 70000000.0f;
    int fails = 0;

    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    auto make_bf16 = [&](size_t n) {
        std::vector<__nv_bfloat16> v(n);
        for (auto& x : v) x = __float2bfloat16(dist(rng));
        return v;
    };

    // ---- rmsnorm ----
    printf("[1/8] rmsnorm_forward_bf16 ... ");
    {
        auto h_x = make_bf16((size_t)B * T * H);
        auto h_g = make_bf16((size_t)H);
        __nv_bfloat16 *d_x, *d_g, *d_y;
        cudaMalloc(&d_x, h_x.size() * 2); cudaMalloc(&d_g, h_g.size() * 2);
        cudaMalloc(&d_y, B * T * H * 2);
        cudaMemcpy(d_x, h_x.data(), h_x.size() * 2, cudaMemcpyHostToDevice);
        cudaMemcpy(d_g, h_g.data(), h_g.size() * 2, cudaMemcpyHostToDevice);
        auto s = viper::ops::rmsnorm_forward_bf16(d_x, d_g, d_y, B * T, H, EPS, 0);
        cudaDeviceSynchronize();
        printf("%s\n", s == cudaSuccess ? "PASS" : "FAIL");
        if (s != cudaSuccess) ++fails;
        cudaFree(d_x); cudaFree(d_g); cudaFree(d_y);
    }

    // ---- rope ----
    printf("[2/8] rope ... ");
    {
        auto h_q = make_bf16((size_t)B * N_HEADS * T * HEAD_DIM);
        __nv_bfloat16 *d_q, *d_k;
        float *d_cos, *d_sin;
        cudaMalloc(&d_q, h_q.size() * 2); cudaMalloc(&d_k, h_q.size() * 2);
        cudaMemcpy(d_q, h_q.data(), h_q.size() * 2, cudaMemcpyHostToDevice);
        cudaMemcpy(d_k, h_q.data(), h_q.size() * 2, cudaMemcpyHostToDevice);
        cudaMalloc(&d_cos, (size_t)T * HEAD_DIM * 4);
        cudaMalloc(&d_sin, (size_t)T * HEAD_DIM * 4);
        auto s1 = viper::ops::rope_precompute_cos_sin(d_cos, d_sin, 0, T, THETA, HEAD_DIM, 0);
        auto s2 = viper::ops::rope_apply_inplace_bf16(d_q, d_k, d_cos, d_sin, B, N_HEADS, N_HEADS, T, HEAD_DIM, 0);
        cudaDeviceSynchronize();
        printf("%s\n", (s1 == cudaSuccess && s2 == cudaSuccess) ? "PASS" : "FAIL");
        if (s1 != cudaSuccess || s2 != cudaSuccess) ++fails;
        cudaFree(d_q); cudaFree(d_k); cudaFree(d_cos); cudaFree(d_sin);
    }

    // ---- embedding ----
    printf("[3/8] embedding_gather_bf16_i32 ... ");
    {
        auto h_table = make_bf16((size_t)V * H);
        std::vector<int32_t> h_ids(B * T);
        for (auto& v : h_ids) v = rng() % V;
        __nv_bfloat16 *d_table, *d_y; int32_t* d_ids;
        cudaMalloc(&d_table, h_table.size() * 2);
        cudaMalloc(&d_ids, h_ids.size() * 4);
        cudaMalloc(&d_y, B * T * H * 2);
        cudaMemcpy(d_table, h_table.data(), h_table.size() * 2, cudaMemcpyHostToDevice);
        cudaMemcpy(d_ids, h_ids.data(), h_ids.size() * 4, cudaMemcpyHostToDevice);
        auto s = viper::ops::embedding_gather_bf16_i32(d_table, d_ids, d_y, B, T, V, H, 0);
        cudaDeviceSynchronize();
        printf("%s\n", s == cudaSuccess ? "PASS" : "FAIL");
        if (s != cudaSuccess) ++fails;
        cudaFree(d_table); cudaFree(d_ids); cudaFree(d_y);
    }

    // ---- swiglu ----
    printf("[4/8] swiglu_out_of_place_bf16 ... ");
    {
        auto h_g = make_bf16((size_t)I);
        auto h_u = make_bf16((size_t)I);
        __nv_bfloat16 *d_g, *d_u, *d_y;
        cudaMalloc(&d_g, I * 2); cudaMalloc(&d_u, I * 2);
        cudaMalloc(&d_y, I * 2);
        cudaMemcpy(d_g, h_g.data(), I * 2, cudaMemcpyHostToDevice);
        cudaMemcpy(d_u, h_u.data(), I * 2, cudaMemcpyHostToDevice);
        auto s = viper::ops::swiglu_out_of_place_bf16(d_g, d_u, d_y, I, 0);
        cudaDeviceSynchronize();
        printf("%s\n", s == cudaSuccess ? "PASS" : "FAIL");
        if (s != cudaSuccess) ++fails;
        cudaFree(d_g); cudaFree(d_u); cudaFree(d_y);
    }

    // ---- residual ----
    printf("[5/8] residual_add_bf16 ... ");
    {
        auto h_x = make_bf16((size_t)H);
        auto h_r = make_bf16((size_t)H);
        __nv_bfloat16 *d_x, *d_r, *d_y;
        cudaMalloc(&d_x, H * 2); cudaMalloc(&d_r, H * 2);
        cudaMalloc(&d_y, H * 2);
        cudaMemcpy(d_x, h_x.data(), H * 2, cudaMemcpyHostToDevice);
        cudaMemcpy(d_r, h_r.data(), H * 2, cudaMemcpyHostToDevice);
        auto s = viper::ops::residual_add_bf16(d_x, d_r, d_y, H, 0);
        cudaDeviceSynchronize();
        printf("%s\n", s == cudaSuccess ? "PASS" : "FAIL");
        if (s != cudaSuccess) ++fails;
        cudaFree(d_x); cudaFree(d_r); cudaFree(d_y);
    }

    // ---- linear (Q4_G64) ----
    printf("[6/8] linear_q4_g64_bf16 ... ");
    {
        constexpr int M = 4, N = 256, K = 256;
        std::vector<float> h_w_orig(N * K);
        std::vector<__nv_bfloat16> h_x(M * K);
        for (auto& v : h_w_orig) v = dist(rng);
        for (auto& v : h_x) v = __float2bfloat16(dist(rng));
        std::vector<uint8_t> h_w_packed(N * K / 2, 0);
        std::vector<__nv_bfloat16> h_w_scales(N * K / 64, 0.0f);
        for (int n = 0; n < N; ++n) {
            for (int g = 0; g < K / 64; ++g) {
                float max_abs = 0.0f;
                for (int j = 0; j < 64; ++j) {
                    max_abs = std::fmax(max_abs, std::fabs(h_w_orig[n * K + g * 64 + j]));
                }
                float scale = max_abs / 7.0f;
                if (scale < 1e-8f) scale = 1e-8f;
                uint16_t sb; std::memcpy(&sb, &scale, 2);
                // bf16 of scale: take the high 16 bits of the float32 bit pattern.
                float sf; std::memcpy(&sf, &scale, 4);
                uint16_t sfb; std::memcpy(&sfb, &sf, 4); sfb = sfb >> 16;
                h_w_scales[n * (K / 64) + g] = *reinterpret_cast<__nv_bfloat16*>(&sfb);
                for (int j = 0; j < 64; ++j) {
                    float w = h_w_orig[n * K + g * 64 + j];
                    int stored = (int)std::round(w / scale) + 8;
                    if (stored < 0) stored = 0; if (stored > 15) stored = 15;
                    int byte_idx = (g * 64 + j) / 2;
                    if (j % 2 == 0) h_w_packed[n * (K / 2) + byte_idx] |= (uint8_t)(stored & 0x0F);
                    else h_w_packed[n * (K / 2) + byte_idx] |= (uint8_t)((stored & 0x0F) << 4);
                }
            }
        }
        uint8_t *d_w; __nv_bfloat16 *d_s, *d_x, *d_y;
        std::vector<__nv_bfloat16> h_y(M * N);
        cudaMalloc(&d_w, h_w_packed.size());
        cudaMalloc(&d_s, h_w_scales.size() * 2);
        cudaMalloc(&d_x, h_x.size() * 2);
        cudaMalloc(&d_y, h_y.size() * 2);
        cudaMemcpy(d_w, h_w_packed.data(), h_w_packed.size(), cudaMemcpyHostToDevice);
        cudaMemcpy(d_s, h_w_scales.data(), h_w_scales.size() * 2, cudaMemcpyHostToDevice);
        cudaMemcpy(d_x, h_x.data(), h_x.size() * 2, cudaMemcpyHostToDevice);
        auto s = viper::ops::linear_q4_g64_bf16(d_w, d_s, d_x, d_y, M, N, K, 0);
        cudaDeviceSynchronize();
        printf("%s\n", s == cudaSuccess ? "PASS" : "FAIL");
        if (s != cudaSuccess) ++fails;
        cudaFree(d_w); cudaFree(d_s); cudaFree(d_x); cudaFree(d_y);
    }

    // ---- sdpa ----
    printf("[7/8] sdpa_forward_bf16 ... ");
    {
        // T_q=1 (decode), T_k=8 (small cache for smoke test).
        auto h_q = make_bf16((size_t)B * N_HEADS * HEAD_DIM);
        auto h_k = make_bf16((size_t)B * N_KV_HEADS * 8 * HEAD_DIM);
        auto h_v = make_bf16((size_t)B * N_KV_HEADS * 8 * HEAD_DIM);
        __nv_bfloat16 *d_q, *d_k, *d_v, *d_o;
        cudaMalloc(&d_q, h_q.size() * 2);
        cudaMalloc(&d_k, h_k.size() * 2);
        cudaMalloc(&d_v, h_v.size() * 2);
        cudaMalloc(&d_o, h_q.size() * 2);
        cudaMemcpy(d_q, h_q.data(), h_q.size() * 2, cudaMemcpyHostToDevice);
        cudaMemcpy(d_k, h_k.data(), h_k.size() * 2, cudaMemcpyHostToDevice);
        cudaMemcpy(d_v, h_v.data(), h_v.size() * 2, cudaMemcpyHostToDevice);
        float scale = 1.0f / std::sqrt((float)HEAD_DIM);
        auto s = viper::ops::sdpa_forward_bf16(d_q, d_k, d_v, d_o,
            B, N_HEADS, N_KV_HEADS, 1, 8, HEAD_DIM, scale, false, 0);
        cudaDeviceSynchronize();
        printf("%s\n", s == cudaSuccess ? "PASS" : "FAIL");
        if (s != cudaSuccess) ++fails;
        cudaFree(d_q); cudaFree(d_k); cudaFree(d_v); cudaFree(d_o);
    }

    // ---- sampling (greedy argmax) ----
    printf("[8/8] sampling_greedy_bf16 ... ");
    {
        // Random logits, then verify the chosen argmax matches CPU.
        std::vector<__nv_bfloat16> h_logits(V);
        for (auto& v : h_logits) v = __float2bfloat16(dist(rng));
        int expected_argmax = 0; float expected_max = -1e30f;
        for (int i = 0; i < V; ++i) {
            float f = __bfloat162float(h_logits[i]);
            if (f > expected_max) { expected_max = f; expected_argmax = i; }
        }
        __nv_bfloat16* d_logits; int32_t* d_out;
        cudaMalloc(&d_logits, V * 2);
        cudaMalloc(&d_out, 4);
        cudaMemcpy(d_logits, h_logits.data(), V * 2, cudaMemcpyHostToDevice);
        auto s = viper::ops::sampling_greedy_bf16(d_logits, d_out, 1, V, 0);
        cudaDeviceSynchronize();
        int32_t h_out;
        cudaMemcpy(&h_out, d_out, 4, cudaMemcpyDeviceToHost);
        bool ok = (s == cudaSuccess) && (h_out == expected_argmax);
        printf("%s (got=%d expected=%d)\n", ok ? "PASS" : "FAIL", h_out, expected_argmax);
        if (!ok) ++fails;
        cudaFree(d_logits); cudaFree(d_out);
    }

    if (fails == 0) {
        printf("\n[OK  ] 8 op kernels + sampling all run cleanly on the RTX 3070 Ti.\n");
        return 0;
    } else {
        printf("\n[FAIL] %d/9 ops failed.\n", fails);
        return 1;
    }
}
