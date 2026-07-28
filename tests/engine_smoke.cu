/*
 * viper engine smoke test — exercises all 5 verified op kernels
 * in a single .exe. Direct kernel calls (no Tensor wrapper; the
 * C++ host wrappers live in src/viper/<op>.cpp and are linked
 * by the full MSVC+CMake build).
 *
 * PURPOSE: One nvcc invocation that runs rmsnorm, rope, embedding,
 *          swiglu, residual on the GPU and confirms each returns
 *          cudaSuccess. This is the "all ops in one binary" sanity
 *          check before the full engine build adds the host wrapper
 *          layer + linear GEMV + SDPA + sampler.
 *
 * BUILD:   tests\engine_smoke.bat
 * RUN:     .\build\engine_smoke.exe
 */
#include "kernels/ops/rmsnorm_kernel.h"
#include "kernels/ops/rope_kernel.h"
#include "kernels/ops/embedding_kernel.h"
#include "kernels/ops/swiglu_kernel.h"
#include "kernels/ops/residual_kernel.h"

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
    constexpr int HEAD_DIM = 128, N_HEADS = 4;
    constexpr float EPS = 1e-5f, THETA = 70000000.0f;

    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    int fails = 0;

    auto make_bf16 = [&](size_t n) {
        std::vector<__nv_bfloat16> v(n);
        for (auto& x : v) x = __float2bfloat16(dist(rng));
        return v;
    };

    // ---- RMSNorm ----
    printf("[1/5] rmsnorm_forward_bf16 ... ");
    {
        auto h_x = make_bf16((size_t)B * T * H);
        auto h_g = make_bf16((size_t)H);
        std::vector<__nv_bfloat16> h_y(B * T * H);
        __nv_bfloat16 *d_x, *d_g, *d_y;
        VIPER_CHECK(cudaMalloc(&d_x, h_x.size() * 2));
        VIPER_CHECK(cudaMalloc(&d_g, h_g.size() * 2));
        VIPER_CHECK(cudaMalloc(&d_y, h_y.size() * 2));
        VIPER_CHECK(cudaMemcpy(d_x, h_x.data(), h_x.size() * 2, cudaMemcpyHostToDevice));
        VIPER_CHECK(cudaMemcpy(d_g, h_g.data(), h_g.size() * 2, cudaMemcpyHostToDevice));
        auto s = viper::ops::rmsnorm_forward_bf16(d_x, d_g, d_y, B * T, H, EPS, 0);
        VIPER_CHECK(cudaDeviceSynchronize());
        printf("%s\n", s == cudaSuccess ? "PASS" : "FAIL");
        if (s != cudaSuccess) ++fails;
        cudaFree(d_x); cudaFree(d_g); cudaFree(d_y);
    }

    // ---- RoPE ----
    printf("[2/5] rope_precompute_cos_sin + rope_apply_inplace_bf16 ... ");
    {
        auto h_q = make_bf16((size_t)B * N_HEADS * T * HEAD_DIM);
        __nv_bfloat16 *d_q, *d_k;
        VIPER_CHECK(cudaMalloc(&d_q, h_q.size() * 2));
        VIPER_CHECK(cudaMalloc(&d_k, h_q.size() * 2));
        VIPER_CHECK(cudaMemcpy(d_q, h_q.data(), h_q.size() * 2, cudaMemcpyHostToDevice));
        VIPER_CHECK(cudaMemcpy(d_k, h_q.data(), h_q.size() * 2, cudaMemcpyHostToDevice));
        float *d_cos, *d_sin;
        VIPER_CHECK(cudaMalloc(&d_cos, (size_t)T * HEAD_DIM * 4));
        VIPER_CHECK(cudaMalloc(&d_sin, (size_t)T * HEAD_DIM * 4));
        auto s1 = viper::ops::rope_precompute_cos_sin(d_cos, d_sin, 0, T, THETA, HEAD_DIM, 0);
        auto s2 = viper::ops::rope_apply_inplace_bf16(d_q, d_k, d_cos, d_sin, B, N_HEADS, N_HEADS, T, HEAD_DIM, 0);
        VIPER_CHECK(cudaDeviceSynchronize());
        printf("%s\n", (s1 == cudaSuccess && s2 == cudaSuccess) ? "PASS" : "FAIL");
        if (s1 != cudaSuccess || s2 != cudaSuccess) ++fails;
        cudaFree(d_q); cudaFree(d_k); cudaFree(d_cos); cudaFree(d_sin);
    }

    // ---- Embedding ----
    printf("[3/5] embedding_gather_bf16_i32 ... ");
    {
        auto h_table = make_bf16((size_t)V * H);
        std::vector<int32_t> h_ids(B * T);
        for (auto& v : h_ids) v = rng() % V;
        std::vector<__nv_bfloat16> h_y(B * T * H);
        __nv_bfloat16 *d_table, *d_y;
        int32_t* d_ids;
        VIPER_CHECK(cudaMalloc(&d_table, h_table.size() * 2));
        VIPER_CHECK(cudaMalloc(&d_ids, h_ids.size() * 4));
        VIPER_CHECK(cudaMalloc(&d_y, h_y.size() * 2));
        VIPER_CHECK(cudaMemcpy(d_table, h_table.data(), h_table.size() * 2, cudaMemcpyHostToDevice));
        VIPER_CHECK(cudaMemcpy(d_ids, h_ids.data(), h_ids.size() * 4, cudaMemcpyHostToDevice));
        auto s = viper::ops::embedding_gather_bf16_i32(d_table, d_ids, d_y, B, T, V, H, 0);
        VIPER_CHECK(cudaDeviceSynchronize());
        printf("%s\n", s == cudaSuccess ? "PASS" : "FAIL");
        if (s != cudaSuccess) ++fails;
        cudaFree(d_table); cudaFree(d_ids); cudaFree(d_y);
    }

    // ---- SwiGLU ----
    printf("[4/5] swiglu_out_of_place_bf16 ... ");
    {
        auto h_g = make_bf16((size_t)I);
        auto h_u = make_bf16((size_t)I);
        std::vector<__nv_bfloat16> h_y(I);
        __nv_bfloat16 *d_g, *d_u, *d_y;
        VIPER_CHECK(cudaMalloc(&d_g, I * 2));
        VIPER_CHECK(cudaMalloc(&d_u, I * 2));
        VIPER_CHECK(cudaMalloc(&d_y, I * 2));
        VIPER_CHECK(cudaMemcpy(d_g, h_g.data(), I * 2, cudaMemcpyHostToDevice));
        VIPER_CHECK(cudaMemcpy(d_u, h_u.data(), I * 2, cudaMemcpyHostToDevice));
        auto s = viper::ops::swiglu_out_of_place_bf16(d_g, d_u, d_y, I, 0);
        VIPER_CHECK(cudaDeviceSynchronize());
        printf("%s\n", s == cudaSuccess ? "PASS" : "FAIL");
        if (s != cudaSuccess) ++fails;
        cudaFree(d_g); cudaFree(d_u); cudaFree(d_y);
    }

    // ---- Residual ----
    printf("[5/5] residual_add_bf16 ... ");
    {
        auto h_x = make_bf16((size_t)H);
        auto h_r = make_bf16((size_t)H);
        std::vector<__nv_bfloat16> h_y(H);
        __nv_bfloat16 *d_x, *d_r, *d_y;
        VIPER_CHECK(cudaMalloc(&d_x, H * 2));
        VIPER_CHECK(cudaMalloc(&d_r, H * 2));
        VIPER_CHECK(cudaMalloc(&d_y, H * 2));
        VIPER_CHECK(cudaMemcpy(d_x, h_x.data(), H * 2, cudaMemcpyHostToDevice));
        VIPER_CHECK(cudaMemcpy(d_r, h_r.data(), H * 2, cudaMemcpyHostToDevice));
        auto s = viper::ops::residual_add_bf16(d_x, d_r, d_y, H, 0);
        VIPER_CHECK(cudaDeviceSynchronize());
        printf("%s\n", s == cudaSuccess ? "PASS" : "FAIL");
        if (s != cudaSuccess) ++fails;
        cudaFree(d_x); cudaFree(d_r); cudaFree(d_y);
    }

    if (fails == 0) {
        printf("\n[OK  ] All 5 op kernels run cleanly in a single .exe on the RTX 3070 Ti.\n");
        return 0;
    } else {
        printf("\n[FAIL] %d/5 op kernels failed.\n", fails);
        return 1;
    }
}
