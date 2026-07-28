/*
 * viper SwiGLU smoke test — standalone.
 *
 * PURPOSE: out = silu(gate) * up, bf16 in/out, fp32 internal.
 *
 * BUILD:   tests\swiglu_smoke.bat
 * RUN:     .\build\swiglu_smoke.exe
 */
#include "kernels/ops/swiglu_kernel.h"
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#define VIPER_CHECK(call) do {                                                  \
    cudaError_t err = (call);                                                   \
    if (err != cudaSuccess) {                                                   \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,        \
                cudaGetErrorString(err));                                       \
        return 1;                                                               \
    }                                                                           \
} while (0)

int main() {
    constexpr int N = 10752;  // typical MLP intermediate
    constexpr float TOL = 1e-1f;

    int dev = 0;
    cudaDeviceProp prop{};
    VIPER_CHECK(cudaGetDevice(&dev));
    VIPER_CHECK(cudaGetDeviceProperties(&prop, dev));
    printf("Device %d: %s\n", dev, prop.name);

    std::vector<float> h_gate(N), h_up(N);
    std::vector<__nv_bfloat16> h_out(N);
    std::vector<float> h_ref(N);

    std::mt19937 rng(17);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (auto& v : h_gate) v = dist(rng);
    for (auto& v : h_up) v = dist(rng);

    for (int i = 0; i < N; ++i) {
        float g = h_gate[i];
        float u = h_up[i];
        h_ref[i] = (g / (1.0f + std::exp(-g))) * u;
    }

    std::vector<__nv_bfloat16> h_g_bf(N), h_u_bf(N);
    for (int i = 0; i < N; ++i) {
        h_g_bf[i] = __float2bfloat16(h_gate[i]);
        h_u_bf[i] = __float2bfloat16(h_up[i]);
    }

    __nv_bfloat16 *d_gate, *d_up;
    VIPER_CHECK(cudaMalloc(&d_gate, N * sizeof(__nv_bfloat16)));
    VIPER_CHECK(cudaMalloc(&d_up, N * sizeof(__nv_bfloat16)));
    VIPER_CHECK(cudaMemcpy(d_gate, h_g_bf.data(), N * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    VIPER_CHECK(cudaMemcpy(d_up, h_u_bf.data(), N * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));

    VIPER_CHECK(viper::ops::swiglu_inplace_bf16(d_gate, d_up, N, 0));
    VIPER_CHECK(cudaDeviceSynchronize());

    VIPER_CHECK(cudaMemcpy(h_out.data(), d_gate, N * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));

    float max_diff = 0.0f;
    for (int i = 0; i < N; ++i) {
        float gpu_val = __bfloat162float(h_out[i]);
        float diff = std::fabs(gpu_val - h_ref[i]);
        if (diff > max_diff) max_diff = diff;
    }

    cudaFree(d_gate);
    cudaFree(d_up);

    if (max_diff < TOL) {
        printf("[OK  ] SwiGLU smoke test PASS (max_diff=%.2e, tol=%.2e, N=%d)\n", max_diff, TOL, N);
        return 0;
    } else {
        printf("[FAIL] SwiGLU smoke test FAIL (max_diff=%.2e, tol=%.2e)\n", max_diff, TOL);
        return 1;
    }
}
