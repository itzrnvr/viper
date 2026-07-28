/*
 * viper Residual add smoke test — standalone.
 *
 * PURPOSE: out[i] = x[i] + y[i], bf16 in/out, fp32 internal.
 *
 * BUILD:   tests\residual_smoke.bat
 * RUN:     .\build\residual_smoke.exe
 */
#include "kernels/ops/residual_kernel.h"
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
    constexpr int N = 3072;
    constexpr float TOL = 5e-2f;  // bf16-typical

    int dev = 0;
    cudaDeviceProp prop{};
    VIPER_CHECK(cudaGetDevice(&dev));
    VIPER_CHECK(cudaGetDeviceProperties(&prop, dev));
    printf("Device %d: %s\n", dev, prop.name);

    std::vector<float> h_x(N), h_y(N);
    std::vector<__nv_bfloat16> h_out(N);

    std::mt19937 rng(19);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (auto& v : h_x) v = dist(rng);
    for (auto& v : h_y) v = dist(rng);

    std::vector<__nv_bfloat16> h_x_bf(N), h_y_bf(N);
    for (int i = 0; i < N; ++i) {
        h_x_bf[i] = __float2bfloat16(h_x[i]);
        h_y_bf[i] = __float2bfloat16(h_y[i]);
    }

    __nv_bfloat16 *d_x, *d_y, *d_out;
    VIPER_CHECK(cudaMalloc(&d_x, N * sizeof(__nv_bfloat16)));
    VIPER_CHECK(cudaMalloc(&d_y, N * sizeof(__nv_bfloat16)));
    VIPER_CHECK(cudaMalloc(&d_out, N * sizeof(__nv_bfloat16)));
    VIPER_CHECK(cudaMemcpy(d_x, h_x_bf.data(), N * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    VIPER_CHECK(cudaMemcpy(d_y, h_y_bf.data(), N * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));

    VIPER_CHECK(viper::ops::residual_add_bf16(d_x, d_y, d_out, N, 0));
    VIPER_CHECK(cudaDeviceSynchronize());

    VIPER_CHECK(cudaMemcpy(h_out.data(), d_out, N * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));

    float max_diff = 0.0f;
    for (int i = 0; i < N; ++i) {
        float expected = h_x[i] + h_y[i];
        float gpu_val = __bfloat162float(h_out[i]);
        float diff = std::fabs(gpu_val - expected);
        if (diff > max_diff) max_diff = diff;
    }

    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_out);

    if (max_diff < TOL) {
        printf("[OK  ] Residual smoke test PASS (max_diff=%.2e, tol=%.2e, N=%d)\n", max_diff, TOL, N);
        return 0;
    } else {
        printf("[FAIL] Residual smoke test FAIL (max_diff=%.2e, tol=%.2e)\n", max_diff, TOL);
        return 1;
    }
}
