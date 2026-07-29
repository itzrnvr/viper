// Benchmark: measure actual GEMV bandwidth achieved.
#include <cstdio>
#include <chrono>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include "ops/linear_kernel.h"

int main() {
    // Simulate q_proj: N=6144, K=3072
    const int N = 6144, K = 3072;
    size_t packed_sz = (size_t)N * K / 2;
    size_t scales_sz = (size_t)N * (K / 64) * sizeof(__nv_bfloat16);
    size_t act_sz = K * sizeof(__nv_bfloat16);
    size_t out_sz = N * sizeof(__nv_bfloat16);

    uint8_t* d_packed; cudaMalloc(&d_packed, packed_sz);
    __nv_bfloat16* d_scales; cudaMalloc(&d_scales, scales_sz);
    __nv_bfloat16* d_x; cudaMalloc(&d_x, act_sz);
    __nv_bfloat16* d_y; cudaMalloc(&d_y, out_sz);
    cudaMemset(d_packed, 0x88, packed_sz);
    cudaMemset(d_scales, 0, scales_sz);
    cudaMemset(d_x, 0x3C, act_sz);

    // Warmup
    for (int i = 0; i < 10; i++)
        viper::ops::linear_q4_g64_bf16(d_packed, d_scales, d_x, d_y, 1, N, K, 0);
    cudaDeviceSynchronize();

    // Measure
    int reps = 1000;
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < reps; i++)
        viper::ops::linear_q4_g64_bf16(d_packed, d_scales, d_x, d_y, 1, N, K, 0);
    cudaDeviceSynchronize();
    auto t1 = std::chrono::steady_clock::now();
    double us = std::chrono::duration<double, std::micro>(t1 - t0).count() / reps;

    // Bandwidth: packed + scales + activation read
    double total_bytes = packed_sz + scales_sz + act_sz;
    double achieved_gbs = total_bytes / (us * 1e3); // GB/s
    printf("q_proj (N=%d, K=%d): %.1f us/launch\n", N, K, us);
    printf("  Data: %.2f MB, Achieved: %.1f GB/s (%.0f%% of 448 peak)\n",
           total_bytes / 1e6, achieved_gbs, achieved_gbs / 448 * 100);

    // Now test with rmsnorm fusion
    __nv_bfloat16* d_gamma; cudaMalloc(&d_gamma, K * sizeof(__nv_bfloat16));
    cudaMemset(d_gamma, 0x3C, K * sizeof(__nv_bfloat16));

    for (int i = 0; i < 10; i++)
        viper::ops::linear_q4_g64_bf16_rmsnorm(d_packed, d_scales, d_gamma, 1e-5f, d_x, d_y, 1, N, K, 0);
    cudaDeviceSynchronize();

    t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < reps; i++)
        viper::ops::linear_q4_g64_bf16_rmsnorm(d_packed, d_scales, d_gamma, 1e-5f, d_x, d_y, 1, N, K, 0);
    cudaDeviceSynchronize();
    t1 = std::chrono::steady_clock::now();
    us = std::chrono::duration<double, std::micro>(t1 - t0).count() / reps;
    achieved_gbs = total_bytes / (us * 1e3);
    printf("q_proj+rmsnorm: %.1f us/launch, %.1f GB/s (%.0f%%)\n",
           us, achieved_gbs, achieved_gbs / 448 * 100);

    // Test down_proj (largest): N=3072, K=10752
    const int N2 = 3072, K2 = 10752;
    size_t p2 = (size_t)N2 * K2 / 2;
    size_t s2 = (size_t)N2 * (K2 / 64) * sizeof(__nv_bfloat16);
    size_t a2 = K2 * sizeof(__nv_bfloat16);
    uint8_t* d_p2; cudaMalloc(&d_p2, p2);
    __nv_bfloat16* d_s2; cudaMalloc(&d_s2, s2);
    __nv_bfloat16* d_a2; cudaMalloc(&d_a2, a2);
    __nv_bfloat16* d_y2; cudaMalloc(&d_y2, N2 * sizeof(__nv_bfloat16));
    cudaMemset(d_p2, 0x88, p2);
    cudaMemset(d_s2, 0, s2);
    cudaMemset(d_a2, 0x3C, a2);

    for (int i = 0; i < 10; i++)
        viper::ops::linear_q4_g64_bf16(d_p2, d_s2, d_a2, d_y2, 1, N2, K2, 0);
    cudaDeviceSynchronize();
    t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < reps; i++)
        viper::ops::linear_q4_g64_bf16(d_p2, d_s2, d_a2, d_y2, 1, N2, K2, 0);
    cudaDeviceSynchronize();
    t1 = std::chrono::steady_clock::now();
    us = std::chrono::duration<double, std::micro>(t1 - t0).count() / reps;
    total_bytes = p2 + s2 + a2;
    achieved_gbs = total_bytes / (us * 1e3);
    printf("down_proj (N=%d, K=%d): %.1f us/launch, %.1f GB/s (%.0f%%)\n",
           N2, K2, us, achieved_gbs, achieved_gbs / 448 * 100);

    cudaFree(d_packed); cudaFree(d_scales); cudaFree(d_x); cudaFree(d_y); cudaFree(d_gamma);
    cudaFree(d_p2); cudaFree(d_s2); cudaFree(d_a2); cudaFree(d_y2);
    return 0;
}
