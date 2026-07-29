// Benchmark: fused DP4A vs scalar Q4 GEMV
#include <cstdio>
#include <chrono>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "../kernels/ops/linear_kernel.cu"
#include "../kernels/ops/fused_dp4a_kernel.cu"

int main() {
    const int N = 6144, K = 3072;
    const int M = 1;

    size_t w_sz = (size_t)N * K / 2;
    size_t ws_sz = (size_t)N * (K/64) * sizeof(__nv_bfloat16);
    size_t x_sz = K * sizeof(__nv_bfloat16);
    size_t y_sz = N * sizeof(__nv_bfloat16);

    uint8_t* d_w; cudaMalloc(&d_w, w_sz);
    __nv_bfloat16* d_ws; cudaMalloc(&d_ws, ws_sz);
    __nv_bfloat16* d_x; cudaMalloc(&d_x, x_sz);
    __nv_bfloat16* d_y; cudaMalloc(&d_y, y_sz);
    __nv_bfloat16* d_gamma; cudaMalloc(&d_gamma, x_sz);

    srand(42);
    uint8_t* h_w = (uint8_t*)malloc(w_sz);
    __nv_bfloat16* h_x = (__nv_bfloat16*)malloc(x_sz);
    __nv_bfloat16* h_ws = (__nv_bfloat16*)malloc(ws_sz);
    __nv_bfloat16* h_g = (__nv_bfloat16*)malloc(x_sz);
    for (size_t i = 0; i < w_sz; ++i) h_w[i] = rand() & 0xFF;
    for (int i = 0; i < N*(K/64); ++i) h_ws[i] = __float2bfloat16(0.5f + 0.001f*(rand()%1000));
    for (int i = 0; i < K; ++i) h_x[i] = __float2bfloat16((rand()%200-100)/200.0f);
    for (int i = 0; i < K; ++i) h_g[i] = __float2bfloat16(0.5f + (rand()%200)/200.0f);
    cudaMemcpy(d_w, h_w, w_sz, cudaMemcpyHostToDevice);
    cudaMemcpy(d_ws, h_ws, ws_sz, cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, h_x, x_sz, cudaMemcpyHostToDevice);
    cudaMemcpy(d_gamma, h_g, x_sz, cudaMemcpyHostToDevice);

    int reps = 1000;
    int grid = (N+7)/8;

    // --- Scalar Q4 rmsnorm GEMV ---
    for (int i = 0; i < 10; i++)
        viper::ops::linear_q4_g64_bf16_rmsnorm(d_w, d_ws, d_gamma, 1e-5f, d_x, d_y, M, N, K, 0);
    cudaDeviceSynchronize();
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < reps; i++)
        viper::ops::linear_q4_g64_bf16_rmsnorm(d_w, d_ws, d_gamma, 1e-5f, d_x, d_y, M, N, K, 0);
    cudaDeviceSynchronize();
    auto t1 = std::chrono::steady_clock::now();
    double scalar_us = std::chrono::duration<double, std::micro>(t1-t0).count() / reps;

    // --- Fused DP4A rmsnorm GEMV ---
    for (int i = 0; i < 10; i++)
        viper::ops::fused_dp4a_rmsnorm(d_w, d_ws, d_gamma, 1e-5f, d_x, d_y, M, N, K, 0);
    cudaDeviceSynchronize();
    t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < reps; i++)
        viper::ops::fused_dp4a_rmsnorm(d_w, d_ws, d_gamma, 1e-5f, d_x, d_y, M, N, K, 0);
    cudaDeviceSynchronize();
    t1 = std::chrono::steady_clock::now();
    double dp4a_us = std::chrono::duration<double, std::micro>(t1-t0).count() / reps;

    double scalar_bw = (double)w_sz / (scalar_us * 1e3);
    double dp4a_bw = (double)w_sz / (dp4a_us * 1e3);

    printf("=== q_proj (N=%d K=%d) rmsnorm+fused ===\n", N, K);
    printf("  Scalar:  %.1f us, %.1f GB/s (%.0f%%)\n", scalar_us, scalar_bw, scalar_bw/448*100);
    printf("  DP4A:    %.1f us, %.1f GB/s (%.0f%%)\n", dp4a_us, dp4a_bw, dp4a_bw/448*100);
    printf("  Speedup: %.2fx\n", scalar_us / dp4a_us);

    // down_proj
    const int N2 = 3072, K2 = 10752;
    size_t w2 = (size_t)N2 * K2 / 2;
    size_t ws2 = (size_t)N2*(K2/64)*sizeof(__nv_bfloat16);
    size_t x2 = K2*sizeof(__nv_bfloat16);
    uint8_t* d_w2; cudaMalloc(&d_w2, w2);
    __nv_bfloat16* d_ws2; cudaMalloc(&d_ws2, ws2);
    __nv_bfloat16* d_x2; cudaMalloc(&d_x2, x2);
    __nv_bfloat16* d_y2; cudaMalloc(&d_y2, N2*sizeof(__nv_bfloat16));
    __nv_bfloat16* d_g2; cudaMalloc(&d_g2, x2);
    cudaMemset(d_w2, 0x88, w2);
    cudaMemset(d_ws2, 0x3C00, ws2);
    cudaMemset(d_x2, 0x3C00, x2);
    cudaMemset(d_g2, 0x3C00, x2);

    for (int i = 0; i < 10; i++)
        viper::ops::linear_q4_g64_bf16_rmsnorm(d_w2, d_ws2, d_g2, 1e-5f, d_x2, d_y2, M, N2, K2, 0);
    cudaDeviceSynchronize();
    t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < reps; i++)
        viper::ops::linear_q4_g64_bf16_rmsnorm(d_w2, d_ws2, d_g2, 1e-5f, d_x2, d_y2, M, N2, K2, 0);
    cudaDeviceSynchronize();
    t1 = std::chrono::steady_clock::now();
    scalar_us = std::chrono::duration<double, std::micro>(t1-t0).count() / reps;

    for (int i = 0; i < 10; i++)
        viper::ops::fused_dp4a_rmsnorm(d_w2, d_ws2, d_g2, 1e-5f, d_x2, d_y2, M, N2, K2, 0);
    cudaDeviceSynchronize();
    t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < reps; i++)
        viper::ops::fused_dp4a_rmsnorm(d_w2, d_ws2, d_g2, 1e-5f, d_x2, d_y2, M, N2, K2, 0);
    cudaDeviceSynchronize();
    t1 = std::chrono::steady_clock::now();
    dp4a_us = std::chrono::duration<double, std::micro>(t1-t0).count() / reps;

    scalar_bw = (double)w2 / (scalar_us * 1e3);
    dp4a_bw = (double)w2 / (dp4a_us * 1e3);

    printf("\n=== down_proj (N=%d K=%d) rmsnorm+fused ===\n", N2, K2);
    printf("  Scalar:  %.1f us, %.1f GB/s (%.0f%%)\n", scalar_us, scalar_bw, scalar_bw/448*100);
    printf("  DP4A:    %.1f us, %.1f GB/s (%.0f%%)\n", dp4a_us, dp4a_bw, dp4a_bw/448*100);
    printf("  Speedup: %.2fx\n", scalar_us / dp4a_us);

    free(h_w); free(h_x); free(h_ws); free(h_g);
    cudaFree(d_w); cudaFree(d_ws); cudaFree(d_x); cudaFree(d_y); cudaFree(d_gamma);
    cudaFree(d_w2); cudaFree(d_ws2); cudaFree(d_x2); cudaFree(d_y2); cudaFree(d_g2);
    return 0;
}
