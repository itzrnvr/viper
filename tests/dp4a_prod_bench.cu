// Benchmark + correctness test for production DP4A GEMV kernel
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

// Include the kernel directly
#include "../kernels/ops/dp4a_q4_kernel.cu"

// Reference: scalar Q4 GEMV on CPU
float scalar_q4_gemv(const uint8_t* w_packed, const __nv_bfloat16* w_scales,
                     const __nv_bfloat16* x, int N, int K, int n) {
    const uint8_t* w_row = w_packed + (size_t)n * (K / 2);
    const __nv_bfloat16* s_row = w_scales + (size_t)n * (K / 64);
    float acc = 0.0f;
    for (int g = 0; g < K / 64; ++g) {
        float sc = __bfloat162float(s_row[g]);
        float group_sum = 0.0f;
        for (int i = 0; i < 64; ++i) {
            int k = g * 64 + i;
            uint8_t b = w_row[k / 2];
            int w_val = (k & 1) ? ((b >> 4) - 8) : ((b & 0xF) - 8);
            group_sum += (float)w_val * __bfloat162float(x[k]);
        }
        acc += group_sum * sc;
    }
    return acc;
}

int main() {
    const int N = 6144, K = 3072;
    const int M = 1;

    // Host data: random weights and activations
    srand(42);
    size_t packed_sz = (size_t)N * K / 2;
    size_t ws_sz = (size_t)N * (K / 64) * sizeof(__nv_bfloat16);
    size_t x_sz = K * sizeof(__nv_bfloat16);

    uint8_t* h_w = (uint8_t*)malloc(packed_sz);
    __nv_bfloat16* h_ws = (__nv_bfloat16*)malloc(ws_sz);
    __nv_bfloat16* h_x = (__nv_bfloat16*)malloc(x_sz);

    for (size_t i = 0; i < packed_sz; ++i)
        h_w[i] = (uint8_t)(rand() & 0xFF);
    for (int i = 0; i < N * (K / 64); ++i)
        h_ws[i] = __float2bfloat16(0.5f + 0.001f * (rand() % 1000));
    for (int i = 0; i < K; ++i)
        h_x[i] = __float2bfloat16((rand() % 200 - 100) / 100.0f);

    // Device data
    uint8_t* d_w; cudaMalloc(&d_w, packed_sz);
    __nv_bfloat16* d_ws; cudaMalloc(&d_ws, ws_sz);
    __nv_bfloat16* d_x; cudaMalloc(&d_x, x_sz);
    __nv_bfloat16* d_y; cudaMalloc(&d_y, N * sizeof(__nv_bfloat16));
    cudaMemcpy(d_w, h_w, packed_sz, cudaMemcpyHostToDevice);
    cudaMemcpy(d_ws, h_ws, ws_sz, cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, h_x, x_sz, cudaMemcpyHostToDevice);

    size_t smem = K + K/64*2 + 128 + K*2;

    // Run DP4A kernel
    viper::ops::dp4a_q4_v3_kernel<<<(N+7)/8, 256, smem>>>(
        d_w, d_ws, d_x, d_y, nullptr, M, N, K, nullptr, 0.0f, nullptr);
    cudaDeviceSynchronize();

    // Copy result
    __nv_bfloat16* h_y = (__nv_bfloat16*)malloc(N * sizeof(__nv_bfloat16));
    cudaMemcpy(h_y, d_y, N * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);

    // Correctness check: compare first 8 outputs
    printf("=== Correctness Check (DP4A vs Scalar CPU) ===\n");
    float max_err = 0.0f;
    for (int n = 0; n < 8; ++n) {
        float gpu = __bfloat162float(h_y[n]);
        float cpu = scalar_q4_gemv(h_w, h_ws, h_x, N, K, n);
        float err = fabsf(gpu - cpu) / (fabsf(cpu) + 1e-6f);
        if (err > max_err) max_err = err;
        printf("  n=%d: gpu=%.4f cpu=%.4f err=%.4f%%\n", n, gpu, cpu, err * 100);
    }
    printf("Max relative error: %.4f%%\n", max_err * 100);

    // Benchmark
    int reps = 1000;
    for (int i = 0; i < 10; i++)
        viper::ops::dp4a_q4_v3_kernel<<<(N+7)/8, 256, smem>>>(
            d_w, d_ws, d_x, d_y, nullptr, M, N, K, nullptr, 0.0f, nullptr);
    cudaDeviceSynchronize();

    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < reps; i++)
        viper::ops::dp4a_q4_v3_kernel<<<(N+7)/8, 256, smem>>>(
            d_w, d_ws, d_x, d_y, nullptr, M, N, K, nullptr, 0.0f, nullptr);
    cudaDeviceSynchronize();
    auto t1 = std::chrono::steady_clock::now();
    double us = std::chrono::duration<double, std::micro>(t1 - t0).count() / reps;
    double bw = (double)packed_sz / (us * 1e3);
    printf("\n=== DP4A Production q_proj (N=%d K=%d) ===\n", N, K);
    printf("  %.1f us, %.2f MB weights, %.1f GB/s (%.0f%%)\n",
           us, (double)packed_sz/1e6, bw, bw/448*100);

    // down_proj
    const int N2 = 3072, K2 = 10752;
    size_t p2 = (size_t)N2 * K2 / 2;
    uint8_t* d_w2; cudaMalloc(&d_w2, p2);
    __nv_bfloat16* d_ws2; cudaMalloc(&d_ws2, N2*(K2/64)*sizeof(__nv_bfloat16));
    __nv_bfloat16* d_x2; cudaMalloc(&d_x2, K2*sizeof(__nv_bfloat16));
    __nv_bfloat16* d_y2; cudaMalloc(&d_y2, N2*sizeof(__nv_bfloat16));
    cudaMemset(d_w2, 0x88, p2);
    cudaMemset(d_ws2, 0x3C00, N2*(K2/64)*sizeof(__nv_bfloat16));
    cudaMemset(d_x2, 0x3C00, K2*sizeof(__nv_bfloat16));

    for (int i = 0; i < 10; i++)
        viper::ops::dp4a_q4_v3_kernel<<<(N2+7)/8, 256, smem>>>(
            d_w2, d_ws2, d_x2, d_y2, nullptr, M, N2, K2, nullptr, 0.0f, nullptr);
    cudaDeviceSynchronize();

    t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < reps; i++)
        viper::ops::dp4a_q4_v3_kernel<<<(N2+7)/8, 256, smem>>>(
            d_w2, d_ws2, d_x2, d_y2, nullptr, M, N2, K2, nullptr, 0.0f, nullptr);
    cudaDeviceSynchronize();
    t1 = std::chrono::steady_clock::now();
    us = std::chrono::duration<double, std::micro>(t1 - t0).count() / reps;
    bw = (double)p2 / (us * 1e3);
    printf("\n=== DP4A Production down_proj (N=%d K=%d) ===\n", N2, K2);
    printf("  %.1f us, %.2f MB weights, %.1f GB/s (%.0f%%)\n",
           us, (double)p2/1e6, bw, bw/448*100);

    free(h_w); free(h_ws); free(h_x); free(h_y);
    cudaFree(d_w); cudaFree(d_ws); cudaFree(d_x); cudaFree(d_y);
    cudaFree(d_w2); cudaFree(d_ws2); cudaFree(d_x2); cudaFree(d_y2);
    return 0;
}
