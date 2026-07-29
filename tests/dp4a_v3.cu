// DP4A GEMV v3: grid-stride access pattern matching the streaming benchmark.
// No SMEM caching, no per-group scales — just raw throughput test.
#include <cstdio>
#include <chrono>
#include <cuda_runtime.h>

__global__ void dp4a_gemv_v3(
    const uint32_t* __restrict__ w32,    // [N, K/8] Q4 packed as uint32
    const int32_t* __restrict__ x32,      // [K/4] INT8 activations as int32
    float* __restrict__ y,
    int N, int K) {
    const int warp_id = threadIdx.x >> 5;
    const int lane_id = threadIdx.x & 31;
    const int n = blockIdx.x * (blockDim.x >> 5) + warp_id;
    if (n >= N) return;

    const uint32_t* w_row = w32 + (size_t)n * (K / 8);
    const int n_quads = K / 8;  // number of uint32 weight loads per output

    int32_t partial = 0;

    // Grid-stride within warp: lane t processes elements t, t+32, t+64, ...
    for (int i = lane_id; i < n_quads; i += 32) {
        // Load 4 bytes = 8 Q4 nibbles
        uint32_t packed4 = w_row[i];

        // Convert Q4 → INT8 using byte_perm + vsubss4
        uint32_t lo = packed4 & 0x0F0F0F0F;
        uint32_t hi = (packed4 >> 4) & 0x0F0F0F0F;
        int32_t slo = __vsubss4(lo, 0x08080808);
        int32_t shi = __vsubss4(hi, 0x08080808);
        int32_t w_lo = __byte_perm(slo, shi, 0x5140);
        int32_t w_hi = __byte_perm(slo, shi, 0x7362);

        // Load 2 int32 of INT8 activations (8 bytes total)
        int32_t x_lo = x32[i * 2];
        int32_t x_hi = x32[i * 2 + 1];

        // DP4A: 4 multiply-adds per call
        partial = __dp4a(w_lo, x_lo, partial);
        partial = __dp4a(w_hi, x_hi, partial);
    }

    // Warp reduce
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        partial += __shfl_xor_sync(0xffffffff, partial, off);

    if (lane_id == 0) y[n] = (float)partial;
}

int main() {
    const int N = 6144, K = 3072;
    size_t w_sz = (size_t)N * K / 8 * sizeof(uint32_t);
    size_t x_sz = K / 4 * sizeof(int32_t);

    uint32_t* d_w; cudaMalloc(&d_w, w_sz);
    int32_t* d_x; cudaMalloc(&d_x, x_sz);
    float* d_y; cudaMalloc(&d_y, N * sizeof(float));
    cudaMemset(d_w, 0x88, w_sz);
    cudaMemset(d_x, 0x01, x_sz);

    // Warmup
    for (int i = 0; i < 10; i++)
        dp4a_gemv_v3<<<(N+7)/8, 256>>>(d_w, d_x, d_y, N, K);
    cudaDeviceSynchronize();

    int reps = 1000;
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < reps; i++)
        dp4a_gemv_v3<<<(N+7)/8, 256>>>(d_w, d_x, d_y, N, K);
    cudaDeviceSynchronize();
    auto t1 = std::chrono::steady_clock::now();
    double us = std::chrono::duration<double, std::micro>(t1 - t0).count() / reps;

    double total = w_sz + x_sz;  // weight + activation
    double achieved = total / (us * 1e3);
    printf("DP4A-v3 q_proj (N=%d K=%d): %.1f us, %.2f MB, %.1f GB/s (%.0f%%)\n",
           N, K, us, total/1e6, achieved, achieved/448*100);

    // down_proj
    const int N2 = 3072, K2 = 10752;
    size_t w2 = (size_t)N2 * K2 / 8 * sizeof(uint32_t);
    size_t x2 = K2 / 4 * sizeof(int32_t);
    uint32_t* d_w2; cudaMalloc(&d_w2, w2);
    int32_t* d_x2; cudaMalloc(&d_x2, x2);
    float* d_y2; cudaMalloc(&d_y2, N2*sizeof(float));
    cudaMemset(d_w2, 0x88, w2);
    cudaMemset(d_x2, 0x01, x2);

    for (int i = 0; i < 10; i++)
        dp4a_gemv_v3<<<(N2+7)/8, 256>>>(d_w2, d_x2, d_y2, N2, K2);
    cudaDeviceSynchronize();

    t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < reps; i++)
        dp4a_gemv_v3<<<(N2+7)/8, 256>>>(d_w2, d_x2, d_y2, N2, K2);
    cudaDeviceSynchronize();
    t1 = std::chrono::steady_clock::now();
    us = std::chrono::duration<double, std::micro>(t1 - t0).count() / reps;
    total = w2 + x2;
    achieved = total / (us * 1e3);
    printf("DP4A-v3 down_proj (N=%d K=%d): %.1f us, %.2f MB, %.1f GB/s (%.0f%%)\n",
           N2, K2, us, total/1e6, achieved, achieved/448*100);

    // Compare: scalar Q4 for reference
    printf("\n--- For comparison, current scalar Q4 achieves ~198 GB/s (44%%) ---\n");

    cudaFree(d_w); cudaFree(d_x); cudaFree(d_y);
    cudaFree(d_w2); cudaFree(d_x2); cudaFree(d_y2);
    return 0;
}
