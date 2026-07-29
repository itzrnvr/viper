// Peak DP4A throughput benchmark: measures raw DP4A bandwidth with streaming loads
#include <cstdio>
#include <chrono>
#include <cuda_runtime.h>

__global__ void dp4a_stream(const int32_t* __restrict__ a, const int32_t* __restrict__ b, int* c, int N) {
    int acc = 0;
    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < N; i += gridDim.x * blockDim.x) {
        acc = __dp4a(a[i], b[i], acc);
    }
    atomicAdd(c, acc);
}

int main() {
    const int N = 6144 * 3072 / 4; // Same total data as q_proj
    size_t bytes = (size_t)N * sizeof(int32_t) * 2; // a + b
    
    int32_t *d_a, *d_b;
    int* d_c;
    cudaMalloc(&d_a, N * sizeof(int32_t));
    cudaMalloc(&d_b, N * sizeof(int32_t));
    cudaMalloc(&d_c, sizeof(int));
    cudaMemset(d_a, 0x11, N * sizeof(int32_t));
    cudaMemset(d_b, 0x22, N * sizeof(int32_t));
    
    int block = 256;
    int grid = 48 * 8; // 384 blocks
    
    // Warmup
    for (int i = 0; i < 10; i++) {
        cudaMemset(d_c, 0, sizeof(int));
        dp4a_stream<<<grid, block>>>(d_a, d_b, d_c, N);
    }
    cudaDeviceSynchronize();
    
    // Benchmark
    int reps = 100;
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < reps; i++) {
        cudaMemsetAsync(d_c, 0, sizeof(int));
        dp4a_stream<<<grid, block>>>(d_a, d_b, d_c, N);
    }
    cudaDeviceSynchronize();
    auto t1 = std::chrono::steady_clock::now();
    double us = std::chrono::duration<double, std::micro>(t1 - t0).count() / reps;
    
    double achieved = bytes / (us * 1e3);
    printf("DP4A streaming: %.1f us, %.1f GB/s (%.0f%% of 448)\n",
           us, achieved, achieved / 448 * 100);
    
    // Compare: memcpy bandwidth (theoretical max)
    auto t2 = std::chrono::steady_clock::now();
    for (int i = 0; i < reps; i++)
        cudaMemcpyAsync(d_a, d_b, N * sizeof(int32_t), cudaMemcpyDeviceToDevice);
    cudaDeviceSynchronize();
    auto t3 = std::chrono::steady_clock::now();
    us = std::chrono::duration<double, std::micro>(t3 - t2).count() / reps;
    double memcpy_bw = N * sizeof(int32_t) / (us * 1e3);
    printf("D2D memcpy: %.1f us, %.1f GB/s (%.0f%% of 448)\n",
           us, memcpy_bw, memcpy_bw / 448 * 100);
    
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    return 0;
}
