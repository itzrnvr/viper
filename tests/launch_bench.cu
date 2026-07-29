// Benchmark: measure per-launch overhead on this GPU/OS.
#include <cstdio>
#include <chrono>
#include <cuda_runtime.h>

__global__ void noop_kernel() {}
__global__ void small_kernel(int* x) { if (threadIdx.x == 0) *x = 42; }

int main() {
    int* d_x;
    cudaMalloc(&d_x, 4);
    
    // Warmup
    for (int i = 0; i < 100; i++) noop_kernel<<<1, 32>>>();
    cudaDeviceSynchronize();
    
    // Measure 1000 noop launches
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < 1000; i++) noop_kernel<<<1, 32>>>();
    cudaDeviceSynchronize();
    auto t1 = std::chrono::steady_clock::now();
    double us = std::chrono::duration<double, std::micro>(t1 - t0).count() / 1000.0;
    printf("Noop launch (1 block, 32 threads): %.1f us/launch\n", us);
    
    // Measure 1000 noop launches with more blocks
    t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < 1000; i++) noop_kernel<<<384, 256>>>();
    cudaDeviceSynchronize();
    t1 = std::chrono::steady_clock::now();
    us = std::chrono::duration<double, std::micro>(t1 - t0).count() / 1000.0;
    printf("Noop launch (384 blocks, 256 threads): %.1f us/launch\n", us);
    
    // Measure 1000 small kernel launches
    t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < 1000; i++) small_kernel<<<1, 32>>>(d_x);
    cudaDeviceSynchronize();
    t1 = std::chrono::steady_clock::now();
    us = std::chrono::duration<double, std::micro>(t1 - t0).count() / 1000.0;
    printf("Small kernel (1 block, 32 threads, 1 write): %.1f us/launch\n", us);
    
    // Measure cudaMemcpy overhead
    t0 = std::chrono::steady_clock::now();
    int host;
    for (int i = 0; i < 1000; i++) cudaMemcpy(&host, d_x, 4, cudaMemcpyDeviceToHost);
    t1 = std::chrono::steady_clock::now();
    us = std::chrono::duration<double, std::micro>(t1 - t0).count() / 1000.0;
    printf("cudaMemcpy D2H (4 bytes): %.1f us\n", us);
    
    // Measure cudaMemcpyAsync overhead  
    t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < 1000; i++) cudaMemcpyAsync(d_x, &host, 4, cudaMemcpyHostToDevice, 0);
    cudaDeviceSynchronize();
    t1 = std::chrono::steady_clock::now();
    us = std::chrono::duration<double, std::micro>(t1 - t0).count() / 1000.0;
    printf("cudaMemcpyAsync H2D (4 bytes): %.1f us\n", us);
    
    cudaFree(d_x);
    return 0;
}
