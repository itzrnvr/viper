#include <cstdio>
#include <chrono>
#include <cuda_runtime.h>
int main() {
    // Reset device to clear WDDM state
    cudaDeviceReset();
    cudaSetDevice(0);
    
    // Force GPU to boost with sustained workload
    int N = 10000000;
    float *a, *b;
    cudaMalloc(&a, N * 4);
    cudaMalloc(&b, N * 4);
    
    // Warmup with sustained copies
    for (int i = 0; i < 100; i++)
        cudaMemcpy(b, a, N * 4, cudaMemcpyDeviceToDevice);
    cudaDeviceSynchronize();
    
    // Benchmark
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < 100; i++)
        cudaMemcpy(b, a, N * 4, cudaMemcpyDeviceToDevice);
    cudaDeviceSynchronize();
    auto t1 = std::chrono::steady_clock::now();
    double us = std::chrono::duration<double, std::micro>(t1 - t0).count() / 100;
    double gb = (double)N * 4 / (us * 1e3);
    printf("After reset: D2D memcpy %.1f GB/s (%.0f%% of 448)\n", gb, gb/448*100);
    
    cudaFree(a); cudaFree(b);
    return 0;
}
