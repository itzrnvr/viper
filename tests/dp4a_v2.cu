// DP4A GEMV v2: proper work distribution — ALL 32 threads active per iteration.
#include <cstdio>
#include <chrono>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

__device__ __forceinline__ int32_t pack_q4(uint8_t b0, uint8_t b1) {
    int8_t q0=(int8_t)((b0&0xF)-8), q1=(int8_t)((b0>>4)-8);
    int8_t q2=(int8_t)((b1&0xF)-8), q3=(int8_t)((b1>>4)-8);
    return ((int32_t)(uint8_t)q0)|((int32_t)(uint8_t)q1<<8)|
           ((int32_t)(uint8_t)q2<<16)|((int32_t)(uint8_t)q3<<24);
}

// Each warp handles 1 output channel. All 32 threads process contiguous K elements.
// Thread t handles elements [t*stride .. t*stride+stride-1].
__global__ void dp4a_gemv_v2(
    const uint8_t* __restrict__ w_packed,       // [N, K/2]
    const __nv_bfloat16* __restrict__ w_scales,   // [N, K/64] — unused for now
    const int32_t* __restrict__ x_int8,           // [K/4]
    float* __restrict__ y,                         // [N]
    int N, int K) {
    const int warp_id = threadIdx.x >> 5;
    const int lane_id = threadIdx.x & 31;
    const int n = blockIdx.x * (blockDim.x >> 5) + warp_id;
    if (n >= N) return;

    const uint8_t* w_row = w_packed + (size_t)n * (K / 2);
    
    // Each thread processes K/32 elements contiguously
    const int elems_per_thread = K / 32;  // 96 for K=3072
    const int start_elem = lane_id * elems_per_thread;
    
    int32_t partial = 0;
    
    // Process 4 elements per DP4A call
    for (int i = 0; i < elems_per_thread; i += 4) {
        int elem = start_elem + i;
        // Load 2 bytes of Q4 weights = 4 nibbles
        int byte_idx = elem >> 1;
        int32_t w_pack = pack_q4(w_row[byte_idx], w_row[byte_idx + 1]);
        // Load 1 int32 of INT8 activations
        int32_t x_pack = x_int8[elem >> 2];
        // DP4A!
        partial = __dp4a(w_pack, x_pack, partial);
    }
    
    // Warp reduce
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        partial += __shfl_xor_sync(0xffffffff, partial, off);
    
    if (lane_id == 0) {
        // Apply average scale (approximation for benchmark)
        y[n] = (float)partial * 0.01f;
    }
}

int main() {
    const int N = 6144, K = 3072;
    size_t packed_sz = (size_t)N * K / 2;
    size_t x_int8_sz = K / 4 * sizeof(int32_t);
    
    uint8_t* d_wp; cudaMalloc(&d_wp, packed_sz);
    int32_t* d_xi; cudaMalloc(&d_xi, x_int8_sz);
    float* d_y; cudaMalloc(&d_y, N * sizeof(float));
    cudaMemset(d_wp, 0x88, packed_sz);
    cudaMemset(d_xi, 0x11, x_int8_sz);
    
    // Warmup
    for (int i = 0; i < 10; i++)
        dp4a_gemv_v2<<<(N+7)/8, 256>>>(d_wp, nullptr, d_xi, d_y, N, K);
    cudaDeviceSynchronize();
    
    // Benchmark
    int reps = 1000;
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < reps; i++)
        dp4a_gemv_v2<<<(N+7)/8, 256>>>(d_wp, nullptr, d_xi, d_y, N, K);
    cudaDeviceSynchronize();
    auto t1 = std::chrono::steady_clock::now();
    double us = std::chrono::duration<double, std::micro>(t1 - t0).count() / reps;
    
    double total = packed_sz + x_int8_sz;
    double achieved = total / (us * 1e3);
    printf("DP4A v2 q_proj (N=%d K=%d): %.1f us, %.1f GB/s (%.0f%%)\n",
           N, K, us, achieved, achieved/448*100);
    
    // Test down_proj (K=10752)
    const int N2 = 3072, K2 = 10752;
    size_t p2 = (size_t)N2 * K2 / 2;
    size_t x2 = K2 / 4 * sizeof(int32_t);
    uint8_t* d_wp2; cudaMalloc(&d_wp2, p2);
    int32_t* d_xi2; cudaMalloc(&d_xi2, x2);
    float* d_y2; cudaMalloc(&d_y2, N2 * sizeof(float));
    cudaMemset(d_wp2, 0x88, p2);
    cudaMemset(d_xi2, 0x11, x2);
    
    for (int i = 0; i < 10; i++)
        dp4a_gemv_v2<<<(N2+7)/8, 256>>>(d_wp2, nullptr, d_xi2, d_y2, N2, K2);
    cudaDeviceSynchronize();
    
    t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < reps; i++)
        dp4a_gemv_v2<<<(N2+7)/8, 256>>>(d_wp2, nullptr, d_xi2, d_y2, N2, K2);
    cudaDeviceSynchronize();
    t1 = std::chrono::steady_clock::now();
    us = std::chrono::duration<double, std::micro>(t1 - t0).count() / reps;
    total = p2 + x2;
    achieved = total / (us * 1e3);
    printf("DP4A v2 down_proj (N=%d K=%d): %.1f us, %.1f GB/s (%.0f%%)\n",
           N2, K2, us, achieved, achieved/448*100);
    
    cudaFree(d_wp); cudaFree(d_xi); cudaFree(d_y);
    cudaFree(d_wp2); cudaFree(d_xi2); cudaFree(d_y2);
    return 0;
}
