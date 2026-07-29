/*
 * DP4A Q4-INT8 GEMV benchmark kernel.
 * 
 * Processes Q4 weights × INT8 activations using __dp4a().
 * Expected: 80%+ bandwidth utilization (vs 44% for scalar Q4 GEMV).
 *
 * ACTIVATIONS must be pre-quantized to INT8 with per-group FP16 scale.
 * WEIGHTS are Q4 (nibble-8), same format as existing engine.
 */
#include <cstdio>
#include <chrono>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

// Extract 4 Q4 nibbles from 2 bytes, pack as 4 INT8 in an int32.
// Nibble value n → INT8 (n - 8), range [-8, 7].
__device__ __forceinline__ int32_t pack_q4_dp4a(uint8_t lo_byte, uint8_t hi_byte) {
    // lo_byte has nibbles [w0, w1], hi_byte has [w2, w3]
    int8_t q0 = (int8_t)((lo_byte & 0xF) - 8);
    int8_t q1 = (int8_t)((lo_byte >> 4) - 8);
    int8_t q2 = (int8_t)((hi_byte & 0xF) - 8);
    int8_t q3 = (int8_t)((hi_byte >> 4) - 8);
    // Pack: int32 with q0 in byte 0 (little-endian)
    union { int32_t i; int8_t b[4]; } u;
    u.b[0] = q0; u.b[1] = q1; u.b[2] = q2; u.b[3] = q3;
    return u.i;
}

// DP4A GEMV: Q4 weights × INT8 activations.
// Grid: ceil(N/8) × M. Block: 256 threads (8 warps, each handles 1 output channel).
__global__ void dp4a_gemv_kernel(
    const uint8_t* __restrict__ w_packed,      // [N, K/2]
    const __nv_bfloat16* __restrict__ w_scales,  // [N, K/64]
    const int32_t* __restrict__ x_int8,          // [K/4] packed INT8
    const __nv_bfloat16* __restrict__ x_scales,  // [K/64]
    float* __restrict__ y,                        // [M, N] float output
    int M, int N, int K) {

    const int m = blockIdx.y;
    const int warp_id = threadIdx.x >> 5;
    const int lane_id = threadIdx.x & 31;
    const int n = blockIdx.x * 8 + warp_id;
    if (n >= N || m >= M) return;

    const uint8_t* w_row = w_packed + (size_t)n * (K / 2);
    const __nv_bfloat16* ws_row = w_scales + (size_t)n * (K / 64);
    const __nv_bfloat16* xs_row = x_scales;

    float acc = 0.0f;
    const int n_groups = K / 64;  // 48 groups for K=3072

    // Each thread processes elements at stride 32*4=128 within each group.
    // Group = 64 elements = 32 bytes packed weights = 16 int32 activations.
    // Each thread in warp handles: 64/32 = 2 elements per group (1 DP4A call).
    // But DP4A processes 4 elements. So each thread handles 4 elements per group.
    // 64/4 = 16 DP4A calls per group, divided by 32 threads = 0.5 per thread.
    // This doesn't divide evenly. Let me use a different approach.

    // Simpler: each thread processes K/32 = 96 elements sequentially.
    // 96/4 = 24 DP4A calls per thread.
    // Process in groups of 64 for scale application.
    
    const int elems_per_thread = K / 32;  // 96 for K=3072
    const int dp4a_per_thread = elems_per_thread / 4;  // 24

    for (int g = 0; g < n_groups; ++g) {
        int32_t group_sum = 0;
        int g_start = g * 64;
        
        // Each thread handles 2 DP4A calls per group (64/32 = 2 elements, but DP4A does 4)
        // Actually: 64 elements per group / 32 threads = 2 elements per thread per group
        // DP4A needs 4 elements. So each thread handles 2 DP4A calls covering 8 elements
        // from 2 different positions in the group.
        
        // Thread t handles elements [t*2, t*2+1] and [t*2+32, t*2+33] within the group.
        // That's 4 elements: need to pair them as 2 DP4A calls.
        // But they're not contiguous. Let me use contiguous chunks instead.
        
        // Alternative: thread t handles 4 elements at positions [t*2, t*2+1, t*2+2, t*2+3] 
        // within the group. 32 threads × 4 = 128 > 64. Some threads are idle.
        // With 16 active threads: each handles 4 elements = 1 DP4A call.
        
        // SIMPLEST: use first 16 threads of the warp for DP4A, rest are idle.
        // OR: distribute work as 2 elements per thread, accumulate separately.
        
        // Let me just do: each thread processes ALL 64 elements of the group sequentially.
        // This is wasteful (32 threads × 64 = 2048 > 64), but correct for measurement.
        // The warp reduction handles the overcounting.
        
        // NO — that multiplies the result by 32. Bad.
        
        // Correct approach: 
        // 64 elements / 4 per DP4A = 16 DP4A calls per group per warp.
        // 16 DP4A calls / 32 threads = 0.5 per thread. 
        // So each thread does 0 or 1 DP4A call per group.
        // Thread 0-15: each does 1 DP4A call (4 elements)
        // Thread 16-31: idle
        
        if (lane_id < 16) {
            int byte_off = g_start / 2 + lane_id * 2;  // 2 bytes = 4 Q4 values
            uint8_t b0 = w_row[byte_off];
            uint8_t b1 = w_row[byte_off + 1];
            int32_t w_pack = pack_q4_dp4a(b0, b1);
            int32_t x_pack = x_int8[(g_start + lane_id * 4) / 4];
            group_sum = __dp4a(w_pack, x_pack, 0);
        }
        
        // Apply group scale
        float ws = __bfloat162float(ws_row[g]);
        float xs = __bfloat162float(xs_row[g]);
        acc += (float)group_sum * ws * xs;
    }

    // Warp reduce
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        acc += __shfl_xor_sync(0xffffffff, acc, off);

    if (lane_id == 0)
        y[m * N + n] = acc;
}

// Simple INT8 quantizer for benchmarking
__global__ void quantize_int8_simple(
    const __nv_bfloat16* __restrict__ x,
    int32_t* __restrict__ x_int8,
    __nv_bfloat16* __restrict__ x_scales,
    int K) {
    const int g = blockIdx.x;
    const int tid = threadIdx.x;
    const int start = g * 64;
    
    // Find max abs
    __shared__ float smax;
    if (tid == 0) smax = 0.0f;
    __syncthreads();
    
    for (int i = tid; i < 64; i += blockDim.x) {
        float v = fabsf(__bfloat162float(x[start + i]));
        atomicMax((int*)&smax, __float_as_int(v));
    }
    __syncthreads();
    
    float scale = fmaxf(smax / 127.0f, 1e-8f);
    float inv_scale = 1.0f / scale;
    if (tid == 0) x_scales[g] = __float2bfloat16(scale);
    
    // Quantize and pack
    for (int i = tid; i < 64; i += blockDim.x * 4) {
        if (i + 3 < 64) {
            int8_t q[4];
            for (int j = 0; j < 4; j++)
                q[j] = (int8_t)roundf(__bfloat162float(x[start + i + j]) * inv_scale);
            x_int8[(start + i) / 4] = ((int32_t)(uint8_t)q[0]) |
                                      ((int32_t)(uint8_t)q[1] << 8) |
                                      ((int32_t)(uint8_t)q[2] << 16) |
                                      ((int32_t)(uint8_t)q[3] << 24);
        }
    }
}

int main() {
    const int N = 6144, K = 3072;
    size_t packed_sz = (size_t)N * K / 2;
    size_t ws_sz = (size_t)N * (K / 64) * sizeof(__nv_bfloat16);
    size_t x_int8_sz = K / 4 * sizeof(int32_t);
    size_t xs_sz = K / 64 * sizeof(__nv_bfloat16);
    
    uint8_t* d_wp; cudaMalloc(&d_wp, packed_sz);
    __nv_bfloat16* d_ws; cudaMalloc(&d_ws, ws_sz);
    int32_t* d_xi; cudaMalloc(&d_xi, x_int8_sz);
    __nv_bfloat16* d_xs; cudaMalloc(&d_xs, xs_sz);
    float* d_y; cudaMalloc(&d_y, N * sizeof(float));
    __nv_bfloat16* d_x; cudaMalloc(&d_x, K * sizeof(__nv_bfloat16));
    
    cudaMemset(d_wp, 0x88, packed_sz);
    cudaMemset(d_ws, 0x3C00, ws_sz); // bf16 1.0
    cudaMemset(d_x, 0x3C00, K * sizeof(__nv_bfloat16));
    
    // Quantize activations
    quantize_int8_simple<<<K/64, 64>>>(d_x, d_xi, d_xs, K);
    cudaDeviceSynchronize();
    
    // Warmup
    for (int i = 0; i < 10; i++)
        dp4a_gemv_kernel<<<(N+7)/8, 256>>>(d_wp, d_ws, d_xi, d_xs, d_y, 1, N, K);
    cudaDeviceSynchronize();
    
    // Benchmark
    int reps = 1000;
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < reps; i++)
        dp4a_gemv_kernel<<<(N+7)/8, 256>>>(d_wp, d_ws, d_xi, d_xs, d_y, 1, N, K);
    cudaDeviceSynchronize();
    auto t1 = std::chrono::steady_clock::now();
    double us = std::chrono::duration<double, std::micro>(t1 - t0).count() / reps;
    
    double total_bytes = packed_sz + ws_sz + x_int8_sz + xs_sz;
    double achieved = total_bytes / (us * 1e3);
    printf("DP4A q_proj (N=%d, K=%d): %.1f us/launch\n", N, K, us);
    printf("  Data: %.2f MB, Achieved: %.1f GB/s (%.0f%% of 448)\n",
           total_bytes / 1e6, achieved, achieved / 448 * 100);
    
    cudaFree(d_wp); cudaFree(d_ws); cudaFree(d_xi); cudaFree(d_xs);
    cudaFree(d_y); cudaFree(d_x);
    return 0;
}
