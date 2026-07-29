/*
 * OPTIMIZED DP4A GEMV with __byte_perm + __vsubss4 Q4 conversion.
 *
 * Key insight: the previous pack_q4 took 24 instructions per 8 elements.
 * Using __byte_perm for nibble interleaving and __vsubss4 for sign correction,
 * we reduce it to 7 instructions. This makes the kernel BANDWIDTH-BOUND.
 *
 * Expected: 85% bandwidth (383 GB/s) — same as raw streaming.
 */
#include <cstdio>
#include <chrono>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

// Sign-correct packed nibbles to INT8 using per-byte SIMD subtract.
// __vsubss4 subtracts packed signed bytes WITHOUT cross-byte borrow.
__device__ __forceinline__ int32_t q4_to_int8_packed(uint32_t packed4) {
    // packed4: 4 bytes, each with 2 nibbles [lo, hi]
    // Extract low nibbles: mask each byte to [0, 15]
    uint32_t lo = packed4 & 0x0F0F0F0F;           // bytes: [q0, q2, q4, q6]
    // Extract high nibbles: shift right 4, mask
    uint32_t hi = (packed4 >> 4) & 0x0F0F0F0F;     // bytes: [q1, q3, q5, q7]
    // Subtract 8 from each byte (signed, no borrow): [-8, 7]
    int32_t slo = __vsubss4(lo, 0x08080808);
    int32_t shi = __vsubss4(hi, 0x08080808);
    // Interleave: [q0,q1,q2,q3] and [q4,q5,q6,q7]
    // lo bytes: [0]=q0, [1]=q2, [2]=q4, [3]=q6
    // hi bytes: [0]=q1, [1]=q3, [2]=q5, [3]=q7
    // lo[0]=slo[0], hi[0]=shi[0], lo[1]=slo[1], hi[1]=shi[1]
    // __byte_perm(a, b, sel): output = bytes from a(0-3) || b(4-7)
    // For [lo[0], hi[0], lo[1], hi[1]]: sel = 0x5140
    // For [lo[2], hi[2], lo[3], hi[3]]: sel = 0x7362
    int32_t pack_lo = __byte_perm(slo, shi, 0x5140);  // [q0, q1, q2, q3]
    // Return the low 4-element pack (high pack computed separately if needed)
    return pack_lo;
}

// Full DP4A GEMV: processes 8 Q4 elements per DP4A call, all 32 threads active.
// Each warp handles 1 output channel. Each thread handles K/32 elements contiguously.
__global__ void dp4a_gemv_opt(
    const uint8_t* __restrict__ w_packed,       // [N, K/2]
    const __nv_bfloat16* __restrict__ w_scales,   // [N, K/64]
    const int8_t* __restrict__ x_int8,            // [K] INT8 activations
    const __nv_bfloat16* __restrict__ x_scales,   // [K/64]
    float* __restrict__ y,                         // [N]
    int N, int K) {
    const int warp_id = threadIdx.x >> 5;
    const int lane_id = threadIdx.x & 31;
    const int n = blockIdx.x * (blockDim.x >> 5) + warp_id;
    if (n >= N) return;

    const uint8_t* w_row = w_packed + (size_t)n * (K / 2);
    const __nv_bfloat16* s_row = w_scales + (size_t)n * (K / 64);

    // Cache activations in SMEM for coalesced access
    extern __shared__ int8_t smem[];
    int8_t* x_smem = smem;
    __nv_bfloat16* xs_smem = (__nv_bfloat16*)(smem + K);  // activation scales
    for (int i = threadIdx.x; i < K; i += blockDim.x)
        x_smem[i] = x_int8[i];
    for (int i = threadIdx.x; i < K / 64; i += blockDim.x)
        xs_smem[i] = x_scales[i];
    __syncthreads();

    // Each thread processes K/32 elements
    const int elems_per_thread = K / 32;  // 96 for K=3072
    const int groups_per_thread = elems_per_thread / 64;  // 1.5 for K=3072
    
    float acc = 0.0f;

    // Process in groups of 64 (for scale application)
    // Each thread handles elems_per_thread/64 = 1 or 2 groups
    for (int g = 0; g < groups_per_thread; ++g) {
        int g_start = lane_id * elems_per_thread + g * 64;
        int g_idx = g_start / 64;
        
        int32_t partial = 0;
        
        // Process 8 elements per iteration (1 uint32 load + 2 DP4A)
        for (int i = 0; i < 64; i += 8) {
            int elem = g_start + i;
            int byte_idx = elem >> 1;
            
            // Load 4 bytes of Q4 weights (8 nibbles)
            uint32_t packed4 = *reinterpret_cast<const uint32_t*>(w_row + byte_idx);
            
            // Extract low nibbles and sign-correct
            uint32_t lo = packed4 & 0x0F0F0F0F;
            uint32_t hi = (packed4 >> 4) & 0x0F0F0F0F;
            int32_t slo = __vsubss4(lo, 0x08080808);
            int32_t shi = __vsubss4(hi, 0x08080808);
            int32_t w_lo = __byte_perm(slo, shi, 0x5140);  // [q0,q1,q2,q3]
            int32_t w_hi = __byte_perm(slo, shi, 0x7362);  // [q4,q5,q6,q7]
            
            // Load 4+4 INT8 activations as int32
            int32_t x_lo = *reinterpret_cast<const int32_t*>(x_smem + elem);
            int32_t x_hi = *reinterpret_cast<const int32_t*>(x_smem + elem + 4);
            
            // DP4A: 4 multiply-adds per instruction
            partial = __dp4a(w_lo, x_lo, partial);
            partial = __dp4a(w_hi, x_hi, partial);
        }
        
        // Apply per-group scales
        float ws = __bfloat162float(s_row[g_idx]);
        float xs = __bfloat162float(xs_smem[g_idx]);
        acc += (float)partial * ws * xs;
    }
    
    // Handle remaining elements (if K/32 not divisible by 64)
    int rem_start = lane_id * elems_per_thread + groups_per_thread * 64;
    for (int i = rem_start; i < lane_id * elems_per_thread + elems_per_thread; i += 8) {
        if (i + 8 > K) break;
        int byte_idx = i >> 1;
        uint32_t packed4 = *reinterpret_cast<const uint32_t*>(w_row + byte_idx);
        uint32_t lo = packed4 & 0x0F0F0F0F;
        uint32_t hi = (packed4 >> 4) & 0x0F0F0F0F;
        int32_t slo = __vsubss4(lo, 0x08080808);
        int32_t shi = __vsubss4(hi, 0x08080808);
        int32_t w_lo = __byte_perm(slo, shi, 0x5140);
        int32_t w_hi = __byte_perm(slo, shi, 0x7362);
        int32_t x_lo = *reinterpret_cast<const int32_t*>(x_smem + i);
        int32_t x_hi = *reinterpret_cast<const int32_t*>(x_smem + i + 4);
        int g_idx = i / 64;
        float ws = __bfloat162float(s_row[g_idx]);
        float xs = __bfloat162float(xs_smem[g_idx]);
        int32_t partial = __dp4a(w_lo, x_lo, 0);
        partial = __dp4a(w_hi, x_hi, partial);
        acc += (float)partial * ws * xs;
    }

    // Warp reduce
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        acc += __shfl_xor_sync(0xffffffff, acc, off);

    if (lane_id == 0)
        y[n] = acc;
}

int main() {
    const int N = 6144, K = 3072;
    size_t packed_sz = (size_t)N * K / 2;
    size_t ws_sz = (size_t)N * (K / 64) * sizeof(__nv_bfloat16);
    size_t x_int8_sz = K * sizeof(int8_t);
    size_t xs_sz = K / 64 * sizeof(__nv_bfloat16);
    
    uint8_t* d_wp; cudaMalloc(&d_wp, packed_sz);
    __nv_bfloat16* d_ws; cudaMalloc(&d_ws, ws_sz);
    int8_t* d_xi; cudaMalloc(&d_xi, x_int8_sz);
    __nv_bfloat16* d_xs; cudaMalloc(&d_xs, xs_sz);
    float* d_y; cudaMalloc(&d_y, N * sizeof(float));
    
    cudaMemset(d_wp, 0x88, packed_sz);  // nibbles [8,8] → values [0,0]
    cudaMemset(d_ws, 0x3C00, ws_sz);    // bf16 1.0
    cudaMemset(d_xi, 1, x_int8_sz);     // all 1s
    cudaMemset(d_xs, 0x3C00, xs_sz);    // bf16 1.0
    
    size_t smem = K * sizeof(int8_t) + (K / 64) * sizeof(__nv_bfloat16) + 128;
    
    // Warmup
    for (int i = 0; i < 10; i++)
        dp4a_gemv_opt<<<(N+7)/8, 256, smem>>>(d_wp, d_ws, d_xi, d_xs, d_y, N, K);
    cudaDeviceSynchronize();
    
    // Benchmark
    int reps = 1000;
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < reps; i++)
        dp4a_gemv_opt<<<(N+7)/8, 256, smem>>>(d_wp, d_ws, d_xi, d_xs, d_y, N, K);
    cudaDeviceSynchronize();
    auto t1 = std::chrono::steady_clock::now();
    double us = std::chrono::duration<double, std::micro>(t1 - t0).count() / reps;
    
    // Bandwidth: weights + activations (activations cached in SMEM, only read once per block)
    double weight_bytes = packed_sz + ws_sz;
    double act_bytes = x_int8_sz + xs_sz;  // read once per block from SMEM
    double total = weight_bytes;  // dominant
    double achieved = total / (us * 1e3);
    printf("DP4A-opt q_proj (N=%d K=%d): %.1f us, weight %.2f MB, %.1f GB/s (%.0f%%)\n",
           N, K, us, weight_bytes/1e6, achieved, achieved/448*100);
    
    // down_proj
    const int N2 = 3072, K2 = 10752;
    size_t p2 = (size_t)N2 * K2 / 2;
    size_t ws2 = (size_t)N2 * (K2/64) * sizeof(__nv_bfloat16);
    size_t xi2 = K2 * sizeof(int8_t);
    size_t xs2 = K2/64 * sizeof(__nv_bfloat16);
    uint8_t* d_wp2; cudaMalloc(&d_wp2, p2);
    __nv_bfloat16* d_ws2; cudaMalloc(&d_ws2, ws2);
    int8_t* d_xi2; cudaMalloc(&d_xi2, xi2);
    __nv_bfloat16* d_xs2; cudaMalloc(&d_xs2, xs2);
    float* d_y2; cudaMalloc(&d_y2, N2*sizeof(float));
    cudaMemset(d_wp2, 0x88, p2);
    cudaMemset(d_ws2, 0x3C00, ws2);
    cudaMemset(d_xi2, 1, xi2);
    cudaMemset(d_xs2, 0x3C00, xs2);
    
    size_t smem2 = K2*sizeof(int8_t) + (K2/64)*sizeof(__nv_bfloat16) + 128;
    
    for (int i = 0; i < 10; i++)
        dp4a_gemv_opt<<<(N2+7)/8, 256, smem2>>>(d_wp2, d_ws2, d_xi2, d_xs2, d_y2, N2, K2);
    cudaDeviceSynchronize();
    
    t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < reps; i++)
        dp4a_gemv_opt<<<(N2+7)/8, 256, smem2>>>(d_wp2, d_ws2, d_xi2, d_xs2, d_y2, N2, K2);
    cudaDeviceSynchronize();
    t1 = std::chrono::steady_clock::now();
    us = std::chrono::duration<double, std::micro>(t1 - t0).count() / reps;
    achieved = (double)(p2+ws2) / (us * 1e3);
    printf("DP4A-opt down_proj (N=%d K=%d): %.1f us, weight %.2f MB, %.1f GB/s (%.0f%%)\n",
           N2, K2, us, (double)(p2+ws2)/1e6, achieved, achieved/448*100);
    
    cudaFree(d_wp); cudaFree(d_ws); cudaFree(d_xi); cudaFree(d_xs); cudaFree(d_y);
    cudaFree(d_wp2); cudaFree(d_ws2); cudaFree(d_xi2); cudaFree(d_xs2); cudaFree(d_y2);
    return 0;
}
