/*
 * viper Linear Q4_G64 GEMV kernel — SMEM-cached activation + vectorized loads.
 *
 * Key insight: all warps in a block read the SAME activation vector x.
 * By caching x in shared memory (6 KB), we eliminate 7× redundant HBM
 * reads per block. The HBM bandwidth is freed for weight reads.
 *
 * Optimizations:
 * 1. 8 warps/block (256 threads) → full SM occupancy
 * 2. x cached in SMEM (cooperative load at block start)
 * 3. uint32 vectorized weight loads
 * 4. Fused residual add (optional)
 */
#include "linear_kernel.h"
#include <cuda_runtime.h>

namespace viper {
namespace ops {

// Shared memory size for x: H bf16 values.
// For H=3072: 6144 bytes. 8 blocks/SM × 6144 = 48 KB < 100 KB SMEM limit.
constexpr int SMEM_X_MAX = 11264;  // max K (down_proj K=10752)

__global__ void linear_q4_g64_warp_kernel(
    const uint8_t* __restrict__ w_packed,
    const __nv_bfloat16* __restrict__ w_scales,
    const __nv_bfloat16* __restrict__ x,
    __nv_bfloat16* __restrict__ y,
    const __nv_bfloat16* __restrict__ residual,
    int M, int N, int K) {

    const int m = blockIdx.y;
    const int warp_id = threadIdx.x >> 5;
    const int lane_id = threadIdx.x & 31;
    const int n = blockIdx.x * 8 + warp_id;
    if (n >= N || m >= M) {
        // Still participate in SMEM load + sync to avoid deadlock.
    }

    // --- Cache x in shared memory ---
    extern __shared__ __nv_bfloat16 smem_x[];
    const __nv_bfloat16* x_row_global = x + (size_t)m * K;

    // Cooperative load: 256 threads load K bf16 values.
    for (int i = threadIdx.x; i < K; i += blockDim.x) {
        smem_x[i] = x_row_global[i];
    }
    __syncthreads();

    if (n >= N || m >= M) return;
    // Now read x from SMEM (fast) instead of HBM.
    const __nv_bfloat16* x_row = smem_x;

    const uint8_t* w_row = w_packed + (size_t)n * (K / 2);
    const __nv_bfloat16* s_row = w_scales + (size_t)n * (K / 64);

    float acc = 0.0f;
    const int n_bytes = K / 2;
    const int vec_end = n_bytes - (n_bytes % 128);

    for (int base = 0; base < vec_end; base += 128) {
        int byte_off = base + lane_id * 4;
        uint32_t packed4 = *reinterpret_cast<const uint32_t*>(w_row + byte_off);

        int w0 = (packed4        & 0xF) - 8;
        int w1 = ((packed4 >> 4) & 0xF) - 8;
        int w2 = ((packed4 >> 8) & 0xF) - 8;
        int w3 = ((packed4 >>12) & 0xF) - 8;
        int w4 = ((packed4 >>16) & 0xF) - 8;
        int w5 = ((packed4 >>20) & 0xF) - 8;
        int w6 = ((packed4 >>24) & 0xF) - 8;
        int w7 = ((packed4 >>28) & 0xF) - 8;

        int xk = byte_off * 2;
        float xv0 = __bfloat162float(x_row[xk    ]);
        float xv1 = __bfloat162float(x_row[xk + 1]);
        float xv2 = __bfloat162float(x_row[xk + 2]);
        float xv3 = __bfloat162float(x_row[xk + 3]);
        float xv4 = __bfloat162float(x_row[xk + 4]);
        float xv5 = __bfloat162float(x_row[xk + 5]);
        float xv6 = __bfloat162float(x_row[xk + 6]);
        float xv7 = __bfloat162float(x_row[xk + 7]);

        float sc = __bfloat162float(s_row[byte_off / 32]);

        acc += sc * ((float)w0*xv0 + (float)w1*xv1 + (float)w2*xv2 + (float)w3*xv3
                   + (float)w4*xv4 + (float)w5*xv5 + (float)w6*xv6 + (float)w7*xv7);
    }

    // Tail (if K/2 not divisible by 128 — typically doesn't execute).
    for (int bi = vec_end + lane_id; bi < n_bytes; bi += 32) {
        float sc = __bfloat162float(s_row[bi / 32]);
        uint8_t b = w_row[bi];
        int w0 = (b & 0xF) - 8;
        int w1 = (b >> 4) - 8;
        int k0 = bi * 2;
        acc += (float)w0 * sc * __bfloat162float(x_row[k0    ]);
        acc += (float)w1 * sc * __bfloat162float(x_row[k0 + 1]);
    }

    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        acc += __shfl_xor_sync(0xffffffff, acc, off);

    if (lane_id == 0) {
        if (residual)
            acc += __bfloat162float(residual[m * N + n]);
        y[m * N + n] = __float2bfloat16(acc);
    }
}

cudaError_t linear_q4_g64_bf16(
    const uint8_t* w_packed,
    const __nv_bfloat16* w_scales,
    const __nv_bfloat16* x,
    __nv_bfloat16* y,
    int M, int N, int K,
    cudaStream_t stream) {
    if (!w_packed || !w_scales || !x || !y || M <= 0 || N <= 0 || K <= 0)
        return cudaErrorInvalidValue;
    if (K % 64 != 0 || K > SMEM_X_MAX)
        return cudaErrorInvalidValue;

    int n_blocks = (N + 7) / 8;
    dim3 grid(n_blocks, M);
    dim3 block(256);
    size_t smem_bytes = K * sizeof(__nv_bfloat16);
    linear_q4_g64_warp_kernel<<<grid, block, smem_bytes, stream>>>(
        w_packed, w_scales, x, y, nullptr, M, N, K);
    return cudaGetLastError();
}

cudaError_t linear_q4_g64_bf16_residual(
    const uint8_t* w_packed,
    const __nv_bfloat16* w_scales,
    const __nv_bfloat16* x,
    __nv_bfloat16* y,
    const __nv_bfloat16* residual,
    int M, int N, int K,
    cudaStream_t stream) {
    if (!w_packed || !w_scales || !x || !y || !residual || M <= 0 || N <= 0 || K <= 0)
        return cudaErrorInvalidValue;
    if (K % 64 != 0 || K > SMEM_X_MAX)
        return cudaErrorInvalidValue;

    int n_blocks = (N + 7) / 8;
    dim3 grid(n_blocks, M);
    dim3 block(256);
    size_t smem_bytes = K * sizeof(__nv_bfloat16);
    linear_q4_g64_warp_kernel<<<grid, block, smem_bytes, stream>>>(
        w_packed, w_scales, x, y, residual, M, N, K);
    return cudaGetLastError();
}

}  // namespace ops
}  // namespace viper
