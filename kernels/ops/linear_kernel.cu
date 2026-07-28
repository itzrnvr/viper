/*
 * viper Linear Q4_G64 GEMV kernel — warp-per-channel design.
 *
 * Each WARP handles one output channel. 32 threads in the warp cooperate
 * over K with stride 32, giving coalesced reads. 4 warps per block = 
 * 4 output channels per block → N/4 blocks total.
 *
 * This balances coalescing (threads read consecutive bytes) with block
 * scheduling overhead (N/4 blocks, not N blocks).
 */
#include "linear_kernel.h"
#include <cuda_runtime.h>

namespace viper {
namespace ops {

// 4 warps per block (128 threads), each warp handles 1 output channel.
// Grid: (N/4, M) blocks.
__global__ void linear_q4_g64_warp_kernel(
    const uint8_t* __restrict__ w_packed,
    const __nv_bfloat16* __restrict__ w_scales,
    const __nv_bfloat16* __restrict__ x,
    __nv_bfloat16* __restrict__ y,
    int M, int N, int K) {
    const int m = blockIdx.y;
    const int warp_id = threadIdx.x >> 5;  // 0..3
    const int lane_id = threadIdx.x & 31;
    const int n = blockIdx.x * 4 + warp_id;
    if (n >= N || m >= M) return;

    const uint8_t* w_row = w_packed + (size_t)n * (K / 2);
    const __nv_bfloat16* s_row = w_scales + (size_t)n * (K / 64);
    const __nv_bfloat16* x_row = x + (size_t)m * K;

    float acc = 0.0f;
    const int n_bytes = K / 2;

    // Each thread reads bytes at stride 32 (coalesced within the warp).
    for (int byte_idx = lane_id; byte_idx < n_bytes; byte_idx += 32) {
        int group = byte_idx / 32;
        float scale = __bfloat162float(s_row[group]);

        uint8_t b = w_row[byte_idx];
        int w0 = (b & 0x0F) - 8;
        int w1 = (b >> 4) - 8;

        int k0 = byte_idx * 2;
        float x0 = __bfloat162float(x_row[k0]);
        float x1 = __bfloat162float(x_row[k0 + 1]);

        acc += (float)w0 * scale * x0;
        acc += (float)w1 * scale * x1;
    }

    // Warp reduce.
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_xor_sync(0xffffffff, acc, offset);
    }

    if (lane_id == 0) {
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
    if (!w_packed || !w_scales || !x || !y || M <= 0 || N <= 0 || K <= 0) {
        return cudaErrorInvalidValue;
    }
    if (K % 64 != 0) {
        return cudaErrorInvalidValue;
    }

    // 4 warps per block = 4 output channels per block.
    int n_blocks = (N + 3) / 4;
    dim3 grid(n_blocks, M);
    dim3 block(128);  // 4 warps
    linear_q4_g64_warp_kernel<<<grid, block, 0, stream>>>(
        w_packed, w_scales, x, y, M, N, K);
    return cudaGetLastError();
}

}  // namespace ops
}  // namespace viper
