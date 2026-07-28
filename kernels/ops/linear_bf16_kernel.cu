/*
 * viper Linear BF16 GEMV/GEMM kernel — implementation.
 *
 * Naive but correct. For T=1 decode this is GEMV; one block per
 * output row, threads cooperate over K.
 */
#include "linear_bf16_kernel.h"
#include <cuda_runtime.h>

namespace viper {
namespace ops {

// One block per output row (n). Block size = 128 threads. Each thread
// accumulates K/128 elements, then warp+block reduce to a single value.
__global__ void linear_bf16_kernel(
    const __nv_bfloat16* __restrict__ w,    // [N, K]
    const __nv_bfloat16* __restrict__ x,    // [M, K]
    __nv_bfloat16* __restrict__ y,          // [M, N]
    int M, int N, int K) {
    const int m = blockIdx.y;
    const int n = blockIdx.x;
    if (m >= M || n >= N) return;
    const int tid = threadIdx.x;
    const int block_size = blockDim.x;

    const __nv_bfloat16* w_row = w + n * K;
    const __nv_bfloat16* x_row = x + m * K;

    float acc = 0.0f;
    for (int k = tid; k < K; k += block_size) {
        acc += __bfloat162float(w_row[k]) * __bfloat162float(x_row[k]);
    }

    // Warp reduce.
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_xor_sync(0xffffffff, acc, offset);
    }

    // Block reduce via shared memory (one float per warp).
    __shared__ float warp_sums[32];  // up to 32 warps = 1024 threads
    int warp_id = tid >> 5;
    int lane_id = tid & 31;
    if (lane_id == 0) warp_sums[warp_id] = acc;
    __syncthreads();
    if (warp_id == 0) {
        int n_warps = (block_size + 31) / 32;
        float v = (lane_id < n_warps) ? warp_sums[lane_id] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1) {
            v += __shfl_xor_sync(0xffffffff, v, offset);
        }
        if (lane_id == 0) {
            y[m * N + n] = __float2bfloat16(v);
        }
    }
}

cudaError_t linear_bf16(
    const __nv_bfloat16* w, const __nv_bfloat16* x, __nv_bfloat16* y,
    int M, int N, int K, cudaStream_t stream) {
    if (!w || !x || !y || M <= 0 || N <= 0 || K <= 0) {
        return cudaErrorInvalidValue;
    }
    if (K < 32) {
        return cudaErrorInvalidValue;  // need at least one warp
    }
    dim3 grid(N, M);
    dim3 block(128);
    linear_bf16_kernel<<<grid, block, 0, stream>>>(w, x, y, M, N, K);
    return cudaGetLastError();
}

}  // namespace ops
}  // namespace viper
