/*
 * viper Linear BF16 GEMV/GEMM kernel — header.
 *
 * PURPOSE: BF16 in/out, FP32 accumulate. Used for the lm_head matmul
 *          (510 M params, BF16, untied). The Q4 linear path is separate.
 *
 * ALGORITHM:
 *   - y[M, N] = x[M, K] @ w[N, K].T
 *   - One block per output row. Threads cooperate over K.
 *   - For T=1 (decode), each block computes one output element via
 *     a dot product over K.
 *
 * CORRECTNESS:
 *   - BF16 inputs, FP32 accumulator, BF16 output.
 *   - Vectorized float4 loads when K divisible by 8.
 */
#ifndef VIPER_LINEAR_BF16_KERNEL_H
#define VIPER_LINEAR_BF16_KERNEL_H

#include <cuda_bf16.h>
#include <cstdint>

namespace viper {
namespace ops {

cudaError_t linear_bf16(
    const __nv_bfloat16* __restrict__ w,    // [N, K]
    const __nv_bfloat16* __restrict__ x,    // [M, K]
    __nv_bfloat16* __restrict__ y,          // [M, N]
    int M, int N, int K,
    cudaStream_t stream);

}  // namespace ops
}  // namespace viper

#endif  // VIPER_LINEAR_BF16_KERNEL_H
