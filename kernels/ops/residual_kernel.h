/*
 * viper Residual add kernel — header
 *
 * PURPOSE: out[i] = x[i] + y[i], element-wise, bf16 in/out.
 *
 * Used twice per layer-step:
 *   1. residual + attn_output (after o_proj)
 *   2. residual + mlp_output (after down_proj)
 *
 * SAFETY:
 *   - No allocation.
 *   - Streamed.
 */
#ifndef VIPER_RESIDUAL_KERNEL_H
#define VIPER_RESIDUAL_KERNEL_H

#include <cuda_bf16.h>
#include <cstdint>

namespace viper {
namespace ops {

// out[i] = x[i] + y[i]
cudaError_t residual_add_bf16(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ y,
    __nv_bfloat16* __restrict__ out,
    int N,
    cudaStream_t stream);

// In-place: x[i] += y[i]
cudaError_t residual_add_inplace_bf16(
    __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ y,
    int N,
    cudaStream_t stream);

}  // namespace ops
}  // namespace viper

#endif  // VIPER_RESIDUAL_KERNEL_H
