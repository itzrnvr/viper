/*
 * viper SwiGLU activation kernel — header
 *
 * PURPOSE: out[i] = silu(gate[i]) * up[i], element-wise, bf16 in/out.
 *
 *   silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
 *
 * Used between gate/up projection and down projection in the MLP path.
 *
 * SAFETY:
 *   - No allocation.
 *   - Streamed; caller passes stream.
 */
#ifndef VIPER_SWIGLU_KERNEL_H
#define VIPER_SWIGLU_KERNEL_H

#include <cuda_bf16.h>
#include <cstdint>

namespace viper {
namespace ops {

// SwiGLU forward, in-place: gate[i] = silu(gate[i]) * up[i]
cudaError_t swiglu_inplace_bf16(
    __nv_bfloat16* __restrict__ gate,
    const __nv_bfloat16* __restrict__ up,
    int N,
    cudaStream_t stream);

// Out-of-place variant: out[i] = silu(gate[i]) * up[i]
cudaError_t swiglu_out_of_place_bf16(
    const __nv_bfloat16* __restrict__ gate,
    const __nv_bfloat16* __restrict__ up,
    __nv_bfloat16* __restrict__ out,
    int N,
    cudaStream_t stream);

}  // namespace ops
}  // namespace viper

#endif  // VIPER_SWIGLU_KERNEL_H
