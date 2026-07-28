/*
 * viper RMSNorm forward kernel — header
 *
 * PURPOSE: RMSNorm (y = rsqrt(mean(x^2) + eps) * x * gamma), bf16 in/out, fp32
 *          internal accumulation. Used as the pre-norm for both sublayers
 *          (input_layernorm, post_attention_layernorm) and as the final
 *          inter-loop norm (skip_loop_final_norm=False on Nanbeige4.2-3B).
 *
 * KEY DECISIONS:
 * - Single-block-per-row layout. One CUDA block handles one row of H hidden.
 * - 128 threads per block, 4 warps. Each thread handles H/128 elements via
 *   vectorized float4 (8 bf16) loads — exactly 3 float4 = 24 elements per
 *   thread when H = 3072. Falls back to scalar tail for non-aligned H.
 * - fp32 accumulation in registers; warp shuffle for intra-warp reduce;
 *   shared memory (single float) for inter-warp reduce.
 * - gamma stays in registers for the multiply; loaded once with the input.
 *
 * GOTCHAS:
 * - Do NOT allocate dynamic SMEM for x; x is streamed through registers to
 *   keep SMEM under 4 KiB (the gamma scratch + the broadcast scalar).
 * - rsqrt is computed once per row, broadcast via SMEM. Do not re-compute.
 * - bf16 conversion uses __float22bfloat162_rn for paired conversion when
 *   possible; scalar __float2bfloat16 for tail.
 */
#ifndef VIPER_RMSNORM_KERNEL_H
#define VIPER_RMSNORM_KERNEL_H

#include <cuda_bf16.h>
#include <cstdint>

namespace viper {
namespace ops {

// RMSNorm forward, bf16 in/out, fp32 internal, gamma in bf16.
//   out[i] = (x[i] * rsqrt(mean(x^2) + eps)) * gamma[i], cast back to bf16.
//
// x:      [rows, H], bf16, row-major, device pointer
// gamma:  [H], bf16, device pointer
// out:    [rows, H], bf16, row-major, device pointer
// rows:   number of rows (e.g. B * T or B * 1 for decode)
// H:      hidden size (must be > 0; padded internally if not multiple of 24)
// eps:    epsilon (Nanbeige4.2-3B uses 1e-5)
// stream: CUDA stream
//
// Returns Status::OK on success, Status::OOM if VRAM insufficient,
// Status::INVALID_ARG if H == 0 or pointers are null.
cudaError_t rmsnorm_forward_bf16(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ gamma,
    __nv_bfloat16* __restrict__ out,
    int rows,
    int H,
    float eps,
    cudaStream_t stream);

}  // namespace ops
}  // namespace viper

#endif  // VIPER_RMSNORM_KERNEL_H
