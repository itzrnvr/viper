/*
 * viper RoPE kernel — header
 *
 * PURPOSE: Rotary Position Embedding with theta=70_000_000, no scaling,
 *          head_dim=128, applied in-place to Q and K tensors. Used per
 *          layer-step in the Nanbeige4.2-3B attention path.
 *
 * MATH:
 *   inv_freq[i] = 1 / theta^(2i / head_dim), for i in [0, head_dim/2)
 *   For each position p, compute cos[p, i] = cos(p * inv_freq[i]),
 *   sin[p, i] = sin(p * inv_freq[i]), broadcast to full head_dim by tiling.
 *   rotate_half(x) = cat([-x[head_dim/2:], x[:head_dim/2]], dim=-1)
 *   out = x * cos + rotate_half(x) * sin
 *
 * KEY DECISIONS:
 * - In-place on Q and K. Both are [B, num_heads, T, head_dim] bf16.
 * - Cos/sin table computed once per (B, T) batch in a small helper kernel,
 *   then cached. Tables live in a (T_max, head_dim) buffer in device memory.
 * - fp32 cos/sin computation (the source's torch.autocast(enabled=False) in
 *   modeling_nanbeige.py line 939); bf16 RoPE silently drifts at long context.
 *
 * GOTCHAS:
 * - The cos/sin table is small (T * head_dim * 2 * 4 bytes fp32 = 1 MB per
 *   1 K context). Precompute it once, never per-step.
 * - position_ids are shared across both loop iterations of the model
 *   (cache_position is computed ONCE before the loop). So the cos/sin table
 *   is also shared across both loops.
 */
#ifndef VIPER_ROPE_KERNEL_H
#define VIPER_ROPE_KERNEL_H

#include <cuda_bf16.h>
#include <cstdint>

namespace viper {
namespace ops {

// Precompute cos/sin table for a range of positions.
//   inv_freq[i] = 1 / theta^(2i / head_dim), for i in [0, head_dim/2)
//   cos[pos, i] = cos(pos * inv_freq[i]), for pos in [pos_start, pos_start + T)
//                 tiled to full head_dim by [i, i + head_dim/2]
//   sin[pos, i] = sin(pos * inv_freq[i]), same tiling
//
// cos_table, sin_table: [T, head_dim], fp32, row-major, device pointers
// pos_start: starting absolute position (usually past_seen_tokens at decode)
// T: number of positions
// theta: RoPE theta (70000000 for Nanbeige4.2-3B)
// head_dim: must be 128
// stream: CUDA stream
//
// Returns cudaSuccess or cudaErrorInvalidValue on bad args.
cudaError_t rope_precompute_cos_sin(
    float* cos_table,
    float* sin_table,
    int pos_start,
    int T,
    float theta,
    int head_dim,
    cudaStream_t stream);

// Apply RoPE in-place to Q and K tensors using the precomputed cos/sin table.
//   Q, K: [B, num_heads, T, head_dim], bf16, row-major (assuming BHSD layout)
//   cos_table, sin_table: [T, head_dim], fp32 (from rope_precompute_cos_sin)
//   B, num_heads_q, num_heads_kv, T, head_dim: shapes
//   stream: CUDA stream
//
// GQA: Q has num_heads_q heads (48), K has num_heads_kv heads (8).
//      Q uses all cos/sin entries; K reuses each cos/sin entry num_heads_q / num_heads_kv times.
cudaError_t rope_apply_inplace_bf16(
    __nv_bfloat16* __restrict__ Q,
    __nv_bfloat16* __restrict__ K,
    const float* __restrict__ cos_table,
    const float* __restrict__ sin_table,
    int B,
    int num_heads_q,
    int num_heads_kv,
    int T,
    int head_dim,
    cudaStream_t stream);

}  // namespace ops
}  // namespace viper

#endif  // VIPER_ROPE_KERNEL_H
