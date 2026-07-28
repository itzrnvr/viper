/*
 * viper SDPA (Scaled Dot-Product Attention) kernel — header.
 *
 * PURPOSE: Causal SDPA with FA-2 online softmax + GQA-6:1 in-kernel
 *          indexing. No repeat_kv materialization — the K/V cache
 *          is read with h_q // 6 as the index.
 *
 * ALGORITHM:
 *   - One block per (batch, query_head, query_block).
 *   - Threads cooperate over the KV sequence in tiles of BLOCK_N.
 *   - Online softmax: track running m (max), l (sum), acc (output).
 *   - On each new K/V tile: compute QK^T, scale, mask (causal), softmax,
 *     accumulate into acc with the log-sum-exp correction.
 *
 * CORRECTNESS:
 *   - GQA: K has n_kv heads; Q has n_heads. h_q // group_size -> h_kv.
 *   - Causal mask: only attend to k <= q (for prefill); for decode T=1,
 *     no mask needed.
 *   - fp32 accumulator for numerical stability.
 *
 * SAFETY:
 *   - No global allocations.
 *   - SMEM usage: tile for Q (BLOCK_M x D), K (BLOCK_N x D), V
 *     (BLOCK_N x D), S (BLOCK_M x BLOCK_N).
 */
#ifndef VIPER_SDPA_KERNEL_H
#define VIPER_SDPA_KERNEL_H

#include <cuda_bf16.h>
#include <cstdint>

namespace viper {
namespace ops {

// Causal SDPA forward, FA-2 online softmax, GQA-6:1 in-kernel.
//   Q: [B, n_heads, T_q, head_dim] bf16
//   K: [B, n_kv_heads, T_k, head_dim] bf16
//   V: [B, n_kv_heads, T_k, head_dim] bf16
//   O: [B, n_heads, T_q, head_dim] bf16 (output)
//   n_heads, n_kv_heads: 48 and 8 for Nanbeige4.2-3B (GQA 6:1).
//   T_q, T_k: query and key sequence lengths.
//   is_causal: true for prefill, false for decode (T_q=1, no mask).
//   scale: 1/sqrt(head_dim).
cudaError_t sdpa_forward_bf16(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    __nv_bfloat16* __restrict__ O,
    int B, int n_heads, int n_kv_heads, int T_q, int T_k, int head_dim,
    float scale, bool is_causal, cudaStream_t stream);

}  // namespace ops
}  // namespace viper

#endif  // VIPER_SDPA_KERNEL_H
