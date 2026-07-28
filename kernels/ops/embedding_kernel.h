/*
 * viper Embedding gather kernel — header
 *
 * PURPOSE: Gather rows from the embedding table by token id.
 *          out[b, t, :] = embed_table[token_ids[b, t], :]
 *
 * Used for:
 *   - Input embedding lookup (model.embed_tokens.weight, vocab=166144, hidden=3072)
 *   - LM head is a separate matmul (out = hidden @ lm_head.T) — NOT a gather,
 *     because lm_head is untied and we want matmul-friendly layout.
 *
 * SHAPES:
 *   - table: [V, H] bf16, row-major
 *   - token_ids: [B, T] int32 or int64
 *   - out: [B, T, H] bf16
 *
 * SAFETY:
 *   - Bounds check on token id < V.
 *   - No allocation inside the kernel.
 */
#ifndef VIPER_EMBEDDING_KERNEL_H
#define VIPER_EMBEDDING_KERNEL_H

#include <cuda_bf16.h>
#include <cstdint>

namespace viper {
namespace ops {

// Gather rows from a bf16 table by int32 token ids.
cudaError_t embedding_gather_bf16_i32(
    const __nv_bfloat16* __restrict__ table,
    const int32_t* __restrict__ token_ids,
    __nv_bfloat16* __restrict__ out,
    int B,
    int T,
    int V,
    int H,
    cudaStream_t stream);

// Same with int64 token ids (PyTorch default).
cudaError_t embedding_gather_bf16_i64(
    const __nv_bfloat16* __restrict__ table,
    const int64_t* __restrict__ token_ids,
    __nv_bfloat16* __restrict__ out,
    int B,
    int T,
    int V,
    int H,
    cudaStream_t stream);

}  // namespace ops
}  // namespace viper

#endif  // VIPER_EMBEDDING_KERNEL_H
