/*
 * viper Sampling kernel — header.
 *
 * PURPOSE: Token sampling. For v1: greedy argmax on the GPU (single
 *          kernel, warp-reduce). Top-k / top-p / temperature are
 *          post-processed on the host after a partial top-k gather.
 *
 *          For v1 we implement greedy only (temperature -> 0 path).
 *          Top-k / top-p are straightforward extensions; deferred.
 *
 * ALGORITHM (greedy):
 *   - For each (batch), find the argmax over vocab.
 *   - One block per batch; one thread per vocab chunk.
 *   - Warp-reduce to find the local argmax per warp; then block
 *     reduces across warps.
 *
 * CORRECTNESS:
 *   - bf16 logits in, i32 token id out.
 *   - max_abs tie-breaking by lowest index (argmax semantics).
 */
#ifndef VIPER_SAMPLING_KERNEL_H
#define VIPER_SAMPLING_KERNEL_H

#include <cuda_bf16.h>
#include <cstdint>

namespace viper {
namespace ops {

// Greedy argmax over the last dim of logits.
//   logits: [B, V] bf16
//   out_token: [B] i32 (host or device)
cudaError_t sampling_greedy_bf16(
    const __nv_bfloat16* __restrict__ logits,
    int32_t* __restrict__ out_token,
    int B, int V, cudaStream_t stream);

}  // namespace ops
}  // namespace viper

#endif  // VIPER_SAMPLING_KERNEL_H
