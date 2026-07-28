/*
 * viper megakernel — the entire 44-step forward in one CUDA launch.
 *
 * PURPOSE: Eliminate the ~800+ kernel launches per token that currently
 *          dominate the per-token cost. One grid launch runs the whole
 *          forward pass: embedding -> 44 layer-steps -> lm_head -> sample.
 *
 * DESIGN:
 *   - Persistent grid-sync megakernel. One block per weight tile.
 *   - Each block loads its tile via cp.async, runs the GEMV for both
 *     loop passes on the same tile, then moves to the next tile.
 *   - The activation frontier (the "current hidden state") moves
 *     through global memory between blocks; the weight tile stays
 *     in shared memory for the block's lifetime.
 *   - For Q4_G64 weights, a 128x128 tile is 8 KB — fits in SMEM.
 *
 * CORRECTNESS:
 *   - Same math as the per-op path, just fused.
 *   - fp32 accumulation in registers, bf16 output.
 *   - The loop dispatch is host-side: for loop_idx in 0..1, for layer in
 *     0..21, for tile in 0..n_tiles-1. The kernel is launched once per
 *     (layer, loop, tile) triple; the host controls the order.
 *
 * SAFETY:
 *   - No dynamic SMEM allocations inside the kernel.
 *   - cp.async for weight streaming; the tile is read from HBM once.
 *   - The host manages the loop order and the frontier buffer.
 */
#ifndef VIPER_MEGAKERNEL_H
#define VIPER_MEGAKERNEL_H

#include <cuda_bf16.h>
#include <cstdint>

namespace viper {
namespace ops {

// The megakernel's "op" is a tuple describing one step of the forward.
// The host builds a vector of these and launches the kernel once.
struct MegakernelStep {
    int loop_idx;    // 0 or 1
    int layer_idx;   // 0..21
    int tile_idx;    // weight tile index within the layer
    int tile_m;      // tile row offset
    int tile_n;      // tile col offset
    int tile_k;      // tile depth
    // Which op to run for this step
    int op_id;       // 0=rmsnorm, 1=qkv, 2=rope, 3=sdpa, 4=o, 5=residual, 6=mlp_gate_up, 7=swiglu, 8=mlp_down, 9=residual, 10=lm_head, 11=sample
};

// Launch the megakernel for the full 44-step forward.
//   steps: the list of MegakernelStep to run
//   n_steps: count
//   hidden_in: [T, H] current activation (from embedding or previous step)
//   hidden_out: [T, H] next activation
//   kv_cache: the 44-slot KV cache
//   ... (more args for the specific ops)
cudaError_t megakernel_forward(
    const MegakernelStep* steps,
    int n_steps,
    const __nv_bfloat16* hidden_in,
    __nv_bfloat16* hidden_out,
    void* kv_cache,
    const void* weights,
    int T, int H, int I, int nQ, int nKV, int HD, int V,
    cudaStream_t stream);

}  // namespace ops
}  // namespace viper

#endif  // VIPER_MEGAKERNEL_H
