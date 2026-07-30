/*
 * viper Flash Decoding attention kernel (rewritten 2026-07-30).
 *
 * Online softmax with tiled K/V processing — O(TILE) SMEM regardless
 * of context length. Enables 128K+ context with constant memory usage.
 *
 * Algorithm (per query head):
 *   m = -inf, l = 0, acc[D] = 0
 *   for each tile of K/V positions:
 *     scores[tile] = Q · K_tile^T * scale     (per-warp, zero block syncs)
 *     m_new = max(m, max(scores))              (BUG FIX: was max(scores) only → NaN)
 *     l = l * exp(m - m_new) + sum(exp(scores - m_new))
 *     acc = acc * exp(m - m_new) + sum(exp(scores - m_new) * V_tile)
 *     m = m_new
 *   output = acc / l
 *
 * Design (advisor-recommended):
 *   - Q loaded into SMEM once (no per-position register pressure)
 *   - Each warp owns TILE/4 positions, uses warp-only reductions
 *   - Only 6 __syncthreads per tile (was 2×TILE=256 in old version)
 *   - Inter-warp reduction via serial loop (not partial-mask __shfl)
 *
 * BUG HISTORY (2026-07-30):
 *   1. NaN: m_new = warp_max[0] (tile max only). When a later tile's max < running
 *      max, expf(m_run - m_new) exploded → inf → inf/inf = NaN. Fix: fmaxf(m_run, ...).
 *   2. UB: __shfl_xor_sync(0xffffffff, ...) called by only 4 lanes (wid==0 && lid<4).
 *      All masked threads must execute shfl. Fix: serial if(tid==0) loop.
 *   3. Perf: 2 __syncthreads per K position (2×TILE per tile). Fix: per-warp ownership.
 *   4. Dead code: old lines 70-80 computed a dot product that was discarded.
 */
#ifndef VIPER_FLASH_DECODE_H
#define VIPER_FLASH_DECODE_H

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace viper {
namespace ops {

// Flash decode for single-token attention with BF16 KV cache.
// Grid: (nQ) blocks. Block: 128 threads (4 warps × 32 lanes).
// HD must be 128 (4 dims per lane). TILE must be divisible by 4.
template <int HD, int TILE>
__global__ void flash_decode_bf16_kernel(
    const __nv_bfloat16* __restrict__ q,         // [nQ, HD]
    const __nv_bfloat16* __restrict__ k_cache,   // [T, nKV, HD]
    const __nv_bfloat16* __restrict__ v_cache,   // [T, nKV, HD]
    __nv_bfloat16* __restrict__ out,              // [nQ, HD]
    int nQ, int nKV, int T_ctx, float scale) {

    const int h = blockIdx.x;
    if (h >= nQ) return;
    const int h_kv = (int)((long long)h * nKV / nQ);
    const int tid = threadIdx.x;
    const int wid = tid >> 5;
    const int lid = tid & 31;

    // --- Load Q into SMEM (once, cooperative) ---
    __shared__ float q_smem[HD];
    if (tid < HD) q_smem[tid] = __bfloat162float(q[h * HD + tid]);
    __syncthreads();

    // --- Shared scratch ---
    __shared__ float tile_scores[TILE];
    __shared__ float smem_max[4];
    __shared__ float smem_sum[4];

    // --- Per-thread online softmax accumulators (one per output dim) ---
    float m_run = -1e30f;
    float l_run = 0.0f;
    float acc = (tid < HD) ? 0.0f : 0.0f;  // each thread owns one dim

    const size_t kv_stride = (size_t)nKV * HD;
    const size_t kv_off = (size_t)h_kv * HD;
    const int pos_per_warp = TILE / 4;  // each warp handles this many positions

    // --- Process K/V in tiles ---
    for (int t0 = 0; t0 < T_ctx; t0 += TILE) {
        const int tile_n = min(TILE, T_ctx - t0);

        // === Phase 1: Compute attention scores (per-warp, ZERO block syncs) ===
        // Each warp owns positions [wid*pos_per_warp, wid*pos_per_warp+pos_per_warp-1]
        for (int pi = 0; pi < pos_per_warp; ++pi) {
            const int tile_pos = wid * pos_per_warp + pi;
            if (tile_pos >= tile_n) break;
            const int t = t0 + tile_pos;
            const __nv_bfloat16* k_row = k_cache + (size_t)t * kv_stride + kv_off;

            // Each lane handles HD/32 = 4 dims
            float dot = 0.f;
            #pragma unroll
            for (int d = 0; d < HD; d += 32)
                dot += q_smem[d + lid] * __bfloat162float(k_row[d + lid]);

            // Warp-only reduction (no block sync)
            for (int off = 16; off > 0; off >>= 1)
                dot += __shfl_xor_sync(0xffffffff, dot, off);

            if (lid == 0) tile_scores[tile_pos] = dot * scale;
        }
        __syncthreads();  // ← single sync per tile for scores

        // === Phase 2: Online softmax update ===
        // Find tile max across all tile_n positions
        float m_tile = -1e30f;
        for (int ti = tid; ti < tile_n; ti += blockDim.x)
            m_tile = fmaxf(m_tile, tile_scores[ti]);
        for (int off = 16; off > 0; off >>= 1)
        if (lid == 0) smem_max[wid] = m_tile;
        __syncthreads();

        if (tid == 0) {
            float m = smem_max[0];
            for (int i = 1; i < 4; ++i) m = fmaxf(m, smem_max[i]);
            smem_max[0] = m;
        }
        __syncthreads();

        // BUG FIX: m_new must be max(running, tile), not tile alone
        const float m_new = fmaxf(m_run, smem_max[0]);
        const float scale_old = expf(m_run - m_new);

        // Compute exp(score - m_new) and sum
        float l_tile = 0.f;
        for (int ti = tid; ti < tile_n; ti += blockDim.x) {
            tile_scores[ti] = expf(tile_scores[ti] - m_new);
            l_tile += tile_scores[ti];
        }
        for (int off = 16; off > 0; off >>= 1)
            l_tile += __shfl_xor_sync(0xffffffff, l_tile, off);
        if (lid == 0) smem_sum[wid] = l_tile;
        __syncthreads();

        if (tid == 0) {
            float l_all = 0;
            for (int i = 0; i < 4; ++i) l_all += smem_sum[i];
            smem_sum[0] = l_all;
        }
        __syncthreads();

        l_run = l_run * scale_old + smem_sum[0];
        acc *= scale_old;
        m_run = m_new;

        // === Phase 3: Accumulate weighted V (each thread = one dim) ===
        if (tid < HD) {
            for (int ti = 0; ti < tile_n; ++ti) {
                const int t = t0 + ti;
                const float weight = tile_scores[ti];
                const __nv_bfloat16* v_row = v_cache + (size_t)t * kv_stride + kv_off;
                acc += weight * __bfloat162float(v_row[tid]);
            }
        }
        __syncthreads();  // tile_scores can be reused for next tile
    }

    // === Final normalization ===
    const float inv_l = (l_run > 1e-30f) ? (1.0f / l_run) : 0.0f;
    if (tid < HD)
        out[h * HD + tid] = __float2bfloat16(acc * inv_l);
}

// Launcher
cudaError_t flash_decode_bf16(
    const __nv_bfloat16* q,
    const __nv_bfloat16* k_cache,
    const __nv_bfloat16* v_cache,
    __nv_bfloat16* out,
    int nQ, int nKV, int HD, int T_ctx, float scale,
    cudaStream_t stream) {
    if (HD != 128) return cudaErrorInvalidValue;
    constexpr int TILE = 128;
    flash_decode_bf16_kernel<128, TILE><<<nQ, 128, 0, stream>>>(
        q, k_cache, v_cache, out, nQ, nKV, T_ctx, scale);
    return cudaGetLastError();
}

}  // namespace ops
}  // namespace viper

#endif
