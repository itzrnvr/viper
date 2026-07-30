/*
 * viper Q4 KV Cache — 4-bit quantized key-value cache.
 *
 * Format per position per layer:
 *   K: [nKV, HD/2] packed Q4 (2 values per byte) + [nKV] FP16 scales
 *   V: [nKV, HD/2] packed Q4 + [nKV] FP16 scales
 *
 * Packing: 2 × 4-bit values → 1 byte. Offset encoding: stored = value + 8.
 *
 * Memory per position per layer:
 *   Q4:   2 × (nKV × HD/2 + nKV × 2) = 2 × (512 + 16) = 1056 bytes
 *   Q6:   2 × (nKV × HD/4 × 3 + nKV × 2) = 2 × (192 + 16) = 416 bytes
 *   Wait, Q6 is smaller? No: Q6 per head = HD/4 × 3 = 96 bytes, Q4 per head = HD/2 = 64 bytes.
 *   Q4: 2 × (8 × 64 + 8 × 2) = 2 × 528 = 1056 bytes ← CORRECT
 *   Q6: 2 × (8 × 96 + 8 × 2) = 2 × 784 = 1568 bytes
 *   So Q4 IS smaller than Q6. ✓
 *
 *   BF16: 2 × 8 × 128 × 2 = 4096 bytes
 *   Q4 is 4× smaller than BF16.
 *
 * 128K context on 8GB VRAM:
 *   Q4 KV: 44 slots × 128K × 1056 = 5.7 GB + 2.3 GB model = 8.0 GB ✓
 *   BF16 KV: 44 × 128K × 4096 = 22.5 GB ✗ (impossible)
 *
 * Quality: 4-bit KV cache has measurable attention quality loss.
 * TurboQuant addresses this by using Q8 for sensitive heads, Q4 for others.
 *
 * BUG HISTORY (2026-07-30):
 *   Same two bugs as Q8 (inter-warp reduction missing + scale store eaten by
 *   SWAP). The scale_store deletion was the THIRD instance of the same pattern:
 *   SWAP edits eating the line after scale computation. With correct scales,
 *   Q4 changes from random garbage to degraded-but-coherent output.
 *
 *   Verified with --cache-type 3, prompt "What is 2+2?":
 *     Before scale fix: '?/ico number开工th舍ip...' (random)
 *     After scale fix:  '<think>{"question": "What is 2' (coherent, degraded)
 *
 *   This matches production Q4_0 KV cache behavior (llama.cpp). Per-head Q4 is
 *   viable for extreme VRAM savings (128K context on 8GB). If quality needs
 * improving: switch to per-32-block scales (like Q4_0) instead of per-128-head.
 */
#ifndef VIPER_Q4_KV_CACHE_H
#define VIPER_Q4_KV_CACHE_H

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace viper {
namespace ops {

// Quantize BF16 KV to Q4 cache with per-32-block scaling (like Q4_0).
// Each warp independently handles one block of 32 values — no inter-warp reduction.
// Grid: (nKV_heads) blocks, 128 threads (4 warps × 32 lanes = 4 blocks).
__global__ void kv_to_q4_cache_kernel(
    const __nv_bfloat16* __restrict__ kv_src,     // [nKV, HD]
    uint8_t* __restrict__ kv_cache,                // [max_seq, nKV, HD/2]
    __nv_bfloat16* __restrict__ kv_scales,         // [max_seq, nKV, HD/32] per-block scales
    int pos, int nKV, int HD) {
    const int h = blockIdx.x;
    if (h >= nKV) return;
    const int tid = threadIdx.x;
    const int wid = tid >> 5;    // warp ID = block ID (0-3 for HD=128)
    const int lid = tid & 31;   // lane within warp = dim within block
    const int nBlocks = HD / 32;

    uint8_t* cache_row = kv_cache + (size_t)pos * nKV * (HD / 2) + (size_t)h * (HD / 2);
    __nv_bfloat16* scale_row = kv_scales + (size_t)pos * nKV * nBlocks + (size_t)h * nBlocks;

    // Each warp handles dims [wid*32, wid*32+31] independently
    const int d = wid * 32 + lid;
    float val = (d < HD) ? fabsf(__bfloat162float(kv_src[h * HD + d])) : 0.0f;

    // Warp-only reduction for block max (no inter-warp needed!)
    for (int off = 16; off > 0; off >>= 1)
        val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, off));
    float gmax = val;  // all lanes now have the block max

    float scale = fmaxf(gmax / 7.0f, 1e-8f);
    if (lid == 0) scale_row[wid] = __float2bfloat16(scale);

    float inv_scale = 1.0f / scale;
    // Each of 16 lanes packs 2 consecutive dims into 1 byte
    if (lid < 16) {
        int base = wid * 32 + lid * 2;
        int q0 = max(-8, min(7, (int)roundf(
            __bfloat162float(kv_src[h * HD + base]) * inv_scale)));
        int q1 = max(-8, min(7, (int)roundf(
            __bfloat162float(kv_src[h * HD + base + 1]) * inv_scale)));
        cache_row[wid * 16 + lid] = (uint8_t)((q0 + 8) | ((q1 + 8) << 4));
    }
}

// Dequantize Q4 KV for one position to BF16 (for attention bridge).
// Grid: (nKV) blocks, 128 threads.
__global__ void q4_kv_dequant_pos_kernel(
    const uint8_t* __restrict__ kv_q4,            // [nKV, HD/2] at position pos
    const __nv_bfloat16* __restrict__ scales,      // [nKV] at position pos
    __nv_bfloat16* __restrict__ out_bf16,          // [nKV, HD]
    int nKV, int HD) {
    const int h = blockIdx.x;
    if (h >= nKV) return;
    const int tid = threadIdx.x;
    float sc = __bfloat162float(scales[h]);
    if (tid < HD / 2) {
        uint8_t packed = kv_q4[h * (HD / 2) + tid];
        int q0 = (packed & 0xF) - 8;
        int q1 = (packed >> 4) - 8;
        out_bf16[h * HD + tid * 2] = __float2bfloat16(q0 * sc);
        out_bf16[h * HD + tid * 2 + 1] = __float2bfloat16(q1 * sc);
    }
}

// Dequantize ALL positions Q4→BF16 (for attention bridge approach).
// Grid: (T_ctx, nKV).
__global__ void q4_kv_dequant_range_kernel(
    const uint8_t* __restrict__ q4_cache,          // [T, nKV, HD/2]
    const __nv_bfloat16* __restrict__ scales,       // [T, nKV]
    __nv_bfloat16* __restrict__ out_bf16,           // [T, nKV, HD]
    int T_ctx, int nKV, int HD) {
    const int t = blockIdx.x;
    const int h = blockIdx.y;
    if (t >= T_ctx || h >= nKV) return;
    const int tid = threadIdx.x;
    float sc = __bfloat162float(scales[(size_t)t * nKV + h]);
    const uint8_t* src = q4_cache + (size_t)t * nKV * (HD / 2) + h * (HD / 2);
    __nv_bfloat16* dst = out_bf16 + (size_t)t * nKV * HD + h * HD;
    for (int i = tid; i < HD / 2; i += blockDim.x) {
        uint8_t packed = src[i];
        dst[i * 2] = __float2bfloat16(((int)(packed & 0xF) - 8) * sc);
        dst[i * 2 + 1] = __float2bfloat16(((int)(packed >> 4) - 8) * sc);
    }
}

// Launchers
cudaError_t k_to_q4_cache(
    const __nv_bfloat16* k_src, uint8_t* k_cache, __nv_bfloat16* k_scales,
    int pos, int nKV, int HD, cudaStream_t stream) {
    kv_to_q4_cache_kernel<<<nKV, 128, 0, stream>>>(k_src, k_cache, k_scales, pos, nKV, HD);
    return cudaGetLastError();
}

cudaError_t v_to_q4_cache(
    const __nv_bfloat16* v_src, uint8_t* v_cache, __nv_bfloat16* v_scales,
    int pos, int nKV, int HD, cudaStream_t stream) {
    kv_to_q4_cache_kernel<<<nKV, 128, 0, stream>>>(v_src, v_cache, v_scales, pos, nKV, HD);
    return cudaGetLastError();
}

cudaError_t q4_kv_dequant_range(
    const uint8_t* q4_cache, const __nv_bfloat16* scales,
    __nv_bfloat16* out_bf16, int T_ctx, int nKV, int HD,
    cudaStream_t stream) {
    dim3 grid(T_ctx, nKV);
    q4_kv_dequant_range_kernel<<<grid, 128, 0, stream>>>(
        q4_cache, scales, out_bf16, T_ctx, nKV, HD);
    return cudaGetLastError();
}

}  // namespace ops
}  // namespace viper

#endif
