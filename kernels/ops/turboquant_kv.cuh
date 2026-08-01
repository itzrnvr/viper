/*
 * viper TurboQuant KV Cache — mixed-precision per head with PER-32-BLOCK scaling.
 *
 * Not all attention heads are equally sensitive to quantization.
 * TurboQuant assigns higher precision to sensitive heads, lower to insensitive:
 *   Heads 0-1: Q8  (near-lossless for critical attention paths)
 *   Heads 2-3: Q6  (medium precision, good quality/size trade-off)
 *   Heads 4-7: Q4  (aggressive, for less sensitive heads)
 *
 * ALL heads use per-32-block scaling (4 scales per head) — same fix as Q4/Q6/Q8.
 *
 * FORMAT:
 *   Data: contiguous per-head blocks with format-specific sizes (see TurboQuantConfig)
 *   Scales: [nKV, HD/32] FP16 per-block (UNIFORM layout regardless of head format)
 *   Total: ~720 bytes data + 64 bytes scales = 784 bytes/pos/layer (81% vs BF16)
 *
 * VERIFIED: ppl=22.78 vs BF16 20.90 (+9.1%). Task accuracy: ~50%.
 *   TurboQuant is WORSE than uniform Q4 (ppl 20.58). The mixed-precision
 *   approach doesn't help — uniform per-block Q4 is simpler and better.
 *   Kept for reference; use --cache-type 3 (Q4) for best quality/VRAM.
 *
 * BUG HISTORY:
 *   1. Per-head scaling → 82.5% perplexity degradation (38.13 vs 20.90)
 *   2. +32 Q6 bias encoding → incompatible with sign-extension unpack
 *   3. Scale store double-offset (scale_ptr[h] when scale_ptr already had +h)
 *   All fixed by per-32-block rewrite. Final: +9.1% degradation (usable but suboptimal).
 */
#ifndef VIPER_TURBOQUANT_KV_H
#define VIPER_TURBOQUANT_KV_H

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace viper {
namespace ops {

struct TurboQuantConfig {
    static constexpr int N_HEADS = 8;
    // 0=Q8, 1=Q6, 2=Q4
    int fmt[N_HEADS] = {0, 0, 1, 1, 2, 2, 2, 2};  // 2×Q8, 2×Q6, 4×Q4

    static int head_bytes(int fmt, int HD) {
        switch (fmt) {
            case 0: return HD;            // Q8
            case 1: return HD / 4 * 3;    // Q6
            case 2: return HD / 2;        // Q4
            default: return HD * 2;       // BF16
        }
    }

    int total_bytes(int HD) const {
        int total = 0;
        for (int h = 0; h < N_HEADS; ++h)
            total += head_bytes(fmt[h], HD);
        return total;
    }

    void compute_offsets(int* offsets, int HD) const {
        offsets[0] = 0;
        for (int h = 0; h < N_HEADS; ++h)
            offsets[h + 1] = offsets[h] + head_bytes(fmt[h], HD);
    }
};

// Quantize one KV head with per-32-block scaling (matches standalone Q8/Q6/Q4).
// Each warp handles one block of 32 dims independently — no inter-warp reduction.
__global__ void kv_to_turbo_cache_kernel(
    const __nv_bfloat16* __restrict__ kv_src,
    uint8_t* __restrict__ kv_cache,
    __nv_bfloat16* __restrict__ kv_scales,   // [max_seq, nKV, HD/32] per-block
    const int* __restrict__ head_fmt,
    const int* __restrict__ head_offsets,
    int pos, int nKV, int HD, int total_offset) {
    const int h = blockIdx.x;
    if (h >= nKV) return;
    const int tid = threadIdx.x;
    const int wid = tid >> 5, lid = tid & 31;
    const int nBlocks = HD / 32;
    int fmt = head_fmt[h];

    uint8_t* cache_ptr = kv_cache + (size_t)pos * total_offset + head_offsets[h];
    __nv_bfloat16* scale_row = kv_scales + (size_t)pos * nKV * nBlocks + (size_t)h * nBlocks;

    // Each warp handles dims [wid*32, wid*32+31] — warp-only max reduction
    const int d = wid * 32 + lid;
    float val = (d < HD) ? fabsf(__bfloat162float(kv_src[h * HD + d])) : 0.0f;
    for (int off = 16; off > 0; off >>= 1)
        val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, off));

    float divisor = (fmt == 0) ? 127.0f : (fmt == 1) ? 31.0f : 7.0f;
    float scale = fmaxf(val / divisor, 1e-8f);
    if (lid == 0) scale_row[wid] = __float2bfloat16(scale);

    float inv_scale = 1.0f / scale;

    if (fmt == 0) {
        // Q8: each lane writes its dim
        if (d < HD)
            cache_ptr[d] = (int8_t)__float2int_rn(
                __bfloat162float(kv_src[h * HD + d]) * inv_scale);
    } else if (fmt == 1) {
        // Q6: lanes 0-7 pack 4 values per 3 bytes (two's complement)
        if (lid < 8) {
            int base = wid * 32 + lid * 4;
            int q[4];
            for (int i = 0; i < 4; ++i)
                q[i] = max(-32, min(31, (int)roundf(
                    __bfloat162float(kv_src[h * HD + base + i]) * inv_scale)));
            int off = wid * 24 + lid * 3;
            cache_ptr[off]     = (uint8_t)((q[0] & 0x3F) | ((q[1] & 0x3F) << 6));
            cache_ptr[off + 1] = (uint8_t)(((q[1] >> 2) & 0x3F) | ((q[2] & 0x3F) << 4));
            cache_ptr[off + 2] = (uint8_t)(((q[2] >> 4) & 0x3F) | ((q[3] & 0x3F) << 2));
        }
    } else {
        // Q4: lanes 0-15 pack 2 values per byte (offset encoding +8)
        if (lid < 16) {
            int base = wid * 32 + lid * 2;
            int q0 = max(-8, min(7, (int)roundf(
                __bfloat162float(kv_src[h * HD + base]) * inv_scale)));
            int q1 = max(-8, min(7, (int)roundf(
                __bfloat162float(kv_src[h * HD + base + 1]) * inv_scale)));
            cache_ptr[wid * 16 + lid] = (uint8_t)((q0 + 8) | ((q1 + 8) << 4));
        }
    }
}

cudaError_t kv_to_turbo_cache(
    const __nv_bfloat16* kv_src,
    uint8_t* kv_cache, __nv_bfloat16* kv_scales,
    const int* head_fmt, int* head_offsets,
    int pos, int nKV, int HD, int total_offset, cudaStream_t stream) {
    kv_to_turbo_cache_kernel<<<nKV, 128, 0, stream>>>(
        kv_src, kv_cache, kv_scales, head_fmt, head_offsets,
        pos, nKV, HD, total_offset);
    return cudaGetLastError();
}

}  // namespace ops
}  // namespace viper

#endif
