/*
 * viper TurboQuant KV Cache — adaptive mixed-precision KV quantization.
 *
 * Not all attention heads are equally sensitive to quantization.
 * TurboQuant assigns Q8 to sensitive heads and Q4 to insensitive ones,
 * maximizing quality per bit.
 *
 * Default allocation for Nanbeige4.2-3B (nKV=8):
 *   Heads 0-1: Q8  (1.0 bytes/elem) — highest sensitivity (near-lossless)
 *   Heads 2-3: Q6  (0.75 bytes/elem) — medium sensitivity
 *   Heads 4-7: Q4  (0.5 bytes/elem) — lowest sensitivity
 *   Average: 0.69 bytes/elem
 *
 * Memory per position per layer:
 *   TurboQuant: 2 × (2×128×1 + 2×128×0.75 + 4×128×0.5 + 8×2) = 2 × 804 = 1608 bytes
 *   Q8:         2080 bytes (TurboQuant 23% smaller)
 *   Q4:         1056 bytes (TurboQuant 52% larger, but much better quality)
 *   BF16:       4096 bytes
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

// Quantize one KV head to the specified format and write to cache.
__global__ void kv_to_turbo_cache_kernel(
    const __nv_bfloat16* __restrict__ kv_src,
    uint8_t* __restrict__ kv_cache,
    __nv_bfloat16* __restrict__ kv_scales,
    const int* __restrict__ head_fmt,
    const int* __restrict__ head_offsets,
    int pos, int nKV, int HD, int total_offset) {
    const int h = blockIdx.x;
    if (h >= nKV) return;
    const int tid = threadIdx.x;
    int fmt = head_fmt[h];

    uint8_t* cache_ptr = kv_cache + (size_t)pos * total_offset + head_offsets[h];
    __nv_bfloat16* scale_ptr = kv_scales + (size_t)pos * nKV + h;  // already includes head offset

    float gmax = 0.f;
    for (int d = tid; d < HD; d += blockDim.x)
        gmax = fmaxf(gmax, fabsf(__bfloat162float(kv_src[h * HD + d])));
    for (int off = 16; off > 0; off >>= 1)
        gmax = fmaxf(gmax, __shfl_xor_sync(0xffffffff, gmax, off));
    __shared__ float tq_warp_max[8];
    const int wid = tid >> 5, lid = tid & 31;
    if (lid == 0) tq_warp_max[wid] = gmax;
    __syncthreads();
    __shared__ float tq_s_gmax;
    if (tid == 0) {
        float m = tq_warp_max[0];
        for (int i = 1; i < (blockDim.x >> 5); ++i) m = fmaxf(m, tq_warp_max[i]);
        tq_s_gmax = m;
    }
    __syncthreads();

    float divisor = (fmt == 0) ? 127.0f : (fmt == 1) ? 31.0f : 7.0f;
    float scale = fmaxf(tq_s_gmax / divisor, 1e-8f);
    if (tid == 0) scale_ptr[0] = __float2bfloat16(scale);
    __syncthreads();

    float inv_scale = 1.0f / scale;

    if (fmt == 0) {
        // Q8
        for (int d = tid; d < HD; d += blockDim.x)
            cache_ptr[d] = (int8_t)__float2int_rn(
                __bfloat162float(kv_src[h * HD + d]) * inv_scale);
    } else if (fmt == 1) {
        // Q6: 3 bytes per 4 elements (two's complement, no bias)
        if (tid < HD / 4) {
            int base = tid * 4;
            int q[4];
            for (int i = 0; i < 4; ++i)
                q[i] = max(-32, min(31, (int)roundf(
                    __bfloat162float(kv_src[h * HD + base + i]) * inv_scale)));
            cache_ptr[tid * 3]     = (uint8_t)((q[0] & 0x3F) | ((q[1] & 0x3F) << 6));
            cache_ptr[tid * 3 + 1] = (uint8_t)(((q[1] >> 2) & 0x3F) | ((q[2] & 0x3F) << 4));
            cache_ptr[tid * 3 + 2] = (uint8_t)(((q[2] >> 4) & 0x3F) | ((q[3] & 0x3F) << 2));
        }
    } else {
        // Q4: 1 byte per 2 elements (offset encoding +8)
        if (tid < HD / 2) {
            int q0 = max(-8, min(7, (int)roundf(
                __bfloat162float(kv_src[h * HD + tid * 2]) * inv_scale)));
            int q1 = max(-8, min(7, (int)roundf(
                __bfloat162float(kv_src[h * HD + tid * 2 + 1]) * inv_scale)));
            cache_ptr[tid] = (uint8_t)((q0 + 8) | ((q1 + 8) << 4));
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
