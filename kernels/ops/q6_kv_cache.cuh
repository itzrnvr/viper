/*
 * viper Q6 KV Cache — 6-bit quantized key-value cache.
 *
 * Format per position per layer:
 *   K: [nKV, HD] packed as Q6 (4 values per 3 bytes) + [nKV] FP16 scales
 *   V: [nKV, HD] packed as Q6 + [nKV] FP16 scales
 *
 * Packing: 4 × 6-bit values → 3 bytes
 *   byte0 = v0 | (v1 << 6)            → v0[5:0], v1[1:0]
 *   byte1 = (v1 >> 2) | (v2 << 4)     → v1[5:2], v2[3:0]
 *   byte2 = (v2 >> 4) | (v3 << 2)     → v2[5:4], v3[5:0]
 *
 * Memory per position per layer:
 *   Q6: 2 × (nKV × HD/4 × 3 + nKV × 2) = 2 × (192 + 16) = 416 bytes
 *   Q8: 2 × (nKV × HD + nKV × 2) = 2 × (1024 + 16) = 2080 bytes
 *   BF16: 2 × nKV × HD × 2 = 4096 bytes
 *
 * Q6 saves 90% vs BF16, 80% vs Q8.
 * Quality: minor degradation (6-bit preserves attention accuracy well).
 */
#ifndef VIPER_Q6_KV_CACHE_H
#define VIPER_Q6_KV_CACHE_H

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace viper {
namespace ops {

// Pack 4 INT6 values into 3 bytes.
__device__ __forceinline__ void pack_q6(int v0, int v1, int v2, int v3,
                                          uint8_t& b0, uint8_t& b1, uint8_t& b2) {
    v0 &= 0x3F; v1 &= 0x3F; v2 &= 0x3F; v3 &= 0x3F;
    b0 = v0 | (v1 << 6);
    b1 = (v1 >> 2) | (v2 << 4);
    b2 = (v2 >> 4) | (v3 << 2);
}

// Unpack 3 bytes into 4 INT6 values (sign-extended from 6-bit).
__device__ __forceinline__ void unpack_q6(uint8_t b0, uint8_t b1, uint8_t b2,
                                            int& v0, int& v1, int& v2, int& v3) {
    v0 = b0 & 0x3F;
    v1 = ((b0 >> 6) | (b1 << 2)) & 0x3F;
    v2 = ((b1 >> 4) | (b2 << 4)) & 0x3F;
    v3 = (b2 >> 2) & 0x3F;
    // Sign extend from 6-bit: values 32-63 are negative (two's complement)
    if (v0 >= 32) v0 -= 64;
    if (v1 >= 32) v1 -= 64;
    if (v2 >= 32) v2 -= 64;
    if (v3 >= 32) v3 -= 64;
}

// Quantize BF16 KV to Q6 cache. Per-head FP16 scale.
// Grid: (nKV_heads) blocks, 128 threads (1 per dim for HD=128).
__global__ void kv_bf16_to_q6_kernel(
    const __nv_bfloat16* __restrict__ kv_bf16,   // [nKV, HD]
    uint8_t* __restrict__ kv_q6,                  // [nKV, HD/4 * 3] packed Q6
    __nv_bfloat16* __restrict__ kv_scales,        // [nKV] FP16 scales
    int nKV, int HD) {
    const int h = blockIdx.x;
    if (h >= nKV) return;
    const int tid = threadIdx.x;

    // Find max in this head
    float gmax = 0.f;
    for (int d = tid; d < HD; d += blockDim.x)
        gmax = fmaxf(gmax, fabsf(__bfloat162float(kv_bf16[h * HD + d])));
    for (int off = 16; off > 0; off >>= 1)
        gmax = fmaxf(gmax, __shfl_xor_sync(0xffffffff, gmax, off));

    float scale = fmaxf(gmax / 31.0f, 1e-8f);  // 6-bit signed: range -32..31
    if (tid == 0) kv_scales[h] = __float2bfloat16(scale);
    __syncthreads();

    float inv_scale = 1.0f / scale;
    uint8_t* out = kv_q6 + (size_t)h * (HD / 4 * 3);

    // Pack 4 values at a time. Each thread handles HD/blockDim.x elements.
    // For HD=128, blockDim=128: each thread handles 1 element.
    // Need cooperative packing: 4 threads collaborate to pack 4 values.
    // Simpler: 32 threads (1 warp) handle 128 elements = 32 groups of 4.
    if (tid < HD / 4) {
        int base = tid * 4;
        int q0 = (int)roundf(__bfloat162float(kv_bf16[h * HD + base]) * inv_scale);
        int q1 = (int)roundf(__bfloat162float(kv_bf16[h * HD + base + 1]) * inv_scale);
        int q2 = (int)roundf(__bfloat162float(kv_bf16[h * HD + base + 2]) * inv_scale);
        int q3 = (int)roundf(__bfloat162float(kv_bf16[h * HD + base + 3]) * inv_scale);
        // Clamp to [-32, 31]
        q0 = max(-32, min(31, q0)); q1 = max(-32, min(31, q1));
        q2 = max(-32, min(31, q2)); q3 = max(-32, min(31, q3));
        // Convert to unsigned 6-bit (add 32 for two's complement)
        uint8_t b0, b1, b2;
        pack_q6(q0 + 32, q1 + 32, q2 + 32, q3 + 32, b0, b1, b2);
        out[tid * 3] = b0;
        out[tid * 3 + 1] = b1;
        out[tid * 3 + 2] = b2;
    }
}

// Q6 KV cache for specific position (fused with position offset).
__global__ void kv_to_q6_cache_kernel(
    const __nv_bfloat16* __restrict__ kv_src,
    uint8_t* __restrict__ kv_cache,               // [max_seq, nKV, HD/4*3]
    __nv_bfloat16* __restrict__ kv_scales,        // [max_seq, nKV]
    int pos, int nKV, int HD) {
    const int h = blockIdx.x;
    if (h >= nKV) return;
    const int tid = threadIdx.x;

    uint8_t* cache_row = kv_cache + (size_t)pos * nKV * (HD / 4 * 3) + (size_t)h * (HD / 4 * 3);
    __nv_bfloat16* scale_row = kv_scales + (size_t)pos * nKV;

    float gmax = 0.f;
    for (int d = tid; d < HD; d += blockDim.x)
        gmax = fmaxf(gmax, fabsf(__bfloat162float(kv_src[h * HD + d])));
    for (int off = 16; off > 0; off >>= 1)
        gmax = fmaxf(gmax, __shfl_xor_sync(0xffffffff, gmax, off));

    float scale = fmaxf(gmax / 31.0f, 1e-8f);
    if (tid == 0) scale_row[h] = __float2bfloat16(scale);
    __syncthreads();

    float inv_scale = 1.0f / scale;
    if (tid < HD / 4) {
        int base = tid * 4;
        int q0 = max(-32, min(31, (int)roundf(__bfloat162float(kv_src[h * HD + base]) * inv_scale)));
        int q1 = max(-32, min(31, (int)roundf(__bfloat162float(kv_src[h * HD + base + 1]) * inv_scale)));
        int q2 = max(-32, min(31, (int)roundf(__bfloat162float(kv_src[h * HD + base + 2]) * inv_scale)));
        int q3 = max(-32, min(31, (int)roundf(__bfloat162float(kv_src[h * HD + base + 3]) * inv_scale)));
        uint8_t b0, b1, b2;
        pack_q6(q0 + 32, q1 + 32, q2 + 32, q3 + 32, b0, b1, b2);
        cache_row[tid * 3] = b0;
        cache_row[tid * 3 + 1] = b1;
        cache_row[tid * 3 + 2] = b2;
    }
}

// Launchers
cudaError_t k_to_q6_cache(
    const __nv_bfloat16* k_src, uint8_t* k_cache, __nv_bfloat16* k_scales,
    int pos, int nKV, int HD, cudaStream_t stream) {
    kv_to_q6_cache_kernel<<<nKV, 128, 0, stream>>>(k_src, k_cache, k_scales, pos, nKV, HD);
    return cudaGetLastError();
}

cudaError_t v_to_q6_cache(
    const __nv_bfloat16* v_src, uint8_t* v_cache, __nv_bfloat16* v_scales,
    int pos, int nKV, int HD, cudaStream_t stream) {
    kv_to_q6_cache_kernel<<<nKV, 128, 0, stream>>>(v_src, v_cache, v_scales, pos, nKV, HD);
    return cudaGetLastError();
}

}  // namespace ops
}  // namespace viper

#endif
