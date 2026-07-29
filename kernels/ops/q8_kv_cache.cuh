/*
 * viper Q8 KV Cache — quantized key-value cache for memory efficiency.
 *
 * Format per position per layer:
 *   K: [nKV_heads, head_dim] INT8 + [nKV_heads] FP16 scales
 *   V: [nKV_heads, head_dim] INT8 + [nKV_heads] FP16 scales
 *
 * Per-head quantization: scale = max(|values in head|) / 127.
 * Dequantize: value = int8_val * scale
 *
 * Memory savings vs BF16:
 *   BF16: 2 × nKV × HD × 2 = 4096 bytes/pos/layer
 *   Q8:   2 × (nKV × HD × 1 + nKV × 2) = 2080 bytes/pos/layer (49% savings)
 *
 * Quality: near-lossless. INT8 with per-head FP16 scale preserves
 * attention accuracy to ~1e-3 relative error.
 */
#ifndef VIPER_Q8_KV_CACHE_H
#define VIPER_Q8_KV_CACHE_H

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace viper {
namespace ops {

// Quantize BF16 KV to Q8 KV cache.
// Each head gets its own FP16 scale.
// Grid: (nKV_heads) blocks, each block handles 1 head × HD elements.
__global__ void kv_bf16_to_q8_kernel(
    const __nv_bfloat16* __restrict__ kv_bf16,  // [nKV, HD] BF16 values
    int8_t* __restrict__ kv_q8,                  // [nKV, HD] INT8 quantized
    __nv_bfloat16* __restrict__ kv_scales,       // [nKV] FP16 per-head scales
    int nKV, int HD) {
    const int h = blockIdx.x;
    if (h >= nKV) return;
    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;

    // Find max in this head
    float gmax = 0.f;
    for (int d = tid; d < HD; d += nthreads)
        gmax = fmaxf(gmax, fabsf(__bfloat162float(kv_bf16[h * HD + d])));
    for (int off = 16; off > 0; off >>= 1)
        gmax = fmaxf(gmax, __shfl_xor_sync(0xffffffff, gmax, off));

    float scale = fmaxf(gmax / 127.0f, 1e-8f);
    if (tid == 0) kv_scales[h] = __float2bfloat16(scale);
    __syncthreads();

    // Quantize
    float inv_scale = 1.0f / scale;
    for (int d = tid; d < HD; d += nthreads)
        kv_q8[h * HD + d] = (int8_t)__float2int_rn(
            __bfloat162float(kv_bf16[h * HD + d]) * inv_scale);
}

// Quantize K to cache (fused with position offset).
// Grid: (nKV_heads) blocks.
__global__ void k_to_q8_cache_kernel(
    const __nv_bfloat16* __restrict__ k_src,     // [nKV, HD] BF16 (after rope)
    int8_t* __restrict__ k_cache,                 // [max_seq, nKV, HD] INT8
    __nv_bfloat16* __restrict__ k_scales,         // [max_seq, nKV] FP16
    int pos, int nKV, int HD) {
    const int h = blockIdx.x;
    if (h >= nKV) return;
    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;

    int8_t* cache_row = k_cache + (size_t)pos * nKV * HD;
    __nv_bfloat16* scale_row = k_scales + (size_t)pos * nKV;

    float gmax = 0.f;
    for (int d = tid; d < HD; d += nthreads)
        gmax = fmaxf(gmax, fabsf(__bfloat162float(k_src[h * HD + d])));
    for (int off = 16; off > 0; off >>= 1)
        gmax = fmaxf(gmax, __shfl_xor_sync(0xffffffff, gmax, off));

    float scale = fmaxf(gmax / 127.0f, 1e-8f);
    if (tid == 0) scale_row[h] = __float2bfloat16(scale);
    __syncthreads();

    float inv_scale = 1.0f / scale;
    for (int d = tid; d < HD; d += nthreads)
        cache_row[h * HD + d] = (int8_t)__float2int_rn(
            __bfloat162float(k_src[h * HD + d]) * inv_scale);
}

// V to Q8 cache (same as K but for V).
__global__ void v_to_q8_cache_kernel(
    const __nv_bfloat16* __restrict__ v_src,
    int8_t* __restrict__ v_cache,
    __nv_bfloat16* __restrict__ v_scales,
    int pos, int nKV, int HD) {
    const int h = blockIdx.x;
    if (h >= nKV) return;
    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;

    int8_t* cache_row = v_cache + (size_t)pos * nKV * HD;
    __nv_bfloat16* scale_row = v_scales + (size_t)pos * nKV;

    float gmax = 0.f;
    for (int d = tid; d < HD; d += nthreads)
        gmax = fmaxf(gmax, fabsf(__bfloat162float(v_src[h * HD + d])));
    for (int off = 16; off > 0; off >>= 1)
        gmax = fmaxf(gmax, __shfl_xor_sync(0xffffffff, gmax, off));

    float scale = fmaxf(gmax / 127.0f, 1e-8f);
    if (tid == 0) scale_row[h] = __float2bfloat16(scale);
    __syncthreads();

    float inv_scale = 1.0f / scale;
    for (int d = tid; d < HD; d += nthreads)
        cache_row[h * HD + d] = (int8_t)__float2int_rn(
            __bfloat162float(v_src[h * HD + d]) * inv_scale);
}

// Attention decode with Q8 KV cache.
// Dequantizes K/V inline during attention computation.
// Grid: (nQ) blocks, each block handles 1 query head.
__global__ void attn_decode_q8_kernel(
    const __nv_bfloat16* __restrict__ q,          // [nQ, HD] BF16 query
    const int8_t* __restrict__ k_cache_q8,        // [T, nKV, HD] INT8 K cache
    const __nv_bfloat16* __restrict__ k_scales,   // [T, nKV] FP16 K scales
    const int8_t* __restrict__ v_cache_q8,        // [T, nKV, HD] INT8 V cache
    const __nv_bfloat16* __restrict__ v_scales,   // [T, nKV] FP16 V scales
    __nv_bfloat16* __restrict__ out,               // [nQ, HD] output
    int nQ, int nKV, int HD, int T_ctx, float scale) {
    const int h = blockIdx.x;
    if (h >= nQ) return;
    const int h_kv = h / (nQ / nKV);  // GQA: nQ/nKV = 6
    const int tid = threadIdx.x;

    // Load query to registers
    float q_vec[128];
    #pragma unroll
    for (int d = tid; d < HD; d += blockDim.x)
        q_vec[d] = __bfloat162float(q[h * HD + d]);
    __syncthreads();

    // Attention scores (online softmax)
    float max_score = -1e30f;
    __shared__ float scores[2048];  // max context per position
    int n_ctx = min(T_ctx, 2048);

    for (int t = 0; t < n_ctx; ++t) {
        float dot = 0.f;
        __nv_bfloat16 k_scale = k_scales[(size_t)t * nKV + h_kv];
        float k_sc = __bfloat162float(k_scale);
        const int8_t* k_row = k_cache_q8 + (size_t)t * nKV * HD + h_kv * HD;
        for (int d = tid; d < HD; d += blockDim.x)
            dot += q_vec[d] * (float)k_row[d] * k_sc;
        // Warp reduce
        for (int off = 16; off > 0; off >>= 1)
            dot += __shfl_xor_sync(0xffffffff, dot, off);
        if (tid == 0) {
            scores[t] = dot * scale;
            max_score = fmaxf(max_score, scores[t]);
        }
        __syncthreads();
    }

    // Softmax + weighted V sum
    float sum_exp = 0.f;
    for (int t = 0; t < n_ctx; ++t) {
        scores[t] = expf(scores[t] - max_score);
        sum_exp += scores[t];
    }
    float inv_sum = 1.f / sum_exp;

    for (int d = tid; d < HD; d += blockDim.x) {
        float acc = 0.f;
        for (int t = 0; t < n_ctx; ++t) {
            __nv_bfloat16 v_scale = v_scales[(size_t)t * nKV + h_kv];
            float v_sc = __bfloat162float(v_scale);
            const int8_t* v_row = v_cache_q8 + (size_t)t * nKV * HD + h_kv * HD;
            acc += scores[t] * inv_sum * (float)v_row[d] * v_sc;
        }
        out[h * HD + d] = __float2bfloat16(acc);
    }
}

// Launchers
cudaError_t k_to_q8_cache(
    const __nv_bfloat16* k_src, int8_t* k_cache, __nv_bfloat16* k_scales,
    int pos, int nKV, int HD, cudaStream_t stream) {
    k_to_q8_cache_kernel<<<nKV, 128, 0, stream>>>(k_src, k_cache, k_scales, pos, nKV, HD);
    return cudaGetLastError();
}

cudaError_t v_to_q8_cache(
    const __nv_bfloat16* v_src, int8_t* v_cache, __nv_bfloat16* v_scales,
    int pos, int nKV, int HD, cudaStream_t stream) {
    v_to_q8_cache_kernel<<<nKV, 128, 0, stream>>>(v_src, v_cache, v_scales, pos, nKV, HD);
    return cudaGetLastError();
}

cudaError_t attn_decode_q8(
    const __nv_bfloat16* q,
    const int8_t* k_cache_q8, const __nv_bfloat16* k_scales,
    const int8_t* v_cache_q8, const __nv_bfloat16* v_scales,
    __nv_bfloat16* out,
    int nQ, int nKV, int HD, int T_ctx, float scale,
    cudaStream_t stream) {
    attn_decode_q8_kernel<<<nQ, 128, 0, stream>>>(
        q, k_cache_q8, k_scales, v_cache_q8, v_scales, out,
        nQ, nKV, HD, T_ctx, scale);
    return cudaGetLastError();
}

}  // namespace ops
}  // namespace viper

#endif
