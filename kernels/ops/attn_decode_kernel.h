// Decode attention kernel (T=1 query) with GQA + KV cache.
//
// One block per Q head. Scans the KV cache in chunks of 1024 positions
// with flash-style online softmax (chunk max, rescale, accumulate).
// GQA: Q head h reads KV head h * nKV / nQ (no repeat_kv materialization).
//
// Layout: q [nQ, D]; k_cache/v_cache [T_ctx, nKV, D]; out [nQ, D].
#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace viper { namespace ops {

cudaError_t attn_decode_bf16(
    const __nv_bfloat16* __restrict__ q,        // [nQ, D]
    const __nv_bfloat16* __restrict__ k_cache,  // [T_ctx, nKV, D]
    const __nv_bfloat16* __restrict__ v_cache,  // [T_ctx, nKV, D]
    __nv_bfloat16* __restrict__ out,            // [nQ, D]
    int nQ, int nKV, int D, int T_ctx,
    float scale,
    cudaStream_t stream);

// Q8 KV cache variant: reads INT8 K/V + FP16 per-head scales.
// Same flash-style online softmax, dequantizes inline.
cudaError_t attn_decode_q8(
    const __nv_bfloat16* __restrict__ q,
    const int8_t* __restrict__ k_cache_q8,     // [T_ctx, nKV, D]
    const __nv_bfloat16* __restrict__ k_scales, // [T_ctx, nKV]
    const int8_t* __restrict__ v_cache_q8,     // [T_ctx, nKV, D]
    const __nv_bfloat16* __restrict__ v_scales, // [T_ctx, nKV]
    __nv_bfloat16* __restrict__ out,
    int nQ, int nKV, int D, int T_ctx,
    float scale,
    cudaStream_t stream);

// Batch attention with causal masking (for speculative decode verification).
// Grid: (nQ, M). Token m attends to positions 0..pos_start+m.
//   q: [M, nQ, D]
//   k_cache/v_cache: [T_total, nKV, D] — must already contain entries for pos_start..pos_start+M-1
//   out: [M, nQ, D]
cudaError_t attn_batch_bf16(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k_cache,
    const __nv_bfloat16* __restrict__ v_cache,
    __nv_bfloat16* __restrict__ out,
    int M, int nQ, int nKV, int D,
    int pos_start,
    float scale,
    cudaStream_t stream);

// Batch KV cache append: copies M K/V vectors into the cache.
//   new_k/new_v: [M, nKV, D]
//   k_cache/v_cache: [T_total, nKV, D]
cudaError_t kv_append_batch_bf16(
    const __nv_bfloat16* new_k,
    const __nv_bfloat16* new_v,
    __nv_bfloat16* k_cache,
    __nv_bfloat16* v_cache,
    int M, int nKV, int D, int pos_start,
    cudaStream_t stream);

} }  // namespace viper::ops
