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

} }  // namespace viper::ops
