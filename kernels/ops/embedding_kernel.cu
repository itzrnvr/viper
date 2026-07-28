/*
 * viper Embedding gather kernel — implementation
 *
 * PURPOSE: see embedding_kernel.h
 *
 * IMPLEMENTATION:
 *   Each block handles one (b, t) row. Threads cooperate over H columns.
 *   Vectorized float4 loads/stores when H is multiple of 8.
 *
 * CORRECTNESS:
 *   - Bounds check on token id < V; OOB writes zeros.
 *   - Hidden dim must be multiple of 8 for the vectorized path.
 *
 * SAFETY:
 *   - No allocation inside kernel.
 *   - SMEM: 0 bytes (streaming copy via registers and L1).
 */
#include "embedding_kernel.h"
#include <cuda_runtime.h>

namespace viper {
namespace ops {

template <int BLOCK>
__global__ void embedding_gather_kernel(
    const __nv_bfloat16* __restrict__ table,
    const int32_t* __restrict__ token_ids_i32,
    const int64_t* __restrict__ token_ids_i64,
    __nv_bfloat16* __restrict__ out,
    int B,
    int T,
    int V,
    int H,
    bool use_i64) {
    const int b = blockIdx.y;
    const int t = blockIdx.x;
    const int tid = threadIdx.x;

    const int row_idx = b * T + t;
    const int32_t token = use_i64 ? static_cast<int32_t>(token_ids_i64[row_idx])
                                   : token_ids_i32[row_idx];
    if (token < 0 || token >= V) {
        for (int i = tid; i < H; i += BLOCK) {
            out[row_idx * H + i] = __float2bfloat16(0.0f);
        }
        return;
    }

    const __nv_bfloat16* src = table + token * H;
    __nv_bfloat16* dst = out + row_idx * H;

    constexpr int VEC = 8;
    const int per_thread = H / BLOCK;
    if ((per_thread % VEC) == 0) {
        const int n_vec = per_thread / VEC;
        for (int i = 0; i < n_vec; ++i) {
            const int idx = tid * VEC + i * BLOCK * VEC;
            const float4 v = *reinterpret_cast<const float4*>(src + idx);
            *reinterpret_cast<float4*>(dst + idx) = v;
        }
    } else {
        for (int i = tid; i < H; i += BLOCK) {
            dst[i] = src[i];
        }
    }
}

cudaError_t embedding_gather_bf16_i32(
    const __nv_bfloat16* table,
    const int32_t* token_ids,
    __nv_bfloat16* out,
    int B,
    int T,
    int V,
    int H,
    cudaStream_t stream) {
    if (!table || !token_ids || !out || B <= 0 || T <= 0 || V <= 0 || H <= 0) {
        return cudaErrorInvalidValue;
    }
    if (H % 8 != 0) {
        return cudaErrorInvalidValue;
    }
    constexpr int BLOCK = 128;
    dim3 grid(T, B);
    dim3 block(BLOCK);
    embedding_gather_kernel<BLOCK><<<grid, block, 0, stream>>>(
        table, token_ids, nullptr, out, B, T, V, H, false);
    return cudaGetLastError();
}

cudaError_t embedding_gather_bf16_i64(
    const __nv_bfloat16* table,
    const int64_t* token_ids,
    __nv_bfloat16* out,
    int B,
    int T,
    int V,
    int H,
    cudaStream_t stream) {
    if (!table || !token_ids || !out || B <= 0 || T <= 0 || V <= 0 || H <= 0) {
        return cudaErrorInvalidValue;
    }
    if (H % 8 != 0) {
        return cudaErrorInvalidValue;
    }
    constexpr int BLOCK = 128;
    dim3 grid(T, B);
    dim3 block(BLOCK);
    embedding_gather_kernel<BLOCK><<<grid, block, 0, stream>>>(
        table, nullptr, token_ids, out, B, T, V, H, true);
    return cudaGetLastError();
}

}  // namespace ops
}  // namespace viper
