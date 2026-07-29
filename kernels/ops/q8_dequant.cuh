/*
 * Q8 KV cache dequantize: reads Q8 INT8 cache + FP16 scales, writes BF16 scratch.
 * Used before the existing attn_decode_bf16 call.
 * Proves the Q8 write/read pipeline before writing a native Q8 attention kernel.
 */
#ifndef VIPER_Q8_KV_DEQUANT_H
#define VIPER_Q8_KV_DEQUANT_H
#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace viper {
namespace ops {

// Dequantize Q8 K or V cache for one position to BF16.
// Grid: (nKV) blocks, each handles 1 head × HD elements.
__global__ void q8_kv_dequant_kernel(
    const int8_t* __restrict__ q8_data,       // [nKV, HD] at position pos
    const __nv_bfloat16* __restrict__ scales,  // [nKV] at position pos
    __nv_bfloat16* __restrict__ out_bf16,      // [nKV, HD]
    int nKV, int HD) {
    const int h = blockIdx.x;
    if (h >= nKV) return;
    const int tid = threadIdx.x;
    float sc = __bfloat162float(scales[h]);
    for (int d = tid; d < HD; d += blockDim.x)
        out_bf16[h * HD + d] = __float2bfloat16((float)q8_data[h * HD + d] * sc);
}

// Dequantize ALL positions from pos 0 to T_ctx-1.
// Grid: (T_ctx, nKV).
__global__ void q8_kv_dequant_range_kernel(
    const int8_t* __restrict__ q8_cache,       // [T, nKV, HD]
    const __nv_bfloat16* __restrict__ scales,   // [T, nKV]
    __nv_bfloat16* __restrict__ out_bf16,       // [T, nKV, HD]
    int T_ctx, int nKV, int HD) {
    const int t = blockIdx.x;
    const int h = blockIdx.y;
    if (t >= T_ctx || h >= nKV) return;
    const int tid = threadIdx.x;
    float sc = __bfloat162float(scales[(size_t)t * nKV + h]);
    const int8_t* src = q8_cache + (size_t)t * nKV * HD + h * HD;
    __nv_bfloat16* dst = out_bf16 + (size_t)t * nKV * HD + h * HD;
    for (int d = tid; d < HD; d += blockDim.x)
        dst[d] = __float2bfloat16((float)src[d] * sc);
}

cudaError_t q8_kv_dequant_range(
    const int8_t* q8_cache, const __nv_bfloat16* scales,
    __nv_bfloat16* out_bf16, int T_ctx, int nKV, int HD,
    cudaStream_t stream) {
    dim3 grid(T_ctx, nKV);
    q8_kv_dequant_range_kernel<<<grid, 128, 0, stream>>>(
        q8_cache, scales, out_bf16, T_ctx, nKV, HD);
    return cudaGetLastError();
}

}  // namespace ops
}  // namespace viper
#endif
