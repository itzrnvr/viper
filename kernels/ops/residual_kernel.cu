/*
 * viper Residual add kernel — implementation
 *
 * PURPOSE: see residual_kernel.h
 *
 * IMPLEMENTATION:
 *   Vectorized float4 (8 bf16) loads/stores. fp32 accumulation per element.
 *
 * SAFETY:
 *   - No allocation.
 *   - SMEM: 0 bytes.
 */
#include "residual_kernel.h"
#include <cuda_runtime.h>

namespace viper {
namespace ops {

template <bool INPLACE>
__global__ void residual_add_kernel(
    __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ y,
    __nv_bfloat16* __restrict__ out,
    int N) {
    constexpr int VEC = 8;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int n_vec = N / VEC;

    if (tid < n_vec) {
        float4 x_pack = *reinterpret_cast<const float4*>(x + tid * VEC);
        float4 y_pack = *reinterpret_cast<const float4*>(y + tid * VEC);

        const __nv_bfloat16* x_arr = reinterpret_cast<const __nv_bfloat16*>(&x_pack);
        const __nv_bfloat16* y_arr = reinterpret_cast<const __nv_bfloat16*>(&y_pack);

        float4 out_pack;
        __nv_bfloat16* out_arr = reinterpret_cast<__nv_bfloat16*>(&out_pack);

        #pragma unroll
        for (int i = 0; i < VEC; ++i) {
            float xv = __bfloat162float(x_arr[i]);
            float yv = __bfloat162float(y_arr[i]);
            out_arr[i] = __float2bfloat16(xv + yv);
        }

        if (INPLACE) {
            *reinterpret_cast<float4*>(x + tid * VEC) = out_pack;
        } else {
            *reinterpret_cast<float4*>(out + tid * VEC) = out_pack;
        }
    } else {
        const int i = tid * VEC + (tid - n_vec);
        if (i < N) {
            float xv = __bfloat162float(x[i]);
            float yv = __bfloat162float(y[i]);
            if (INPLACE) {
                x[i] = __float2bfloat16(xv + yv);
            } else {
                out[i] = __float2bfloat16(xv + yv);
            }
        }
    }
}

cudaError_t residual_add_bf16(
    const __nv_bfloat16* x,
    const __nv_bfloat16* y,
    __nv_bfloat16* out,
    int N,
    cudaStream_t stream) {
    if (!x || !y || !out || N <= 0) return cudaErrorInvalidValue;
    constexpr int BLOCK = 256;
    const int n_vec = N / 8;
    int grid = (n_vec + BLOCK - 1) / BLOCK;
    residual_add_kernel<false><<<grid, BLOCK, 0, stream>>>(const_cast<__nv_bfloat16*>(x), y, out, N);
    return cudaGetLastError();
}

cudaError_t residual_add_inplace_bf16(
    __nv_bfloat16* x,
    const __nv_bfloat16* y,
    int N,
    cudaStream_t stream) {
    if (!x || !y || N <= 0) return cudaErrorInvalidValue;
    constexpr int BLOCK = 256;
    const int n_vec = N / 8;
    int grid = (n_vec + BLOCK - 1) / BLOCK;
    residual_add_kernel<true><<<grid, BLOCK, 0, stream>>>(x, y, nullptr, N);
    return cudaGetLastError();
}

}  // namespace ops
}  // namespace viper
