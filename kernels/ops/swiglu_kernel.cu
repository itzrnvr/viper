/*
 * viper SwiGLU activation kernel — implementation
 *
 * PURPOSE: see swiglu_kernel.h
 *
 * IMPLEMENTATION:
 *   Each thread processes VEC=8 elements via float4. Vectorized loads
 *   and stores. silu computed in fp32.
 *
 * SAFETY:
 *   - No allocation.
 *   - SMEM: 0 bytes.
 */
#include "swiglu_kernel.h"
#include <cuda_runtime.h>

namespace viper {
namespace ops {

__device__ __forceinline__ float silu_f32(float x) {
    return x / (1.0f + __expf(-x));
}

__global__ void swiglu_inplace_kernel(
    __nv_bfloat16* __restrict__ gate,
    const __nv_bfloat16* __restrict__ up,
    int N) {
    constexpr int VEC = 8;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int n_vec = N / VEC;

    if (tid < n_vec) {
        float4 g_pack = *reinterpret_cast<const float4*>(gate + tid * VEC);
        float4 u_pack = *reinterpret_cast<const float4*>(up + tid * VEC);

        const __nv_bfloat16* g_arr = reinterpret_cast<const __nv_bfloat16*>(&g_pack);
        const __nv_bfloat16* u_arr = reinterpret_cast<const __nv_bfloat16*>(&u_pack);

        float4 out_pack;
        __nv_bfloat16* out_arr = reinterpret_cast<__nv_bfloat16*>(&out_pack);

        #pragma unroll
        for (int i = 0; i < VEC; ++i) {
            float g = __bfloat162float(g_arr[i]);
            float u = __bfloat162float(u_arr[i]);
            out_arr[i] = __float2bfloat16(silu_f32(g) * u);
        }

        *reinterpret_cast<float4*>(gate + tid * VEC) = out_pack;
    } else {
        const int i = tid * VEC + (tid - n_vec);
        if (i < N) {
            float g = __bfloat162float(gate[i]);
            float u = __bfloat162float(up[i]);
            gate[i] = __float2bfloat16(silu_f32(g) * u);
        }
    }
}

cudaError_t swiglu_inplace_bf16(
    __nv_bfloat16* gate,
    const __nv_bfloat16* up,
    int N,
    cudaStream_t stream) {
    if (!gate || !up || N <= 0) return cudaErrorInvalidValue;
    constexpr int BLOCK = 256;
    const int n_vec = N / 8;
    int grid = (n_vec + BLOCK - 1) / BLOCK;
    swiglu_inplace_kernel<<<grid, BLOCK, 0, stream>>>(gate, up, N);
    return cudaGetLastError();
}

cudaError_t swiglu_out_of_place_bf16(
    const __nv_bfloat16* gate,
    const __nv_bfloat16* up,
    __nv_bfloat16* out,
    int N,
    cudaStream_t stream) {
    if (!gate || !up || !out || N <= 0) return cudaErrorInvalidValue;
    constexpr int BLOCK = 256;
    const int n_vec = N / 8;
    int grid = (n_vec + BLOCK - 1) / BLOCK;
    cudaMemcpyAsync(out, gate, N * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice, stream);
    swiglu_inplace_kernel<<<grid, BLOCK, 0, stream>>>(out, up, N);
    return cudaGetLastError();
}

}  // namespace ops
}  // namespace viper
