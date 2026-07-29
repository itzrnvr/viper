/*
 * viper RMSNorm + Q8 Quantize fused kernel.
 *
 * Does rmsnorm AND quantizes to INT8 with per-group-of-64 scales in ONE pass.
 * Output: x_q8[K] INT8 + x_scales[K/64] FLOAT — ready for DP4A GEMV.
 * Also outputs x_norm[K] BF16 for residual connections.
 *
 * This eliminates 1 separate quantize launch per rmsnorm call.
 * With 88 rmsnorm calls per forward: saves 88 launches (~1ms on WDDM).
 *
 * __syncthreads cost: 0.21us each (measured). Total 3 syncs = 0.63us. Negligible.
 */
#include "rmsnorm_kernel.h"
#include <cuda_runtime.h>

namespace viper {
namespace ops {

// RMSNorm + Q8 quantize: one kernel, outputs both BF16 norm and INT8 Q8.
// Grid: 1 block per row. Block: 256 threads.
__global__ void rmsnorm_quantize_kernel(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ gamma,
    __nv_bfloat16* __restrict__ out_norm,  // [K] BF16 normalized (for residual paths)
    int8_t* __restrict__ out_q8,            // [K] INT8 quantized (for DP4A GEMV)
    float* __restrict__ out_scales,         // [K/64] per-group activation scales
    int H,
    float eps) {
    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;

    extern __shared__ char smem[];
    __nv_bfloat16* sx = (__nv_bfloat16*)smem;  // [H] BF16 activations

    // Phase 1: Load x to SMEM
    for (int i = tid; i < H; i += nthreads)
        sx[i] = x[i];
    __syncthreads();

    // Phase 2: Compute sum of squares
    float ss = 0.0f;
    for (int i = tid; i < H; i += nthreads) {
        float v = __bfloat162float(sx[i]);
        ss += v * v;
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        ss += __shfl_xor_sync(0xffffffff, ss, off);
    __shared__ float warp_sums[8];
    const int wid = tid >> 5, lid = tid & 31;
    if (lid == 0) warp_sums[wid] = ss;
    __syncthreads();
    if (tid < 32) {
        float t = (tid < 8) ? warp_sums[tid] : 0.0f;
        #pragma unroll
        for (int off = 4; off > 0; off >>= 1)
            t += __shfl_xor_sync(0xffffffff, t, off);
        if (tid == 0) warp_sums[0] = rsqrtf(t / (float)H + eps);
    }
    __syncthreads();
    float inv_rms = warp_sums[0];

    // Phase 3: Normalize + write BF16 output
    for (int i = tid; i < H; i += nthreads) {
        float v = __bfloat162float(sx[i]) * inv_rms * __bfloat162float(gamma[i]);
        out_norm[i] = __float2bfloat16(v);
        sx[i] = __float2bfloat16(v);  // also update SMEM for quantization
    }
    __syncthreads();

    // Phase 4: Per-group quantization to INT8
    const int ngroups = H / 64;
    const int groups_per_warp = (ngroups + 7) / 8;
    const int my_g_start = wid * groups_per_warp;
    const int my_g_end = min(my_g_start + groups_per_warp, ngroups);

    for (int g = my_g_start; g < my_g_end; ++g) {
        const int base = g * 64;
        // Each of 32 threads handles 2 elements (64/32 = 2)
        float gmax = 0.f;
        #pragma unroll
        for (int s = 0; s < 2; ++s) {
            int idx = base + lid + s * 32;
            if (idx < H) {
                float v = __bfloat162float(sx[idx]);
                gmax = fmaxf(gmax, fabsf(v));
            }
        }
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            gmax = fmaxf(gmax, __shfl_xor_sync(0xffffffff, gmax, off));

        float scale = gmax / 127.0f;
        if (lid == 0) out_scales[g] = scale;

        #pragma unroll
        for (int s = 0; s < 2; ++s) {
            int idx = base + lid + s * 32;
            if (idx < H) {
                float v = __bfloat162float(sx[idx]);
                out_q8[idx] = (int8_t)__float2int_rn(v / scale);
            }
        }
    }
}

cudaError_t rmsnorm_quantize_bf16(
    const __nv_bfloat16* x,
    const __nv_bfloat16* gamma,
    __nv_bfloat16* out_norm,
    int8_t* out_q8,
    float* out_scales,
    int H,
    float eps,
    cudaStream_t stream) {
    if (!x || !gamma || !out_norm || !out_q8 || !out_scales || H <= 0)
        return cudaErrorInvalidValue;
    if (H % 64 != 0) return cudaErrorInvalidValue;
    size_t smem = H * sizeof(__nv_bfloat16);
    rmsnorm_quantize_kernel<<<1, 256, smem, stream>>>(
        x, gamma, out_norm, out_q8, out_scales, H, eps);
    return cudaGetLastError();
}

}  // namespace ops
}  // namespace viper
