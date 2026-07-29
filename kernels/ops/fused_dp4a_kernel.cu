/*
 * viper FUSED DP4A Q4 GEMV — rmsnorm + single-scale quantize + DP4A in ONE kernel.
 *
 * SMEM layout: [K bf16 activations] → overwritten by [K int8] after quantize.
 * Single global activation scale (not per-group) for full thread utilization.
 *
 * Flow:
 *   1. Load BF16 to SMEM (with optional swiglu fusion) — ALL threads
 *   2. rmsnorm reduction (sum_sq + max_abs in one pass) — ALL threads
 *   3. Normalize + quantize to INT8 in SMEM — ALL threads
 *   4. DP4A GEMV with per-group weight scales — grid-stride pattern
 */
#ifndef VIPER_FUSED_DP4A_KERNEL_CU
#define VIPER_FUSED_DP4A_KERNEL_CU

#include "linear_kernel.h"
#include <cuda_runtime.h>

namespace viper {
namespace ops {

__global__ void fused_dp4a_q4_kernel(
    const uint8_t* __restrict__ w_packed,
    const __nv_bfloat16* __restrict__ w_scales,
    const __nv_bfloat16* __restrict__ x,
    __nv_bfloat16* __restrict__ y,
    const __nv_bfloat16* __restrict__ residual,
    int M, int N, int K,
    const __nv_bfloat16* __restrict__ gamma,
    float eps,
    const __nv_bfloat16* __restrict__ swiglu_up)
{
    const int m_idx = blockIdx.y;
    const int warp_id = threadIdx.x >> 5;
    const int lane_id = threadIdx.x & 31;
    const int n = blockIdx.x * (blockDim.x >> 5) + warp_id;

    // SMEM: K bf16 (overwritten by K int8 after quantize)
    extern __shared__ char smem[];
    __nv_bfloat16* xbf = (__nv_bfloat16*)smem;   // [K] bf16 → reused
    int8_t* xq = (int8_t*)smem;                    // [K] int8 (same memory)
    // Small persistent area for scale + rmsnorm result
    __shared__ float s_act_scale;
    __shared__ float s_inv_rms;

    // Phase 1: Load BF16 activations (with optional swiglu)
    const __nv_bfloat16* xg = x + (size_t)m_idx * K;
    for (int i = threadIdx.x; i < K; i += blockDim.x) {
        if (swiglu_up) {
            float g = __bfloat162float(xg[i]);
            float u = __bfloat162float(swiglu_up[i]);
            xbf[i] = __float2bfloat16(g / (1.0f + __expf(-g)) * u);
        } else {
            xbf[i] = xg[i];
        }
    }

    // Phase 2: rmsnorm (sum_sq + max_abs in one reduction)
    float ss = 0.0f, mx = 0.0f;
    for (int i = threadIdx.x; i < K; i += blockDim.x) {
        float v = __bfloat162float(xbf[i]);
        ss += v * v;
        mx = fmaxf(mx, fabsf(v));
    }
    // Warp reduce ss and mx
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        ss += __shfl_xor_sync(0xffffffff, ss, off);
        mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, off));
    }
    __shared__ float wss[8], wmx[8];
    if ((threadIdx.x & 31) == 0) {
        wss[threadIdx.x >> 5] = ss;
        wmx[threadIdx.x >> 5] = mx;
    }
    __syncthreads();
    if (threadIdx.x < 32) {
        float tss = (threadIdx.x < 8) ? wss[threadIdx.x] : 0.0f;
        float tmx = (threadIdx.x < 8) ? wmx[threadIdx.x] : 0.0f;
        #pragma unroll
        for (int off = 4; off > 0; off >>= 1) {
            tss += __shfl_xor_sync(0xffffffff, tss, off);
            tmx = fmaxf(tmx, __shfl_xor_sync(0xffffffff, tmx, off));
        }
        if (threadIdx.x == 0) {
            float inv_rms = gamma ? rsqrtf(tss / (float)K + eps) : 1.0f;
            float gamma_max = fmaxf(tmx * inv_rms, 1e-8f);
            // Also account for gamma magnitude (assume max gamma ~ 2.0)
            s_inv_rms = inv_rms;
            s_act_scale = gamma_max * 2.0f / 127.0f;  // conservative scale
        }
    }
    __syncthreads();

    float inv_rms = s_inv_rms;
    float act_scale = s_act_scale;
    float inv_act_scale = 1.0f / act_scale;

    // Phase 3: Normalize + quantize to INT8 (overwrite xbf with xq)
    for (int i = threadIdx.x; i < K; i += blockDim.x) {
        float v = __bfloat162float(xbf[i]) * inv_rms;
        if (gamma) v *= __bfloat162float(gamma[i]);
        int q = __float2int_rn(v * inv_act_scale);
        xq[i] = (int8_t)(q > 127 ? 127 : (q < -128 ? -128 : q));
    }
    __syncthreads();

    if (n >= N || m_idx >= M) return;

    // Phase 4: DP4A GEMV (grid-stride, per-group weight scales, single act scale)
    const uint32_t* w32 = reinterpret_cast<const uint32_t*>(
        w_packed + (size_t)n * (K / 2));
    const __nv_bfloat16* s_row = w_scales + (size_t)n * (K / 64);
    const int32_t* xq32 = reinterpret_cast<const int32_t*>(xq);
    const int n_quads = K / 8;

    float acc = 0.0f;
    for (int i = lane_id; i < n_quads; i += 32) {
        uint32_t packed4 = w32[i];
        uint32_t lo = packed4 & 0x0F0F0F0F;
        uint32_t hi = (packed4 >> 4) & 0x0F0F0F0F;
        int32_t slo = __vsubss4(lo, 0x08080808);
        int32_t shi = __vsubss4(hi, 0x08080808);
        int32_t w_lo = __byte_perm(slo, shi, 0x5140);
        int32_t w_hi = __byte_perm(slo, shi, 0x7362);
        int32_t x_lo = xq32[i * 2];
        int32_t x_hi = xq32[i * 2 + 1];
        int32_t raw = __dp4a(w_lo, x_lo, 0);
        raw = __dp4a(w_hi, x_hi, raw);
        int g = i / 8;
        acc += (float)raw * __bfloat162float(s_row[g]) * act_scale;
    }

    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        acc += __shfl_xor_sync(0xffffffff, acc, off);

    if (lane_id == 0) {
        if (residual)
            acc += __bfloat162float(residual[m_idx * N + n]);
        y[m_idx * N + n] = __float2bfloat16(acc);
    }
}

// Launchers matching existing API
cudaError_t fused_dp4a_rmsnorm(
    const uint8_t* w, const __nv_bfloat16* s, const __nv_bfloat16* gamma,
    float eps, const __nv_bfloat16* x, __nv_bfloat16* y,
    int M, int N, int K, cudaStream_t stream) {
    size_t smem = K * 2;  // bf16 → int8 reuse
    fused_dp4a_q4_kernel<<<(N+7)/8, 256, smem, stream>>>(
        w, s, x, y, nullptr, M, N, K, gamma, eps, nullptr);
    return cudaGetLastError();
}

cudaError_t fused_dp4a_residual(
    const uint8_t* w, const __nv_bfloat16* s, const __nv_bfloat16* x,
    __nv_bfloat16* y, const __nv_bfloat16* residual,
    int M, int N, int K, cudaStream_t stream) {
    size_t smem = K * 2;
    fused_dp4a_q4_kernel<<<(N+7)/8, 256, smem, stream>>>(
        w, s, x, y, residual, M, N, K, nullptr, 0.0f, nullptr);
    return cudaGetLastError();
}

cudaError_t fused_dp4a_residual_swiglu(
    const uint8_t* w, const __nv_bfloat16* s, const __nv_bfloat16* gate,
    const __nv_bfloat16* up, __nv_bfloat16* y, const __nv_bfloat16* residual,
    int M, int N, int K, cudaStream_t stream) {
    size_t smem = K * 2;
    fused_dp4a_q4_kernel<<<(N+7)/8, 256, smem, stream>>>(
        w, s, gate, y, residual, M, N, K, nullptr, 0.0f, up);
    return cudaGetLastError();
}

}  // namespace ops
}  // namespace viper

#endif
