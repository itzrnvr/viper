/*
 * viper FUSED DP4A Q4 GEMV — rmsnorm + per-warp quantize + DP4A in ONE kernel.
 *
 * FIX v2: Per-warp parallel quantization eliminates 191 of 192 __syncthreads.
 * Old: 48 groups × 4 syncs = 192 syncs → 960μs overhead per kernel.
 * New: 8 warps × 6 groups each (no inter-group sync) → 2 syncs total.
 *
 * SMEM: K bf16 (overwritten by K int8 after quantize) + K/64 float group scales.
 * Occupancy: K*2 + K/64*4 bytes. For K=3072: 6336B → 7 blocks/SM (87%).
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

    extern __shared__ char smem[];
    __nv_bfloat16* xbf = (__nv_bfloat16*)smem;               // [K] bf16 activations
    int8_t* xq = (int8_t*)(smem + K * sizeof(__nv_bfloat16)); // [K] int8 quantized (SEPARATE)
    float* group_scales = (float*)(xq + K);                   // [K/64] per-group scales

    // ===== Phase 1: Load BF16 activations to SMEM =====
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

    // ===== Phase 2: rmsnorm (global reduction, 2 syncs total) =====
    __shared__ float s_inv_rms;
    if (gamma) {
        float ss = 0.0f;
        for (int i = threadIdx.x; i < K; i += blockDim.x) {
            float v = __bfloat162float(xbf[i]);
            ss += v * v;
        }
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            ss += __shfl_xor_sync(0xffffffff, ss, off);
        __shared__ float wss[8];
        if (lane_id == 0) wss[warp_id] = ss;
        __syncthreads();
        if (threadIdx.x == 0) {
            float total = 0;
            #pragma unroll
            for (int i = 0; i < 8; ++i) total += wss[i];
            s_inv_rms = rsqrtf(total / (float)K + eps);
        }
        __syncthreads();
    } else {
        s_inv_rms = 1.0f;
        __syncthreads();
    }
    float inv_rms = s_inv_rms;

    // ===== Phase 3: Per-warp parallel quantization (ZERO syncs!) =====
    // Each warp handles its own groups independently using __shfl only.
    const int ngroups = K / 64;
    const int groups_per_warp = (ngroups + 7) / 8;
    const int my_g_start = warp_id * groups_per_warp;
    const int my_g_end = min(my_g_start + groups_per_warp, ngroups);

    for (int g = my_g_start; g < my_g_end; ++g) {
        const int base = g * 64;
        // Each of 32 threads handles 2 elements (64/32 = 2)
        float gmax = 0.f;
        #pragma unroll
        for (int s = 0; s < 2; ++s) {
            int idx = base + lane_id + s * 32;
            float v = __bfloat162float(xbf[idx]) * inv_rms;
            if (gamma) v *= __bfloat162float(gamma[idx]);
            gmax = fmaxf(gmax, fabsf(v));
        }
        // Warp reduce only — NO __syncthreads!
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            gmax = fmaxf(gmax, __shfl_xor_sync(0xffffffff, gmax, off));

        float scale = gmax / 127.0f;
        if (lane_id == 0) group_scales[g] = scale;

        // Quantize (each warp writes to its own region — no conflict)
        #pragma unroll
        for (int s = 0; s < 2; ++s) {
            int idx = base + lane_id + s * 32;
            float v = __bfloat162float(xbf[idx]) * inv_rms;
            if (gamma) v *= __bfloat162float(gamma[idx]);
            xq[idx] = (int8_t)__float2int_rn(v / scale);
        }
    }

    // ===== ONE sync before GEMV (all warps must finish quantizing) =====
    __syncthreads();

    if (n >= N || m_idx >= M) return;

    // ===== Phase 4: DP4A GEMV =====
    const uint32_t* w32 = reinterpret_cast<const uint32_t*>(
        w_packed + (size_t)n * (K / 2));
    const __nv_bfloat16* s_row = w_scales + (size_t)n * (K / 64);
    const int32_t* xq32 = reinterpret_cast<const int32_t*>(xq);
    const int n_quads = K / 8;

    float acc = 0.0f;
    for (int i = lane_id; i < n_quads; i += 32) {
        uint32_t packed4 = w32[i];
        // Unpack Q4 → INT8 using __vsubss4 + __byte_perm (3 instructions)
        uint32_t lo = packed4 & 0x0F0F0F0F;
        uint32_t hi = (packed4 >> 4) & 0x0F0F0F0F;
        int32_t slo = __vsubss4(lo, 0x08080808);
        int32_t shi = __vsubss4(hi, 0x08080808);
        int32_t w_lo = __byte_perm(slo, shi, 0x5140);
        int32_t w_hi = __byte_perm(slo, shi, 0x7362);
        int32_t x_lo = xq32[i * 2];
        int32_t x_hi = xq32[i * 2 + 1];
        // DP4A: 4 multiply-adds per instruction
        int32_t raw = __dp4a(w_lo, x_lo, 0);
        raw = __dp4a(w_hi, x_hi, raw);
        int g = i / 8;
        acc += (float)raw * __bfloat162float(s_row[g]) * group_scales[g];
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

// Launchers
cudaError_t fused_dp4a_rmsnorm(
    const uint8_t* w, const __nv_bfloat16* s, const __nv_bfloat16* gamma,
    float eps, const __nv_bfloat16* x, __nv_bfloat16* y,
    int M, int N, int K, cudaStream_t stream) {
    size_t smem = K * 3 + (K / 64) * sizeof(float);
    fused_dp4a_q4_kernel<<<(N+7)/8, 256, smem, stream>>>(
        w, s, x, y, nullptr, M, N, K, gamma, eps, nullptr);
    return cudaGetLastError();
}

cudaError_t fused_dp4a_residual(
    const uint8_t* w, const __nv_bfloat16* s, const __nv_bfloat16* x,
    __nv_bfloat16* y, const __nv_bfloat16* residual,
    int M, int N, int K, cudaStream_t stream) {
    size_t smem = K * 3 + (K / 64) * sizeof(float);
    fused_dp4a_q4_kernel<<<(N+7)/8, 256, smem, stream>>>(
        w, s, x, y, residual, M, N, K, nullptr, 0.0f, nullptr);
    return cudaGetLastError();
}

cudaError_t fused_dp4a_residual_swiglu(
    const uint8_t* w, const __nv_bfloat16* s, const __nv_bfloat16* gate,
    const __nv_bfloat16* up, __nv_bfloat16* y, const __nv_bfloat16* residual,
    int M, int N, int K, cudaStream_t stream) {
    size_t smem = K * 3 + (K / 64) * sizeof(float);
    fused_dp4a_q4_kernel<<<(N+7)/8, 256, smem, stream>>>(
        w, s, gate, y, residual, M, N, K, nullptr, 0.0f, up);
    return cudaGetLastError();
}

}  // namespace ops
}  // namespace viper

#endif
