/*
 * viper DP4A Q4 GEMV — v3 production with fused rmsnorm+quantize.
 *
 * SMEM layout: [K bf16 activations] → overwritten by [K int8 quantized]
 * Phases:
 *   1. Load BF16 activations to SMEM (with optional swiglu)
 *   2. Fused rmsnorm + INT8 quantization (single pass after reduction)
 *   3. DP4A GEMV using 4-groups-per-iteration pattern
 *
 * The DP4A inner loop replaces scalar FMA with:
 *   __byte_perm + __vsubss4 for Q4→INT8 (7 inst / 8 elems)
 *   __dp4a for 4-wide dot product (1 inst / 4 MACs)
 */
#include "linear_kernel.h"
#include <cuda_runtime.h>

namespace viper {
namespace ops {

__global__ void dp4a_q4_v3_kernel(
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
    const int m = blockIdx.y;
    const int warp_id = threadIdx.x >> 5;
    const int lane_id = threadIdx.x & 31;
    const int n = blockIdx.x * (blockDim.x >> 5) + warp_id;

    // SMEM: reuse same buffer for bf16→int8 (int8 is half size)
    extern __shared__ char smem[];
    // Use first K*2 bytes for initial bf16 load, then reuse as K int8 + K/64 scales
    int8_t* xq = (int8_t*)smem;                    // [K] int8 (overwritten)
    __nv_bfloat16* xs_scales = (__nv_bfloat16*)(smem + K); // [K/64] activation scales
    // Use temp area for initial bf16 load + rmsnorm (in second half of smem)
    __nv_bfloat16* xbf = (__nv_bfloat16*)(smem + K + K / 64 * 2 + 128); // [K] bf16 temp

    // Phase 1: Load activations to xbf (with optional swiglu)
    const __nv_bfloat16* x_global = x + (size_t)m * K;
    for (int i = threadIdx.x; i < K; i += blockDim.x) {
        if (swiglu_up) {
            float g = __bfloat162float(x_global[i]);
            float u = __bfloat162float(swiglu_up[i]);
            xbf[i] = __float2bfloat16(g / (1.0f + __expf(-g)) * u);
        } else {
            xbf[i] = x_global[i];
        }
    }

    // Phase 2: Fused rmsnorm + INT8 quantization
    float inv_rms = 1.0f;
    if (gamma) {
        // Sum of squares reduction
        float ss = 0.0f;
        for (int i = threadIdx.x; i < K; i += blockDim.x) {
            float v = __bfloat162float(xbf[i]);
            ss += v * v;
        }
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            ss += __shfl_xor_sync(0xffffffff, ss, off);
        __shared__ float ws[8];
        if ((threadIdx.x & 31) == 0) ws[threadIdx.x >> 5] = ss;
        __syncthreads();
        if (threadIdx.x < 32) {
            float t = (threadIdx.x < 8) ? ws[threadIdx.x] : 0.0f;
            #pragma unroll
            for (int off = 4; off > 0; off >>= 1)
                t += __shfl_xor_sync(0xffffffff, t, off);
            if (threadIdx.x == 0) ws[0] = rsqrtf(t / (float)K + eps);
        }
        __syncthreads();
        inv_rms = ws[0];
    }

    // Per-group quantization: find max-abs per group, quantize
    const int n_groups = K / 64;
    for (int g = threadIdx.x; g < n_groups; g += blockDim.x) {
        int start = g * 64;
        float maxabs = 1e-8f;
        for (int i = 0; i < 64; ++i) {
            float v = fabsf(__bfloat162float(xbf[start + i]) * inv_rms
                           * (gamma ? __bfloat162float(gamma[start + i]) : 1.0f));
            if (v > maxabs) maxabs = v;
        }
        float sc = maxabs / 127.0f;
        float inv_sc = 1.0f / sc;
        xs_scales[g] = __float2bfloat16(sc);
        for (int i = 0; i < 64; ++i) {
            float v = __bfloat162float(xbf[start + i]) * inv_rms
                     * (gamma ? __bfloat162float(gamma[start + i]) : 1.0f);
            int q = __float2int_rn(v * inv_sc);
            xq[start + i] = (int8_t)(q > 127 ? 127 : (q < -128 ? -128 : q));
        }
    }
    __syncthreads();

    // xbf no longer needed — free its SMEM implicitly
    if (n >= N || m >= M) return;

    // Phase 3: DP4A GEMV
    const uint8_t* w_row = w_packed + (size_t)n * (K / 2);
    const __nv_bfloat16* s_row = w_scales + (size_t)n * (K / 64);
    const int32_t* xq32 = reinterpret_cast<const int32_t*>(xq);

    float acc = 0.0f;
    const int group_in_lane = lane_id >> 3;  // 0-3
    const int quad_in_group = lane_id & 7;   // 0-7

    for (int gbase = 0; gbase < n_groups; gbase += 4) {
        int g = gbase + group_in_lane;
        if (g >= n_groups) break;

        // Weight: 32 bytes per group, 4 bytes per quad
        int byte_off = g * 32 + quad_in_group * 4;
        uint32_t packed4 = *reinterpret_cast<const uint32_t*>(w_row + byte_off);

        // Q4 → INT8 via byte_perm + vsubss4
        uint32_t lo = packed4 & 0x0F0F0F0F;
        uint32_t hi = (packed4 >> 4) & 0x0F0F0F0F;
        int32_t slo = __vsubss4(lo, 0x08080808);
        int32_t shi = __vsubss4(hi, 0x08080808);
        int32_t w_lo = __byte_perm(slo, shi, 0x5140);
        int32_t w_hi = __byte_perm(slo, shi, 0x7362);

        // Activation: elem offset = byte_off * 2
        int elem = byte_off * 2;
        int32_t x_lo = xq32[elem / 4];
        int32_t x_hi = xq32[elem / 4 + 1];

        int32_t partial = __dp4a(w_lo, x_lo, 0);
        partial = __dp4a(w_hi, x_hi, partial);

        // Reduce within 8-lane group
        #pragma unroll
        for (int off = 4; off > 0; off >>= 1)
            partial += __shfl_xor_sync(0xffffffff, partial, off);

        if (quad_in_group == 0) {
            float ws = __bfloat162float(s_row[g]);
            float xs = __bfloat162float(xs_scales[g]);
            acc += (float)partial * ws * xs;
        }
    }

    // Cross-group warp reduce
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        acc += __shfl_xor_sync(0xffffffff, acc, off);

    if (lane_id == 0) {
        if (residual)
            acc += __bfloat162float(residual[m * N + n]);
        y[m * N + n] = __float2bfloat16(acc);
    }
}

// Launchers
cudaError_t dp4a_q4_g64_bf16(
    const uint8_t* w, const __nv_bfloat16* s, const __nv_bfloat16* x,
    __nv_bfloat16* y, int M, int N, int K, cudaStream_t stream) {
    // SMEM: K (int8) + K/64*2 (scales) + 128 (pad) + K*2 (bf16 temp)
    size_t smem = K + K/64*2 + 128 + K*2;
    dp4a_q4_v3_kernel<<<(N+7)/8, 256, smem, stream>>>(
        w, s, x, y, nullptr, M, N, K, nullptr, 0.0f, nullptr);
    return cudaGetLastError();
}

cudaError_t dp4a_q4_g64_bf16_rmsnorm(
    const uint8_t* w, const __nv_bfloat16* s, const __nv_bfloat16* gamma,
    float eps, const __nv_bfloat16* x, __nv_bfloat16* y,
    int M, int N, int K, cudaStream_t stream) {
    size_t smem = K + K/64*2 + 128 + K*2;
    dp4a_q4_v3_kernel<<<(N+7)/8, 256, smem, stream>>>(
        w, s, x, y, nullptr, M, N, K, gamma, eps, nullptr);
    return cudaGetLastError();
}

cudaError_t dp4a_q4_g64_bf16_residual(
    const uint8_t* w, const __nv_bfloat16* s, const __nv_bfloat16* x,
    __nv_bfloat16* y, const __nv_bfloat16* residual,
    int M, int N, int K, cudaStream_t stream) {
    size_t smem = K + K/64*2 + 128 + K*2;
    dp4a_q4_v3_kernel<<<(N+7)/8, 256, smem, stream>>>(
        w, s, x, y, residual, M, N, K, nullptr, 0.0f, nullptr);
    return cudaGetLastError();
}

cudaError_t dp4a_q4_g64_bf16_residual_swiglu(
    const uint8_t* w, const __nv_bfloat16* s, const __nv_bfloat16* gate,
    const __nv_bfloat16* up, __nv_bfloat16* y, const __nv_bfloat16* residual,
    int M, int N, int K, cudaStream_t stream) {
    size_t smem = K + K/64*2 + 128 + K*2;
    dp4a_q4_v3_kernel<<<(N+7)/8, 256, smem, stream>>>(
        w, s, gate, y, residual, M, N, K, nullptr, 0.0f, up);
    return cudaGetLastError();
}

}  // namespace ops
}  // namespace viper
