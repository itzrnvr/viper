/*
 * viper Multi-M Q4 GEMV kernel.
 *
 * CRITICAL OPTIMIZATION: Each block processes ALL M tokens for its output
 * channel, reading weights ONCE from DRAM (L1-cached for subsequent tokens).
 *
 * Without this: forward_batch(M=5) reads 5× weights → 5× slower than M=1.
 * With this: forward_batch(M=5) reads 1× weights → same speed as M=1.
 *
 * Also supports fused rmsnorm and residual.
 */
#ifndef VIPER_LINEAR_MULTIM_KERNEL_CU
#define VIPER_LINEAR_MULTIM_KERNEL_CU

#include "linear_kernel.h"
#include <cuda_runtime.h>

namespace viper {
namespace ops {

constexpr int MULTIM_MAX_M = 16;
constexpr int MULTIM_SMEM_MAX = 11264;

// Multi-M kernel: processes M tokens per block, weights read ONCE.
// Grid: (N+7)/8 blocks, 256 threads each.
__global__ void linear_q4_g64_multim_kernel(
    const uint8_t* __restrict__ w_packed,
    const __nv_bfloat16* __restrict__ w_scales,
    const __nv_bfloat16* __restrict__ x,          // [M, K]
    __nv_bfloat16* __restrict__ y,                 // [M, N]
    const __nv_bfloat16* __restrict__ residual,    // [M, N] or null
    int M, int N, int K,
    const __nv_bfloat16* __restrict__ gamma,
    float eps)
{
    const int warp_id = threadIdx.x >> 5;
    const int lane_id = threadIdx.x & 31;
    const int n = blockIdx.x * (blockDim.x >> 5) + warp_id;

    // SMEM: [M, K] bf16 activations
    extern __shared__ __nv_bfloat16 smem_x[];

    // Phase 1: Load ALL M activation vectors + optional rmsnorm
    for (int m = 0; m < M; ++m) {
        const __nv_bfloat16* x_global = x + (size_t)m * K;
        __nv_bfloat16* x_smem = smem_x + (size_t)m * K;
        for (int i = threadIdx.x; i < K; i += blockDim.x)
            x_smem[i] = x_global[i];
        __syncthreads();

        if (gamma) {
            // rmsnorm for this token
            float ss = 0.0f;
            for (int i = threadIdx.x; i < K; i += blockDim.x) {
                float v = __bfloat162float(x_smem[i]);
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
            float inv = ws[0];
            for (int i = threadIdx.x; i < K; i += blockDim.x)
                x_smem[i] = __float2bfloat16(
                    __bfloat162float(x_smem[i]) * inv * __bfloat162float(gamma[i]));
            __syncthreads();
        }
    }

    if (n >= N) return;

    // Phase 2: GEMV — read weights ONCE, compute ALL M tokens
    const uint8_t* w_row = w_packed + (size_t)n * (K / 2);
    const __nv_bfloat16* s_row = w_scales + (size_t)n * (K / 64);

    float acc[MULTIM_MAX_M];
    #pragma unroll
    for (int m = 0; m < MULTIM_MAX_M; ++m) acc[m] = 0.0f;

    const int n_bytes = K / 2;
    const int vec_end = n_bytes - (n_bytes % 128);

    for (int base = 0; base < vec_end; base += 128) {
        int byte_off = base + lane_id * 4;
        // Read weight ONCE from DRAM (L1 cached for subsequent iterations)
        uint32_t packed4 = *reinterpret_cast<const uint32_t*>(w_row + byte_off);

        // Dequant ONCE
        int w0 = (packed4        & 0xF) - 8;
        int w1 = ((packed4 >> 4) & 0xF) - 8;
        int w2 = ((packed4 >> 8) & 0xF) - 8;
        int w3 = ((packed4 >>12) & 0xF) - 8;
        int w4 = ((packed4 >>16) & 0xF) - 8;
        int w5 = ((packed4 >>20) & 0xF) - 8;
        int w6 = ((packed4 >>24) & 0xF) - 8;
        int w7 = ((packed4 >>28) & 0xF) - 8;

        float sc = __bfloat162float(s_row[byte_off / 32]);

        // Compute dot product for ALL M tokens (activations from SMEM)
        int xk = byte_off * 2;
        #pragma unroll
        for (int m = 0; m < MULTIM_MAX_M; ++m) {
            if (m >= M) break;
            const __nv_bfloat16* xm = smem_x + (size_t)m * K;
            float xv0 = __bfloat162float(xm[xk    ]);
            float xv1 = __bfloat162float(xm[xk + 1]);
            float xv2 = __bfloat162float(xm[xk + 2]);
            float xv3 = __bfloat162float(xm[xk + 3]);
            float xv4 = __bfloat162float(xm[xk + 4]);
            float xv5 = __bfloat162float(xm[xk + 5]);
            float xv6 = __bfloat162float(xm[xk + 6]);
            float xv7 = __bfloat162float(xm[xk + 7]);
            acc[m] += sc * ((float)w0*xv0 + (float)w1*xv1 + (float)w2*xv2 + (float)w3*xv3
                          + (float)w4*xv4 + (float)w5*xv5 + (float)w6*xv6 + (float)w7*xv7);
        }
    }

    // Tail
    for (int bi = vec_end + lane_id; bi < n_bytes; bi += 32) {
        float sc = __bfloat162float(s_row[bi / 32]);
        uint8_t b = w_row[bi];
        int w0 = (b & 0xF) - 8;
        int w1 = (b >> 4) - 8;
        int k0 = bi * 2;
        #pragma unroll
        for (int m = 0; m < MULTIM_MAX_M; ++m) {
            if (m >= M) break;
            const __nv_bfloat16* xm = smem_x + (size_t)m * K;
            acc[m] += (float)w0 * sc * __bfloat162float(xm[k0    ]);
            acc[m] += (float)w1 * sc * __bfloat162float(xm[k0 + 1]);
        }
    }

    // Warp reduce for each M
    #pragma unroll
    for (int m = 0; m < MULTIM_MAX_M; ++m) {
        if (m >= M) break;
        float a = acc[m];
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            a += __shfl_xor_sync(0xffffffff, a, off);
        if (lane_id == 0) {
            if (residual)
                a += __bfloat162float(residual[m * N + n]);
            y[m * N + n] = __float2bfloat16(a);
        }
    }
}

// Launchers
cudaError_t linear_q4_multim_rmsnorm(
    const uint8_t* w, const __nv_bfloat16* s, const __nv_bfloat16* gamma,
    float eps, const __nv_bfloat16* x, __nv_bfloat16* y,
    int M, int N, int K, cudaStream_t stream) {
    if (K % 64 != 0 || K > MULTIM_SMEM_MAX || M > MULTIM_MAX_M)
        return cudaErrorInvalidValue;
    size_t smem = (size_t)M * K * sizeof(__nv_bfloat16);
    linear_q4_g64_multim_kernel<<<(N+7)/8, 256, smem, stream>>>(
        w, s, x, y, nullptr, M, N, K, gamma, eps);
    return cudaGetLastError();
}

cudaError_t linear_q4_multim_residual(
    const uint8_t* w, const __nv_bfloat16* s, const __nv_bfloat16* x,
    __nv_bfloat16* y, const __nv_bfloat16* residual,
    int M, int N, int K, cudaStream_t stream) {
    if (K % 64 != 0 || K > MULTIM_SMEM_MAX || M > MULTIM_MAX_M)
        return cudaErrorInvalidValue;
    size_t smem = (size_t)M * K * sizeof(__nv_bfloat16);
    linear_q4_g64_multim_kernel<<<(N+7)/8, 256, smem, stream>>>(
        w, s, x, y, residual, M, N, K, nullptr, 0.0f);
    return cudaGetLastError();
}

}  // namespace ops
}  // namespace viper

#endif
