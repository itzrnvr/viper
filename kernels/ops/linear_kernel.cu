/*
 * viper Linear Q4_G64 GEMV kernel — SMEM-cached activation + vectorized loads.
 *
 * Key insight: all warps in a block read the SAME activation vector x.
 * By caching x in shared memory (6 KB), we eliminate 7× redundant HBM
 * reads per block. The HBM bandwidth is freed for weight reads.
 *
 * Optimizations:
 * 1. 8 warps/block (256 threads) → full SM occupancy
 * 2. x cached in SMEM (cooperative load at block start)
 * 3. uint32 vectorized weight loads
 * 4. Fused residual add (optional)
 */
#include "linear_kernel.h"
#include <cuda_runtime.h>

namespace viper {
namespace ops {

// Shared memory size for x: H bf16 values.
// For H=3072: 6144 bytes. 8 blocks/SM × 6144 = 48 KB < 100 KB SMEM limit.
constexpr int SMEM_X_MAX = 11264;  // max K (down_proj K=10752)

__global__ void linear_q4_g64_warp_kernel(
    const uint8_t* __restrict__ w_packed,
    const __nv_bfloat16* __restrict__ w_scales,
    const __nv_bfloat16* __restrict__ x,
    __nv_bfloat16* __restrict__ y,
    const __nv_bfloat16* __restrict__ residual,
    int M, int N, int K,
    const __nv_bfloat16* __restrict__ gamma,  // rmsnorm weights (null = skip)
    float eps,
    const __nv_bfloat16* __restrict__ swiglu_up) {  // up proj for fused swiglu

    const int m = blockIdx.y;
    const int warp_id = threadIdx.x >> 5;
    const int lane_id = threadIdx.x & 31;
    const int n = blockIdx.x * 8 + warp_id;
    if (n >= N || m >= M) {
        // Still participate in SMEM load + sync to avoid deadlock.
    }

    // --- Cache x in shared memory (with optional swiglu fusion) ---
    extern __shared__ __nv_bfloat16 smem_x[];
    const __nv_bfloat16* x_row_global = x + (size_t)m * K;
    for (int i = threadIdx.x; i < K; i += blockDim.x) {
        if (swiglu_up) {
            float g = __bfloat162float(x_row_global[i]);
            float u = __bfloat162float(swiglu_up[i]);
            smem_x[i] = __float2bfloat16(g / (1.0f + __expf(-g)) * u);
        } else {
            smem_x[i] = x_row_global[i];
        }
    }

    // Optional fused rmsnorm: normalize x in SMEM before GEMV.
    if (gamma) {
        float ss = 0.0f;
        for (int i = threadIdx.x; i < K; i += blockDim.x) {
            float v = __bfloat162float(smem_x[i]);
            ss += v * v;
        }
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            ss += __shfl_xor_sync(0xffffffff, ss, off);
        __shared__ float warp_sums[8];
        if ((threadIdx.x & 31) == 0) warp_sums[threadIdx.x >> 5] = ss;
        __syncthreads();
        if (threadIdx.x < 32) {
            float t = (threadIdx.x < 8) ? warp_sums[threadIdx.x] : 0.0f;
            #pragma unroll
            for (int off = 4; off > 0; off >>= 1)
                t += __shfl_xor_sync(0xffffffff, t, off);
            if (threadIdx.x == 0) warp_sums[0] = rsqrtf(t / (float)K + eps);
        }
        __syncthreads();
        float inv_rms = warp_sums[0];
        for (int i = threadIdx.x; i < K; i += blockDim.x)
            smem_x[i] = __float2bfloat16(
                __bfloat162float(smem_x[i]) * inv_rms * __bfloat162float(gamma[i]));
        __syncthreads();
    }

    const __nv_bfloat16* x_row = smem_x;

    const uint8_t* w_row = w_packed + (size_t)n * (K / 2);
    const __nv_bfloat16* s_row = w_scales + (size_t)n * (K / 64);

    float acc = 0.0f;
    const int n_bytes = K / 2;
    const int vec_end = n_bytes - (n_bytes % 128);

    for (int base = 0; base < vec_end; base += 128) {
        int byte_off = base + lane_id * 4;
        uint32_t packed4 = *reinterpret_cast<const uint32_t*>(w_row + byte_off);

        int w0 = (packed4        & 0xF) - 8;
        int w1 = ((packed4 >> 4) & 0xF) - 8;
        int w2 = ((packed4 >> 8) & 0xF) - 8;
        int w3 = ((packed4 >>12) & 0xF) - 8;
        int w4 = ((packed4 >>16) & 0xF) - 8;
        int w5 = ((packed4 >>20) & 0xF) - 8;
        int w6 = ((packed4 >>24) & 0xF) - 8;
        int w7 = ((packed4 >>28) & 0xF) - 8;

        int xk = byte_off * 2;
        float xv0 = __bfloat162float(x_row[xk    ]);
        float xv1 = __bfloat162float(x_row[xk + 1]);
        float xv2 = __bfloat162float(x_row[xk + 2]);
        float xv3 = __bfloat162float(x_row[xk + 3]);
        float xv4 = __bfloat162float(x_row[xk + 4]);
        float xv5 = __bfloat162float(x_row[xk + 5]);
        float xv6 = __bfloat162float(x_row[xk + 6]);
        float xv7 = __bfloat162float(x_row[xk + 7]);

        float sc = __bfloat162float(s_row[byte_off / 32]);

        acc += sc * ((float)w0*xv0 + (float)w1*xv1 + (float)w2*xv2 + (float)w3*xv3
                   + (float)w4*xv4 + (float)w5*xv5 + (float)w6*xv6 + (float)w7*xv7);
    }

    // Tail (if K/2 not divisible by 128 — typically doesn't execute).
    for (int bi = vec_end + lane_id; bi < n_bytes; bi += 32) {
        float sc = __bfloat162float(s_row[bi / 32]);
        uint8_t b = w_row[bi];
        int w0 = (b & 0xF) - 8;
        int w1 = (b >> 4) - 8;
        int k0 = bi * 2;
        acc += (float)w0 * sc * __bfloat162float(x_row[k0    ]);
        acc += (float)w1 * sc * __bfloat162float(x_row[k0 + 1]);
    }

    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        acc += __shfl_xor_sync(0xffffffff, acc, off);

    if (lane_id == 0) {
        if (residual)
            acc += __bfloat162float(residual[m * N + n]);
        y[m * N + n] = __float2bfloat16(acc);
    }
}

// SMEM carveout experiment: reverted — reducing L1 hurts more than SMEM helps.


cudaError_t linear_q4_g64_bf16(
    const uint8_t* w_packed,
    const __nv_bfloat16* w_scales,
    const __nv_bfloat16* x,
    __nv_bfloat16* y,
    int M, int N, int K,
    cudaStream_t stream) {
    if (!w_packed || !w_scales || !x || !y || M <= 0 || N <= 0 || K <= 0)
        return cudaErrorInvalidValue;
    if (K % 64 != 0 || K > SMEM_X_MAX)
        return cudaErrorInvalidValue;

    int n_blocks = (N + 7) / 8;
    dim3 grid(n_blocks, M);
    dim3 block(256);
    size_t smem_bytes = K * sizeof(__nv_bfloat16);
    linear_q4_g64_warp_kernel<<<grid, block, smem_bytes, stream>>>(
        w_packed, w_scales, x, y, nullptr, M, N, K, nullptr, 0.0f, nullptr);
    return cudaGetLastError();
}

cudaError_t linear_q4_g64_bf16_residual(
    const uint8_t* w_packed,
    const __nv_bfloat16* w_scales,
    const __nv_bfloat16* x,
    __nv_bfloat16* y,
    const __nv_bfloat16* residual,
    int M, int N, int K,
    cudaStream_t stream) {
    if (!w_packed || !w_scales || !x || !y || !residual || M <= 0 || N <= 0 || K <= 0)
        return cudaErrorInvalidValue;
    if (K % 64 != 0 || K > SMEM_X_MAX)
        return cudaErrorInvalidValue;

    int n_blocks = (N + 7) / 8;
    dim3 grid(n_blocks, M);
    dim3 block(256);
    size_t smem_bytes = K * sizeof(__nv_bfloat16);
    linear_q4_g64_warp_kernel<<<grid, block, smem_bytes, stream>>>(
        w_packed, w_scales, x, y, residual, M, N, K, nullptr, 0.0f, nullptr);
    return cudaGetLastError();
}

// Fused rmsnorm + Q4 GEMV: normalizes x in SMEM, then does the matmul.
// Eliminates the separate rmsnorm kernel launch.
cudaError_t linear_q4_g64_bf16_rmsnorm(
    const uint8_t* w_packed,
    const __nv_bfloat16* w_scales,
    const __nv_bfloat16* gamma,
    float eps,
    const __nv_bfloat16* x,
    __nv_bfloat16* y,
    int M, int N, int K,
    cudaStream_t stream) {
    if (!w_packed || !w_scales || !x || !y || !gamma || M <= 0 || N <= 0 || K <= 0)
        return cudaErrorInvalidValue;
    if (K % 64 != 0 || K > SMEM_X_MAX)
        return cudaErrorInvalidValue;
    int n_blocks = (N + 7) / 8;
    dim3 grid(n_blocks, M);
    dim3 block(256);
    size_t smem_bytes = K * sizeof(__nv_bfloat16);
    linear_q4_g64_warp_kernel<<<grid, block, smem_bytes, stream>>>(
        w_packed, w_scales, x, y, nullptr, M, N, K, gamma, eps, nullptr);
    return cudaGetLastError();
}

// Fused swiglu + residual GEMV: reads gate, applies silu(gate)*up, then GEMV + residual.
cudaError_t linear_q4_g64_bf16_residual_swiglu(
    const uint8_t* w_packed, const __nv_bfloat16* w_scales,
    const __nv_bfloat16* gate, const __nv_bfloat16* up,
    __nv_bfloat16* y, const __nv_bfloat16* residual,
    int M, int N, int K, cudaStream_t stream) {
    if (!w_packed || !w_scales || !gate || !up || !y || !residual || M <= 0 || N <= 0 || K <= 0)
        return cudaErrorInvalidValue;
    if (K % 64 != 0 || K > SMEM_X_MAX)
        return cudaErrorInvalidValue;
    int n_blocks = (N + 7) / 8;
    dim3 grid(n_blocks, M);
    dim3 block(256);
    size_t smem_bytes = K * sizeof(__nv_bfloat16);
    linear_q4_g64_warp_kernel<<<grid, block, smem_bytes, stream>>>(
        w_packed, w_scales, gate, y, residual, M, N, K, nullptr, 0.0f, up);
    return cudaGetLastError();
}

// Fused 2-matrix GEMV: processes two weight matrices in a single launch.
// Uses blockIdx.z to select matrix. Both must have same N, K, and input x.
// Eliminates 1 kernel launch per call (saves 44 launches for k+v, 44 for gate+up).
__global__ void linear_q4_g64_fused2_kernel(
    const uint8_t* __restrict__ w0, const __nv_bfloat16* __restrict__ s0,
    const uint8_t* __restrict__ w1, const __nv_bfloat16* __restrict__ s1,
    const __nv_bfloat16* __restrict__ x,
    __nv_bfloat16* __restrict__ y0, __nv_bfloat16* __restrict__ y1,
    int M, int N, int K,
    const __nv_bfloat16* __restrict__ gamma, float eps) {
    const int z = blockIdx.z;
    const uint8_t* w_packed = (z == 0) ? w0 : w1;
    const __nv_bfloat16* w_scales = (z == 0) ? s0 : s1;
    __nv_bfloat16* y = (z == 0) ? y0 : y1;
    const int m = blockIdx.y;
    const int warp_id = threadIdx.x >> 5;
    const int lane_id = threadIdx.x & 31;
    const int n = blockIdx.x * 8 + warp_id;
    if (n >= N) {
        // participate in SMEM load to avoid deadlock
    }
    extern __shared__ __nv_bfloat16 smem_x[];
    const __nv_bfloat16* x_row_global = x + (size_t)m * K;
    for (int i = threadIdx.x; i < K; i += blockDim.x) smem_x[i] = x_row_global[i];
    __syncthreads();
    if (gamma) {
        float ss = 0.0f;
        for (int i = threadIdx.x; i < K; i += blockDim.x) {
            float v = __bfloat162float(smem_x[i]); ss += v * v;
        }
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) ss += __shfl_xor_sync(0xffffffff, ss, off);
        __shared__ float fw_sums[8];
        if ((threadIdx.x & 31) == 0) fw_sums[threadIdx.x >> 5] = ss;
        __syncthreads();
        if (threadIdx.x < 32) {
            float t = (threadIdx.x < 8) ? fw_sums[threadIdx.x] : 0.0f;
            #pragma unroll
            for (int off = 4; off > 0; off >>= 1) t += __shfl_xor_sync(0xffffffff, t, off);
            if (threadIdx.x == 0) fw_sums[0] = rsqrtf(t / (float)K + eps);
        }
        __syncthreads();
        float inv = fw_sums[0];
        for (int i = threadIdx.x; i < K; i += blockDim.x)
            smem_x[i] = __float2bfloat16(__bfloat162float(smem_x[i]) * inv * __bfloat162float(gamma[i]));
        __syncthreads();
    }
    if (n >= N || m >= M) return;
    const __nv_bfloat16* x_row = smem_x;
    const uint8_t* w_row = w_packed + (size_t)n * (K / 2);
    const __nv_bfloat16* s_row = w_scales + (size_t)n * (K / 64);
    float acc = 0.0f;
    const int n_bytes = K / 2;
    const int vec_end = n_bytes - (n_bytes % 128);
    for (int base = 0; base < vec_end; base += 128) {
        int byte_off = base + lane_id * 4;
        uint32_t p4 = *reinterpret_cast<const uint32_t*>(w_row + byte_off);
        int w0b = (p4 & 0xF) - 8, w1b = ((p4>>4)&0xF) - 8, w2b = ((p4>>8)&0xF) - 8, w3b = ((p4>>12)&0xF) - 8;
        int w4b = ((p4>>16)&0xF) - 8, w5b = ((p4>>20)&0xF) - 8, w6b = ((p4>>24)&0xF) - 8, w7b = ((p4>>28)&0xF) - 8;
        int xk = byte_off * 2;
        float xv0 = __bfloat162float(x_row[xk]), xv1 = __bfloat162float(x_row[xk+1]);
        float xv2 = __bfloat162float(x_row[xk+2]), xv3 = __bfloat162float(x_row[xk+3]);
        float xv4 = __bfloat162float(x_row[xk+4]), xv5 = __bfloat162float(x_row[xk+5]);
        float xv6 = __bfloat162float(x_row[xk+6]), xv7 = __bfloat162float(x_row[xk+7]);
        float sc = __bfloat162float(s_row[byte_off / 32]);
        acc += sc * ((float)w0b*xv0 + (float)w1b*xv1 + (float)w2b*xv2 + (float)w3b*xv3
                   + (float)w4b*xv4 + (float)w5b*xv5 + (float)w6b*xv6 + (float)w7b*xv7);
    }
    for (int bi = vec_end + lane_id; bi < n_bytes; bi += 32) {
        float sc = __bfloat162float(s_row[bi / 32]);
        uint8_t b = w_row[bi];
        int w0b = (b & 0xF) - 8, w1b = (b >> 4) - 8;
        int k0 = bi * 2;
        acc += (float)w0b * sc * __bfloat162float(x_row[k0]);
        acc += (float)w1b * sc * __bfloat162float(x_row[k0 + 1]);
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_xor_sync(0xffffffff, acc, off);
    if (lane_id == 0) y[m * N + n] = __float2bfloat16(acc);
}

cudaError_t linear_q4_g64_bf16_fused2(
    const uint8_t* w0, const __nv_bfloat16* s0,
    const uint8_t* w1, const __nv_bfloat16* s1,
    const __nv_bfloat16* x,
    __nv_bfloat16* y0, __nv_bfloat16* y1,
    int M, int N, int K,
    cudaStream_t stream) {
    if (K % 64 != 0 || K > SMEM_X_MAX) return cudaErrorInvalidValue;
    int n_blocks = (N + 7) / 8;
    dim3 grid(n_blocks, M, 2);
    dim3 block(256);
    size_t smem = K * sizeof(__nv_bfloat16);
    linear_q4_g64_fused2_kernel<<<grid, block, smem, stream>>>(
        w0, s0, w1, s1, x, y0, y1, M, N, K, nullptr, 0.0f);
    return cudaGetLastError();
}

// Fused rmsnorm + 2-matrix GEMV: normalizes x, then processes both matrices.
cudaError_t linear_q4_g64_bf16_fused2_rmsnorm(
    const uint8_t* w0, const __nv_bfloat16* s0,
    const uint8_t* w1, const __nv_bfloat16* s1,
    const __nv_bfloat16* gamma, float eps,
    const __nv_bfloat16* x,
    __nv_bfloat16* y0, __nv_bfloat16* y1,
    int M, int N, int K,
    cudaStream_t stream) {
    if (K % 64 != 0 || K > SMEM_X_MAX) return cudaErrorInvalidValue;
    int n_blocks = (N + 7) / 8;
    dim3 grid(n_blocks, M, 2);
    dim3 block(256);
    size_t smem = K * sizeof(__nv_bfloat16);
    linear_q4_g64_fused2_kernel<<<grid, block, smem, stream>>>(
        w0, s0, w1, s1, x, y0, y1, M, N, K, gamma, eps);
    return cudaGetLastError();
}
}  // namespace ops
}  // namespace viper
