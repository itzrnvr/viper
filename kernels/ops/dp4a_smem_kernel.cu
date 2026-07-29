/*
 * viper DP4A GEMV with pre-quantized Q8 activations from SMEM.
 *
 * Reads INT8 activations (quantized by rmsnorm_quantize) from SMEM.
 * Uses __dp4a for 4 multiply-adds per instruction.
 * Uses __vsubss4 + __byte_perm for Q4 weight unpacking (6 inst vs 24 scalar).
 *
 * Total instructions per 8 weight elements: ~11 (vs 35 scalar) → 3.2x fewer.
 * Fully memory-bound: ~370 GB/s expected (98% of streaming peak).
 *
 * SMEM: K bytes (INT8) + K/64 * 4 bytes (scales) — HALF of scalar's K*2 bytes.
 */
#ifndef VIPER_DP4A_SMEM_KERNEL_H
#define VIPER_DP4A_SMEM_KERNEL_H

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace viper {
namespace ops {

// DP4A GEMV with pre-quantized Q8 activations.
// DP4A GEMV with pre-quantized Q8 activations from L1 cache (NO SMEM, NO sync).
__global__ void dp4a_smem_gemv_kernel(
    const uint8_t* __restrict__ w_packed,
    const __nv_bfloat16* __restrict__ w_scales,
    const int8_t* __restrict__ xq_global,
    const float* __restrict__ xq_scales,
    __nv_bfloat16* __restrict__ y,
    const __nv_bfloat16* __restrict__ residual,
    int M, int N, int K) {
    const int m = blockIdx.y;
    const int warp_id = threadIdx.x >> 5;
    const int lane_id = threadIdx.x & 31;
    const int n = blockIdx.x * (blockDim.x >> 5) + warp_id;
    if (n >= N || m >= M) return;
    const uint32_t* w32 = reinterpret_cast<const uint32_t*>(w_packed + (size_t)n * (K / 2));
    const __nv_bfloat16* s_row = w_scales + (size_t)n * (K / 64);
    const int32_t* xq32 = reinterpret_cast<const int32_t*>(xq_global + (size_t)m * K);
    const float* xs_row = xq_scales + (size_t)m * (K / 64);

    float acc = 0.0f;
    const int n_quads = K / 8;  // 8 weight elements per uint32, processed as 2 DP4A

    for (int i = lane_id; i < n_quads; i += 32) {
        uint32_t packed4 = w32[i];
        // Q4 weight unpack: __vsubss4 + __byte_perm (6 instructions)
        uint32_t lo = packed4 & 0x0F0F0F0F;
        uint32_t hi = (packed4 >> 4) & 0x0F0F0F0F;
        int32_t w_lo = __byte_perm(__vsubss4(lo, 0x08080808), __vsubss4(hi, 0x08080808), 0x5140);
        int32_t w_hi = __byte_perm(__vsubss4(lo, 0x08080808), __vsubss4(hi, 0x08080808), 0x7362);
        // Q8 activations: read as int32 (4 INT8 packed)
        int32_t x_lo = xq32[i * 2];
        int32_t x_hi = xq32[i * 2 + 1];
        // DP4A: 4 multiply-adds per instruction
        int32_t dot = __dp4a(w_lo, x_lo, 0);
        dot = __dp4a(w_hi, x_hi, dot);
        // Apply scales
        int g = i / 8;
        acc += __bfloat162float(s_row[g]) * xs_row[g] * (float)dot;
    }

    // Warp reduce
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        acc += __shfl_xor_sync(0xffffffff, acc, off);

    if (lane_id == 0) {
        if (residual)
            acc += __bfloat162float(residual[m * N + n]);
        y[m * N + n] = __float2bfloat16(acc);
    }
}

// Launcher
cudaError_t dp4a_smem_gemv(
    const uint8_t* w, const __nv_bfloat16* s,
    const int8_t* xq, const float* xs,
    __nv_bfloat16* y,
    int M, int N, int K, cudaStream_t stream) {
    if (K % 64 != 0) return cudaErrorInvalidValue;
    dp4a_smem_gemv_kernel<<<(N+7)/8, 256, 0, stream>>>(
        w, s, xq, xs, y, nullptr, M, N, K);
    return cudaGetLastError();
}

cudaError_t dp4a_smem_gemv_residual(
    const uint8_t* w, const __nv_bfloat16* s,
    const int8_t* xq, const float* xs,
    __nv_bfloat16* y, const __nv_bfloat16* residual,
    int M, int N, int K, cudaStream_t stream) {
    if (K % 64 != 0) return cudaErrorInvalidValue;
    dp4a_smem_gemv_kernel<<<(N+7)/8, 256, 0, stream>>>(
        w, s, xq, xs, y, residual, M, N, K);
}

// Fused2 DP4A: process 2 weight matrices with same Q8 input in 1 launch.
// blockIdx.z selects matrix (0 or 1). Saves 1 launch per k/v and gate/up.
__global__ void dp4a_smem_gemv_fused2_kernel(
    const uint8_t* __restrict__ w0, const __nv_bfloat16* __restrict__ s0,
    const uint8_t* __restrict__ w1, const __nv_bfloat16* __restrict__ s1,
    const int8_t* __restrict__ xq_global,
    const float* __restrict__ xq_scales,
    __nv_bfloat16* __restrict__ y0, __nv_bfloat16* __restrict__ y1,
    int M, int N, int K) {
    const int z = blockIdx.z;
    const uint8_t* w_packed = (z == 0) ? w0 : w1;
    const __nv_bfloat16* w_scales = (z == 0) ? s0 : s1;
    __nv_bfloat16* y = (z == 0) ? y0 : y1;

    const int m = blockIdx.y;
    const int warp_id = threadIdx.x >> 5;
    const int lane_id = threadIdx.x & 31;
    const int n = blockIdx.x * (blockDim.x >> 5) + warp_id;

    const int8_t* xq_row = xq_global + (size_t)m * K;
    const float* xs_row = xq_scales + (size_t)m * (K / 64);
    if (n >= N || m >= M) return;
    const uint32_t* w32 = reinterpret_cast<const uint32_t*>(w_packed + (size_t)n * (K / 2));
    const __nv_bfloat16* s_row = w_scales + (size_t)n * (K / 64);
    const int32_t* xq32 = reinterpret_cast<const int32_t*>(xq_row);
    const int n_quads = K / 8;
    float acc = 0.0f;
    for (int i = lane_id; i < n_quads; i += 32) {
        uint32_t packed4 = w32[i];
        uint32_t lo = packed4 & 0x0F0F0F0F;
        uint32_t hi = (packed4 >> 4) & 0x0F0F0F0F;
        int32_t w_lo = __byte_perm(__vsubss4(lo, 0x08080808), __vsubss4(hi, 0x08080808), 0x5140);
        int32_t w_hi = __byte_perm(__vsubss4(lo, 0x08080808), __vsubss4(hi, 0x08080808), 0x7362);
        int32_t dot = __dp4a(w_lo, xq32[i * 2], 0);
        dot = __dp4a(w_hi, xq32[i * 2 + 1], dot);
        int g = i / 8;
        acc += __bfloat162float(s_row[g]) * xs_row[g] * (float)dot;
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_xor_sync(0xffffffff, acc, off);
    if (lane_id == 0) y[m * N + n] = __float2bfloat16(acc);
}

cudaError_t dp4a_smem_gemv_fused2(
    const uint8_t* w0, const __nv_bfloat16* s0,
    const uint8_t* w1, const __nv_bfloat16* s1,
    const int8_t* xq, const float* xs,
    __nv_bfloat16* y0, __nv_bfloat16* y1,
    int M, int N, int K, cudaStream_t stream) {
    if (K % 64 != 0) return cudaErrorInvalidValue;
    int n_blocks = (N + 7) / 8;
    dp4a_smem_gemv_fused2_kernel<<<dim3(n_blocks, M, 2), 256, 0, stream>>>(
        w0, s0, w1, s1, xq, xs, y0, y1, M, N, K);
    return cudaGetLastError();
}
}  // namespace ops
}  // namespace viper

#endif
