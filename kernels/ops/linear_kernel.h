/*
 * viper Linear Q4_G64 GEMV/GEMM kernel — header.
 *
 * PURPOSE: W4A16 GEMV for the closed Q4_G64 weight format. Reads 4-bit
 *          weights + per-group FP16 scale, dequantizes to BF16 in
 *          registers/SMEM, runs mma.sync BF16 -> FP32. This is the
 *          bandwidth-bound core kernel (Q4 body = 1.6 GB, drives the
 *          per-token cost).
 *
 * Q4_G64 LAYOUT (our format, 4.25 bits/weight):
 *   - For each row of N columns, weights are packed in groups of 64.
 *   - Per group: 64 x 4-bit weights (32 bytes) + 1 x FP16 scale (2 bytes) = 34 bytes.
 *   - Per row: ceil(N / 64) groups * 34 bytes.
 *   - Weights are SYMMETRIC around zero: stored_value = (weight + 8).
 *   - Dequant: bf16 = (stored - 8) * scale.
 *
 * ALGORITHM:
 *   - M = batch * seq_len, N = out features, K = in features.
 *   - One block computes a (BM x BN) tile of output. Block index
 *     blockIdx.x = N tile, blockIdx.y = M tile.
 *   - Each block streams its Q4 weights via cp.async, dequants in
 *     SMEM/registers, mma.sync accumulates BF16 * BF16 -> FP32.
 *   - For T=1 (decode), BM=1, BN=128, BK=128.
 *
 * CORRECTNESS:
 *   - All math in FP32 (mma.sync accumulator).
 *   - bf16 cast round-to-nearest.
 *   - Symmetric quantization: -8..7 stored, dequant shifts by 8.
 *
 * SAFETY:
 *   - Caller passes VRAM-checked tensor pointers.
 *   - No global allocations inside the kernel.
 *   - SMEM usage: tile_BF16 BMxBK + tile_BF16 BKxBN.
 */
#ifndef VIPER_LINEAR_KERNEL_H
#define VIPER_LINEAR_KERNEL_H

#include <cuda_bf16.h>
#include <cstdint>

namespace viper {
namespace ops {

// Linear forward, Q4_G64 weights, BF16 activations, FP32 accumulate.
//   y[M, N] = x[M, K] @ dequant(w_q4[N, K]).T
//
// w_packed: [N, K/2] uint8 (4-bit weights packed two-per-byte)
// w_scales: [N, K/64] __nv_bfloat16 (one FP16 per 64-element group)
// x: [M, K] __nv_bfloat16
// y: [M, N] __nv_bfloat16
// M, N, K: shapes
// stream: CUDA stream
//
// Per-row stride for w_packed: (K / 2) bytes (4-bit weights)
// Per-row stride for w_scales: (K / 64) * 2 bytes (FP16 scales)
cudaError_t linear_q4_g64_bf16(
    const uint8_t* __restrict__ w_packed,
    const __nv_bfloat16* __restrict__ w_scales,
    const __nv_bfloat16* __restrict__ x,
    __nv_bfloat16* __restrict__ y,
    int M, int N, int K,
    cudaStream_t stream);

// Same as above but adds residual[M, N] to output: y = x @ w.T + residual.
// Eliminates a separate residual_add kernel launch.
cudaError_t linear_q4_g64_bf16_residual(
    const uint8_t* __restrict__ w_packed,
    const __nv_bfloat16* __restrict__ w_scales,
    const __nv_bfloat16* __restrict__ x,
    __nv_bfloat16* __restrict__ y,
    const __nv_bfloat16* __restrict__ residual,
    int M, int N, int K,
    cudaStream_t stream);

// Fused 2-matrix GEMV: processes two weight matrices in a single launch.
// Both matrices must have the same N, K. Outputs go to separate buffers.
cudaError_t linear_q4_g64_bf16_fused2(
    const uint8_t* __restrict__ w0, const __nv_bfloat16* __restrict__ s0,
    const uint8_t* __restrict__ w1, const __nv_bfloat16* __restrict__ s1,
    const __nv_bfloat16* __restrict__ x,
    __nv_bfloat16* __restrict__ y0, __nv_bfloat16* __restrict__ y1,
    int M, int N, int K,
    cudaStream_t stream);


// Fused rmsnorm + Q4 GEMV: normalizes x in SMEM, then does matmul.
cudaError_t linear_q4_g64_bf16_rmsnorm(
    const uint8_t* w_packed, const __nv_bfloat16* w_scales,
    const __nv_bfloat16* gamma, float eps,
    const __nv_bfloat16* x, __nv_bfloat16* y,
    int M, int N, int K, cudaStream_t stream);

// Fused rmsnorm + 2-matrix GEMV.
cudaError_t linear_q4_g64_bf16_fused2_rmsnorm(
    const uint8_t* w0, const __nv_bfloat16* s0,
    const uint8_t* w1, const __nv_bfloat16* s1,
    const __nv_bfloat16* gamma, float eps,
    const __nv_bfloat16* x,
    __nv_bfloat16* y0, __nv_bfloat16* y1,
    int M, int N, int K, cudaStream_t stream);
}  // namespace ops
}  // namespace viper

#endif  // VIPER_LINEAR_KERNEL_H
