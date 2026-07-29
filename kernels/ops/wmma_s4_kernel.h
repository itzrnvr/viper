#ifndef VIPER_WMMA_S4_KERNEL_H
#define VIPER_WMMA_S4_KERNEL_H
#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace viper {
namespace ops {

// WMMA s4 GEMV with activation quantization and rmsnorm fusion.
// Uses experimental::precision::s4 with m8n8k32 tiles.
// Achieves 359 GB/s (80% of peak) — 1.81× scalar Q4.
//
// Q4 weights are DIRECTLY compatible (same nibble format as s4).
// BF16 activations are quantized to s4 per-group-of-32 on-the-fly.
//
// Tile: M=8 (padded from actual M), N=8, K=32.
// Grid: ceil(N/8) / 8 warps_per_block. Block: 256 threads.
//
// NOTE: This kernel processes M=8 tokens per call. For M=1 decode:
// pad to M=8 (7/8 wasted compute, but tensor core compute is free).
// For M=5 spec decode: pad to M=8 (3/8 wasted).

// Kernel declaration (defined in wmma_s4_kernel.cu)
__global__ void wmma_s4_gemv_kernel(
    const uint8_t* __restrict__ w_q4,
    const __nv_bfloat16* __restrict__ w_scales,
    const __nv_bfloat16* __restrict__ x,
    float* __restrict__ y,
    const __nv_bfloat16* __restrict__ residual,
    int M, int N, int K,
    const __nv_bfloat16* __restrict__ gamma,
    float eps);

// Launcher
inline cudaError_t wmma_s4_gemv(
    const uint8_t* w, const __nv_bfloat16* s,
    const __nv_bfloat16* x, float* y,
    const __nv_bfloat16* residual,
    int M, int N, int K,
    const __nv_bfloat16* gamma, float eps,
    cudaStream_t stream = 0) {
    // SMEM: 8*K*2 (bf16 act) + 8*K/8*4 (s4 packed) + K/32*2 (act scales) + 256
    size_t smem = 8 * K * 2 + 8 * (K / 8) * 4 + (K / 32) * 2 + 256;
    int n_blocks = (N / 8 + 7) / 8;
    wmma_s4_gemv_kernel<<<n_blocks, 256, smem, stream>>>(
        w, s, x, y, residual, M, N, K, gamma, eps);
    return cudaGetLastError();
}

}  // namespace ops
}  // namespace viper

#endif
