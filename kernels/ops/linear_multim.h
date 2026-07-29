#ifndef VIPER_LINEAR_MULTIM_H
#define VIPER_LINEAR_MULTIM_H
#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace viper {
namespace ops {

// Multi-M GEMV: processes ALL M tokens per block. Weights read ONCE from DRAM.
// Each warp handles 1 output channel for all M tokens (activations from L2 cache).
cudaError_t linear_q4_multim(
    const uint8_t* w, const __nv_bfloat16* s,
    const __nv_bfloat16* x, __nv_bfloat16* y,
    int M, int N, int K, cudaStream_t stream);

cudaError_t linear_q4_multim_residual(
    const uint8_t* w, const __nv_bfloat16* s,
    const __nv_bfloat16* x, __nv_bfloat16* y,
    const __nv_bfloat16* residual,
    int M, int N, int K, cudaStream_t stream);

// Quantize BF16 activations to INT8 with per-group-of-64 scales.
// Output: xq[M*K] INT8 + xs[M*K/64] FLOAT scales.
cudaError_t quantize_to_q8(
    const __nv_bfloat16* x, int8_t* xq, float* xs,
    int M, int K, cudaStream_t stream);

// DP4A GEMV with PRE-QUANTIZED Q8 activations (no internal quantize, no SMEM, no sync).
// Use after quantize_to_q8 for the same activation vector.
cudaError_t linear_q4_dp4a_prequantized(
    const uint8_t* w, const __nv_bfloat16* s,
    const int8_t* xq, const float* xs,
    __nv_bfloat16* y,
    int M, int N, int K, cudaStream_t stream);

// Same with residual add.
cudaError_t linear_q4_dp4a_prequantized_residual(
    const uint8_t* w, const __nv_bfloat16* s,
    const int8_t* xq, const float* xs,
    __nv_bfloat16* y, const __nv_bfloat16* residual,
    int M, int N, int K, cudaStream_t stream);

}  // namespace ops
}  // namespace viper

#endif
