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

}  // namespace ops
}  // namespace viper

#endif
