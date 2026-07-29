#ifndef VIPER_FUSED_DP4A_KERNEL_H
#define VIPER_FUSED_DP4A_KERNEL_H
#include <cuda_bf1616.h>
#include <cuda_runtime.h>

namespace viper {
namespace ops {

cudaError_t fused_dp4a_rmsnorm(
    const uint8_t* w, const __nv_bfloat16* s, const __nv_bfloat16* gamma,
    float eps, const __nv_bfloat16* x, __nv_bfloat16* y,
    int M, int N, int K, cudaStream_t stream);

cudaError_t fused_dp4a_residual(
    const uint8_t* w, const __nv_bfloat16* s, const __nv_bfloat16* x,
    __nv_bfloat16* y, const __nv_bfloat16* residual,
    int M, int N, int K, cudaStream_t stream);

cudaError_t fused_dp4a_residual_swiglu(
    const uint8_t* w, const __nv_bfloat16* s, const __nv_bfloat16* gate,
    const __nv_bfloat16* up, __nv_bfloat16* y, const __nv_bfloat16* residual,
    int M, int N, int K, cudaStream_t stream);

}  // namespace ops
}  // namespace viper

#endif
