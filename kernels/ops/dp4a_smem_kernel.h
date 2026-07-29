#ifndef VIPER_DP4A_SMEM_H
#define VIPER_DP4A_SMEM_H
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace viper {
namespace ops {

cudaError_t dp4a_smem_gemv(
    const uint8_t* w, const __nv_bfloat16* s,
    const int8_t* xq, const float* xs,
    __nv_bfloat16* y,
    int M, int N, int K, cudaStream_t stream);

cudaError_t dp4a_smem_gemv_residual(
    const uint8_t* w, const __nv_bfloat16* s,
    const int8_t* xq, const float* xs,
    __nv_bfloat16* y, const __nv_bfloat16* residual,
    int M, int N, int K, cudaStream_t stream);

cudaError_t dp4a_smem_gemv_fused2(
    const uint8_t* w0, const __nv_bfloat16* s0,
    const uint8_t* w1, const __nv_bfloat16* s1,
    const int8_t* xq, const float* xs,
    __nv_bfloat16* y0, __nv_bfloat16* y1,
    int M, int N, int K, cudaStream_t stream);

}  // namespace ops
}  // namespace viper
#endif
