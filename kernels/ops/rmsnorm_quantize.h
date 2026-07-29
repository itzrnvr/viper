#ifndef VIPER_RMSNORM_QUANTIZE_H
#define VIPER_RMSNORM_QUANTIZE_H
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace viper {
namespace ops {

// Fused rmsnorm + Q8 quantize. Outputs both BF16 norm and INT8 Q8.
cudaError_t rmsnorm_quantize_bf16(
    const __nv_bfloat16* x,
    const __nv_bfloat16* gamma,
    __nv_bfloat16* out_norm,
    int8_t* out_q8,
    float* out_scales,
    int H,
    float eps,
    cudaStream_t stream);

}  // namespace ops
}  // namespace viper
#endif
