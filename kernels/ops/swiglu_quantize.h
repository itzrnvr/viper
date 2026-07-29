#ifndef VIPER_SWIGLU_QUANTIZE_H
#define VIPER_SWIGLU_QUANTIZE_H
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace viper {
namespace ops {

// Fused swiglu + Q8 quantize. Computes silu(gate)*up AND quantizes to INT8.
cudaError_t swiglu_quantize_bf16(
    const __nv_bfloat16* gate,
    const __nv_bfloat16* up,
    __nv_bfloat16* out_bf16,
    int8_t* out_q8,
    float* out_scales,
    int I,
    cudaStream_t stream);

}  // namespace ops
}  // namespace viper
#endif
