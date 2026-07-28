// SwiGLU — Tensor -> CUDA kernel.
//
// Contract: swiglu_forward(gate, up, y)
//   gate, up: [N] bf16
//   y:        [N] bf16 (output)
#include "viper/ops.h"
#include "viper/tensor.h"
#include "viper/cuda_check.h"
#include "viper/safety.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace viper::ops {

Status swiglu_forward(const Tensor& gate, const Tensor& up, Tensor& y) {
    if (gate.dtype() != DType::BF16 || up.dtype() != DType::BF16 ||
        y.dtype() != DType::BF16) {
        return Status(StatusCode::INVALID_ARGUMENT,
                      "swiglu_forward: all tensors must be BF16");
    }
    if (gate.shape().numel() != up.shape().numel() ||
        gate.shape().numel() != y.shape().numel()) {
        return Status(StatusCode::SHAPE_MISMATCH,
                      "swiglu_forward: gate/up/y must have the same numel");
    }
    const int N = static_cast<int>(gate.shape().numel());

    // In-place would be nice but the contract is out-of-place; copy
    // gate -> y, then apply silu(y) * up.
    cudaError_t err = swiglu_out_of_place_bf16(
        gate.data_as<__nv_bfloat16>(),
        up.data_as<__nv_bfloat16>(),
        y.data_as<__nv_bfloat16>(),
        N, /*stream=*/0);
    return safety::classify_cuda(err);
}

}  // namespace viper::ops
