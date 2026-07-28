// Residual add — Tensor -> CUDA kernel.
//
// Contract: residual_forward(x, residual, y)
//   x, residual: [N] bf16
//   y:           [N] bf16 (output)
#include "viper/ops.h"
#include "viper/tensor.h"
#include "viper/cuda_check.h"
#include "viper/safety.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace viper::ops {

Status residual_forward(const Tensor& x, const Tensor& residual, Tensor& y) {
    if (x.dtype() != DType::BF16 || residual.dtype() != DType::BF16 ||
        y.dtype() != DType::BF16) {
        return Status(StatusCode::INVALID_ARGUMENT,
                      "residual_forward: all tensors must be BF16");
    }
    if (x.shape().numel() != residual.shape().numel() ||
        x.shape().numel() != y.shape().numel()) {
        return Status(StatusCode::SHAPE_MISMATCH,
                      "residual_forward: x/residual/y must have the same numel");
    }
    const int N = static_cast<int>(x.shape().numel());

    cudaError_t err = residual_add_bf16(
        x.data_as<__nv_bfloat16>(),
        residual.data_as<__nv_bfloat16>(),
        y.data_as<__nv_bfloat16>(),
        N, /*stream=*/0);
    return safety::classify_cuda(err);
}

}  // namespace viper::ops
