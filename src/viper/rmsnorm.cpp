// RMSNorm forward — Tensor -> CUDA kernel.
//
// Contract (include/viper/ops.h):
//   rmsnorm_forward(x, weight, eps, y)
//     x:     [rows, H] bf16, contiguous
//     weight:[H] bf16
//     y:     [rows, H] bf16 (output)
//
// Kernel: viper::ops::rmsnorm_forward_bf16 (kernels/ops/rmsnorm_kernel.cu)
//   one block per row, fp32 internal accumulation, warp+block reduce.
//
// Validates shape + dtype, then calls the kernel on the default stream.
#include "viper/ops.h"
#include "viper/tensor.h"
#include "viper/cuda_check.h"
#include "viper/safety.h"

#include <cuda_bf16.h>

namespace viper::ops {

Status rmsnorm_forward(const Tensor& x, const Tensor& weight, f32 eps,
                       Tensor& y) {
    if (x.dtype() != DType::BF16 || weight.dtype() != DType::BF16 ||
        y.dtype() != DType::BF16) {
        return Status(StatusCode::INVALID_ARGUMENT,
                      "rmsnorm_forward: all tensors must be BF16");
    }
    if (x.shape().rank() != 2 || y.shape().rank() != 2 ||
        weight.shape().rank() != 1) {
        return Status(StatusCode::SHAPE_MISMATCH,
                      "rmsnorm_forward: x and y must be [rows, H], weight must be [H]");
    }
    if (weight.shape()[0] != x.shape()[1]) {
        return Status(StatusCode::SHAPE_MISMATCH,
                      "rmsnorm_forward: weight[0] must equal x[1]");
    }
    if (x.shape()[0] != y.shape()[0] || x.shape()[1] != y.shape()[1]) {
        return Status(StatusCode::SHAPE_MISMATCH,
                      "rmsnorm_forward: x and y must have identical shape");
    }

    const int rows = static_cast<int>(x.shape()[0]);
    const int H    = static_cast<int>(x.shape()[1]);

    cudaError_t err = rmsnorm_forward_bf16(
        x.data_as<__nv_bfloat16>(),
        weight.data_as<__nv_bfloat16>(),
        y.data_as<__nv_bfloat16>(),
        rows, H, eps, /*stream=*/0);

    return safety::classify_cuda(err);
}

}  // namespace viper::ops
