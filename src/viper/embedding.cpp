// Embedding gather — Tensor -> CUDA kernel.
//
// Contract: embedding_forward(table, token_ids, y)
//   table:    [V, H] bf16
//   token_ids:[B, T] int32
//   y:        [B, T, H] bf16 (output)
#include "viper/ops.h"
#include "viper/tensor.h"
#include "viper/cuda_check.h"
#include "viper/safety.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace viper::ops {

Status embedding_forward(const Tensor& embedding_table,
                         const Tensor& token_ids, Tensor& y) {
    if (embedding_table.dtype() != DType::BF16 || y.dtype() != DType::BF16) {
        return Status(StatusCode::INVALID_ARGUMENT,
                      "embedding_forward: table and y must be BF16");
    }
    if (token_ids.dtype() != DType::INT32) {
        return Status(StatusCode::INVALID_ARGUMENT,
                      "embedding_forward: token_ids must be INT32");
    }
    if (embedding_table.shape().rank() != 2 || y.shape().rank() != 3 ||
        token_ids.shape().rank() != 2) {
        return Status(StatusCode::SHAPE_MISMATCH,
                      "embedding_forward: rank mismatch");
    }
    const int V = static_cast<int>(embedding_table.shape()[0]);
    const int H = static_cast<int>(embedding_table.shape()[1]);
    const int B = static_cast<int>(token_ids.shape()[0]);
    const int T = static_cast<int>(token_ids.shape()[1]);
    if (y.shape()[0] != B || y.shape()[1] != T || y.shape()[2] != H) {
        return Status(StatusCode::SHAPE_MISMATCH,
                      "embedding_forward: y must be [B, T, H]");
    }
    if (H % 8 != 0) {
        return Status(StatusCode::INVALID_ARGUMENT,
                      "embedding_forward: H must be a multiple of 8");
    }

    cudaError_t err = embedding_gather_bf16_i32(
        embedding_table.data_as<__nv_bfloat16>(),
        token_ids.data_as<int32_t>(),
        y.data_as<__nv_bfloat16>(),
        B, T, V, H, /*stream=*/0);

    return safety::classify_cuda(err);
}

}  // namespace viper::ops
