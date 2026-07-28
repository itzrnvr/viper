// RoPE forward — Tensor -> CUDA kernel (cos/sin precompute + apply).
//
// Contract (include/viper/ops.h):
//   rope_forward(q, k, position_ids, theta, head_dim)
//     q, k:        [B, T, n_heads, head_dim] bf16, in-place
//     position_ids:[B, T] int32, absolute positions (0-indexed)
//     theta:       RoPE base (Nanbeige4.2-3B: 70000000)
//     head_dim:    must be 128
//
// Strategy:
//   1. Allocate cos/sin tables on device (T * head_dim * 4 bytes each).
//   2. Precompute via rope_precompute_cos_sin.
//   3. Apply in-place via rope_apply_inplace_bf16.
#include "viper/ops.h"
#include "viper/tensor.h"
#include "viper/cuda_check.h"
#include "viper/safety.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace viper::ops {

Status rope_forward(Tensor& q, Tensor& k, const Tensor& position_ids,
                    f32 theta, i32 head_dim) {
    if (q.dtype() != DType::BF16 || k.dtype() != DType::BF16) {
        return Status(StatusCode::INVALID_ARGUMENT,
                      "rope_forward: q and k must be BF16");
    }
    if (position_ids.dtype() != DType::INT32) {
        return Status(StatusCode::INVALID_ARGUMENT,
                      "rope_forward: position_ids must be INT32");
    }
    if (head_dim != 128) {
        return Status(StatusCode::INVALID_ARGUMENT,
                      "rope_forward: only head_dim=128 supported");
    }
    if (q.shape().rank() != 4 || k.shape().rank() != 4) {
        return Status(StatusCode::SHAPE_MISMATCH,
                      "rope_forward: q and k must be [B, T, n_heads, head_dim]");
    }
    // Shapes must match.
    for (int i = 0; i < 4; ++i) {
        if (q.shape()[i] != k.shape()[i]) {
            return Status(StatusCode::SHAPE_MISMATCH,
                          "rope_forward: q and k shape mismatch");
        }
    }
    const int B = static_cast<int>(q.shape()[0]);
    const int T = static_cast<int>(q.shape()[1]);
    const int n_q = static_cast<int>(q.shape()[2]);
    const int n_kv = static_cast<int>(k.shape()[2]);

    // Read position_ids[0] for the cos/sin precompute base. We support
    // a single starting position (B=1 or all-B-same); for heterogeneous
    // batches, the engine pads to a single position.
    std::vector<int32_t> h_pos(1);
    cudaError_t e = cudaMemcpy(h_pos.data(), position_ids.data_as<int32_t>(),
                              sizeof(int32_t), cudaMemcpyDeviceToHost);
    if (e != cudaSuccess) {
        return safety::classify_cuda(e);
    }
    const int pos_start = h_pos[0];

    // Allocate cos/sin tables.
    float *d_cos = nullptr, *d_sin = nullptr;
    const size_t cs_bytes = static_cast<size_t>(T) * head_dim * sizeof(float);
    if (cudaMalloc(&d_cos, cs_bytes) != cudaSuccess) {
        return Status(StatusCode::OUT_OF_MEMORY, "rope_forward: cos alloc");
    }
    if (cudaMalloc(&d_sin, cs_bytes) != cudaSuccess) {
        cudaFree(d_cos);
        return Status(StatusCode::OUT_OF_MEMORY, "rope_forward: sin alloc");
    }

    // Precompute.
    VIPER_CHECK_CUDA(rope_precompute_cos_sin(d_cos, d_sin, pos_start, T,
                                            theta, head_dim, 0));

    // Apply in-place.
    VIPER_CHECK_CUDA(rope_apply_inplace_bf16(
        q.data_as<__nv_bfloat16>(),
        k.data_as<__nv_bfloat16>(),
        d_cos, d_sin, B, n_q, n_kv, T, head_dim, 0));

    cudaFree(d_cos);
    cudaFree(d_sin);
    return safety::classify_cuda(cudaGetLastError());
}

}  // namespace viper::ops
