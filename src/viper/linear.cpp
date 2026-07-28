// Linear (matmul + bias) — stub. Closed switch on weight dtype: BF16 / FP32
// are lossless, Q4_G64 / Q5_G64 / Q6_G64 / W8_G32 are lossy and gated by
// --quality-lossy. Real kernels live in kernels/ops/linear_kernel.cu (main
// thread owns).
#include "viper/ops.h"

#include "viper/quant.h"

namespace viper::ops {

Status linear_forward(const Tensor& w, const Tensor& b, const Tensor& x,
                      Tensor& y) {
    (void)b;
    (void)x;
    (void)y;
    switch (w.dtype()) {
        case DType::BF16:
            return Status::Unimplemented(
                "linear_forward BF16: kernel lands in kernels/ops/linear_kernel.cu (M2)");
        case DType::FP32:
            return Status::Unimplemented(
                "linear_forward FP32: kernel lands in kernels/ops/linear_kernel.cu (M2)");
        case DType::Q4_G64:
            return Status::Unimplemented(
                "linear_forward Q4_G64: gated by --quality-lossy; kernel lands in M3");
        case DType::Q5_G64:
        case DType::Q6_G64:
            return Status::Unimplemented(
                "linear_forward Q5/Q6: gated by --quality-lossy; kernel lands in M3");
        case DType::W8_G32:
            return Status::Unimplemented(
                "linear_forward W8_G32: gated by --quality-lossy; kernel lands in M3");
        default:
            return Status(StatusCode::INVALID_ARGUMENT,
                          "linear_forward: bad weight dtype");
    }
}

}  // namespace viper::ops