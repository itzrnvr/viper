// Residual add — stub. Real impl lives in kernels/ops/residual_kernel.cu.
#include "viper/ops.h"

namespace viper::ops {

Status residual_forward(const Tensor& x, const Tensor& residual, Tensor& y) {
    (void)x;
    (void)residual;
    (void)y;
    return Status::Unimplemented(
        "residual_forward: kernel lands in kernels/ops/residual_kernel.cu (M2)");
}

}  // namespace viper::ops