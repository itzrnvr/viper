// RMSNorm forward — stub. Real BF16 implementation lives in
// kernels/ops/rmsnorm_kernel.cu (main thread owns). This stub matches the
// public interface in include/viper/ops.h so callers can link against the
// engine even before the kernel lands.
#include "viper/ops.h"

namespace viper::ops {

Status rmsnorm_forward(const Tensor& x, const Tensor& weight, f32 eps,
                       Tensor& y) {
    (void)x;
    (void)weight;
    (void)eps;
    (void)y;
    return Status::Unimplemented(
        "rmsnorm_forward: kernel lands in kernels/ops/rmsnorm_kernel.cu (M2)");
}

}  // namespace viper::ops