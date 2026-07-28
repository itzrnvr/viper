// SwiGLU activation — stub. Real impl lives in kernels/ops/swiglu_kernel.cu.
#include "viper/ops.h"

namespace viper::ops {

Status swiglu_forward(const Tensor& gate, const Tensor& up, Tensor& y) {
    (void)gate;
    (void)up;
    (void)y;
    return Status::Unimplemented(
        "swiglu_forward: kernel lands in kernels/ops/swiglu_kernel.cu (M2)");
}

}  // namespace viper::ops