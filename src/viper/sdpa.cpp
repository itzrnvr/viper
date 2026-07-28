// Scaled-dot-product attention (causal) — stub. Real FA-2 impl lives in
// kernels/ops/sdpa_kernel.cu (main thread owns).
#include "viper/ops.h"

namespace viper::ops {

Status sdpa_forward(const Tensor& q, const Tensor& k, const Tensor& v,
                    Tensor& out) {
    (void)q;
    (void)k;
    (void)v;
    (void)out;
    return Status::Unimplemented(
        "sdpa_forward: FA-2 kernel lands in kernels/ops/sdpa_kernel.cu (M2)");
}

}  // namespace viper::ops