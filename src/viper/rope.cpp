// RoPE forward — stub. Real BF16 implementation lives in
// kernels/ops/rope_kernel.cu (main thread owns).
#include "viper/ops.h"

namespace viper::ops {

Status rope_forward(Tensor& q, Tensor& k, const Tensor& position_ids, f32 theta,
                    i32 head_dim) {
    (void)q;
    (void)k;
    (void)position_ids;
    (void)theta;
    (void)head_dim;
    return Status::Unimplemented(
        "rope_forward: kernel lands in kernels/ops/rope_kernel.cu (M2)");
}

}  // namespace viper::ops