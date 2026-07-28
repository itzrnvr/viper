// Sampling — stub. Real impl lives in kernels/ops/sampling_kernel.cu.
#include "viper/ops.h"

namespace viper::ops {

Status sampling_forward(const Tensor& logits, i32 top_k, f32 top_p, f32 temperature,
                        u32 seed, i32& out_token) {
    (void)logits;
    (void)top_k;
    (void)top_p;
    (void)temperature;
    (void)seed;
    out_token = -1;
    return Status::Unimplemented(
        "sampling_forward: kernel lands in kernels/ops/sampling_kernel.cu (M4)");
}

}  // namespace viper::ops