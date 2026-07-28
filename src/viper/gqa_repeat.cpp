// GQA repeat-interleave — stub. Real impl lives in kernels/ops/gqa_repeat_kernel.cu.
#include "viper/ops.h"

namespace viper::ops {

Status gqa_repeat_forward(const Tensor& kv, Tensor& out) {
    (void)kv;
    (void)out;
    return Status::Unimplemented(
        "gqa_repeat_forward: kernel lands in kernels/ops/gqa_repeat_kernel.cu (M2)");
}

}  // namespace viper::ops