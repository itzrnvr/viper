// Embedding gather — stub. Real BF16 implementation lives in
// kernels/ops/embedding_kernel.cu (main thread owns).
#include "viper/ops.h"

namespace viper::ops {

Status embedding_forward(const Tensor& embedding_table,
                         const Tensor& token_ids, Tensor& y) {
    (void)embedding_table;
    (void)token_ids;
    (void)y;
    return Status::Unimplemented(
        "embedding_forward: kernel lands in kernels/ops/embedding_kernel.cu (M2)");
}

}  // namespace viper::ops