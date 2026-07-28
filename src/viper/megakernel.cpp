// Megakernel launcher host stub. Real persistent grid-sync kernel impl lands
// in M3 (main thread owns kernels/ops/megakernel_launch.cu). This stub
// matches the public interface in include/viper/megakernel.h so callers
// can link against the engine before the kernel lands.
#include "viper/megakernel.h"

namespace viper::megakernel {

Status launch_persistent_forward(NanbeigeModel& model,
                                 const std::vector<i32>& token_ids,
                                 std::vector<i32>& out_tokens,
                                 const LaunchConfig& cfg) {
    (void)model;
    (void)token_ids;
    (void)cfg;
    out_tokens.clear();
    return Status::Unimplemented(
        "launch_persistent_forward: persistent megakernel impl lands in M3 — "
        "grid-sync via cooperative_groups, no CUDA Graphs");
}

Status grid_sync() {
    auto err = cudaDeviceSynchronize();
    if (err != cudaSuccess) return ::viper::safety::classify_cuda(err);
    return Status::Ok();
}

}  // namespace viper::megakernel