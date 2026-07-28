// Megakernel launcher — persistent grid-sync kernel replaces CUDA Graphs.
//
// Design contract (parent directive):
//   - NO cudaGraph / cudaGraphLaunch / cudaStreamBeginCapture anywhere in the
//     engine. CUDA Graphs destabilize the GPU and crash the PC.
//   - One persistent kernel holds the activation frontier in shared memory,
//     streams weight tiles via cp.async, and grid-syncs between layers via
//     cooperative groups.
//   - This header reserves the entry points; concrete megakernel impl is
//     landed in M3. Skeleton phase wires a stub that launches individual ops
//     with grid-sync fences between them.
#pragma once

#include "viper/common.h"
#include "viper/model.h"
#include "viper/tensor.h"

namespace viper::megakernel {

struct LaunchConfig {
    i32 grid_x = 0;            // 0 = auto from seq_len
    i32 max_iters = 1;         // safety cap: forward iterations
    i32 max_tokens = 32;       // safety cap: tokens generated per call
    i32 timeout_ms = 60'000;   // per-op safety timeout
    bool persistent = true;   // false = single-shot launch (debug only)
};

// Launch the persistent 44-step forward pass. Skeleton returns UNIMPLEMENTED
// with a clear message; future phases fill in the persistent kernel.
Status launch_persistent_forward(NanbeigeModel& model,
                                 const std::vector<i32>& token_ids,
                                 std::vector<i32>& out_tokens,
                                 const LaunchConfig& cfg);

// Synchronize grid (cooperative_groups::grid_group::sync() in the kernel).
// Called from host for testing — wraps cudaDeviceSynchronize + a stream fence.
Status grid_sync();

}  // namespace viper::megakernel