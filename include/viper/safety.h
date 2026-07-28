// Resource / safety guards. Every public API in the engine funnels through
// here before touching the GPU.
//
// Policy (parent directive):
//   - cudaMemGetInfo before every major allocation; refuse if free VRAM < 1 GB
//     (return Status::OUT_OF_MEMORY)
//   - Host alloc guards: refuse if free host RAM < 4 GB
//   - Per-op timeout: 60s default, configurable
//   - Optional NVML temp check (warn 80C, abort 87C)
//   - All cudaError* are caught and classified to one of OK/OOM/TIMEOUT/
//     INVALID_ARGUMENT/UNIMPLEMENTED/INTERNAL
#pragma once

#include <cstdint>

#include "viper/common.h"

namespace viper::safety {

struct DeviceMem {
    u64 free_bytes = 0;
    u64 total_bytes = 0;
};

struct HostMem {
    u64 free_bytes = 0;
    u64 total_bytes = 0;
};

// Free VRAM headroom floor — engine refuses any allocation when free VRAM is
// below this. Keeps the OS and other processes from getting wedged.
constexpr u64 kMinDeviceFreeBytes = 1ULL << 30;  // 1 GiB

// Free host RAM floor — keeps Windows from paging to death.
constexpr u64 kMinHostFreeBytes = 4ULL << 30;    // 4 GiB

// NVML thresholds (Celsius).
constexpr i32 kNvmlWarnTempC = 80;
constexpr i32 kNvmlAbortTempC = 87;

// Query current device 0 free/total VRAM.
Status device_memory(DeviceMem& out);

// Query current host free/total RAM via GlobalMemoryStatusEx.
Status host_memory(HostMem& out);

// Pre-allocation guard. Returns OOM if either:
//   - free VRAM < requested nbytes + kMinDeviceFreeBytes, OR
//   - free VRAM < kMinDeviceFreeBytes outright.
Status guard_device_alloc(usize nbytes);

// Pre-allocation host guard. Returns OOM if free RAM drops below
// kMinHostFreeBytes after the hypothetical allocation.
Status guard_host_alloc(usize nbytes);

// Optional NVML temperature check. Reads GPU 0 temperature via nvidia-smi
// (no link-time dependency on nvml.lib). temp_c is set even when the policy
// allows it.
enum class TempVerdict { Ok, Warn, Abort };
Status nvml_temperature(i32& temp_c, TempVerdict& verdict);

// Map a cudaError_t to our Status codes. Returns StatusCode::OK when the
// argument is cudaSuccess.
Status classify_cuda(cudaError_t err);

// Translate our Status into a printable line. Useful for log_message.
std::string_view status_name(StatusCode code) noexcept;

}  // namespace viper::safety