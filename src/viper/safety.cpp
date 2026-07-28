// Resource / safety guards. Concrete implementations.
#include "viper/safety.h"

#include <array>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <sstream>
#include <stdexcept>

#include <cuda_runtime.h>
#ifdef _WIN32
#include <windows.h>
#endif

namespace viper::safety {

Status device_memory(DeviceMem& out) {
    u64 free_b = 0, total_b = 0;
    auto err = cudaMemGetInfo(&free_b, &total_b);
    if (err != cudaSuccess) {
        return Status(StatusCode::CUDA_ERROR, cudaGetErrorString(err));
    }
    out.free_bytes = free_b;
    out.total_bytes = total_b;
    return Status::Ok();
}

Status host_memory(HostMem& out) {
#ifdef _WIN32
    MEMORYSTATUSEX ms{};
    ms.dwLength = sizeof(ms);
    if (!GlobalMemoryStatusEx(&ms)) {
        return Status(StatusCode::INTERNAL, "GlobalMemoryStatusEx failed");
    }
    out.free_bytes = static_cast<u64>(ms.ullAvailPhys);
    out.total_bytes = static_cast<u64>(ms.ullTotalPhys);
    return Status::Ok();
#else
    return Status(StatusCode::UNIMPLEMENTED, "host_memory not implemented on this OS");
#endif
}

Status guard_device_alloc(usize nbytes) {
    DeviceMem m;
    if (auto s = device_memory(m); !s.ok()) return s;
    if (m.free_bytes < kMinDeviceFreeBytes) {
        std::ostringstream os;
        os << "free VRAM " << (m.free_bytes / (1 << 20)) << " MiB below floor "
           << (kMinDeviceFreeBytes / (1 << 20)) << " MiB";
        return Status(StatusCode::OUT_OF_MEMORY, os.str());
    }
    if (m.free_bytes - nbytes < kMinDeviceFreeBytes) {
        std::ostringstream os;
        os << "alloc " << (nbytes / (1 << 20)) << " MiB would leave "
           << ((m.free_bytes - nbytes) / (1 << 20))
           << " MiB free (floor " << (kMinDeviceFreeBytes / (1 << 20)) << " MiB)";
        return Status(StatusCode::OUT_OF_MEMORY, os.str());
    }
    return Status::Ok();
}

Status guard_host_alloc(usize nbytes) {
    HostMem m;
    if (auto s = host_memory(m); !s.ok()) return s;
    if (m.free_bytes < kMinHostFreeBytes + nbytes) {
        std::ostringstream os;
        os << "host alloc " << (nbytes / (1 << 20)) << " MiB exceeds headroom "
           << "(free " << (m.free_bytes / (1 << 30)) << " GiB, floor "
           << (kMinHostFreeBytes / (1 << 30)) << " GiB)";
        return Status(StatusCode::OUT_OF_MEMORY, os.str());
    }
    return Status::Ok();
}

namespace {

// Run nvidia-smi and read GPU 0 temperature. Avoids linking nvml.lib.
bool read_nvidia_smi_temp(i32& out_c) {
#ifdef _WIN32
    std::array<char, 128> cmd{};
    std::snprintf(cmd.data(), cmd.size(),
                  "nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits");
    FILE* pipe = _popen(cmd.data(), "r");
    if (!pipe) return false;
    char line[64] = {};
    char* got = std::fgets(line, sizeof(line), pipe);
    int rc = _pclose(pipe);
    if (!got || rc != 0) return false;
    char* end = nullptr;
    long v = std::strtol(line, &end, 10);
    if (end == line) return false;
    out_c = static_cast<i32>(v);
    return true;
#else
    (void)out_c;
    return false;
#endif
}

}  // namespace

Status nvml_temperature(i32& temp_c, TempVerdict& verdict) {
    temp_c = -1;
    verdict = TempVerdict::Ok;
    if (!read_nvidia_smi_temp(temp_c)) {
        // NVML not available — soft-skip rather than fail the run.
        return Status::Ok();
    }
    if (temp_c >= kNvmlAbortTempC) {
        verdict = TempVerdict::Abort;
        std::ostringstream os;
        os << "GPU temperature " << temp_c << "C >= abort " << kNvmlAbortTempC << "C";
        return Status(StatusCode::INTERNAL, os.str());
    }
    if (temp_c >= kNvmlWarnTempC) {
        verdict = TempVerdict::Warn;
        VIPER_LOG(Warn, "GPU temperature above warn threshold");
    }
    return Status::Ok();
}

Status classify_cuda(cudaError_t err) {
    if (err == cudaSuccess) return Status::Ok();
    switch (err) {
        case cudaErrorInvalidValue:
        case cudaErrorInvalidDevicePointer:
        case cudaErrorInvalidMemcpyDirection:
            return Status(StatusCode::INVALID_ARGUMENT, cudaGetErrorString(err));
        case cudaErrorMemoryAllocation:
        case cudaErrorCudartUnloading:
            return Status(StatusCode::OUT_OF_MEMORY, cudaGetErrorString(err));
        case cudaErrorNotYetImplemented:
            return Status(StatusCode::UNIMPLEMENTED, cudaGetErrorString(err));
        case cudaErrorLaunchTimeout:
            return Status(StatusCode::INTERNAL,
                          std::string("launch timeout: ") + cudaGetErrorString(err));
        default:
            return Status(StatusCode::CUDA_ERROR, cudaGetErrorString(err));
    }
}

std::string_view status_name(StatusCode code) noexcept {
    switch (code) {
        case StatusCode::OK:               return "OK";
        case StatusCode::UNIMPLEMENTED:    return "UNIMPLEMENTED";
        case StatusCode::INVALID_ARGUMENT: return "INVALID_ARGUMENT";
        case StatusCode::CUDA_ERROR:       return "CUDA_ERROR";
        case StatusCode::SHAPE_MISMATCH:   return "SHAPE_MISMATCH";
        case StatusCode::OUT_OF_MEMORY:    return "OUT_OF_MEMORY";
        case StatusCode::IO_ERROR:         return "IO_ERROR";
        case StatusCode::INTERNAL:         return "INTERNAL";
    }
    return "UNKNOWN";
}

}  // namespace viper::safety