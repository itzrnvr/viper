// CUDA error checking — VIPER_CHECK_CUDA macro + classify_cuda helper.
//
// Every CUDA call in the engine is funneled through one of these so we can
// catch cudaError* and translate to a Status. Per-op timeout is enforced
// elsewhere (safety.h + megakernel.h).
#pragma once

#include <cuda_runtime.h>

#include "viper/status.h"

namespace viper::safety {
Status classify_cuda(cudaError_t err);
}

#define VIPER_CHECK_CUDA(expr)                                                  \
    do {                                                                        \
        cudaError_t _err = (expr);                                              \
        if (_err != cudaSuccess) {                                              \
            return ::viper::Status(::viper::StatusCode::CUDA_ERROR,             \
                                    cudaGetErrorString(_err));                   \
        }                                                                       \
    } while (0)