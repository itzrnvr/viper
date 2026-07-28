// Viper common types and error helpers — umbrella header.
//
// Scalar types, logging, the VIPER_CHECK / VIPER_ASSERT macros. Status and
// CUDA-error checking live in status.h and cuda_check.h respectively.
#pragma once

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <string_view>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "viper/cuda_check.h"
#include "viper/status.h"

namespace viper {

// ---------------------------------------------------------------------------
// Scalar types
// ---------------------------------------------------------------------------
using i8 = std::int8_t;
using i16 = std::int16_t;
using i32 = std::int32_t;
using i64 = std::int64_t;
using u8 = std::uint8_t;
using u16 = std::uint16_t;
using u32 = std::uint32_t;
using u64 = std::uint64_t;
using usize = std::size_t;
using f32 = float;
using f64 = double;

// GPU half-precision aliases (NVIDIA __nv_bfloat16 / __half).
using bf16 = __nv_bfloat16;
using fp16 = __half;

// ---------------------------------------------------------------------------
// Logging
// ---------------------------------------------------------------------------
enum class LogLevel { Debug = 0, Info = 1, Warn = 2, Error = 3 };

void log_message(LogLevel level, std::string_view msg);

#define VIPER_LOG(level, msg) ::viper::log_message(::viper::LogLevel::level, (msg))

#define VIPER_CHECK(cond, msg)                                                  \
    do {                                                                        \
        if (!(cond)) {                                                          \
            return ::viper::Status(::viper::StatusCode::INVALID_ARGUMENT,       \
                                    std::string("check failed: ") + (msg));     \
        }                                                                       \
    } while (0)

#if defined(VIPER_ENABLE_ASSERTS) && VIPER_ENABLE_ASSERTS
#define VIPER_ASSERT(cond)                                                      \
    do {                                                                        \
        if (!(cond)) {                                                          \
            std::fprintf(stderr, "ASSERT failed at %s:%d: %s\n", __FILE__,      \
                          __LINE__, #cond);                                     \
            std::abort();                                                       \
        }                                                                       \
    } while (0)
#else
#define VIPER_ASSERT(cond) ((void)0)
#endif

}  // namespace viper