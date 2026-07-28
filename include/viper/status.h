// Viper status codes + Status value type.
//
// All public APIs return Status; consumers check Status::code() and surface
// the message. Status::Ok() is the canonical success value.
#pragma once

#include <string>
#include <string_view>

namespace viper {

enum class StatusCode : int {
    OK = 0,
    UNIMPLEMENTED = 1,
    INVALID_ARGUMENT = 2,
    CUDA_ERROR = 3,
    SHAPE_MISMATCH = 4,
    OUT_OF_MEMORY = 5,
    IO_ERROR = 6,
    INTERNAL = 99,
};

class Status {
  public:
    Status() = default;
    explicit Status(StatusCode code, std::string message = "")
        : code_(code), message_(std::move(message)) {}

    StatusCode code() const noexcept { return code_; }
    bool ok() const noexcept { return code_ == StatusCode::OK; }
    explicit operator bool() const noexcept { return ok(); }
    std::string_view message() const noexcept { return message_; }

    static Status Ok() { return Status(StatusCode::OK); }
    static Status Unimplemented(std::string_view what = "") {
        return Status(StatusCode::UNIMPLEMENTED, std::string(what));
    }

  private:
    StatusCode code_ = StatusCode::OK;
    std::string message_;
};

}  // namespace viper