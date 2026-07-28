// Viper tensor — multi-dimensional array view over a raw device buffer.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>
#include <vector>

#include "viper/common.h"

namespace viper {

// ---------------------------------------------------------------------------
// Data type enum (kept tiny: only what we need for v1).
// ---------------------------------------------------------------------------
enum class DType : int {
    FP32 = 0,
    BF16 = 1,
    FP16 = 2,
    INT8 = 3,
    UINT8 = 4,
    INT32 = 5,
    UINT32 = 6,
    Q4_G64 = 100,  // quant format registry ids
    Q5_G64 = 101,
    Q6_G64 = 102,
    W8_G32 = 103,
};

std::string_view dtype_name(DType d) noexcept;
usize dtype_sizeof(DType d) noexcept;

// ---------------------------------------------------------------------------
// Shape
// ---------------------------------------------------------------------------
class Shape {
  public:
    using Dim = i64;

    static constexpr usize kMaxDims = 8;

    Shape() = default;
    Shape(std::initializer_list<Dim> dims) {
        VIPER_ASSERT(dims.size() <= kMaxDims);
        for (auto d : dims) dims_[rank_++] = d;
    }
    explicit Shape(std::span<const Dim> dims) {
        VIPER_ASSERT(dims.size() <= kMaxDims);
        for (auto d : dims) dims_[rank_++] = d;
    }

    Dim operator[](usize i) const noexcept { return dims_[i]; }
    Dim& operator[](usize i) noexcept { return dims_[i]; }

    usize rank() const noexcept { return rank_; }
    Dim dim(usize i) const noexcept { return dims_[i]; }
    i64 numel() const noexcept {
        i64 n = 1;
        for (usize i = 0; i < rank_; ++i) n *= dims_[i];
        return n;
    }

    bool operator==(const Shape& other) const noexcept {
        if (rank_ != other.rank_) return false;
        for (usize i = 0; i < rank_; ++i)
            if (dims_[i] != other.dims_[i]) return false;
        return true;
    }
    bool operator!=(const Shape& other) const noexcept { return !(*this == other); }

    // Row-major contiguous: stride[i] = product(dims[i+1:]).
    std::array<Dim, kMaxDims> strides() const noexcept {
        std::array<Dim, kMaxDims> s{};
        Dim acc = 1;
        for (i64 i = rank_ - 1; i >= 0; --i) {
            s[i] = acc;
            acc *= dims_[i];
        }
        return s;
    }

  private:
    std::array<Dim, kMaxDims> dims_{};
    usize rank_ = 0;
};

// ---------------------------------------------------------------------------
// Tensor — non-owning view over a device buffer.
// ---------------------------------------------------------------------------
class Tensor {
  public:
    Tensor() = default;
    Tensor(void* data, DType dtype, Shape shape)
        : data_(data), dtype_(dtype), shape_(std::move(shape)) {}

    void* data() const noexcept { return data_; }
    DType dtype() const noexcept { return dtype_; }
    const Shape& shape() const noexcept { return shape_; }
    i64 numel() const noexcept { return shape_.numel(); }
    usize nbytes() const noexcept {
        return static_cast<usize>(numel()) * dtype_sizeof(dtype_);
    }

    bool is_contiguous() const noexcept {
        auto s = shape_.strides();
        i64 expected = 1;
        for (i64 i = shape_.rank() - 1; i >= 0; --i) {
            if (s[i] != expected) return false;
            expected *= shape_[i];
        }
        return true;
    }

    // Convenience helpers — typed pointer cast.
    template <typename T>
    T* data_as() const noexcept {
        return reinterpret_cast<T*>(data_);
    }

  private:
    void* data_ = nullptr;
    DType dtype_ = DType::FP32;
    Shape shape_;
};

// ---------------------------------------------------------------------------
// DeviceBuffer — owning device memory wrapper.
// ---------------------------------------------------------------------------
class DeviceBuffer {
  public:
    DeviceBuffer() = default;
    explicit DeviceBuffer(usize nbytes) { alloc(nbytes); }
    ~DeviceBuffer() { free(); }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    DeviceBuffer(DeviceBuffer&& o) noexcept : ptr_(o.ptr_), nbytes_(o.nbytes_) {
        o.ptr_ = nullptr;
        o.nbytes_ = 0;
    }
    DeviceBuffer& operator=(DeviceBuffer&& o) noexcept {
        if (this != &o) {
            free();
            ptr_ = o.ptr_;
            nbytes_ = o.nbytes_;
            o.ptr_ = nullptr;
            o.nbytes_ = 0;
        }
        return *this;
    }

    Status alloc(usize nbytes);
    void free() noexcept;
    void* ptr() const noexcept { return ptr_; }
    usize nbytes() const noexcept { return nbytes_; }

    Status copy_from_host(const void* src, usize nbytes);
    Status copy_to_host(void* dst, usize nbytes) const;

  private:
    void* ptr_ = nullptr;
    usize nbytes_ = 0;
};

}  // namespace viper