// Viper tensor — multi-dimensional array view over a raw device buffer.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "viper/common.h"

namespace viper {

enum class DType : int {
    FP32 = 0,
    BF16 = 1,
    FP16 = 2,
    INT8 = 3,
    UINT8 = 4,
    INT32 = 5,
    UINT32 = 6,
    Q4_G64 = 100,
    Q5_G64 = 101,
    Q6_G64 = 102,
    W8_G32 = 103,
};

const char* dtype_name(DType d) noexcept;
usize dtype_sizeof(DType d) noexcept;

class Shape {
public:
    using Dim = i64;
    static constexpr usize kMaxDims = 8;

    Shape() = default;
    Shape(std::initializer_list<Dim> dims) {
        VIPER_ASSERT(dims.size() <= kMaxDims);
        size_t i = 0;
        for (auto d : dims) dims_[i++] = d;
        rank_ = i;
    }
    // Pointer + size constructor (replaces std::span for MSVC 14.29 compat).
    Shape(const Dim* dims, usize n) {
        VIPER_ASSERT(n <= kMaxDims);
        for (size_t i = 0; i < n; ++i) dims_[i] = dims[i];
        rank_ = n;
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

    std::array<Dim, kMaxDims> strides() const noexcept {
        std::array<Dim, kMaxDims> s{};
        Dim acc = 1;
        for (i64 i = (i64)rank_ - 1; i >= 0; --i) {
            s[(size_t)i] = acc;
            acc *= dims_[(size_t)i];
        }
        return s;
    }

private:
    std::array<Dim, kMaxDims> dims_{};
    usize rank_ = 0;
};

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
        for (i64 i = (i64)shape_.rank() - 1; i >= 0; --i) {
            if (s[(size_t)i] != expected) return false;
            expected *= shape_[(size_t)i];
        }
        return true;
    }

    template <typename T>
    T* data_as() const noexcept {
        return reinterpret_cast<T*>(data_);
    }

private:
    void* data_ = nullptr;
    DType dtype_ = DType::FP32;
    Shape shape_;
};

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
