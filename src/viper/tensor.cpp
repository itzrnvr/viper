// DeviceBuffer + dtype helpers.
#include "viper/tensor.h"

#include "viper/safety.h"

namespace viper {

std::string_view dtype_name(DType d) noexcept {
    switch (d) {
        case DType::FP32:   return "FP32";
        case DType::BF16:   return "BF16";
        case DType::FP16:   return "FP16";
        case DType::INT8:   return "INT8";
        case DType::UINT8:  return "UINT8";
        case DType::INT32:  return "INT32";
        case DType::UINT32: return "UINT32";
        case DType::Q4_G64: return "Q4_G64";
        case DType::Q5_G64: return "Q5_G64";
        case DType::Q6_G64: return "Q6_G64";
        case DType::W8_G32: return "W8_G32";
    }
    return "UNKNOWN";
}

usize dtype_sizeof(DType d) noexcept {
    switch (d) {
        case DType::FP32:   return 4;
        case DType::BF16:   return 2;
        case DType::FP16:   return 2;
        case DType::INT8:   return 1;
        case DType::UINT8:  return 1;
        case DType::INT32:  return 4;
        case DType::UINT32: return 4;
        // Quantized formats: pack/header only; per-element size varies.
        case DType::Q4_G64: return 1;  // 4-bit packed
        case DType::Q5_G64: return 1;
        case DType::Q6_G64: return 1;
        case DType::W8_G32: return 1;
    }
    return 0;
}

Status DeviceBuffer::alloc(usize nbytes) {
    free();
    if (nbytes == 0) {
        nbytes_ = 0;
        ptr_ = nullptr;
        return Status::Ok();
    }
    if (auto s = safety::guard_device_alloc(nbytes); !s.ok()) return s;
    auto err = cudaMalloc(&ptr_, nbytes);
    if (err != cudaSuccess) {
        ptr_ = nullptr;
        nbytes_ = 0;
        return Status(StatusCode::CUDA_ERROR, cudaGetErrorString(err));
    }
    nbytes_ = nbytes;
    return Status::Ok();
}

void DeviceBuffer::free() noexcept {
    if (ptr_) {
        cudaFree(ptr_);
        ptr_ = nullptr;
    }
    nbytes_ = 0;
}

Status DeviceBuffer::copy_from_host(const void* src, usize nbytes) {
    if (nbytes > nbytes_) {
        return Status(StatusCode::INVALID_ARGUMENT, "copy_from_host: nbytes > buffer");
    }
    auto err = cudaMemcpy(ptr_, src, nbytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
        return Status(StatusCode::CUDA_ERROR, cudaGetErrorString(err));
    return Status::Ok();
}

Status DeviceBuffer::copy_to_host(void* dst, usize nbytes) const {
    if (nbytes > nbytes_) {
        return Status(StatusCode::INVALID_ARGUMENT, "copy_to_host: nbytes > buffer");
    }
    auto err = cudaMemcpy(dst, ptr_, nbytes, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
        return Status(StatusCode::CUDA_ERROR, cudaGetErrorString(err));
    return Status::Ok();
}

}  // namespace viper