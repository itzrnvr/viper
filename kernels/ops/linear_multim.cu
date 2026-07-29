/*
 * viper Multi-M Q4 GEMV production kernel with DP4A.
 *
 * Each block processes ALL M tokens for its output channels.
 * Weights read ONCE from DRAM. Uses __dp4a for 4 multiply-adds per instruction.
 *
 * Key optimization: Q4 weights are unpacked to INT8, activations are
 * pre-quantized to INT8 (Q8 format), then __dp4a computes the dot product.
 * This reduces instruction count from ~24 to ~10 per element,
 * making the kernel memory-bound instead of compute-bound for M>1.
 */
#include "linear_multim.h"
#include <cuda_runtime.h>
#include <cuda_bf16.h>

namespace viper {
namespace ops {

// Quantize BF16 activations to INT8 with per-group-of-64 scales.
// Input: x[M, K] BF16. Output: xq[M, K] INT8 + xs[M, K/64] float scales.
__global__ void quantize_activations_kernel(
    const __nv_bfloat16* __restrict__ x,  // [M, K]
    int8_t* __restrict__ xq,               // [M, K]
    float* __restrict__ xs,                // [M, K/64] scales
    int M, int K) {
    const int m = blockIdx.x;  // one block per token
    if (m >= M) return;
    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;
    const int ngroups = K / 64;

    // Quantize each group of 64 independently
    for (int g = 0; g < ngroups; ++g) {
        const int base = g * 64;
        // Find max in this group (all threads cooperate)
        float local_max = 0.f;
        for (int i = tid; i < 64; i += nthreads) {
            local_max = fmaxf(local_max, fabsf(__bfloat162float(x[m * K + base + i])));
        }
        for (int off = 16; off > 0; off >>= 1)
            local_max = fmaxf(local_max, __shfl_xor_sync(0xffffffff, local_max, off));
        __shared__ float warp_max[8];
        const int wid = tid / 32, lid = tid % 32;
        if (lid == 0) warp_max[wid] = local_max;
        __syncthreads();
        float gmax = warp_max[0];
        #pragma unroll
        for (int i = 1; i < 8; ++i) gmax = fmaxf(gmax, warp_max[i]);
        __syncthreads();

        const float scale = gmax / 127.0f;
        if (tid == 0) xs[m * ngroups + g] = scale;
        __syncthreads();

        for (int i = tid; i < 64; i += nthreads) {
            float v = __bfloat162float(x[m * K + base + i]);
            xq[m * K + base + i] = (int8_t)__float2int_rn(v / scale);
        }
        __syncthreads();
    }
}

// DP4A multi-M kernel: Q4 weights × Q8 activations → FP32 output
__global__ void linear_q4_dp4a_multim_kernel(
    const uint8_t* __restrict__ w_packed,   // [N, K/2] Q4 packed
    const __nv_bfloat16* __restrict__ w_scales,  // [N, K/64] scales
    const int8_t* __restrict__ xq,           // [M, K] INT8 quantized
    const float* __restrict__ xs,            // [M, K/64] activation scales
    __nv_bfloat16* __restrict__ y,           // [M, N] output
    const __nv_bfloat16* __restrict__ residual, // [M, N] or null
    int M, int N, int K) {
    const int warp_id = threadIdx.x >> 5;
    const int lane_id = threadIdx.x & 31;
    const int n = blockIdx.x * (blockDim.x >> 5) + warp_id;
    if (n >= N) return;

    const uint8_t* w_row = w_packed + (size_t)n * (K / 2);
    const __nv_bfloat16* s_row = w_scales + (size_t)n * (K / 64);

    float acc[16];
    #pragma unroll
    for (int m = 0; m < 16; ++m) acc[m] = 0.0f;

    const int n_bytes = K / 2;
    const int vec_end = n_bytes - (n_bytes % 128);

    for (int base = 0; base < vec_end; base += 128) {
        int byte_off = base + lane_id * 4;
        uint32_t p4 = *reinterpret_cast<const uint32_t*>(w_row + byte_off);

        // Unpack Q4 → INT8 (two's complement: subtract 8)
        int8_t w0 = (p4 & 0xF) - 8, w1 = ((p4>>4)&0xF) - 8;
        int8_t w2 = ((p4>>8)&0xF) - 8, w3 = ((p4>>12)&0xF) - 8;
        int8_t w4 = ((p4>>16)&0xF) - 8, w5 = ((p4>>20)&0xF) - 8;
        int8_t w6 = ((p4>>24)&0xF) - 8, w7 = ((p4>>28)&0xF) - 8;

        float sc = __bfloat162float(s_row[byte_off / 32]);
        int xk = byte_off * 2;
        int group = byte_off / 32;

        // Pack 4 weights into one int32 for DP4A
        int w_packed_0 = (w0 & 0xFF) | ((w1 & 0xFF) << 8) | ((w2 & 0xFF) << 16) | ((w3 & 0xFF) << 24);
        int w_packed_1 = (w4 & 0xFF) | ((w5 & 0xFF) << 8) | ((w6 & 0xFF) << 16) | ((w7 & 0xFF) << 24);

        #pragma unroll
        for (int m = 0; m < 16; ++m) {
            if (m >= M) break;
            int a0 = *reinterpret_cast<const int*>(xq + m * K + xk);
            int a1 = *reinterpret_cast<const int*>(xq + m * K + xk + 4);
            float as = xs[m * (K / 64) + group];
            acc[m] += sc * as * (float)(
                __dp4a(w_packed_0, a0, 0) + __dp4a(w_packed_1, a1, 0));
        }
    }

    // Warp reduce per token
    #pragma unroll
    for (int m = 0; m < 16; ++m) {
        if (m >= M) break;
        float a = acc[m];
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            a += __shfl_xor_sync(0xffffffff, a, off);
        if (lane_id == 0) {
            if (residual)
                a += __bfloat162float(residual[m * N + n]);
            y[m * N + n] = __float2bfloat16(a);
        }
    }
}

// Buffers for quantized activations (allocated once, reused)
static int8_t* d_xq_ = nullptr;
static float* d_xs_ = nullptr;
static int xq_capacity_ = 0;

static bool ensure_xq(int M, int K) {
    int needed = M * K;
    int scale_needed = M * (K / 64);
    if (needed <= xq_capacity_) return true;
    if (d_xq_) cudaFree(d_xq_);
    if (d_xs_) cudaFree(d_xs_);
    xq_capacity_ = needed + 1024;
    return cudaMalloc(&d_xq_, xq_capacity_) == cudaSuccess &&
           cudaMalloc(&d_xs_, scale_needed * sizeof(float) + 64) == cudaSuccess;
}

cudaError_t linear_q4_multim(
    const uint8_t* w, const __nv_bfloat16* s,
    const __nv_bfloat16* x, __nv_bfloat16* y,
    int M, int N, int K, cudaStream_t stream) {
    if (!w || !s || !x || !y || M <= 0 || N <= 0 || K <= 0 || M > 16)
        return cudaErrorInvalidValue;
    if (!ensure_xq(M, K)) return cudaErrorMemoryAllocation;

    // Quantize activations (one block per token for consistent scale)
    quantize_activations_kernel<<<M, 256, 0, stream>>>(x, d_xq_, d_xs_, M, K);

    // DP4A GEMV
    linear_q4_dp4a_multim_kernel<<<(N+7)/8, 256, 0, stream>>>(
        w, s, d_xq_, d_xs_, y, nullptr, M, N, K);
    return cudaGetLastError();
}

cudaError_t linear_q4_multim_residual(
    const uint8_t* w, const __nv_bfloat16* s,
    const __nv_bfloat16* x, __nv_bfloat16* y,
    const __nv_bfloat16* residual,
    int M, int N, int K, cudaStream_t stream) {
    if (!w || !s || !x || !y || !residual || M <= 0 || N <= 0 || K <= 0 || M > 16)
        return cudaErrorInvalidValue;
    if (!ensure_xq(M, K)) return cudaErrorMemoryAllocation;

    quantize_activations_kernel<<<M, 256, 0, stream>>>(x, d_xq_, d_xs_, M, K);

    linear_q4_dp4a_multim_kernel<<<(N+7)/8, 256, 0, stream>>>(
        w, s, d_xq_, d_xs_, y, residual, M, N, K);
    return cudaGetLastError();
}

// Public quantize function
cudaError_t quantize_to_q8(
    const __nv_bfloat16* x, int8_t* xq, float* xs,
    int M, int K, cudaStream_t stream) {
    if (!x || !xq || !xs || M <= 0 || K <= 0) return cudaErrorInvalidValue;
    quantize_activations_kernel<<<M, 256, 0, stream>>>(x, xq, xs, M, K);
    return cudaGetLastError();
}

// DP4A GEMV with pre-quantized activations (no quantize inside, no SMEM, no sync)
cudaError_t linear_q4_dp4a_prequantized(
    const uint8_t* w, const __nv_bfloat16* s,
    const int8_t* xq, const float* xs,
    __nv_bfloat16* y,
    int M, int N, int K, cudaStream_t stream) {
    if (!w || !s || !xq || !xs || !y || M <= 0 || N <= 0 || K <= 0 || M > 16)
        return cudaErrorInvalidValue;
    linear_q4_dp4a_multim_kernel<<<(N+7)/8, 256, 0, stream>>>(
        w, s, xq, xs, y, nullptr, M, N, K);
    return cudaGetLastError();
}

cudaError_t linear_q4_dp4a_prequantized_residual(
    const uint8_t* w, const __nv_bfloat16* s,
    const int8_t* xq, const float* xs,
    __nv_bfloat16* y, const __nv_bfloat16* residual,
    int M, int N, int K, cudaStream_t stream) {
    if (!w || !s || !xq || !xs || !y || !residual || M <= 0 || N <= 0 || K <= 0 || M > 16)
        return cudaErrorInvalidValue;
    linear_q4_dp4a_multim_kernel<<<(N+7)/8, 256, 0, stream>>>(
        w, s, xq, xs, y, residual, M, N, K);
    return cudaGetLastError();
}
}  // namespace ops
}  // namespace viper
