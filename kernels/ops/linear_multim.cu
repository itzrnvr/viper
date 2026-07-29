/*
 * viper Multi-M Q4 GEMV production kernel.
 *
 * Each block processes ALL M tokens for its output channels.
 * Weights read ONCE from DRAM. Activations from L2 cache.
 *
 * Without this: forward_batch(M=5) reads weights 5x → 5x slower.
 * With this: forward_batch(M=5) reads weights 1x → ~1.3x slower per token.
 */
#include "linear_multim.h"
#include <cuda_runtime.h>

namespace viper {
namespace ops {

__global__ void linear_q4_multim_kernel(
    const uint8_t* __restrict__ w_packed,
    const __nv_bfloat16* __restrict__ w_scales,
    const __nv_bfloat16* __restrict__ x,       // [M, K]
    __nv_bfloat16* __restrict__ y,              // [M, N]
    const __nv_bfloat16* __restrict__ residual, // [M, N] or null
    int M, int N, int K) {
    const int warp_id = threadIdx.x >> 5;
    const int lane_id = threadIdx.x & 31;
    const int n = blockIdx.x * (blockDim.x >> 5) + warp_id;
    if (n >= N) return;

    const uint8_t* w_row = w_packed + (size_t)n * (K / 2);
    const __nv_bfloat16* s_row = w_scales + (size_t)n * (K / 64);

    float acc[8];  // max M=8
    #pragma unroll
    for (int m = 0; m < 8; ++m) acc[m] = 0.0f;

    const int n_bytes = K / 2;
    const int vec_end = n_bytes - (n_bytes % 128);

    for (int base = 0; base < vec_end; base += 128) {
        int byte_off = base + lane_id * 4;
        // Read weight ONCE — L1 caches across M loop
        uint32_t p4 = *reinterpret_cast<const uint32_t*>(w_row + byte_off);

        int w0 = (p4 & 0xF) - 8, w1 = ((p4>>4)&0xF) - 8;
        int w2 = ((p4>>8)&0xF) - 8, w3 = ((p4>>12)&0xF) - 8;
        int w4 = ((p4>>16)&0xF) - 8, w5 = ((p4>>20)&0xF) - 8;
        int w6 = ((p4>>24)&0xF) - 8, w7 = ((p4>>28)&0xF) - 8;

        float sc = __bfloat162float(s_row[byte_off / 32]);
        int xk = byte_off * 2;

        // Compute for ALL M tokens (activations from L2)
        #pragma unroll
        for (int m = 0; m < 8; ++m) {
            if (m >= M) break;
            const __nv_bfloat16* xm = x + (size_t)m * K;
            acc[m] += sc * (
                (float)w0 * __bfloat162float(xm[xk]) +
                (float)w1 * __bfloat162float(xm[xk+1]) +
                (float)w2 * __bfloat162float(xm[xk+2]) +
                (float)w3 * __bfloat162float(xm[xk+3]) +
                (float)w4 * __bfloat162float(xm[xk+4]) +
                (float)w5 * __bfloat162float(xm[xk+5]) +
                (float)w6 * __bfloat162float(xm[xk+6]) +
                (float)w7 * __bfloat162float(xm[xk+7]));
        }
    }

    // Warp reduce per token
    #pragma unroll
    for (int m = 0; m < 8; ++m) {
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

cudaError_t linear_q4_multim(
    const uint8_t* w, const __nv_bfloat16* s,
    const __nv_bfloat16* x, __nv_bfloat16* y,
    int M, int N, int K, cudaStream_t stream) {
    if (!w || !s || !x || !y || M <= 0 || N <= 0 || K <= 0 || M > 8)
        return cudaErrorInvalidValue;
    linear_q4_multim_kernel<<<(N+7)/8, 256, 0, stream>>>(
        w, s, x, y, nullptr, M, N, K);
    return cudaGetLastError();
}

cudaError_t linear_q4_multim_residual(
    const uint8_t* w, const __nv_bfloat16* s,
    const __nv_bfloat16* x, __nv_bfloat16* y,
    const __nv_bfloat16* residual,
    int M, int N, int K, cudaStream_t stream) {
    if (!w || !s || !x || !y || !residual || M <= 0 || N <= 0 || K <= 0 || M > 8)
        return cudaErrorInvalidValue;
    linear_q4_multim_kernel<<<(N+7)/8, 256, 0, stream>>>(
        w, s, x, y, residual, M, N, K);
    return cudaGetLastError();
}

}  // namespace ops
}  // namespace viper
