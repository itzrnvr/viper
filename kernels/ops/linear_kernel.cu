/*
 * viper Linear Q4_G64 GEMV kernel — implementation.
 *
 * LAYOUT (our Q4_G64 format):
 *   For each row of N columns, weights packed in groups of 64.
 *   Per group: 64 x 4-bit weights (32 bytes) + 1 x FP16 scale (2 bytes).
 *   Per row: (K/64) groups * 34 bytes.
 *   Symmetric quantization: stored = weight + 8 (range 0..15).
 *
 * DECODE PATH (T=1, M=1): one block per (m, n_tile), threads cooperate
 *   over K. For T>1 (prefill), larger M tiles reduce per-token cost.
 *
 * For v1 we implement the decode path (M variable, single output row
 * per block at a time, BN=64 output channels per block).
 *
 * CORRECTNESS:
 *   - Dequant: bf16 = (stored - 8) * scale, per group of 64.
 *   - mma.sync NOT used for T=1 (memory-bound, no compute).
 *   - Pure FMA accumulation in FP32.
 */
#include "linear_kernel.h"
#include <cuda_runtime.h>

namespace viper {
namespace ops {

// Decode-path GEMV: M rows in parallel, one block per (m, n_tile).
// Block: 128 threads, BN=64 output channels per block.
// Each thread computes N_PER_THREAD = BN/128 = 0.5 outputs (need 2 threads per output).
// Use BN=128 instead, N_PER_THREAD = 1 output per thread.
template <int BN>
__global__ void linear_q4_g64_decode_kernel(
    const uint8_t* __restrict__ w_packed,    // [N, K/2]
    const __nv_bfloat16* __restrict__ w_scales, // [N, K/64]
    const __nv_bfloat16* __restrict__ x,     // [M, K]
    __nv_bfloat16* __restrict__ y,           // [M, N]
    int M, int N, int K) {
    const int m = blockIdx.y;
    const int n_tile = blockIdx.x;
    const int n_base = n_tile * BN;
    const int tid = threadIdx.x;

    if (m >= M || n_base >= N) return;

    // Each thread handles ONE output channel.
    const int n = n_base + tid;
    if (n >= N) return;

    // Pointers for this row.
    const uint8_t* w_row = w_packed + n * (K / 2);
    const __nv_bfloat16* s_row = w_scales + n * (K / 64);
    const __nv_bfloat16* x_row = x + m * K;

    // Accumulate.
    float acc = 0.0f;

    // Process K in chunks of 64 (one group at a time).
    // Each iteration: load 64 weights (32 bytes), 1 scale (2 bytes),
    // dequant, FMA with 64 x values.
    constexpr int GROUP = 64;
    const int n_groups = K / GROUP;
    for (int g = 0; g < n_groups; ++g) {
        const float scale = __bfloat162float(s_row[g]);
        // Load 64 4-bit weights from 32 bytes. Each byte holds two
        // weights: low nibble = even position, high nibble = odd.
        // We process 8 weights per byte in the inner loop.
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            uint8_t b0 = w_row[g * 32 + j * 4 + 0];
            uint8_t b1 = w_row[g * 32 + j * 4 + 1];
            uint8_t b2 = w_row[g * 32 + j * 4 + 2];
            uint8_t b3 = w_row[g * 32 + j * 4 + 3];
            // Unpack 8 weights (each byte = 2 weights).
            int w[8];
            w[0] = (b0 & 0x0F) - 8;
            w[1] = (b0 >> 4) - 8;
            w[2] = (b1 & 0x0F) - 8;
            w[3] = (b1 >> 4) - 8;
            w[4] = (b2 & 0x0F) - 8;
            w[5] = (b2 >> 4) - 8;
            w[6] = (b3 & 0x0F) - 8;
            w[7] = (b3 >> 4) - 8;
            int base = g * GROUP + j * 8;
            #pragma unroll
            for (int k = 0; k < 8; ++k) {
                float x_val = __bfloat162float(x_row[base + k]);
                acc += (float)w[k] * scale * x_val;
            }
        }
    }

    y[m * N + n] = __float2bfloat16(acc);
}

cudaError_t linear_q4_g64_bf16(
    const uint8_t* w_packed,
    const __nv_bfloat16* w_scales,
    const __nv_bfloat16* x,
    __nv_bfloat16* y,
    int M, int N, int K,
    cudaStream_t stream) {
    if (!w_packed || !w_scales || !x || !y || M <= 0 || N <= 0 || K <= 0) {
        return cudaErrorInvalidValue;
    }
    if (K % 64 != 0) {
        return cudaErrorInvalidValue;  // Q4_G64 requires K divisible by 64
    }
    if (N % 128 != 0) {
        return cudaErrorInvalidValue;  // decode kernel requires N divisible by 128
    }

    constexpr int BN = 128;
    dim3 grid(N / BN, M);
    dim3 block(BN);
    linear_q4_g64_decode_kernel<BN><<<grid, block, 0, stream>>>(
        w_packed, w_scales, x, y, M, N, K);
    return cudaGetLastError();
}

}  // namespace ops
}  // namespace viper
