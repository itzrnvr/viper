/*
 * viper RoPE kernel — implementation (corrected for partner read race).
 *
 * PURPOSE: see rope_kernel.h
 *
 * RACE FIX:
 *   The original kernel had a cross-warp race: thread tid reads
 *   x[partner] (where partner = tid +/- half) and writes x[tid].
 *   Thread `tid + half` does the symmetric read/write. Without a
 *   __syncthreads() between read and write, tid=0 (warp 0) could
 *   read x[64] AFTER tid=64 (warp 2) has already overwritten it.
 *
 *   Fix: each block loads its D-tile into shared memory with one
 *   store per element, __syncthreads, then the rest of the kernel
 *   reads partner values from shared memory (consistent across the
 *   block). Final writes go to global memory.
 *
 * CORRECTNESS:
 *   - inv_freq computed in fp32.
 *   - cos/sin in fp32; bf16 multiply cast through fp32.
 *   - rotate_half reads from shared memory after sync, not global.
 *
 * SAFETY:
 *   - No global allocations inside kernels.
 *   - SMEM usage: D * 2 bytes per block (D=128 → 256 bytes).
 *   - Block size = head_dim; grid = (B, T, num_heads).
 */
#include "rope_kernel.h"
#include <cuda_runtime.h>
#include <cmath>

namespace viper {
namespace ops {

__global__ void rope_cos_sin_kernel(
    float* __restrict__ cos_table,
    float* __restrict__ sin_table,
    int pos_start,
    int T,
    float theta,
    int head_dim) {
    const int t = blockIdx.x;
    if (t >= T) return;
    const int half = head_dim >> 1;
    const int tid = threadIdx.x;

    if (tid < half) {
        const float exponent = -2.0f * static_cast<float>(tid) / static_cast<float>(head_dim);
        const float inv_freq = __expf(exponent * __logf(theta));
        const float pos = static_cast<float>(pos_start + t);
        const float angle = pos * inv_freq;
        const float c = cosf(angle);
        const float s = sinf(angle);
        cos_table[t * head_dim + tid] = c;
        cos_table[t * head_dim + tid + half] = c;
        sin_table[t * head_dim + tid] = s;
        sin_table[t * head_dim + tid + half] = s;
    }
}

cudaError_t rope_precompute_cos_sin(
    float* cos_table,
    float* sin_table,
    int pos_start,
    int T,
    float theta,
    int head_dim,
    cudaStream_t stream) {
    if (!cos_table || !sin_table || T <= 0 || head_dim != 128) {
        return cudaErrorInvalidValue;
    }
    dim3 grid(T);
    dim3 block(128);
    rope_cos_sin_kernel<<<grid, block, 0, stream>>>(
        cos_table, sin_table, pos_start, T, theta, head_dim);
    return cudaGetLastError();
}

template <int HEAD_DIM>
__global__ void rope_apply_q_or_k_kernel(
    __nv_bfloat16* __restrict__ x,
    const float* __restrict__ cos_table,
    const float* __restrict__ sin_table,
    int B,
    int H,
    int T) {
    const int b = blockIdx.x;
    const int t = blockIdx.y;
    const int h = blockIdx.z;
    const int tid = threadIdx.x;
    constexpr int half = HEAD_DIM / 2;

    // Load the full D-tile into shared memory. Each thread loads one
    // element; with HEAD_DIM threads per block, this is one load per
    // thread. After this, the partner read from SMEM is race-free.
    __shared__ __nv_bfloat16 tile[HEAD_DIM];

    const int x_idx = ((b * H + h) * T + t) * HEAD_DIM + tid;
    tile[tid] = x[x_idx];
    __syncthreads();

    if (tid < HEAD_DIM) {
        const int partner = (tid < half) ? (tid + half) : (tid - half);
        const int cs_idx = t * HEAD_DIM + tid;

        const float x_val = __bfloat162float(tile[tid]);
        const float x_partner = __bfloat162float(tile[partner]);
        const float c = cos_table[cs_idx];
        const float s = sin_table[cs_idx];

        const float rotate = (tid < half) ? -x_partner : x_partner;
        const float out_val = x_val * c + rotate * s;
        x[x_idx] = __float2bfloat16(out_val);
    }
}

cudaError_t rope_apply_inplace_bf16(
    __nv_bfloat16* Q,
    __nv_bfloat16* K,
    const float* cos_table,
    const float* sin_table,
    int B,
    int num_heads_q,
    int num_heads_kv,
    int T,
    int head_dim,
    cudaStream_t stream) {
    if (!Q || !K || !cos_table || !sin_table || head_dim != 128 || T <= 0 || B <= 0
        || num_heads_q <= 0 || num_heads_kv <= 0) {
        return cudaErrorInvalidValue;
    }

    dim3 grid_q(B, T, num_heads_q);
    dim3 block(head_dim);
    rope_apply_q_or_k_kernel<128><<<grid_q, block, 0, stream>>>(
        Q, cos_table, sin_table, B, num_heads_q, T);

    dim3 grid_k(B, T, num_heads_kv);
    rope_apply_q_or_k_kernel<128><<<grid_k, block, 0, stream>>>(
        K, cos_table, sin_table, B, num_heads_kv, T);

    return cudaGetLastError();
}

}  // namespace ops
}  // namespace viper
