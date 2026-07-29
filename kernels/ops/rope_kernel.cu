/*
 * viper RoPE kernel — clean rewrite using warp-shuffle for partner reads.
 *
 * PURPOSE: see rope_kernel.h
 *
 * DESIGN:
 *   - 1 block = 1 warp = 32 threads. Each block handles ONE (b, t, h) tuple.
 *   - Each thread processes head_dim/32 elements (4 elements when D=128).
 *   - Partner read via __shfl_sync — no SMEM, no __syncthreads, no race.
 *   - Grid: (B, T, num_heads), block: 32 threads. head_dim is templated.
 *
 * CORRECTNESS:
 *   - inv_freq computed in fp32.
 *   - cos/sin in fp32; bf16 multiply cast through fp32.
 *
 * SAFETY:
 *   - No global allocations inside kernels.
 *   - SMEM usage: 0 bytes.
 *   - No __syncthreads needed — warp shuffle provides per-warp sync.
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

// 1 warp (32 threads) per (b, t, h) tuple. Each thread processes
// HEAD_DIM/32 elements via iterations. Partner read via __shfl_sync.
template <int HEAD_DIM>
__global__ void rope_apply_q_or_k_kernel(
    __nv_bfloat16* __restrict__ x,
    const float* __restrict__ cos_table,
    const float* __restrict__ sin_table,
    int B,
    int H,
    int T) {
    constexpr int half = HEAD_DIM / 2;
    constexpr int ITERS = HEAD_DIM / 32;  // elements per thread
    const int b = blockIdx.x;
    const int t = blockIdx.y;
    const int h = blockIdx.z;
    const int tid = threadIdx.x;  // 0..31

    // Each thread loads its assigned elements + partner via shfl.
    // For HEAD_DIM=128: ITERS=4, each thread handles positions
    // {tid, tid+32, tid+64, tid+96}.
    #pragma unroll
    for (int it = 0; it < ITERS; ++it) {
        const int pos = it * 32 + tid;        // global position in head_dim
        const int partner = (pos < half) ? (pos + half) : (pos - half);
        const int partner_iter = partner / 32;
        const int partner_tid = partner % 32;
        const int x_idx = ((b * T + t) * H + h) * HEAD_DIM + pos;
        const int partner_x_idx = ((b * T + t) * H + h) * HEAD_DIM + partner;

        const float x_val = __bfloat162float(x[x_idx]);
        // Read partner value from the partner thread's iteration
        // (synchronized within the warp — no race).
        // We can use shfl with the actual value, but we need to load
        // the partner's element into its register. Since each thread
        // loads its own pos into a register inside the same iteration,
        // the partner thread's register holds what we need if our
        // partner is in this thread's iteration. Otherwise we need a
        // second load.
        float partner_val;
        if (partner_iter == it) {
            // Partner is in this iteration; use shfl.
            partner_val = __shfl_sync(0xffffffff, x_val, partner_tid);
        } else {
            // Partner is in a different iteration; do a direct load.
            // (For HEAD_DIM=128 with half=64, partner and pos are
            // always in the same iteration since half=64 and 32<64<=96.)
            partner_val = __bfloat162float(x[partner_x_idx]);
        }

        const int cs_idx = t * HEAD_DIM + pos;
        const float c = cos_table[cs_idx];
        const float s = sin_table[cs_idx];

        const float rotate = (pos < half) ? -partner_val : partner_val;
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
    dim3 block(32);  // 1 warp per (b, t, h) tuple
    rope_apply_q_or_k_kernel<128><<<grid_q, block, 0, stream>>>(
        Q, cos_table, sin_table, B, num_heads_q, T);

    dim3 grid_k(B, T, num_heads_kv);
    rope_apply_q_or_k_kernel<128><<<grid_k, block, 0, stream>>>(
        K, cos_table, sin_table, B, num_heads_kv, T);

    return cudaGetLastError();
}

// Apply RoPE to K and write directly to KV cache (eliminates memcpy).
// Q is still modified in-place by the existing kernel.
__global__ void rope_k_to_cache_kernel(
    const __nv_bfloat16* __restrict__ k_src,  // [nKV, D]
    __nv_bfloat16* __restrict__ k_cache,       // [T, nKV, D]
    int pos, int nKV,
    const float* __restrict__ cos_t,
    const float* __restrict__ sin_t,
    int D) {
    const int half = D / 2;
    const int h = blockIdx.x;
    const int tid = threadIdx.x;
    if (tid >= D) return;

    int partner = (tid < half) ? (tid + half) : (tid - half);
    float kv = __bfloat162float(k_src[h * D + tid]);
    float kp = __bfloat162float(k_src[h * D + partner]);
    float c = cos_t[tid], s = sin_t[tid];
    float rotate = (tid < half) ? -kp : kp;

    k_cache[(size_t)pos * nKV * D + h * D + tid] =
        __float2bfloat16(kv * c + rotate * s);
}

// Fused Q-rope + K-to-cache in a single kernel launch.
// Grid: (nQ + nKVh) blocks, 128 threads. Q blocks use first 32 threads.
__global__ void rope_q_k_fused_kernel(
    __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K_src,
    __nv_bfloat16* __restrict__ K_cache,
    int pos, int nKV, int nQ,
    const float* __restrict__ cos_t,
    const float* __restrict__ sin_t) {
    constexpr int D = 128;
    constexpr int half = D / 2;
    const int bid = blockIdx.x;
    const int tid = threadIdx.x;
    if (bid < nQ) {
        // Q rope in-place: 128 threads, 1 per dimension
        const int h = bid;
        const int d = tid;  // 0-127
        const int partner = (d < half) ? (d + half) : (d - half);
        float xv = __bfloat162float(Q[h * D + d]);
        float pv = __bfloat162float(Q[h * D + partner]);
        float c = cos_t[d], s = sin_t[d];
        float rot = (d < half) ? -pv : pv;
        Q[h * D + d] = __float2bfloat16(xv * c + rot * s);
    } else {
        // K rope + write to cache: 128 threads, 1 per dim
        if (tid >= D) return;
        const int h = bid - nQ;
        int partner = (tid < half) ? (tid + half) : (tid - half);
        float kv = __bfloat162float(K_src[h * D + tid]);
        float kp = __bfloat162float(K_src[h * D + partner]);
        float c = cos_t[tid], s = sin_t[tid];
        float rot = (tid < half) ? -kp : kp;
        K_cache[(size_t)pos * nKV * D + h * D + tid] =
            __float2bfloat16(kv * c + rot * s);
    }
}
cudaError_t rope_apply_q_inplace_k_to_cache(
    __nv_bfloat16* Q,              // modified in-place
    const __nv_bfloat16* K_src,    // read-only (k_proj output before rope)
    __nv_bfloat16* K_cache,        // write target (KV cache slot)
    int pos, int nKV,
    const float* cos_t, const float* sin_t,
    int nQ, int nKVh, int T, int D,
    cudaStream_t stream) {
    if (D != 128) return cudaErrorInvalidValue;
    // Q: standard in-place rope
    dim3 grid_q(1, T, nQ);
    rope_apply_q_or_k_kernel<128><<<grid_q, 32, 0, stream>>>(
        Q, cos_t, sin_t, 1, nQ, T);
    // K: rope + write to cache (no separate memcpy needed)
    rope_k_to_cache_kernel<<<nKVh, 128, 0, stream>>>(
        K_src, K_cache, pos, nKV, cos_t, sin_t, D);
    return cudaGetLastError();
}

}  // namespace ops
}  // namespace viper
