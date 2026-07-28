// Decode attention: block per Q head, chunked online-softmax scan.
//
// Block: 128 threads (one per head-dim element for D=128).
// Chunk: 1024 KV positions. Each thread owns D-slice i of the output and
// computes 8 dots per chunk (t = tid + 128*j, j=0..7).
//
// Numerics: fp32 accumulation throughout; bf16 in/out.
#include "attn_decode_kernel.h"

namespace viper { namespace ops {

constexpr int kChunk = 1024;

__global__ void attn_decode_kernel(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k_cache,
    const __nv_bfloat16* __restrict__ v_cache,
    __nv_bfloat16* __restrict__ out,
    int nQ, int nKV, int D, int T_ctx,
    float scale) {
    const int h = blockIdx.x;
    if (h >= nQ) return;
    const int h_kv = (int)((long long)h * nKV / nQ);
    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;  // must equal D for the output mapping

    __shared__ float q_vec[128];
    __shared__ float dots[kChunk];
    __shared__ float red[32];
    __shared__ float s_scalars[2];  // [0]=chunk max, [1]=chunk sum partial

    // Stage q for this head.
    if (tid < D) q_vec[tid] = __bfloat162float(q[h * D + tid]);
    __syncthreads();

    float m_run = -1e30f;   // running max
    float l_run = 0.0f;     // running sum
    float acc = 0.0f;       // this thread's output element (dim = tid)

    const size_t kv_stride = (size_t)nKV * D;
    const size_t kv_off = (size_t)h_kv * D;

    for (int c0 = 0; c0 < T_ctx; c0 += kChunk) {
        const int c_len = min(kChunk, T_ctx - c0);
        // 1) dots for this chunk.
        for (int j = tid; j < c_len; j += nthreads) {
            const __nv_bfloat16* k_row = k_cache + (size_t)(c0 + j) * kv_stride + kv_off;
            float d = 0.0f;
            for (int i = 0; i < D; ++i) d += q_vec[i] * __bfloat162float(k_row[i]);
            dots[j] = d * scale;
        }
        __syncthreads();
        // 2) chunk max (block reduce, thread 0 writes s_scalars[0]).
        {
            float mx = -1e30f;
            for (int j = tid; j < c_len; j += nthreads) mx = fmaxf(mx, dots[j]);
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1)
                mx = fmaxf(mx, __shfl_down_sync(0xffffffffu, mx, off));
            if ((tid & 31) == 0) red[tid >> 5] = mx;
            __syncthreads();
            if (tid < 32) {
                float v = (tid < (nthreads >> 5)) ? red[tid] : -1e30f;
                #pragma unroll
                for (int off = 16; off > 0; off >>= 1)
                    v = fmaxf(v, __shfl_down_sync(0xffffffffu, v, off));
                if (tid == 0) s_scalars[0] = v;
            }
            __syncthreads();
        }
        const float m_new = fmaxf(m_run, s_scalars[0]);
        const float rescale = __expf(m_run - m_new);
        // 3) accumulate. l_part: per-thread subset of positions (for the
        // block-reduced denominator). acc: output dim tid must sum the
        // weighted V over ALL positions in the chunk.
        float l_part = 0.0f;
        float acc_new = acc * rescale;
        for (int j = tid; j < c_len; j += nthreads) {
            l_part += __expf(dots[j] - m_new);
        }
        if (tid < D) {
            for (int j = 0; j < c_len; ++j) {
                const float w = __expf(dots[j] - m_new);
                const __nv_bfloat16* v_row = v_cache + (size_t)(c0 + j) * kv_stride + kv_off;
                acc_new += w * __bfloat162float(v_row[tid]);
            }
        }
        acc = acc_new;
        // 4) block-reduce l_part -> chunk sum; combine into running sum.
        {
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1)
                l_part += __shfl_down_sync(0xffffffffu, l_part, off);
            if ((tid & 31) == 0) red[tid >> 5] = l_part;
            __syncthreads();
            if (tid < 32) {
                float v = (tid < (nthreads >> 5)) ? red[tid] : 0.0f;
                #pragma unroll
                for (int off = 16; off > 0; off >>= 1)
                    v += __shfl_down_sync(0xffffffffu, v, off);
                if (tid == 0) s_scalars[1] = v;
            }
            __syncthreads();
        }
        l_run = l_run * rescale + s_scalars[1];
        m_run = m_new;
        __syncthreads();
    }

    if (tid < D) out[h * D + tid] = __float2bfloat16(acc / fmaxf(l_run, 1e-30f));
}

cudaError_t attn_decode_bf16(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k_cache,
    const __nv_bfloat16* __restrict__ v_cache,
    __nv_bfloat16* __restrict__ out,
    int nQ, int nKV, int D, int T_ctx,
    float scale,
    cudaStream_t stream) {
    if (T_ctx <= 0) {
        cudaMemsetAsync(out, 0, (size_t)nQ * D * 2, stream);
        return cudaSuccess;
    }
    if (D != 128) return cudaErrorInvalidValue;  // v1: head_dim 128 only
    attn_decode_kernel<<<nQ, 128, 0, stream>>>(
        q, k_cache, v_cache, out, nQ, nKV, D, T_ctx, scale);
    return cudaGetLastError();
}

} }  // namespace viper::ops
