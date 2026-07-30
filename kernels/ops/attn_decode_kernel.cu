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

// ---- Q8 KV cache attention variant ----
// Same flash-style online softmax, but reads INT8 K/V + FP16 per-head scales.
// Dequantizes inline: value = int8_val * scale.
__global__ void attn_decode_q8_kernel(
    const __nv_bfloat16* __restrict__ q,
    const int8_t* __restrict__ k_cache_q8,    // [T, nKV, D] INT8
    const __nv_bfloat16* __restrict__ k_scales, // [T, nKV] FP16 per-head scales
    const int8_t* __restrict__ v_cache_q8,    // [T, nKV, D] INT8
    const __nv_bfloat16* __restrict__ v_scales, // [T, nKV] FP16
    __nv_bfloat16* __restrict__ out,
    int nQ, int nKV, int D, int T_ctx,
    float scale) {
    const int h = blockIdx.x;
    if (h >= nQ) return;
    const int h_kv = (int)((long long)h * nKV / nQ);
    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;

    __shared__ float q_vec[128];
    __shared__ float dots[kChunk];
    __shared__ float red[32];
    __shared__ float s_scalars[2];

    if (tid < D) q_vec[tid] = __bfloat162float(q[h * D + tid]);
    __syncthreads();

    float m_run = -1e30f, l_run = 0.0f, acc = 0.0f;
    const size_t kv_stride = (size_t)nKV * D;
    const size_t kv_off = (size_t)h_kv * D;

    for (int c0 = 0; c0 < T_ctx; c0 += kChunk) {
        const int c_len = min(kChunk, T_ctx - c0);
        // Dots: dequantize K inline
        for (int j = tid; j < c_len; j += nthreads) {
            const int8_t* k_row = k_cache_q8 + (size_t)(c0 + j) * kv_stride + kv_off;
            float k_sc = __bfloat162float(k_scales[(size_t)(c0 + j) * nKV + h_kv]);
            float d = 0.0f;
            for (int i = 0; i < D; ++i) d += q_vec[i] * ((float)k_row[i] * k_sc);
            dots[j] = d * scale;
        }
        __syncthreads();
        // Chunk max
        { float mx = -1e30f;
          for (int j = tid; j < c_len; j += nthreads) mx = fmaxf(mx, dots[j]);
          for (int off = 16; off > 0; off >>= 1) mx = fmaxf(mx, __shfl_down_sync(0xffffffffu, mx, off));
          if ((tid & 31) == 0) red[tid >> 5] = mx;
          __syncthreads();
          if (tid < 32) { float v = (tid < (nthreads >> 5)) ? red[tid] : -1e30f;
            for (int off = 16; off > 0; off >>= 1) v = fmaxf(v, __shfl_down_sync(0xffffffffu, v, off));
            if (tid == 0) s_scalars[0] = v; }
          __syncthreads(); }
        const float m_new = fmaxf(m_run, s_scalars[0]);
        const float rescale = __expf(m_run - m_new);
        // Accumulate V: dequantize inline
        float l_part = 0.0f;
        float acc_new = acc * rescale;
        for (int j = tid; j < c_len; j += nthreads) l_part += __expf(dots[j] - m_new);
        if (tid < D) {
            for (int j = 0; j < c_len; ++j) {
                const float w = __expf(dots[j] - m_new);
                const int8_t* v_row = v_cache_q8 + (size_t)(c0 + j) * kv_stride + kv_off;
                float v_sc = __bfloat162float(v_scales[(size_t)(c0 + j) * nKV + h_kv]);
                acc_new += w * ((float)v_row[tid] * v_sc);
            }
        }
        acc = acc_new;
        { for (int off = 16; off > 0; off >>= 1) l_part += __shfl_down_sync(0xffffffffu, l_part, off);
          if ((tid & 31) == 0) red[tid >> 5] = l_part;
          __syncthreads();
          if (tid < 32) { float v = (tid < (nthreads >> 5)) ? red[tid] : 0.0f;
            for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffffu, v, off);
            if (tid == 0) s_scalars[1] = v; }
          __syncthreads(); }
        l_run = l_run * rescale + s_scalars[1];
        m_run = m_new;
        __syncthreads();
    }
    if (tid < D) out[h * D + tid] = __float2bfloat16(acc / fmaxf(l_run, 1e-30f));
}

cudaError_t attn_decode_q8(
    const __nv_bfloat16* q,
    const int8_t* k_cache_q8, const __nv_bfloat16* k_scales,
    const int8_t* v_cache_q8, const __nv_bfloat16* v_scales,
    __nv_bfloat16* out,
    int nQ, int nKV, int D, int T_ctx, float scale,
    cudaStream_t stream) {
    if (T_ctx <= 0) { cudaMemsetAsync(out, 0, (size_t)nQ * D * 2, stream); return cudaSuccess; }
    if (D != 128) return cudaErrorInvalidValue;
    attn_decode_q8_kernel<<<nQ, 128, 0, stream>>>(
        q, k_cache_q8, k_scales, v_cache_q8, v_scales, out, nQ, nKV, D, T_ctx, scale);
    return cudaGetLastError();
}

// ---- Q4 KV cache attention variant ----
// Same flash-style online softmax, reads packed 4-bit K/V (2 values per byte).
// Offset encoding: stored = value + 8 (range 0-15 for values -8..7).
__global__ void attn_decode_q4_kernel(
    const __nv_bfloat16* __restrict__ q,
    const uint8_t* __restrict__ k_cache_q4,     // [T, nKV, D/2] packed Q4
    const __nv_bfloat16* __restrict__ k_scales,  // [T, nKV] FP16
    const uint8_t* __restrict__ v_cache_q4,     // [T, nKV, D/2]
    const __nv_bfloat16* __restrict__ v_scales,  // [T, nKV]
    __nv_bfloat16* __restrict__ out,
    int nQ, int nKV, int D, int T_ctx, float scale) {
    const int h = blockIdx.x;
    if (h >= nQ) return;
    const int h_kv = (int)((long long)h * nKV / nQ);
    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;
    const int halfD = D / 2;

    __shared__ float q_vec[128];
    __shared__ float dots[kChunk];
    __shared__ float red[32];
    __shared__ float s_scalars[2];
    if (tid < D) q_vec[tid] = __bfloat162float(q[h * D + tid]);
    __syncthreads();

    float m_run = -1e30f, l_run = 0.0f, acc = 0.0f;
    const size_t kv_stride = (size_t)nKV * halfD;
    const size_t kv_off = (size_t)h_kv * halfD;

    for (int c0 = 0; c0 < T_ctx; c0 += kChunk) {
        const int c_len = min(kChunk, T_ctx - c0);
        for (int j = tid; j < c_len; j += nthreads) {
            const uint8_t* k_row = k_cache_q4 + (size_t)(c0 + j) * kv_stride + kv_off;
            float k_sc = __bfloat162float(k_scales[(size_t)(c0 + j) * nKV + h_kv]);
            float d = 0.0f;
            for (int i = 0; i < D; i += 2) {
                uint8_t packed = k_row[i / 2];
                d += q_vec[i] * (float)((packed & 0xF) - 8) * k_sc;
                d += q_vec[i + 1] * (float)(((packed >> 4) & 0xF) - 8) * k_sc;
            }
            dots[j] = d * scale;
        }
        __syncthreads();
        { float mx = -1e30f;
          for (int j = tid; j < c_len; j += nthreads) mx = fmaxf(mx, dots[j]);
          for (int off = 16; off > 0; off >>= 1) mx = fmaxf(mx, __shfl_down_sync(0xffffffffu, mx, off));
          if ((tid & 31) == 0) red[tid >> 5] = mx; __syncthreads();
          if (tid < 32) { float v = (tid < (nthreads >> 5)) ? red[tid] : -1e30f;
            for (int off = 16; off > 0; off >>= 1) v = fmaxf(v, __shfl_down_sync(0xffffffffu, v, off));
            if (tid == 0) s_scalars[0] = v; } __syncthreads(); }
        const float m_new = fmaxf(m_run, s_scalars[0]);
        const float rescale = __expf(m_run - m_new);
        float l_part = 0.0f;
        float acc_new = acc * rescale;
        for (int j = tid; j < c_len; j += nthreads) l_part += __expf(dots[j] - m_new);
        if (tid < D) {
            for (int j = 0; j < c_len; ++j) {
                const float w = __expf(dots[j] - m_new);
                const uint8_t* v_row = v_cache_q4 + (size_t)(c0 + j) * kv_stride + kv_off;
                float v_sc = __bfloat162float(v_scales[(size_t)(c0 + j) * nKV + h_kv]);
                int v0 = (v_row[tid / 2] & 0xF) - 8;
                int v1 = ((v_row[tid / 2] >> 4) & 0xF) - 8;
                float val = (tid & 1) ? (float)v1 * v_sc : (float)v0 * v_sc;
                acc_new += w * val;
            }
        }
        acc = acc_new;
        { for (int off = 16; off > 0; off >>= 1) l_part += __shfl_down_sync(0xffffffffu, l_part, off);
          if ((tid & 31) == 0) red[tid >> 5] = l_part; __syncthreads();
          if (tid < 32) { float v = (tid < (nthreads >> 5)) ? red[tid] : 0.0f;
            for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffffu, v, off);
            if (tid == 0) s_scalars[1] = v; } __syncthreads(); }
        l_run = l_run * rescale + s_scalars[1];
        m_run = m_new;
        __syncthreads();
    }
    if (tid < D) out[h * D + tid] = __float2bfloat16(acc / fmaxf(l_run, 1e-30f));
}

cudaError_t attn_decode_q4(
    const __nv_bfloat16* q,
    const uint8_t* k_cache_q4, const __nv_bfloat16* k_scales,
    const uint8_t* v_cache_q4, const __nv_bfloat16* v_scales,
    __nv_bfloat16* out,
    int nQ, int nKV, int D, int T_ctx, float scale,
    cudaStream_t stream) {
    if (T_ctx <= 0) { cudaMemsetAsync(out, 0, (size_t)nQ * D * 2, stream); return cudaSuccess; }
    if (D != 128) return cudaErrorInvalidValue;
    attn_decode_q4_kernel<<<nQ, 128, 0, stream>>>(
        q, k_cache_q4, k_scales, v_cache_q4, v_scales, out, nQ, nKV, D, T_ctx, scale);
    return cudaGetLastError();
}
// ---- Q6 KV cache attention variant ----
// Same online softmax, reads packed 6-bit K/V (4 values per 3 bytes).
// Two's complement: pack_q6 masks with & 0x3F, unpack sign-extends (>=32 → -64).
__global__ void attn_decode_q6_kernel(
    const __nv_bfloat16* __restrict__ q,
    const uint8_t* __restrict__ k_cache_q6,     // [T, nKV, D/4*3] packed Q6
    const __nv_bfloat16* __restrict__ k_scales,  // [T, nKV] FP16
    const uint8_t* __restrict__ v_cache_q6,     // [T, nKV, D/4*3]
    const __nv_bfloat16* __restrict__ v_scales,  // [T, nKV]
    __nv_bfloat16* __restrict__ out,
    int nQ, int nKV, int D, int T_ctx, float scale) {
    const int h = blockIdx.x;
    if (h >= nQ) return;
    const int h_kv = (int)((long long)h * nKV / nQ);
    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;
    const int packed_width = D / 4 * 3;  // bytes per head (96 for D=128)

    __shared__ float q_vec[128];
    __shared__ float dots[kChunk];
    __shared__ float red[32];
    __shared__ float s_scalars[2];
    if (tid < D) q_vec[tid] = __bfloat162float(q[h * D + tid]);
    __syncthreads();

    float m_run = -1e30f, l_run = 0.0f, acc = 0.0f;
    const size_t kv_stride = (size_t)nKV * packed_width;
    const size_t kv_off = (size_t)h_kv * packed_width;

    for (int c0 = 0; c0 < T_ctx; c0 += kChunk) {
        const int c_len = min(kChunk, T_ctx - c0);
        // Dots: unpack 4 Q6 values per 3 bytes, inline dequantize
        for (int j = tid; j < c_len; j += nthreads) {
            const uint8_t* k_row = k_cache_q6 + (size_t)(c0 + j) * kv_stride + kv_off;
            float k_sc = __bfloat162float(k_scales[(size_t)(c0 + j) * nKV + h_kv]);
            float d = 0.0f;
            for (int i = 0; i < D; i += 4) {
                const uint8_t* base = k_row + (i / 4) * 3;
                int v0 = base[0] & 0x3F;
                int v1 = ((base[0] >> 6) | (base[1] << 2)) & 0x3F;
                int v2 = ((base[1] >> 4) | (base[2] << 4)) & 0x3F;
                int v3 = (base[2] >> 2) & 0x3F;
                if (v0 >= 32) v0 -= 64;
                if (v1 >= 32) v1 -= 64;
                if (v2 >= 32) v2 -= 64;
                if (v3 >= 32) v3 -= 64;
                d += q_vec[i]   * (float)v0 * k_sc;
                d += q_vec[i+1] * (float)v1 * k_sc;
                d += q_vec[i+2] * (float)v2 * k_sc;
                d += q_vec[i+3] * (float)v3 * k_sc;
            }
            dots[j] = d * scale;
        }
        __syncthreads();
        // Chunk max
        { float mx = -1e30f;
          for (int j = tid; j < c_len; j += nthreads) mx = fmaxf(mx, dots[j]);
          for (int off = 16; off > 0; off >>= 1) mx = fmaxf(mx, __shfl_down_sync(0xffffffffu, mx, off));
          if ((tid & 31) == 0) red[tid >> 5] = mx; __syncthreads();
          if (tid < 32) { float v = (tid < (nthreads >> 5)) ? red[tid] : -1e30f;
            for (int off = 16; off > 0; off >>= 1) v = fmaxf(v, __shfl_down_sync(0xffffffffu, v, off));
            if (tid == 0) s_scalars[0] = v; } __syncthreads(); }
        const float m_new = fmaxf(m_run, s_scalars[0]);
        const float rescale = __expf(m_run - m_new);
        // Accumulate V: unpack single Q6 value at tid
        float l_part = 0.0f;
        float acc_new = acc * rescale;
        for (int j = tid; j < c_len; j += nthreads) l_part += __expf(dots[j] - m_new);
        if (tid < D) {
            int group = tid >> 2;    // tid / 4
            int sub = tid & 3;        // tid % 4
            for (int j = 0; j < c_len; ++j) {
                const float w = __expf(dots[j] - m_new);
                const uint8_t* v_row = v_cache_q6 + (size_t)(c0 + j) * kv_stride + kv_off;
                float v_sc = __bfloat162float(v_scales[(size_t)(c0 + j) * nKV + h_kv]);
                const uint8_t* base = v_row + group * 3;
                int v;
                switch (sub) {
                    case 0:  v = base[0] & 0x3F; break;
                    case 1:  v = ((base[0] >> 6) | (base[1] << 2)) & 0x3F; break;
                    case 2:  v = ((base[1] >> 4) | (base[2] << 4)) & 0x3F; break;
                    default: v = (base[2] >> 2) & 0x3F; break;
                }
                if (v >= 32) v -= 64;
                acc_new += w * (float)v * v_sc;
            }
        }
        acc = acc_new;
        { for (int off = 16; off > 0; off >>= 1) l_part += __shfl_down_sync(0xffffffffu, l_part, off);
          if ((tid & 31) == 0) red[tid >> 5] = l_part; __syncthreads();
          if (tid < 32) { float v = (tid < (nthreads >> 5)) ? red[tid] : 0.0f;
            for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffffu, v, off);
            if (tid == 0) s_scalars[1] = v; } __syncthreads(); }
        l_run = l_run * rescale + s_scalars[1];
        m_run = m_new;
        __syncthreads();
    }
    if (tid < D) out[h * D + tid] = __float2bfloat16(acc / fmaxf(l_run, 1e-30f));
}

cudaError_t attn_decode_q6(
    const __nv_bfloat16* q,
    const uint8_t* k_cache_q6, const __nv_bfloat16* k_scales,
    const uint8_t* v_cache_q6, const __nv_bfloat16* v_scales,
    __nv_bfloat16* out,
    int nQ, int nKV, int D, int T_ctx, float scale,
    cudaStream_t stream) {
    if (T_ctx <= 0) { cudaMemsetAsync(out, 0, (size_t)nQ * D * 2, stream); return cudaSuccess; }
    if (D != 128) return cudaErrorInvalidValue;
    attn_decode_q6_kernel<<<nQ, 128, 0, stream>>>(
        q, k_cache_q6, k_scales, v_cache_q6, v_scales, out, nQ, nKV, D, T_ctx, scale);
    return cudaGetLastError();
}

// ---- Batch attention with causal masking (for speculative decode) ----
// Grid: (nQ, M). Each block handles one (head, token) pair.
// Token m attends to positions 0..pos_start+m (causal).
__global__ void attn_batch_kernel(
    const __nv_bfloat16* __restrict__ q,        // [M, nQ, D]
    const __nv_bfloat16* __restrict__ k_cache,  // [T_total, nKV, D]
    const __nv_bfloat16* __restrict__ v_cache,  // [T_total, nKV, D]
    __nv_bfloat16* __restrict__ out,            // [M, nQ, D]
    int M, int nQ, int nKV, int D,
    int pos_start,
    float scale) {
    const int h = blockIdx.x;
    const int m = blockIdx.y;
    if (h >= nQ || m >= M) return;
    const int h_kv = (int)((long long)h * nKV / nQ);
    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;
    const int T_ctx = pos_start + m + 1;  // causal limit

    __shared__ float q_vec[128];
    __shared__ float dots[kChunk];
    __shared__ float red[32];
    __shared__ float s_scalars[2];

    const __nv_bfloat16* q_row = q + (size_t)m * nQ * D + (size_t)h * D;
    if (tid < D) q_vec[tid] = __bfloat162float(q_row[tid]);
    __syncthreads();

    float m_run = -1e30f, l_run = 0.0f, acc = 0.0f;
    const size_t kv_stride = (size_t)nKV * D;
    const size_t kv_off = (size_t)h_kv * D;

    for (int c0 = 0; c0 < T_ctx; c0 += kChunk) {
        const int c_len = min(kChunk, T_ctx - c0);
        for (int j = tid; j < c_len; j += nthreads) {
            const __nv_bfloat16* k_row = k_cache + (size_t)(c0 + j) * kv_stride + kv_off;
            float d = 0.0f;
            for (int i = 0; i < D; ++i) d += q_vec[i] * __bfloat162float(k_row[i]);
            dots[j] = d * scale;
        }
        __syncthreads();
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
        float l_part = 0.0f;
        float acc_new = acc * rescale;
        for (int j = tid; j < c_len; j += nthreads)
            l_part += __expf(dots[j] - m_new);
        if (tid < D) {
            for (int j = 0; j < c_len; ++j) {
                const float w = __expf(dots[j] - m_new);
                const __nv_bfloat16* v_row = v_cache + (size_t)(c0 + j) * kv_stride + kv_off;
                acc_new += w * __bfloat162float(v_row[tid]);
            }
        }
        acc = acc_new;
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

    if (tid < D)
        out[(size_t)m * nQ * D + (size_t)h * D + tid] =
            __float2bfloat16(acc / fmaxf(l_run, 1e-30f));
}

cudaError_t attn_batch_bf16(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k_cache,
    const __nv_bfloat16* __restrict__ v_cache,
    __nv_bfloat16* __restrict__ out,
    int M, int nQ, int nKV, int D,
    int pos_start,
    float scale,
    cudaStream_t stream) {
    if (D != 128) return cudaErrorInvalidValue;
    dim3 grid(nQ, M);
    attn_batch_kernel<<<grid, 128, 0, stream>>>(
        q, k_cache, v_cache, out, M, nQ, nKV, D, pos_start, scale);
    return cudaGetLastError();
}

// ---- Batch KV cache append ----
// Copies M K/V vectors [M, nKV, D] into the cache at positions pos_start..pos_start+M-1.
__global__ void kv_append_batch_kernel(
    const __nv_bfloat16* __restrict__ new_k,
    const __nv_bfloat16* __restrict__ new_v,
    __nv_bfloat16* __restrict__ k_cache,
    __nv_bfloat16* __restrict__ v_cache,
    int M, int nKV, int D, int pos_start) {
    int m = blockIdx.x;
    int idx = threadIdx.x;
    int total = nKV * D;
    if (idx < total) {
        int cache_off = (pos_start + m) * total + idx;
        int new_off = m * total + idx;
        k_cache[cache_off] = new_k[new_off];
        v_cache[cache_off] = new_v[new_off];
    }
}

cudaError_t kv_append_batch_bf16(
    const __nv_bfloat16* new_k,
    const __nv_bfloat16* new_v,
    __nv_bfloat16* k_cache,
    __nv_bfloat16* v_cache,
    int M, int nKV, int D, int pos_start,
    cudaStream_t stream) {
    int total = nKV * D;
    int block = min(total, 1024);
    kv_append_batch_kernel<<<M, block, 0, stream>>>(
        new_k, new_v, k_cache, v_cache, M, nKV, D, pos_start);
    return cudaGetLastError();
}

} }  // namespace viper::ops
