/*
 * viper DP4A Q4 GEMV — v4 with FUSED RoPE and INPUT-SIDE RESIDUAL.
 *
 * NEW FUSIONS (vs v3):
 *   - Input-side residual: x += residual BEFORE rmsnorm (saves 2 HBM ops/layer)
 *   - RoPE fusion: Q/K elements rotated IN-KERNEL after GEMV (saves 1 kernel launch)
 *
 * Based on dp4a_q4_kernel.cu v3. Same DP4A inner loop, same SMEM layout.
 *
 * BUILD: nvcc -arch=sm_86 -std=c++20 -c dp4a_q4_v4_kernel.cu
 */
#include "linear_kernel.h"
#include <cuda_runtime.h>

namespace viper {
namespace ops {

// ============================================================
// KERNEL: Fused residual + rmsnorm + Q4 GEMV + RoPE
// ============================================================
// For Q/K projections: after GEMV, apply RoPE before writing output.
// For V projections: no RoPE (standard GEMV).
// For MLP projections: no RoPE (standard GEMV with input-side residual).
//
// RoPE requirements:
//   - HEAD_DIM=128, half=64
//   - Q has nQ = num_heads * HEAD_DIM = 48 * 128 = 6144 elements
//   - K has nKV = num_kv_heads * HEAD_DIM = 8 * 128 = 1024 elements
//   - Each element pairs with element ±half in the same head
//
// Strategy: use a separate SMEM buffer for Q/K GEMV output.
//   Phase 1: Load x (+ optional residual_in + swiglu) → SMEM
//   Phase 2: Fused rmsnorm + INT8 quant
//   Phase 3: DP4A GEMV → write to SMEM output buffer (NOT global)
//   Phase 4: Sync. Apply RoPE in SMEM. Write rotated result to global.
//
// This kernel handles ONE projection (Q OR K OR V OR MLP) per launch.
// The QKV trio is handled by 3 sequential launches of this kernel,
// with rope params set for Q and K, NULL for V.

template <bool APPLY_ROPE, int HEAD_DIM = 128>
__global__ void dp4a_q4_v4_kernel(
    const uint8_t* __restrict__ w_packed,
    const __nv_bfloat16* __restrict__ w_scales,
    const __nv_bfloat16* __restrict__ x,
    __nv_bfloat16* __restrict__ y,
    // Input-side residual (fused with rmsnorm)
    const __nv_bfloat16* __restrict__ residual_in,  // NEW: add to x BEFORE norm
    int M, int N, int K,
    // RMSNorm params
    const __nv_bfloat16* __restrict__ gamma,
    float eps,
    // SwiGLU fusion
    const __nv_bfloat16* __restrict__ swiglu_up,
    // Output-side residual (for O/down projections — kept for compat)
    const __nv_bfloat16* __restrict__ residual_out,
    // RoPE params (only used when APPLY_ROPE=true)
    const float* __restrict__ rope_cos,   // [HEAD_DIM] precomputed cos for this position
    const float* __restrict__ rope_sin,   // [HEAD_DIM] precomputed sin for this position
    int rope_offset,                      // offset into y for this head's Q/K block
    int rope_n_heads                      // number of heads in this projection
) {
    constexpr int half = HEAD_DIM / 2;
    const int m = blockIdx.y;
    const int warp_id = threadIdx.x >> 5;
    const int lane_id = threadIdx.x & 31;
    const int n = blockIdx.x * (blockDim.x >> 5) + warp_id;

    // SMEM layout:
    // [K int8] + [K/64 scales] + [128 pad] + [K bf16 temp] + [optional: HEAD_DIM output buf]
    extern __shared__ char smem_v4[];
    int8_t* xq = (int8_t*)smem_v4;
    __nv_bfloat16* xs_scales = (__nv_bfloat16*)(smem_v4 + K);
    __nv_bfloat16* xbf = (__nv_bfloat16*)(smem_v4 + K + K / 64 * 2 + 128);

    // RoPE output buffer (only needed for Q/K projections with APPLY_ROPE)
    // Uses space AFTER xbf (xbf is freed after Phase 2, so we reuse)
    // Layout: [HEAD_DIM] for the current head's output
    __nv_bfloat16* rope_buf = xbf;  // reuse xbf buffer (freed after quant)

    // ---- Phase 1: Load activations (+ optional input residual + swiglu) ----
    const __nv_bfloat16* x_global = x + (size_t)m * K;
    for (int i = threadIdx.x; i < K; i += blockDim.x) {
        float val;
        if (swiglu_up) {
            float g = __bfloat162float(x_global[i]);
            float u = __bfloat162float(swiglu_up[i]);
            val = g / (1.0f + __expf(-g)) * u;
        } else {
            val = __bfloat162float(x_global[i]);
        }
        // NEW: input-side residual — add BEFORE normalization
        if (residual_in) {
            val += __bfloat162float(residual_in[m * K + i]);
        }
        xbf[i] = __float2bfloat16(val);
    }

    // ---- Phase 2: Fused rmsnorm + INT8 quantization ----
    float inv_rms = 1.0f;
    if (gamma) {
        float ss = 0.0f;
        for (int i = threadIdx.x; i < K; i += blockDim.x) {
            float v = __bfloat162float(xbf[i]);
            ss += v * v;
        }
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            ss += __shfl_xor_sync(0xffffffff, ss, off);
        __shared__ float ws_v4[8];
        if ((threadIdx.x & 31) == 0) ws_v4[threadIdx.x >> 5] = ss;
        __syncthreads();
        if (threadIdx.x < 32) {
            float t = (threadIdx.x < 8) ? ws_v4[threadIdx.x] : 0.0f;
            #pragma unroll
            for (int off = 4; off > 0; off >>= 1)
                t += __shfl_xor_sync(0xffffffff, t, off);
            if (threadIdx.x == 0) ws_v4[0] = rsqrtf(t / (float)K + eps);
        }
        __syncthreads();
        inv_rms = ws_v4[0];
    }

    const int n_groups = K / 64;
    for (int g = threadIdx.x; g < n_groups; g += blockDim.x) {
        int start = g * 64;
        float maxabs = 1e-8f;
        for (int i = 0; i < 64; ++i) {
            float v = fabsf(__bfloat162float(xbf[start + i]) * inv_rms
                           * (gamma ? __bfloat162float(gamma[start + i]) : 1.0f));
            if (v > maxabs) maxabs = v;
        }
        float sc = maxabs / 127.0f;
        float inv_sc = 1.0f / sc;
        xs_scales[g] = __float2bfloat16(sc);
        for (int i = 0; i < 64; ++i) {
            float v = __bfloat162float(xbf[start + i]) * inv_rms
                     * (gamma ? __bfloat162float(gamma[start + i]) : 1.0f);
            int q = __float2int_rn(v * inv_sc);
            xq[start + i] = (int8_t)(q > 127 ? 127 : (q < -128 ? -128 : q));
        }
    }
    __syncthreads();

    // xbf no longer needed — SMEM reused for RoPE output if needed
    if (n >= N || m >= M) return;

    // ---- Phase 3: DP4A GEMV (identical to v3) ----
    const uint8_t* w_row = w_packed + (size_t)n * (K / 2);
    const __nv_bfloat16* s_row = w_scales + (size_t)n * (K / 64);
    const int32_t* xq32 = reinterpret_cast<const int32_t*>(xq);

    float acc = 0.0f;
    const int group_in_lane = lane_id >> 3;
    const int quad_in_group = lane_id & 7;

    for (int gbase = 0; gbase < n_groups; gbase += 4) {
        int g = gbase + group_in_lane;
        if (g >= n_groups) break;

        int byte_off = g * 32 + quad_in_group * 4;
        uint32_t packed4 = *reinterpret_cast<const uint32_t*>(w_row + byte_off);

        uint32_t lo = packed4 & 0x0F0F0F0F;
        uint32_t hi = (packed4 >> 4) & 0x0F0F0F0F;
        int32_t slo = __vsubss4(lo, 0x08080808);
        int32_t shi = __vsubss4(hi, 0x08080808);
        int32_t w_lo = __byte_perm(slo, shi, 0x5140);
        int32_t w_hi = __byte_perm(slo, shi, 0x7362);

        int elem = byte_off * 2;
        int32_t x_lo = xq32[elem / 4];
        int32_t x_hi = xq32[elem / 4 + 1];

        int32_t partial = __dp4a(w_lo, x_lo, 0);
        partial = __dp4a(w_hi, x_hi, partial);

        #pragma unroll
        for (int off = 4; off > 0; off >>= 1)
            partial += __shfl_xor_sync(0xffffffff, partial, off);

        if (quad_in_group == 0) {
            float ws = __bfloat162float(s_row[g]);
            float xs = __bfloat162float(xs_scales[g]);
            acc += (float)partial * ws * xs;
        }
    }

    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        acc += __shfl_xor_sync(0xffffffff, acc, off);

    // ---- Phase 3b: Output (with optional output-side residual) ----
    if (lane_id == 0) {
        if (residual_out)
            acc += __bfloat162float(residual_out[m * N + n]);
        y[m * N + n] = __float2bfloat16(acc);
    }

    // NOTE: RoPE fusion for Q/K requires a DIFFERENT approach.
    // The DP4A GEMV is embarrassingly parallel across output elements.
    // RoPE pairs element n with element n±half (within the same head).
    // These paired elements may be in different thread blocks.
    //
    // For TRUE RoPE fusion into the GEMV, we need a SEPARATE kernel
    // variant that computes ALL elements of one head in a single block,
    // so the paired elements are accessible via SMEM + __syncthreads.
    //
    // That variant is dp4a_q4_qkv_rope_kernel below.
}

// ============================================================
// KERNEL: Fused QKV Projection + RoPE (megakernel for one head)
// ============================================================
// One block computes ALL HEAD_DIM=128 output elements for ONE head.
// After GEMV: all elements in registers/SMEM → apply RoPE in-place.
// This eliminates the separate RoPE kernel launch entirely.
//
// Grid: (num_heads, M) for Q, (num_kv_heads, M) for K
// Block: 128 threads (1 thread per output element)
// SMEM: K (int8) + K/64*2 (scales) + 128 (pad) + HEAD_DIM (bf16 output buf)

template <int HEAD_DIM>
__global__ void dp4a_q4_qkv_rope_kernel(
    const uint8_t* __restrict__ w_packed,
    const __nv_bfloat16* __restrict__ w_scales,
    const __nv_bfloat16* __restrict__ x,
    __nv_bfloat16* __restrict__ y,         // Q or K output
    const __nv_bfloat16* __restrict__ residual_in,
    int M, int K,
    const __nv_bfloat16* __restrict__ gamma,
    float eps,
    const float* __restrict__ rope_cos,
    const float* __restrict__ rope_sin
) {
    constexpr int half = HEAD_DIM / 2;
    const int m = blockIdx.y;      // batch/sequence position
    const int head = blockIdx.x;   // which head
    const int tid = threadIdx.x;   // 0..127, one per output element

    extern __shared__ char smem_qkv[];
    int8_t* xq = (int8_t*)smem_qkv;
    __nv_bfloat16* xs_scales = (__nv_bfloat16*)(smem_qkv + K);
    __nv_bfloat16* xbf = (__nv_bfloat16*)(smem_qkv + K + K/64*2 + 128);
    __nv_bfloat16* out_buf = (__nv_bfloat16*)(smem_qkv + K + K/64*2 + 128 + K*2);

    // Phase 1: Load x + residual_in (fused)
    const __nv_bfloat16* x_global = x + (size_t)m * K;
    for (int i = tid; i < K; i += HEAD_DIM) {
        float val = __bfloat162float(x_global[i]);
        if (residual_in) val += __bfloat162float(residual_in[m * K + i]);
        xbf[i] = __float2bfloat16(val);
    }

    // Phase 2: Fused rmsnorm + quant
    float ss = 0.0f;
    for (int i = tid; i < K; i += HEAD_DIM) {
        float v = __bfloat162float(xbf[i]);
        ss += v * v;
    }
    // Reduce across 128 threads (4 warps)
    __shared__ float reduce_ws[4];
    if ((tid & 31) == 0) reduce_ws[tid >> 5] = ss;
    __syncthreads();
    if (tid < 32) {
        float t = (tid < 4) ? reduce_ws[tid] : 0.0f;
        #pragma unroll
        for (int off = 2; off > 0; off >>= 1)
            t += __shfl_xor_sync(0xffffffff, t, off);
        if (tid == 0) reduce_ws[0] = rsqrtf(t / (float)K + eps);
    }
    __syncthreads();
    float inv_rms = reduce_ws[0];

    const int n_groups = K / 64;
    for (int g = tid; g < n_groups; g += HEAD_DIM) {
        int start = g * 64;
        float maxabs = 1e-8f;
        for (int i = 0; i < 64; ++i) {
            float v = fabsf(__bfloat162float(xbf[start+i]) * inv_rms
                           * __bfloat162float(gamma[start+i]));
            if (v > maxabs) maxabs = v;
        }
        float sc = maxabs / 127.0f;
        xs_scales[g] = __float2bfloat16(sc);
        float inv_sc = 1.0f / sc;
        for (int i = 0; i < 64; ++i) {
            float v = __bfloat162float(xbf[start+i]) * inv_rms * __bfloat162float(gamma[start+i]);
            int q = __float2int_rn(v * inv_sc);
            xq[start+i] = (int8_t)(q > 127 ? 127 : (q < -128 ? -128 : q));
        }
    }
    __syncthreads();

    // Phase 3: DP4A GEMV — this thread computes output element `tid` for `head`
    // Weight row for this head's element tid:
    const uint8_t* w_row = w_packed + (size_t)(head * HEAD_DIM + tid) * (K / 2);
    const __nv_bfloat16* s_row = w_scales + (size_t)(head * HEAD_DIM + tid) * (K / 64);
    const int32_t* xq32 = reinterpret_cast<const int32_t*>(xq);

    float acc = 0.0f;
    // Each thread iterates over all groups for its output element
    for (int g = 0; g < n_groups; g++) {
        int byte_off = g * 32;  // 32 bytes per group in weight
        // Each thread reads 4 bytes from the weight for this group
        // (simplified: each thread handles its own element's weight row)
        // Note: this is a scalar inner loop, not the warp-cooperative DP4A.
        // For full DP4A speed, we'd need 8 threads cooperating per element.
        // This is a correctness-first implementation; optimization later.

        float group_acc = 0.0f;
        for (int qi = 0; qi < 64; qi += 4) {
            int elem = g * 64 + qi;
            // Load 4 weight values (packed as 2 bytes → 4 int4)
            int w_byte_off = g * 32 + (qi / 4) * 2;
            uint32_t packed2 = w_row[w_byte_off] | (w_row[w_byte_off + 1] << 8);
            // Unpack 4 int4 values
            int8_t w0 = (packed2 & 0xF) - 8;
            int8_t w1 = ((packed2 >> 4) & 0xF) - 8;
            int8_t w2 = ((packed2 >> 8) & 0xF) - 8;
            int8_t w3 = ((packed2 >> 12) & 0xF) - 8;

            // Load 4 activation values
            float a0 = (float)xq[elem]   * __bfloat162float(xs_scales[g]);
            float a1 = (float)xq[elem+1] * __bfloat162float(xs_scales[g]);
            float a2 = (float)xq[elem+2] * __bfloat162float(xs_scales[g]);
            float a3 = (float)xq[elem+3] * __bfloat162float(xs_scales[g]);

            group_acc += w0*a0 + w1*a1 + w2*a2 + w3*a3;
        }
        acc += group_acc * __bfloat162float(s_row[g]);
    }

    // Store GEMV result to SMEM output buffer
    out_buf[tid] = __float2bfloat16(acc);
    __syncthreads();

    // Phase 4: FUSED RoPE — apply rotation in SMEM (no separate kernel!)
    // Each thread rotates its element using its partner's value from SMEM
    {
        const int partner = (tid < half) ? (tid + half) : (tid - half);
        const float x_val = __bfloat162float(out_buf[tid]);
        const float partner_val = __bfloat162float(out_buf[partner]);
        const float c = rope_cos[tid];
        const float s = rope_sin[tid];
        const float rotate = (tid < half) ? -partner_val : partner_val;
        const float rotated = x_val * c + rotate * s;

        // Write rotated Q/K to global output
        y[m * (HEAD_DIM * gridDim.x) + head * HEAD_DIM + tid] = __float2bfloat16(rotated);
    }
}

// ============================================================
// LAUNCHERS
// ============================================================

// Fused residual + rmsnorm + GEMV (for input to attention/MLP blocks)
// Replaces: separate residual_add + rmsnorm + GEMV = 3 launches → 1 launch
cudaError_t dp4a_q4_fused_residual_norm_gemm(
    const uint8_t* w, const __nv_bfloat16* s,
    const __nv_bfloat16* x, const __nv_bfloat16* residual_in,
    __nv_bfloat16* y, int M, int N, int K,
    const __nv_bfloat16* gamma, float eps,
    cudaStream_t stream)
{
    size_t smem = K + K/64*2 + 128 + K*2;
    dp4a_q4_v4_kernel<false><<<(N+7)/8, 256, smem, stream>>>(
        w, s, x, y, residual_in, M, N, K, gamma, eps,
        nullptr, nullptr, nullptr, 0, 0);
    return cudaGetLastError();
}

// Fused QKV projection + RoPE (for Q and K)
// Replaces: QKV GEMV + RoPE kernel = 2 launches → 1 launch
// Launch ONE per head: grid = (num_heads, M)
cudaError_t dp4a_q4_qkv_with_rope(
    const uint8_t* w, const __nv_bfloat16* s,
    const __nv_bfloat16* x, const __nv_bfloat16* residual_in,
    __nv_bfloat16* y,
    int M, int K, int num_heads,
    const __nv_bfloat16* gamma, float eps,
    const float* rope_cos, const float* rope_sin,
    cudaStream_t stream)
{
    // SMEM: K (int8) + K/64*2 (scales) + 128 (pad) + K*2 (bf16) + 128*2 (output buf)
    size_t smem = K + K/64*2 + 128 + K*2 + 256;
    dim3 grid(num_heads, M);
    dim3 block(128);  // 1 thread per output element
    dp4a_q4_qkv_rope_kernel<128><<<grid, block, smem, stream>>>(
        w, s, x, y, residual_in, M, K, gamma, eps, rope_cos, rope_sin);
    return cudaGetLastError();
}

}  // namespace ops
}  // namespace viper
