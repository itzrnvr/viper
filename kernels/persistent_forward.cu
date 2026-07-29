/*
 * viper persistent forward kernel — entire decode in ONE CUDA launch.
 *
 * Replaces ~400 kernel launches per token with 1 launch + atomic barriers.
 * On Windows WDDM where each launch costs ~25μs, this saves ~10ms/token.
 *
 * DESIGN:
 *   - Persistent grid: 384 blocks (48 SMs × 8 blocks/SM), 256 threads/block
 *   - Each block loops over output channels for GEMV phases
 *   - Atomic barrier between phases (sense-reversing)
 *   - No SMEM (x read from L1 cache) for maximum occupancy
 *
 * BARRIER:
 *   Uses atomicAdd + spin-wait on global memory. All blocks must be
 *   resident (grid size ≤ max_active_blocks). Verified at launch.
 */
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>

namespace viper {
namespace ops {

// ---- Atomic grid barrier (sense-reversing) ----
__device__ unsigned int d_barrier_count = 0;
__device__ volatile unsigned int d_barrier_sense = 0;

__device__ __forceinline__ void grid_sync() {
    __syncthreads();
    __threadfence();
    if (threadIdx.x == 0) {
        unsigned int my_sense = d_barrier_sense;
        unsigned int c = atomicAdd(&d_barrier_count, 1);
        if (c == gridDim.x - 1) {
            d_barrier_count = 0;
            __threadfence_system();
            d_barrier_sense = my_sense + 1;
        } else {
            while (d_barrier_sense <= my_sense) {}
        }
    }
    __syncthreads();
}

// ---- GEMV device function (persistent block pattern) ----
// Each block processes output channels in a stride loop.
// Optional rmsnorm fusion and residual add.
__device__ void gemv_op(
    const uint8_t* __restrict__ w, const __nv_bfloat16* __restrict__ s,
    const __nv_bfloat16* __restrict__ x, __nv_bfloat16* __restrict__ y,
    const __nv_bfloat16* __restrict__ residual,
    const __nv_bfloat16* __restrict__ gamma, float eps,
    int N, int K) {
    const int warp_id = threadIdx.x >> 5;
    const int lane_id = threadIdx.x & 31;
    const int warps_per_block = blockDim.x >> 5;  // 8

    // Each block handles channels: blockIdx.x*8, blockIdx.x*8+gridDim.x*8, ...
    for (int base = blockIdx.x * warps_per_block; base < N; base += gridDim.x * warps_per_block) {
        int n = base + warp_id;
        if (n >= N) continue;

        const uint8_t* w_row = w + (size_t)n * (K / 2);
        const __nv_bfloat16* s_row = s + (size_t)n * (K / 64);
        float acc = 0.0f;
        const int nb = K / 2;

        // Vectorized loop: 4 bytes/thread/iter
        for (int bi = lane_id * 4; bi < nb; bi += 32 * 4) {
            uint32_t p4 = *reinterpret_cast<const uint32_t*>(w_row + bi);
            int w0=(p4&0xF)-8, w1=((p4>>4)&0xF)-8, w2=((p4>>8)&0xF)-8, w3=((p4>>12)&0xF)-8;
            int w4=((p4>>16)&0xF)-8, w5=((p4>>20)&0xF)-8, w6=((p4>>24)&0xF)-8, w7=((p4>>28)&0xF)-8;
            int xk = bi * 2;
            float sc = __bfloat162float(s_row[bi / 32]);
            acc += sc * ((float)w0*__bfloat162float(x[xk]) + (float)w1*__bfloat162float(x[xk+1])
                       + (float)w2*__bfloat162float(x[xk+2]) + (float)w3*__bfloat162float(x[xk+3])
                       + (float)w4*__bfloat162float(x[xk+4]) + (float)w5*__bfloat162float(x[xk+5])
                       + (float)w6*__bfloat162float(x[xk+6]) + (float)w7*__bfloat162float(x[xk+7]));
        }
        // Warp reduce
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            acc += __shfl_xor_sync(0xffffffff, acc, off);
        if (lane_id == 0) {
            if (residual) acc += __bfloat162float(residual[n]);
            y[n] = __float2bfloat16(acc);
        }
    }
}

// ---- RMSNorm device function (block 0 only) ----
__device__ void rmsnorm_op(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ gamma,
    __nv_bfloat16* __restrict__ out,
    int H, float eps) {
    if (blockIdx.x > 0) return;
    const int tid = threadIdx.x;
    // Load and compute sum of squares
    float ss = 0.0f;
    for (int i = tid; i < H; i += blockDim.x) {
        float v = __bfloat162float(x[i]);
        ss += v * v;
    }
    // Warp reduce
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        ss += __shfl_xor_sync(0xffffffff, ss, off);
    // Inter-warp reduce
    __shared__ float ws[8];
    if ((tid & 31) == 0) ws[tid >> 5] = ss;
    __syncthreads();
    float total = 0.0f;
    if (tid < 32) {
        total = (tid < 8) ? ws[tid] : 0.0f;
        #pragma unroll
        for (int off = 4; off > 0; off >>= 1)
            total += __shfl_xor_sync(0xffffffff, total, off);
        if (tid == 0) ws[0] = rsqrtf(total / (float)H + eps);
    }
    __syncthreads();
    float inv = ws[0];
    // Normalize and write
    for (int i = tid; i < H; i += blockDim.x)
        out[i] = __float2bfloat16(__bfloat162float(x[i]) * inv * __bfloat162float(gamma[i]));
}

// ---- SwiGLU device function (first few blocks) ----
__device__ void swiglu_op(
    __nv_bfloat16* __restrict__ gate,
    const __nv_bfloat16* __restrict__ up,
    int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int i = tid; i < N; i += stride) {
        float g = __bfloat162float(gate[i]);
        float u = __bfloat162float(up[i]);
        gate[i] = __float2bfloat16(g / (1.0f + __expf(-g)) * u);
    }
}

// ---- RoPE device function (block 0 only) ----
__device__ void rope_op(
    __nv_bfloat16* __restrict__ q, __nv_bfloat16* __restrict__ k,
    const float* __restrict__ cos_t, const float* __restrict__ sin_t,
    int nQ, int nKVh, int HD) {
    if (blockIdx.x > 0) return;
    const int half = HD / 2;
    const int tid = threadIdx.x;
    // Process Q heads
    for (int h = 0; h < nQ; ++h) {
        for (int d = tid; d < HD; d += blockDim.x) {
            int partner = (d < half) ? (d + half) : (d - half);
            float qv = __bfloat162float(q[h * HD + d]);
            float qp = __bfloat162float(q[h * HD + partner]);
            float c = cos_t[d], s = sin_t[d];
            float rotate = (d < half) ? -qp : qp;
            q[h * HD + d] = __float2bfloat16(qv * c + rotate * s);
        }
    }
    // Process K heads
    for (int h = 0; h < nKVh; ++h) {
        for (int d = tid; d < HD; d += blockDim.x) {
            int partner = (d < half) ? (d + half) : (d - half);
            float kv = __bfloat162float(k[h * HD + d]);
            float kp = __bfloat162float(k[h * HD + partner]);
            float c = cos_t[d], s = sin_t[d];
            float rotate = (d < half) ? -kp : kp;
            k[h * HD + d] = __float2bfloat16(kv * c + rotate * s);
        }
    }
}

// ---- KV append device function (block 0 only) ----
__device__ void kv_append_op(
    const __nv_bfloat16* __restrict__ k_src,
    __nv_bfloat16* __restrict__ k_cache,
    int nKVh, int HD, int pos) {
    if (blockIdx.x > 0) return;
    int total = nKVh * HD;
    for (int i = threadIdx.x; i < total; i += blockDim.x)
        k_cache[(size_t)pos * total + i] = k_src[i];
}

// ---- Attention device function (one block per Q head) ----
__device__ void attn_op(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k_cache,
    const __nv_bfloat16* __restrict__ v_cache,
    __nv_bfloat16* __restrict__ out,
    int nQ, int nKV, int D, int T_ctx, float scale) {
    const int h = blockIdx.x;
    if (h >= nQ) return;
    const int h_kv = (int)((long long)h * nKV / nQ);
    const int tid = threadIdx.x;
    const size_t kv_stride = (size_t)nKV * D;
    const size_t kv_off = (size_t)h_kv * D;

    // Load Q
    __shared__ float q_vec[128];
    if (tid < D) q_vec[tid] = __bfloat162float(q[h * D + tid]);
    __syncthreads();

    // Online softmax with block-wide dot product reduction
    __shared__ float warp_dots[8];
    float m_run = -1e30f, l_run = 0.0f, acc = 0.0f;
    for (int p = 0; p < T_ctx; ++p) {
        const __nv_bfloat16* k_row = k_cache + (size_t)p * kv_stride + kv_off;
        float dot = 0.0f;
        for (int i = tid; i < D; i += blockDim.x)
            dot += q_vec[i] * __bfloat162float(k_row[i]);
        // Warp reduce
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            dot += __shfl_xor_sync(0xffffffff, dot, off);
        // Inter-warp reduce
        if ((tid & 31) == 0) warp_dots[tid >> 5] = dot;
        __syncthreads();
        if (tid < 32) {
            float v = (tid < 8) ? warp_dots[tid] : 0.0f;
            #pragma unroll
            for (int off = 4; off > 0; off >>= 1)
                v += __shfl_xor_sync(0xffffffff, v, off);
            if (tid == 0) warp_dots[0] = v;
        }
        __syncthreads();
        dot = warp_dots[0] * scale;
        // Rescale accumulator
        float m_new = fmaxf(m_run, dot);
        float rescale = __expf(m_run - m_new);
        float w = __expf(dot - m_new);
        acc = acc * rescale;
        l_run = l_run * rescale + w;
        m_run = m_new;
        // Weighted V (each thread handles one dimension)
        const __nv_bfloat16* v_row = v_cache + (size_t)p * kv_stride + kv_off;
        if (tid < D)
            acc += w * __bfloat162float(v_row[tid]);
        __syncthreads();  // ensure all threads finish V before next iteration
    }
    if (tid < D)
        out[h * D + tid] = __float2bfloat16(acc / fmaxf(l_run, 1e-30f));
}

// Namespace continues to end of file — kernel and launch function
// need access to device functions and barrier variables.

// ---- Device-side struct matching host GpuLinearQ4/GpuLayer ----
struct DLinear { const uint8_t* packed; const __nv_bfloat16* scales; int out_f, in_f; };
struct DLayer {
    DLinear q, k, v, o, gate, up, down;
    const __nv_bfloat16* input_ln;
    const __nv_bfloat16* post_ln;
};

// ---- Main persistent decode kernel ----
// ONE launch for the entire forward pass. Replaces ~400 kernel launches.
__global__ __launch_bounds__(256, 6) void persistent_decode_kernel(
    const DLayer* __restrict__ layers,
    int n_layers, int n_loops,
    const __nv_bfloat16* __restrict__ embed,
    const uint8_t* __restrict__ lm_packed,
    const __nv_bfloat16* __restrict__ lm_scales,
    const __nv_bfloat16* __restrict__ final_norm,
    __nv_bfloat16* __restrict__ x,
    __nv_bfloat16* __restrict__ x_norm,
    __nv_bfloat16* __restrict__ q_buf,
    __nv_bfloat16* __restrict__ kb_buf,
    __nv_bfloat16* __restrict__ attn_buf,
    __nv_bfloat16* __restrict__ g_buf,
    __nv_bfloat16* __restrict__ u_buf,
    __nv_bfloat16* __restrict__ logits_buf,
    __nv_bfloat16** __restrict__ kv_k_ptrs,
    __nv_bfloat16** __restrict__ kv_v_ptrs,
    const float* __restrict__ cos_pos,
    const float* __restrict__ sin_pos,
    int H, int I, int nQ, int nKVh, int HD, int vocab, int pos,
    int32_t token,
    float eps, float attn_scale,
    int32_t* out_token) {

    // Reset barrier state (first thread of first block).
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        d_barrier_count = 0;
        d_barrier_sense = 0;
    }
    grid_sync();

    // Phase 0: Embedding gather (block 0 only)
    if (blockIdx.x == 0) {
        for (int i = threadIdx.x; i < H; i += blockDim.x)
            x[i] = embed[(size_t)token * H + i];
    }
    grid_sync();

    float scale = attn_scale;

    for (int loop = 0; loop < n_loops; ++loop) {
        for (int l = 0; l < n_layers; ++l) {
            const DLayer& lw = layers[l];
            int slot = loop * n_layers + l;
            __nv_bfloat16* k_cache = kv_k_ptrs[slot];
            __nv_bfloat16* v_cache = kv_v_ptrs[slot];

            // --- Attention sublayer ---
            rmsnorm_op(x, lw.input_ln, x_norm, H, eps);
            grid_sync();

            gemv_op(lw.q.packed, lw.q.scales, x_norm, q_buf, nullptr, nullptr, 0, lw.q.out_f, lw.q.in_f);
            gemv_op(lw.k.packed, lw.k.scales, x_norm, kb_buf, nullptr, nullptr, 0, lw.k.out_f, lw.k.in_f);
            gemv_op(lw.v.packed, lw.v.scales, x_norm, v_cache + (size_t)pos * nKVh * HD, nullptr, nullptr, 0, lw.v.out_f, lw.v.in_f);
            grid_sync();

            rope_op(q_buf, kb_buf, cos_pos, sin_pos, nQ, nKVh, HD);
            grid_sync();

            kv_append_op(kb_buf, k_cache, nKVh, HD, pos);
            grid_sync();

            attn_op(q_buf, k_cache, v_cache, attn_buf, nQ, nKVh, HD, pos + 1, scale);
            grid_sync();

            gemv_op(lw.o.packed, lw.o.scales, attn_buf, x, x, nullptr, 0, lw.o.out_f, lw.o.in_f);
            grid_sync();

            // --- MLP sublayer ---
            rmsnorm_op(x, lw.post_ln, x_norm, H, eps);
            grid_sync();

            gemv_op(lw.gate.packed, lw.gate.scales, x_norm, g_buf, nullptr, nullptr, 0, lw.gate.out_f, lw.gate.in_f);
            gemv_op(lw.up.packed, lw.up.scales, x_norm, u_buf, nullptr, nullptr, 0, lw.up.out_f, lw.up.in_f);
            grid_sync();

            swiglu_op(g_buf, u_buf, I);
            grid_sync();

            gemv_op(lw.down.packed, lw.down.scales, g_buf, x, x, nullptr, 0, lw.down.out_f, lw.down.in_f);
            grid_sync();
        }
        // Inter-loop norm
        rmsnorm_op(x, final_norm, x, H, eps);
        grid_sync();
    }

    // lm_head
    gemv_op(lm_packed, lm_scales, x, logits_buf, nullptr, nullptr, 0, vocab, H);
    grid_sync();

    // Greedy sampling (block 0 only)
    if (blockIdx.x == 0) {
        __shared__ float s_vals[128];
        __shared__ int s_idxs[128];
        float best_val = -1e30f;
        int best_idx = 0;
        for (int i = threadIdx.x; i < vocab; i += blockDim.x) {
            float v = __bfloat162float(logits_buf[i]);
            if (v > best_val) { best_val = v; best_idx = i; }
        }
        // Block reduce for max
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            float other_v = __shfl_xor_sync(0xffffffff, best_val, off);
            int other_i = __shfl_xor_sync(0xffffffff, best_idx, off);
            if (other_v > best_val) { best_val = other_v; best_idx = other_i; }
        }
        if ((threadIdx.x & 31) == 0) {
            s_vals[threadIdx.x >> 5] = best_val;
            s_idxs[threadIdx.x >> 5] = best_idx;
        }
        __syncthreads();
        if (threadIdx.x < 8) {
            float v = s_vals[threadIdx.x];
            int i = s_idxs[threadIdx.x];
            #pragma unroll
            for (int off = 4; off > 0; off >>= 1) {
                float ov = __shfl_xor_sync(0xffffffff, v, off);
                int oi = __shfl_xor_sync(0xffffffff, i, off);
                if (ov > v) { v = ov; i = oi; }
            }
            if (threadIdx.x == 0) *out_token = i;
        }
    }
}

// Host-side launch wrapper
cudaError_t launch_persistent_decode(
    const void* d_layers, int n_layers, int n_loops,
    const __nv_bfloat16* embed,
    const uint8_t* lm_packed, const __nv_bfloat16* lm_scales,
    const __nv_bfloat16* final_norm,
    __nv_bfloat16* x, __nv_bfloat16* x_norm,
    __nv_bfloat16* q_buf, __nv_bfloat16* kb_buf,
    __nv_bfloat16* attn_buf, __nv_bfloat16* g_buf, __nv_bfloat16* u_buf,
    __nv_bfloat16* logits_buf,
    __nv_bfloat16** kv_k_ptrs, __nv_bfloat16** kv_v_ptrs,
    const float* cos_pos, const float* sin_pos,
    int H, int I, int nQ, int nKVh, int HD, int vocab, int pos,
    int32_t token, float eps, float attn_scale,
    int32_t* out_token,
    int grid_size, cudaStream_t stream) {
    // Query actual max active blocks to avoid deadlock.
    int max_per_sm = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_per_sm,
        persistent_decode_kernel, 256, 0);
    int device_id = 0;
    cudaDeviceProp prop;
    cudaGetDevice(&device_id);
    cudaGetDeviceProperties(&prop, device_id);
    int max_grid = max_per_sm * prop.multiProcessorCount;
    if (grid_size > max_grid) {
        fprintf(stderr, "[persistent] grid %d > max %d (blocks/SM=%d, SMs=%d), clamping\n",
                grid_size, max_grid, max_per_sm, prop.multiProcessorCount);
        grid_size = max_grid;
    }
    void* args[] = {
        (void*)&d_layers, &n_layers, &n_loops, &embed,
        &lm_packed, &lm_scales, &final_norm,
        &x, &x_norm, &q_buf, &kb_buf, &attn_buf, &g_buf, &u_buf, &logits_buf,
        &kv_k_ptrs, &kv_v_ptrs, &cos_pos, &sin_pos,
        &H, &I, &nQ, &nKVh, &HD, &vocab, &pos, &token,
        &eps, &attn_scale, &out_token
    };
    cudaLaunchKernel((const void*)persistent_decode_kernel,
                     dim3(grid_size), dim3(256), args, 0, stream);
    return cudaGetLastError();
}

}  // namespace ops
}  // namespace viper
