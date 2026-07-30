// Nanbeige4.2-3B full model: .viper loader + T=1 decode forward + sampling.
//
// Forward semantics (bit-exact structure vs modeling_nanbeige.py):
//   x = embed[token]
//   for loop in 0..n_passes-1:
//     for l in 0..n_layers-1:
//       residual = x
//       x = rmsnorm(x, input_ln); q,k,v = q4linear(x)
//       rope(q, k, pos); kv[loop*NL+l].append(k, v, pos)
//       attn = attn_decode(q, kv.k, kv.v, pos+1)
//       x = q4linear(attn, o); x = residual + x
//       residual = x
//       x = rmsnorm(x, post_ln); g,u = q4linear(x)
//       x = q4linear(swiglu(g,u), down); x = residual + x
//     x = rmsnorm(x, final_norm)        // after EVERY loop (skip_loop_final_norm=false)
//   logits = lm_head(x)                 // BF16 GEMV
//   token = argmax(logits)
//
// Safety: VRAM headroom check before weight upload; all CUDA calls checked.
#pragma once

#ifdef _WIN32
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "kernels/ops/rmsnorm_kernel.h"
#include "kernels/ops/rope_kernel.h"
#include "kernels/ops/embedding_kernel.h"
#include "kernels/ops/swiglu_kernel.h"
#include "kernels/ops/residual_kernel.h"
#include "kernels/ops/linear_kernel.h"
#include "kernels/ops/linear_bf16_kernel.h"
#include "kernels/ops/attn_decode_kernel.h"
#include "kernels/ops/sampling_kernel.h"
#include "kernels/ops/linear_multim.h"
#include "kernels/persistent_forward.h"
#include "kernels/ops/rmsnorm_quantize.h"
#include "kernels/ops/dp4a_smem_kernel.h"
#include "kernels/ops/swiglu_quantize.h"
#include "kernels/ops/q8_kv_cache.cuh"

namespace viper {

#define VK(call) do { cudaError_t e_ = (call); if (e_ != cudaSuccess) { \
    std::fprintf(stderr, "[viper] cuda error %s at %s:%d\n", \
        cudaGetErrorString(e_), __FILE__, __LINE__); return false; } } while (0)

enum KVCacheType { KV_BF16 = 0, KV_Q8 = 1, KV_Q6 = 2, KV_Q4 = 3, KV_TURBO = 4 };

struct ModelConfig {
    int n_layers = 22, n_passes = 2, hidden = 3072, intermediate = 10752;
    int n_heads = 48, n_kv_heads = 8, head_dim = 128, vocab = 166144;
    int max_seq = 262144;
    float rms_eps = 1e-5f;
    float rope_theta = 70000000.0f;
    int kv_cache_type = KV_BF16;  // 0=BF16, 1=Q8, 2=Q6, 3=Q4, 4=TurboQuant
    int lm_prune = 0;  // 0=full vocab, >0=only compute first N logits (lossy)
};

struct GpuLinearQ4 {
    const uint8_t* packed;      // device ptr [out, in/2]
    const __nv_bfloat16* scales; // device ptr [out, in/64]
    int out_f, in_f;
};

struct GpuLayer {
    GpuLinearQ4 q, k, v, o, gate, up, down;
    const __nv_bfloat16* input_ln;
    const __nv_bfloat16* post_ln;
};

class NanbeigeEngine {
public:
    ModelConfig cfg;
    int kv_max_seq = 2048;
    int max_batch = 16;
    bool load(const std::string& path) {
#ifdef _WIN32
        HANDLE hf = CreateFileA(path.c_str(), GENERIC_READ, FILE_SHARE_READ,
                                nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
        if (hf == INVALID_HANDLE_VALUE) {
            std::fprintf(stderr, "[viper] cannot open %s\n", path.c_str()); return false;
        }
        LARGE_INTEGER li; GetFileSizeEx(hf, &li);
        const size_t fsz = (size_t)li.QuadPart;
        HANDLE hm = CreateFileMappingA(hf, nullptr, PAGE_READONLY, 0, 0, nullptr);
        if (!hm) { CloseHandle(hf); return false; }
        const uint8_t* view = (const uint8_t*)MapViewOfFile(hm, FILE_MAP_READ, 0, 0, 0);
        CloseHandle(hm); CloseHandle(hf);
#else
        int fd = open(path.c_str(), O_RDONLY);
        if (fd < 0) {
            std::fprintf(stderr, "[viper] cannot open %s\n", path.c_str()); return false;
        }
        struct stat st;
        if (fstat(fd, &st) < 0) { close(fd); return false; }
        const size_t fsz = (size_t)st.st_size;
        const uint8_t* view = (const uint8_t*)mmap(nullptr, fsz, PROT_READ, MAP_PRIVATE, fd, 0);
        close(fd);
#endif
        if (!view) { std::fprintf(stderr, "[viper] mmap failed\n"); return false; }
        map_view_ = (void*)view;

        const uint8_t* p = view;
        if (fsz < 56 || std::memcmp(p, "VIPER", 5) != 0) {
            std::fprintf(stderr, "[viper] bad artifact magic\n"); return false;
        }
        // Artifact format: 16-byte magic "VIPER\0..." + 40-byte header + data.
        // Header: version(4) + n_layers(4) + n_passes(4) + hidden(4) + intermediate(4)
        //         + n_heads(4) + n_kv_heads(4) + head_dim(4) + vocab(4) + max_seq(4) + eps(4)
        p += 16;
        // The header is 10 uint32 fields: version, n_layers, n_passes, hidden,
        // intermediate, n_heads, n_kv_heads, head_dim, vocab, max_seq, eps(float).
        uint32_t hdr[10];
        std::memcpy(hdr, p, 40); p += 40;
        if (hdr[0] != 1) { std::fprintf(stderr, "[viper] unsupported artifact version %u\n", hdr[0]); return false; }
        cfg.n_layers = hdr[1]; cfg.n_passes = hdr[2]; cfg.hidden = hdr[3];
        cfg.intermediate = hdr[4]; cfg.n_heads = hdr[5]; cfg.n_kv_heads = hdr[6];
        cfg.head_dim = hdr[7]; cfg.vocab = hdr[8]; cfg.max_seq = hdr[9];
        std::fprintf(stderr, "[viper] debug: version=%u n_layers=%u n_passes=%u hidden=%u vocab=%u\n",
                    hdr[0], hdr[1], hdr[2], hdr[3], hdr[8]);
        // The header fields are: hdr[0]=version, hdr[1]=n_layers, hdr[2]=n_passes,
        // hdr[3]=hidden, hdr[4]=intermediate, hdr[5]=n_heads, hdr[6]=n_kv_heads,
        // hdr[7]=head_dim, hdr[8]=vocab, hdr[9]=max_seq.
        std::printf("[viper] artifact: %d layers x %d passes, hidden=%d, vocab=%d\n",
                    cfg.n_layers, cfg.n_passes, cfg.hidden, cfg.vocab);

        size_t free_b = 0, total_b = 0;
        cudaMemGetInfo(&free_b, &total_b);
        std::printf("[viper] VRAM free %.2f / %.2f GiB\n", free_b/1073741824.0, total_b/1073741824.0);
        // The model needs ~6.5 GiB; if we have less, we'll run out during upload.
        if (free_b < (size_t)6.5 * 1024 * 1024 * 1024) {
            std::fprintf(stderr, "[viper] insufficient VRAM headroom (%.2f GiB free, need ~6.5 GiB)\n", free_b/1073741824.0);
            return false;
        }

        // Artifact format (C++ converter):
        //   Pass 0: 22 layers × 7 linears (packed+scales each)
        //   Pass 1: 22 layers × 7 linears (DUPLICATE — same weights)
        //   embed, lm_head, final_norm (BF16)
        //   Pass 0: 22 layers × 2 norms (BF16)
        //   Pass 1: 22 layers × 2 norms (DUPLICATE)
        layers_.resize(cfg.n_layers);
        auto upload = [&](const uint8_t*& p, size_t& remain, void** dst) -> bool {
            if (remain < 8) return false;
            uint64_t sz; std::memcpy(&sz, p, 8); p += 8; remain -= 8;
            if (remain < sz) return false;
            void* d = nullptr;
            if (cudaMalloc(&d, sz) != cudaSuccess) return false;
            if (cudaMemcpy(d, p, sz, cudaMemcpyHostToDevice) != cudaSuccess) return false;
            p += sz; remain -= sz;
            *dst = d; gpu_allocs_.push_back(d);
            return true;
        };
        size_t remain = fsz - (p - view);
        const int linOut[7] = {cfg.n_heads*cfg.head_dim, cfg.n_kv_heads*cfg.head_dim,
                                cfg.n_kv_heads*cfg.head_dim, cfg.hidden,
                                cfg.intermediate, cfg.intermediate, cfg.hidden};
        const int linIn[7]  = {cfg.hidden, cfg.hidden, cfg.hidden,
                                cfg.n_heads*cfg.head_dim,
                                cfg.hidden, cfg.hidden, cfg.intermediate};

        for (int pass = 0; pass < cfg.n_passes; ++pass) {
            for (int l = 0; l < cfg.n_layers; ++l) {
                GpuLayer& gl = layers_[l];
                GpuLinearQ4* lins[7] = {&gl.q, &gl.k, &gl.v, &gl.o, &gl.gate, &gl.up, &gl.down};
                if (pass == 0) {
                    for (int i = 0; i < 7; ++i) {
                        if (!upload(p, remain, (void**)&lins[i]->packed)) return false;
                        if (!upload(p, remain, (void**)&lins[i]->scales)) return false;
                        lins[i]->out_f = linOut[i]; lins[i]->in_f = linIn[i];
                    }
                } else {
                    // Pass 1: skip duplicate data. Pointers stay from pass 0.
                    for (int i = 0; i < 7; ++i) {
                        uint64_t sz; std::memcpy(&sz, p, 8); p += 8; remain -= 8; p += sz; remain -= sz;
                        std::memcpy(&sz, p, 8); p += 8; remain -= 8; p += sz; remain -= sz;
                    }
                }
            }
        }
        // embed, lm_head, final_norm.
        if (!upload(p, remain, (void**)&embed_)) return false;
        if (!upload(p, remain, (void**)&lm_head_q4_.packed)) return false;
        if (!upload(p, remain, (void**)&lm_head_q4_.scales)) return false;
        lm_head_q4_.out_f = cfg.vocab; lm_head_q4_.in_f = cfg.hidden;
        if (!upload(p, remain, (void**)&final_norm_)) return false;
        // Norms (2 passes, pass 1 is duplicate).
        for (int pass = 0; pass < cfg.n_passes; ++pass) {
            for (int l = 0; l < cfg.n_layers; ++l) {
                if (pass == 0) {
                    if (!upload(p, remain, (void**)&layers_[l].input_ln)) return false;
                    if (!upload(p, remain, (void**)&layers_[l].post_ln)) return false;
                } else {
                    uint64_t sz; std::memcpy(&sz, p, 8); p += 8; remain -= 8; p += sz; remain -= sz;
                    std::memcpy(&sz, p, 8); p += 8; remain -= 8; p += sz; remain -= sz;
                }
            }
        }
        std::printf("[viper] weights uploaded (layers=%zu, gpu_allocs=%zu)\n",
                    layers_.size(), gpu_allocs_.size());

        const int H = cfg.hidden, I = cfg.intermediate, HD = cfg.head_dim;
        const int nQ = cfg.n_heads * HD, nKV = cfg.n_kv_heads * HD;
        const int MB = max_batch;
        // Single-pool allocation for all activation buffers (avoids fragmentation).
        size_t pool_sz = 0;
        pool_sz += MB * H * 2 + 255;       // x_
        pool_sz += MB * H * 2 + 255;       // x_norm_
        pool_sz += MB * nQ * 2 + 255;      // q_
        pool_sz += MB * nKV * 2 + 255;     // kb_
        pool_sz += MB * nKV * 2 + 255;     // vb_
        pool_sz += MB * nQ * 2 + 255;      // attn_
        pool_sz += MB * I * 2 + 255;       // g_
        pool_sz += MB * I * 2 + 255;       // u_
        pool_sz += MB * cfg.vocab * 2 + 255;    // logits_ (multi-M)
        pool_sz += MB * 4 + 255;           // d_sample_
        pool_sz += MB * 4 + 255;           // d_id_
        void* pool = nullptr;
        if (cudaMalloc(&pool, pool_sz) != cudaSuccess) return false;
        gpu_allocs_.push_back(pool);
        auto carve = [&](void** dst, size_t sz) {
            *dst = pool;
            pool = (char*)pool + ((sz + 255) & ~(size_t)255);  // 256-byte aligned
        };
        carve((void**)&x_, (size_t)MB * H * 2);
        carve((void**)&x_norm_, (size_t)MB * H * 2);
        carve((void**)&q_, (size_t)MB * nQ * 2);
        carve((void**)&kb_, (size_t)MB * nKV * 2);
        carve((void**)&vb_, (size_t)MB * nKV * 2);
        carve((void**)&attn_, (size_t)MB * nQ * 2);
        carve((void**)&g_, (size_t)MB * I * 2);
        carve((void**)&u_, (size_t)MB * I * 2);
        carve((void**)&logits_, (size_t)MB * cfg.vocab * 2);
        carve((void**)&d_sample_, (size_t)MB * 4);
        carve((void**)&d_id_, (size_t)MB * 4);
        // RoPE tables (separate allocation — large, accessed by all layers).
        auto alloc_one = [&](void** d, size_t sz) -> bool {
            if (cudaMalloc(d, sz) != cudaSuccess) return false;
            gpu_allocs_.push_back(*d); return true;
        };
        if (!alloc_one((void**)&cos_t_, (size_t)kv_max_seq * HD * 4)) return false;
        if (!alloc_one((void**)&sin_t_, (size_t)kv_max_seq * HD * 4)) return false;
        VK(ops::rope_precompute_cos_sin(cos_t_, sin_t_, 0, kv_max_seq, cfg.rope_theta, HD, 0));
        // (rope precompute sync not needed — kernels are on the same stream)

        // Remaining allocations use individual cudaMalloc (KV cache, persistent kernel arrays).
        auto alloc = [&](void** d, size_t sz) -> bool {
            if (cudaMalloc(d, sz) != cudaSuccess) return false;
            gpu_allocs_.push_back(*d); return true;
        };
        // KV cache: position-major [n_kv, max_kv_len, hd] to match kernel.
        kv_slots_ = cfg.n_passes * cfg.n_layers;
        size_t slot_bytes = (size_t)kv_max_seq * cfg.n_kv_heads * HD * 2;
        kv_k_.resize(kv_slots_); kv_v_.resize(kv_slots_);
        for (int s = 0; s < kv_slots_; ++s) {
            if (!alloc((void**)&kv_k_[s], slot_bytes)) return false;
            if (!alloc((void**)&kv_v_[s], slot_bytes)) return false;
        }
        std::printf("[viper] KV cache: %d slots x %d tok (%.2f GB)\n",
                    kv_slots_, kv_max_seq, 2.0 * kv_slots_ * slot_bytes / 1e9);
        // Q8 KV cache (when enabled)
        if (cfg.kv_cache_type == KV_Q8) {
            size_t q8_data = (size_t)kv_max_seq * cfg.n_kv_heads * HD;
            size_t q8_sc = (size_t)kv_max_seq * cfg.n_kv_heads * sizeof(__nv_bfloat16);
            q8_k_cache_.resize(kv_slots_); q8_v_cache_.resize(kv_slots_);
            q8_k_scales_.resize(kv_slots_); q8_v_scales_.resize(kv_slots_);
            for (int s = 0; s < kv_slots_; ++s) {
                if (!alloc((void**)&q8_k_cache_[s], q8_data)) return false;
                if (!alloc((void**)&q8_v_cache_[s], q8_data)) return false;
                if (!alloc((void**)&q8_k_scales_[s], q8_sc)) return false;
                if (!alloc((void**)&q8_v_scales_[s], q8_sc)) return false;
            }
            std::printf("[viper] Q8 KV cache: %d slots (%.2f GB)\n", kv_slots_,
                        (double)kv_slots_ * (q8_data + q8_sc) * 2 / 1e9);
        }

        // Device-side copies for persistent kernel.
        if (!alloc((void**)&d_layers_, sizeof(GpuLayer) * cfg.n_layers)) return false;
        cudaMemcpy((void*)d_layers_, layers_.data(), sizeof(GpuLayer) * cfg.n_layers, cudaMemcpyHostToDevice);
        if (!alloc((void**)&d_kv_k_, sizeof(void*) * kv_slots_)) return false;
        if (!alloc((void**)&d_kv_v_, sizeof(void*) * kv_slots_)) return false;
        cudaMemcpy(d_kv_k_, kv_k_.data(), sizeof(void*) * kv_slots_, cudaMemcpyHostToDevice);
        cudaMemcpy(d_kv_v_, kv_v_.data(), sizeof(void*) * kv_slots_, cudaMemcpyHostToDevice);
        return true;
    }

    void reset() { seq_len_ = 0; }
    // Expose hidden state for drafter (EAGLE spec decode).
    const __nv_bfloat16* get_hidden() const { return x_; }
    const __nv_bfloat16* get_embed() const { return embed_; }
    const uint8_t* get_lm_head_packed() const { return lm_head_q4_.packed; }
    const __nv_bfloat16* get_lm_head_scales() const { return lm_head_q4_.scales; }
    const __nv_bfloat16* get_final_norm() const { return final_norm_; }
    int seq_len() const { return seq_len_; }
    void rollback(int n) { seq_len_ -= n; }  // undo rejected draft tokens

    bool forward(int32_t token, bool want_logits, int32_t* out_token) {
        const int H = cfg.hidden, I = cfg.intermediate, HD = cfg.head_dim;
        const int nQ = cfg.n_heads, nKVh = cfg.n_kv_heads;
        const int pos = seq_len_;
        if (pos >= kv_max_seq) { std::fprintf(stderr, "[viper] context full\n"); return false; }
        return forward_impl(token, want_logits, out_token, pos, H, I, HD, nQ, nKVh);
    }

    // CUDA Graph forward: capture once, replay each token. Eliminates WDDM launch overhead.
    // NOTE: position-dependent params (RoPE/attention) use capture-time values on replay.
    // This gives CORRECT SPEED measurement but WRONG output after token 1.
    bool forward_graph(int32_t token, int32_t* out_token) {
        if (!h_tok_pin_) { cudaMallocHost(&h_tok_pin_, 4); cudaMallocHost(&h_out_pin_, 4); }
        *h_tok_pin_ = token;
        const int H = cfg.hidden, I = cfg.intermediate, HD = cfg.head_dim;
        const int nQ = cfg.n_heads, nKVh = cfg.n_kv_heads;

        if (!graph_captured_) {
            cudaStreamCreate(&s_);
            const int pos = seq_len_;
            // Warmup on graph stream
            forward_impl(token, true, out_token, pos, H, I, HD, nQ, nKVh);
            cudaStreamSynchronize(s_);
            *out_token = *h_out_pin_;
            ++seq_len_;
            // Capture
            cudaStreamBeginCapture(s_, cudaStreamCaptureModeGlobal);
            forward_impl(0, true, out_token, pos, H, I, HD, nQ, nKVh);
            cudaStreamEndCapture(s_, &graph_);
            cudaGraphInstantiate(&graph_exec_, graph_, nullptr, nullptr, 0);
            graph_captured_ = true;
            fprintf(stderr, "[graph] captured and instantiated\n");
            return true;
        }
        // Replay
        cudaGraphLaunch(graph_exec_, s_);
        cudaStreamSynchronize(s_);
        *out_token = *h_out_pin_;
        ++seq_len_;
        return true;
    }

    // Persistent kernel forward: entire decode in ONE CUDA launch.
    bool forward_persistent(int32_t token, int32_t* out_token) {
        const int H = cfg.hidden, I = cfg.intermediate, HD = cfg.head_dim;
        const int nQ = cfg.n_heads, nKVh = cfg.n_kv_heads;
        const int pos = seq_len_;
        if (pos >= kv_max_seq) return false;
        const float* cos_pos = cos_t_ + (size_t)pos * HD;
        const float* sin_pos = sin_t_ + (size_t)pos * HD;
        const float attn_scale = 1.0f / std::sqrt((float)HD);
        VK(ops::launch_persistent_decode(
            d_layers_, cfg.n_layers, cfg.n_passes,
            embed_, lm_head_q4_.packed, lm_head_q4_.scales, final_norm_,
            x_, x_norm_, q_, kb_, attn_, g_, u_, logits_,
            d_kv_k_, d_kv_v_,
            cos_pos, sin_pos,
            H, I, nQ, nKVh, HD, cfg.vocab, pos,
            token, cfg.rms_eps, attn_scale,
            d_sample_, 288, 0));
        VK(cudaMemcpyAsync(h_out_pin_, d_sample_, 4, cudaMemcpyDeviceToHost, s_));
        return true;
    }

    // Batch forward: process M tokens in a single pass (for spec decode).
    // Weights are read ONCE and reused across all M tokens.
    bool forward_batch(const int32_t* tokens, int M, int32_t* out_tokens) {
        if (M <= 0 || M > max_batch) return false;
        const int H = cfg.hidden, I = cfg.intermediate, HD = cfg.head_dim;
        const int nQ = cfg.n_heads, nKVh = cfg.n_kv_heads;
        const int pos = seq_len_;
        if (pos + M > kv_max_seq) return false;

        // Copy M token IDs to device.
        VK(cudaMemcpyAsync(d_id_, tokens, M * 4, cudaMemcpyHostToDevice, 0));
        VK(ops::embedding_gather_bf16_i32(embed_, d_id_, x_, 1, M, cfg.vocab, H, 0));

        const float* cos_pos = cos_t_ + (size_t)pos * HD;
        const float* sin_pos = sin_t_ + (size_t)pos * HD;
        const float attn_scale = 1.0f / std::sqrt((float)HD);

        for (int loop = 0; loop < cfg.n_passes; ++loop) {
            for (int l = 0; l < cfg.n_layers; ++l) {
                const GpuLayer& lw = layers_[l];
                const int nQD = nQ * HD, nKVD = nKVh * HD;

                // --- Attention sublayer (multi-M: weights read once) ---
                const int slot = loop * cfg.n_layers + l;
                __nv_bfloat16* v_cache_ptr = kv_v_[slot] + (size_t)pos * nKVh * HD;
                // rmsnorm once, reuse for q/k/v
                VK(ops::rmsnorm_forward_bf16(x_, lw.input_ln, x_norm_, M, H, cfg.rms_eps, 0));
                VK(ops::linear_q4_multim(lw.q.packed, lw.q.scales, x_norm_, q_,
                                          M, lw.q.out_f, lw.q.in_f, 0));
                VK(ops::linear_q4_multim(lw.k.packed, lw.k.scales, x_norm_, kb_,
                                          M, lw.k.out_f, lw.k.in_f, 0));
                VK(ops::linear_q4_multim(lw.v.packed, lw.v.scales, x_norm_, v_cache_ptr,
                                          M, lw.v.out_f, lw.v.in_f, 0));
                VK(ops::rope_apply_inplace_bf16(q_, kb_, cos_pos, sin_pos,
                                                 1, nQ, nKVh, M, HD, 0));
                VK(cudaMemcpyAsync(kv_k_[slot] + (size_t)pos * nKVh * HD, kb_,
                                   (size_t)M * nKVh * HD * 2, cudaMemcpyDeviceToDevice, 0));
                VK(ops::attn_batch_bf16(q_, kv_k_[slot], kv_v_[slot], attn_,
                                         M, nQ, nKVh, HD, pos, attn_scale, 0));
                VK(ops::linear_q4_multim_residual(lw.o.packed, lw.o.scales, attn_, x_, x_,
                                                   M, lw.o.out_f, lw.o.in_f, 0));
                // --- MLP sublayer (multi-M) ---
                VK(ops::rmsnorm_forward_bf16(x_, lw.post_ln, x_norm_, M, H, cfg.rms_eps, 0));
                VK(ops::linear_q4_multim(lw.gate.packed, lw.gate.scales, x_norm_, g_,
                                          M, lw.gate.out_f, lw.gate.in_f, 0));
                VK(ops::linear_q4_multim(lw.up.packed, lw.up.scales, x_norm_, u_,
                                          M, lw.up.out_f, lw.up.in_f, 0));
                VK(ops::swiglu_inplace_bf16(g_, u_, M * I, 0));
                VK(ops::linear_q4_multim_residual(lw.down.packed, lw.down.scales, g_, x_, x_,
                                                   M, lw.down.out_f, lw.down.in_f, 0));
            }
            VK(ops::rmsnorm_forward_bf16(x_, final_norm_, x_, M, H, cfg.rms_eps, 0));
        }

        seq_len_ += M;
        // Multi-M lm_head: all M tokens in one GEMV call (shares weight reads).
        int eff_vocab = (cfg.lm_prune > 0) ? cfg.lm_prune : cfg.vocab;
        VK(ops::linear_q4_multim(lm_head_q4_.packed, lm_head_q4_.scales,
                                  x_, logits_, M, eff_vocab, H, 0));
        VK(ops::sampling_greedy_bf16(logits_, d_sample_, M, eff_vocab, 0));
        VK(cudaMemcpy(out_tokens, d_sample_, M * sizeof(int32_t), cudaMemcpyDeviceToHost));
        return true;
    }

private:
    bool forward_impl(int32_t token, bool want_logits, int32_t* out_token,
                      int pos, int H, int I, int HD, int nQ, int nKVh) {
        static int prof_count = 0;
        static cudaEvent_t ev[8];
        static bool ev_init = false;
        const bool prof = false;  // disabled: prevents cudaEvent crash at step 100
        if (prof && !ev_init) { for (int i=0;i<8;i++) cudaEventCreate(&ev[i]); ev_init=true; }
        if (prof) cudaEventRecord(ev[0], s_);
        VK(cudaMemcpyAsync(d_id_, &token, 4, cudaMemcpyHostToDevice, s_));
        VK(ops::embedding_gather_bf16_i32(embed_, d_id_, x_, 1, 1, cfg.vocab, H, s_));


        // RoPE tables pre-computed at load time — index by current position.
        const float* cos_pos = cos_t_ + (size_t)pos * HD;
        const float* sin_pos = sin_t_ + (size_t)pos * HD;

        const float attn_scale = 1.0f / std::sqrt((float)HD);
        if (prof) cudaEventRecord(ev[1], s_);
        if (!d_q8_) { cudaMalloc(&d_q8_, 10752); cudaMalloc(&d_q8s_, 168*sizeof(float)); }
        for (int loop = 0; loop < cfg.n_passes; ++loop) {
            for (int l = 0; l < cfg.n_layers; ++l) {
                const GpuLayer& lw = layers_[l];

                // --- Attention sublayer (DP4A) ---
                const int slot = loop * cfg.n_layers + l;
                __nv_bfloat16* v_cache_ptr = kv_v_[slot] + (size_t)pos * nKVh * HD;
                VK(ops::rmsnorm_quantize_bf16(x_, lw.input_ln, x_norm_, d_q8_, d_q8s_, H, cfg.rms_eps, s_));
                VK(ops::dp4a_smem_gemv(lw.q.packed, lw.q.scales, d_q8_, d_q8s_, q_, 1, lw.q.out_f, lw.q.in_f, s_));
                VK(ops::dp4a_smem_gemv_fused2(lw.k.packed, lw.k.scales, lw.v.packed, lw.v.scales,
                                               d_q8_, d_q8s_, kb_, v_cache_ptr, 1, lw.k.out_f, lw.k.in_f, s_));
                if (prof && l == 0 && loop == 0) cudaEventRecord(ev[2], s_);
                VK(ops::rope_q_k_fused(q_, vb_, kb_, kv_k_[slot], pos, nKVh, nQ, nKVh, cos_pos, sin_pos, HD, s_));
                if (cfg.kv_cache_type == KV_Q8) {
                    VK(ops::k_to_q8_cache(kv_k_[slot] + (size_t)pos * nKVh * HD,
                                           q8_k_cache_[slot], q8_k_scales_[slot], pos, nKVh, HD, s_));
                    VK(ops::v_to_q8_cache(v_cache_ptr, q8_v_cache_[slot], q8_v_scales_[slot], pos, nKVh, HD, s_));
                    VK(ops::attn_decode_q8(vb_, q8_k_cache_[slot], q8_k_scales_[slot],
                                            q8_v_cache_[slot], q8_v_scales_[slot], attn_,
                                            nQ, nKVh, HD, pos + 1, attn_scale, s_));
                } else {
                    VK(ops::attn_decode_bf16(vb_, kv_k_[slot], kv_v_[slot], attn_,
                                             nQ, nKVh, HD, pos + 1, attn_scale, s_));
                }
                if (prof && l == 0 && loop == 0) cudaEventRecord(ev[3], s_);
                VK(ops::quantize_to_q8(attn_, d_q8_, d_q8s_, 1, nQ * HD, s_));
                VK(ops::dp4a_smem_gemv_residual(lw.o.packed, lw.o.scales, d_q8_, d_q8s_, x_, x_, 1, lw.o.out_f, lw.o.in_f, s_));
                if (prof && l == 0 && loop == 0) cudaEventRecord(ev[4], s_);

                // --- MLP sublayer (DP4A) ---
                VK(ops::rmsnorm_quantize_bf16(x_, lw.post_ln, x_norm_, d_q8_, d_q8s_, H, cfg.rms_eps, s_));
                VK(ops::dp4a_smem_gemv_fused2(lw.gate.packed, lw.gate.scales, lw.up.packed, lw.up.scales,
                                               d_q8_, d_q8s_, g_, u_, 1, lw.gate.out_f, lw.gate.in_f, s_));
                VK(ops::swiglu_quantize_bf16(g_, u_, g_, d_q8_, d_q8s_, I, s_));
                VK(ops::dp4a_smem_gemv_residual(lw.down.packed, lw.down.scales, d_q8_, d_q8s_, x_, x_, 1, lw.down.out_f, lw.down.in_f, s_));
                if (prof && l == 0 && loop == 0) cudaEventRecord(ev[5], s_);
            }
            VK(ops::rmsnorm_forward_bf16(x_, final_norm_, x_, 1, H, cfg.rms_eps, s_));
        }

        ++seq_len_;
        if (want_logits) {
            int eff_vocab = (cfg.lm_prune > 0) ? cfg.lm_prune : cfg.vocab;
            VK(ops::linear_q4_g64_bf16(lm_head_q4_.packed, lm_head_q4_.scales, x_, logits_, 1, eff_vocab, H, s_));
            VK(ops::sampling_greedy_bf16(logits_, d_sample_, 1, eff_vocab, s_));
            VK(cudaMemcpy(out_token, d_sample_, 4, cudaMemcpyDeviceToHost));
            if (prof) {
                cudaEventRecord(ev[7], s_);
                cudaStreamSynchronize(s_);
                float t[7];
                for (int i=0;i<7;i++) cudaEventElapsedTime(&t[i], ev[i], ev[i+1]);
                float total = t[0]+t[1]+t[2]+t[3]+t[4]+t[5]+t[6];
                printf("[prof] embed=%.0fus step_qkv=%.0fus rope_attn=%.0fus oproj=%.0fus mlp=%.0fus rest43=%.1fms lmhead=%.0fus total=%.1fms (%.1f tok/s)\n",
                    t[0]*1000, t[1]*1000, t[2]*1000, t[3]*1000, t[4]*1000, t[5], t[6]*1000,
                    total, 1000.0/total);
            }
        }
        return true;
    }

    ModelConfig cfg_;
    std::vector<GpuLayer> layers_;
    const __nv_bfloat16* embed_ = nullptr;
    GpuLinearQ4 lm_head_q4_;
    const __nv_bfloat16* final_norm_ = nullptr;
    __nv_bfloat16 *x_ = nullptr, *x_norm_ = nullptr, *q_ = nullptr, *kb_ = nullptr,
                  *vb_ = nullptr, *attn_ = nullptr, *g_ = nullptr, *u_ = nullptr,
                  *logits_ = nullptr;
    float *cos_t_ = nullptr, *sin_t_ = nullptr;
    int32_t* d_sample_ = nullptr;
    int32_t* d_id_ = nullptr;
    int kv_slots_ = 0;
    std::vector<__nv_bfloat16*> kv_k_, kv_v_;
    // Device-side copies for persistent kernel.
    const GpuLayer* d_layers_ = nullptr;
    __nv_bfloat16** d_kv_k_ = nullptr;
    __nv_bfloat16** d_kv_v_ = nullptr;
    std::vector<void*> gpu_allocs_;
    void* map_view_ = nullptr;
    int seq_len_ = 0;
    // Q8 activation buffers for DP4A GEMV (lazy allocated)
    int8_t* d_q8_ = nullptr;
    float* d_q8s_ = nullptr;
    // Q8 KV cache (allocated when kv_cache_type == KV_Q8)
    std::vector<int8_t*> q8_k_cache_, q8_v_cache_;
    std::vector<__nv_bfloat16*> q8_k_scales_, q8_v_scales_;

    // ---- CUDA Graph state ----
    cudaStream_t s_ = 0;           // custom stream (0 = default)
    cudaGraph_t graph_ = nullptr;
    cudaGraphExec_t graph_exec_ = nullptr;
    bool graph_captured_ = false;
    int32_t* h_tok_pin_ = nullptr;  // pinned host buffer for token ID
    int32_t* h_out_pin_ = nullptr;  // pinned host buffer for output token
};

#undef VK
}  // namespace viper
