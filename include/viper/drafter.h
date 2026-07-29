/*
 * viper Drafter — 1-layer speculative drafting model for EAGLE-style spec decode.
 *
 * The drafter takes the base model's hidden state and autoregressively
 * generates K draft tokens. Each forward pass is ~1ms (1 layer vs 44).
 *
 * Expected acceptance: 70-85% with trained weights.
 * Combined with multi-M batch verify: ~140-260 tok/s effective.
 *
 * Usage:
 *   Drafter drafter; drafter.load("drafter.viper");
 *   int K = drafter.generate(hidden, token, drafts, 4);
 */
#ifndef VIPER_DRAFTER_H
#define VIPER_DRAFTER_H

#include "viper/model_impl.cuh"

#define DVK(call) do { cudaError_t e_ = (call); if (e_ != cudaSuccess) { \
    std::fprintf(stderr, "[drafter] cuda error %s at %s:%d\n", \
        cudaGetErrorString(e_), __FILE__, __LINE__); return 0; } } while (0)

namespace viper {

class Drafter {
public:
    ModelConfig cfg;
    int kv_max_seq = 2048;

    // Drafter weights (1 layer only)
    ~Drafter() { for (void* p : gpu_allocs_) cudaFree(p); }
    GpuLayer layer;
    const __nv_bfloat16* embed = nullptr;
    GpuLinearQ4 lm_head;
    const __nv_bfloat16* final_norm = nullptr;

    // Buffers
    __nv_bfloat16 *h_ = nullptr, *h_norm_ = nullptr, *q_ = nullptr, *kb_ = nullptr;
    __nv_bfloat16 *attn_ = nullptr, *g_ = nullptr, *u_ = nullptr, *logits_ = nullptr;
    int32_t* d_sample_ = nullptr;
    int32_t* d_id_ = nullptr;
    float *cos_t_ = nullptr, *sin_t_ = nullptr;
    std::vector<__nv_bfloat16*> kv_k_, kv_v_;
    int seq_len_ = 0;
    std::vector<void*> gpu_allocs_;
    void* map_view_ = nullptr;

    bool load(const std::string& path) {
        // Same .viper format as base model, but n_layers=1, n_passes=1
#ifdef _WIN32
        HANDLE hf = CreateFileA(path.c_str(), GENERIC_READ, FILE_SHARE_READ,
                                nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
        if (hf == INVALID_HANDLE_VALUE) return false;
        LARGE_INTEGER li; GetFileSizeEx(hf, &li);
        size_t fsz = li.QuadPart;
        HANDLE hm = CreateFileMappingA(hf, nullptr, PAGE_READONLY, 0, 0, nullptr);
        if (!hm) { CloseHandle(hf); return false; }
        const uint8_t* view = (const uint8_t*)MapViewOfFile(hm, FILE_MAP_READ, 0, 0, 0);
        CloseHandle(hm); CloseHandle(hf);
#else
        int fd = open(path.c_str(), O_RDONLY);
        if (fd < 0) return false;
        struct stat st; fstat(fd, &st); size_t fsz = st.st_size;
        const uint8_t* view = (const uint8_t*)mmap(nullptr, fsz, PROT_READ, MAP_PRIVATE, fd, 0);
        close(fd);
#endif
        if (!view) return false;
        map_view_ = (void*)view;

        const uint8_t* p = view;
        if (fsz < 56 || memcmp(p, "VIPER", 5) != 0) { fprintf(stderr, "[drafter] bad magic\n"); return false; }
        p += 16;
        uint32_t hdr[10]; memcpy(hdr, p, 40); p += 40;
        cfg.n_layers = hdr[1]; cfg.n_passes = hdr[2];
        cfg.hidden = hdr[3]; cfg.intermediate = hdr[4];
        cfg.n_heads = hdr[5]; cfg.n_kv_heads = hdr[6];
        cfg.head_dim = hdr[7]; cfg.vocab = hdr[8];
        cfg.rms_eps = 1e-5f; cfg.rope_theta = 70000000.0f;

        auto upload = [&](const uint8_t*& pp, void** dst) -> bool {
        fprintf(stderr, "[drafter] hdr: layers=%d passes=%d hidden=%d hd=%d vocab=%d fsz=%zu\n",
                cfg.n_layers, cfg.n_passes, cfg.hidden, cfg.head_dim, cfg.vocab, fsz);
            uint64_t sz; memcpy(&sz, pp, 8); pp += 8;
            void* d = nullptr;
            if (cudaMalloc(&d, sz) != cudaSuccess) return false;
            if (cudaMemcpy(d, pp, sz, cudaMemcpyHostToDevice) != cudaSuccess) return false;
            pp += sz; *dst = d; gpu_allocs_.push_back(d); return true;
        };

        // 1 layer × 7 linears
        const int linOut[7] = {cfg.n_heads*cfg.head_dim, cfg.n_kv_heads*cfg.head_dim,
                               cfg.n_kv_heads*cfg.head_dim, cfg.hidden,
                               cfg.intermediate, cfg.intermediate, cfg.hidden};
        const int linIn[7] = {cfg.hidden, cfg.hidden, cfg.hidden,
                              cfg.n_heads*cfg.head_dim, cfg.hidden, cfg.hidden, cfg.intermediate};
        GpuLinearQ4* lins[7] = {&layer.q, &layer.k, &layer.v, &layer.o,
                                 &layer.gate, &layer.up, &layer.down};
        for (int i = 0; i < 7; ++i) {
            if (!upload(p, (void**)&lins[i]->packed)) return false;
            if (!upload(p, (void**)&lins[i]->scales)) return false;
            lins[i]->out_f = linOut[i]; lins[i]->in_f = linIn[i];
        }
        // SKIP embed, lm_head, final_norm — shared with base model via set_shared()
        // Just advance the file pointer past these sections
        for (int skip = 0; skip < 4; ++skip) {  // embed, lm_packed, lm_scales, final_norm
            uint64_t sz; memcpy(&sz, p, 8); p += 8 + sz;
        }
        // Norms
        if (!upload(p, (void**)&layer.input_ln)) return false;
        if (!upload(p, (void**)&layer.post_ln)) return false;

        // Allocate buffers
        const int H = cfg.hidden, I = cfg.intermediate, HD = cfg.head_dim;
        const int nQ = cfg.n_heads, nKVh = cfg.n_kv_heads;
        auto alloc = [&](void** d, size_t sz) -> bool {
            if (cudaMalloc(d, sz) != cudaSuccess) return false;
            gpu_allocs_.push_back(*d); return true;
        };
        alloc((void**)&h_, H * 2);
        alloc((void**)&h_norm_, H * 2);
        alloc((void**)&q_, nQ * HD * 2);
        alloc((void**)&kb_, nKVh * HD * 2);
        alloc((void**)&attn_, nQ * HD * 2);
        alloc((void**)&g_, I * 2);
        alloc((void**)&u_, I * 2);
        alloc((void**)&logits_, cfg.vocab * 2);
        alloc((void**)&d_sample_, 4);
        alloc((void**)&d_id_, 4);

        // RoPE tables
        alloc((void**)&cos_t_, kv_max_seq * HD * 4);
        alloc((void**)&sin_t_, kv_max_seq * HD * 4);
        ops::rope_precompute_cos_sin(cos_t_, sin_t_, 0, kv_max_seq, cfg.rope_theta, HD, 0);

        // KV cache (1 slot for the 1-layer drafter)
        kv_k_.resize(1); kv_v_.resize(1);
        size_t slot_bytes = (size_t)kv_max_seq * nKVh * HD * 2;
        alloc((void**)&kv_k_[0], slot_bytes);
        alloc((void**)&kv_v_[0], slot_bytes);

        printf("[viper] drafter loaded: %d layer, hidden=%d, vocab=%d\n",
               cfg.n_layers, cfg.hidden, cfg.vocab);
        return true;
    }

    // Share embed/lm_head/final_norm with the base model (saves 1.3 GB VRAM)
    void set_shared(const __nv_bfloat16* embed_ptr,
                    const uint8_t* lm_packed, const __nv_bfloat16* lm_scales,
                    int lm_out, int lm_in,
                    const __nv_bfloat16* fnorm) {
        embed = embed_ptr;
        lm_head.packed = lm_packed;
        lm_head.scales = lm_scales;
        lm_head.out_f = lm_out;
        lm_head.in_f = lm_in;
        final_norm = fnorm;
    }

    void reset() { seq_len_ = 0; }
    void rollback(int n) { seq_len_ -= n; }

    // Generate K draft tokens from base model's hidden state.
    // base_hidden: device pointer to base model's final hidden state [H]
    // base_token: the token that produced base_hidden
    // out_tokens: host array for K draft tokens
    // Returns: number of drafts generated
    int generate(const __nv_bfloat16* base_hidden, int32_t base_token,
                 int32_t* out_tokens, int max_k) {
        const int H = cfg.hidden, HD = cfg.head_dim;
        const int nQ = cfg.n_heads, nKVh = cfg.n_kv_heads;

        // Initialize hidden state from base model
        cudaMemcpy(h_, base_hidden, H * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice);

        // Process the base token at the current position
        int32_t current_token = base_token;
        int pos = seq_len_;

        for (int k = 0; k < max_k; ++k) {
            // Embed current token and add to hidden state
            cudaMemcpy(d_id_, &current_token, 4, cudaMemcpyHostToDevice);
            // h_ = h_ + embed[current_token]
            // (simplified: just use embed directly as input)
            // EAGLE input: h_ = h_ + embed(token)
            cudaMemcpy(d_id_, &current_token, 4, cudaMemcpyHostToDevice);
            DVK(ops::embedding_gather_bf16_i32(embed, d_id_, h_norm_, 1, 1, cfg.vocab, H, 0));
            DVK(ops::residual_add_inplace_bf16(h_, h_norm_, H, 0));

            // rmsnorm on the combined input
            DVK(ops::rmsnorm_forward_bf16(h_, layer.input_ln, h_norm_, 1, H, cfg.rms_eps, 0));

            // q_proj
            DVK(ops::linear_q4_g64_bf16(layer.q.packed, layer.q.scales, h_norm_, q_,
                                       1, layer.q.out_f, layer.q.in_f, 0));
            // k_proj
            DVK(ops::linear_q4_g64_bf16(layer.k.packed, layer.k.scales, h_norm_, kb_,
                                       1, layer.k.out_f, layer.k.in_f, 0));
            // v_proj → KV cache
            __nv_bfloat16* v_ptr = kv_v_[0] + (size_t)pos * nKVh * HD;
            DVK(ops::linear_q4_g64_bf16(layer.v.packed, layer.v.scales, h_norm_, v_ptr,
                                       1, layer.v.out_f, layer.v.in_f, 0));

            // rope + k to cache
            const float* cos_pos = cos_t_ + (size_t)pos * HD;
            const float* sin_pos = sin_t_ + (size_t)pos * HD;
            DVK(ops::rope_apply_q_inplace_k_to_cache(
                q_, kb_, kv_k_[0], pos, nKVh, cos_pos, sin_pos,
                nQ, nKVh, 1, HD, 0));

            // attention
            float attn_scale = 1.0f / sqrtf((float)HD);
            DVK(ops::attn_decode_bf16(q_, kv_k_[0], kv_v_[0], attn_,
                                     nQ, nKVh, HD, pos + 1, attn_scale, 0));

            // o_proj + residual
            DVK(ops::linear_q4_g64_bf16_residual(layer.o.packed, layer.o.scales,
                                                attn_, h_, h_, 1, layer.o.out_f, layer.o.in_f, 0));

            // MLP: gate+up
            DVK(ops::linear_q4_g64_bf16_fused2_rmsnorm(
                layer.gate.packed, layer.gate.scales,
                layer.up.packed, layer.up.scales,
                layer.post_ln, cfg.rms_eps, h_, g_, u_,
                1, layer.gate.out_f, layer.gate.in_f, 0));
            // swiglu + down + residual
            DVK(ops::linear_q4_g64_bf16_residual_swiglu(
                layer.down.packed, layer.down.scales, g_, u_, h_, h_,
                1, layer.down.out_f, layer.down.in_f, 0));

            // Final norm
            DVK(ops::rmsnorm_forward_bf16(h_, final_norm, h_, 1, H, cfg.rms_eps, 0));

            // lm_head + sample
            DVK(ops::linear_q4_g64_bf16(lm_head.packed, lm_head.scales, h_,
                                       logits_, 1, cfg.vocab, H, 0));
            DVK(ops::sampling_greedy_bf16(logits_, d_sample_, 1, cfg.vocab, 0));
            DVK(cudaMemcpy(&out_tokens[k], d_sample_, 4, cudaMemcpyDeviceToHost));

            current_token = out_tokens[k];
            ++pos;
            ++seq_len_;
        }

        return max_k;
    }

#undef DVK
};

}  // namespace viper

#endif  // VIPER_DRAFTER_H
