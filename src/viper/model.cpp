// viper model forward — 44-step loop dispatch. Wires real SDPA (with
// causal mask) and BF16 lm_head. Accepts T tokens (full sequence each
// step; no KV cache in v1). O(n^2) but correct.

#include "viper/model.h"
#include "viper/ops.h"
#include "viper/tensor.h"
#include "viper/status.h"
#include "viper/cuda_check.h"
#include "viper/safety.h"
#include "viper/kv_cache.h"
#include "viper/quant.h"

#include "kernels/ops/rmsnorm_kernel.h"
#include "kernels/ops/rope_kernel.h"
#include "kernels/ops/embedding_kernel.h"
#include "kernels/ops/swiglu_kernel.h"
#include "kernels/ops/residual_kernel.h"
#include "kernels/ops/linear_kernel.h"
#include "kernels/ops/linear_bf16_kernel.h"
#include "kernels/ops/sdpa_kernel.h"
#include "kernels/ops/sampling_kernel.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
#include <cstring>

namespace viper {

struct LinearWeightsQ4G64 {
    const uint8_t* w_packed;
    const __nv_bfloat16* scales;
    int out_features;
    int in_features;
};

struct LayerWeights {
    LinearWeightsQ4G64 q_proj, k_proj, v_proj, o_proj;
    LinearWeightsQ4G64 gate_proj, up_proj, down_proj;
    const __nv_bfloat16* input_layernorm;
    const __nv_bfloat16* post_attention_layernorm;
};

struct ModelWeights {
    int n_layers = 22;
    int n_passes = 2;
    int hidden = 3072;
    int intermediate = 10752;
    int n_heads = 48;
    int n_kv_heads = 8;
    int head_dim = 128;
    int vocab = 166144;
    int max_seq = 262144;
    const __nv_bfloat16* embed_tokens;
    const __nv_bfloat16* lm_head;
    std::vector<LayerWeights> layers;
    const __nv_bfloat16* final_norm;
};

struct Activations {
    DeviceBuffer hidden;
    DeviceBuffer residual;
    DeviceBuffer q_buf;
    DeviceBuffer k_buf;
    DeviceBuffer v_buf;
    DeviceBuffer attn_out;
    DeviceBuffer gate_buf;
    DeviceBuffer up_buf;
    DeviceBuffer mlp_out;
    DeviceBuffer logits;
    int current_T = 0;
};

class NanbeigeModelImpl {
public:
    ModelWeights weights;
    Activations act;
    cudaStream_t stream = 0;

    Status load(const void* artifact_data, size_t artifact_size) {
        const uint8_t* p = static_cast<const uint8_t*>(artifact_data);
        if (artifact_size < 16 + 10 * 4) return Status(StatusCode::IO_ERROR, "artifact too small");
        if (std::memcmp(p, "VIPER", 5) != 0) {
            return Status(StatusCode::IO_ERROR, "bad magic");
        }
        p += 16;
        uint32_t v;
        std::memcpy(&v, p, 4); p += 4;
        if (v != 1) return Status(StatusCode::INVALID_ARGUMENT, "unsupported version");
        std::memcpy(&v, p, 4); p += 4; weights.n_layers = v;
        std::memcpy(&v, p, 4); p += 4; weights.n_passes = v;
        std::memcpy(&v, p, 4); p += 4; weights.hidden = v;
        std::memcpy(&v, p, 4); p += 4; weights.intermediate = v;
        std::memcpy(&v, p, 4); p += 4; weights.n_heads = v;
        std::memcpy(&v, p, 4); p += 4; weights.n_kv_heads = v;
        std::memcpy(&v, p, 4); p += 4; weights.head_dim = v;
        std::memcpy(&v, p, 4); p += 4; weights.vocab = v;
        std::memcpy(&v, p, 4); p += 4; weights.max_seq = v;

        weights.layers.assign(weights.n_layers * weights.n_passes, LayerWeights{});

        for (int pass = 0; pass < weights.n_passes; ++pass) {
            for (int l = 0; l < weights.n_layers; ++l) {
                LayerWeights& lw = weights.layers[pass * weights.n_layers + l];
                p = read_linear(p, lw.q_proj, weights.n_heads * weights.head_dim, weights.hidden);
                p = read_linear(p, lw.k_proj, weights.n_kv_heads * weights.head_dim, weights.hidden);
                p = read_linear(p, lw.v_proj, weights.n_kv_heads * weights.head_dim, weights.hidden);
                p = read_linear(p, lw.o_proj, weights.hidden, weights.n_heads * weights.head_dim);
                p = read_linear(p, lw.gate_proj, weights.intermediate, weights.hidden);
                p = read_linear(p, lw.up_proj, weights.intermediate, weights.hidden);
                p = read_linear(p, lw.down_proj, weights.hidden, weights.intermediate);
            }
        }
        p = read_bf16(p, weights.vocab * weights.hidden * 2, &weights.embed_tokens);
        p = read_bf16(p, weights.vocab * weights.hidden * 2, &weights.lm_head);
        p = read_bf16(p, weights.hidden * 2, &weights.final_norm);
        for (int pass = 0; pass < weights.n_passes; ++pass) {
            for (int l = 0; l < weights.n_layers; ++l) {
                LayerWeights& lw = weights.layers[pass * weights.n_layers + l];
                p = read_bf16(p, weights.hidden * 2, &lw.input_layernorm);
                p = read_bf16(p, weights.hidden * 2, &lw.post_attention_layernorm);
            }
        }
        return Status::Ok();
    }

    Status forward(const int32_t* input_ids, int T,
                   int32_t* next_token, int& next_token_len) {
        if (T <= 0 || T > 4096) return Status(StatusCode::INVALID_ARGUMENT, "T out of range");
        const int B = 1;
        const int H = weights.hidden;
        const int I = weights.intermediate;
        const int HD = weights.head_dim;
        const int nQ = weights.n_heads;
        const int nKV = weights.n_kv_heads;
        const int V = weights.vocab;
        const size_t BT = (size_t)B * T;

        Status s = Status::Ok();
        if (act.current_T < T || !act.hidden.ptr()) {
            s = act.hidden.alloc(BT * H * 2);
            if (s.ok()) s = act.residual.alloc(BT * H * 2);
            if (s.ok()) s = act.q_buf.alloc(BT * nQ * HD * 2);
            if (s.ok()) s = act.k_buf.alloc(BT * nKV * HD * 2);
            if (s.ok()) s = act.v_buf.alloc(BT * nKV * HD * 2);
            if (s.ok()) s = act.attn_out.alloc(BT * nQ * HD * 2);
            if (s.ok()) s = act.gate_buf.alloc(BT * I * 2);
            if (s.ok()) s = act.up_buf.alloc(BT * I * 2);
            if (s.ok()) s = act.mlp_out.alloc(BT * H * 2);
            if (s.ok()) s = act.logits.alloc(BT * V * 2);
            if (s.ok()) act.current_T = T;
        }
        if (!s.ok()) return s;

        // Embedding lookup.
        {
            Tensor table(const_cast<__nv_bfloat16*>(weights.embed_tokens),
                         DType::BF16, Shape({(i64)V, (i64)H}));
            Tensor ids(const_cast<int32_t*>(input_ids), DType::INT32, Shape({(i64)B, (i64)T}));
            Tensor out(act.hidden.ptr(), DType::BF16, Shape({(i64)B * T, (i64)H}));
            s = ops::embedding_forward(table, ids, out);
            if (!s.ok()) return s;
        }

        // Position ids: [0, 1, ..., T-1] -- real arange, not bf16 garbage.
        std::vector<int32_t> pos_ids(T);
        for (int i = 0; i < T; ++i) pos_ids[i] = i;
        int32_t* d_pos = nullptr;
        cudaMalloc(&d_pos, T * sizeof(int32_t));
        cudaMemcpy(d_pos, pos_ids.data(), T * sizeof(int32_t), cudaMemcpyHostToDevice);

        for (int loop_idx = 0; loop_idx < weights.n_passes; ++loop_idx) {
            for (int l = 0; l < weights.n_layers; ++l) {
                const LayerWeights& lw = weights.layers[loop_idx * weights.n_layers + l];

                // residual save: hidden -> residual
                {
                    Tensor x(act.hidden.ptr(), DType::BF16, Shape({(i64)BT, (i64)H}));
                    Tensor y(act.residual.ptr(), DType::BF16, Shape({(i64)BT, (i64)H}));
                    s = ops::residual_forward(x, x, y);
                    if (!s.ok()) { cudaFree(d_pos); return s; }
                }
                // input_layernorm: hidden = norm(hidden)
                {
                    Tensor x(act.hidden.ptr(), DType::BF16, Shape({(i64)BT, (i64)H}));
                    Tensor g(const_cast<__nv_bfloat16*>(lw.input_layernorm), DType::BF16, Shape({(i64)H}));
                    Tensor y(act.hidden.ptr(), DType::BF16, Shape({(i64)BT, (i64)H}));
                    s = ops::rmsnorm_forward(x, g, 1e-5f, y);
                    if (!s.ok()) { cudaFree(d_pos); return s; }
                }
                // qkv projections
                {
                    Tensor x(act.hidden.ptr(), DType::BF16, Shape({(i64)BT, (i64)H}));
                    Tensor q(act.q_buf.ptr(), DType::BF16, Shape({(i64)BT, (i64)nQ * HD}));
                    s = run_linear(x, lw.q_proj, q);
                    if (!s.ok()) { cudaFree(d_pos); return s; }
                }
                {
                    Tensor x(act.hidden.ptr(), DType::BF16, Shape({(i64)BT, (i64)H}));
                    Tensor k(act.k_buf.ptr(), DType::BF16, Shape({(i64)BT, (i64)nKV * HD}));
                    s = run_linear(x, lw.k_proj, k);
                    if (!s.ok()) { cudaFree(d_pos); return s; }
                }
                {
                    Tensor x(act.hidden.ptr(), DType::BF16, Shape({(i64)BT, (i64)H}));
                    Tensor v(act.v_buf.ptr(), DType::BF16, Shape({(i64)BT, (i64)nKV * HD}));
                    s = run_linear(x, lw.v_proj, v);
                    if (!s.ok()) { cudaFree(d_pos); return s; }
                }
                // RoPE
                {
                    Tensor q(act.q_buf.ptr(), DType::BF16,
                              Shape({(i64)B, (i64)T, (i64)nQ, (i64)HD}));
                    Tensor k(act.k_buf.ptr(), DType::BF16,
                              Shape({(i64)B, (i64)T, (i64)nKV, (i64)HD}));
                    Tensor pid(d_pos, DType::INT32, Shape({(i64)B, (i64)T}));
                    s = ops::rope_forward(q, k, pid, 70000000.0f, HD);
                    if (!s.ok()) { cudaFree(d_pos); return s; }
                }
                // SDPA: causal. T_q=T_k=T. Batched attention.
                {
                    __nv_bfloat16* q_p = reinterpret_cast<__nv_bfloat16*>(act.q_buf.ptr());
                    __nv_bfloat16* k_p = reinterpret_cast<__nv_bfloat16*>(act.k_buf.ptr());
                    __nv_bfloat16* v_p = reinterpret_cast<__nv_bfloat16*>(act.v_buf.ptr());
                    __nv_bfloat16* o_p = reinterpret_cast<__nv_bfloat16*>(act.attn_out.ptr());
                    float scale = 1.0f / std::sqrt((float)HD);
                    s = safety::classify_cuda(sdpa_forward_bf16(
                        q_p, k_p, v_p, o_p,
                        B, nQ, nKV, T, T, HD, scale, /*is_causal=*/true, 0));
                    if (!s.ok()) { cudaFree(d_pos); return s; }
                }
                // o_proj: hidden = o_proj(attn_out)
                {
                    Tensor attn(act.attn_out.ptr(), DType::BF16, Shape({(i64)BT, (i64)nQ * HD}));
                    Tensor y(act.hidden.ptr(), DType::BF16, Shape({(i64)BT, (i64)H}));
                    s = run_linear(attn, lw.o_proj, y);
                    if (!s.ok()) { cudaFree(d_pos); return s; }
                }
                // residual add: hidden = residual + hidden
                {
                    Tensor r(act.residual.ptr(), DType::BF16, Shape({(i64)BT, (i64)H}));
                    Tensor x(act.hidden.ptr(), DType::BF16, Shape({(i64)BT, (i64)H}));
                    Tensor y(act.hidden.ptr(), DType::BF16, Shape({(i64)BT, (i64)H}));
                    s = ops::residual_forward(r, x, y);
                    if (!s.ok()) { cudaFree(d_pos); return s; }
                }
                // post_attention_layernorm
                {
                    Tensor x(act.hidden.ptr(), DType::BF16, Shape({(i64)BT, (i64)H}));
                    Tensor g(const_cast<__nv_bfloat16*>(lw.post_attention_layernorm), DType::BF16, Shape({(i64)H}));
                    Tensor y(act.hidden.ptr(), DType::BF16, Shape({(i64)BT, (i64)H}));
                    s = ops::rmsnorm_forward(x, g, 1e-5f, y);
                    if (!s.ok()) { cudaFree(d_pos); return s; }
                }
                // gate, up
                {
                    Tensor x(act.hidden.ptr(), DType::BF16, Shape({(i64)BT, (i64)H}));
                    Tensor g(act.gate_buf.ptr(), DType::BF16, Shape({(i64)BT, (i64)I}));
                    s = run_linear(x, lw.gate_proj, g);
                    if (!s.ok()) { cudaFree(d_pos); return s; }
                }
                {
                    Tensor x(act.hidden.ptr(), DType::BF16, Shape({(i64)BT, (i64)H}));
                    Tensor u(act.up_buf.ptr(), DType::BF16, Shape({(i64)BT, (i64)I}));
                    s = run_linear(x, lw.up_proj, u);
                    if (!s.ok()) { cudaFree(d_pos); return s; }
                }
                // swiglu
                {
                    Tensor g(act.gate_buf.ptr(), DType::BF16, Shape({(i64)BT, (i64)I}));
                    Tensor u(act.up_buf.ptr(), DType::BF16, Shape({(i64)BT, (i64)I}));
                    Tensor y(act.gate_buf.ptr(), DType::BF16, Shape({(i64)BT, (i64)I}));
                    s = ops::swiglu_forward(g, u, y);
                    if (!s.ok()) { cudaFree(d_pos); return s; }
                }
                // down_proj
                {
                    Tensor x(act.gate_buf.ptr(), DType::BF16, Shape({(i64)BT, (i64)I}));
                    Tensor y(act.hidden.ptr(), DType::BF16, Shape({(i64)BT, (i64)H}));
                    s = run_linear(x, lw.down_proj, y);
                    if (!s.ok()) { cudaFree(d_pos); return s; }
                }
                // residual add
                {
                    Tensor r(act.residual.ptr(), DType::BF16, Shape({(i64)BT, (i64)H}));
                    Tensor x(act.hidden.ptr(), DType::BF16, Shape({(i64)BT, (i64)H}));
                    Tensor y(act.hidden.ptr(), DType::BF16, Shape({(i64)BT, (i64)H}));
                    s = ops::residual_forward(r, x, y);
                    if (!s.ok()) { cudaFree(d_pos); return s; }
                }
            }
        }

        // lm_head: only the last token's hidden. logits = last_hidden @ W^T.
        __nv_bfloat16* last_hidden =
            reinterpret_cast<__nv_bfloat16*>(act.hidden.ptr()) + (T - 1) * H;
        {
            cudaError_t e = linear_bf16(
                weights.lm_head,
                last_hidden,
                reinterpret_cast<__nv_bfloat16*>(act.logits.ptr()),
                1, V, H, 0);
            if (e != cudaSuccess) {
                cudaFree(d_pos);
                return safety::classify_cuda(e);
            }
        }
        // Greedy sample.
        {
            cudaError_t e = sampling_greedy_bf16(
                reinterpret_cast<const __nv_bfloat16*>(act.logits.ptr()),
                next_token, 1, V, 0);
            cudaFree(d_pos);
            if (e != cudaSuccess) return safety::classify_cuda(e);
            next_token_len = 1;
            return Status::Ok();
        }
    }

private:
    const uint8_t* read_linear(const uint8_t* p, LinearWeightsQ4G64& lw,
                                int rows, int cols) {
        uint64_t packed_sz;
        std::memcpy(&packed_sz, p, 8); p += 8;
        lw.w_packed = p;
        p += packed_sz;
        uint64_t scales_sz;
        std::memcpy(&scales_sz, p, 8); p += 8;
        lw.scales = reinterpret_cast<const __nv_bfloat16*>(p);
        p += scales_sz;
        lw.out_features = rows;
        lw.in_features = cols;
        return p;
    }
    const uint8_t* read_bf16(const uint8_t* p, int64_t bytes,
                              const __nv_bfloat16** out) {
        int64_t sz;
        std::memcpy(&sz, p, 8); p += 8;
        *out = reinterpret_cast<const __nv_bfloat16*>(p);
        p += sz;
        (void)bytes;
        return p;
    }
    Status run_linear(const Tensor& x, const LinearWeightsQ4G64& w, Tensor& y) {
        return safety::classify_cuda(linear_q4_g64_bf16(
            w.w_packed, w.scales,
            x.data_as<__nv_bfloat16>(),
            y.data_as<__nv_bfloat16>(),
            (int)x.shape()[0], w.out_features, w.in_features, 0));
    }
};

NanbeigeModel::NanbeigeModel() : impl_(new NanbeigeModelImpl()) {}
NanbeigeModel::~NanbeigeModel() { delete impl_; }
Status NanbeigeModel::load(const void* data, size_t size) {
    return impl_->load(data, size);
}
Status NanbeigeModel::forward(const int32_t* input_ids, int T,
                              int32_t* next_token, int& next_token_len) {
    return impl_->forward(input_ids, T, next_token, next_token_len);
}

}  // namespace viper
