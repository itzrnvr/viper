// Viper operator signatures — every op returns Status (never throws).
//
// For this skeleton phase the heavy ops (linear / sdpa / swiglu / residual /
// sampling / gqa_repeat) are linked to stubs that return UNIMPLEMENTED. RMSNorm,
// RoPE, and embedding have real kernels (kernels/ops/*.cu) wired in.
#pragma once

#include "viper/common.h"
#include "viper/tensor.h"

namespace viper::ops {

// y = rmsnorm(x, weight, eps). Row-major x of shape [rows, cols].
// output shape [rows, cols] same dtype as input.
Status rmsnorm_forward(const Tensor& x, const Tensor& weight, f32 eps,
                       Tensor& y);

// Apply rotary embeddings to q and k in-place.
//   q, k: [B, T, n_heads_q or n_kv, head_dim] in bf16.
//   position_ids: [B, T] in i32.
//   theta: RoPE base (70'000'000 for Nanbeige4.2-3B).
Status rope_forward(Tensor& q, Tensor& k, const Tensor& position_ids, f32 theta,
                    i32 head_dim);

// Gather row-wise: y[i, t, :] = embedding_table[token_ids[i, t]].
// token_ids: [B, T] i32; embedding_table: [vocab, hidden] bf16; y: [B, T, hidden].
Status embedding_forward(const Tensor& embedding_table,
                         const Tensor& token_ids, Tensor& y);

// Linear (matmul + bias). qtype drives a closed switch on the weight format
// (see viper/quant.h). UNIMPLEMENTED in skeleton phase for Q4_G64/Q5_G64/etc.
Status linear_forward(const Tensor& w, const Tensor& b, const Tensor& x,
                      Tensor& y);

// Repeat-interleave KV heads for grouped-query attention: K/V are stored as
// [B, T, n_kv, head_dim] and expanded to [B, T, n_q, head_dim] on demand.
Status gqa_repeat_forward(const Tensor& kv, Tensor& out);

// Scaled-dot-product attention (causal). UNIMPLEMENTED in skeleton phase.
Status sdpa_forward(const Tensor& q, const Tensor& k, const Tensor& v,
                    Tensor& out);

// SwiGLU activation. UNIMPLEMENTED in skeleton phase.
Status swiglu_forward(const Tensor& gate, const Tensor& up, Tensor& y);

// Residual add: y = x + residual.
Status residual_forward(const Tensor& x, const Tensor& residual, Tensor& y);

// Sampling — pick next token. UNIMPLEMENTED in skeleton phase.
Status sampling_forward(const Tensor& logits, i32 top_k, f32 top_p, f32 temperature,
                        u32 seed, i32& out_token);

}  // namespace viper::ops