// NanbeigeModel interface — load a .viper artifact, run forward, sample.
#pragma once

#include <memory>
#include <string>
#include <vector>
#include <cstdint>

#include "viper/common.h"
#include "viper/kv_cache.h"
#include "viper/tensor.h"

namespace viper {

struct ModelConfig {
    i32 vocab_size = 166144;
    i32 hidden_size = 3072;
    i32 n_layers_per_pass = 22;
    i32 n_passes = 2;
    i32 n_q_heads = 48;
    i32 n_kv_heads = 8;
    i32 head_dim = 128;
    i32 inter_size = 10752;
    i32 max_seq_len = 262144;
    f32 rope_theta = 70000000.0f;
    f32 rms_eps = 1e-6f;
};

class NanbeigeModel {
public:
    NanbeigeModel();
    ~NanbeigeModel();

    // Load .viper artifact from memory.
    Status load(const void* data, size_t size);

    // Run forward on input_ids[0..T-1]. Returns next token in next_token.
    // v1: T must be 1 (single token at a time). Each call computes one
    // forward pass and samples one token. For multi-token decode, the
    // caller re-runs with the full accumulated sequence.
    Status forward(const int32_t* input_ids, int T,
                   int32_t* next_token, int& next_token_len);

    const ModelConfig& config() const noexcept { return config_; }

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    ModelConfig config_;
};

}  // namespace viper
