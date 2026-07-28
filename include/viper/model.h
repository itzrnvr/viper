// NanbeigeModel interface — load a .viper artifact, run forward, sample.
#pragma once

#include <memory>
#include <string>

#include "viper/common.h"
#include "viper/kv_cache.h"
#include "viper/tensor.h"

namespace viper {

struct ModelConfig {
    i32 vocab_size = 166144;
    i32 hidden_size = 3072;
    i32 n_layers_per_pass = 22;
    i32 n_passes = 2;
    i32 n_q_heads = 24;
    i32 n_kv_heads = 8;
    i32 head_dim = 128;
    i32 inter_size = 8192;
    i32 max_seq_len = 262144;
    f32 rope_theta = 70000000.0f;
    f32 rms_eps = 1e-6f;
};

class NanbeigeModel {
  public:
    NanbeigeModel();
    ~NanbeigeModel();

    // Load .viper artifact (placeholder in skeleton phase). Returns OK if the
    // header parses, UNIMPLEMENTED for weights / transforms.
    Status load(const std::string& artifact_path);

    // Run one decode step. token_ids is the prompt's flat token list.
    // logits[i] is filled with a sampled token id.
    Status forward(const std::vector<i32>& token_ids, std::vector<i32>& logits);

    // Sample next token from current logits (greedy in skeleton phase).
    Status sample(i32& out_token);

    const ModelConfig& config() const noexcept { return config_; }
    KvCache& kv_cache() noexcept { return *kv_cache_; }

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    ModelConfig config_;
    std::unique_ptr<KvCache> kv_cache_;
};

}  // namespace viper