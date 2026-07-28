// KV cache manager — 44 slots, indexed as cache_layer_idx = loop_idx * 22 + layer_idx.
//
// Nanbeige4.2-3B has 22 transformer layers per pass and runs 2 passes (loop 0
// and loop 1) — the cache layer slot for a logical (loop, layer) pair is
// loop_idx * 22 + layer_idx. position_ids are shared across both passes.
#pragma once

#include <array>
#include <vector>

#include "viper/common.h"
#include "viper/tensor.h"

namespace viper {

struct KvCacheConfig {
    i32 n_layers_per_pass = 22;     // transformer depth per pass
    i32 n_passes = 2;               // total pass count (the "loops")
    i32 n_kv_heads = 8;
    i32 head_dim = 128;
    i32 max_seq_len = 262144;       // 262k context window
    DType kv_dtype = DType::BF16;   // bit-exact default; INT8/INT2 are lossy opts
};

// Total slot count.
constexpr i32 kCacheSlots(const KvCacheConfig& c) noexcept {
    return c.n_passes * c.n_layers_per_pass;
}

class KvCache {
  public:
    explicit KvCache(KvCacheConfig cfg);

    const KvCacheConfig& config() const noexcept { return cfg_; }
    i32 n_slots() const noexcept { return kCacheSlots(cfg_); }

    // Map (loop_idx, layer_idx) -> cache slot id.
    static constexpr i32 cache_layer_idx(i32 loop_idx, i32 layer_idx,
                                         const KvCacheConfig& c) noexcept {
        return loop_idx * c.n_layers_per_pass + layer_idx;
    }

    // Allocate all backing buffers. Refuses if VRAM headroom < 1 GB.
    Status alloc();

    // Get the K or V tensor for a given (loop, layer). Shape:
    //   [max_seq_len, n_kv_heads, head_dim] (BF16 by default).
    Tensor k(i32 loop_idx, i32 layer_idx);
    Tensor v(i32 loop_idx, i32 layer_idx);

    // Append a single step's KV. seq_len is the new sequence length (must grow
    // monotonically across calls).
    Status append(i32 loop_idx, i32 layer_idx, const Tensor& new_k,
                  const Tensor& new_v, i32 seq_len);

    i32 current_seq_len() const noexcept { return current_seq_len_; }
    void reset() noexcept { current_seq_len_ = 0; }

  private:
    KvCacheConfig cfg_;
    std::vector<DeviceBuffer> k_buffers_;  // n_slots() entries
    std::vector<DeviceBuffer> v_buffers_;  // n_slots() entries
    i32 current_seq_len_ = 0;
};

}  // namespace viper