// KV cache manager.
#include "viper/kv_cache.h"

#include "viper/safety.h"

namespace viper {

KvCache::KvCache(KvCacheConfig cfg) : cfg_(cfg) {}

Status KvCache::alloc() {
    k_buffers_.clear();
    v_buffers_.clear();
    k_buffers_.resize(static_cast<usize>(n_slots()));
    v_buffers_.resize(static_cast<usize>(n_slots()));

    const usize per_slot_bytes = static_cast<usize>(cfg_.max_seq_len) *
                                 static_cast<usize>(cfg_.n_kv_heads) *
                                 static_cast<usize>(cfg_.head_dim) *
                                 dtype_sizeof(cfg_.kv_dtype);
    for (i32 slot = 0; slot < n_slots(); ++slot) {
        if (auto s = k_buffers_[slot].alloc(per_slot_bytes); !s.ok()) return s;
        if (auto s = v_buffers_[slot].alloc(per_slot_bytes); !s.ok()) return s;
    }
    current_seq_len_ = 0;
    return Status::Ok();
}

Tensor KvCache::k(i32 loop_idx, i32 layer_idx) {
    const i32 slot = cache_layer_idx(loop_idx, layer_idx, cfg_);
    const usize bytes = static_cast<usize>(cfg_.max_seq_len) *
                        static_cast<usize>(cfg_.n_kv_heads) *
                        static_cast<usize>(cfg_.head_dim) *
                        dtype_sizeof(cfg_.kv_dtype);
    return Tensor(k_buffers_[slot].ptr(), cfg_.kv_dtype,
                  Shape{cfg_.max_seq_len, cfg_.n_kv_heads, cfg_.head_dim});
}

Tensor KvCache::v(i32 loop_idx, i32 layer_idx) {
    const i32 slot = cache_layer_idx(loop_idx, layer_idx, cfg_);
    return Tensor(v_buffers_[slot].ptr(), cfg_.kv_dtype,
                  Shape{cfg_.max_seq_len, cfg_.n_kv_heads, cfg_.head_dim});
}

Status KvCache::append(i32 loop_idx, i32 layer_idx, const Tensor& new_k,
                       const Tensor& new_v, i32 seq_len) {
    if (seq_len < current_seq_len_) {
        return Status(StatusCode::INVALID_ARGUMENT,
                      "KV append: seq_len must be monotonically increasing");
    }
    if (seq_len > cfg_.max_seq_len) {
        return Status(StatusCode::INVALID_ARGUMENT, "KV append: seq_len > max_seq_len");
    }
    // Skeleton: no-op copy. Real impl memcpy_async from new_k/new_v into slot.
    (void)loop_idx;
    (void)layer_idx;
    (void)new_k;
    (void)new_v;
    current_seq_len_ = seq_len;
    return Status::Ok();
}

}  // namespace viper