// NanbeigeModel — placeholder loader + forward stub.
#include "viper/model.h"

#include <fstream>

#include "viper/quant.h"

namespace viper {

struct NanbeigeModel::Impl {
    bool loaded = false;
};

NanbeigeModel::NanbeigeModel() : impl_(std::make_unique<Impl>()) {
    kv_cache_ = std::make_unique<KvCache>(KvCacheConfig{
        .n_layers_per_pass = config_.n_layers_per_pass,
        .n_passes = config_.n_passes,
        .n_kv_heads = config_.n_kv_heads,
        .head_dim = config_.head_dim,
        .max_seq_len = config_.max_seq_len,
    });
}

NanbeigeModel::~NanbeigeModel() = default;

Status NanbeigeModel::load(const std::string& artifact_path) {
    std::ifstream in(artifact_path, std::ios::binary | std::ios::ate);
    if (!in) {
        return Status(StatusCode::IO_ERROR,
                      "open .viper artifact failed: " + artifact_path);
    }
    const auto sz = in.tellg();
    in.seekg(0);
    if (sz < 16) {
        return Status(StatusCode::IO_ERROR, "artifact too small");
    }
    // Header parse: magic "VIPER\0\0" (8 bytes) + config blob.
    char magic[8] = {};
    in.read(magic, sizeof(magic));
    constexpr char kMagic[8] = {'V', 'I', 'P', 'E', 'R', 0, 0, 0};
    if (std::memcmp(magic, kMagic, 8) != 0) {
        return Status(StatusCode::IO_ERROR, "bad magic (not a .viper file)");
    }
    impl_->loaded = true;
    // Skeleton: weights / tensors not materialized. Forward returns
    // UNIMPLEMENTED with a clear message.
    return Status::Ok();
}

Status NanbeigeModel::forward(const std::vector<i32>& token_ids,
                              std::vector<i32>& logits) {
    if (!impl_->loaded) {
        return Status(StatusCode::INVALID_ARGUMENT, "model not loaded");
    }
    logits.assign(static_cast<usize>(config_.vocab_size), 0);
    return Status(StatusCode::UNIMPLEMENTED,
                  "NanbeigeModel::forward stub — megakernel impl lands in M3");
}

Status NanbeigeModel::sample(i32& out_token) {
    out_token = -1;
    return Status(StatusCode::UNIMPLEMENTED, "NanbeigeModel::sample stub");
}

}  // namespace viper