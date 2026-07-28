// viper_convert — HF safetensors -> .viper format. PURE C++, no CUDA deps.
// Reads the Nanbeige4.2-3B safetensors and writes a closed .viper artifact.
//
// Usage: viper_convert <input_dir> <output.viper>

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <fstream>
#include <vector>
#include <string>
#include <unordered_map>

struct TensorInfo {
    std::string dtype;
    std::vector<int64_t> shape;
    int64_t data_offset;
    int64_t data_length;
};

static bool read_safetensors(const std::string& path,
                              std::unordered_map<std::string, TensorInfo>& info,
                              std::vector<uint8_t>& data) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) return false;
    int64_t size = f.tellg();
    f.seekg(0, std::ios::beg);
    int64_t header_len = 0;
    f.read(reinterpret_cast<char*>(&header_len), 8);
    if (header_len <= 0 || header_len > 100 * 1024 * 1024) return false;
    std::string header(header_len, '\0');
    f.read(header.data(), header_len);

    size_t pos = 0;
    while (pos < header.size()) {
        size_t key_start = header.find('"', pos);
        if (key_start == std::string::npos) break;
        size_t key_end = header.find('"', key_start + 1);
        if (key_end == std::string::npos) break;
        std::string key = header.substr(key_start + 1, key_end - key_start - 1);
        if (key == "__metadata__") {
            pos = header.find('}', key_end) + 1;
            continue;
        }
        size_t brace = header.find('{', key_end);
        size_t brace_end = header.find('}', brace);
        if (brace == std::string::npos || brace_end == std::string::npos) break;

        TensorInfo t;
        size_t dtype_pos = header.find("\"dtype\"", brace);
        size_t dq1 = header.find('"', dtype_pos + 7);
        size_t dq2 = header.find('"', dq1 + 1);
        t.dtype = header.substr(dq1 + 1, dq2 - dq1 - 1);

        size_t shape_pos = header.find("\"shape\"", brace);
        size_t lb = header.find('[', shape_pos);
        size_t rb = header.find(']', lb);
        std::string shape_str = header.substr(lb + 1, rb - lb - 1);
        size_t p = 0;
        while (p < shape_str.size()) {
            while (p < shape_str.size() && (shape_str[p] == ' ' || shape_str[p] == ',')) ++p;
            int64_t v = 0;
            while (p < shape_str.size() && shape_str[p] >= '0' && shape_str[p] <= '9') {
                v = v * 10 + (shape_str[p] - '0');
                ++p;
            }
            t.shape.push_back(v);
        }
        size_t offsets_pos = header.find("\"data_offsets\"", brace);
        size_t ob1 = header.find('[', offsets_pos);
        size_t ob2 = header.find(',', ob1);
        size_t ob3 = header.find(']', ob2);
        int64_t start = std::stoll(header.substr(ob1 + 1, ob2 - ob1 - 1));
        int64_t end = std::stoll(header.substr(ob2 + 1, ob3 - ob2 - 1));
        t.data_offset = start;
        t.data_length = end - start;

        info[key] = t;
        pos = brace_end + 1;
    }
    int64_t data_start = 8 + header_len;
    int64_t data_size = size - data_start;
    data.resize(data_size);
    f.seekg(data_start, std::ios::beg);
    f.read(reinterpret_cast<char*>(data.data()), data_size);
    return true;
}

static void quantize_q4_g64(const uint16_t* w, int N, int K,
                           std::vector<uint8_t>& packed,
                           std::vector<uint16_t>& scales_bf16) {
    packed.assign(N * K / 2, 0);
    scales_bf16.assign(N * K / 64, 0);
    // CRITICAL: proper BF16→float32 conversion (shift to upper 16 bits).
    auto bf16_to_f32 = [](uint16_t bits) -> float {
        uint32_t u = (uint32_t)bits << 16;
        float v; std::memcpy(&v, &u, 4);
        return v;
    };
    for (int n = 0; n < N; ++n) {
        for (int g = 0; g < K / 64; ++g) {
            float max_abs = 0.0f;
            for (int j = 0; j < 64; ++j) {
                float v = std::fabs(bf16_to_f32(w[n * K + g * 64 + j]));
                if (v > max_abs) max_abs = v;
            }
            float scale = max_abs / 7.0f;
            if (scale < 1e-8f) scale = 1e-8f;
            uint32_t su; std::memcpy(&su, &scale, 4);
            scales_bf16[n * (K / 64) + g] = (uint16_t)(su >> 16);
            for (int j = 0; j < 64; ++j) {
                float v = bf16_to_f32(w[n * K + g * 64 + j]);
                int stored = (int)std::round(v / scale) + 8;
                if (stored < 0) stored = 0;
                if (stored > 15) stored = 15;
                int byte_idx = (g * 64 + j) / 2;
                if (j % 2 == 0) {
                    packed[n * (K / 2) + byte_idx] |= (uint8_t)(stored & 0x0F);
                } else {
                    packed[n * (K / 2) + byte_idx] |= (uint8_t)((stored & 0x0F) << 4);
                }
            }
        }
    }
}

int main(int argc, char** argv) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <input_dir> <output.viper>\n", argv[0]);
        return 1;
    }
    std::string input_dir = argv[1];
    std::string output_path = argv[2];

    std::vector<std::string> shard_paths;
    std::string single = input_dir + "/model.safetensors";
    std::ifstream test(single);
    if (test.good()) {
        shard_paths.push_back(single);
    } else {
        for (int i = 1; i <= 10; ++i) {
            char buf[256];
            snprintf(buf, sizeof(buf), "%s/model-%05d-of-00005.safetensors",
                     input_dir.c_str(), i);
            std::ifstream t(buf);
            if (t.good()) shard_paths.push_back(buf);
        }
        if (shard_paths.empty()) {
            // Try 2-shard model.
            for (int i = 1; i <= 2; ++i) {
                char buf[256];
                snprintf(buf, sizeof(buf), "%s/model-%05d-of-00002.safetensors",
                         input_dir.c_str(), i);
                std::ifstream t(buf);
                if (t.good()) shard_paths.push_back(buf);
            }
        }
    }
    if (shard_paths.empty()) {
        fprintf(stderr, "no safetensors shards found in %s\n", input_dir.c_str());
        return 1;
    }
    printf("found %zu shard(s)\n", shard_paths.size());

    // The model has 2 shards. Load both and merge into a single map.
    std::unordered_map<std::string, TensorInfo> all_info;
    std::unordered_map<std::string, std::vector<uint8_t>> all_data;
    for (size_t shard_idx = 0; shard_idx < shard_paths.size(); ++shard_idx) {
        std::unordered_map<std::string, TensorInfo> info;
        std::vector<uint8_t> data;
        if (!read_safetensors(shard_paths[shard_idx], info, data)) {
            fprintf(stderr, "failed to read %s\n", shard_paths[shard_idx].c_str());
            return 1;
        }
        printf("shard %zu: %zu tensors, %lld bytes\n", shard_idx, info.size(), (long long)data.size());
        // Move into the combined map. Each shard's data is offset by 0 in its own buffer.
        for (auto& [k, v] : info) {
            std::vector<uint8_t> td(data.begin() + v.data_offset,
                                     data.begin() + v.data_offset + v.data_length);
            all_info[k] = v;
            all_data[k] = std::move(td);
        }
    }

    // Now write the .viper artifact.
    std::ofstream out(output_path, std::ios::binary);
    if (!out) { fprintf(stderr, "cannot open output %s\n", output_path.c_str()); return 1; }

    // Header.
    out.write("VIPER\0\0\0\0\0\0\0\0\0\0", 16);
    uint32_t v32 = 1; out.write(reinterpret_cast<char*>(&v32), 4);  // version
    v32 = 22; out.write(reinterpret_cast<char*>(&v32), 4);  // n_layers
    v32 = 2;  out.write(reinterpret_cast<char*>(&v32), 4);  // n_passes
    v32 = 3072; out.write(reinterpret_cast<char*>(&v32), 4);  // hidden
    v32 = 10752; out.write(reinterpret_cast<char*>(&v32), 4);  // intermediate
    v32 = 48; out.write(reinterpret_cast<char*>(&v32), 4);  // n_heads
    v32 = 8; out.write(reinterpret_cast<char*>(&v32), 4);  // n_kv_heads
    v32 = 128; out.write(reinterpret_cast<char*>(&v32), 4);  // head_dim
    v32 = 166144; out.write(reinterpret_cast<char*>(&v32), 4);  // vocab
    v32 = 262144; out.write(reinterpret_cast<char*>(&v32), 4);  // max_seq

    const int n_layers = 22;
    const int n_passes = 2;
    const int H = 3072;
    const int I = 10752;
    const int n_heads = 48;
    const int n_kv_heads = 8;
    const int D = 128;

    // Quantize the linears.
    auto write_linear = [&](const std::string& key, int rows, int cols) {
        auto it = all_info.find(key);
        if (it == all_info.end()) {
            fprintf(stderr, "missing tensor %s\n", key.c_str());
            return false;
        }
        const uint16_t* bf16 = reinterpret_cast<const uint16_t*>(all_data[key].data());
        std::vector<uint8_t> packed;
        std::vector<uint16_t> scales;
        quantize_q4_g64(bf16, rows, cols, packed, scales);
        uint64_t sz = packed.size();
        out.write(reinterpret_cast<char*>(&sz), 8);
        out.write(reinterpret_cast<const char*>(packed.data()), packed.size());
        sz = scales.size() * sizeof(uint16_t);
        out.write(reinterpret_cast<char*>(&sz), 8);
        out.write(reinterpret_cast<const char*>(scales.data()), scales.size() * 2);
        printf("  %s: Q4 %d x %d (%.2f MB)\n", key.c_str(), rows, cols,
               (packed.size() + scales.size() * 2) / 1e6);
        return true;
    };
    auto write_bf16 = [&](const std::string& key) {
        auto it = all_info.find(key);
        if (it == all_info.end()) {
            fprintf(stderr, "missing tensor %s\n", key.c_str());
            return false;
        }
        int64_t sz = it->second.data_length;
        out.write(reinterpret_cast<char*>(&sz), 8);
        out.write(reinterpret_cast<const char*>(all_data[key].data()), sz);
        printf("  %s: BF16 (%.2f MB)\n", key.c_str(), sz / 1e6);
        return true;
    };

    for (int p = 0; p < n_passes; ++p) {
        for (int l = 0; l < n_layers; ++l) {
            char buf[256];
            const char* pre = "model.layers";
            snprintf(buf, sizeof(buf), "%s.%d.self_attn.q_proj.weight", pre, l);
            if (!write_linear(buf, n_heads * D, H)) return 1;
            snprintf(buf, sizeof(buf), "%s.%d.self_attn.k_proj.weight", pre, l);
            if (!write_linear(buf, n_kv_heads * D, H)) return 1;
            snprintf(buf, sizeof(buf), "%s.%d.self_attn.v_proj.weight", pre, l);
            if (!write_linear(buf, n_kv_heads * D, H)) return 1;
            snprintf(buf, sizeof(buf), "%s.%d.self_attn.o_proj.weight", pre, l);
            if (!write_linear(buf, H, n_heads * D)) return 1;
            snprintf(buf, sizeof(buf), "%s.%d.mlp.gate_proj.weight", pre, l);
            if (!write_linear(buf, I, H)) return 1;
            snprintf(buf, sizeof(buf), "%s.%d.mlp.up_proj.weight", pre, l);
            if (!write_linear(buf, I, H)) return 1;
            snprintf(buf, sizeof(buf), "%s.%d.mlp.down_proj.weight", pre, l);
            if (!write_linear(buf, H, I)) return 1;
        }
    }

    if (!write_bf16("model.embed_tokens.weight")) return 1;
    if (!write_bf16("lm_head.weight")) return 1;
    if (!write_bf16("model.norm.weight")) return 1;
    for (int p = 0; p < n_passes; ++p) {
        for (int l = 0; l < n_layers; ++l) {
            char buf[256];
            snprintf(buf, sizeof(buf), "model.layers.%d.input_layernorm.weight", l);
            if (!write_bf16(buf)) return 1;
            snprintf(buf, sizeof(buf), "model.layers.%d.post_attention_layernorm.weight", l);
            if (!write_bf16(buf)) return 1;
        }
    }

    out.close();
    printf("wrote %s\n", output_path.c_str());
    return 0;
}
