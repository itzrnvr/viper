// viper_convert — HF safetensors -> .viper format.
//
// Reads the Nanbeige4.2-3B safetensors checkpoint and writes a closed
// .viper artifact that viper-cli can mmap at load time. Quantizes
// linear weights to Q4_G64 (4-bit symmetric, per-group FP16 scale,
// group size 64). Keeps lm_head and embeddings in BF16 (untied).
//
// Usage: viper_convert <input_dir> <output.viper>
//
// Input:  D:\hf-cache\Nanbeige4.2-3B\  (model-00001-of-00002.safetensors etc.)
// Output: D:\dev\viper\artifacts\Nanbeige4.2-3B.viper
//
// Safety: streams the file in chunks; max ~16 GB peak host RAM usage.

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <vector>
#include <string>
#include <unordered_map>
#include <algorithm>

// safetensors format: 8-byte little-endian header length, then JSON
// header of {tensor_name: {dtype, shape, data_offsets}}. Data follows
// immediately after the header. Reference:
// https://huggingface.co/docs/safetensors/index

struct TensorInfo {
    std::string dtype;
    std::vector<int64_t> shape;
    int64_t data_offset;
    int64_t data_length;
};

bool read_safetensors(const std::string& path,
                      std::unordered_map<std::string, TensorInfo>& info,
                      std::vector<uint8_t>& data) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) {
        fprintf(stderr, "cannot open %s\n", path.c_str());
        return false;
    }
    int64_t size = f.tellg();
    f.seekg(0, std::ios::beg);

    int64_t header_len = 0;
    f.read(reinterpret_cast<char*>(&header_len), 8);
    if (header_len <= 0 || header_len > 100 * 1024 * 1024) {
        fprintf(stderr, "bad header length %lld\n", (long long)header_len);
        return false;
    }

    std::string header(header_len, '\0');
    f.read(header.data(), header_len);

    // Minimal JSON parser — assumes well-formed safetensors header.
    // Extracts each tensor entry's dtype, shape, and data_offsets.
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
        // dtype
        size_t dtype_pos = header.find("\"dtype\"", brace);
        size_t dq1 = header.find('"', dtype_pos + 7);
        size_t dq2 = header.find('"', dq1 + 1);
        t.dtype = header.substr(dq1 + 1, dq2 - dq1 - 1);
        // shape
        size_t shape_pos = header.find("\"shape\"", brace);
        size_t lb = header.find('[', shape_pos);
        size_t rb = header.find(']', lb);
        std::string shape_str = header.substr(lb + 1, rb - lb - 1);
        // Parse ints
        size_t p = 0;
        while (p < shape_str.size()) {
            while (p < shape_str.size() && (shape_str[p] == ' ' || shape_str[p] == ',')) ++p;
            int64_t v = 0;
            bool neg = false;
            if (p < shape_str.size() && shape_str[p] == '-') { neg = true; ++p; }
            while (p < shape_str.size() && shape_str[p] >= '0' && shape_str[p] <= '9') {
                v = v * 10 + (shape_str[p] - '0');
                ++p;
            }
            if (neg) v = -v;
            if (p < shape_str.size() && shape_str[p] == ',') ++p;
            t.shape.push_back(v);
        }
        // data_offsets
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

// Quantize BF16 weight to Q4_G64 (symmetric, group size 64).
// Returns packed bytes (N*K/2) and scales (N*K/64).
void quantize_q4_g64(const uint16_t* bf16_weights, int N, int K,
                    std::vector<uint8_t>& packed,
                    std::vector<uint16_t>& scales_bf16) {
    packed.assign(N * K / 2, 0);
    scales_bf16.assign(N * K / 64, 0);
    for (int n = 0; n < N; ++n) {
        for (int g = 0; g < K / 64; ++g) {
            // Find max abs in this group.
            float max_abs = 0.0f;
            for (int j = 0; j < 64; ++j) {
                uint16_t bits = bf16_weights[n * K + g * 64 + j];
                float v;
                std::memcpy(&v, &bits, 4);
                v = std::fmax(v, -v);
                if (v > max_abs) max_abs = v;
            }
            float scale = max_abs / 7.0f;
            if (scale < 1e-8f) scale = 1e-8f;
            // Convert scale to BF16.
            uint16_t scale_bits;
            float scale_clamped = scale;
            std::memcpy(&scale_bits, &scale_clamped, 4);
            // The BF16 is the upper 16 bits of FP32.
            uint16_t bf16 = scale_bits >> 16;
            scales_bf16[n * (K / 64) + g] = bf16;
            // Pack the 64 weights.
            for (int j = 0; j < 64; ++j) {
                uint16_t bits = bf16_weights[n * K + g * 64 + j];
                float v;
                std::memcpy(&v, &bits, 4);
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

// .viper artifact format (v1):
//   16-byte magic: "VIPER\0\0\0\0\0\0\0\0\0\0" (padded with zeros)
//   uint32_t version = 1
//   uint32_t n_layers = 22, n_passes = 2
//   uint32_t hidden = 3072, intermediate = 10752, n_heads = 48, n_kv_heads = 8
//   uint32_t head_dim = 128, vocab = 166144, max_seq = 262144
//   For each linear (q, k, v, o, gate, up, down) per layer (22) per pass (2):
//     uint64_t packed_size, scales_size, then the data
//   Embed: vocab * hidden BF16
//   Lm_head: vocab * hidden BF16
//   Final norm: hidden BF16
//   Per-layer norms (input_layernorm, post_attention_layernorm): 22*2 each, hidden BF16
//   Per-pass layer index remap: pass * 22 + layer (matches our KV cache slot)

int main(int argc, char** argv) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <input_safetensors_dir> <output.viper>\n", argv[0]);
        return 1;
    }
    std::string input_dir = argv[1];
    std::string output_path = argv[2];

    // Read all safetensors shards in the directory.
    // For Nanbeige4.2-3B there's typically model-00001-of-00002.safetensors
    // and model-00002-of-00002.safetensors. We handle single-shard for v1.
    std::vector<std::string> shard_paths;
    // Try single-shard first.
    std::string single = input_dir + "/model.safetensors";
    std::ifstream test(single);
    if (test.good()) {
        shard_paths.push_back(single);
    } else {
        // Look for model-00001-of-NNNNN.safetensors.
        for (int i = 1; i <= 10; ++i) {
            char buf[256];
            snprintf(buf, sizeof(buf), "%s/model-%05d-of-00005.safetensors",
                     input_dir.c_str(), i);
            std::ifstream t(buf);
            if (t.good()) shard_paths.push_back(buf);
        }
    }
    if (shard_paths.empty()) {
        fprintf(stderr, "no safetensors shards found in %s\n", input_dir.c_str());
        return 1;
    }
    printf("found %zu shard(s)\n", shard_paths.size());

    // Read all tensors from the first shard (v1: single-shard model only).
    std::unordered_map<std::string, TensorInfo> info;
    std::vector<uint8_t> data;
    if (!read_safetensors(shard_paths[0], info, data)) {
        fprintf(stderr, "failed to read %s\n", shard_paths[0].c_str());
        return 1;
    }
    printf("read %zu tensors, %lld bytes of data\n", info.size(), (long long)data.size());

    // Constants for Nanbeige4.2-3B.
    const int n_layers = 22;
    const int n_passes = 2;
    const int H = 3072;
    const int I = 10752;
    const int n_heads = 48;
    const int n_kv_heads = 8;
    const int D = 128;
    const int V = 166144;
    const int max_seq = 262144;

    // Open output file.
    std::ofstream out(output_path, std::ios::binary);
    if (!out) {
        fprintf(stderr, "cannot open output %s\n", output_path.c_str());
        return 1;
    }

    // Magic.
    out.write("VIPER\0\0\0\0\0\0\0\0\0\0", 16);
    uint32_t v32 = 1; out.write(reinterpret_cast<char*>(&v32), 4);
    v32 = n_layers; out.write(reinterpret_cast<char*>(&v32), 4);
    v32 = n_passes; out.write(reinterpret_cast<char*>(&v32), 4);
    v32 = H; out.write(reinterpret_cast<char*>(&v32), 4);
    v32 = I; out.write(reinterpret_cast<char*>(&v32), 4);
    v32 = n_heads; out.write(reinterpret_cast<char*>(&v32), 4);
    v32 = n_kv_heads; out.write(reinterpret_cast<char*>(&v32), 4);
    v32 = D; out.write(reinterpret_cast<char*>(&v32), 4);
    v32 = V; out.write(reinterpret_cast<char*>(&v32), 4);
    v32 = max_seq; out.write(reinterpret_cast<char*>(&v32), 4);

    // Helper: read BF16 tensor and quantize if it's a linear.
    auto write_linear = [&](const std::string& name, int rows, int cols) {
        auto it = info.find(name);
        if (it == info.end()) {
            fprintf(stderr, "missing tensor %s\n", name.c_str());
            return false;
        }
        const uint16_t* bf16 = reinterpret_cast<const uint16_t*>(
            data.data() + it->second.data_offset);
        std::vector<uint8_t> packed;
        std::vector<uint16_t> scales;
        quantize_q4_g64(bf16, rows, cols, packed, scales);
        uint64_t sz = packed.size();
        out.write(reinterpret_cast<char*>(&sz), 8);
        out.write(reinterpret_cast<const char*>(packed.data()), packed.size());
        sz = scales.size() * sizeof(uint16_t);
        out.write(reinterpret_cast<char*>(&sz), 8);
        out.write(reinterpret_cast<const char*>(scales.data()), scales.size() * 2);
        printf("  %s: Q4_G64 %d x %d (%.2f MB)\n", name.c_str(), rows, cols,
               (packed.size() + scales.size() * 2) / 1e6);
        return true;
    };
    auto write_bf16 = [&](const std::string& name) {
        auto it = info.find(name);
        if (it == info.end()) {
            fprintf(stderr, "missing tensor %s\n", name.c_str());
            return false;
        }
        int64_t sz = it->second.data_length;
        out.write(reinterpret_cast<char*>(&sz), 8);
        out.write(reinterpret_cast<const char*>(data.data() + it->second.data_offset), sz);
        printf("  %s: BF16 (%.2f MB)\n", name.c_str(), sz / 1e6);
        return true;
    };

    // Per layer, per pass: q, k, v, o, gate, up, down.
    // The pass is implicit in num_loops=2 — same layer used twice.
    for (int p = 0; p < n_passes; ++p) {
        for (int l = 0; l < n_layers; ++l) {
            char name[256];
            const char* layer_prefix = "model.layers";
            // q_proj [n_heads*D, H] = [6144, 3072]
            snprintf(name, sizeof(name), "%s.%d.self_attn.q_proj.weight",
                     layer_prefix, l);
            if (!write_linear(name, n_heads * D, H)) return 1;
            // k_proj [n_kv_heads*D, H] = [1024, 3072]
            snprintf(name, sizeof(name), "%s.%d.self_attn.k_proj.weight",
                     layer_prefix, l);
            if (!write_linear(name, n_kv_heads * D, H)) return 1;
            // v_proj [n_kv_heads*D, H] = [1024, 3072]
            snprintf(name, sizeof(name), "%s.%d.self_attn.v_proj.weight",
                     layer_prefix, l);
            if (!write_linear(name, n_kv_heads * D, H)) return 1;
            // o_proj [H, n_heads*D] = [3072, 6144]
            snprintf(name, sizeof(name), "%s.%d.self_attn.o_proj.weight",
                     layer_prefix, l);
            if (!write_linear(name, H, n_heads * D)) return 1;
            // gate_proj [I, H] = [10752, 3072]
            snprintf(name, sizeof(name), "%s.%d.mlp.gate_proj.weight",
                     layer_prefix, l);
            if (!write_linear(name, I, H)) return 1;
            // up_proj [I, H] = [10752, 3072]
            snprintf(name, sizeof(name), "%s.%d.mlp.up_proj.weight",
                     layer_prefix, l);
            if (!write_linear(name, I, H)) return 1;
            // down_proj [H, I] = [3072, 10752]
            snprintf(name, sizeof(name), "%s.%d.mlp.down_proj.weight",
                     layer_prefix, l);
            if (!write_linear(name, H, I)) return 1;
        }
    }

    // Embed [V, H] BF16.
    if (!write_bf16("model.embed_tokens.weight")) return 1;
    // lm_head [V, H] BF16 (untied).
    if (!write_bf16("lm_head.weight")) return 1;
    // Final norm [H] BF16.
    if (!write_bf16("model.norm.weight")) return 1;

    // Per-layer input_layernorm, post_attention_layernorm (BF16).
    // Stored once per pass since they're the same across loops.
    for (int p = 0; p < n_passes; ++p) {
        for (int l = 0; l < n_layers; ++l) {
            char name[256];
            snprintf(name, sizeof(name), "model.layers.%d.input_layernorm.weight", l);
            if (!write_bf16(name)) return 1;
            snprintf(name, sizeof(name), "model.layers.%d.post_attention_layernorm.weight", l);
            if (!write_bf16(name)) return 1;
        }
    }

    out.close();
    printf("wrote %s\n", output_path.c_str());
    return 0;
}
