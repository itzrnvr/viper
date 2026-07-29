// viper Tokenizer — proper BPE encoding + exact decode.
//
// CRITICAL: The model was trained with BPE (Byte Pair Encoding).
// Greedy-longest-match produces WRONG token IDs, causing garbage output.
// This implementation uses the actual BPE merge rules from tokenizer.json.
//
// Format:
//   vocab.bin: u32 n, u32 bos, u32 eos, u32 im_start, u32 im_end,
//              then per id: u32 len, utf8 bytes
//   merges.bin: u32 n_merges, then per merge: u16 len_a, bytes_a, u16 len_b, bytes_b
//
// Encoding:
//   1. Split text by special tokens (<|im_start|>, <|im_end|>, etc.)
//   2. For each non-special segment: Metaspace + BPE
//   3. Metaspace: replace ' ' with ▁ (U+2581), prepend ▁
//   4. BPE: start with UTF-8 chars, greedily merge lowest-rank pairs
#pragma once

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <unordered_map>

namespace viper {

class Tokenizer {
public:
    // Load vocab.bin. Auto-loads merges.bin from the same directory.
    bool load(const std::string& path) {
        // Try to load merges.bin from the same directory
        std::string dir = path.substr(0, path.find_last_of("/\\") + 1);
        std::string merges = dir + "merges.bin";
        return load_vocab(path) && (load_merges(merges) || true);
    }

    int bos() const { return (int)bos_; }
    int eos() const { return (int)eos_; }
    int im_start() const { return (int)im_start_; }
    int im_end() const { return (int)im_end_; }

    std::vector<int32_t> encode(const std::string& text) const {
        std::vector<int32_t> ids;

        // Split by special tokens: <|...|>
        size_t i = 0;
        while (i < text.size()) {
            // Check for special token
            if (text[i] == '<') {
                size_t close = text.find("|>", i);
                if (close != std::string::npos) {
                    std::string cand = text.substr(i, close + 2 - i);
                    auto it = vocab_map_.find(cand);
                    if (it != vocab_map_.end()) {
                        ids.push_back(it->second);
                        i = close + 2;
                        continue;
                    }
                }
            }
            // Find next special token or end
            size_t next_special = text.find("<|", i);
            if (next_special == std::string::npos) next_special = text.size();

            // BPE-encode the segment [i, next_special)
            std::string segment = text.substr(i, next_special - i);
            if (!segment.empty()) {
                auto seg_ids = bpe_encode(segment);
                ids.insert(ids.end(), seg_ids.begin(), seg_ids.end());
            }
            i = next_special;
        }
        return ids;
    }

    std::string decode(int32_t id) const {
        if (id < 0 || (uint32_t)id >= n_) return "";
        std::string s = strings_[id];
        // ▁ -> space
        const std::string sp = "\xe2\x96\x81";
        std::string out;
        for (size_t j = 0; j < s.size();) {
            if (j + 3 <= s.size() && s.compare(j, 3, sp) == 0) { out += ' '; j += 3; }
            else { out += s[j++]; }
        }
        return out;
    }

private:
    bool load_vocab(const std::string& path) {
        FILE* f = std::fopen(path.c_str(), "rb");
        if (!f) { std::fprintf(stderr, "[tok] cannot open %s\n", path.c_str()); return false; }
        uint32_t hdr[5];
        if (std::fread(hdr, 4, 5, f) != 5) { std::fclose(f); return false; }
        n_ = hdr[0]; bos_ = hdr[1]; eos_ = hdr[2]; im_start_ = hdr[3]; im_end_ = hdr[4];
        strings_.resize(n_);
        vocab_map_.clear();
        for (uint32_t i = 0; i < n_; ++i) {
            uint32_t len;
            if (std::fread(&len, 4, 1, f) != 1) { std::fclose(f); return false; }
            strings_[i].resize(len);
            if (len && std::fread(&strings_[i][0], 1, len, f) != len) { std::fclose(f); return false; }
            if (!strings_[i].empty()) vocab_map_[strings_[i]] = (int32_t)i;
        }
        std::fclose(f);
        std::printf("[tok] vocab %d tokens, bos=%d eos=%d im_start=%d im_end=%d\n",
                    n_, bos_, eos_, im_start_, im_end_);
        return true;
    }

    bool load_merges(const std::string& path) {
        FILE* f = std::fopen(path.c_str(), "rb");
        if (!f) {
            std::fprintf(stderr, "[tok] WARNING: merges.bin not found, using greedy fallback\n");
            return false;
        }
        uint32_t n_merges;
        if (std::fread(&n_merges, 4, 1, f) != 1) { std::fclose(f); return false; }
        merge_ranks_.clear();
        merge_ranks_.reserve(n_merges);
        for (uint32_t r = 0; r < n_merges; ++r) {
            uint16_t la, lb;
            if (std::fread(&la, 2, 1, f) != 1) { std::fclose(f); return false; }
            std::string a(la, '\0');
            if (la && std::fread(&a[0], 1, la, f) != la) { std::fclose(f); return false; }
            if (std::fread(&lb, 2, 1, f) != 1) { std::fclose(f); return false; }
            std::string b(lb, '\0');
            if (lb && std::fread(&b[0], 1, lb, f) != lb) { std::fclose(f); return false; }
            merge_ranks_[a + "\x01" + b] = (int32_t)r;  // \x01 as separator
        }
        std::fclose(f);
        std::printf("[tok] loaded %u BPE merges\n", n_merges);
        return true;
    }

    // BPE encode a text segment (no special tokens).
    std::vector<int32_t> bpe_encode(const std::string& text) const {
        // Metaspace: replace ' ' with ▁, prepend ▁
        const std::string sp = "\xe2\x96\x81";  // ▁ (3 bytes UTF-8)
        std::string t = sp + text;
        // Replace spaces with ▁
        std::string processed;
        processed.reserve(t.size() + 3);
        for (size_t j = 0; j < t.size(); ++j) {
            if (t[j] == ' ') processed += sp;
            else processed += t[j];
        }

        // Split into UTF-8 character tokens
        std::vector<std::string> tokens;
        size_t pos = 0;
        while (pos < processed.size()) {
            uint8_t c = (uint8_t)processed[pos];
            size_t charlen = 1;
            if (c >= 0xF0) charlen = 4;
            else if (c >= 0xE0) charlen = 3;
            else if (c >= 0xC0) charlen = 2;
            if (pos + charlen > processed.size()) charlen = 1;
            tokens.push_back(processed.substr(pos, charlen));
            pos += charlen;
        }

        if (merge_ranks_.empty()) {
            // Fallback: greedy longest match (old behavior)
            return greedy_encode(tokens);
        }

        // BPE: repeatedly merge lowest-rank pair
        while (tokens.size() > 1) {
            int best_rank = INT32_MAX;
            int best_idx = -1;
            for (size_t j = 0; j + 1 < tokens.size(); ++j) {
                std::string key = tokens[j] + "\x01" + tokens[j + 1];
                auto it = merge_ranks_.find(key);
                if (it != merge_ranks_.end() && it->second < best_rank) {
                    best_rank = it->second;
                    best_idx = (int)j;
                }
            }
            if (best_idx < 0) break;
            tokens[best_idx] = tokens[best_idx] + tokens[best_idx + 1];
            tokens.erase(tokens.begin() + best_idx + 1);
        }

        // Look up token IDs
        std::vector<int32_t> ids;
        for (auto& tok : tokens) {
            auto it = vocab_map_.find(tok);
            if (it != vocab_map_.end()) {
                ids.push_back(it->second);
            }
            // else: unknown token, skip
        }
        return ids;
    }

    // Greedy fallback (used if merges.bin not loaded)
    std::vector<int32_t> greedy_encode(const std::vector<std::string>& chars) const {
        std::vector<int32_t> ids;
        size_t i = 0;
        while (i < chars.size()) {
            // Try longest match from current position
            std::string acc;
            int best_id = -1;
            for (size_t j = i; j < chars.size(); ++j) {
                acc += chars[j];
                auto it = vocab_map_.find(acc);
                if (it != vocab_map_.end()) best_id = it->second;
            }
            if (best_id >= 0) { ids.push_back(best_id); ++i; }
            else ++i;  // skip unknown
        }
        return ids;
    }

    uint32_t n_ = 0, bos_ = 0, eos_ = 0, im_start_ = 0, im_end_ = 0;
    std::vector<std::string> strings_;
    std::unordered_map<std::string, int32_t> vocab_map_;
    std::unordered_map<std::string, int32_t> merge_ranks_;  // key: "a\x01b"
};

}  // namespace viper
