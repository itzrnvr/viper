// Greedy longest-match tokenizer over the exported vocab.bin.
//
// v1 simplification: greedy longest-prefix match (not true BPE merge order).
// Correct for decode (id -> string is exact); encode is approximate but
// adequate for interactive use. True BPE with merge ranks is a follow-up.
//
// SentencePiece convention: space is U+2581 (▁). We map ' ' <-> ▁ on the
// encode/decode boundary and prepend ▁ to the first word.
#pragma once

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace viper {

class Tokenizer {
public:
    bool load(const std::string& path) {
        FILE* f = std::fopen(path.c_str(), "rb");
        if (!f) { std::fprintf(stderr, "[tok] cannot open %s\n", path.c_str()); return false; }
        uint32_t hdr[5];
        if (std::fread(hdr, 4, 5, f) != 5) { std::fclose(f); return false; }
        n_ = hdr[0]; bos_ = hdr[1]; eos_ = hdr[2]; im_start_ = hdr[3]; im_end_ = hdr[4];
        strings_.resize(n_);
        for (uint32_t i = 0; i < n_; ++i) {
            uint32_t len;
            if (std::fread(&len, 4, 1, f) != 1) { std::fclose(f); return false; }
            strings_[i].resize(len);
            if (len && std::fread(&strings_[i][0], 1, len, f) != len) { std::fclose(f); return false; }
        }
        std::fclose(f);
        std::printf("[tok] vocab %d tokens, bos=%d eos=%d im_start=%d im_end=%d\n",
                    n_, bos_, eos_, im_start_, im_end_);
        return true;
    }

    int bos() const { return (int)bos_; }
    int eos() const { return (int)eos_; }
    int im_start() const { return (int)im_start_; }
    int im_end() const { return (int)im_end_; }

    // Greedy longest-match encode. Special tokens (<|...|>) are matched first.
    std::vector<int32_t> encode(const std::string& text) const {
        std::vector<int32_t> ids;
        size_t i = 0;
        // SentencePiece: prepend ▁ if the text starts a sequence.
        std::string sp = "\xe2\x96\x81";  // ▁
        std::string t = sp + text;
        // Map spaces to ▁.
        for (size_t j = 0; j < t.size(); ++j)
            if (t[j] == ' ') t.replace(j, 1, sp), j += sp.size() - 1;

        while (i < t.size()) {
            // Special token match: <|...|>
            if (t[i] == '<') {
                size_t close = t.find("|>", i);
                if (close != std::string::npos) {
                    std::string cand = t.substr(i, close + 2 - i);
                    int id = find(cand);
                    if (id >= 0) { ids.push_back(id); i = close + 2; continue; }
                }
            }
            // Longest prefix match.
            int best_id = -1; size_t best_len = 0;
            for (uint32_t id = 0; id < n_; ++id) {
                const std::string& s = strings_[id];
                if (s.empty() || s.size() <= best_len) continue;
                if (i + s.size() > t.size()) continue;
                if (s[0] == '<') continue;  // specials handled above
                if (std::memcmp(t.data() + i, s.data(), s.size()) == 0) {
                    best_id = (int)id; best_len = s.size();
                }
            }
            if (best_id < 0) {
                // Byte fallback: skip one byte (unmatched).
                ++i;
            } else {
                ids.push_back(best_id);
                i += best_len;
            }
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
    int find(const std::string& s) const {
        for (uint32_t id = 0; id < n_; ++id)
            if (strings_[id] == s) return (int)id;
        return -1;
    }
    uint32_t n_ = 0, bos_ = 0, eos_ = 0, im_start_ = 0, im_end_ = 0;
    std::vector<std::string> strings_;
};

}  // namespace viper
