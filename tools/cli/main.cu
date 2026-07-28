/*
 * viper_cli — end-to-end inference with n-gram speculative decoding.
 *
 *   viper_cli.exe --model artifacts\Nanbeige4.2-3B.viper
 *                 --vocab artifacts\vocab.bin --prompt "Hello" --max-tokens 128
 *
 * Spec decode: drafts K tokens from an n-gram cache, verifies them in a single
 * batch forward pass (weights read once for all K+1 tokens). Accepts the
 * longest matching prefix. Effective speedup = mean_accepted / 1.
 */
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <chrono>

#include "viper/model_impl.cuh"
#include "viper/tokenizer.h"

static std::string argval(int argc, char** argv, const char* key, const char* dflt) {
    for (int i = 1; i + 1 < argc; ++i)
        if (std::strcmp(argv[i], key) == 0) return argv[i + 1];
    return dflt;
}

// ---- N-gram cache for speculative drafting ----
// Maps a window of N tokens → the token that followed. Used to draft
// candidate continuations without running the model.
struct NgramCache {
    static constexpr int N = 3;
    static constexpr int SIZE = 1 << 16;
    static constexpr int MASK = SIZE - 1;

    struct Entry { uint32_t hash; int32_t value; };
    std::vector<Entry> table;

    NgramCache() : table(SIZE, {0, -1}) {}

    static uint32_t hash3(int32_t a, int32_t b, int32_t c) {
        uint32_t h = 2166136261u;
        h ^= (uint32_t)a; h *= 16777619u;
        h ^= (uint32_t)b; h *= 16777619u;
        h ^= (uint32_t)c; h *= 16777619u;
        return h;
    }

    void add(int32_t a, int32_t b, int32_t c, int32_t next) {
        uint32_t h = hash3(a, b, c) & MASK;
        table[h] = {h + 1, next};  // store hash+1 so 0 means empty
    }

    int32_t lookup(int32_t a, int32_t b, int32_t c) const {
        uint32_t h = hash3(a, b, c) & MASK;
        const Entry& e = table[h];
        if (e.hash == h + 1) return e.value;
        return -1;
    }

    // Chain-draft up to max_k tokens from history.
    int draft(const std::vector<int32_t>& hist, int32_t* out, int max_k) const {
        int n = hist.size();
        if (n < N) return 0;
        int32_t a = hist[n - 3], b = hist[n - 2], c = hist[n - 1];
        int k = 0;
        while (k < max_k) {
            int32_t next = lookup(a, b, c);
            if (next < 0) break;
            out[k++] = next;
            a = b; b = c; c = next;
        }
        return k;
    }

    // Record all n-grams from a token sequence into the cache.
    void ingest(const std::vector<int32_t>& tokens) {
        for (int i = 0; i + N < (int)tokens.size(); ++i)
            add(tokens[i], tokens[i+1], tokens[i+2], tokens[i+3]);
    }
};

int main(int argc, char** argv) {
    std::string modelp = argval(argc, argv, "--model", "D:/dev/viper/artifacts/nbg42.viper");
    std::string vocabp = argval(argc, argv, "--vocab", "D:/dev/viper/artifacts/vocab.bin");
    std::string prompt  = argval(argc, argv, "--prompt", "Hello, who are you?");
    int max_tokens = std::atoi(argval(argc, argv, "--max-tokens", "128").c_str());
    int spec_k     = std::atoi(argval(argc, argv, "--spec-k", "4").c_str());

    viper::Tokenizer tok;
    if (!tok.load(vocabp)) { std::fprintf(stderr, "[cli] tok load failed\n"); return 1; }
    std::printf("[cli] tokenizer loaded\n");

    viper::NanbeigeEngine engine;
    if (!engine.load(modelp)) { std::fprintf(stderr, "[cli] engine load failed\n"); return 1; }

    // ChatML template.
    std::string full = "<|im_start|>user\n" + prompt + "<|im_end|>\n<|im_start|>assistant\n";
    std::vector<int32_t> ids = tok.encode(full);
    std::printf("[cli] prompt tokens: %zu  spec_k=%d\n", ids.size(), spec_k);
    std::fflush(stdout);

    auto t0 = std::chrono::steady_clock::now();

    // Prefill.
    int32_t next = -1;
    for (size_t i = 0; i < ids.size(); ++i) {
        bool last = (i + 1 == ids.size());
        if (!engine.forward(ids[i], last, &next)) return 1;
    }
    auto t1 = std::chrono::steady_clock::now();
    double ttft = std::chrono::duration<double>(t1 - t0).count();

    // Generate with n-gram speculative decoding.
    NgramCache ngram;
    std::vector<int32_t> history;  // all committed tokens for n-gram context
    history.push_back(next);
    ngram.ingest(ids);  // seed cache from prompt
    ngram.ingest(history);

    int n_gen = 0;
    int n_steps = 0, n_accepted_total = 0;

    auto tg0 = std::chrono::steady_clock::now();

    while (n_gen < max_tokens) {
        // 1. Draft K tokens from n-gram cache.
        int32_t draft[8];
        int K = (spec_k > 0 && history.size() >= NgramCache::N)
              ? ngram.draft(history, draft, std::min(spec_k, 8)) : 0;
        ++n_steps;

        if (K > 0) {
            // 2. Batch forward: [next, draft0, ..., draftK-1].
            int32_t batch[9];
            batch[0] = next;
            for (int i = 0; i < K; ++i) batch[i + 1] = draft[i];
            int M = K + 1;
            int32_t predicted[9];
            if (!engine.forward_batch(batch, M, predicted)) { K = 0; }  // fallback
            else {
                // 3. Accept longest matching prefix.
                int accepted = 0;
                for (int i = 0; i < K; ++i) {
                    if (predicted[i] == draft[i]) accepted++;
                    else break;
                }
                n_accepted_total += accepted;

                // 4. Roll back rejected tokens from KV cache.
                if (K - accepted > 0)
                    engine.rollback(K - accepted);

                // 5. Commit 1 + accepted tokens.
                for (int i = 0; i <= accepted; ++i) {
                    int32_t t = predicted[i];
                    if (t == tok.eos() || t == tok.im_end()) goto done;
                    std::string piece = tok.decode(t);
                    std::fwrite(piece.data(), 1, piece.size(), stdout);
                    std::fflush(stdout);
                    ++n_gen;
                    history.push_back(t);

                    // Update n-gram cache.
                    int h = history.size();
                    if (h >= 4) {
                        ngram.add(history[h-4], history[h-3], history[h-2], t);
                    }
                }

                next = predicted[accepted];  // last committed token
                continue;
            }
        }

        // Fallback: single-token forward (no draft or batch failed).
        if (next == tok.eos() || next == tok.im_end()) break;
        std::string piece = tok.decode(next);
        std::fwrite(piece.data(), 1, piece.size(), stdout);
        std::fflush(stdout);
        ++n_gen;

        int32_t prev = next;
        if (!engine.forward(next, true, &next)) return 1;
        history.push_back(prev);
        int h = history.size();
        if (h >= 4)
            ngram.add(history[h-4], history[h-3], history[h-2], prev);
    }

done:
    auto tg1 = std::chrono::steady_clock::now();
    double gen_s = std::chrono::duration<double>(tg1 - tg0).count();
    double tps = gen_s > 0 ? n_gen / gen_s : 0.0;
    double accept_rate = n_steps > 0 ? (double)n_accepted_total / (n_steps * std::max(spec_k,1)) : 0;

    std::printf("\n[cli] ttft=%.2fs  gen=%d tok  %.1f tok/s  "
                "(steps=%d, drafted_accept_rate=%.1f%%)\n",
                ttft, n_gen, tps, n_steps, accept_rate * 100);
    return 0;
}
