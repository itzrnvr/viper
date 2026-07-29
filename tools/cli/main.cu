/*
 * viper_cli — end-to-end inference with Jacobi speculative decoding.
 *
 * Jacobi spec decode: uses the model's OWN predictions from the previous
 * step as draft tokens. Over multiple iterations, the guesses converge to
 * the correct sequence. No separate drafter model needed.
 *
 * Algorithm:
 *   1. Process [T0, G1, G2, ..., GK] in one batch → [P0, P1, ..., PK]
 *   2. Accept longest prefix where Gi == P(i-1)
 *   3. Use [P1, P2, ..., PK] as next step's guesses
 *   4. Repeat
 *
 * Acceptance rate depends on output predictability:
 *   - Code/math/factual: high (50-80%)
 *   - Creative text: lower (20-40%)
 */
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <chrono>

#include "viper/model_impl.cuh"
#include "viper/drafter.h"
#include "viper/tokenizer.h"

static std::string argval(int argc, char** argv, const char* key, const char* dflt) {
    for (int i = 1; i + 1 < argc; ++i)
        if (std::strcmp(argv[i], key) == 0) return argv[i + 1];
    return dflt;
}

// ---- N-gram cache for speculative drafting ----
struct NgramCache {
    static constexpr int N = 2;
    static constexpr int SIZE = 1 << 16;
    static constexpr int MASK = SIZE - 1;
    struct Entry { uint32_t hash; int32_t value; };
    std::vector<Entry> table;
    NgramCache() : table(SIZE, {0, -1}) {}
    static uint32_t hash2(int32_t a, int32_t b) {
        uint32_t h = 2166136261u; h ^= (uint32_t)a; h *= 16777619u;
        h ^= (uint32_t)b; h *= 16777619u; return h;
    }
    void add(int32_t a, int32_t b, int32_t next) {
        uint32_t h = hash2(a, b) & MASK; table[h] = {h + 1, next};
    }
    int32_t lookup(int32_t a, int32_t b) const {
        uint32_t h = hash2(a, b) & MASK; const Entry& e = table[h];
        return (e.hash == h + 1) ? e.value : -1;
    }
    int draft(const std::vector<int32_t>& hist, int32_t* out, int max_k) const {
        int n = hist.size(); if (n < N) return 0;
        int32_t a = hist[n-2], b = hist[n-1]; int k = 0;
        while (k < max_k) { int32_t v = lookup(a, b); if (v < 0) break; out[k++] = v; a = b; b = v; }
        return k;
    }
    void ingest(const std::vector<int32_t>& tokens) {
        for (int i = 0; i + N < (int)tokens.size(); ++i) add(tokens[i], tokens[i+1], tokens[i+2]);
    }
};

int main(int argc, char** argv) {
    std::string modelp = argval(argc, argv, "--model", "D:/dev/viper/artifacts/Nanbeige4.2-3B.viper");
    std::string vocabp = argval(argc, argv, "--vocab", "D:/dev/viper/artifacts/vocab.bin");
    std::string drafterp = argval(argc, argv, "--drafter", "");
    std::string prompt  = argval(argc, argv, "--prompt", "Hello, who are you?");
    int max_tokens = std::atoi(argval(argc, argv, "--max-tokens", "128").c_str());
    int spec_k     = std::atoi(argval(argc, argv, "--spec-k", "4").c_str());
    int fastMode  = std::atoi(argval(argc, argv, "--fast", "0").c_str());
    int usePersistent = std::atoi(argval(argc, argv, "--persistent", "0").c_str());
    int useGraph = std::atoi(argval(argc, argv, "--graph", "0").c_str());
    int lmPrune = std::atoi(argval(argc, argv, "--lm-head-prune", "0").c_str());
    int cacheType = std::atoi(argval(argc, argv, "--cache-type", "0").c_str());  // 0=bf16 1=q8 2=q6 3=q4
    if (usePersistent > 0 || useGraph > 0) spec_k = 0;  // graph/persistent: single-token decode

    viper::Tokenizer tok;
    if (!tok.load(vocabp)) { std::fprintf(stderr, "[cli] tok load failed\n"); return 1; }
    std::printf("[cli] tokenizer loaded\n");

    viper::NanbeigeEngine engine;
    engine.max_batch = std::max(spec_k + 1, 5);
    if (!engine.load(modelp)) { std::fprintf(stderr, "[cli] engine load failed\n"); return 1; }
    if (fastMode > 0) {
        engine.cfg.n_passes = 1;  // loop-0 only: halves weight reads
        std::printf("[cli] FAST MODE: n_passes=1 (22 layers, ~2x speed)\n");
    }
    if (lmPrune > 0) {
        engine.cfg.lm_prune = lmPrune;
        std::fprintf(stderr, "[cli] WARNING: lm_head pruning = %d (LOSSY: greedy output may differ)\n", lmPrune);
    }
    if (cacheType > 0) {
        std::fprintf(stderr, "[cli] ERROR: cache type %d not yet wired (Q8 allocation pending)\n", cacheType);
        return 1;
    }

    // Load EAGLE drafter if specified.
    viper::Drafter* drafter = nullptr;
    if (!drafterp.empty()) {
        drafter = new viper::Drafter();
        if (!drafter->load(drafterp)) {
            std::fprintf(stderr, "[cli] drafter load failed, using n-gram fallback\n");
            delete drafter; drafter = nullptr;
        } else {
            // Share embed/lm_head/final_norm with base model (saves 1.3 GB VRAM)
            drafter->set_shared(engine.get_embed(),
                                engine.get_lm_head_packed(), engine.get_lm_head_scales(),
                                engine.cfg.vocab, engine.cfg.hidden,
                                engine.get_final_norm());
            std::printf("[cli] drafter loaded (shared weights)\n");
        }
    }

    // ChatML template.
    std::string full = "<|im_start|>user\n" + prompt + "<|im_end|>\n<|im_start|>assistant\n";
    std::vector<int32_t> ids = tok.encode(full);
    std::printf("[cli] prompt tokens: %zu  spec_k=%d\n", ids.size(), spec_k);
    std::fflush(stdout);

    auto t0 = std::chrono::steady_clock::now();

    // Prefill: batch mode (fast) or sequential (exact).
    int prefill_batch = std::atoi(argval(argc, argv, "--prefill-batch", "8").c_str());
    int32_t next = -1;

    if (prefill_batch > 0 && ids.size() > 1) {
        // Batch prefill: process prompt in chunks (weight reads shared across M tokens).
        int32_t predicted[17];
        size_t i = 0;
        while (i < ids.size()) {
            int M = std::min((int)(ids.size() - i), engine.max_batch);
            if (M < 2) {
                if (!engine.forward(ids[i], true, &next)) return 1;
            } else {
                if (!engine.forward_batch(ids.data() + i, M, predicted)) return 1;
                next = predicted[M - 1];
            }
            i += M;
        }
    } else {
        for (size_t i = 0; i < ids.size(); ++i) {
            bool last = (i + 1 == ids.size());
            if (!engine.forward(ids[i], last, &next)) return 1;
        }
    }
    auto t1 = std::chrono::steady_clock::now();
    double ttft = std::chrono::duration<double>(t1 - t0).count();

    // N-gram speculative decoding.
    constexpr int MAX_K = 16;
    int K = std::min(spec_k, MAX_K);
    int32_t drafts[MAX_K];

    // Build n-gram cache from prompt + generation history.
    NgramCache ngram;
    ngram.ingest(ids);
    std::vector<int32_t> history(ids.begin(), ids.end());
    history.push_back(next);

    // Initialize drafts from n-gram cache.
    for (int i = 0; i < K; ++i) drafts[i] = 198;

    int n_gen = 0;
    int n_steps = 0, n_accepted_total = 0, n_batch_steps = 0;

    auto tg0 = std::chrono::steady_clock::now();

    while (n_gen < max_tokens) {
        ++n_steps;

        if (spec_k > 0) {
            // Draft K tokens: use EAGLE drafter if available, else n-gram.
            if (drafter) {
                drafter->reset();
                const __nv_bfloat16* hidden = engine.get_hidden();
                drafter->generate(hidden, next, drafts, K);
            } else {
                int n_drafted = ngram.draft(history, drafts, K);
                for (int i = n_drafted; i < K; ++i) drafts[i] = 198;
            }

            // Build batch: [next, draft1, ..., draftK]
            int32_t batch[MAX_K + 1];
            batch[0] = next;
            for (int i = 0; i < K; ++i) batch[i + 1] = drafts[i];
            int M = K + 1;

            int32_t predicted[MAX_K + 1];
            if (engine.forward_batch(batch, M, predicted)) {
                ++n_batch_steps;

                int accepted = 0;
                for (int i = 0; i < K; ++i) {
                    if (predicted[i] == drafts[i]) accepted++;
                    else break;
                }
                n_accepted_total += accepted;

                if (K - accepted > 0)
                    engine.rollback(K - accepted);

                for (int i = 0; i <= accepted; ++i) {
                    int32_t t = predicted[i];
                    if (t == tok.eos() || t == tok.im_end()) goto done;
                    std::string piece = tok.decode(t);
                    std::fwrite(piece.data(), 1, piece.size(), stdout);
                    std::fflush(stdout);
                    ++n_gen;
                    if (n_gen >= max_tokens) goto done;
                    // Update n-gram cache + history.
                    int h = history.size();
                    if (h >= 2) ngram.add(history[h-2], history[h-1], t);
                    history.push_back(t);
                }

                next = predicted[accepted];
                continue;
            }
        }

        // Fallback: single-token forward.
        if (next == tok.eos() || next == tok.im_end()) break;
        std::string piece = tok.decode(next);
        std::fwrite(piece.data(), 1, piece.size(), stdout);
        std::fflush(stdout);
        ++n_gen;

        if (useGraph > 0) {
            if (!engine.forward_graph(next, &next)) return 1;
        } else if (usePersistent > 0) {
            if (!engine.forward_persistent(next, &next)) return 1;
        } else {
            if (!engine.forward(next, true, &next)) return 1;
        }
    }

done:
    auto tg1 = std::chrono::steady_clock::now();
    double gen_s = std::chrono::duration<double>(tg1 - tg0).count();
    double tps = gen_s > 0 ? n_gen / gen_s : 0.0;
    double accept_rate = n_batch_steps > 0 ? (double)n_accepted_total / (n_batch_steps * K) : 0;
    double mean_accepted = n_batch_steps > 0 ? (double)n_accepted_total / n_batch_steps : 0;

    std::printf("\n[cli] ttft=%.2fs  gen=%d tok  %.1f tok/s  "
                "(steps=%d, batch=%d, mean_acc=%.2f, accept_rate=%.1f%%)\n",
                ttft, n_gen, tps, n_steps, n_batch_steps,
                mean_accepted, accept_rate * 100);
    return 0;
}
