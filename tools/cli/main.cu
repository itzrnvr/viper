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
#include "viper/tokenizer.h"

static std::string argval(int argc, char** argv, const char* key, const char* dflt) {
    for (int i = 1; i + 1 < argc; ++i)
        if (std::strcmp(argv[i], key) == 0) return argv[i + 1];
    return dflt;
}

int main(int argc, char** argv) {
    std::string modelp = argval(argc, argv, "--model", "D:/dev/viper/artifacts/Nanbeige4.2-3B.viper");
    std::string vocabp = argval(argc, argv, "--vocab", "D:/dev/viper/artifacts/vocab.bin");
    std::string prompt  = argval(argc, argv, "--prompt", "Hello, who are you?");
    int max_tokens = std::atoi(argval(argc, argv, "--max-tokens", "128").c_str());
    int spec_k     = std::atoi(argval(argc, argv, "--spec-k", "4").c_str());

    viper::Tokenizer tok;
    if (!tok.load(vocabp)) { std::fprintf(stderr, "[cli] tok load failed\n"); return 1; }
    std::printf("[cli] tokenizer loaded\n");

    viper::NanbeigeEngine engine;
    engine.max_batch = std::max(spec_k + 1, 5);
    if (!engine.load(modelp)) { std::fprintf(stderr, "[cli] engine load failed\n"); return 1; }

    // ChatML template.
    std::string full = "<|im_start|>user\n" + prompt + "<|im_end|>\n<|im_start|>assistant\n";
    std::vector<int32_t> ids = tok.encode(full);
    std::printf("[cli] prompt tokens: %zu  spec_k=%d\n", ids.size(), spec_k);
    std::fflush(stdout);

    auto t0 = std::chrono::steady_clock::now();

    // Prefill (sequential for output quality).
    int32_t next = -1;
    for (size_t i = 0; i < ids.size(); ++i) {
        bool last = (i + 1 == ids.size());
        if (!engine.forward(ids[i], last, &next)) return 1;
    }
    auto t1 = std::chrono::steady_clock::now();
    double ttft = std::chrono::duration<double>(t1 - t0).count();

    // Jacobi speculative decoding.
    constexpr int MAX_K = 8;
    int K = std::min(spec_k, MAX_K);

    // Draft buffer: initialized with space token (common continuation).
    // Will be updated with model predictions each step.
    int32_t drafts[MAX_K];
    int32_t prev_preds[MAX_K + 1];  // predictions from last batch
    bool have_preds = false;

    // Initialize drafts to a common token (newline/space)
    for (int i = 0; i < K; ++i) drafts[i] = 198;  // '\n' in most tokenizers

    int n_gen = 0;
    int n_steps = 0, n_accepted_total = 0, n_batch_steps = 0;

    auto tg0 = std::chrono::steady_clock::now();

    while (n_gen < max_tokens) {
        ++n_steps;

        if (spec_k > 0) {
            // Build batch: [next, draft1, draft2, ..., draftK]
            int32_t batch[MAX_K + 1];
            batch[0] = next;
            for (int i = 0; i < K; ++i) batch[i + 1] = drafts[i];
            int M = K + 1;

            int32_t predicted[MAX_K + 1];
            if (engine.forward_batch(batch, M, predicted)) {
                ++n_batch_steps;

                // Accept longest matching prefix.
                // predicted[0] should be the real next token (given `next`)
                // predicted[i] is the model's prediction given draft[i]
                // Accept draft[i] if predicted[i] == draft[i]
                int accepted = 0;
                for (int i = 0; i < K; ++i) {
                    if (predicted[i] == drafts[i]) accepted++;
                    else break;
                }
                n_accepted_total += accepted;

                // Roll back rejected draft tokens from KV cache.
                if (K - accepted > 0)
                    engine.rollback(K - accepted);

                // Commit 1 + accepted tokens.
                // predicted[0] is always committed (it's the verified next token).
                // predicted[1..accepted] are also committed (verified drafts).
                for (int i = 0; i <= accepted; ++i) {
                    int32_t t = predicted[i];
                    if (t == tok.eos() || t == tok.im_end()) goto done;
                    std::string piece = tok.decode(t);
                    std::fwrite(piece.data(), 1, piece.size(), stdout);
                    std::fflush(stdout);
                    ++n_gen;
                    if (n_gen >= max_tokens) goto done;
                }

                // Update next token.
                next = predicted[accepted];

                // Update drafts: use predictions from this step.
                // predicted[accepted+1..K] are predictions for positions after
                // the accepted tokens. Use them as next step's drafts.
                for (int i = 0; i < K; ++i) {
                    int src = accepted + 1 + i;
                    if (src <= K)
                        drafts[i] = predicted[src];
                    else
                        drafts[i] = 198;  // fallback
                }
                have_preds = true;
                continue;
            }
        }

        // Fallback: single-token forward.
        if (next == tok.eos() || next == tok.im_end()) break;
        std::string piece = tok.decode(next);
        std::fwrite(piece.data(), 1, piece.size(), stdout);
        std::fflush(stdout);
        ++n_gen;

        if (!engine.forward(next, true, &next)) return 1;
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
