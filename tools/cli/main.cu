// viper_cli — end-to-end inference: load .viper, tokenize, prefill, generate.
//
//   viper_cli.exe --model artifacts\nbg42.viper --vocab artifacts\vocab.bin
//                 --prompt "Hello" --max-tokens 128 [--serve-once]
//
// Streams generated text to stdout as it samples. Greedy decoding (v1).
// No CUDA Graphs (user constraint). All CUDA errors checked.

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <chrono>

#include "viper/model_impl.cuh"
#include "viper/tokenizer.h"

static std::string arg_val(int argc, char** argv, const char* key, const char* dflt) {
    for (int i = 1; i + 1 < argc; ++i)
        if (std::strcmp(argv[i], key) == 0) return argv[i + 1];
    return dflt;
}

int main(int argc, char** argv) {
    std::string model_p = arg_val(argc, argv, "--model", "D:/dev/viper/artifacts/nbg42.viper");
    std::string vocab_p = arg_val(argc, argv, "--vocab", "D:/dev/viper/artifacts/vocab.bin");
    std::string prompt  = arg_val(argc, argv, "--prompt", "Hello, who are you?");
    int max_tokens = std::atoi(arg_val(argc, argv, "--max-tokens", "128").c_str());

    viper::Tokenizer tok;
    if (!tok.load(vocab_p)) return 1;
    viper::NanbeigeEngine engine;
    if (!engine.load(model_p)) return 1;

    // ChatML template (Nanbeige default).
    std::string full = "<|im_start|>user\n" + prompt + "<|im_end|>\n<|im_start|>assistant\n";
    std::vector<int32_t> ids = tok.encode(full);
    std::printf("[cli] prompt tokens: %zu\n", ids.size());

    auto t0 = std::chrono::steady_clock::now();

    // Prefill: feed prompt tokens, logits only on the last.
    int32_t next = -1;
    for (size_t i = 0; i < ids.size(); ++i) {
        bool last = (i + 1 == ids.size());
        if (!engine.forward(ids[i], last, &next)) return 1;
    }
    auto t1 = std::chrono::steady_clock::now();
    double ttft = std::chrono::duration<double>(t1 - t0).count();

    // Generate.
    std::string out_text;
    int n_gen = 0;
    auto tg0 = std::chrono::steady_clock::now();
    for (int i = 0; i < max_tokens; ++i) {
        if (next == tok.eos() || next == tok.im_end()) break;
        out_text += tok.decode(next);
        std::string piece = tok.decode(next);
        std::fwrite(piece.data(), 1, piece.size(), stdout);
        std::fflush(stdout);
        ++n_gen;
        if (!engine.forward(next, true, &next)) return 1;
    }
    auto tg1 = std::chrono::steady_clock::now();
    double gen_s = std::chrono::duration<double>(tg1 - tg0).count();

    std::printf("\n[cli] ttft=%.2fs  gen=%d tok  %.1f tok/s\n",
                ttft, n_gen, gen_s > 0 ? n_gen / gen_s : 0.0);
    return 0;
}
