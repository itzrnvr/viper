// viper_cli — load a .viper artifact, run forward passes, print tokens.
//
// For v1: each decode step re-runs the full accumulated sequence through
// the model (no KV cache). O(n^2) but correct. Gives a real token so
// you can verify the engine works end-to-end. KV cache is a perf
// optimization that comes after.

#include "viper/model.h"
#include "viper/status.h"
#include "viper/tensor.h"
#include "viper/cuda_check.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <string>
#include <vector>
#include <fstream>

static std::vector<int32_t> simple_tokenize(const std::string& text) {
    std::vector<int32_t> ids;
    ids.push_back(166100);  // BOS
    size_t i = 0;
    while (i < text.size()) {
        while (i < text.size() && (text[i] == ' ' || text[i] == '\n' || text[i] == '\t')) ++i;
        size_t j = i;
        while (j < text.size() && text[j] != ' ' && text[j] != '\n' && text[j] != '\t') ++j;
        if (j > i) {
            uint32_t h = 2166136261u;
            for (size_t k = i; k < j; ++k) h = (h ^ text[k]) * 16777619u;
            ids.push_back(100 + (int)(h % 165900u));
        }
        i = j;
    }
    return ids;
}

int main(int argc, char** argv) {
    std::string artifact_path = "D:/dev/viper/artifacts/Nanbeige4.2-3B.viper";
    std::string prompt = "The capital of France is";
    int max_tokens = 10;
    bool decode_loop = false;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--artifact" && i + 1 < argc) artifact_path = argv[++i];
        else if (a == "--prompt" && i + 1 < argc) prompt = argv[++i];
        else if (a == "--max-tokens" && i + 1 < argc) max_tokens = std::atoi(argv[++i]);
        else if (a == "--loop") decode_loop = true;
        else { fprintf(stderr, "unknown arg: %s\n", a.c_str()); return 1; }
    }

    std::ifstream f(artifact_path, std::ios::binary | std::ios::ate);
    if (!f) { fprintf(stderr, "cannot open %s\n", artifact_path.c_str()); return 1; }
    size_t size = f.tellg();
    f.seekg(0, std::ios::beg);
    std::vector<uint8_t> data(size);
    f.read(reinterpret_cast<char*>(data.data()), size);
    f.close();
    printf("loaded artifact: %zu bytes\n", size);

    viper::NanbeigeModel model;
    viper::Status s = model.load(data.data(), data.size());
    if (!s.ok()) {
        fprintf(stderr, "model load failed\n");
        return 1;
    }
    printf("model loaded\n");

    std::vector<int32_t> ids = simple_tokenize(prompt);
    printf("prompt: %s  ->  %zu tokens\n", prompt.c_str(), ids.size());

    int32_t* d_ids;
    cudaMalloc(&d_ids, ids.size() * sizeof(int32_t));
    int32_t* d_next;
    cudaMalloc(&d_next, sizeof(int32_t));
    int32_t h_next = 0;

    auto t0 = std::chrono::steady_clock::now();
    for (int t = 0; t < max_tokens; ++t) {
        cudaMemcpy(d_ids, ids.data(), ids.size() * sizeof(int32_t), cudaMemcpyHostToDevice);
        int dummy_len = 0;
        s = model.forward(d_ids, (int)ids.size(), d_next, dummy_len);
        if (!s.ok()) {
            fprintf(stderr, "forward failed\n");
            cudaFree(d_ids); cudaFree(d_next);
            return 1;
        }
        cudaMemcpy(&h_next, d_next, sizeof(int32_t), cudaMemcpyDeviceToHost);
        printf("step %d: next token = %d\n", t + 1, h_next);
        if (h_next == 166101) { printf("EOS\n"); break; }
        if (decode_loop) ids.push_back(h_next);
    }
    auto t1 = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double>(t1 - t0).count();
    if (max_tokens > 1) {
        printf("\n%d steps in %.2f sec = %.2f steps/sec\n", max_tokens, elapsed, max_tokens / elapsed);
    }

    cudaFree(d_ids);
    cudaFree(d_next);
    return 0;
}
