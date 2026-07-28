/*
 * viper forward profile — measures per-op time on the real model.
 * Used to find where the 10.6 tok/s is going and to guide megakernel design.
 */
#include "viper/model_impl.cuh"
#include "viper/tokenizer.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <chrono>
#include <vector>

#define VIPER_CHECK(call) do {                                                  \
    cudaError_t _e = (call);                                                    \
    if (_e != cudaSuccess) {                                                    \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,        \
                cudaGetErrorString(_e));                                       \
        return 1;                                                               \
    }                                                                           \
} while (0)

int main() {
    viper::NanbeigeEngine engine;
    if (!engine.load("D:/dev/viper/artifacts/Nanbeige4.2-3B.viper")) {
        fprintf(stderr, "load failed\n");
        return 1;
    }
    viper::Tokenizer tok;
    if (!tok.load("D:/dev/viper/artifacts/vocab.bin")) {
        fprintf(stderr, "tok failed\n");
        return 1;
    }

    std::string full = "