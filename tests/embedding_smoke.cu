/*
 * viper Embedding gather smoke test — standalone.
 *
 * PURPOSE: Verify the gather kernel copies the right rows from the
 *          embedding table.
 *
 * BUILD:   tests\embedding_smoke.bat
 * RUN:     .\build\embedding_smoke.exe
 */
#include "kernels/ops/embedding_kernel.h"
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <random>
#include <vector>

#define VIPER_CHECK(call) do {                                                  \
    cudaError_t err = (call);                                                   \
    if (err != cudaSuccess) {                                                   \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,        \
                cudaGetErrorString(err));                                       \
        return 1;                                                               \
    }                                                                           \
} while (0)

int main() {
    constexpr int V = 4096;
    constexpr int H = 3072;
    constexpr int B = 2;
    constexpr int T = 8;

    int dev = 0;
    cudaDeviceProp prop{};
    VIPER_CHECK(cudaGetDevice(&dev));
    VIPER_CHECK(cudaGetDeviceProperties(&prop, dev));
    printf("Device %d: %s\n", dev, prop.name);

    // Host: random table + token ids.
    std::vector<__nv_bfloat16> h_table(V * H);
    std::vector<int32_t> h_ids(B * T);
    std::vector<__nv_bfloat16> h_out(B * T * H);

    std::mt19937 rng(13);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (auto& v : h_table) v = __float2bfloat16(dist(rng));
    for (auto& v : h_ids) v = rng() % V;

    // CPU reference.
    std::vector<__nv_bfloat16> h_ref(B * T * H);
    for (int b = 0; b < B; ++b) {
        for (int t = 0; t < T; ++t) {
            int tok = h_ids[b * T + t];
            for (int i = 0; i < H; ++i) {
                h_ref[(b * T + t) * H + i] = h_table[tok * H + i];
            }
        }
    }

    __nv_bfloat16 *d_table, *d_out;
    int32_t* d_ids;
    VIPER_CHECK(cudaMalloc(&d_table, V * H * sizeof(__nv_bfloat16)));
    VIPER_CHECK(cudaMalloc(&d_ids, B * T * sizeof(int32_t)));
    VIPER_CHECK(cudaMalloc(&d_out, B * T * H * sizeof(__nv_bfloat16)));
    VIPER_CHECK(cudaMemcpy(d_table, h_table.data(), V * H * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    VIPER_CHECK(cudaMemcpy(d_ids, h_ids.data(), B * T * sizeof(int32_t), cudaMemcpyHostToDevice));

    VIPER_CHECK(viper::ops::embedding_gather_bf16_i32(d_table, d_ids, d_out, B, T, V, H, 0));
    VIPER_CHECK(cudaDeviceSynchronize());

    VIPER_CHECK(cudaMemcpy(h_out.data(), d_out, B * T * H * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));

    // Compare.
    int mismatches = 0;
    for (size_t i = 0; i < h_ref.size(); ++i) {
        if (h_out[i] != h_ref[i]) ++mismatches;
    }

    cudaFree(d_table);
    cudaFree(d_ids);
    cudaFree(d_out);

    if (mismatches == 0) {
        printf("[OK  ] Embedding smoke test PASS (no mismatches across %zu elements)\n", h_ref.size());
        return 0;
    } else {
        printf("[FAIL] Embedding smoke test FAIL (%d / %zu mismatches)\n", mismatches, h_ref.size());
        return 1;
    }
}
