/*
 * viper Linear Q4_G64 smoke test — standalone, dequantized reference.
 */
#include "kernels/ops/linear_kernel.h"
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <random>
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
    constexpr int M = 4, N = 256, K = 256;
    constexpr float TOL = 1.0f;  // bf16 round-trip on accumulated dot products

    int dev = 0;
    cudaDeviceProp prop{};
    VIPER_CHECK(cudaGetDevice(&dev));
    VIPER_CHECK(cudaGetDeviceProperties(&prop, dev));
    printf("Device %d: %s\n", dev, prop.name);

    std::vector<float> h_w_orig(N * K);
    std::vector<__nv_bfloat16> h_x(M * K);
    std::mt19937 rng(33);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (auto& v : h_w_orig) v = dist(rng);
    for (auto& v : h_x) v = __float2bfloat16(dist(rng));

    std::vector<uint8_t> h_w_packed(N * K / 2, 0);
    std::vector<__nv_bfloat16> h_w_scales(N * K / 64, 0.0f);
    std::vector<float> h_w_dequant(N * K, 0.0f);
    for (int n = 0; n < N; ++n) {
        for (int g = 0; g < K / 64; ++g) {
            float max_abs = 0.0f;
            for (int j = 0; j < 64; ++j) {
                max_abs = std::fmax(max_abs, std::fabs(h_w_orig[n * K + g * 64 + j]));
            }
            float scale = max_abs / 7.0f;
            if (scale < 1e-8f) scale = 1e-8f;
            h_w_scales[n * (K / 64) + g] = __float2bfloat16(scale);
            for (int j = 0; j < 64; ++j) {
                float w = h_w_orig[n * K + g * 64 + j];
                int stored = (int)std::round(w / scale) + 8;
                if (stored < 0) stored = 0;
                if (stored > 15) stored = 15;
                int byte_idx = (g * 64 + j) / 2;
                if (j % 2 == 0) {
                    h_w_packed[n * (K / 2) + byte_idx] |= (uint8_t)(stored & 0x0F);
                } else {
                    h_w_packed[n * (K / 2) + byte_idx] |= (uint8_t)((stored & 0x0F) << 4);
                }
                h_w_dequant[n * K + g * 64 + j] = (stored - 8) * scale;
            }
        }
    }

    std::vector<float> h_y_ref(M * N, 0.0f);
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float acc = 0.0f;
            for (int k = 0; k < K; ++k) {
                acc += h_w_dequant[n * K + k] * __bfloat162float(h_x[m * K + k]);
            }
            h_y_ref[m * N + n] = acc;
        }
    }

    uint8_t *d_w;
    __nv_bfloat16 *d_s, *d_x, *d_y;
    std::vector<__nv_bfloat16> h_y(M * N);
    VIPER_CHECK(cudaMalloc(&d_w, h_w_packed.size()));
    VIPER_CHECK(cudaMalloc(&d_s, h_w_scales.size() * 2));
    VIPER_CHECK(cudaMalloc(&d_x, h_x.size() * 2));
    VIPER_CHECK(cudaMalloc(&d_y, h_y.size() * 2));
    VIPER_CHECK(cudaMemcpy(d_w, h_w_packed.data(), h_w_packed.size(), cudaMemcpyHostToDevice));
    VIPER_CHECK(cudaMemcpy(d_s, h_w_scales.data(), h_w_scales.size() * 2, cudaMemcpyHostToDevice));
    VIPER_CHECK(cudaMemcpy(d_x, h_x.data(), h_x.size() * 2, cudaMemcpyHostToDevice));

    VIPER_CHECK(viper::ops::linear_q4_g64_bf16(d_w, d_s, d_x, d_y, M, N, K, 0));
    VIPER_CHECK(cudaDeviceSynchronize());
    VIPER_CHECK(cudaMemcpy(h_y.data(), d_y, h_y.size() * 2, cudaMemcpyDeviceToHost));

    float max_diff = 0.0f;
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float gpu_val = __bfloat162float(h_y[m * N + n]);
            float diff = std::fabs(gpu_val - h_y_ref[m * N + n]);
            if (diff > max_diff) max_diff = diff;
        }
    }

    cudaFree(d_w); cudaFree(d_s); cudaFree(d_x); cudaFree(d_y);

    printf("max abs diff vs dequantized reference: %.2e (tol=%.2e)\n", max_diff, TOL);
    if (max_diff < TOL) {
        printf("[OK  ] Linear Q4_G64 smoke test PASS (M=%d, N=%d, K=%d)\n", M, N, K);
        return 0;
    } else {
        printf("[FAIL] Linear Q4_G64 smoke test FAIL\n");
        return 1;
    }
}
