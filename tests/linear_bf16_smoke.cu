/*
 * viper Linear BF16 smoke test — standalone, against CPU FP32 reference.
 *
 * Checks both absolute diff (loose, for bf16 round-trip) AND argmax
 * agreement (which is what lm_head actually feeds for sampling).
 */
#include "kernels/ops/linear_bf16_kernel.h"
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
    constexpr int M = 4, N = 512, K = 512;
    constexpr float TOL = 1.0f;

    int dev = 0;
    cudaDeviceProp prop{};
    VIPER_CHECK(cudaGetDevice(&dev));
    VIPER_CHECK(cudaGetDeviceProperties(&prop, dev));
    printf("Device %d: %s\n", dev, prop.name);

    std::vector<__nv_bfloat16> h_w(N * K);
    std::vector<__nv_bfloat16> h_x(M * K);
    std::mt19937 rng(47);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (auto& v : h_w) v = __float2bfloat16(dist(rng));
    for (auto& v : h_x) v = __float2bfloat16(dist(rng));

    std::vector<float> h_y_ref(M * N, 0.0f);
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float acc = 0.0f;
            for (int k = 0; k < K; ++k) {
                acc += __bfloat162float(h_w[n * K + k]) * __bfloat162float(h_x[m * K + k]);
            }
            h_y_ref[m * N + n] = acc;
        }
    }

    __nv_bfloat16 *d_w, *d_x, *d_y;
    std::vector<__nv_bfloat16> h_y(M * N);
    VIPER_CHECK(cudaMalloc(&d_w, h_w.size() * 2));
    VIPER_CHECK(cudaMalloc(&d_x, h_x.size() * 2));
    VIPER_CHECK(cudaMalloc(&d_y, h_y.size() * 2));
    VIPER_CHECK(cudaMemcpy(d_w, h_w.data(), h_w.size() * 2, cudaMemcpyHostToDevice));
    VIPER_CHECK(cudaMemcpy(d_x, h_x.data(), h_x.size() * 2, cudaMemcpyHostToDevice));

    VIPER_CHECK(viper::ops::linear_bf16(d_w, d_x, d_y, M, N, K, 0));
    VIPER_CHECK(cudaDeviceSynchronize());
    VIPER_CHECK(cudaMemcpy(h_y.data(), d_y, h_y.size() * 2, cudaMemcpyDeviceToHost));

    float max_diff = 0.0f;
    for (size_t i = 0; i < h_y.size(); ++i) {
        float gpu_val = __bfloat162float(h_y[i]);
        float diff = std::fabs(gpu_val - h_y_ref[i]);
        if (diff > max_diff) max_diff = diff;
    }

    int argmax_correct = 0;
    for (int m = 0; m < M; ++m) {
        int ref_arg = 0; float ref_max = -1e30f;
        int gpu_arg = 0; float gpu_max = -1e30f;
        for (int n = 0; n < N; ++n) {
            if (h_y_ref[m * N + n] > ref_max) { ref_max = h_y_ref[m * N + n]; ref_arg = n; }
            float gv = __bfloat162float(h_y[m * N + n]);
            if (gv > gpu_max) { gpu_max = gv; gpu_arg = n; }
        }
        if (ref_arg == gpu_arg) ++argmax_correct;
    }

    cudaFree(d_w); cudaFree(d_x); cudaFree(d_y);

    printf("max abs diff: %.2e (tol=%.2e)\n", max_diff, TOL);
    printf("argmax agreement: %d / %d\n", argmax_correct, M);
    if (max_diff < TOL && argmax_correct == M) {
        printf("[OK  ] Linear BF16 smoke test PASS\n");
        return 0;
    } else {
        printf("[FAIL] Linear BF16 smoke test FAIL\n");
        return 1;
    }
}
