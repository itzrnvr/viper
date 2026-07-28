/*
 * viper RMSNorm test
 *
 * PURPOSE: Verify the CUDA RMSNorm kernel matches a CPU FP32 reference
 *          within bf16 tolerance (max abs diff < 5e-3).
 *
 * USAGE: built into the tests target; runs as part of the test suite.
 *
 * SAFETY: small tensors only (rows=2, H=3072) — no GPU memory pressure.
 */
#include "rmsnorm_kernel.h"
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cmath>
#include <cstdio>
#include <vector>
#include <random>

#define VIPER_CHECK(call)                                                      \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,   \
                    cudaGetErrorString(err));                                  \
            return 1;                                                          \
        }                                                                      \
    } while (0)

namespace {

// CPU reference: y = rsqrt(mean(x^2) + eps) * x * gamma, all fp32.
void rmsnorm_cpu_ref(const float* x, const float* gamma, float* out, int rows, int H, float eps) {
    for (int r = 0; r < rows; ++r) {
        float ss = 0.0f;
        for (int i = 0; i < H; ++i) {
            float v = x[r * H + i];
            ss += v * v;
        }
        float rsqrt_val = 1.0f / std::sqrt(ss / static_cast<float>(H) + eps);
        for (int i = 0; i < H; ++i) {
            out[r * H + i] = (x[r * H + i] * rsqrt_val) * gamma[i];
        }
    }
}

}  // namespace

int test_rmsnorm() {
    constexpr int ROWS = 2;
    constexpr int H = 3072;
    constexpr float EPS = 1e-5f;
    constexpr float TOL = 5e-3f;  // bf16 tolerance

    // Host buffers.
    std::vector<float> h_x(ROWS * H);
    std::vector<float> h_gamma(H);
    std::vector<float> h_out_cpu(ROWS * H);
    std::vector<float> h_out_gpu(ROWS * H);

    // Random init.
    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (auto& v : h_x) v = dist(rng);
    for (auto& v : h_gamma) v = dist(rng);

    // CPU reference (fp32 throughout).
    rmsnorm_cpu_ref(h_x.data(), h_gamma.data(), h_out_cpu.data(), ROWS, H, EPS);

    // Cast inputs to bf16, copy to device.
    std::vector<__nv_bfloat16> h_x_bf(ROWS * H);
    std::vector<__nv_bfloat16> h_g_bf(H);
    std::vector<__nv_bfloat16> h_out_bf(ROWS * H);
    for (int i = 0; i < ROWS * H; ++i) h_x_bf[i] = __float2bfloat16(h_x[i]);
    for (int i = 0; i < H; ++i) h_g_bf[i] = __float2bfloat16(h_gamma[i]);

    __nv_bfloat16 *d_x, *d_g, *d_out;
    VIPER_CHECK(cudaMalloc(&d_x, ROWS * H * sizeof(__nv_bfloat16)));
    VIPER_CHECK(cudaMalloc(&d_g, H * sizeof(__nv_bfloat16)));
    VIPER_CHECK(cudaMalloc(&d_out, ROWS * H * sizeof(__nv_bfloat16)));

    VIPER_CHECK(cudaMemcpy(d_x, h_x_bf.data(), ROWS * H * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    VIPER_CHECK(cudaMemcpy(d_g, h_g_bf.data(), H * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));

    // Run kernel.
    VIPER_CHECK(viper::ops::rmsnorm_forward_bf16(d_x, d_g, d_out, ROWS, H, EPS, 0));

    VIPER_CHECK(cudaMemcpy(h_out_bf.data(), d_out, ROWS * H * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));

    // Compare: cast bf16 -> fp32, compute max abs diff vs fp32 CPU ref.
    float max_diff = 0.0f;
    for (int i = 0; i < ROWS * H; ++i) {
        float gpu_val = __bfloat162float(h_out_bf[i]);
        float diff = std::fabs(gpu_val - h_out_cpu[i]);
        if (diff > max_diff) max_diff = diff;
    }

    cudaFree(d_x);
    cudaFree(d_g);
    cudaFree(d_out);

    if (max_diff < TOL) {
        printf("RMSNorm test PASS (max_diff=%.2e, tol=%.2e)\n", max_diff, TOL);
        return 0;
    } else {
        printf("RMSNorm test FAIL (max_diff=%.2e, tol=%.2e)\n", max_diff, TOL);
        return 1;
    }
}
