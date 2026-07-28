/*
 * viper RoPE smoke test — standalone, with debug prints.
 *
 * PURPOSE: Verify the RoPE CUDA kernel produces output within bf16
 *          tolerance of a CPU FP32 reference.
 *
 * BUILD:   tests\rope_smoke.bat
 * RUN:     .\build\rope_smoke.exe
 */
#include "kernels/ops/rope_kernel.h"
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cmath>
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

namespace {

void rope_cpu_ref(float* x, const float* cos_table, const float* sin_table,
                  int B, int H, int T, int D) {
    for (int b = 0; b < B; ++b) {
        for (int h = 0; h < H; ++h) {
            for (int t = 0; t < T; ++t) {
                int half = D / 2;
                for (int tid = 0; tid < D; ++tid) {
                    int partner = (tid < half) ? (tid + half) : (tid - half);
                    int x_idx = ((b * H + h) * T + t) * D + tid;
                    int p_idx = ((b * H + h) * T + t) * D + partner;
                    int cs_idx = t * D + tid;
                    float x_val = x[x_idx];
                    float x_partner = x[p_idx];
                    float c = cos_table[cs_idx];
                    float s = sin_table[cs_idx];
                    float rotate = (tid < half) ? -x_partner : x_partner;
                    x[x_idx] = x_val * c + rotate * s;
                }
            }
        }
    }
}

}  // namespace

int main() {
    constexpr int B = 1;
    constexpr int T = 4;
    constexpr int N_HEADS = 4;
    constexpr int HEAD_DIM = 128;
    constexpr float THETA = 70000000.0f;
    constexpr float TOL = 5e-2f;

    int dev = 0;
    cudaDeviceProp prop{};
    VIPER_CHECK(cudaGetDevice(&dev));
    VIPER_CHECK(cudaGetDeviceProperties(&prop, dev));
    printf("Device %d: %s\n", dev, prop.name);

    std::vector<float> h_x(B * N_HEADS * T * HEAD_DIM);
    std::vector<float> h_cos(T * HEAD_DIM);
    std::vector<float> h_sin(T * HEAD_DIM);

    std::mt19937 rng(7);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (auto& v : h_x) v = dist(rng);

    for (int t = 0; t < T; ++t) {
        int half = HEAD_DIM / 2;
        for (int i = 0; i < half; ++i) {
            float exponent = -2.0f * static_cast<float>(i) / static_cast<float>(HEAD_DIM);
            float inv_freq = std::exp(exponent * std::log(THETA));
            float angle = static_cast<float>(t) * inv_freq;
            h_cos[t * HEAD_DIM + i] = std::cos(angle);
            h_cos[t * HEAD_DIM + i + half] = std::cos(angle);
            h_sin[t * HEAD_DIM + i] = std::sin(angle);
            h_sin[t * HEAD_DIM + i + half] = std::sin(angle);
        }
    }

    std::vector<float> h_x_ref = h_x;
    rope_cpu_ref(h_x_ref.data(), h_cos.data(), h_sin.data(), B, N_HEADS, T, HEAD_DIM);

    std::vector<__nv_bfloat16> h_x_bf(B * N_HEADS * T * HEAD_DIM);
    for (size_t i = 0; i < h_x.size(); ++i) h_x_bf[i] = __float2bfloat16(h_x[i]);

    __nv_bfloat16 *d_x = nullptr;
    __nv_bfloat16 *d_k = nullptr;
    float *d_cos = nullptr, *d_sin = nullptr;
    VIPER_CHECK(cudaMalloc(&d_x, h_x_bf.size() * sizeof(__nv_bfloat16)));
    VIPER_CHECK(cudaMalloc(&d_k, h_x_bf.size() * sizeof(__nv_bfloat16)));
    VIPER_CHECK(cudaMalloc(&d_cos, h_cos.size() * sizeof(float)));
    VIPER_CHECK(cudaMalloc(&d_sin, h_sin.size() * sizeof(float)));

    VIPER_CHECK(cudaMemcpy(d_x, h_x_bf.data(), h_x_bf.size() * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    VIPER_CHECK(cudaMemcpy(d_k, h_x_bf.data(), h_x_bf.size() * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    VIPER_CHECK(cudaMemcpy(d_cos, h_cos.data(), h_cos.size() * sizeof(float), cudaMemcpyHostToDevice));
    VIPER_CHECK(cudaMemcpy(d_sin, h_sin.data(), h_sin.size() * sizeof(float), cudaMemcpyHostToDevice));

    VIPER_CHECK(viper::ops::rope_apply_inplace_bf16(d_x, d_k, d_cos, d_sin, B, N_HEADS, N_HEADS, T, HEAD_DIM, 0));
    VIPER_CHECK(cudaDeviceSynchronize());

    std::vector<__nv_bfloat16> h_x_out(h_x_bf.size());
    VIPER_CHECK(cudaMemcpy(h_x_out.data(), d_x, h_x_bf.size() * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));

    // Print first 4 + first-from-second-half values to compare.
    printf("GPU output (row 0, head 0, t 0):\n");
    for (int i = 0; i < 4; ++i) printf("  i=%d gpu=%.4f cpu=%.4f\n", i,
        __bfloat162float(h_x_out[i]), h_x_ref[i]);
    printf("GPU output (row 0, head 0, t 0, i 60..67):\n");
    for (int i = 60; i < 68; ++i) printf("  i=%d gpu=%.4f cpu=%.4f\n", i,
        __bfloat162float(h_x_out[i]), h_x_ref[i]);

    float max_diff = 0.0f;
    for (size_t i = 0; i < h_x_ref.size(); ++i) {
        float gpu_val = __bfloat162float(h_x_out[i]);
        float diff = std::fabs(gpu_val - h_x_ref[i]);
        if (diff > max_diff) max_diff = diff;
    }

    cudaFree(d_x);
    int max_idx = -1;
    int max_b = -1, max_t = -1, max_h = -1, max_i = -1;
    for (size_t i = 0; i < h_x_ref.size(); ++i) {
        float gpu_val = __bfloat162float(h_x_out[i]);
        float diff = std::fabs(gpu_val - h_x_ref[i]);
        if (diff > max_diff) {
            max_diff = diff;
            max_idx = (int)i;
            int total = B * N_HEADS * T * HEAD_DIM;
            int stride = N_HEADS * T * HEAD_DIM;
            int b = (int)i / stride;
            int rem = (int)i % stride;
            int h = rem / (T * HEAD_DIM);
            int rem2 = rem % (T * HEAD_DIM);
            int t = rem2 / HEAD_DIM;
            int ii = rem2 % HEAD_DIM;
            max_b = b; max_h = h; max_t = t; max_i = ii;
        }
    }
    printf("max diff at b=%d h=%d t=%d i=%d: gpu=%.4f cpu=%.4f (input=%.4f)\n",
        max_b, max_h, max_t, max_i,
        __bfloat162float(h_x_out[max_idx]), h_x_ref[max_idx], h_x[max_idx]);
    cudaFree(d_cos);
    cudaFree(d_sin);

    if (max_diff < TOL) {
        printf("[OK  ] RoPE smoke test PASS (max_diff=%.2e, tol=%.2e)\n", max_diff, TOL);
        return 0;
    } else {
        printf("[FAIL] RoPE smoke test FAIL (max_diff=%.2e, tol=%.2e)\n", max_diff, TOL);
        return 1;
    }
}
