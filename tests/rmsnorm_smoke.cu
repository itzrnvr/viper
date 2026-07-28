/*
 * viper RMSNorm smoke test — standalone, no MSVC/CMake.
 *
 * PURPOSE: Verify the RMSNorm CUDA kernel compiles with nvcc and runs on
 *          the GPU, producing output within bf16 tolerance of a CPU FP32
 *          reference.
 *
 * BUILD:   tests\rmsnorm_smoke.bat   (vcvars64 + nvcc + run, one shot)
 *
 * RUN:     .\build\rmsnorm_smoke.exe
 *
 * SAFETY: Small tensors (rows=2, H=3072, total ~12 KB). No VRAM pressure.
 *         Safe to run on the actual 3070 Ti; we verify before any other
 *         engine code touches the GPU.
 */
#include "kernels/ops/rmsnorm_kernel.h"
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#define VIPER_CHECK(call)                                                       \
    do {                                                                        \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,    \
                    cudaGetErrorString(err));                                   \
            return 1;                                                           \
        }                                                                       \
    } while (0)

namespace {

// CPU FP32 reference: y = rsqrt(mean(x^2) + eps) * x * gamma
void rmsnorm_cpu_ref(const float* x, const float* gamma, float* out,
                     int rows, int H, float eps) {
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

int main() {
    constexpr int ROWS = 2;
    constexpr int H = 3072;
    constexpr float EPS = 1e-5f;
    constexpr float TOL = 1e-1f;  // bf16-typical: rsqrt amplifies small errors

    // 1. Print GPU info so we know we're really running on the 3070 Ti.
    int dev = 0;
    cudaDeviceProp prop{};
    VIPER_CHECK(cudaGetDevice(&dev));
    VIPER_CHECK(cudaGetDeviceProperties(&prop, dev));
    printf("Device %d: %s (sm_%d%d, %d SMs, %.1f GB)\n",
           dev, prop.name, prop.major, prop.minor, prop.multiProcessorCount,
           prop.totalGlobalMem / 1e9);

    // 2. Host buffers, random init.
    std::vector<float> h_x(ROWS * H);
    std::vector<float> h_gamma(H);
    std::vector<float> h_out_cpu(ROWS * H);
    std::vector<__nv_bfloat16> h_out_gpu(ROWS * H);

    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (auto& v : h_x) v = dist(rng);
    for (auto& v : h_gamma) v = dist(rng);

    // 3. CPU reference (FP32 throughout).
    rmsnorm_cpu_ref(h_x.data(), h_gamma.data(), h_out_cpu.data(), ROWS, H, EPS);

    // 4. Cast to bf16, copy to device.
    std::vector<__nv_bfloat16> h_x_bf(ROWS * H);
    std::vector<__nv_bfloat16> h_g_bf(H);
    for (int i = 0; i < ROWS * H; ++i) h_x_bf[i] = __float2bfloat16(h_x[i]);
    for (int i = 0; i < H; ++i) h_g_bf[i] = __float2bfloat16(h_gamma[i]);

    __nv_bfloat16 *d_x = nullptr, *d_g = nullptr, *d_out = nullptr;
    VIPER_CHECK(cudaMalloc(&d_x, ROWS * H * sizeof(__nv_bfloat16)));
    VIPER_CHECK(cudaMalloc(&d_g, H * sizeof(__nv_bfloat16)));
    VIPER_CHECK(cudaMalloc(&d_out, ROWS * H * sizeof(__nv_bfloat16)));

    VIPER_CHECK(cudaMemcpy(d_x, h_x_bf.data(), ROWS * H * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    VIPER_CHECK(cudaMemcpy(d_g, h_g_bf.data(), H * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));

    // 5. Run the kernel.
    VIPER_CHECK(viper::ops::rmsnorm_forward_bf16(d_x, d_g, d_out, ROWS, H, EPS, 0));
    VIPER_CHECK(cudaDeviceSynchronize());

    VIPER_CHECK(cudaMemcpy(h_out_gpu.data(), d_out, ROWS * H * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));

    // 6. Compare.
    float max_diff = 0.0f;
    for (int i = 0; i < ROWS * H; ++i) {
        float gpu_val = __bfloat162float(h_out_gpu[i]);
        float diff = std::fabs(gpu_val - h_out_cpu[i]);
        if (diff > max_diff) max_diff = diff;
    }

    printf("row 0 gpu first 8: ");
    for (int i = 0; i < 8; ++i) {
        printf("%.4f ", __bfloat162float(h_out_gpu[i]));
    }
    printf("\nrow 0 cpu first 8: ");
    for (int i = 0; i < 8; ++i) {
        printf("%.4f ", h_out_cpu[i]);
    }
    printf("\n");

    cudaFree(d_x);
    cudaFree(d_g);
    cudaFree(d_out);

    if (max_diff < TOL) {
        printf("[OK  ] RMSNorm smoke test PASS (max_diff=%.2e, tol=%.2e)\n", max_diff, TOL);
        return 0;
    } else {
        printf("[FAIL] RMSNorm smoke test FAIL (max_diff=%.2e, tol=%.2e)\n", max_diff, TOL);
        return 1;
    }
}
