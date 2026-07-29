/*
 * DP4A GEMV v4: grid-stride + per-lane scale application + single warp reduce.
 *
 * Key insight: apply weight_scale[g] × act_scale[g] per-lane DURING the loop,
 * accumulate to float, then do ONE warp reduce at the end.
 * Eliminates 3 shfl_xor per group (saves ~100+ instructions per warp).
 */
#include <cstdio>
#include <chrono>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

__global__ void quantize_int8_kernel(
    const __nv_bfloat16* __restrict__ x, int8_t* __restrict__ xq,
    __nv_bfloat16* __restrict__ xs, int K) {
    const int g = blockIdx.x;
    const int tid = threadIdx.x;
    const int start = g * 64;
    float my_val = (tid < 64) ? fabsf(__bfloat162float(x[start + tid])) : 0.0f;
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        my_val = fmaxf(my_val, __shfl_xor_sync(0xffffffff, my_val, off));
    __shared__ float wm[2];
    if ((tid & 31) == 0) wm[tid >> 5] = my_val;
    __syncthreads();
    if (tid < 32) {
        float mx = fmaxf(fmaxf(wm[0], wm[1]), 1e-8f);
        float sc = mx / 127.0f;
        if (tid == 0) { xs[g] = __float2bfloat16(sc); wm[0] = 1.0f / sc; }
    }
    __syncthreads();
    float inv = wm[0];
    if (tid < 64) {
        int q = __float2int_rn(__bfloat162float(x[start + tid]) * inv);
        xq[start + tid] = (int8_t)(q > 127 ? 127 : (q < -128 ? -128 : q));
    }
}

// DP4A GEMV v4: grid-stride, per-lane scale, single warp reduce.
// Each lane processes quads at stride 32. Applies group scale per-quad.
__global__ void dp4a_gemv_v4(
    const uint32_t* __restrict__ w32,       // [N, K/8] Q4 packed as uint32
    const __nv_bfloat16* __restrict__ w_scales, // [N, K/64]
    const int32_t* __restrict__ xq32,        // [K/4] INT8 activations as int32
    const __nv_bfloat16* __restrict__ xs,     // [K/64] act scales
    __nv_bfloat16* __restrict__ y,            // [N]
    const __nv_bfloat16* __restrict__ residual,
    int N, int K) {
    const int warp_id = threadIdx.x >> 5;
    const int lane_id = threadIdx.x & 31;
    const int n = blockIdx.x * (blockDim.x >> 5) + warp_id;
    if (n >= N) return;

    const uint32_t* w_row = w32 + (size_t)n * (K / 8);
    const __nv_bfloat16* s_row = w_scales + (size_t)n * (K / 64);
    const int n_quads = K / 8;

    float acc = 0.0f;

    // Grid-stride: lane t processes quads t, t+32, t+64, ...
    for (int i = lane_id; i < n_quads; i += 32) {
        // Load 4 bytes Q4 weights
        uint32_t packed4 = w_row[i];

        // Q4 → INT8 via byte_perm + vsubss4
        uint32_t lo = packed4 & 0x0F0F0F0F;
        uint32_t hi = (packed4 >> 4) & 0x0F0F0F0F;
        int32_t slo = __vsubss4(lo, 0x08080808);
        int32_t shi = __vsubss4(hi, 0x08080808);
        int32_t w_lo = __byte_perm(slo, shi, 0x5140);
        int32_t w_hi = __byte_perm(slo, shi, 0x7362);

        // Load INT8 activations
        int32_t x_lo = xq32[i * 2];
        int32_t x_hi = xq32[i * 2 + 1];

        // DP4A
        int32_t raw = __dp4a(w_lo, x_lo, 0);
        raw = __dp4a(w_hi, x_hi, raw);

        // Apply per-group scales (group = quad / 8)
        int g = i / 8;
        float ws = __bfloat162float(s_row[g]);
        float xsv = __bfloat162float(xs[g]);
        acc += (float)raw * ws * xsv;
    }

    // Single warp reduce (5 shfl_xor total, not per-group)
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        acc += __shfl_xor_sync(0xffffffff, acc, off);

    if (lane_id == 0) {
        if (residual)
            acc += __bfloat162float(residual[n]);
        y[n] = __float2bfloat16(acc);
    }
}

int main() {
    const int N = 6144, K = 3072;
    size_t w_sz = (size_t)N * K / 8 * sizeof(uint32_t);
    size_t ws_sz = (size_t)N * (K/64) * sizeof(__nv_bfloat16);
    size_t x_sz = K * sizeof(__nv_bfloat16);
    size_t xq_sz = K / 4 * sizeof(int32_t);
    size_t xs_sz = K/64 * sizeof(__nv_bfloat16);
    size_t y_sz = N * sizeof(__nv_bfloat16);

    uint32_t* d_w; cudaMalloc(&d_w, w_sz);
    __nv_bfloat16* d_ws; cudaMalloc(&d_ws, ws_sz);
    __nv_bfloat16* d_x; cudaMalloc(&d_x, x_sz);
    int32_t* d_xq; cudaMalloc(&d_xq, xq_sz);
    __nv_bfloat16* d_xs; cudaMalloc(&d_xs, xs_sz);
    __nv_bfloat16* d_y; cudaMalloc(&d_y, y_sz);

    srand(42);
    uint8_t* h_w = (uint8_t*)malloc(w_sz);
    __nv_bfloat16* h_x = (__nv_bfloat16*)malloc(x_sz);
    __nv_bfloat16* h_ws = (__nv_bfloat16*)malloc(ws_sz);
    for (size_t i = 0; i < w_sz; ++i) h_w[i] = rand() & 0xFF;
    for (int i = 0; i < N*(K/64); ++i) h_ws[i] = __float2bfloat16(0.5f + 0.001f*(rand()%1000));
    for (int i = 0; i < K; ++i) h_x[i] = __float2bfloat16((rand()%200-100)/100.0f);
    cudaMemcpy(d_w, h_w, w_sz, cudaMemcpyHostToDevice);
    cudaMemcpy(d_ws, h_ws, ws_sz, cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, h_x, x_sz, cudaMemcpyHostToDevice);

    quantize_int8_kernel<<<K/64, 64>>>(d_x, (int8_t*)d_xq, d_xs, K);
    cudaDeviceSynchronize();

    int reps = 1000;
    for (int i = 0; i < 10; i++)
        dp4a_gemv_v4<<<(N+7)/8, 256>>>(d_w, d_ws, d_xq, d_xs, d_y, nullptr, N, K);
    cudaDeviceSynchronize();

    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < reps; i++)
        dp4a_gemv_v4<<<(N+7)/8, 256>>>(d_w, d_ws, d_xq, d_xs, d_y, nullptr, N, K);
    cudaDeviceSynchronize();
    auto t1 = std::chrono::steady_clock::now();
    double us = std::chrono::duration<double, std::micro>(t1 - t0).count() / reps;
    double bw = (double)w_sz / (us * 1e3);
    printf("=== DP4A v4 q_proj (N=%d K=%d) ===\n", N, K);
    printf("  GEMV: %.1f us, %.1f GB/s (%.0f%%)\n", us, bw, bw/448*100);

    // Quantize+GEMV
    t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < reps; i++) {
        quantize_int8_kernel<<<K/64, 64>>>(d_x, (int8_t*)d_xq, d_xs, K);
        dp4a_gemv_v4<<<(N+7)/8, 256>>>(d_w, d_ws, d_xq, d_xs, d_y, nullptr, N, K);
    }
    cudaDeviceSynchronize();
    t1 = std::chrono::steady_clock::now();
    us = std::chrono::duration<double, std::micro>(t1 - t0).count() / reps;
    printf("  Quantize+GEMV: %.1f us\n", us);

    // down_proj
    const int N2 = 3072, K2 = 10752;
    size_t w2 = (size_t)N2 * K2 / 8 * sizeof(uint32_t);
    size_t ws2 = (size_t)N2*(K2/64)*sizeof(__nv_bfloat16);
    size_t x2 = K2*sizeof(__nv_bfloat16);
    size_t xq2 = K2/4*sizeof(int32_t);
    size_t xs2 = K2/64*sizeof(__nv_bfloat16);
    uint32_t* d_w2; cudaMalloc(&d_w2, w2);
    __nv_bfloat16* d_ws2; cudaMalloc(&d_ws2, ws2);
    __nv_bfloat16* d_x2; cudaMalloc(&d_x2, x2);
    int32_t* d_xq2; cudaMalloc(&d_xq2, xq2);
    __nv_bfloat16* d_xs2; cudaMalloc(&d_xs2, xs2);
    __nv_bfloat16* d_y2; cudaMalloc(&d_y2, N2*sizeof(__nv_bfloat16));
    cudaMemset(d_w2, 0x88, w2);
    cudaMemset(d_ws2, 0x3C00, ws2);
    cudaMemset(d_x2, 0x3C00, x2);

    quantize_int8_kernel<<<K2/64, 64>>>(d_x2, (int8_t*)d_xq2, d_xs2, K2);
    cudaDeviceSynchronize();

    for (int i = 0; i < 10; i++)
        dp4a_gemv_v4<<<(N2+7)/8, 256>>>(d_w2, d_ws2, d_xq2, d_xs2, d_y2, nullptr, N2, K2);
    cudaDeviceSynchronize();

    t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < reps; i++)
        dp4a_gemv_v4<<<(N2+7)/8, 256>>>(d_w2, d_ws2, d_xq2, d_xs2, d_y2, nullptr, N2, K2);
    cudaDeviceSynchronize();
    t1 = std::chrono::steady_clock::now();
    us = std::chrono::duration<double, std::micro>(t1 - t0).count() / reps;
    bw = (double)w2 / (us * 1e3);
    printf("\n=== DP4A v4 down_proj (N=%d K=%d) ===\n", N2, K2);
    printf("  GEMV: %.1f us, %.1f GB/s (%.0f%%)\n", us, bw, bw/448*100);

    t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < reps; i++) {
        quantize_int8_kernel<<<K2/64, 64>>>(d_x2, (int8_t*)d_xq2, d_xs2, K2);
        dp4a_gemv_v4<<<(N2+7)/8, 256>>>(d_w2, d_ws2, d_xq2, d_xs2, d_y2, nullptr, N2, K2);
    }
    cudaDeviceSynchronize();
    t1 = std::chrono::steady_clock::now();
    us = std::chrono::duration<double, std::micro>(t1 - t0).count() / reps;
    printf("  Quantize+GEMV: %.1f us\n", us);

    printf("\n--- Scalar Q4: ~198 GB/s (44%%). DP4A v3 (no scales): ~324 GB/s (72%%) ---\n");

    free(h_w); free(h_x); free(h_ws);
    cudaFree(d_w); cudaFree(d_ws); cudaFree(d_x); cudaFree(d_xq);
    cudaFree(d_xs); cudaFree(d_y);
    cudaFree(d_w2); cudaFree(d_ws2); cudaFree(d_x2); cudaFree(d_xq2);
    cudaFree(d_xs2); cudaFree(d_y2);
    return 0;
}
