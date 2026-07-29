/*
 * Clean DP4A GEMV: NO SMEM, reads INT8 activations from DRAM (L2 cached).
 * Per-group scales via 4-groups-per-iteration pattern.
 *
 * Pipeline: rmsnorm_kernel → quantize_int8_kernel → dp4a_gemv_clean
 */
#include <cstdio>
#include <chrono>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

// Quantize BF16 → INT8 with per-group-of-64 FP16 scale.
// Grid: K/64 blocks, 64 threads each. Fully parallel.
__global__ void quantize_int8_kernel(
    const __nv_bfloat16* __restrict__ x,    // [K] BF16
    int8_t* __restrict__ xq,                  // [K] INT8
    __nv_bfloat16* __restrict__ xs,           // [K/64] FP16 scales
    int K) {
    const int g = blockIdx.x;
    const int tid = threadIdx.x;
    const int start = g * 64;

    // Cooperative max-abs within group (64 threads, 64 elements)
    float my_val = (tid < 64) ? fabsf(__bfloat162float(x[start + tid])) : 0.0f;

    // Warp reduce (64 threads = 2 warps)
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        my_val = fmaxf(my_val, __shfl_xor_sync(0xffffffff, my_val, off));

    __shared__ float warp_max[2];
    if ((tid & 31) == 0) warp_max[tid >> 5] = my_val;
    __syncthreads();

    if (tid < 32) {
        float mx = fmaxf(warp_max[0], warp_max[1]);
        mx = fmaxf(mx, 1e-8f);
        float sc = mx / 127.0f;
        float inv = 1.0f / sc;
        if (tid == 0) xs[g] = __float2bfloat16(sc);
        // Broadcast inv via shared memory
        if (tid == 0) warp_max[0] = inv;
    }
    __syncthreads();
    float inv = warp_max[0];

    // Quantize
    if (tid < 64) {
        int q = __float2int_rn(__bfloat162float(x[start + tid]) * inv);
        xq[start + tid] = (int8_t)(q > 127 ? 127 : (q < -128 ? -128 : q));
    }
}

// DP4A GEMV: reads Q4 weights + INT8 activations from DRAM. No SMEM.
// 4 groups per warp iteration. Lanes 0-7→g0, 8-15→g1, 16-23→g2, 24-31→g3.
__global__ void dp4a_gemv_clean(
    const uint8_t* __restrict__ w_packed,       // [N, K/2] Q4
    const __nv_bfloat16* __restrict__ w_scales,   // [N, K/64]
    const int8_t* __restrict__ xq,                // [K] INT8
    const __nv_bfloat16* __restrict__ xs,          // [K/64]
    __nv_bfloat16* __restrict__ y,                 // [N] BF16
    const __nv_bfloat16* __restrict__ residual,    // optional
    int N, int K) {
    const int warp_id = threadIdx.x >> 5;
    const int lane_id = threadIdx.x & 31;
    const int n = blockIdx.x * (blockDim.x >> 5) + warp_id;
    if (n >= N) return;

    const uint8_t* w_row = w_packed + (size_t)n * (K / 2);
    const __nv_bfloat16* s_row = w_scales + (size_t)n * (K / 64);
    const int32_t* xq32 = reinterpret_cast<const int32_t*>(xq);

    const int n_groups = K / 64;
    const int group_in_lane = lane_id >> 3;
    const int quad_in_group = lane_id & 7;

    float acc = 0.0f;

    for (int gbase = 0; gbase < n_groups; gbase += 4) {
        int g = gbase + group_in_lane;
        if (g >= n_groups) break;

        int byte_off = g * 32 + quad_in_group * 4;
        uint32_t packed4 = *reinterpret_cast<const uint32_t*>(w_row + byte_off);

        // Q4 → INT8
        uint32_t lo = packed4 & 0x0F0F0F0F;
        uint32_t hi = (packed4 >> 4) & 0x0F0F0F0F;
        int32_t slo = __vsubss4(lo, 0x08080808);
        int32_t shi = __vsubss4(hi, 0x08080808);
        int32_t w_lo = __byte_perm(slo, shi, 0x5140);
        int32_t w_hi = __byte_perm(slo, shi, 0x7362);

        int elem = byte_off * 2;
        int32_t x_lo = xq32[elem / 4];
        int32_t x_hi = xq32[elem / 4 + 1];

        int32_t partial = __dp4a(w_lo, x_lo, 0);
        partial = __dp4a(w_hi, x_hi, partial);

        #pragma unroll
        for (int off = 4; off > 0; off >>= 1)
            partial += __shfl_xor_sync(0xffffffff, partial, off);

        if (quad_in_group == 0) {
            float ws = __bfloat162float(s_row[g]);
            float xsv = __bfloat162float(xs[g]);
            acc += (float)partial * ws * xsv;
        }
    }

    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        acc += __shfl_xor_sync(0xffffffff, acc, off);

    if (lane_id == 0) {
        if (residual)
            acc += __bfloat162float(residual[n]);
        y[n] = __float2bfloat16(acc);
    }
}

// Also provide fused2 version (2 matrices, blockIdx.z)
__global__ void dp4a_gemv_fused2_clean(
    const uint8_t* __restrict__ w0, const __nv_bfloat16* __restrict__ s0,
    const uint8_t* __restrict__ w1, const __nv_bfloat16* __restrict__ s1,
    const int8_t* __restrict__ xq, const __nv_bfloat16* __restrict__ xs,
    __nv_bfloat16* __restrict__ y0, __nv_bfloat16* __restrict__ y1,
    int N, int K) {
    const int z = blockIdx.z;
    const uint8_t* wp = (z == 0) ? w0 : w1;
    const __nv_bfloat16* sp = (z == 0) ? s0 : s1;
    __nv_bfloat16* yp = (z == 0) ? y0 : y1;
    const int warp_id = threadIdx.x >> 5;
    const int lane_id = threadIdx.x & 31;
    const int n = blockIdx.x * (blockDim.x >> 5) + warp_id;
    if (n >= N) return;

    const uint8_t* w_row = wp + (size_t)n * (K / 2);
    const __nv_bfloat16* s_row = sp + (size_t)n * (K / 64);
    const int32_t* xq32 = reinterpret_cast<const int32_t*>(xq);

    const int n_groups = K / 64;
    const int group_in_lane = lane_id >> 3;
    const int quad_in_group = lane_id & 7;

    float acc = 0.0f;
    for (int gbase = 0; gbase < n_groups; gbase += 4) {
        int g = gbase + group_in_lane;
        if (g >= n_groups) break;
        int byte_off = g * 32 + quad_in_group * 4;
        uint32_t p4 = *reinterpret_cast<const uint32_t*>(w_row + byte_off);
        uint32_t lo = p4 & 0x0F0F0F0F, hi = (p4 >> 4) & 0x0F0F0F0F;
        int32_t slo = __vsubss4(lo, 0x08080808), shi = __vsubss4(hi, 0x08080808);
        int32_t wl = __byte_perm(slo, shi, 0x5140), wh = __byte_perm(slo, shi, 0x7362);
        int elem = byte_off * 2;
        int32_t partial = __dp4a(wl, xq32[elem/4], 0);
        partial = __dp4a(wh, xq32[elem/4+1], partial);
        #pragma unroll
        for (int off = 4; off > 0; off >>= 1)
            partial += __shfl_xor_sync(0xffffffff, partial, off);
        if (quad_in_group == 0) {
            acc += (float)partial * __bfloat162float(s_row[g]) * __bfloat162float(xs[g]);
        }
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        acc += __shfl_xor_sync(0xffffffff, acc, off);
    if (lane_id == 0) yp[n] = __float2bfloat16(acc);
}

int main() {
    const int N = 6144, K = 3072;

    // Allocate
    size_t w_sz = (size_t)N * K / 2;
    size_t ws_sz = (size_t)N * (K/64) * sizeof(__nv_bfloat16);
    size_t x_sz = K * sizeof(__nv_bfloat16);
    size_t xq_sz = K * sizeof(int8_t);
    size_t xs_sz = K/64 * sizeof(__nv_bfloat16);
    size_t y_sz = N * sizeof(__nv_bfloat16);

    uint8_t* d_w; cudaMalloc(&d_w, w_sz);
    __nv_bfloat16* d_ws; cudaMalloc(&d_ws, ws_sz);
    __nv_bfloat16* d_x; cudaMalloc(&d_x, x_sz);
    int8_t* d_xq; cudaMalloc(&d_xq, xq_sz);
    __nv_bfloat16* d_xs; cudaMalloc(&d_xs, xs_sz);
    __nv_bfloat16* d_y; cudaMalloc(&d_y, y_sz);

    // Init with random data
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

    // Step 1: Quantize
    quantize_int8_kernel<<<K/64, 64>>>(d_x, d_xq, d_xs, K);
    cudaDeviceSynchronize();

    // Step 2: DP4A GEMV
    int reps = 1000;

    // Warmup
    for (int i = 0; i < 10; i++)
        dp4a_gemv_clean<<<(N+7)/8, 256>>>(d_w, d_ws, d_xq, d_xs, d_y, nullptr, N, K);
    cudaDeviceSynchronize();

    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < reps; i++)
        dp4a_gemv_clean<<<(N+7)/8, 256>>>(d_w, d_ws, d_xq, d_xs, d_y, nullptr, N, K);
    cudaDeviceSynchronize();
    auto t1 = std::chrono::steady_clock::now();
    double us = std::chrono::duration<double, std::micro>(t1 - t0).count() / reps;
    double bw = (double)w_sz / (us * 1e3);
    printf("=== DP4A Clean q_proj (N=%d K=%d) ===\n", N, K);
    printf("  GEMV: %.1f us, %.1f GB/s (%.0f%%)\n", us, bw, bw/448*100);

    // Also benchmark quantize+gemv together
    t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < reps; i++) {
        quantize_int8_kernel<<<K/64, 64>>>(d_x, d_xq, d_xs, K);
        dp4a_gemv_clean<<<(N+7)/8, 256>>>(d_w, d_ws, d_xq, d_xs, d_y, nullptr, N, K);
    }
    cudaDeviceSynchronize();
    t1 = std::chrono::steady_clock::now();
    us = std::chrono::duration<double, std::micro>(t1 - t0).count() / reps;
    printf("  Quantize+GEMV: %.1f us total\n", us);

    // down_proj
    const int N2 = 3072, K2 = 10752;
    size_t w2 = (size_t)N2 * K2 / 2;
    size_t ws2 = (size_t)N2*(K2/64)*sizeof(__nv_bfloat16);
    size_t x2 = K2*sizeof(__nv_bfloat16);
    size_t xq2 = K2*sizeof(int8_t);
    size_t xs2 = K2/64*sizeof(__nv_bfloat16);
    uint8_t* d_w2; cudaMalloc(&d_w2, w2);
    __nv_bfloat16* d_ws2; cudaMalloc(&d_ws2, ws2);
    __nv_bfloat16* d_x2; cudaMalloc(&d_x2, x2);
    int8_t* d_xq2; cudaMalloc(&d_xq2, xq2);
    __nv_bfloat16* d_xs2; cudaMalloc(&d_xs2, xs2);
    __nv_bfloat16* d_y2; cudaMalloc(&d_y2, N2*sizeof(__nv_bfloat16));
    cudaMemset(d_w2, 0x88, w2);
    cudaMemset(d_ws2, 0x3C00, ws2);
    cudaMemset(d_x2, 0x3C00, x2);

    quantize_int8_kernel<<<K2/64, 64>>>(d_x2, d_xq2, d_xs2, K2);
    cudaDeviceSynchronize();

    for (int i = 0; i < 10; i++)
        dp4a_gemv_clean<<<(N2+7)/8, 256>>>(d_w2, d_ws2, d_xq2, d_xs2, d_y2, nullptr, N2, K2);
    cudaDeviceSynchronize();

    t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < reps; i++)
        dp4a_gemv_clean<<<(N2+7)/8, 256>>>(d_w2, d_ws2, d_xq2, d_xs2, d_y2, nullptr, N2, K2);
    cudaDeviceSynchronize();
    t1 = std::chrono::steady_clock::now();
    us = std::chrono::duration<double, std::micro>(t1 - t0).count() / reps;
    bw = (double)w2 / (us * 1e3);
    printf("\n=== DP4A Clean down_proj (N=%d K=%d) ===\n", N2, K2);
    printf("  GEMV: %.1f us, %.1f GB/s (%.0f%%)\n", us, bw, bw/448*100);

    t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < reps; i++) {
        quantize_int8_kernel<<<K2/64, 64>>>(d_x2, d_xq2, d_xs2, K2);
        dp4a_gemv_clean<<<(N2+7)/8, 256>>>(d_w2, d_ws2, d_xq2, d_xs2, d_y2, nullptr, N2, K2);
    }
    cudaDeviceSynchronize();
    t1 = std::chrono::steady_clock::now();
    us = std::chrono::duration<double, std::micro>(t1 - t0).count() / reps;
    printf("  Quantize+GEMV: %.1f us total\n", us);

    printf("\n--- Current scalar Q4: ~198 GB/s (44%%) ---\n");

    free(h_w); free(h_x); free(h_ws);
    cudaFree(d_w); cudaFree(d_ws); cudaFree(d_x); cudaFree(d_xq);
    cudaFree(d_xs); cudaFree(d_y);
    cudaFree(d_w2); cudaFree(d_ws2); cudaFree(d_x2); cudaFree(d_xq2);
    cudaFree(d_xs2); cudaFree(d_y2);
    return 0;
}
