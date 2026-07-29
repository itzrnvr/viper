// Benchmark: multi-M GEMV (no SMEM, L2-cached activations)
// vs current separate-blocks-per-token approach.
#include <cstdio>
#include <chrono>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

// Multi-M: each warp handles 1 output channel for ALL M tokens.
// Weights read ONCE from DRAM. Activations from L2 cache.
__global__ void multim_gemv(
    const uint8_t* __restrict__ w_packed,
    const __nv_bfloat16* __restrict__ w_scales,
    const __nv_bfloat16* __restrict__ x,       // [M, K]
    __nv_bfloat16* __restrict__ y,              // [M, N]
    int M, int N, int K) {
    const int warp_id = threadIdx.x >> 5;
    const int lane_id = threadIdx.x & 31;
    const int n = blockIdx.x * (blockDim.x >> 5) + warp_id;
    if (n >= N) return;

    const uint8_t* w_row = w_packed + (size_t)n * (K / 2);
    const __nv_bfloat16* s_row = w_scales + (size_t)n * (K / 64);

    float acc[8];  // max M=8
    for (int m = 0; m < M; ++m) acc[m] = 0.0f;

    const int n_bytes = K / 2;
    const int vec_end = n_bytes - (n_bytes % 128);

    for (int base = 0; base < vec_end; base += 128) {
        int byte_off = base + lane_id * 4;
        uint32_t p4 = *reinterpret_cast<const uint32_t*>(w_row + byte_off);

        int w0 = (p4 & 0xF) - 8, w1 = ((p4>>4)&0xF) - 8;
        int w2 = ((p4>>8)&0xF) - 8, w3 = ((p4>>12)&0xF) - 8;
        int w4 = ((p4>>16)&0xF) - 8, w5 = ((p4>>20)&0xF) - 8;
        int w6 = ((p4>>24)&0xF) - 8, w7 = ((p4>>28)&0xF) - 8;

        float sc = __bfloat162float(s_row[byte_off / 32]);
        int xk = byte_off * 2;

        #pragma unroll
        for (int m = 0; m < 8; ++m) {
            if (m >= M) break;
            const __nv_bfloat16* xm = x + (size_t)m * K;
            acc[m] += sc * (
                (float)w0 * __bfloat162float(xm[xk]) +
                (float)w1 * __bfloat162float(xm[xk+1]) +
                (float)w2 * __bfloat162float(xm[xk+2]) +
                (float)w3 * __bfloat162float(xm[xk+3]) +
                (float)w4 * __bfloat162float(xm[xk+4]) +
                (float)w5 * __bfloat162float(xm[xk+5]) +
                (float)w6 * __bfloat162float(xm[xk+6]) +
                (float)w7 * __bfloat162float(xm[xk+7]));
        }
    }

    #pragma unroll
    for (int m = 0; m < 8; ++m) {
        if (m >= M) break;
        float a = acc[m];
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            a += __shfl_xor_sync(0xffffffff, a, off);
        if (lane_id == 0) y[m * N + n] = __float2bfloat16(a);
    }
}

// Current approach: separate blocks per token (M in grid.y)
__global__ void separate_gemv(
    const uint8_t* __restrict__ w_packed,
    const __nv_bfloat16* __restrict__ w_scales,
    const __nv_bfloat16* __restrict__ x,
    __nv_bfloat16* __restrict__ y,
    int M, int N, int K) {
    const int m = blockIdx.y;
    const int warp_id = threadIdx.x >> 5;
    const int lane_id = threadIdx.x & 31;
    const int n = blockIdx.x * (blockDim.x >> 5) + warp_id;
    if (n >= N) return;

    const uint8_t* w_row = w_packed + (size_t)n * (K / 2);
    const __nv_bfloat16* s_row = w_scales + (size_t)n * (K / 64);
    const __nv_bfloat16* xm = x + (size_t)m * K;

    float acc = 0.0f;
    const int n_bytes = K / 2;
    const int vec_end = n_bytes - (n_bytes % 128);
    for (int base = 0; base < vec_end; base += 128) {
        int byte_off = base + lane_id * 4;
        uint32_t p4 = *reinterpret_cast<const uint32_t*>(w_row + byte_off);
        int w0=(p4&0xF)-8, w1=((p4>>4)&0xF)-8, w2=((p4>>8)&0xF)-8, w3=((p4>>12)&0xF)-8;
        int w4=((p4>>16)&0xF)-8, w5=((p4>>20)&0xF)-8, w6=((p4>>24)&0xF)-8, w7=((p4>>28)&0xF)-8;
        float sc = __bfloat162float(s_row[byte_off / 32]);
        int xk = byte_off * 2;
        acc += sc * ((float)w0*__bfloat162float(xm[xk]) + (float)w1*__bfloat162float(xm[xk+1])
                   + (float)w2*__bfloat162float(xm[xk+2]) + (float)w3*__bfloat162float(xm[xk+3])
                   + (float)w4*__bfloat162float(xm[xk+4]) + (float)w5*__bfloat162float(xm[xk+5])
                   + (float)w6*__bfloat162float(xm[xk+6]) + (float)w7*__bfloat162float(xm[xk+7]));
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        acc += __shfl_xor_sync(0xffffffff, acc, off);
    if (lane_id == 0) y[m * N + n] = __float2bfloat16(acc);
}

int main() {
    const int N = 6144, K = 3072;
    const int Ms[] = {1, 2, 3, 4, 5};
    size_t w_sz = (size_t)N * K / 2;
    size_t ws_sz = (size_t)N * (K/64) * sizeof(__nv_bfloat16);
    size_t x_max = 5 * K * sizeof(__nv_bfloat16);
    size_t y_max = 5 * N * sizeof(__nv_bfloat16);

    uint8_t* d_w; cudaMalloc(&d_w, w_sz);
    __nv_bfloat16* d_ws; cudaMalloc(&d_ws, ws_sz);
    __nv_bfloat16* d_x; cudaMalloc(&d_x, x_max);
    __nv_bfloat16* d_y; cudaMalloc(&d_y, y_max);
    cudaMemset(d_w, 0x88, w_sz);
    cudaMemset(d_ws, 0x3C00, ws_sz);
    cudaMemset(d_x, 0x3C00, x_max);

    int reps = 500;

    printf("=== Multi-M vs Separate (q_proj N=%d K=%d) ===\n", N, K);
    printf("  M  | Multi-M (us)  Per-tok | Separate (us) Per-tok | Speedup\n");
    printf("  ---+-----------------------+------------------------+--------\n");

    for (int mi = 0; mi < 5; ++mi) {
        int M = Ms[mi];

        // Multi-M
        for (int i = 0; i < 10; i++)
            multim_gemv<<<(N+7)/8, 256>>>(d_w, d_ws, d_x, d_y, M, N, K);
        cudaDeviceSynchronize();
        auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < reps; i++)
            multim_gemv<<<(N+7)/8, 256>>>(d_w, d_ws, d_x, d_y, M, N, K);
        cudaDeviceSynchronize();
        auto t1 = std::chrono::steady_clock::now();
        double multi_us = std::chrono::duration<double, std::micro>(t1-t0).count() / reps;

        // Separate
        dim3 grid_sep((N+7)/8, M);
        for (int i = 0; i < 10; i++)
            separate_gemv<<<grid_sep, 256>>>(d_w, d_ws, d_x, d_y, M, N, K);
        cudaDeviceSynchronize();
        t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < reps; i++)
            separate_gemv<<<grid_sep, 256>>>(d_w, d_ws, d_x, d_y, M, N, K);
        cudaDeviceSynchronize();
        t1 = std::chrono::steady_clock::now();
        double sep_us = std::chrono::duration<double, std::micro>(t1-t0).count() / reps;

        printf("  %d  |   %7.1f    %5.1f   |   %7.1f    %5.1f    |  %.2fx\n",
               M, multi_us, multi_us/M, sep_us, sep_us/M, sep_us/multi_us);
    }

    // down_proj
    const int N2 = 3072, K2 = 10752;
    size_t w2 = (size_t)N2 * K2 / 2;
    uint8_t* d_w2; cudaMalloc(&d_w2, w2);
    __nv_bfloat16* d_ws2; cudaMalloc(&d_ws2, N2*(K2/64)*sizeof(__nv_bfloat16));
    __nv_bfloat16* d_x2; cudaMalloc(&d_x2, 5*K2*sizeof(__nv_bfloat16));
    __nv_bfloat16* d_y2; cudaMalloc(&d_y2, 5*N2*sizeof(__nv_bfloat16));
    cudaMemset(d_w2, 0x88, w2);
    cudaMemset(d_ws2, 0x3C00, N2*(K2/64)*sizeof(__nv_bfloat16));
    cudaMemset(d_x2, 0x3C00, 5*K2*sizeof(__nv_bfloat16));

    printf("\n=== Multi-M vs Separate (down_proj N=%d K=%d) ===\n", N2, K2);
    printf("  M  | Multi-M (us)  Per-tok | Separate (us) Per-tok | Speedup\n");
    printf("  ---+-----------------------+------------------------+--------\n");

    for (int mi = 0; mi < 5; ++mi) {
        int M = Ms[mi];
        for (int i = 0; i < 10; i++)
            multim_gemv<<<(N2+7)/8, 256>>>(d_w2, d_ws2, d_x2, d_y2, M, N2, K2);
        cudaDeviceSynchronize();
        auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < reps; i++)
            multim_gemv<<<(N2+7)/8, 256>>>(d_w2, d_ws2, d_x2, d_y2, M, N2, K2);
        cudaDeviceSynchronize();
        auto t1 = std::chrono::steady_clock::now();
        double multi_us = std::chrono::duration<double, std::micro>(t1-t0).count() / reps;

        dim3 grid_sep((N2+7)/8, M);
        for (int i = 0; i < 10; i++)
            separate_gemv<<<grid_sep, 256>>>(d_w2, d_ws2, d_x2, d_y2, M, N2, K2);
        cudaDeviceSynchronize();
        t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < reps; i++)
            separate_gemv<<<grid_sep, 256>>>(d_w2, d_ws2, d_x2, d_y2, M, N2, K2);
        cudaDeviceSynchronize();
        t1 = std::chrono::steady_clock::now();
        double sep_us = std::chrono::duration<double, std::micro>(t1-t0).count() / reps;

        printf("  %d  |   %7.1f    %5.1f   |   %7.1f    %5.1f    |  %.2fx\n",
               M, multi_us, multi_us/M, sep_us, sep_us/M, sep_us/multi_us);
    }

    cudaFree(d_w); cudaFree(d_ws); cudaFree(d_x); cudaFree(d_y);
    cudaFree(d_w2); cudaFree(d_ws2); cudaFree(d_x2); cudaFree(d_y2);
    return 0;
}
