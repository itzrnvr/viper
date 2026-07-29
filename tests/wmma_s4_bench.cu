/*
 * WMMA INT4 tensor core GEMV benchmark.
 * 
 * Uses mma.sync.aligned.m16n8k64.s4 to do INT4×INT4 matrix multiply.
 * The key insight: tensor cores eliminate the scalar dequantization
 * instructions that bottleneck the current Q4 GEMV at 44% bandwidth.
 *
 * Expected: 85% bandwidth (383 GB/s) — same data, no instruction overhead.
 */
#include <cstdio>
#include <chrono>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <mma.h>

using namespace nvcuda;

// WMMA s4 fragment dimensions: M=16, N=8, K=64
// A: [16, 64] INT4 row_major (activation, padded to M=16)
// B: [64, 8] INT4 col_major (weights)
// C: [16, 8] INT32 accumulator

// Repack Q4 row-major weights to WMMA col-major format.
// Input: [N, K] row-major packed INT4 (2 nibbles per byte)
// Output: [K, N] col-major packed INT4 (2 nibbles per byte)
// Each output byte at [k, n] has: low nibble = B[k][2n], high nibble = B[k][2n+1]
// Wait, col_major means K varies fastest.
// B[k][n] is stored at byte offset (k + n*K) / 2
// For k even: low nibble. For k odd: high nibble.
void repack_q4_to_colmajor(
    const uint8_t* row_major,  // [N, K/2] packed
    uint8_t* col_major,         // [N, K/2] packed, col-major
    int N, int K) {
    // For each output byte position [k_pair, n]:
    // k_pair = k/2, where k = 0, 2, 4, ...
    // col_major byte at (k_pair + n * K/2)
    // Contains: low nibble = B[k][n] = weight[n][k], high nibble = B[k+1][n] = weight[n][k+1]
    for (int n = 0; n < N; ++n) {
        for (int k = 0; k < K; k += 2) {
            // Extract weight[n][k] and weight[n][k+1] from row-major
            int byte_idx = n * (K/2) + k/2;
            uint8_t packed = row_major[byte_idx];
            int w0 = (packed & 0xF) - 8;      // Already signed [-8, 7]
            int w1 = ((packed >> 4) & 0xF) - 8;
            // Convert back to unsigned nibble [0, 15] for INT4 storage: add 8
            uint8_t u0 = (uint8_t)(w0 + 8);    // [0, 15]
            uint8_t u1 = (uint8_t)(w1 + 8);
            // Store in col-major: byte at (k/2 + n * K/2)
            col_major[k/2 + n * (K/2)] = u0 | (u1 << 4);
        }
    }
}

// WMMA s4 GEMV kernel.
// Each warp computes 8 output channels (one N-tile).
// Block has 8 warps → 64 output channels per block.
__global__ void wmma_s4_gemv(
    const uint32_t* __restrict__ A_packed,  // [16, K/8] packed INT4 (activation, M=16 padded)
    const uint32_t* __restrict__ B_packed,  // [K/8, N] col-major packed INT4 (weights)
    int32_t* __restrict__ C,                 // [16, N] INT32 output
    int N, int K) {
    
    const int warp_id = threadIdx.x >> 5;
    const int lane_id = threadIdx.x & 31;
    
    // Each block handles 64 output channels (8 warps × 8 channels/warp)
    const int n_base = blockIdx.x * 8 * 8;  // 8 warps × 8 N per warp
    const int n_tile = n_base + warp_id * 8;  // This warp's starting N
    
    if (n_tile >= N) return;
    
    // WMMA fragments
    wmma::fragment<wmma::matrix_a, 16, 8, 64, wmma::precision::s4, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 8, 64, wmma::precision::s4, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 8, 64, int32_t> c_frag;
    
    wmma::fill_fragment(c_frag, 0);
    
    // Iterate over K in tiles of 64
    const int n_k_tiles = K / 64;
    for (int kt = 0; kt < n_k_tiles; ++kt) {
        // Load A tile: [16, 64] from A_packed at K-offset kt*64
        // A is row-major packed INT4. Leading dim = K (in elements).
        // Each row has 64 INT4 = 32 bytes = 8 uint32.
        // Row stride in uint32 = K/8.
        wmma::load_matrix_sync(a_frag, A_packed + kt * 8, K / 8);
        
        // Load B tile: [64, 8] from B_packed at K-offset kt*64, N-offset n_tile
        // B is col-major packed INT4. Leading dim = K (in elements).
        // Each column has 64 INT4 = 32 bytes = 8 uint32.
        // Column stride in uint32 = K/8.
        wmma::load_matrix_sync(b_frag, B_packed + kt * 8 + n_tile * (K/8), K / 8);
        
        // MMA: C += A × B
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }
    
    // Store result: [16, N] INT32
    // We only need row 0 (M=1 padded to 16)
    wmma::store_matrix_sync(C + n_tile * 16, c_frag, 16, wmma::mem_col_major);
}

int main() {
    // Test with q_proj dimensions: N=6144, K=3072
    const int N = 6144, K = 3072;
    const int M = 16;  // WMMA minimum
    
    // Allocate host data
    size_t a_sz = (size_t)M * K / 8 * sizeof(uint32_t);  // [16, K/8] packed INT4
    size_t b_sz = (size_t)N * K / 8 * sizeof(uint32_t);   // [K/8, N] col-major packed INT4
    size_t c_sz = (size_t)M * N * sizeof(int32_t);
    
    uint32_t* h_a = (uint32_t*)malloc(a_sz);
    uint32_t* h_b = (uint32_t*)malloc(b_sz);
    int32_t* h_c = (int32_t*)malloc(c_sz);
    
    // Initialize: all zeros (values = -8 in INT4)
    memset(h_a, 0x00, a_sz);
    memset(h_b, 0x00, b_sz);
    
    // Allocate device
    uint32_t* d_a; cudaMalloc(&d_a, a_sz);
    uint32_t* d_b; cudaMalloc(&d_b, b_sz);
    int32_t* d_c; cudaMalloc(&d_c, c_sz);
    cudaMemcpy(d_a, h_a, a_sz, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, b_sz, cudaMemcpyHostToDevice);
    
    // Grid: ceil(N/64) blocks, each block has 8 warps × 8 N per warp = 64 N
    int grid = (N + 63) / 64;
    
    // Warmup
    for (int i = 0; i < 10; i++) {
        wmma_s4_gemv<<<grid, 256>>>(d_a, d_b, d_c, N, K);
    }
    cudaDeviceSynchronize();
    
    // Benchmark
    int reps = 1000;
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < reps; i++) {
        wmma_s4_gemv<<<grid, 256>>>(d_a, d_b, d_c, N, K);
    }
    cudaDeviceSynchronize();
    auto t1 = std::chrono::steady_clock::now();
    double us = std::chrono::duration<double, std::micro>(t1 - t0).count() / reps;
    
    // Bandwidth: weight data only (B matrix)
    double b_bytes = (double)b_sz;
    double achieved = b_bytes / (us * 1e3);
    printf("WMMA s4 q_proj (N=%d K=%d): %.1f us, weight %.2f MB, %.1f GB/s (%.0f%%)\n",
           N, K, us, b_bytes/1e6, achieved, achieved/448*100);
    
    // Test down_proj: N=3072, K=10752
    const int N2 = 3072, K2 = 10752;
    size_t b2_sz = (size_t)N2 * K2 / 8 * sizeof(uint32_t);
    uint32_t* d_b2; cudaMalloc(&d_b2, b2_sz);
    uint32_t* d_a2; cudaMalloc(&d_a2, M * K2 / 8 * sizeof(uint32_t));
    int32_t* d_c2; cudaMalloc(&d_c2, M * N2 * sizeof(int32_t));
    cudaMemset(d_b2, 0, b2_sz);
    cudaMemset(d_a2, 0, M * K2 / 8 * sizeof(uint32_t));
    
    int grid2 = (N2 + 63) / 64;
    for (int i = 0; i < 10; i++)
        wmma_s4_gemv<<<grid2, 256>>>(d_a2, d_b2, d_c2, N2, K2);
    cudaDeviceSynchronize();
    
    t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < reps; i++)
        wmma_s4_gemv<<<grid2, 256>>>(d_a2, d_b2, d_c2, N2, K2);
    cudaDeviceSynchronize();
    t1 = std::chrono::steady_clock::now();
    us = std::chrono::duration<double, std::micro>(t1 - t0).count() / reps;
    achieved = (double)b2_sz / (us * 1e3);
    printf("WMMA s4 down_proj (N=%d K=%d): %.1f us, weight %.2f MB, %.1f GB/s (%.0f%%)\n",
           N2, K2, us, (double)b2_sz/1e6, achieved, achieved/448*100);
    
    free(h_a); free(h_b); free(h_c);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    cudaFree(d_a2); cudaFree(d_b2); cudaFree(d_c2);
    return 0;
}
