// Test: PTX mma.sync.s8 on sm_86 (bypasses WMMA C++ API)
#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>

__global__ void test_mma_s8_ptx() {
    // mma.sync.aligned.m16n8k32.row.col.s8.s8.s32.s32
    // A: 2 b32 per thread (packed s8), B: 1 b32, D/C: 4 s32
    int32_t a0 = 0x01010101, a1 = 0x01010101;  // A fragment (4 s8 per b32)
    int32_t b0 = 0x01010101;                     // B fragment
    int32_t d0 = 0, d1 = 0, d2 = 0, d3 = 0;     // D/C accumulators

    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s8.s8.s32.s32 "
        "{%0, %1, %2, %3}, {%4, %5}, {%6}, {%0, %1, %2, %3};\n"
        : "+r"(d0), "+r"(d1), "+r"(d2), "+r"(d3)
        : "r"(a0), "r"(a1), "r"(b0)
    );

    if (threadIdx.x == 0) {
        printf("PTX mma.sync.s8 result: d0=%d d1=%d d2=%d d3=%d\n", d0, d1, d2, d3);
    }
}

// Also test s4
__global__ void test_mma_s4_ptx() {
    // mma.sync.aligned.m16n8k64.row.col.s4.s4.s32.s32
    int32_t a0 = 0x88888888, a1 = 0x88888888;  // A fragment (8 s4 per b32)
    int32_t b0 = 0x88888888, b1 = 0x88888888;  // B fragment
    int32_t d0 = 0, d1 = 0, d2 = 0, d3 = 0;

    asm volatile(
        "mma.sync.aligned.m16n8k64.row.col.s4.s4.s32.s32 "
        "{%0, %1, %2, %3}, {%4, %5}, {%6, %7}, {%0, %1, %2, %3};\n"
        : "+r"(d0), "+r"(d1), "+r"(d2), "+r"(d3)
        : "r"(a0), "r"(a1), "r"(b0), "r"(b1)
    );

    if (threadIdx.x == 0) {
        printf("PTX mma.sync.s4 result: d0=%d d1=%d d2=%d d3=%d\n", d0, d1, d2, d3);
    }
}

int main() {
    printf("Testing PTX mma.sync on sm_86...\n");
    test_mma_s8_ptx<<<1, 32>>>();
    cudaDeviceSynchronize();
    printf("s8: %s\n", cudaGetLastError() == cudaSuccess ? "OK" : cudaGetErrorString(cudaGetLastError()));

    test_mma_s4_ptx<<<1, 32>>>();
    cudaDeviceSynchronize();
    printf("s4: %s\n", cudaGetLastError() == cudaSuccess ? "OK" : cudaGetErrorString(cudaGetLastError()));
    return 0;
}
