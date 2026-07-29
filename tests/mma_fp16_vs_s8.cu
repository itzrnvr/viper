// Test: FP16 MMA (should work) vs INT8 MMA (failing)
#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

__global__ void test_fp16_mma() {
    // FP16 MMA - should work on sm_86
    uint32_t a0 = 0, a1 = 0, a2 = 0, a3 = 0;
    uint32_t b0 = 0, b1 = 0;
    float c0 = 0, c1 = 0, c2 = 0, c3 = 0;

    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f16.f16.f32.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1)
    );
    if (threadIdx.x == 0) printf("FP16 MMA: OK (result: %f)\n", c0);
}

__global__ void test_s8_mma() {
    // INT8 MMA
    int32_t a0 = 0x01010101, a1 = 0x01010101;
    int32_t b0 = 0x01010101;
    int32_t d0 = 0, d1 = 0, d2 = 0, d3 = 0;

    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s8.s8.s32.s32 "
        "{%0, %1, %2, %3}, {%4, %5}, {%6}, {%0, %1, %2, %3};\n"
        : "+r"(d0), "+r"(d1), "+r"(d2), "+r"(d3)
        : "r"(a0), "r"(a1), "r"(b0)
    );
    if (threadIdx.x == 0) printf("S8 MMA: OK (result: %d)\n", d0);
}

int main() {
    printf("=== Testing MMA on sm_86 ===\n");
    test_fp16_mma<<<1, 32>>>();
    cudaDeviceSynchronize();
    printf("FP16 status: %s\n", cudaGetLastError() == cudaSuccess ? "PASS" : cudaGetErrorString(cudaGetLastError()));

    test_s8_mma<<<1, 32>>>();
    cudaDeviceSynchronize();
    printf("S8 status: %s\n", cudaGetLastError() == cudaSuccess ? "PASS" : cudaGetErrorString(cudaGetLastError()));
    return 0;
}
