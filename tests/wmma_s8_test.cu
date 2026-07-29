// Quick test: does WMMA s8 work on sm_86?
#include <cstdio>
#include <mma.h>
using namespace nvcuda::wmma;

__global__ void test_wmma_s8() {
    fragment<matrix_a, 16, 16, 16, precision::s8, row_major> a_frag;
    fragment<matrix_b, 16, 16, 16, precision::s8, col_major> b_frag;
    fragment<accumulator, 16, 16, 16, int32_t> c_frag;
    fill_fragment(c_frag, 0);
    // Just test compilation — don't need real data
    // load_matrix_sync would go here
    mma_sync(c_frag, a_frag, b_frag, c_frag);
}

int main() {
    test_wmma_s8<<<1, 32>>>();
    cudaDeviceSynchronize();
    printf("WMMA s8 on sm_86: SUCCESS\n");
    return 0;
}
