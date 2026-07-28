/*
 * viper Sampling kernel — implementation (greedy).
 */
#include "sampling_kernel.h"
#include <cuda_runtime.h>

namespace viper {
namespace ops {

// One block per batch, 256 threads, each scans V/256 elements.
__global__ void sampling_greedy_kernel(
    const __nv_bfloat16* __restrict__ logits,
    int32_t* __restrict__ out_token,
    int B, int V) {
    const int b = blockIdx.x;
    if (b >= B) return;
    const int tid = threadIdx.x;
    const int block_size = blockDim.x;

    // Each thread finds the local argmax in its strided portion.
    float local_max = -INFINITY;
    int local_idx = 0;
    for (int v = tid; v < V; v += block_size) {
        float val = __bfloat162float(logits[b * V + v]);
        if (val > local_max) {
            local_max = val;
            local_idx = v;
        }
    }

    // Warp reduce.
    for (int offset = 16; offset > 0; offset >>= 1) {
        float other_val = __shfl_xor_sync(0xffffffff, local_max, offset);
        int other_idx = __shfl_xor_sync(0xffffffff, local_idx, offset);
        if (other_val > local_max || (other_val == local_max && other_idx < local_idx)) {
            local_max = other_val;
            local_idx = other_idx;
        }
    }

    // Block reduce via shared memory.
    __shared__ float warp_max[8];
    __shared__ int warp_idx[8];
    int warp_id = tid >> 5;
    int lane_id = tid & 31;
    if (lane_id == 0) {
        warp_max[warp_id] = local_max;
        warp_idx[warp_id] = local_idx;
    }
    __syncthreads();

    if (warp_id == 0) {
        float wm = (lane_id < (block_size + 31) / 32) ? warp_max[lane_id] : -INFINITY;
        int wi = (lane_id < (block_size + 31) / 32) ? warp_idx[lane_id] : 0;
        for (int offset = 4; offset > 0; offset >>= 1) {
            float other_val = __shfl_xor_sync(0xffffffff, wm, offset);
            int other_idx = __shfl_xor_sync(0xffffffff, wi, offset);
            if (other_val > wm || (other_val == wm && other_idx < wi)) {
                wm = other_val;
                wi = other_idx;
            }
        }
        if (lane_id == 0) {
            out_token[b] = wi;
        }
    }
}

cudaError_t sampling_greedy_bf16(
    const __nv_bfloat16* logits,
    int32_t* out_token,
    int B, int V, cudaStream_t stream) {
    if (!logits || !out_token || B <= 0 || V <= 0) {
        return cudaErrorInvalidValue;
    }
    dim3 grid(B);
    dim3 block(256);
    sampling_greedy_kernel<<<grid, block, 0, stream>>>(logits, out_token, B, V);
    return cudaGetLastError();
}

}  // namespace ops
}  // namespace viper
