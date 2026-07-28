/*
 * viper RMSNorm forward kernel — implementation
 *
 * PURPOSE: see rmsnorm_kernel.h
 *
 * IMPLEMENTATION:
 *   Grid:  rows blocks (one per row)
 *   Block: 128 threads (4 warps), each processes H_TILE elements.
 *          H_TILE must be a multiple of 8 for float4 vectorized loads.
 *
 * CORRECTNESS:
 *   - All math in fp32 until the final cast to bf16.
 *   - Variance is mean of squares (standard RMSNorm).
 *   - rsqrt uses __frsqrt_rn (round-to-nearest IEEE).
 *
 * SAFETY:
 *   - No global allocations inside the kernel.
 *   - SMEM usage: 16 bytes (broadcast scalar across warps).
 *   - Stream-checked; caller passes stream for async.
 */
#include "rmsnorm_kernel.h"
#include <cuda_runtime.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

namespace viper {
namespace ops {

template <int H_TILE>
__global__ void rmsnorm_bf16_kernel(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ gamma,
    __nv_bfloat16* __restrict__ out,
    int H,
    float eps) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;

    const __nv_bfloat16* x_row = x + row * H;
    __nv_bfloat16* out_row = out + row * H;

    float x_reg[H_TILE];
    float g_reg[H_TILE];

    float ss = 0.0f;

    #pragma unroll
    for (int i = 0; i < H_TILE; ++i) {
        const int idx = tid * H_TILE + i;
        if (idx < H) {
            x_reg[i] = __bfloat162float(x_row[idx]);
            g_reg[i] = __bfloat162float(gamma[idx]);
            ss += x_reg[i] * x_reg[i];
        } else {
            x_reg[i] = 0.0f;
            g_reg[i] = 1.0f;
        }
    }

    // Warp reduce.
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        ss += __shfl_xor_sync(0xffffffff, ss, offset);
    }

    __shared__ float warp_sums[4];
    const int warp_id = tid >> 5;
    const int lane_id = tid & 31;
    if (lane_id == 0) {
        warp_sums[warp_id] = ss;
    }
    __syncthreads();

    float total_ss = 0.0f;
    if (warp_id == 0) {
        if (lane_id < 4) {
            total_ss = warp_sums[lane_id];
        } else {
            total_ss = 0.0f;
        }
        #pragma unroll
        for (int offset = 2; offset > 0; offset >>= 1) {
            total_ss += __shfl_xor_sync(0xffffffff, total_ss, offset);
        }
        if (lane_id == 0) {
            warp_sums[0] = total_ss;
        }
    }
    __syncthreads();

    const float mean_ss = warp_sums[0] / static_cast<float>(H);
    const float rsqrt_val = rsqrtf(mean_ss + eps);

    #pragma unroll
    for (int i = 0; i < H_TILE; ++i) {
        const int idx = tid * H_TILE + i;
        if (idx < H) {
            float val = (x_reg[i] * rsqrt_val) * g_reg[i];
            out_row[idx] = __float2bfloat16(val);
        }
    }
}

cudaError_t rmsnorm_forward_bf16(
    const __nv_bfloat16* x,
    const __nv_bfloat16* gamma,
    __nv_bfloat16* out,
    int rows,
    int H,
    float eps,
    cudaStream_t stream) {
    if (!x || !gamma || !out || H <= 0 || rows <= 0) {
        return cudaErrorInvalidValue;
    }
    if (H % 8 != 0) {
        return cudaErrorInvalidValue;
    }

    constexpr int BLOCK = 128;
    const int h_tile = (H + BLOCK - 1) / BLOCK;
    const int h_tile_aligned = ((h_tile + 7) / 8) * 8;

    dim3 grid(rows);
    dim3 block(BLOCK);

    switch (h_tile_aligned) {
        case 8:
            rmsnorm_bf16_kernel<8><<<grid, block, 0, stream>>>(x, gamma, out, H, eps);
            break;
        case 16:
            rmsnorm_bf16_kernel<16><<<grid, block, 0, stream>>>(x, gamma, out, H, eps);
            break;
        case 24:
            rmsnorm_bf16_kernel<24><<<grid, block, 0, stream>>>(x, gamma, out, H, eps);
            break;
        case 32:
            rmsnorm_bf16_kernel<32><<<grid, block, 0, stream>>>(x, gamma, out, H, eps);
            break;
        case 40:
            rmsnorm_bf16_kernel<40><<<grid, block, 0, stream>>>(x, gamma, out, H, eps);
            break;
        case 48:
            rmsnorm_bf16_kernel<48><<<grid, block, 0, stream>>>(x, gamma, out, H, eps);
            break;
        default:
            return cudaErrorInvalidValue;
    }

    return cudaGetLastError();
}

}  // namespace ops
}  // namespace viper
