/*
 * viper swiglu + Q8 quantize fused kernel.
 *
 * Computes silu(gate) * up AND quantizes to INT8 in one pass.
 * Saves 1 launch per layer-step (was: swiglu + quantize = 2 → now 1).
 *
 * Flow:
 *   for each element i:
 *     v = silu(gate[i]) * up[i] = gate[i] / (1 + exp(-gate[i])) * up[i]
 *     out_bf16[i] = v
 *     find per-group-of-64 max
 *   per-group quantize → out_q8 + out_scales
 */
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace viper {
namespace ops {

__global__ void swiglu_quantize_kernel(
    const __nv_bfloat16* __restrict__ gate,   // [I]
    const __nv_bfloat16* __restrict__ up,      // [I]
    __nv_bfloat16* __restrict__ out_bf16,      // [I] swiglu output (for reference)
    int8_t* __restrict__ out_q8,               // [I] INT8 quantized
    float* __restrict__ out_scales,            // [I/64] per-group scales
    int I) {

    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;

    extern __shared__ char smem[];
    __nv_bfloat16* sx = (__nv_bfloat16*)smem;  // [I] BF16 swiglu output

    // Phase 1: Compute swiglu and store to SMEM
    for (int i = tid; i < I; i += nthreads) {
        float g = __bfloat162float(gate[i]);
        float u = __bfloat162float(up[i]);
        float v = g / (1.0f + __expf(-g)) * u;  // silu(g) * u
        sx[i] = __float2bfloat16(v);
        out_bf16[i] = __float2bfloat16(v);
    }
    __syncthreads();

    // Phase 2: Per-warp parallel quantization (same as rmsnorm_quantize)
    const int ngroups = I / 64;
    const int wid = tid >> 5, lid = tid & 31;
    const int groups_per_warp = (ngroups + 7) / 8;
    const int my_g_start = wid * groups_per_warp;
    const int my_g_end = min(my_g_start + groups_per_warp, ngroups);

    for (int g = my_g_start; g < my_g_end; ++g) {
        const int base = g * 64;
        float gmax = 0.f;
        #pragma unroll
        for (int s = 0; s < 2; ++s) {
            int idx = base + lid + s * 32;
            if (idx < I) {
                float v = __bfloat162float(sx[idx]);
                gmax = fmaxf(gmax, fabsf(v));
            }
        }
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            gmax = fmaxf(gmax, __shfl_xor_sync(0xffffffff, gmax, off));

        float scale = fmaxf(gmax / 127.0f, 1e-8f);
        if (lid == 0) out_scales[g] = scale;

        #pragma unroll
        for (int s = 0; s < 2; ++s) {
            int idx = base + lid + s * 32;
            if (idx < I) {
                float v = __bfloat162float(sx[idx]);
                out_q8[idx] = (int8_t)__float2int_rn(v / scale);
            }
        }
    }
}

cudaError_t swiglu_quantize_bf16(
    const __nv_bfloat16* gate,
    const __nv_bfloat16* up,
    __nv_bfloat16* out_bf16,
    int8_t* out_q8,
    float* out_scales,
    int I,
    cudaStream_t stream) {
    if (!gate || !up || !out_bf16 || !out_q8 || !out_scales || I <= 0)
        return cudaErrorInvalidValue;
    if (I % 64 != 0) return cudaErrorInvalidValue;
    size_t smem = I * sizeof(__nv_bfloat16);
    swiglu_quantize_kernel<<<1, 256, smem, stream>>>(
        gate, up, out_bf16, out_q8, out_scales, I);
    return cudaGetLastError();
}

}  // namespace ops
}  // namespace viper
