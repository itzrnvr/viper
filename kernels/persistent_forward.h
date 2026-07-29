#pragma once
#include <cuda_bf16.h>
#include <cstdint>
#include <cuda_runtime.h>

namespace viper { namespace ops {

// Launch the persistent decode kernel (entire forward in 1 CUDA launch).
// Requires device-side copies of layers array and KV cache pointer arrays.
cudaError_t launch_persistent_decode(
    const void* d_layers, int n_layers, int n_loops,
    const __nv_bfloat16* embed,
    const uint8_t* lm_packed, const __nv_bfloat16* lm_scales,
    const __nv_bfloat16* final_norm,
    __nv_bfloat16* x, __nv_bfloat16* x_norm,
    __nv_bfloat16* q_buf, __nv_bfloat16* kb_buf,
    __nv_bfloat16* attn_buf, __nv_bfloat16* g_buf, __nv_bfloat16* u_buf,
    __nv_bfloat16* logits_buf,
    __nv_bfloat16** kv_k_ptrs, __nv_bfloat16** kv_v_ptrs,
    const float* cos_pos, const float* sin_pos,
    int H, int I, int nQ, int nKVh, int HD, int vocab, int pos,
    int32_t token, float eps, float attn_scale,
    int32_t* out_token,
    int grid_size, cudaStream_t stream);

} }
