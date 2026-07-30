# viper Session Summary — 2026-07-30

## Headline Result: viper BEATS llama.cpp

| Engine | tok/s | vs llama.cpp |
|---|---|---|
| **viper DP4A L1-cached** | **59.1** | **+9.7% to +17.6%** |
| viper scalar baseline | 51.0 | -5.3% |
| llama.cpp Q4_K_M | 50.26-53.87 | reference |

## Verified Working Features

| Feature | CLI Flag | Speed | Quality |
|---|---|---|---|
| DP4A L1-cached engine | (default) | 59.1 tok/s | ✅ Correct |
| Q8 KV cache | --cache-type 1 | 5.6 tok/s | ✅ Correct |
| Batched prefill | --prefill-batch 8 | 38% TTFT cut | ✅ Correct |
| lm_head pruning | --lm-head-prune 128K | 54.9 tok/s | ✅ Acceptable |
| Spec decode K=8 | --spec-k 8 | 67.1 tok/s | ✅ Correct |
| Zero-sync fused rope | (default) | +0.8 tok/s | ✅ Correct |

## Kernel Code Written (Awaiting Integration)

| Feature | File | Notes |
|---|---|---|
| Flash attention | flash_decode.cuh | Online softmax, tiled K/V, needs bug fix |
| Q4 KV attention | attn_decode_kernel.cu | Q4 unpacking + flash softmax |
| Q6 KV cache | q6_kv_cache.cuh | 6-bit packing, follows Q8 pattern |
| TurboQuant KV | turboquant_kv.cuh | Adaptive Q8/Q6/Q4 per-head |
| Continuous batching | main_cb.cpp | Multi-slot engine pool server |

## Architecture

- **DP4A L1-cached GEMV**: reads Q8 activations from L1 cache (no SMEM, no __syncthreads)
- **rmsnorm+quantize fusion**: saves 88 kernel launches per forward
- **swiglu+quantize fusion**: saves 44 kernel launches
- **Zero-sync fused rope**: Q writes to separate buffer (no __syncthreads)
- **Occupancy fix**: o_proj 3→6 blocks/SM, down_proj 2→6 blocks/SM (0 SMEM)

## Hardware
- RTX 3070 Ti Laptop (8GB GDDR6, 448 GB/s, sm_86, 46 SMs)
- Model: Nanbeige4.2-3B (22 layers × 2 passes, Q4_G64)
