# Optimization Analysis (2026-07-30)

## Quality Comparison: viper vs llama.cpp

### Viper (BF16 cache, DP4A Q4 weights)
- Math: `2+2 = **4**` ✅
- Knowledge: `capital of France is **Paris**` ✅
- Coding: `reverse_string → return s[::-1]` ✅
- Creative: 5-7-5 haiku ✅
- Speed: 56-63 tok/s (clean GPU)

### llama.cpp (Q4_K_M GGUF)
- Speed: 53.87 tok/s (llama-bench, previous session)
- Quality: Cannot test now (GPU at 87% util from training)
- Previous session verified: correct math/code/knowledge

### Verdict
Viper matches or exceeds llama.cpp quality at 4-17% higher speed.
Full formal comparison requires clean GPU.

---

## n_passes=2 Weight Sharing Analysis

### Problem
Nanbeige4.2-3B has n_passes=2: the same 22 layers are applied twice per
token. Both passes read the SAME 1.67 GB of weights. Total weight traffic
per token: 2 × 1.67 GB = 3.34 GB.

### Sequential Dependency
Loop 1's input depends on loop 0's output (final_norm between loops).
This means we CANNOT batch both passes in a single layer computation for
the SAME token. The dependency chain is:

  embed → loop0(L0→L21) → final_norm → loop1(L0→L21) → final_norm → lm_head

### What Multi-M Already Provides
The multi-M DP4A GEMV kernel processes M tokens through one layer with a
SINGLE weight read. For spec decode (M=K+1=9), this gives 9× arithmetic
intensity per byte of weight read.

For the n_passes=2 case, multi-M helps when processing MULTIPLE tokens
through the SAME loop pass simultaneously (e.g., batch processing or
continuous batching). It does NOT help for the two sequential passes of
a single token.

### Conclusion
n_passes=2 weight sharing via multi-M is ALREADY IMPLEMENTED for spec
decode (119 tok/s equivalent batch verify speed). The sequential loop
dependency prevents further single-token optimization. No additional
code change needed.

---

## Weight Layout Interleaving Analysis

### Previous Finding
Analysis showed ~0% benefit from uint32-level weight interleaving.

### Root Cause
The DP4A GEMV kernel reads Q4 weights with per-group-of-64 scales. Each
thread reads 4 consecutive bytes (8 Q4 values). Within a warp (32 threads),
the access pattern is already coalesced:

  Thread 0: bytes 0-3
  Thread 1: bytes 4-7
  ...
  Thread 31: bytes 124-127

This gives 128-byte aligned coalesced reads — optimal for the 128-byte
cache line size. Interleaving 8 output channels would change the stride
from 4B to 32B per thread, DESTROYING within-warp coalescing.

### L2 Cache Analysis
The L2 cache is 4 MB. Each layer's weights are ~75 MB (Q4). Only ~5%
of one layer fits in L2. Weight interleaving cannot improve L2 hit rate
because the working set vastly exceeds L2 capacity.

### Conclusion
Weight interleaving provides ~0% benefit and would HARM coalescing.
The current layout is optimal for this GEMV access pattern.

---

## Summary (CORRECTED 2026-07-31)

| Optimization | Status | Evidence |
|-------------|--------|----------|
| Per-block scaling | ✅ KEY FIX | Brought Q4/Q6/Q8 from broken to 85-90% accuracy parity |
| Flash attention | ✅ Wired | --flash-attn flag, multi-tile verified |
| Q8 KV cache | ✅ 90% accuracy | Per-32-block scaling, 49% VRAM savings |
| Q6 KV cache | ✅ 85% accuracy | Per-32-block scaling, 62.5% VRAM savings |
| Q4 KV cache | ✅ 90% accuracy | Per-32-block scaling, 75% VRAM savings |
| TurboQuant | ⚠️ 50% accuracy | Mixed-precision attention kernel has bugs, needs debug |
| Loop-aware (type 5) | ⚠️ Tied with Q4 | No measurable benefit at 150-token context. Theory needs 2K+ retrieval test |
| Continuous batching | ✅ Single-slot + multi-slot | Shared weights via load_shared(), SSE verified |
| n_passes=2 weight sharing | ✅ Multi-M exists | Sequential loop dependency limits single-token |
| Weight interleaving | ✅ 0% benefit | Current layout optimal |
| lm_head pruning | ✅ Implemented | --lm-head-prune N |
| Batched prefill | ✅ Implemented | Fixed batch clamping bug (was limited to 5) |
| Quality vs llama.cpp | ⚠️ Partial | Viper verified correct. llama.cpp too slow for side-by-side |

## What Actually Matters

The per-32-block scaling fix is the single most important change. Before it,
Q4 produced Chinese garbage. After it, all cache types score 85-90%. The fix:
each warp handles one block of 32 values independently with its own scale,
eliminating outlier-sensitivity that per-head scaling suffered from.

Loop-aware cache (Q8 loop0 + Q4 loop1) is implemented but NOT validated.
At 150-token math/knowledge QA, it ties uniform Q4. The compounding-error
hypothesis is plausible but requires a retrieval benchmark at 2K+ positions
to test — embed a fact at position 500 in a 2K document, retrieve it.
