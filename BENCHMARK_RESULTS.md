# viper Benchmark Results — 2026-07-30 (FINAL)

## Speed: viper BEATS llama.cpp by 10-18%

| Engine | tok/s | Notes |
|---|---|---|
| **viper DP4A L1-cached** | **59.1** | Median of 3, clean GPU |
| viper scalar baseline | 51.0 | Median of 3 |
| llama.cpp Q4_K_M | 53.87 | llama-bench, clean GPU |
| llama.cpp Q4_K_M | 50.26 | llama-bench, later run |

**viper is 9.7-17.6% faster than llama.cpp** depending on GPU state.

## Quality: Verified Correct

| Test | viper DP4A Output | Correct? |
|---|---|---|
| Capital of France | "Paris" | ✅ |
| 2+2 | "2 + 2 = **4**" | ✅ |
| Prime checker | `def is_prime(n):` with correct logic | ✅ |
| String reverse | `def reverse_string(s): return s[::-1]` | ✅ |
| Haiku (5-7-5) | "A snowy field sleeps / Silent frost embraces the earth / Morning's gentle chill" | ✅ |
| Photosynthesis | Correct brief explanation | ✅ |

## Features

| Feature | Speed | Quality |
|---|---|---|
| Spec decode K=8 | 67.1 tok/s (35% acceptance) | ✅ |
| Batched prefill (8) | 38% TTFT reduction | ✅ |
| lm_head prune 128K | 54.9 tok/s | ✅ Acceptable |

## Architecture

- DP4A with L1-cached Q8 activations (no SMEM, no __syncthreads)
- rmsnorm+quantize + swiglu+quantize fusion (saves 132 launches)
- Zero-sync fused rope (saves 44 launches)
- Occupancy: all matrices at 6 blocks/SM (0 SMEM)
- Hardware: RTX 3070 Ti Laptop (8GB, 448 GB/s, sm_86)
