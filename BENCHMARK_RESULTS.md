# viper Benchmark Results — 2026-07-30

## Final Verified Results (Clean GPU, Same-Session A/B)

### Decode Speed (greedy, median of 3 runs)

| Engine | tok/s | vs llama.cpp |
|---|---|---|
| **viper DP4A L1-cached** | **59.1** | **+9.7% FASTER** |
| viper scalar (baseline) | 51.0 | -5.3% |
| llama.cpp Q4_K_M | 53.87 | reference |

### Quality Verification

| Test | viper DP4A | viper scalar | Match? |
|---|---|---|---|
| Math: "What is 2+2?" | "2 + 2 = **4**" | "2+2=4" | ✅ Both correct |
| Code: "reverse a string" | `def reverse_string(s): return s[::-1]` | — | ✅ Correct |
| Creative: haiku about winter | Correct haiku | — | ✅ Correct |
| Thinking trace | "Weimplify is asked..." | "Weimplify is asked..." | ✅ Identical artifacts |

### Feature Tests

| Feature | Speed | Notes |
|---|---|---|
| DP4A + spec decode K=8 | 67.1 tok/s | 35% acceptance |
| DP4A + prefill batch 8 | 62.7 tok/s | Correct output |
| DP4A + lm_head prune 32K | 65.5 tok/s | ❌ Too aggressive (garbage) |
| DP4A + lm_head prune 128K | 54.9 tok/s | ✅ Acceptable quality |

### Architecture

- DP4A L1-cached GEMV: reads Q8 activations from L1 (no SMEM, no __syncthreads)
- rmsnorm+quantize fusion: saves 88 kernel launches
- swiglu+quantize fusion: saves 44 kernel launches
- Zero-sync fused rope: saves 44 kernel launches
- Occupancy fix: o_proj 3→6 blocks/SM, down_proj 2→6 blocks/SM
