# Viper Engine — Implementation Plan
## C++/CuTe DSL inference for Nanbeige4.2-3B on RTX 3070 Ti (sm_86)

**Goal:** 1 tok/s → 50-100+ tok/s. No hardware change. Pure software.

## Architecture
```
Python (orchestration) → pybind11 → C++ Engine → CuTe DSL kernels → GPU
```
Python NEVER touches individual layers. C++ runs the full forward. CuTe kernels handle the math.

## Kernel Design (MINIMIZE launches)
- **Fused dequant + GEMM + LoRA**: ONE kernel, weights stay 4-bit, LoRA in registers
- **Fused attention**: Flash Attention 2 style, online softmax, KV cache
- **Fused MLP**: Gate+Up+SwiGLU+Down, intermediates in shared memory
- **Target**: 3-4 kernels per layer-pass (vs current ~30), CUDA Graphs = 1 total

## Phases
0. **Profile + KV cache fix** (free 5-15× speedup) — measure, don't guess
1. **C++ skeleton + cuBLAS + KV cache** (10-20 tok/s) — eliminate bnb + Python
2. **CuTe fused GEMM+LoRA** (30-50 tok/s) — eliminate dequant overhead
3. **Fused attention + CUDA Graphs** (50-100+ tok/s) — eliminate WDDM
4. **Training integration** — same engine for train + eval
