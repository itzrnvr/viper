# Nanbeige4.2-3B FP64 Per-Op Oracle

Python reference oracles that produce ground-truth FP64 outputs for every op
in the Nanbeige4.2-3B forward pass. Each file is runnable as
`python -m tools.oracle.<op>`; `run_all.py` aggregates them.

## Why this exists

Project viper is a C++/CUDA inference engine targeting the same model. To
verify the engine is bit-correct against the upstream `modeling_nanbeige.py`,
every kernel has an FP64 oracle. The oracle is the only thing standing
between a "looks fast" kernel and a "is correct" kernel.

## Layout

```
tools/oracle/
    _common.py        shared upstream-loader + BF16/FP64 helpers
    embedding.py      vocab gather [B,T] -> [B,T,3072]
    rmsnorm.py        y = rsqrt(mean(x^2)+eps)*x*gamma (cross-validates upstream)
    rope.py           theta=70M, head_dim=128, rotate_half=cat([-x[...,64:], x[...,:64]])
    linear.py         q/k/v/o/gate/up/down matmul (cross-validates nn.Linear)
    gqa_repeat.py     K,V [B,8,T,128] -> [B,48,T,128] (n_rep=6)
    sdpa.py           causal SDPA with FP32 softmax (tol 1e-2)
    swiglu.py         silu(gate)*up
    residual.py       x + y
    loop_dispatch.py  for loop_idx in 0..1, layer_idx in 0..21
    kv_cache.py       DynamicCache with cache_layer_idx = layer + loop*22
    sampling.py       greedy + top-k + top-p, eos=166101
    forward.py        end-to-end model forward (loads 8.3 GB model in BF16)
    parity_harness.py aggregate random-input parity for 12 small ops
    run_all.py        orchestrator (runs all 13, prints PARITY PASS/FAIL)
    README.md         this file
```

## How to run

### One op at a time
```bash
python -m tools.oracle.embedding
python -m tools.oracle.rmsnorm
python -m tools.oracle.linear
# ... etc
```

### All 13 ops (orchestrator)
```bash
python tools/oracle/run_all.py
```

### Quick aggregate of the 12 small ops (no model load)
```bash
python -m tools.oracle.parity_harness
```

## Environment

The oracles use the hermes-agent venv:
```
C:/Users/babys/AppData/Local/hermes/hermes-agent/venv/Scripts/python.exe
```
with torch 2.12.1+cu126, numpy 2.x, transformers 5.x. `run_all.py` and the
individual `python -m tools.oracle.<op>` invocations must be issued from a
shell where that venv's Python is active, OR you must point `python` at the
venv explicitly. The CPU torch at `C:/Python313` is NOT used.

## Algorithm details

Each op computes:
1.  **FP64 oracle** — the exact math in FP64. No rounding. Reference truth.
2.  **BF16 reference** — inputs cast to BF16, math done with FP32 internal
    precision where the upstream algorithm already uses FP32 (e.g. RMSNorm
    variance, RoPE cos/sin, SDPA softmax, matmul accumulate), output cast
    back to FP32 for measurement.

The only delta between the two paths is therefore BF16 *input quantization*
(typically 1e-7 to 1e-4 with model-realistic activation magnitudes). This
isolates the algorithm from BF16 final-multiply quirks (upstream RMSNorm does
`gamma * bf16(intermediate)` which adds ~1e-2 error of its own; the oracle
side-steps that by keeping the final multiply in FP32).

## Tolerances

| Op       | tol  |
|----------|------|
| sdpa     | 1e-2 |
| all else | 1e-3 |

SDPA needs 1e-2 because the softmax accumulates over `T*T` scores; the FP32
internal accumulator's rounding bleeds through.

## Memory safety

* `forward.py` is the only file that loads the actual 8.3 GB model.
* `run_all.py` runs `forward.py` ONLY after all 12 small ops pass, so a
  broken kernel doesn't waste GPU/RAM loading weights.
* Per-op tests use small random tensors — never the full model.

## Cross-validation

`rmsnorm.py` and `linear.py` also instantiate the actual upstream classes
(`NanbeigeRMSNorm`, `nn.Linear`) and verify they match the FP64 oracle within
a slightly looser tolerance (5e-2 / 5e-3 respectively) — those tests are
informational and do not gate PASS/FAIL on the primary 1e-3 oracle.