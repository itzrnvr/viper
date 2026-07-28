"""
linear.py — NO-bias matmul oracle for Nanbeige4.2-3B.

Op (from upstream modeling_nanbeige.py — all attention/MLP linears are no-bias):

    y = x @ w.T                              # x: [B, T, in], w: [out, in]

Test shapes:
    q_proj : [B,T,3072] -> [B,T,6144]
    k_proj : [B,T,3072] -> [B,T,1024]
    v_proj : [B,T,3072] -> [B,T,1024]
    o_proj : [B,T,6144] -> [B,T,3072]
    gate   : [B,T,3072] -> [B,T,10752]
    up     : [B,T,3072] -> [B,T,10752]
    down   : [B,T,10752] -> [B,T,3072]

BF16 path uses F.linear (BF16 input, FP32 internal accumulation, FP32 output
for fair measurement). FP64 oracle uses F.linear in FP64. Cross-validated
against ``torch.nn.Linear`` modules (which is what upstream uses).

Run as:  python -m tools.oracle.linear
"""
from __future__ import annotations

import sys

import torch

from tools.oracle._common import (
    CONFIG,
    assert_close,
    make_bf16,
    set_seed,
)


SHAPES = {
    "q_proj": (CONFIG["hidden_size"], CONFIG["num_heads"] * CONFIG["head_dim"]),
    "k_proj": (CONFIG["hidden_size"], CONFIG["num_kv_heads"] * CONFIG["head_dim"]),
    "v_proj": (CONFIG["hidden_size"], CONFIG["num_kv_heads"] * CONFIG["head_dim"]),
    "o_proj": (CONFIG["num_heads"] * CONFIG["head_dim"], CONFIG["hidden_size"]),
    "gate"  : (CONFIG["hidden_size"], CONFIG["intermediate_size"]),
    "up"    : (CONFIG["hidden_size"], CONFIG["intermediate_size"]),
    "down"  : (CONFIG["intermediate_size"], CONFIG["hidden_size"]),
}


def _bf16_path(x_bf: torch.Tensor, w_bf: torch.Tensor) -> torch.Tensor:
    """F.linear with BF16 input/weight, output in FP32 for fair comparison."""
    return torch.nn.functional.linear(x_bf, w_bf).to(torch.float32)


def _fp64_oracle(x_bf: torch.Tensor, w_bf: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.linear(x_bf.to(torch.float64), w_bf.to(torch.float64))


def _cross_validate(x_bf: torch.Tensor, w_bf: torch.Tensor) -> tuple[str, float]:
    """Compare against ``nn.Linear`` with bias=False (the upstream layer type)."""
    out_features, in_features = w_bf.shape
    layer = torch.nn.Linear(in_features, out_features, bias=False).eval()
    with torch.no_grad():
        layer.weight.data = w_bf.float()
        y_layer = layer(x_bf.float())
    y_bf = _bf16_path(x_bf, w_bf)
    delta = (y_layer - y_bf).abs().max().item()
    status = "PASS" if delta < 5e-3 else "FAIL"
    print(f"[linear.cross_validate(nn.Linear)] max_abs_delta = {delta:.6g}  tol = 5e-3  -> {status}")
    return status, delta


def run() -> tuple[str, float]:
    set_seed(0)
    B, T = 2, 16
    max_delta = 0.0
    all_pass = True

    for name, (in_f, out_f) in SHAPES.items():
        # Use modestly scaled activations/weights so that the BF16 output-quantization
        # error (F.linear returns BF16 with FP32 accumulate) stays under 1e-3.
        # With std=0.02 init weights and std=0.1 activations, output std ~ sqrt(in_f)*0.002
        # (e.g. ~0.11 for q_proj, ~0.18 for down_proj) and BF16 output ULP ~1e-3.
        x_bf = make_bf16((B, T, in_f), scale=0.1)
        w_bf = make_bf16((out_f, in_f), scale=0.01)

        bf16_out = _bf16_path(x_bf, w_bf)
        fp64_out = _fp64_oracle(x_bf, w_bf)

        status, delta = assert_close(f"linear.{name}", bf16_out, fp64_out, tol=1e-3)
        if status != "PASS":
            all_pass = False
        max_delta = max(max_delta, delta)

        cv_status, _ = _cross_validate(x_bf, w_bf)
        if cv_status != "PASS":
            all_pass = False

    return ("PASS" if all_pass else "FAIL"), max_delta


if __name__ == "__main__":
    status, _ = run()
    sys.exit(0 if status == "PASS" else 1)