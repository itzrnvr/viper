"""
swiglu.py — SwiGLU MLP activation oracle for Nanbeige4.2-3B.

Op (from upstream modeling_nanbeige.py ``NanbeigeMLP``):

    intermediate = silu(gate_proj(x)) * up_proj(x)
    down         = down_proj(intermediate)

The oracle tests the element-wise SwiGLU step:

    y = silu(gate) * up

Run as:  python -m tools.oracle.swiglu
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


def _silu_fp64(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


def _fp64_oracle(gate_bf: torch.Tensor, up_bf: torch.Tensor) -> torch.Tensor:
    g = gate_bf.to(torch.float64)
    u = up_bf.to(torch.float64)
    return _silu_fp64(g) * u


def _bf16_path(gate_bf: torch.Tensor, up_bf: torch.Tensor) -> torch.Tensor:
    """FP32 silu (matches upstream precision), output FP32 for fair comparison."""
    g = gate_bf.to(torch.float32)
    u = up_bf.to(torch.float32)
    y = torch.nn.functional.silu(g) * u
    return y


def run() -> tuple[str, float]:
    set_seed(0)
    B, T = 2, 16
    intermediate = CONFIG["intermediate_size"]

    # After gate/up projection with std=0.02 weights, output std ~ sqrt(3072) * 0.5 * 0.02 ~ 0.55.
    gate_bf = make_bf16((B, T, intermediate), scale=0.5)
    up_bf = make_bf16((B, T, intermediate), scale=0.5)

    bf16_out = _bf16_path(gate_bf, up_bf)
    fp64_out = _fp64_oracle(gate_bf, up_bf)

    return assert_close("swiglu", bf16_out, fp64_out, tol=1e-3)


if __name__ == "__main__":
    status, _ = run()
    sys.exit(0 if status == "PASS" else 1)