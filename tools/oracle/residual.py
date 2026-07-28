"""
residual.py — element-wise residual addition oracle for Nanbeige4.2-3B.

Op (from upstream modeling_nanbeige.py ``NanbeigeDecoderLayer``):

    residual = hidden_states
    hidden_states = self_attn(input_layernorm(hidden_states))
    hidden_states = residual + hidden_states                  # residual after attention
    ...
    hidden_states = residual + hidden_states                  # residual after MLP

Run as:  python -m tools.oracle.residual
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


def _fp64_oracle(x_bf: torch.Tensor, y_bf: torch.Tensor) -> torch.Tensor:
    return x_bf.to(torch.float64) + y_bf.to(torch.float64)


def _bf16_path(x_bf: torch.Tensor, y_bf: torch.Tensor) -> torch.Tensor:
    """FP32 internal add (matches upstream precision), output FP32 for fair compare."""
    return x_bf.to(torch.float32) + y_bf.to(torch.float32)


def run() -> tuple[str, float]:
    set_seed(0)
    B, T = 2, 16
    H = CONFIG["hidden_size"]

    x_bf = make_bf16((B, T, H), scale=0.5)
    y_bf = make_bf16((B, T, H), scale=0.5)

    bf16_out = _bf16_path(x_bf, y_bf)
    fp64_out = _fp64_oracle(x_bf, y_bf)

    return assert_close("residual", bf16_out, fp64_out, tol=1e-3)


if __name__ == "__main__":
    status, _ = run()
    sys.exit(0 if status == "PASS" else 1)