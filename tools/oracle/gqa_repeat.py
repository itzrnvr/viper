"""
gqa_repeat.py — Grouped-Query-Attention KV repeat oracle for Nanbeige4.2-3B.

Op (from upstream modeling_nanbeige.py ``repeat_kv``):

    n_rep = num_heads // num_kv_heads
    x: [B, num_kv_heads, T, head_dim]
       -> expand [B, num_kv_heads, n_rep, T, head_dim]
       -> reshape [B, num_heads, T, head_dim]

For Nanbeige4.2-3B: num_heads=48, num_kv_heads=8, n_rep=6.

This is a pure data-movement op (no math). Diff between BF16 and FP64 is 0.

Run as:  python -m tools.oracle.gqa_repeat
"""
from __future__ import annotations

import sys

import torch

from tools.oracle._common import (
    CONFIG,
    assert_close,
    load_upstream,
    make_bf16,
    set_seed,
)


def _repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    B, num_kv_heads, T, head_dim = x.shape
    if n_rep == 1:
        return x
    x = x[:, :, None, :, :]                                            # [B, num_kv, 1, T, D]
    x = x.expand(B, num_kv_heads, n_rep, T, head_dim)                  # [B, num_kv, n_rep, T, D]
    return x.reshape(B, num_kv_heads * n_rep, T, head_dim)


def _fp64_oracle(x_bf: torch.Tensor, n_rep: int) -> torch.Tensor:
    return _repeat_kv(x_bf.to(torch.float64), n_rep)


def _bf16_path(x_bf: torch.Tensor, n_rep: int) -> torch.Tensor:
    return _repeat_kv(x_bf, n_rep)


def _cross_validate(x_bf: torch.Tensor, n_rep: int) -> tuple[str, float]:
    mod = load_upstream()
    y_upstream = mod.repeat_kv(x_bf, n_rep)
    y_local = _bf16_path(x_bf, n_rep)
    delta = (y_upstream.float() - y_local.float()).abs().max().item()
    status = "PASS" if delta == 0.0 else "FAIL"
    print(f"[gqa_repeat.cross_validate(upstream)] max_abs_delta = {delta:.6g}  -> {status}")
    return status, delta


def run() -> tuple[str, float]:
    set_seed(0)
    B, T = 2, 16
    num_kv = CONFIG["num_kv_heads"]
    num_heads = CONFIG["num_heads"]
    head_dim = CONFIG["head_dim"]
    n_rep = num_heads // num_kv

    x_bf = make_bf16((B, num_kv, T, head_dim), scale=0.5)

    bf16_out = _bf16_path(x_bf, n_rep)
    fp64_out = _fp64_oracle(x_bf, n_rep)

    main_status, main_delta = assert_close("gqa_repeat", bf16_out, fp64_out, tol=1e-3)
    cv_status, _ = _cross_validate(x_bf, n_rep)

    if main_status != "PASS" or cv_status != "PASS":
        return "FAIL", main_delta
    return "PASS", main_delta


if __name__ == "__main__":
    status, _ = run()
    sys.exit(0 if status == "PASS" else 1)