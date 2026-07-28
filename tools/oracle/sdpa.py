"""
sdpa.py — Causal Scaled-Dot-Product Attention oracle for Nanbeige4.2-3B.

Op (from upstream modeling_nanbeige.py ``NanbeigeSdpaAttention``):

    Q, K, V : [B, num_heads, T, head_dim]    # BF16
    scores  : Q @ K.transpose(-2, -1) / sqrt(head_dim)         # FP32 accumulate
    causal  : is_causal=True (or 4D mask; for full-sequence decode this is None)
    attn    : softmax(scores, dim=-1, dtype=fp32) -> bf16
    out     : attn @ V                                          # FP32 accumulate

Upstream's SDPA call uses ``is_causal=True`` when no explicit mask is given.

Tolerance: 1e-2 (matmul + softmax accumulate large errors for [48, T] scores).

Run as:  python -m tools.oracle.sdpa
"""
from __future__ import annotations

import sys

import torch
import torch.nn.functional as F

from tools.oracle._common import (
    CONFIG,
    assert_close,
    make_bf16,
    set_seed,
)


def _fp64_oracle(q_bf: torch.Tensor, k_bf: torch.Tensor, v_bf: torch.Tensor,
                 is_causal: bool = True) -> torch.Tensor:
    q = q_bf.to(torch.float64)
    k = k_bf.to(torch.float64)
    v = v_bf.to(torch.float64)
    head_dim = q.shape[-1]
    scale = 1.0 / (head_dim ** 0.5)
    scores = (q @ k.transpose(-2, -1)) * scale                         # [B, H, T, T]
    if is_causal:
        T = scores.shape[-1]
        causal_mask = torch.triu(
            torch.full((T, T), float("-inf"), dtype=scores.dtype, device=scores.device),
            diagonal=1,
        )
        scores = scores + causal_mask
    attn = torch.softmax(scores, dim=-1)                                # FP64 softmax
    out = attn @ v
    return out


def _bf16_path(q_bf: torch.Tensor, k_bf: torch.Tensor, v_bf: torch.Tensor,
               is_causal: bool = True) -> torch.Tensor:
    out = F.scaled_dot_product_attention(
        q_bf, k_bf, v_bf,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=is_causal,
    )
    return out.to(torch.float32)


def run() -> tuple[str, float]:
    set_seed(0)
    B, T = 2, 16
    num_heads = CONFIG["num_heads"]
    head_dim = CONFIG["head_dim"]

    q_bf = make_bf16((B, num_heads, T, head_dim), scale=0.5)
    k_bf = make_bf16((B, num_heads, T, head_dim), scale=0.5)
    v_bf = make_bf16((B, num_heads, T, head_dim), scale=0.5)

    bf16_out = _bf16_path(q_bf, k_bf, v_bf)
    fp64_out = _fp64_oracle(q_bf, k_bf, v_bf)

    return assert_close("sdpa", bf16_out, fp64_out, tol=1e-2)


if __name__ == "__main__":
    status, _ = run()
    sys.exit(0 if status == "PASS" else 1)