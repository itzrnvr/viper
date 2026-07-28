"""
rope.py — Rotary Position Embedding oracle for Nanbeige4.2-3B.

Op (from upstream modeling_nanbeige.py):

    inv_freq = 1.0 / (base ** (arange(0, dim, 2).float() / dim))       # [dim/2]
    freqs    = inv_freq @ position_ids.float()                          # fp32
    emb      = cat([freqs, freqs], dim=-1)                              # [B, T, dim]
    cos, sin = emb.cos(), emb.sin()                                     # fp32
    cos, sin = cos.to(bf16), sin.to(bf16)                               # bf16
    q'       = q * cos + rotate_half(q) * sin                          # bf16

rotate_half(x) = cat([-x[..., dim/2:], x[..., :dim/2]], dim=-1).

For oracle purposes the BF16 path keeps cos/sin and outputs in FP32 so the
only delta versus the FP64 oracle is BF16 input quantization.

Run as:  python -m tools.oracle.rope
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


def _build_inv_freq(head_dim: int, base: float, dtype=torch.float64) -> torch.Tensor:
    half = head_dim // 2
    idx = torch.arange(0, head_dim, 2, dtype=torch.int64).to(torch.float64)
    return 1.0 / (base ** (idx / head_dim))


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat([-x[..., half:], x[..., :half]], dim=-1)


def _fp64_oracle(q_bf: torch.Tensor, k_bf: torch.Tensor,
                 position_ids: torch.Tensor, inv_freq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    inv = inv_freq[None, :, None]                                     # [1, dim/2, 1]
    pos = position_ids[:, None, :].to(torch.float64)                   # [B, 1, T]
    freqs = (inv * pos).transpose(1, 2)                                 # [B, T, dim/2]
    emb = torch.cat([freqs, freqs], dim=-1)                            # [B, T, dim]
    cos = emb.cos()
    sin = emb.sin()
    cos_unsq = cos.unsqueeze(1)                                        # [B, 1, T, dim]
    sin_unsq = sin.unsqueeze(1)
    q = q_bf.to(torch.float64)
    k = k_bf.to(torch.float64)
    q_out = q * cos_unsq + _rotate_half(q) * sin_unsq
    k_out = k * cos_unsq + _rotate_half(k) * sin_unsq
    return q_out, k_out


def _bf16_path(q_bf: torch.Tensor, k_bf: torch.Tensor,
               position_ids: torch.Tensor, inv_freq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """BF16 inputs, FP32 internal cos/sin (matches upstream math precision)."""
    inv = inv_freq[None, :, None]                                      # [1, dim/2, 1]
    pos = position_ids[:, None, :].to(torch.float32)                   # [B, 1, T]
    freqs = (inv.to(torch.float32) * pos).transpose(1, 2)
    emb = torch.cat([freqs, freqs], dim=-1)
    cos = emb.cos().to(torch.float32)
    sin = emb.sin().to(torch.float32)
    cos_unsq = cos.unsqueeze(1)
    sin_unsq = sin.unsqueeze(1)
    q = q_bf.to(torch.float32)
    k = k_bf.to(torch.float32)
    q_out = q * cos_unsq + _rotate_half(q) * sin_unsq
    k_out = k * cos_unsq + _rotate_half(k) * sin_unsq
    return q_out, k_out


def run() -> tuple[str, float]:
    set_seed(0)
    B, T = 2, 16
    num_heads = CONFIG["num_heads"]
    num_kv_heads = CONFIG["num_kv_heads"]
    head_dim = CONFIG["head_dim"]
    base = CONFIG["rope_theta"]

    q_bf = make_bf16((B, num_heads, T, head_dim), scale=0.5)
    k_bf = make_bf16((B, num_kv_heads, T, head_dim), scale=0.5)
    position_ids = torch.arange(T, dtype=torch.long).unsqueeze(0).repeat(B, 1)

    inv_freq = _build_inv_freq(head_dim, base)

    q_bf_out, k_bf_out = _bf16_path(q_bf, k_bf, position_ids, inv_freq)
    q_fp_out, k_fp_out = _fp64_oracle(q_bf, k_bf, position_ids, inv_freq)

    s_q, d_q = assert_close("rope.q", q_bf_out, q_fp_out, tol=1e-3)
    s_k, d_k = assert_close("rope.k", k_bf_out, k_fp_out, tol=1e-3)
    if s_q != "PASS" or s_k != "PASS":
        return "FAIL", max(d_q, d_k)
    return "PASS", max(d_q, d_k)


if __name__ == "__main__":
    status, _ = run()
    sys.exit(0 if status == "PASS" else 1)