"""
embedding.py — vocab gather oracle for Nanbeige4.2-3B.

Op: input_ids [B, T] (int64) -> hidden_states [B, T, hidden_size] via
nn.Embedding lookup against embed_tokens.weight [vocab_size, hidden_size].

Reference op: ``nn.Embedding.forward`` from upstream ``modeling_nanbeige.py``.
The weight is the model's ``embed_tokens`` parameter.

Run as:  python -m tools.oracle.embedding
"""
from __future__ import annotations

import sys

import torch

from tools.oracle._common import (
    CONFIG,
    assert_close,
    make_bf16,
    make_int_ids,
    set_seed,
)


def _fp64_oracle(weight: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
    """Exact FP64 embedding gather."""
    return weight.to(torch.float64)[input_ids]


def _bf16_path(weight: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
    """BF16 embedding gather, output in FP32 for fair comparison."""
    out = weight[input_ids]                              # BF16 gather
    return out.to(torch.float32)                          # cast to FP32 for delta measurement


def run() -> tuple[str, float]:
    set_seed(0)
    B, T = 2, 16
    hidden = CONFIG["hidden_size"]
    vocab = CONFIG["vocab_size"]

    input_ids = make_int_ids((B, T), vocab)
    weight = make_bf16((vocab, hidden), scale=0.02)      # model init scale

    bf16_out = _bf16_path(weight, input_ids)
    fp64_out = _fp64_oracle(weight, input_ids)

    return assert_close("embedding", bf16_out, fp64_out, tol=1e-3)


if __name__ == "__main__":
    status, _ = run()
    sys.exit(0 if status == "PASS" else 1)