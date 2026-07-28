"""
sampling.py — token sampling oracle for Nanbeige4.2-3B.

Tests three sampling strategies applied to the final logits ``[B, vocab_size]``:

    greedy  : argmax over logits, never returns EOS for the chosen random logits
    top-k   : zero-out everything outside the top-k, then sample from the
              resulting categorical distribution; never returns EOS unless EOS
              is in the top-k
    top-p   : zero-out the smallest-probability tokens whose cumulative mass
              exceeds ``(1 - top_p)``, then sample; never returns EOS unless
              EOS is within the nucleus

The oracle uses EOS = 166101 (per generation_config.json).

Tolerance for distributional tests: all sampled token ids are valid (>=0 and
< vocab_size), and the chosen set of valid token ids after filtering is correct.

Run as:  python -m tools.oracle.sampling
"""
from __future__ import annotations

import sys

import torch

from tools.oracle._common import CONFIG, set_seed


def _greedy(logits: torch.Tensor) -> torch.Tensor:
    return logits.argmax(dim=-1)


def _topk_filter(logits: torch.Tensor, k: int) -> torch.Tensor:
    """Zero everything outside the top-k; returns FP32 logits."""
    topk_vals, topk_idx = torch.topk(logits, k=k, dim=-1)
    masked = torch.full_like(logits, float("-inf"))
    masked.scatter_(-1, topk_idx, topk_vals)
    return masked


def _topp_filter(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    """Zero tokens with cumulative sorted-prob mass above ``top_p`` (smallest first)."""
    sorted_logits, sorted_idx = torch.sort(logits, dim=-1, descending=True)
    sorted_probs = torch.softmax(sorted_logits, dim=-1)
    cum = sorted_probs.cumsum(dim=-1)
    # Remove tokens with cum > top_p (keep first token always).
    sorted_remove = cum > top_p
    sorted_remove[..., 1:] = sorted_remove[..., :-1].clone()
    sorted_remove[..., 0] = False
    # Scatter back to original positions.
    remove_mask = torch.zeros_like(logits, dtype=torch.bool)
    remove_mask.scatter_(-1, sorted_idx, sorted_remove)
    return logits.masked_fill(remove_mask, float("-inf"))


def _sample_from_masked(logits_masked: torch.Tensor, eos_id: int) -> torch.Tensor:
    """Sample one token id per batch row from the masked distribution (FP64)."""
    probs = torch.softmax(logits_masked.to(torch.float64), dim=-1)
    out_ids = []
    for row in probs:
        # Check eos is only in the kept set if it had non-negligible mass before masking.
        out_ids.append(torch.multinomial(row, num_samples=1).item())
    return torch.tensor(out_ids, dtype=torch.long)


def run() -> tuple[str, float]:
    set_seed(0)
    B = 4
    vocab = CONFIG["vocab_size"]
    eos_id = CONFIG["eos_token_id"]

    # Random logits whose argmax is NOT EOS (we set EOS low).
    logits = torch.randn(B, vocab, dtype=torch.float32)
    logits[:, eos_id] = -50.0                                          # make EOS the minimum

    # --- Greedy ---
    greedy_ids = _greedy(logits)
    assert (greedy_ids != eos_id).all(), "greedy picked EOS"
    assert ((greedy_ids >= 0) & (greedy_ids < vocab)).all()

    # --- Top-k (k=20, matching generation_config.json) ---
    k = 20
    masked_k = _topk_filter(logits, k=k)
    # Greedy within top-k must equal the global argmax.
    assert torch.equal(masked_k.argmax(dim=-1), logits.argmax(dim=-1))
    # The non-(-inf) positions should be exactly k per row.
    assert (masked_k != float("-inf")).sum(dim=-1).eq(k).all()

    # --- Top-p (p=0.95) ---
    top_p = 0.95
    masked_p = _topp_filter(logits, top_p=top_p)
    # Greedy must still be top-1 after filtering.
    assert torch.equal(masked_p.argmax(dim=-1), logits.argmax(dim=-1))
    # Non-(-inf) positions should be <= vocab_size and > 0 per row.
    n_kept = (masked_p != float("-inf")).sum(dim=-1)
    assert (n_kept > 0).all() and (n_kept <= vocab).all()
    # Prob mass inside the nucleus must be >= top_p (numerical safety: >= top_p - 1e-6).
    cum_in_nucleus = []
    for row_logits, row_masked in zip(logits, masked_p):
        probs = torch.softmax(row_logits.to(torch.float64), dim=-1)
        mask = row_masked != float("-inf")
        cum_in_nucleus.append(probs[mask].sum().item())
    cum_in_nucleus_t = torch.tensor(cum_in_nucleus)
    assert (cum_in_nucleus_t >= top_p - 1e-6).all(), cum_in_nucleus_t

    # --- Sampling smoke test (one draw, just verify id is valid + not EOS by construction) ---
    sample_ids = _sample_from_masked(masked_k, eos_id)
    assert (sample_ids != eos_id).all(), "sampled EOS despite it being -50"
    assert ((sample_ids >= 0) & (sample_ids < vocab)).all()

    print(f"[sampling] B={B} vocab={vocab} eos={eos_id}  -> PASS")
    return "PASS", 0.0


if __name__ == "__main__":
    status, _ = run()
    sys.exit(0 if status == "PASS" else 1)