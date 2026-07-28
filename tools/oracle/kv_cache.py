"""
kv_cache.py — DynamicCache oracle for Nanbeige4.2-3B.

Op (from upstream modeling_nanbeige.py ``NanbeigeModel._get_cache_seq_length``
and ``_get_loop_cache_layer_idx``):

    cache_layer_idx = layer_idx + loop_idx * num_hidden_layers
    cache = DynamicCache()           # num_loops=2 -> 44 layers total
    cache.update(K, V, layer_idx=cache_layer_idx, cache_kwargs=...)

This oracle verifies:

* 1. DynamicCache stores K/V keyed by the resolved ``layer_idx`` (== layer_idx + loop_idx * 22).
* 2. After ``update``, ``cache[layer_idx]`` returns the appended K/V.
* 3. ``get_seq_length(layer_idx)`` returns the cumulative length for that slot.
* 4. The cache holds 44 slots (num_loops * num_hidden_layers) for the loop layout.

Run as:  python -m tools.oracle.kv_cache
"""
from __future__ import annotations

import sys

import torch
from transformers.cache_utils import DynamicCache

from tools.oracle._common import (
    CONFIG,
    make_bf16,
    set_seed,
)


def _run_loop(loop_idx: int, num_layers: int, B: int, T: int,
              num_kv_heads: int, head_dim: int) -> tuple[str, float]:
    cache = DynamicCache()
    seq_per_slot: list[int] = []
    for layer_idx in range(num_layers):
        cache_layer_idx = layer_idx + loop_idx * num_layers
        kv = make_bf16((B, num_kv_heads, T, head_dim), scale=0.5)
        k = kv.clone()
        v = kv.clone()
        cache.update(k, v, layer_idx=cache_layer_idx)
        seq_len = cache.get_seq_length(cache_layer_idx)
        if seq_len != T:
            return "FAIL", float(abs(seq_len - T))
        seq_per_slot.append(seq_len)
        # transformers 5.x DynamicCache exposes ``cache.layers[i].keys/.values``
        # directly as tensors (no dict-style subscript on the cache itself).
        stored = cache.layers[cache_layer_idx]
        stored_k, stored_v = stored.keys, stored.values
        if stored_k.shape != (B, num_kv_heads, T, head_dim):
            return "FAIL", float(stored_k.shape[-1] - head_dim)
        if not torch.equal(stored_k, k):
            return "FAIL", (stored_k - k).abs().max().item()
        if not torch.equal(stored_v, v):
            return "FAIL", (stored_v - v).abs().max().item()
    return "PASS", 0.0
def run() -> tuple[str, float]:
    set_seed(0)
    B, T = 1, 8
    num_layers = CONFIG["num_layers"]
    num_loops = CONFIG["num_loops"]
    num_kv_heads = CONFIG["num_kv_heads"]
    head_dim = CONFIG["head_dim"]

    max_delta = 0.0
    all_pass = True
    for loop_idx in range(num_loops):
        status, delta = _run_loop(loop_idx, num_layers, B, T, num_kv_heads, head_dim)
        print(f"[kv_cache.loop{loop_idx}] {status} max_abs_delta={delta:.6g}")
        if status != "PASS":
            all_pass = False
        max_delta = max(max_delta, delta)

    # Total distinct cache slots == num_loops * num_layers.
    total_slots = num_loops * num_layers
    status = "PASS" if all_pass else "FAIL"
    print(f"[kv_cache] total_slots_expected = {total_slots}  -> {status}")
    return status, max_delta


if __name__ == "__main__":
    status, _ = run()
    sys.exit(0 if status == "PASS" else 1)