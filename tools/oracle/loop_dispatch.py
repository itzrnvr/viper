"""
loop_dispatch.py — multi-loop decoder dispatch oracle for Nanbeige4.2-3B.

Op (from upstream modeling_nanbeige.py ``NanbeigeModel.forward``):

    num_loops = config.num_loops                                          # 2
    num_hidden_layers = config.num_hidden_layers                          # 22
    layer_order = [(idx, None) for idx in range(num_hidden_layers)]        # sequential
    for loop_idx in range(num_loops):
        for execution_idx, (layer_idx, _) in enumerate(layer_order):
            decoder_layer = layers[layer_idx]
            ...
            cache_layer_idx = layer_idx + loop_idx * num_hidden_layers
            hidden_states = decoder_layer(...)

So for Nanbeige4.2-3B (num_loops=2, num_hidden_layers=22):

    cache_layer_idx = layer_idx + loop_idx * 22

This oracle verifies the dispatch loop iterates (0..21) twice and that
``cache_layer_idx`` is computed correctly for each pair.

Run as:  python -m tools.oracle.loop_dispatch
"""
from __future__ import annotations

import sys

from tools.oracle._common import CONFIG, set_seed


def _expected_cache_layer_idx(layer_idx: int, loop_idx: int) -> int:
    return layer_idx + loop_idx * CONFIG["num_layers"]


def run() -> tuple[str, float]:
    set_seed(0)
    num_loops = CONFIG["num_loops"]
    num_layers = CONFIG["num_layers"]

    max_delta = 0.0
    pairs_seen: list[tuple[int, int, int]] = []
    for loop_idx in range(num_loops):
        for layer_idx in range(num_layers):
            cache_layer_idx = _expected_cache_layer_idx(layer_idx, loop_idx)
            expected = layer_idx + loop_idx * num_layers
            delta = abs(cache_layer_idx - expected)
            max_delta = max(max_delta, float(delta))
            pairs_seen.append((loop_idx, layer_idx, cache_layer_idx))

    # Spot-check first and last entries.
    assert pairs_seen[0] == (0, 0, 0), pairs_seen[0]
    assert pairs_seen[num_layers - 1] == (0, num_layers - 1, num_layers - 1)
    assert pairs_seen[num_layers] == (1, 0, num_layers)
    assert pairs_seen[-1] == (1, num_layers - 1, 2 * num_layers - 1)

    total = num_loops * num_layers
    status = "PASS" if max_delta == 0.0 else "FAIL"
    print(f"[loop_dispatch] total_pairs = {total}  max_abs_delta = {max_delta:.6g}  -> {status}")
    return status, max_delta


if __name__ == "__main__":
    status, _ = run()
    sys.exit(0 if status == "PASS" else 1)