"""
run_all.py — discover and run every op test under tools/oracle.

Runs all 13 op tests:

    1. 12 small ops (random tensors, no model load)
    2. forward.py (loads Nanbeige4.2-3B in BF16)

Exits 0 and prints ``PARITY PASS`` if all 13 pass; otherwise exits non-zero
and prints ``PARITY FAIL`` with per-op results.

Forward is run LAST and ONLY if all 12 small ops pass — keeps memory low in
the failure case (no model load if upstream code is broken).

Run as:  python tools/oracle/run_all.py
         python -m tools.oracle.run_all        (also works)
"""
from __future__ import annotations

import importlib
import pathlib
import sys

# Make ``tools`` importable when this file is run as a plain script
# (`python tools/oracle/run_all.py` from D:/dev/viper/).
HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


SMALL_OPS = [
    "embedding",
    "rmsnorm",
    "rope",
    "linear",
    "gqa_repeat",
    "sdpa",
    "swiglu",
    "residual",
    "loop_dispatch",
    "kv_cache",
    "sampling",
    "parity_harness",
]

FORWARD_OP = "forward"


def _run_op(name: str) -> tuple[str, float]:
    mod = importlib.import_module(f"tools.oracle.{name}")
    return mod.run()


def run() -> tuple[str, dict[str, str]]:
    results: dict[str, str] = {}

    print("=" * 70)
    print("Phase 1: 12 small op parity tests (no model load)")
    print("=" * 70)
    small_pass = True
    for name in SMALL_OPS:
        try:
            status, _ = _run_op(name)
        except Exception as e:                                            # noqa: BLE001
            status = f"ERROR({type(e).__name__}: {e})"
        results[name] = status
        if status != "PASS":
            small_pass = False

    forward_status = "SKIP"
    if small_pass:
        print()
        print("=" * 70)
        print("Phase 2: end-to-end forward (loads Nanbeige4.2-3B in BF16)")
        print("=" * 70)
        try:
            forward_status, _ = _run_op(FORWARD_OP)
        except Exception as e:                                            # noqa: BLE001
            forward_status = f"ERROR({type(e).__name__}: {e})"
        results[FORWARD_OP] = forward_status
    else:
        results[FORWARD_OP] = "SKIP (small ops failed)"

    print()
    print("=" * 70)
    for name, status in results.items():
        print(f"  {name:20s}  {status}")
    print("=" * 70)

    all_pass = all(s == "PASS" for s in results.values())
    final = "PARITY PASS" if all_pass else "PARITY FAIL"
    print(final)
    return final, results


if __name__ == "__main__":
    final, _ = run()
    sys.exit(0 if final == "PARITY PASS" else 1)