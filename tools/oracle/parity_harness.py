"""
parity_harness.py — aggregate random-input parity test for 12 small ops.

Runs every op except :mod:`tools.oracle.forward` (which loads the 8.3 GB model).
Fails on any op exceeding its tolerance.

Run as:  python -m tools.oracle.parity_harness
"""
from __future__ import annotations

import importlib
import sys


SKIP = {"parity_harness", "run_all", "forward", "_common"}

# Per-op tolerance override (sdpa uses 1e-2, everything else 1e-3).
TOLERANCE_OVERRIDE = {
    "sdpa": 1e-2,
}


def _discover_ops():
    import pathlib
    here = pathlib.Path(__file__).parent
    ops = []
    for p in sorted(here.glob("*.py")):
        name = p.stem
        if name in SKIP or name.startswith("_"):
            continue
        ops.append(name)
    return ops


def run() -> tuple[str, dict[str, str]]:
    results: dict[str, str] = {}
    overall_pass = True
    for name in _discover_ops():
        if name == "forward":
            continue
        mod = importlib.import_module(f"tools.oracle.{name}")
        try:
            status, _ = mod.run()
        except Exception as e:                                            # noqa: BLE001
            status = f"ERROR({type(e).__name__}: {e})"
        results[name] = status
        if status != "PASS":
            overall_pass = False
    print()
    print("=" * 60)
    for name, status in results.items():
        print(f"  {name:20s}  {status}")
    print("=" * 60)
    final = "PARITY PASS" if overall_pass else "PARITY FAIL"
    print(final)
    return final, results


if __name__ == "__main__":
    final, _ = run()
    sys.exit(0 if final == "PARITY PASS" else 1)