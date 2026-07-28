"""
Shared utilities for Nanbeige4.2-3B FP64 oracle tests.

* Loads the upstream ``modeling_nanbeige`` + ``configuration_nanbeige`` modules
  from ``D:/hf-cache/Nanbeige4.2-3B`` without instantiating the 8.3 GB model.
* Provides :func:`make_inputs` to build BF16 tensors with model-realistic
  magnitudes (activations ~N(0, 0.5), weights ~N(0, 0.02), gamma ~N(1, 0.02)).
* Provides :func:`compare` and :func:`assert_close` for FP64-vs-BF16 reporting.
"""
from __future__ import annotations

import importlib.util
import pathlib
import sys
import types

import numpy as np
import torch

CACHE_DIR = pathlib.Path(r"D:/hf-cache/Nanbeige4.2-3B")
CONFIG = {
    "vocab_size": 166144,
    "hidden_size": 3072,
    "intermediate_size": 10752,
    "num_heads": 48,
    "num_kv_heads": 8,
    "head_dim": 128,
    "num_layers": 22,
    "num_loops": 2,
    "rope_theta": 70_000_000.0,
    "rms_norm_eps": 1e-5,
    "eos_token_id": 166101,
    "bos_token_id": 166100,
}

PACKAGE_NAME = "nb_upstream_oracle"


def load_upstream():
    """Lazy-import the upstream Nanbeige modeling + configuration modules.

    The model weights are NOT loaded — only the Python class definitions are
    imported, so this is safe to call from any test (no OOM risk).
    """
    if PACKAGE_NAME in sys.modules and f"{PACKAGE_NAME}.modeling_nanbeige" in sys.modules:
        return sys.modules[f"{PACKAGE_NAME}.modeling_nanbeige"]

    if not CACHE_DIR.exists():
        raise FileNotFoundError(f"Nanbeige cache dir not found: {CACHE_DIR}")

    pkg = types.ModuleType(PACKAGE_NAME)
    pkg.__path__ = [str(CACHE_DIR)]
    sys.modules[PACKAGE_NAME] = pkg

    def _load_sub(name):
        full = f"{PACKAGE_NAME}.{name}"
        spec = importlib.util.spec_from_file_location(full, CACHE_DIR / f"{name}.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[full] = mod
        spec.loader.exec_module(mod)
        return mod

    _load_sub("configuration_nanbeige")
    return _load_sub("modeling_nanbeige")


def set_seed(seed: int = 0) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def make_bf16(shape, scale: float = 0.5, generator=None) -> torch.Tensor:
    """Build a BF16 tensor with the given shape and N(0, scale) distribution."""
    if generator is None:
        x = torch.randn(*shape, dtype=torch.float32) * scale
    else:
        x = torch.randn(*shape, dtype=torch.float32, generator=generator) * scale
    return x.to(torch.bfloat16)


def make_int_ids(shape, vocab_size: int = CONFIG["vocab_size"]) -> torch.Tensor:
    return torch.randint(0, vocab_size, shape, dtype=torch.long)


def make_gamma(hidden: int, scale: float = 0.02, mean: float = 1.0) -> torch.Tensor:
    """RMSNorm gamma with realistic N(mean, scale) distribution."""
    return (torch.randn(hidden, dtype=torch.float32) * scale + mean).to(torch.bfloat16)


def max_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a.to(torch.float64) - b.to(torch.float64)).abs().max().item()


def assert_close(name: str, bf16_out: torch.Tensor, fp64_out: torch.Tensor,
                 tol: float = 1e-3) -> tuple[str, float]:
    delta = max_abs(bf16_out, fp64_out)
    status = "PASS" if delta < tol else "FAIL"
    print(f"[{name}] max_abs_delta = {delta:.6g}  tol = {tol:g}  -> {status}")
    return status, delta


def report(name: str, status: str, delta: float, tol: float) -> None:
    print(f"[{name}] max_abs_delta = {delta:.6g}  tol = {tol:g}  -> {status}")