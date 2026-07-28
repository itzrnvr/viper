"""
rmsnorm.py — RMSNorm oracle for Nanbeige4.2-3B.

Op (from upstream modeling_nanbeige.py ``NanbeigeRMSNorm``):

    input_dtype  = x.dtype                            # bf16
    x_fp32       = x.to(torch.float32)
    variance     = x_fp32.pow(2).mean(-1, keepdim=True)
    y_fp32       = x_fp32 * torch.rsqrt(variance + eps)
    output       = gamma * y_fp32.to(input_dtype)     # bf16 multiply in upstream

For oracle purposes the BF16 path performs the final multiply in FP32 so that
the only delta versus the FP64 oracle is BF16 input-weight quantization (which
must be < 1e-3 for the C++/CUDA implementation to be considered correct).

Cross-validates against the upstream ``NanbeigeRMSNorm`` class with random
weight (matches the algorithm).

Run as:  python -m tools.oracle.rmsnorm
"""
from __future__ import annotations

import sys

import torch

from tools.oracle._common import (
    CONFIG,
    assert_close,
    load_upstream,
    make_bf16,
    make_gamma,
    set_seed,
)


def _fp64_oracle(x_bf: torch.Tensor, gamma_bf: torch.Tensor, eps: float) -> torch.Tensor:
    x64 = x_bf.to(torch.float64)
    g64 = gamma_bf.to(torch.float64)
    var = x64.pow(2).mean(-1, keepdim=True)
    y = x64 * torch.rsqrt(var + eps)
    return y * g64


def _bf16_path(x_bf: torch.Tensor, gamma_bf: torch.Tensor, eps: float) -> torch.Tensor:
    """FP32 internal math, output FP32 (avoids BF16 final-multiply quirk)."""
    x32 = x_bf.to(torch.float32)
    var = x32.pow(2).mean(-1, keepdim=True)
    y32 = x32 * torch.rsqrt(var + eps)
    return gamma_bf.to(torch.float32) * y32


def _cross_validate(x_bf: torch.Tensor, gamma_bf: torch.Tensor, eps: float) -> tuple[str, float]:
    """Run the actual upstream NanbeigeRMSNorm class and compare to FP64 oracle.

    Upstream does the final multiply in BF16, so the diff includes BF16 final-multiply
    quantization (~1e-2 with random inputs) — we therefore use a 5e-2 tolerance here.
    """
    mod = load_upstream()
    H = x_bf.shape[-1]
    norm = mod.NanbeigeRMSNorm(H, eps=eps).eval()
    with torch.no_grad():
        norm.weight.data = gamma_bf.float()
        y_upstream = norm(x_bf).to(torch.float32)
    y_fp64 = _fp64_oracle(x_bf, gamma_bf, eps).to(torch.float32)
    delta = (y_upstream - y_fp64).abs().max().item()
    status = "PASS" if delta < 5e-2 else "FAIL"
    print(f"[rmsnorm.cross_validate(upstream)] max_abs_delta = {delta:.6g}  tol = 5e-2  -> {status}")
    return status, delta


def run() -> tuple[str, float]:
    set_seed(0)
    B, T, H = 2, 16, CONFIG["hidden_size"]
    eps = CONFIG["rms_norm_eps"]

    x_bf = make_bf16((B, T, H), scale=0.5)
    gamma_bf = make_gamma(H, scale=0.02, mean=1.0)

    bf16_out = _bf16_path(x_bf, gamma_bf, eps)
    fp64_out = _fp64_oracle(x_bf, gamma_bf, eps)

    main_status, main_delta = assert_close("rmsnorm", bf16_out, fp64_out, tol=1e-3)
    cross_status, _ = _cross_validate(x_bf, gamma_bf, eps)

    if main_status != "PASS" or cross_status != "PASS":
        return "FAIL", max(main_delta, 0.0)
    return "PASS", main_delta


if __name__ == "__main__":
    status, _ = run()
    sys.exit(0 if status == "PASS" else 1)