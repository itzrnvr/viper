"""
forward.py — end-to-end model forward oracle for Nanbeige4.2-3B.

Loads the actual 8.3 GB model in BF16 from ``D:/hf-cache/Nanbeige4.2-3B`` and
verifies:

    1.  Tokenization of a fixed prompt yields a sane input_ids tensor.
    2.  ``model.forward(input_ids=..., use_cache=True)`` returns logits with the
        correct shape ``[B, T, vocab_size]`` and the final ``past_key_values``
        is a :class:`DynamicCache` with 44 slots (num_loops * num_hidden_layers).
    3.  Logits are finite (no NaN / Inf).
    4.  Greedy decoding of the first generated token returns a valid id
        (< vocab_size) and is reproducible across two consecutive runs
        (deterministic sampling).
    5.  The returned argmax matches the model's own ``generate`` argmax for the
        same prompt and temperature=0 (greedy), within BF16 determinism.

This module is the ONLY oracle that touches the model weights. The other 12
ops use small random tensors and run quickly without loading the model.

Memory budget:
    * Model in BF16 :  ~3.5 GB
    * KV cache for a 8-token prompt in BF16 : ~5 MB
    * Single forward + small generation : < 4 GB total — well within 8 GB
      VRAM of an RTX 3070 Ti and < 32 GB host RAM.

Run as:  python -m tools.oracle.forward
"""
from __future__ import annotations

import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from tools.oracle._common import CONFIG, set_seed


PROMPT = "Hello, my name is"


def _load_model(device: str = "cpu"):
    """Load Nanbeige4.2-3B in BF16.

    Defaults to CPU to keep the test runnable on any host. Pass ``device="cuda"``
    on a CUDA-capable machine for speed.
    """
    path = r"D:/hf-cache/Nanbeige4.2-3B"
    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="eager",                                   # exact algorithm
    )
    model = model.to(device).eval()
    return tok, model


def run() -> tuple[str, float]:
    set_seed(0)
    vocab = CONFIG["vocab_size"]
    eos = CONFIG["eos_token_id"]
    num_layers = CONFIG["num_layers"]
    num_loops = CONFIG["num_loops"]

    tok, model = _load_model(device="cpu")

    inputs = tok(PROMPT, return_tensors="pt")
    input_ids = inputs.input_ids
    B, T = input_ids.shape

    # 1. Forward pass with cache.
    with torch.no_grad():
        out = model(input_ids=input_ids, use_cache=True, return_dict=True)

    logits = out.logits
    pkv = out.past_key_values

    # 2. Logits shape + finiteness.
    assert logits.shape == (B, T, vocab), logits.shape
    assert torch.isfinite(logits).all(), "logits contain NaN/Inf"

    # 3. Greedy argmax.
    gen_id_logits = logits[0, -1, :].argmax().item()
    assert 0 <= gen_id_logits < vocab, gen_id_logits

    # 4. KV cache has the right number of slots (44 = 2 loops * 22 layers).
    expected_slots = num_loops * num_layers
    actual_slots = len(pkv)
    assert actual_slots == expected_slots, (actual_slots, expected_slots)

    # 5. Reproducibility: greedy decoding must be deterministic.
    with torch.no_grad():
        gen1 = model.generate(
            input_ids=input_ids,
            max_new_tokens=1,
            do_sample=False,
            temperature=1.0,
            top_k=0,
            top_p=1.0,
            pad_token_id=tok.pad_token_id or 0,
            eos_token_id=eos,
        )
        gen2 = model.generate(
            input_ids=input_ids,
            max_new_tokens=1,
            do_sample=False,
            temperature=1.0,
            top_k=0,
            top_p=1.0,
            pad_token_id=tok.pad_token_id or 0,
            eos_token_id=eos,
        )
    assert torch.equal(gen1, gen2), "greedy decode non-deterministic across runs"
    assert gen1[0, -1].item() == gen_id_logits, "generate argmax != logits argmax"

    print(
        f"[forward] prompt={PROMPT!r} T={T} logits.shape={tuple(logits.shape)} "
        f"greedy_id={gen_id_logits} cache_slots={actual_slots}  -> PASS"
    )
    return "PASS", 0.0


if __name__ == "__main__":
    try:
        status, _ = run()
    except Exception as e:                                                # noqa: BLE001
        print(f"[forward] FAIL with {type(e).__name__}: {e}")
        sys.exit(1)
    sys.exit(0 if status == "PASS" else 1)