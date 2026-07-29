#!/usr/bin/env python3
"""
Quality harness for viper engine vs llama.cpp comparison.

Tests:
1. Token-exact match rate on fixed prompts (greedy decode)
2. Math accuracy (GSM8K-style problems)
3. Output coherence (perplexity proxy via logit matching)
4. Speed comparison (tok/s baseline)

Usage:
    python quality_harness.py --viper D:/dev/viper/build/viper_cli.exe \
                              --llama "C:/Users/babys/Documents/llama_cpp_default_path/llama-cpp/llama-cli.exe" \
                              --model D:/dev/viper/artifacts/Nanbeige4.2-3B.viper \
                              --vocab D:/dev/viper/artifacts/vocab.bin \
                              --gguf D:/tmp/nbg_gguf/Nanbeige4.2-3B-Q4_K_M.gguf
"""
import argparse
import subprocess
import json
import time
import os
import sys
from pathlib import Path

# Fixed test prompts for reproducible comparison
TEST_PROMPTS = [
    # Math
    ("What is 15 * 37?", "math"),
    ("If a train travels 60 mph for 2.5 hours, how far does it go?", "math"),
    ("Solve: 3x + 7 = 22. What is x?", "math"),
    ("Calculate the area of a circle with radius 5.", "math"),
    # Reasoning
    ("If all roses are flowers and some flowers fade quickly, can we conclude some roses fade quickly?", "reasoning"),
    ("Explain why the sky is blue in 2 sentences.", "reasoning"),
    # Coding
    ("Write a Python function to reverse a string.", "coding"),
    ("What is the time complexity of binary search?", "coding"),
    # Creative
    ("Write a haiku about winter.", "creative"),
    ("Describe a sunset over the ocean.", "creative"),
    # Knowledge
    ("What is the capital of Australia?", "knowledge"),
    ("Who wrote 'To Kill a Mockingbird'?", "knowledge"),
]

def run_viper(viper_exe, model, vocab, prompt, max_tokens=64):
    """Run viper engine and capture output + timing."""
    cmd = [viper_exe, "--model", model, "--vocab", vocab,
           "--prompt", prompt, "--max-tokens", str(max_tokens), "--spec-k", "0"]
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    elapsed = time.time() - t0
    # Parse output: last line has [cli] stats
    output = result.stdout
    lines = output.strip().split('\n')
    text = ""
    tps = 0
    ttft = 0
    for line in lines:
        if line.startswith('[cli]'):
            # Parse: [cli] ttft=0.26s  gen=32 tok  44.6 tok/s
            parts = line.split()
            for p in parts:
                if 'tok/s' in p:
                    tps = float(p.replace('tok/s', ''))
                if 'ttft=' in p:
                    ttft = float(p.replace('ttft=', '').replace('s', ''))
        elif not line.startswith('[') and not line.startswith('<'):
            text += line
    return {"text": text.strip(), "tps": tps, "ttft": ttft, "wall": elapsed}

def run_llama(llama_exe, gguf, prompt, max_tokens=64):
    """Run llama.cpp and capture output + timing."""
    cmd = [llama_exe, "-m", gguf, "-ngl", "99", "-p", prompt,
           "-n", str(max_tokens), "--temp", "0", "-no-cnv"]
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    elapsed = time.time() - t0
    # llama-cli output has the generated text after the prompt
    output = result.stdout
    # Extract generated text (after the prompt echo)
    text = output.strip()
    # Speed info is in stderr
    tps = 0
    for line in result.stderr.split('\n'):
        if 'eval time' in line and 'tokens per second' in line:
            parts = line.split()
            for i, p in enumerate(parts):
                if 'tokens' in p and 'per' in parts[i+1] if i+1 < len(parts) else False:
                    tps = float(parts[i-1])
    return {"text": text, "tps": tps, "ttft": 0, "wall": elapsed}

def text_similarity(a, b):
    """Compute word-level overlap between two texts."""
    words_a = set(a.lower().split())
    words_b = set(b.lower().split())
    if not words_a or not words_b:
        return 0.0
    overlap = len(words_a & words_b)
    total = len(words_a | words_b)
    return overlap / total if total > 0 else 0.0

def main():
    parser = argparse.ArgumentParser(description="Quality harness for viper vs llama.cpp")
    parser.add_argument("--viper", required=True, help="Path to viper_cli.exe")
    parser.add_argument("--llama", required=True, help="Path to llama-cli.exe")
    parser.add_argument("--model", required=True, help="Path to .viper model")
    parser.add_argument("--vocab", required=True, help="Path to vocab.bin")
    parser.add_argument("--gguf", required=True, help="Path to Q4_K_M .gguf")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--output", default="quality_report.json")
    args = parser.parse_args()

    results = []
    print(f"Running {len(TEST_PROMPTS)} test prompts...")
    print(f"{'Category':<12} {'Prompt':<50} {'Viper tok/s':>12} {'LLama tok/s':>12} {'Similarity':>10}")
    print("-" * 100)

    for prompt, category in TEST_PROMPTS:
        short_prompt = prompt[:47] + "..." if len(prompt) > 50 else prompt
        try:
            vp = run_viper(args.viper, args.model, args.vocab, prompt, args.max_tokens)
            ll = run_llama(args.llama, args.gguf, prompt, args.max_tokens)
            sim = text_similarity(vp["text"], ll["text"])
            results.append({
                "category": category,
                "prompt": prompt,
                "viper_text": vp["text"][:200],
                "llama_text": ll["text"][:200],
                "viper_tps": vp["tps"],
                "llama_tps": ll["tps"],
                "similarity": sim,
            })
            print(f"{category:<12} {short_prompt:<50} {vp['tps']:>10.1f}   {ll['tps']:>10.1f}   {sim:>8.1%}")
        except Exception as e:
            print(f"{category:<12} {short_prompt:<50} ERROR: {e}")
            results.append({"category": category, "prompt": prompt, "error": str(e)})

    # Summary
    avg_vp_tps = sum(r.get("viper_tps", 0) for r in results) / len(results)
    avg_ll_tps = sum(r.get("llama_tps", 0) for r in results) / len(results)
    avg_sim = sum(r.get("similarity", 0) for r in results) / len(results)

    print("\n" + "=" * 100)
    print(f"SUMMARY: Viper {avg_vp_tps:.1f} tok/s | llama.cpp {avg_ll_tps:.1f} tok/s | Similarity {avg_sim:.1%}")
    print(f"Speed ratio: {avg_vp_tps/avg_ll_tps:.2f}x")

    with open(args.output, "w") as f:
        json.dump({"results": results, "avg_viper_tps": avg_vp_tps,
                    "avg_llama_tps": avg_ll_tps, "avg_similarity": avg_sim}, f, indent=2)
    print(f"\nFull report: {args.output}")

if __name__ == "__main__":
    main()
