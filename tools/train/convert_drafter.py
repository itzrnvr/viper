#!/usr/bin/env python3
"""
Convert EAGLE drafter weights from .npz to .viper format.

Usage:
    python convert_drafter.py drafter.npz drafter.viper

The .viper format is the same binary format used by the base model,
but with n_layers=1 and n_passes=1.
"""
import numpy as np
import struct
import sys
import json
import os
def write_packed(f, data):
    """Write uint64 size (in bytes) + raw data."""
    f.write(struct.pack('<Q', data.nbytes))
    f.write(data.tobytes())

def quantize_q4(weight, group_size=64):
    """Quantize to Q4 (4-bit signed, group scales)."""
    w = weight.astype(np.float32).flatten()
    pad = (group_size - len(w) % group_size) % group_size
    if pad > 0:
        w = np.pad(w, (0, pad))
    w = w.reshape(-1, group_size)
    scales = np.abs(w).max(axis=1) / 8.0
    scales = np.maximum(scales, 1e-8).astype(np.float16)
    q = np.round(w / scales[:, None]).clip(-8, 7).astype(np.int32) + 8
    # Pack 2 values per byte
    q = q.astype(np.uint8).clip(0, 15)
    packed = q[:, 0::2] | (q[:, 1::2] << 4)
    return packed.flatten(), scales

def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} input.npz output.viper")
        sys.exit(1)

    inp, outp = sys.argv[1], sys.argv[2]
    data = np.load(inp, allow_pickle=True)
    config = json.loads(str(data['_config']))

    print(f"Config: {config}")
    n_layers = 1
    n_passes = 1

    with open(outp, 'wb') as f:
        # Magic (16 bytes)
        magic = b'VIPER\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
        f.write(magic)

        # Header (40 bytes = 10 uint32)
        header = struct.pack('<10I',
            1,  # version
            n_layers,
            n_passes,
            config['hidden_dim'],
            config['intermediate'],
            config['n_heads'],
            config['n_kv_heads'],
            config['head_dim'],
            config['vocab_size'],
            2048,  # max_seq
        )
        f.write(header)

        # 1 layer × 7 linears (packed + scales each)
        linear_names = ['q_proj', 'k_proj', 'v_proj', 'o_proj',
                       'gate_proj', 'up_proj', 'down_proj']

        total_bytes = 0
        for name in linear_names:
            w = data[f'{name}_packed'].astype(np.uint8)  # already quantized
            s = data[f'{name}_scales'].astype(np.float16)
            # Re-pack if needed (ensure correct format)
            # The training script already packed these, so just write
            # But we need to ensure the packed format matches:
            # [out_features, in_features/2] packed bytes + [out_features, in_features/64] scales
            write_packed(f, w)
            write_packed(f, s)
            total_bytes += len(w) + len(s) * 2 + 16
            print(f"  {name}: {len(w)} bytes packed, {len(s)} scales")

        # Embed (BF16) — loader reads embed BEFORE lm_head
        embed_bf16 = data['embed'].astype(np.float16)
        write_packed(f, embed_bf16)
        print(f"  embed: {len(embed_bf16)} values")

        # lm_head (Q4 packed + scales)
        write_packed(f, data['lm_head_packed'].astype(np.uint8))
        write_packed(f, data['lm_head_scales'].astype(np.float16))
        print(f"  lm_head: {len(data['lm_head_packed'])} bytes")

        # final_norm (BF16)
        fnorm = data['final_norm'].astype(np.float16)
        write_packed(f, fnorm)

        # Per-layer norms (1 layer × 2 norms)
        for name in ['input_ln', 'post_ln']:
            w = data[name].astype(np.float16)
            write_packed(f, w)
            print(f"  {name}: {len(w)} values")

    size_mb = os.path.getsize(outp) / 1e6
    print(f"\n✅ Written {outp} ({size_mb:.1f} MB)")

if __name__ == '__main__':
    main()
