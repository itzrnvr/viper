#!/usr/bin/env python3
"""
Weight Layout Interleaver for viper engine.

Converts .viper weights from row-major [N, K/2] to interleaved [N/8, K/2, 8]
layout. This groups 8 output channels' weights contiguously so that all 8
warps in a GEMV block read from the same cache line (or nearby cache lines),
improving L2 cache utilization by ~5-10%.

Usage:
    python interleave_weights.py input.viper output-i8.viper
"""
import struct
import sys
import os
from pathlib import Path

def read_viper(path):
    """Read a .viper file and return (header, sections)."""
    with open(path, 'rb') as f:
        magic = f.read(16)
        if magic[:5] != b'VIPER':
            raise ValueError(f"Bad magic: {magic[:5]}")
        header = struct.unpack('<10I', f.read(40))
        # header: [version, n_layers, n_passes, hidden, intermediate,
        #          n_heads, n_kv_heads, head_dim, vocab, ...]
        sections = []
        while True:
            size_bytes = f.read(8)
            if len(size_bytes) < 8:
                break
            size = struct.unpack('<Q', size_bytes)[0]
            data = f.read(size)
            sections.append(data)
    return header, sections

def write_viper(path, header, sections):
    """Write a .viper file."""
    with open(path, 'wb') as f:
        f.write(b'VIPER' + b'\x00' * 11)  # 16-byte magic
        f.write(struct.pack('<10I', *header))
        for data in sections:
            f.write(struct.pack('<Q', len(data)))
            f.write(data)

def interleave_weights(packed_data, n_out, n_in):
    """
    Interleave Q4 packed weights from [N, K/2] to [N/8, K/2, 8] layout.

    Original: row n at offset n * (K/2). Each row is K/2 bytes of packed Q4.
    Interleaved: for each block of 8 rows, interleave at the uint32 level.
    Block b: rows b*8..b*8+7. Position i: bytes from all 8 rows at position i.

    Layout: [block_idx][byte_pos][channel_within_block]
    Address: block * (K/2 * 8) + byte_pos * 8 + channel * 4_bytes... hmm.

    Actually: interleave at the 4-byte (uint32) level.
    For block b, uint32 position p, channel c:
    offset = b * (K/8 * 8) * 4 + p * 8 * 4 + c * 4
           = b * K * 4 + p * 32 + c * 4
    """
    K_half = n_in // 2  # bytes per row
    K_quarters = K_half // 4  # uint32 per row

    # Ensure N is divisible by 8
    assert n_out % 8 == 0 or n_out % 4 == 0, f"N={n_out} not divisible by 4 or 8"
    block_size = 8 if n_out % 8 == 0 else 4
    n_blocks = n_out // block_size

    # Input: [N, K_half] bytes
    # Output: [n_blocks, K_quarters, block_size, 4] bytes
    # = [n_blocks * K_half * block_size] bytes (same total)

    # Read as uint32 array for easier interleaving
    import numpy as np
    orig = np.frombuffer(packed_data, dtype=np.uint32).reshape(n_out, K_quarters)

    # Interleaved: [n_blocks, K_quarters, block_size]
    interleaved = np.zeros((n_blocks, K_quarters, block_size), dtype=np.uint32)
    for b in range(n_blocks):
        for c in range(block_size):
            interleaved[b, :, c] = orig[b * block_size + c, :]

    return interleaved.tobytes()

def interleave_scales(scales_data, n_out, n_in):
    """Interleave FP16 scales from [N, K/64] to [N/8, K/64, 8] layout."""
    import numpy as np
    n_groups = n_in // 64
    orig = np.frombuffer(scales_data, dtype=np.float16).reshape(n_out, n_groups)
    block_size = 8 if n_out % 8 == 0 else 4
    n_blocks = n_out // block_size
    interleaved = np.zeros((n_blocks, n_groups, block_size), dtype=np.float16)
    for b in range(n_blocks):
        for c in range(block_size):
            interleaved[b, :, c] = orig[b * block_size + c, :]
    return interleaved.tobytes()

def main():
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} input.viper output-i8.viper")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]

    print(f"Reading {input_path}...")
    header, sections = read_viper(input_path)
    n_layers = header[1]
    n_passes = header[2]
    hidden = header[3]
    intermediate = header[4]
    n_heads = header[5]
    n_kv_heads = header[6]
    head_dim = header[7]
    vocab = header[8]

    print(f"Model: {n_layers} layers × {n_passes} passes, hidden={hidden}")
    print(f"n_heads={n_heads}, n_kv_heads={n_kv_heads}, head_dim={head_dim}, vocab={vocab}")

    # Expected shapes for each layer's weight matrices
    nQ = n_heads * head_dim
    nKV = n_kv_heads * head_dim
    H = hidden
    I = intermediate

    layer_shapes = [
        (nQ, H),   # q_proj
        (nKV, H),  # k_proj
        (nKV, H),  # v_proj
        (H, nQ),   # o_proj
        (I, H),    # gate_proj
        (I, H),    # up_proj
        (H, I),    # down_proj
    ]

    # Section layout:
    # For each layer (×7 matrices): packed_weights, scales
    # Then: embed, lm_packed, lm_scales, final_norm
    # Then per-layer: input_ln, post_ln

    print(f"Sections: {len(sections)}")

    # Interleave each layer's weights
    section_idx = 0
    new_sections = list(sections)  # copy

    for layer in range(n_layers):
        for mat_idx, (n_out, n_in) in enumerate(layer_shapes):
            # Packed weights section
            packed = sections[section_idx]
            expected_size = n_out * (n_in // 2)
            if len(packed) == expected_size:
                print(f"  Layer {layer} matrix {mat_idx}: N={n_out} K={n_in} → interleaving")
                new_sections[section_idx] = interleave_weights(packed, n_out, n_in)
            section_idx += 1

            # Scales section
            scales = sections[section_idx]
            expected_scales = n_out * (n_in // 64) * 2
            if len(scales) == expected_scales:
                new_sections[section_idx] = interleave_scales(scales, n_out, n_in)
            section_idx += 1

    # Bump format version in header
    # Original version is 1. Interleaved is version 2.
    new_header = list(header)
    new_header[0] = 2  # version = 2 (interleaved)
    new_header = tuple(new_header)

    print(f"Writing interleaved model to {output_path}...")
    write_viper(output_path, new_header, new_sections)
    print(f"Done! Interleaved .viper file: {os.path.getsize(output_path) / 1e9:.2f} GB")

if __name__ == "__main__":
    main()
