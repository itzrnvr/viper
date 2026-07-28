# viper converter (streaming): HF safetensors -> .viper (Q4_G64 body, BF16 embed/lm_head).
# Processes ONE tensor at a time; peak RAM ~150 MB instead of ~17 GB.
import json, struct, sys, os
import numpy as np
import torch
from safetensors import safe_open

SRC = sys.argv[1] if len(sys.argv) > 1 else "D:/hf-cache/Nanbeige4.2-3B"
DST = sys.argv[2] if len(sys.argv) > 2 else "D:/dev/viper/artifacts/nbg42.viper"

cfg = json.load(open(os.path.join(SRC, "config.json")))
H = cfg["hidden_size"]; I = cfg["intermediate_size"]
NL = cfg["num_hidden_layers"]; NH = cfg["num_attention_heads"]
NKV = cfg["num_key_value_heads"]; HD = cfg.get("head_dim", 128)
V = cfg["vocab_size"]; NP = cfg.get("num_loops", 2)
MS = cfg["max_position_embeddings"]
EPS = cfg.get("rms_norm_eps", 1e-5)
print(f"config: H={H} I={I} NL={NL} NH={NH} NKV={NKV} HD={HD} V={V} loops={NP}", flush=True)

shards = [os.path.join(SRC, f) for f in sorted(os.listdir(SRC)) if f.endswith(".safetensors")]
handles = [safe_open(s, framework="pt") for s in shards]
key2handle = {}
for h in handles:
    for k in h.keys():
        key2handle[k] = h
print(f"{len(key2handle)} tensors across {len(shards)} shards", flush=True)

def get(key):
    return key2handle[key].get_tensor(key).float().numpy()

def bf16_bytes(arr):
    f = np.ascontiguousarray(arr, dtype=np.float32).ravel()
    u = f.view(np.uint32)
    rounding = ((u >> 16) & 1) + 0x7FFF
    u = (u + rounding) >> 16
    return u.astype(np.uint16).tobytes()

def quant_q4_g64(w):
    rows, cols = w.shape
    assert cols % 64 == 0
    packed = np.zeros((rows, cols // 2), dtype=np.uint8)
    scales = np.zeros((rows, cols // 64), dtype=np.float32)
    for g in range(cols // 64):
        grp = w[:, g*64:(g+1)*64]
        mx = np.abs(grp).max(axis=1)
        scale = np.maximum(mx / 7.0, 1e-12)
        scales[:, g] = scale
        q = np.clip(np.round(grp / scale[:, None]).astype(np.int32) + 8, 0, 15).astype(np.uint8)
        for j in range(32):
            packed[:, g*32 + j] = q[:, 2*j] | (q[:, 2*j+1] << 4)
    return packed.tobytes(), bf16_bytes(scales)

out = open(DST, "wb")
out.write(b"VIPER001")
hdr = [NL, NP, H, I, NH, NKV, HD, V, MS, struct.unpack("<I", struct.pack("<f", EPS))[0]]
out.write(struct.pack("<10I", *hdr))

LINEARS = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
           "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]

for l in range(NL):
    prefix = f"model.layers.{l}."
    for name in LINEARS:
        w = get(prefix + name + ".weight")
        pb, sb = quant_q4_g64(w)
        del w
        out.write(struct.pack("<Q", len(pb))); out.write(pb)
        out.write(struct.pack("<Q", len(sb))); out.write(sb)
    for norm in ["input_layernorm", "post_attention_layernorm"]:
        nb = bf16_bytes(get(prefix + norm + ".weight"))
        out.write(struct.pack("<Q", len(nb))); out.write(nb)
    print(f"  layer {l+1}/{NL} ({out.tell()/1e9:.2f} GB)", flush=True)

for key in ["model.embed_tokens.weight", "lm_head.weight", "model.norm.weight"]:
    nb = bf16_bytes(get(key))
    out.write(struct.pack("<Q", len(nb))); out.write(nb)
    print(f"wrote {key}: {len(nb)/1e6:.0f} MB", flush=True)

out.close()
print(f"DONE -> {DST} ({os.path.getsize(DST)/1e9:.2f} GB)", flush=True)
