#!/usr/bin/env python
"""HF reference for viper comparison."""
import os
os.environ['HF_HOME'] = 'D:/hf-cache'
import torch
import importlib.util
from transformers import AutoTokenizer

# Load the upstream modeling_nanbeige.py directly and patch _init_rope.
hub_path = 'D:/hf-cache/modules/transformers_modules/Nanbeige/Nanbeige4_dot_2_hyphen_3B/f56ec5a9650268aa098496734743c25ea778bd2d/modeling_nanbeige.py'
spec = importlib.util.spec_from_file_location("nb", hub_path)
nb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(nb)

# Patch _init_rope to handle rope_scaling=None
orig_init_rope = nb.NanbeigeAttention._init_rope
def safe_init_rope(self):
    if self.config.rope_scaling is None:
        return
    return orig_init_rope(self)
nb.NanbeigeAttention._init_rope = safe_init_rope

tok = AutoTokenizer.from_pretrained('Nanbeige/Nanbeige4.2-3B', use_fast=False, trust_remote_code=True, cache_dir='D:/hf-cache')
print(f"[hf] vocab={tok.vocab_size} bos={tok.bos_token_id} eos={tok.eos_token_id}")

model = nb.NanbeigeForCausalLM.from_pretrained('Nanbeige/Nanbeige4.2-3B', torch_dtype=torch.bfloat16, trust_remote_code=True, cache_dir='D:/hf-cache').cuda().eval()
print(f"[hf] model loaded on {next(model.parameters()).device}")

# Use the same chat template as viper's test
prompt_text = "The capital of France is"
full = f"<|im_start|>user\n{prompt_text}<|im_end|>\n<|im_start|>assistant\n"
ids = tok(full, return_tensors='pt', add_special_tokens=False).input_ids.cuda()
print(f"[hf] prompt tokens ({ids.shape[1]}): {ids[0].tolist()}")
print(f"[hf] prompt: {full!r}")

for step in range(10):
    with torch.no_grad():
        out = model(input_ids=ids, use_cache=False)
    logits = out.logits[0, -1]
    next_id = logits.argmax().item()
    top5 = logits.topk(5)
    print(f"[hf] step {step+1}: argmax={next_id} '{tok.decode([next_id])}' top5={[(t.item(), tok.decode([t.item()])) for t in top5.indices]}")
    if next_id == tok.eos_token_id:
        print("[hf] EOS")
        break
    ids = torch.cat([ids, torch.tensor([[next_id]], device='cuda')], dim=1)
