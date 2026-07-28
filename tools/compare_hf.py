#!/usr/bin/env python
"""HF reference for viper comparison. Handles the relative-import problem
by preloading the config module, then patching the model file's relative
imports to use sys.modules lookup."""
import os
os.environ['HF_HOME'] = 'D:/hf-cache'
import sys
import torch
import importlib.util
from transformers import AutoTokenizer, AutoConfig

hub_dir = 'D:/hf-cache/modules/transformers_modules/Nanbeige/Nanbeige4_dot_2_hyphen_3B/f56ec5a9650268aa098496734743c25ea778bd2d'
config_path = f'{hub_dir}/configuration_nanbeige.py'
model_path = f'{hub_dir}/modeling_nanbeige.py'

# Load config and model as a fake package.
pkg_name = 'nb_viper'
config_mod = importlib.import_module(config_path.replace('/', '.').replace('.py', ''))
# Inject the config into sys.modules so relative imports resolve.
sys.modules[f'{pkg_name}.configuration_nanbeige'] = config_mod

with open(model_path, 'r', encoding='utf-8') as f:
    src = f.read()
# Replace relative imports with absolute lookups via sys.modules.
src = src.replace('from .configuration_nanbeige import', f'from sys import modules as _m; _m["{pkg_name}.configuration_nanbeige"] = _m.get("{pkg_name}.configuration_nanbeige", _m.get("configuration_nanbeige")); from {pkg_name}.configuration_nanbeige import')

spec = importlib.util.spec_from_loader(pkg_name, loader=None, is_package=True)
nb = importlib.util.module_from_spec(spec)
nb.__path__ = [hub_dir]
sys.modules[pkg_name] = nb
exec(compile(src, model_path, 'exec'), nb.__dict__)

# Patch _init_rope to handle rope_scaling=None.
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
    decoded = tok.decode([next_id])
    print(f"[hf] step {step+1}: argmax={next_id} '{decoded}' top5={[(t.item(), tok.decode([t.item()])) for t in top5.indices.tolist()]}")
    if next_id == tok.eos_token_id:
        print("[hf] EOS")
        break
    ids = torch.cat([ids, torch.tensor([[next_id]], device='cuda')], dim=1)
