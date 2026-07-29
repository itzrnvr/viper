#!/usr/bin/env python3
"""
EAGLE Drafter Training for Nanbeige4.2-3B — Colab-ready.

Simplified: uses a 1-layer transformer drafter initialized from the base
model's first layer. Properly vectorized attention via SDPA.

Run on Colab:
    !pip install transformers torch datasets
    !python train_drafter_colab.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import numpy as np
import os, json, struct, argparse
from tqdm import tqdm


def create_drafter(base_model):
    """Create a 1-layer drafter from the base model's first layer."""
    cfg = base_model.config
    # Deep copy the first layer as our drafter
    layer = type(base_model.model.layers[0])(cfg).to(base_model.dtype)
    layer.load_state_dict(base_model.model.layers[0].state_dict())
    
    # Also copy final norm
    norm = type(base_model.model.norm)(cfg).to(base_model.dtype)
    norm.load_state_dict(base_model.model.norm.state_dict())
    
    return layer, norm


def collect_hidden_states(model, tokenizer, texts, max_len=256, device='cuda', max_samples=5000):
    """Collect (hidden_t, hidden_t+1, token_t+1) triples from text."""
    model.eval()
    data = {'hidden': [], 'next_hidden': [], 'tokens': []}
    
    with torch.no_grad():
        for text in tqdm(texts[:max_samples], desc="Collecting"):
            if len(data['hidden']) >= max_samples * 10:
                break
            inputs = tokenizer(text, return_tensors='pt', max_length=max_len,
                             truncation=True).to(device)
            if inputs.input_ids.shape[1] < 4:
                continue
            outputs = model(**inputs, output_hidden_states=True)
            h = outputs.hidden_states[-1][0]  # [seq, hidden]
            
            for t in range(h.shape[0] - 1):
                data['hidden'].append(h[t].cpu())
                data['next_hidden'].append(h[t+1].cpu())
                data['tokens'].append(inputs.input_ids[0, t+1].cpu())
    
    return {
        'hidden': torch.stack(data['hidden']),
        'next_hidden': torch.stack(data['next_hidden']),
        'tokens': torch.stack(data['tokens']),
    }


def train(base_model_name, output_dir, epochs=2, batch_size=16, lr=5e-5):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading {base_model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).to(device).eval()
    
    cfg = model.config
    H = cfg.hidden_size
    print(f"hidden={H}, vocab={cfg.vocab_size}")
    
    # Create drafter (1 layer)
    drafter_layer, drafter_norm = create_drafter(model)
    drafter_layer.to(device)
    drafter_norm.to(device)
    
    # Embedding + lm_head from base model
    embed = model.model.embed_tokens
    lm_head = model.lm_head
    
    # Freeze base model
    for p in model.parameters():
        p.requires_grad = False
    
    # Trainable params: drafter layer + norm
    params = list(drafter_layer.parameters()) + list(drafter_norm.parameters())
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
    
    # Collect training data
    print("Loading text corpus...")
    try:
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        texts = [t for t in ds['text'] if len(t) > 50][:5000]
    except:
        texts = [
            "Write a Python function to sort a list. def sort(lst): return sorted(lst)",
            "The capital of France is Paris. The capital of Japan is Tokyo.",
            "Explain binary search: repeatedly divide the search interval in half.",
            "def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)",
            "Machine learning is a subset of artificial intelligence.",
            "The derivative of x squared is 2x.",
            "SQL query: SELECT * FROM users WHERE age > 18 ORDER BY name;",
            "In computer science, Big O notation describes algorithm complexity.",
        ] * 500
    
    print("Collecting hidden states...")
    data = collect_hidden_states(model, tokenizer, texts, device=device)
    n = len(data['hidden'])
    print(f"Collected {n} samples")
    
    # Training loop
    print(f"\nTraining {epochs} epochs...")
    for epoch in range(epochs):
        perm = torch.randperm(n)
        total_loss = 0
        
        for i in range(0, n, batch_size):
            idx = perm[i:i+batch_size]
            h = data['hidden'][idx].to(device)
            h_target = data['next_hidden'][idx].to(device)
            tokens = data['tokens'][idx].to(device)
            
            # EAGLE input: hidden + embed(token)
            x = h + embed(tokens).to(h.dtype)  # [batch, hidden]
            x = x.unsqueeze(1)  # [batch, 1, hidden] — seq_len=1 for drafter
            
            # Drafter forward (1 layer)
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                layer_out = drafter_layer(x, attention_mask=None,
                                          position_ids=torch.tensor([[0]], device=device))
                pred_hidden = drafter_norm(layer_out[0])  # [batch, 1, hidden]
                pred_hidden = pred_hidden.squeeze(1)  # [batch, hidden]
                
                # Loss: MSE on hidden state + CE on token prediction
                mse_loss = F.mse_loss(pred_hidden.float(), h_target.float())
                
                # Token prediction via shared lm_head
                logits = lm_head(pred_hidden)  # [batch, vocab]
                token_target = data['tokens'][idx].to(device)  # token at t+1
                # Actually, we want to predict the NEXT next token
                # For now: just use hidden MSE
                loss = mse_loss
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            
            total_loss += loss.item()
        
        avg = total_loss / (n / batch_size)
        print(f"  Epoch {epoch+1}: loss={avg:.6f}")
    
    # Save
    print("\nSaving drafter weights...")
    save_drafter_q4(drafter_layer, drafter_norm, model, output_dir)
    print(f"Done! Output: {output_dir}/")


def save_drafter_q4(layer, norm, base_model, output_dir):
    """Export drafter weights as Q4 packed + numpy."""
    cfg = base_model.config
    
    def q4(weight, group=64):
        w = weight.detach().float().cpu().numpy().flatten()
        pad = (group - len(w) % group) % group
        w = np.pad(w, (0, pad)).reshape(-1, group)
        scales = np.abs(w).max(axis=1) / 8.0
        scales = np.maximum(scales, 1e-8).astype(np.float16)
        q = np.round(w / scales[:, None]).clip(-8, 7).astype(np.int32) + 8
        q = q.astype(np.uint8).clip(0, 15)
        packed = q[:, 0::2] | (q[:, 1::2] << 4)
        return packed.flatten(), scales
    
    export = {}
    # Layer weights
    attn = layer.self_attn
    mlp = layer.mlp
    for name, w in [
        ('q_proj', attn.q_proj.weight),
        ('k_proj', attn.k_proj.weight),
        ('v_proj', attn.v_proj.weight),
        ('o_proj', attn.o_proj.weight),
        ('gate_proj', mlp.gate_proj.weight),
        ('up_proj', mlp.up_proj.weight),
        ('down_proj', mlp.down_proj.weight),
        ('lm_head', base_model.lm_head.weight),
    ]:
        p, s = q4(w)
        export[f'{name}_packed'] = p
        export[f'{name}_scales'] = s
        print(f"  {name}: {w.shape} → {len(p)//1024}KB Q4")
    
    # BF16 weights
    export['embed'] = base_model.model.embed_tokens.weight.detach().to(torch.float16).cpu().numpy()
    export['input_ln'] = layer.input_layernorm.weight.detach().to(torch.float16).cpu().numpy()
    export['post_ln'] = layer.post_attention_layernorm.weight.detach().to(torch.float16).cpu().numpy()
    export['final_norm'] = norm.weight.detach().to(torch.float16).cpu().numpy()
    
    export['_config'] = json.dumps({
        'hidden_dim': cfg.hidden_size,
        'n_heads': cfg.num_attention_heads,
        'n_kv_heads': getattr(cfg, 'num_key_value_heads', cfg.num_attention_heads),
        'head_dim': cfg.hidden_size // cfg.num_attention_heads,
        'intermediate': cfg.intermediate_size,
        'vocab_size': cfg.vocab_size,
        'eps': cfg.rms_norm_eps,
    })
    
    path = os.path.join(output_dir, 'drafter.npz')
    np.savez(path, **export)
    size = os.path.getsize(path) / 1e6
    print(f"  Total: {path} ({size:.1f} MB)")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='Nanbeige/Nanbeige4.2-3B')
    parser.add_argument('--output', default='./drafter_output')
    parser.add_argument('--epochs', type=int, default=2)
    args = parser.parse_args()
    train(args.model, args.output, epochs=args.epochs)
