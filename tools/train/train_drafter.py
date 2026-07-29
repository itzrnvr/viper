#!/usr/bin/env python3
"""
EAGLE Drafter Training for Nanbeige4.2-3B.

Trains a 1-layer transformer drafter for speculative decoding.
The drafter takes the base model's hidden state and autoregressively
generates K draft tokens. Expected acceptance: 70-85%.

Usage on Colab:
    !pip install transformers torch
    # Upload Nanbeige4.2-3B to Colab or use HF Hub
    !python train_drafter.py --model Nanbeige/Nanbeige4.2-3B --epochs 3

Output:
    drafter_weights.pt — PyTorch checkpoint
    drafter.viper — Q4 packed weights for viper engine

The drafter architecture (1 layer, same hidden_dim as base):
    Input: h_t (base model hidden) + embed(draft_token)
    Layer: self-attention + MLP (same structure as base model layer)
    Output: h_{t+1} (predicted next hidden state)
    Token: lm_head(h_{t+1}) → draft token

Training: teacher forcing on text corpus.
    Loss = MSE(drafter_output, target_hidden) + CE(lm_head(output), target_token)
"""

import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
from tqdm import tqdm
import json

class EagleDrafter(nn.Module):
    """
    1-layer transformer drafter for speculative decoding.
    Uses the same architecture as a single Nanbeige layer.
    """
    def __init__(self, hidden_dim=3072, n_heads=48, n_kv_heads=8,
                 head_dim=128, intermediate=10752, vocab_size=166144, eps=1e-5):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.intermediate = intermediate
        self.eps = eps

        # Embedding (shared with base model — will be copied)
        self.embed = nn.Embedding(vocab_size, hidden_dim)

        # Attention projections
        self.q_proj = nn.Linear(hidden_dim, n_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, n_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, n_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * head_dim, hidden_dim, bias=False)

        # MLP projections
        self.gate_proj = nn.Linear(hidden_dim, intermediate, bias=False)
        self.up_proj = nn.Linear(hidden_dim, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, hidden_dim, bias=False)

        # Layer norms
        self.input_layernorm = nn.RMSNorm(hidden_dim, eps=eps)
        self.post_attention_layernorm = nn.RMSNorm(hidden_dim, eps=eps)

        # Final norm + LM head (shared with base model)
        self.final_norm = nn.RMSNorm(hidden_dim, eps=eps)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)

    def forward(self, hidden_state, draft_token_id, position,
                kv_cache_k=None, kv_cache_v=None):
        """
        Single-step forward (autoregressive drafting).

        Args:
            hidden_state: [batch, hidden] — base model's output
            draft_token_id: [batch] — token to process at this position
            position: int — sequence position for RoPE
            kv_cache_k/v: [batch, n_kv_heads, head_dim] — drafter's KV cache

        Returns:
            new_hidden: [batch, hidden] — predicted next hidden state
            logits: [batch, vocab] — token prediction logits
            new_kv_k/v: updated KV cache
        """
        batch = hidden_state.shape[0]

        # Input: hidden_state + embedding
        x = hidden_state + self.embed(draft_token_id)  # [batch, hidden]

        # Attention sublayer
        residual = x
        x_norm = self.input_layernorm(x)  # [batch, hidden]

        q = self.q_proj(x_norm)  # [batch, n_q * head_dim]
        k = self.k_proj(x_norm)  # [batch, n_kv * head_dim]
        v = self.v_proj(x_norm)  # [batch, n_kv * head_dim]

        q = q.view(batch, self.n_heads, self.head_dim)
        k = k.view(batch, self.n_kv_heads, self.head_dim)
        v = v.view(batch, self.n_kv_heads, self.head_dim)

        # Apply RoPE (simplified — use base model's rope_theta)
        theta = 70000000.0
        inv_freq = 1.0 / (theta ** (torch.arange(0, self.head_dim, 2,
                          device=q.device).float() / self.head_dim))
        pos_tensor = torch.tensor(float(position), device=q.device)
        freqs = pos_tensor * inv_freq
        cos = torch.cos(freqs).repeat_interleave(2)
        sin = torch.sin(freqs).repeat_interleave(2)

        def rotate_half(x):
            x1, x2 = x[..., :self.head_dim//2], x[..., self.head_dim//2:]
            return torch.cat((-x2, x1), dim=-1)

        q_rot = q * cos + rotate_half(q) * sin
        k_rot = k * cos + rotate_half(k) * sin

        # Append to KV cache
        if kv_cache_k is not None:
            all_k = torch.cat([kv_cache_k, k_rot.unsqueeze(1)], dim=1)
            all_v = torch.cat([kv_cache_v, v.unsqueeze(1)], dim=1)
        else:
            all_k = k_rot.unsqueeze(1)  # [batch, 1, n_kv, head_dim]
            all_v = v.unsqueeze(1)

        # GQA attention
        scale = 1.0 / (self.head_dim ** 0.5)
        # Expand q for GQA: [batch, n_heads, head_dim] → repeat kv group
        head_per_kv = self.n_heads // self.n_kv_heads
        q_expanded = q_rot.repeat_interleave(head_per_kv, dim=0)
        # [n_heads, head_dim] vs [n_kv, head_dim] → need per-head attention

        attn_output = torch.zeros_like(q_rot)
        for h in range(self.n_heads):
            h_kv = h // head_per_kv
            scores = torch.zeros(batch, all_k.shape[1], device=q.device)
            for pos_idx in range(all_k.shape[1]):
                k_vec = all_k[:, pos_idx, h_kv, :]  # [batch, head_dim]
                scores[:, pos_idx] = (q_rot[:, h, :] * k_vec).sum(-1) * scale
            weights = F.softmax(scores, dim=-1)
            for pos_idx in range(all_k.shape[1]):
                v_vec = all_v[:, pos_idx, h_kv, :]
                attn_output[:, h, :] += weights[:, pos_idx:pos_idx+1] * v_vec

        attn_output = attn_output.reshape(batch, -1)  # [batch, n_q * head_dim]
        attn_output = self.o_proj(attn_output)
        x = residual + attn_output

        # MLP sublayer
        residual = x
        x_norm = self.post_attention_layernorm(x)
        gate = self.gate_proj(x_norm)
        up = self.up_proj(x_norm)
        x = F.silu(gate) * up
        x = self.down_proj(x)
        x = residual + x

        # Final norm
        x = self.final_norm(x)

        # Token prediction
        logits = self.lm_head(x)

        return x, logits, all_k, all_v


class HiddenStateDataset(Dataset):
    """Pre-collected hidden states from base model."""
    def __init__(self, data_path):
        self.data = torch.load(data_path)

    def __len__(self):
        return len(self.data['hidden'])

    def __getitem__(self, idx):
        return {
            'hidden': self.data['hidden'][idx],
            'next_hidden': self.data['next_hidden'][idx],
            'token': self.data['tokens'][idx],
            'next_token': self.data['next_tokens'][idx],
        }


def collect_hidden_states(model, tokenizer, texts, max_length=512, device='cuda'):
    """Run base model on texts and collect hidden states."""
    model.eval()
    all_hidden = []
    all_next_hidden = []
    all_tokens = []
    all_next_tokens = []

    with torch.no_grad():
        for text in tqdm(texts, desc="Collecting hidden states"):
            inputs = tokenizer(text, return_tensors='pt', max_length=max_length,
                             truncation=True).to(device)
            outputs = model(**inputs, output_hidden_states=True)

            # Get last hidden state before lm_head
            hidden = outputs.hidden_states[-1]  # [1, seq_len, hidden]

            # Collect pairs: (h_t, h_{t+1}, token_{t+1}, token_{t+2})
            for t in range(hidden.shape[1] - 2):
                all_hidden.append(hidden[0, t].cpu())
                all_next_hidden.append(hidden[0, t+1].cpu())
                all_tokens.append(inputs.input_ids[0, t+1].cpu())
                all_next_tokens.append(inputs.input_ids[0, t+2].cpu())

    return {
        'hidden': torch.stack(all_hidden),
        'next_hidden': torch.stack(all_next_hidden),
        'tokens': torch.stack(all_tokens),
        'next_tokens': torch.stack(all_next_tokens),
    }


def train_drafter(base_model_name, output_dir, epochs=3, batch_size=32,
                  learning_rate=1e-4, max_seq_len=512, num_texts=10000):
    """Train the EAGLE drafter."""

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load base model and tokenizer
    print(f"Loading base model: {base_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).to(device)
    base_model.eval()

    # Get model config
    config = base_model.config
    hidden_dim = config.hidden_size
    print(f"Model config: hidden={hidden_dim}, layers={config.num_hidden_layers}")

    # Create drafter
    drafter = EagleDrafter(
        hidden_dim=hidden_dim,
        n_heads=config.num_attention_heads,
        n_kv_heads=getattr(config, 'num_key_value_heads', config.num_attention_heads),
        head_dim=hidden_dim // config.num_attention_heads,
        intermediate=config.intermediate_size,
        vocab_size=config.vocab_size,
    ).to(device)

    # Initialize drafter from base model's first layer
    print("Initializing drafter from base model layer 0...")
    base_layer = base_model.model.layers[0]
    drafter.q_proj.weight.data = base_layer.self_attn.q_proj.weight.data.clone()
    drafter.k_proj.weight.data = base_layer.self_attn.k_proj.weight.data.clone()
    drafter.v_proj.weight.data = base_layer.self_attn.v_proj.weight.data.clone()
    drafter.o_proj.weight.data = base_layer.self_attn.o_proj.weight.data.clone()
    drafter.gate_proj.weight.data = base_layer.mlp.gate_proj.weight.data.clone()
    drafter.up_proj.weight.data = base_layer.mlp.up_proj.weight.data.clone()
    drafter.down_proj.weight.data = base_layer.mlp.down_proj.weight.data.clone()
    drafter.input_layernorm.weight.data = base_layer.input_layernorm.weight.data.clone()
    drafter.post_attention_layernorm.weight.data = base_layer.post_attention_layernorm.weight.data.clone()
    drafter.embed.weight.data = base_model.model.embed_tokens.weight.data.clone()
    drafter.final_norm.weight.data = base_model.model.norm.weight.data.clone()
    drafter.lm_head.weight.data = base_model.lm_head.weight.data.clone()

    # Collect training data
    print("Collecting hidden states from text corpus...")
    # Use a mix of code, math, and general text
    sample_texts = [
        "Write a Python function to sort a list using quicksort.",
        "Explain the time complexity of binary search.",
        "What is the derivative of sin(x)?",
        "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
        "The capital of France is Paris.",
        "Calculate the area of a circle with radius 5.",
        "Write a SQL query to find the top 10 customers by revenue.",
        "Explain how transformer attention works.",
    ] * (num_texts // 8 + 1)

    data = collect_hidden_states(base_model, tokenizer, sample_texts[:num_texts],
                                  max_length=max_seq_len, device=device)

    # Save training data
    torch.save(data, os.path.join(output_dir, 'training_data.pt'))
    print(f"Collected {len(data['hidden'])} hidden state pairs")

    # Training loop
    optimizer = torch.optim.AdamW(drafter.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    dataset = HiddenStateDataset(os.path.join(output_dir, 'training_data.pt'))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    drafter.train()
    for epoch in range(epochs):
        total_loss = 0
        total_mse = 0
        total_ce = 0

        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            h = batch['hidden'].to(device)
            h_next = batch['next_hidden'].to(device)
            tokens = batch['token'].to(device)
            next_tokens = batch['next_token'].to(device)

            # Forward: predict next hidden state
            pred_hidden, logits, _, _ = drafter(h, tokens, position=0)

            # Loss: MSE on hidden + CE on token
            mse_loss = F.mse_loss(pred_hidden, h_next)
            ce_loss = F.cross_entropy(logits, next_tokens)
            loss = mse_loss + 0.1 * ce_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(drafter.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            total_mse += mse_loss.item()
            total_ce += ce_loss.item()

        scheduler.step()
        n_batches = len(dataloader)
        print(f"Epoch {epoch+1}: loss={total_loss/n_batches:.4f} "
              f"mse={total_mse/n_batches:.6f} ce={total_ce/n_batches:.4f}")

    # Save drafter checkpoint
    torch.save(drafter.state_dict(), os.path.join(output_dir, 'drafter_weights.pt'))
    print(f"Drafter saved to {output_dir}/drafter_weights.pt")

    # Export to Q4 format for viper engine
    export_to_viper(drafter, os.path.join(output_dir, 'drafter.viper'))

    return drafter


def quantize_q4(weight, group_size=64):
    """Quantize weight tensor to 4-bit with group scales."""
    orig_shape = weight.shape
    w = weight.flatten().to(torch.float32)

    # Pad to multiple of group_size
    pad = (group_size - w.shape[0] % group_size) % group_size
    if pad > 0:
        w = torch.cat([w, torch.zeros(pad)])

    w = w.reshape(-1, group_size)
    scales = w.abs().max(dim=1).values / 8.0  # Q4 range [-8, 7]
    scales = scales.clamp(min=1e-8)

    q = torch.round(w / scales.unsqueeze(1)).clamp(-8, 7)
    q_u = (q + 8).to(torch.uint8)  # [0, 15]

    # Pack 2 values per byte
    q_packed = q_u[:, 0::2] | (q_u[:, 1::2] << 4)

    return q_packed.flatten().numpy(), scales.to(torch.float16).numpy()


def export_to_viper(drafter, output_path):
    """Export drafter weights to viper Q4 format."""
    print(f"Exporting drafter to {output_path}...")

    # Quantize each weight matrix
    weights = {
        'q_proj': drafter.q_proj.weight,
        'k_proj': drafter.k_proj.weight,
        'v_proj': drafter.v_proj.weight,
        'o_proj': drafter.o_proj.weight,
        'gate_proj': drafter.gate_proj.weight,
        'up_proj': drafter.up_proj.weight,
        'down_proj': drafter.down_proj.weight,
        'lm_head': drafter.lm_head.weight,
    }

    # Save as numpy npz for now (converter script will handle .viper format)
    export_data = {}
    for name, w in weights.items():
        packed, scales = quantize_q4(w)
        export_data[f'{name}_packed'] = packed
        export_data[f'{name}_scales'] = scales
        print(f"  {name}: {w.shape} → {len(packed)} bytes Q4")

    # Save norms and embedding in fp16
    export_data['embed'] = drafter.embed.weight.data.to(torch.float16).numpy()
    export_data['input_ln'] = drafter.input_layernorm.weight.data.to(torch.float16).numpy()
    export_data['post_ln'] = drafter.post_attention_layernorm.weight.data.to(torch.float16).numpy()
    export_data['final_norm'] = drafter.final_norm.weight.data.to(torch.float16).numpy()

    # Save config
    export_data['_config'] = json.dumps({
        'hidden_dim': drafter.hidden_dim,
        'n_heads': drafter.n_heads,
        'n_kv_heads': drafter.n_kv_heads,
        'head_dim': drafter.head_dim,
        'intermediate': drafter.intermediate,
        'vocab_size': drafter.embed.num_embeddings,
        'eps': drafter.eps,
    })

    np.savez(output_path.replace('.viper', '.npz'), **export_data)
    print(f"Drafter exported ({os.path.getsize(output_path.replace('.viper', '.npz')) / 1e6:.1f} MB)")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train EAGLE drafter for Nanbeige4.2-3B')
    parser.add_argument('--model', default='Nanbeige/Nanbeige4.2-3B',
                        help='HuggingFace model name')
    parser.add_argument('--output', default='./drafter_output',
                        help='Output directory')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num-texts', type=int, default=10000)

    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)

    train_drafter(
        base_model_name=args.model,
        output_dir=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        num_texts=args.num_texts,
    )

    print("\n✅ Drafter training complete!")
    print(f"   Weights: {args.output}/drafter_weights.pt")
    print(f"   Q4 export: {args.output}/drafter.npz")
    print("\nNext steps:")
    print("1. Copy drafter.npz to your viper artifacts directory")
    print("2. Run: python convert_drafter.py drafter.npz drafter.viper")
    print("3. Run: viper_cli --model model.viper --drafter drafter.viper --spec-k 4")
