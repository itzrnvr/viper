"""Quick EAGLE drafter training — runs locally on RTX 3070 Ti."""
import torch, torch.nn as nn, torch.nn.functional as F, numpy as np, os, json
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, DynamicCache
from tqdm import tqdm

OUT = "C:/tmp/drafter_output"
os.makedirs(OUT, exist_ok=True)

# Load model
print("Loading model...")
config = AutoConfig.from_pretrained('Nanbeige/Nanbeige4.2-3B', trust_remote_code=True)
config.rope_scaling = None
model = AutoModelForCausalLM.from_pretrained('Nanbeige/Nanbeige4.2-3B', config=config,
    torch_dtype=torch.bfloat16, trust_remote_code=True).cuda().eval()
tokenizer = AutoTokenizer.from_pretrained('Nanbeige/Nanbeige4.2-3B', trust_remote_code=True)
H = model.config.hidden_size
print(f"hidden={H}, vocab={model.config.vocab_size}")

# Collect hidden states
print("Collecting hidden states...")
texts = [
    "Write a Python function to sort a list using quicksort.",
    "Explain the time complexity of binary search.",
    "What is the derivative of sin(x)?",
    "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
    "The capital of France is Paris.",
    "Calculate the area of a circle with radius 5.",
    "Write a SQL query to find the top 10 customers by revenue.",
    "Explain how transformer attention works.",
    "Write a function to reverse a string in Python.",
    "What is the Pythagorean theorem?",
    "How do you implement a binary search tree?",
    "Write a regex to match email addresses.",
    "Explain the difference between TCP and UDP.",
    "What is dynamic programming?",
    "Write a function to check if a number is prime.",
    "Explain gradient descent.",
    "What is the time complexity of merge sort?",
    "Write a Python decorator for timing functions.",
    "Explain the CAP theorem.",
    "How does HTTPS work?",
] * 50  # 1000 texts

hidden_list, next_hidden_list, token_list = [], [], []
with torch.no_grad():
    for text in tqdm(texts[:500], desc="Collecting"):
        inputs = tokenizer(text, return_tensors='pt', max_length=128, truncation=True).to('cuda')
        if inputs.input_ids.shape[1] < 4: continue
        out = model(**inputs, past_key_values=DynamicCache(), output_hidden_states=True)
        h = out.hidden_states[-1][0]
        for t in range(h.shape[0] - 1):
            hidden_list.append(h[t].cpu())
            next_hidden_list.append(h[t+1].cpu())
            token_list.append(inputs.input_ids[0, t+1].cpu())

hidden = torch.stack(hidden_list)
next_hidden = torch.stack(next_hidden_list)
tokens = torch.stack(token_list)
print(f"Collected {len(hidden)} samples")

# Create drafter (copy layer 0)
print("Creating drafter from layer 0...")
drafter = type(model.model.layers[0])(model.config, layer_idx=0).to(torch.bfloat16).cuda()
drafter.load_state_dict(model.model.layers[0].state_dict())
import copy
drafter_norm = copy.deepcopy(model.model.norm)
drafter_norm.load_state_dict(model.model.norm.state_dict())

embed = model.model.embed_tokens
lm_head = model.lm_head
for p in model.parameters(): p.requires_grad = False

params = list(drafter.parameters()) + list(drafter_norm.parameters())
optimizer = torch.optim.AdamW(params, lr=1e-4, weight_decay=0.01)

# Train
print("Training...")
BS = 16
N = len(hidden)
for epoch in range(2):
    perm = torch.randperm(N)
    total = 0
    for i in range(0, N, BS):
        idx = perm[i:i+BS]
        h = hidden[idx].cuda()
        h_next = next_hidden[idx].cuda()
        tok = tokens[idx].cuda()

        x = (h + embed(tok).to(h.dtype)).unsqueeze(1)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            layer_out = drafter(x, attention_mask=None,
                                position_ids=torch.tensor([[0]], device='cuda'))
            pred = drafter_norm(layer_out[0]).squeeze(1)
            loss = F.mse_loss(pred.float(), h_next.float())

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()
        total += loss.item()
    print(f"  Epoch {epoch+1}: loss={total/(N/BS):.6f}")

# Export Q4
print("Exporting Q4 weights...")
def q4(w, group=64):
    w = w.detach().float().cpu().numpy().flatten()
    pad = (group - len(w) % group) % group
    w = np.pad(w, (0, pad)).reshape(-1, group)
    s = np.maximum(np.abs(w).max(axis=1) / 8.0, 1e-8).astype(np.float16)
    q = np.round(w / s[:, None]).clip(-8, 7).astype(np.int32) + 8
    q = q.astype(np.uint8).clip(0, 15)
    return (q[:, 0::2] | (q[:, 1::2] << 4)).flatten(), s

export = {}
attn = drafter.self_attn
mlp = drafter.mlp
for name, w in [('q_proj', attn.q_proj.weight), ('k_proj', attn.k_proj.weight),
                ('v_proj', attn.v_proj.weight), ('o_proj', attn.o_proj.weight),
                ('gate_proj', mlp.gate_proj.weight), ('up_proj', mlp.up_proj.weight),
                ('down_proj', mlp.down_proj.weight), ('lm_head', lm_head.weight)]:
    p, s = q4(w)
    export[f'{name}_packed'] = p
    export[f'{name}_scales'] = s
    print(f"  {name}: {len(p)//1024}KB")

export['embed'] = embed.weight.detach().to(torch.float16).cpu().numpy()
export['input_ln'] = drafter.input_layernorm.weight.detach().to(torch.float16).cpu().numpy()
export['post_ln'] = drafter.post_attention_layernorm.weight.detach().to(torch.float16).cpu().numpy()
export['final_norm'] = drafter_norm.weight.detach().to(torch.float16).cpu().numpy()
export['_config'] = json.dumps({
    'hidden_dim': H, 'n_heads': model.config.num_attention_heads,
    'n_kv_heads': getattr(model.config, 'num_key_value_heads', model.config.num_attention_heads),
    'head_dim': H // model.config.num_attention_heads,
    'intermediate': model.config.intermediate_size,
    'vocab_size': model.config.vocab_size, 'eps': model.config.rms_norm_eps,
})
np.savez(f"{OUT}/drafter.npz", **export)
print(f"\nDone! {OUT}/drafter.npz ({os.path.getsize(f'{OUT}/drafter.npz')/1e6:.1f} MB)")
