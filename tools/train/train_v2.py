"""Enhanced drafter training — more data, more epochs, better loss."""
import torch, torch.nn as nn, torch.nn.functional as F, numpy as np, os, json, copy
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, DynamicCache
from tqdm import tqdm

OUT = "C:/tmp/drafter_v2"
os.makedirs(OUT, exist_ok=True)

print("Loading model...")
config = AutoConfig.from_pretrained('Nanbeige/Nanbeige4.2-3B', trust_remote_code=True)
config.rope_scaling = None
model = AutoModelForCausalLM.from_pretrained('Nanbeige/Nanbeige4.2-3B', config=config,
    torch_dtype=torch.bfloat16, trust_remote_code=True).cuda().eval()
tokenizer = AutoTokenizer.from_pretrained('Nanbeige/Nanbeige4.2-3B', trust_remote_code=True)
H = model.config.hidden_size

# Diverse training texts
texts = [
    "Write a Python function to sort a list using quicksort. def quicksort(arr):",
    "Explain the time complexity of binary search. Binary search has O(log n).",
    "What is the derivative of sin(x)? The derivative is cos(x).",
    "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
    "The capital of France is Paris. The capital of Japan is Tokyo.",
    "Calculate the area of a circle with radius 5. Area = pi * r^2 = 78.54.",
    "Write a SQL query to find the top 10 customers. SELECT * FROM customers ORDER BY revenue DESC LIMIT 10;",
    "Explain how transformer attention works. Attention computes weighted sum of values.",
    "Write a function to reverse a string. def reverse(s): return s[::-1]",
    "What is the Pythagorean theorem? a^2 + b^2 = c^2.",
    "How do you implement a binary search tree? class Node: def __init__(self, val): self.val = val",
    "Write a regex to match email addresses. [a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+",
    "Explain the difference between TCP and UDP. TCP is connection-oriented, UDP is not.",
    "What is dynamic programming? Breaking problems into overlapping subproblems.",
    "Write a function to check if a number is prime. def is_prime(n):",
    "Explain gradient descent. Gradient descent minimizes loss by following the negative gradient.",
    "What is the time complexity of merge sort? O(n log n).",
    "Write a Python decorator for timing functions. def timer(func):",
    "Explain the CAP theorem. A distributed system can have at most 2 of: Consistency, Availability, Partition tolerance.",
    "How does HTTPS work? HTTPS uses TLS/SSL for encrypted communication.",
    "Write a function to find the maximum element in a list. def find_max(lst):",
    "What is recursion? A function that calls itself with a smaller problem.",
    "Explain Big O notation. Big O describes the upper bound of algorithm complexity.",
    "Write a Python class for a stack. class Stack: def push(self, item):",
    "What is a hash table? A data structure that maps keys to values using a hash function.",
    "Write a function to merge two sorted lists. def merge(a, b):",
    "Explain the difference between processes and threads. Processes have separate memory.",
    "What is REST? Representational State Transfer, an architectural style for APIs.",
    "Write a function to check if a string is a palindrome. def is_palindrome(s):",
    "Explain how databases use indexes. Indexes speed up queries by pre-sorting data.",
    "Hello! How can I help you today?",
    "What is machine learning? Training algorithms to learn patterns from data.",
    "Write a Python function to compute factorial. def factorial(n):",
    "Explain the difference between SQL and NoSQL. SQL is relational, NoSQL is not.",
    "What is the meaning of life? 42, according to Douglas Adams.",
    "Write a function to find the GCD of two numbers. def gcd(a, b):",
    "Explain how neural networks learn. Through backpropagation and gradient descent.",
    "What is cloud computing? Delivering computing services over the internet.",
    "Write a Python function to flatten a nested list. def flatten(lst):",
    "Explain the difference between async and sync programming.",
    "What is Docker? A platform for containerizing applications.",
    "Write a function to detect a cycle in a linked list. def has_cycle(head):",
    "Explain the Chandy-Lamport snapshot algorithm for distributed systems.",
    "What is the difference between GET and POST requests?",
    "Write a Python generator for Fibonacci numbers. def fib():",
    "Explain map-reduce programming model for big data processing.",
] * 100  # 4600 texts

# Collect hidden states
print("Collecting hidden states...")
hidden_list, next_hidden_list, token_list, next_token_list = [], [], [], []
with torch.no_grad():
    for text in tqdm(texts[:2000], desc="Collecting"):
        inputs = tokenizer(text, return_tensors='pt', max_length=128, truncation=True).to('cuda')
        if inputs.input_ids.shape[1] < 4: continue
        out = model(**inputs, past_key_values=DynamicCache(), output_hidden_states=True)
        h = out.hidden_states[-1][0]
        for t in range(h.shape[0] - 2):
            hidden_list.append(h[t].cpu())
            next_hidden_list.append(h[t+1].cpu())
            token_list.append(inputs.input_ids[0, t+1].cpu())
            next_token_list.append(inputs.input_ids[0, t+2].cpu())

hidden = torch.stack(hidden_list)
next_hidden = torch.stack(next_hidden_list)
tokens = torch.stack(token_list)
next_tokens = torch.stack(next_token_list)
N = len(hidden)
print(f"Collected {N} samples")

# Create drafter
print("Creating drafter...")
drafter = type(model.model.layers[0])(model.config, layer_idx=0).to(torch.bfloat16).cuda()
drafter.load_state_dict(model.model.layers[0].state_dict())
drafter_norm = copy.deepcopy(model.model.norm)
embed = model.model.embed_tokens
lm_head = model.lm_head
for p in model.parameters(): p.requires_grad = False
params = list(drafter.parameters()) + list(drafter_norm.parameters())
optimizer = torch.optim.AdamW(params, lr=3e-4, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)

# Train with BOTH hidden MSE and token CE loss
BS = 32
for epoch in range(5):
    perm = torch.randperm(N)
    total_loss = total_mse = total_ce = 0
    for i in range(0, N, BS):
        idx = perm[i:i+BS]
        h = hidden[idx].cuda()
        h_next = next_hidden[idx].cuda()
        tok = tokens[idx].cuda()
        ntok = next_tokens[idx].cuda()

        x = (h + embed(tok).to(h.dtype)).unsqueeze(1)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            layer_out = drafter(x, attention_mask=None,
                                position_ids=torch.tensor([[0]], device='cuda'))
            pred = drafter_norm(layer_out[0]).squeeze(1)
            mse_loss = F.mse_loss(pred.float(), h_next.float())
            logits = lm_head(pred)
            ce_loss = F.cross_entropy(logits, ntok)
            loss = mse_loss + 0.5 * ce_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()
        total_loss += loss.item(); total_mse += mse_loss.item(); total_ce += ce_loss.item()
    scheduler.step()
    nb = N // BS
    print(f"Epoch {epoch+1}: loss={total_loss/nb:.4f} mse={total_mse/nb:.6f} ce={total_ce/nb:.4f}")

# Export Q4
print("Exporting...")
def q4(w, group=64):
    w = w.detach().float().cpu().numpy().flatten()
    pad = (group - len(w) % group) % group
    w = np.pad(w, (0, pad)).reshape(-1, group)
    s = np.maximum(np.abs(w).max(axis=1) / 8.0, 1e-8).astype(np.float16)
    q = np.round(w / s[:, None]).clip(-8, 7).astype(np.int32) + 8
    q = q.astype(np.uint8).clip(0, 15)
    return (q[:, 0::2] | (q[:, 1::2] << 4)).flatten(), s

export = {}
for name, w in [('q_proj', drafter.self_attn.q_proj.weight), ('k_proj', drafter.self_attn.k_proj.weight),
                ('v_proj', drafter.self_attn.v_proj.weight), ('o_proj', drafter.self_attn.o_proj.weight),
                ('gate_proj', drafter.mlp.gate_proj.weight), ('up_proj', drafter.mlp.up_proj.weight),
                ('down_proj', drafter.mlp.down_proj.weight), ('lm_head', lm_head.weight)]:
    p, s = q4(w); export[f'{name}_packed'] = p; export[f'{name}_scales'] = s
export['embed'] = embed.weight.detach().to(torch.float16).cpu().numpy()
export['input_ln'] = drafter.input_layernorm.weight.detach().to(torch.float16).cpu().numpy()
export['post_ln'] = drafter.post_attention_layernorm.weight.detach().to(torch.float16).cpu().numpy()
export['final_norm'] = drafter_norm.weight.detach().to(torch.float16).cpu().numpy()
export['_config'] = np.array(json.dumps({
    'hidden_dim': H, 'n_heads': model.config.num_attention_heads,
    'n_kv_heads': getattr(model.config, 'num_key_value_heads', model.config.num_attention_heads),
    'head_dim': 128, 'intermediate': model.config.intermediate_size,
    'vocab_size': model.config.vocab_size, 'eps': model.config.rms_norm_eps,
}))
np.savez(f"{OUT}/drafter.npz", **export)
print(f"Done! {OUT}/drafter.npz ({os.path.getsize(f'{OUT}/drafter.npz')/1e6:.1f} MB)")
