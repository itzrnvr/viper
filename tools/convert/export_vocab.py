# Export a compact vocab for the C++ greedy-longest-match tokenizer.
# Format: u32 n_tokens, u32 bos_id, u32 eos_id, u32 im_start_id, u32 im_end_id,
#         then per id (0..n-1): u32 byte_len, utf8 bytes (token string with ▁).
import json, struct, sys
from transformers import AutoTokenizer

SRC = "D:/hf-cache/Nanbeige4.2-3B"
DST = "D:/dev/viper/artifacts/vocab.bin"

tok = AutoTokenizer.from_pretrained(SRC, trust_remote_code=True)
vocab = tok.get_vocab()  # {str: id}
n = max(vocab.values()) + 1
strings = [b""] * n
for s, i in vocab.items():
    strings[i] = s.encode("utf-8")

bos = tok.bos_token_id or 0
eos = tok.eos_token_id or 0
im_start = vocab.get("<|im_start|>", 0)
im_end = vocab.get("<|im_end|>", 0)
print(f"vocab n={n} bos={bos} eos={eos} im_start={im_start} im_end={im_end}")

with open(DST, "wb") as f:
    f.write(struct.pack("<5I", n, bos, eos, im_start, im_end))
    for i in range(n):
        f.write(struct.pack("<I", len(strings[i])))
        f.write(strings[i])
print(f"wrote {DST}")

# quick sanity: encode a test prompt with the real tokenizer
test = "<|im_start|>user\nHello, how are you?<|im_end|>\n<|im_start|>assistant\n"
ids = tok(test, add_special_tokens=False)["input_ids"]
print("sample ids:", ids[:20], "len:", len(ids))
print("roundtrip:", tok.decode(ids)[:60])
