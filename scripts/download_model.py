import os
os.environ['HF_HOME'] = 'D:/hf-cache'
from huggingface_hub import snapshot_download
p = snapshot_download(
    repo_id='Nanbeige/Nanbeige4.2-3B',
    local_dir='D:/hf-cache/Nanbeige4.2-3B',
    local_dir_use_symlinks=False,
    allow_patterns=['*.json', '*.py', '*.safetensors', 'tokenizer*', '*.md']
)
print('OK', p)
