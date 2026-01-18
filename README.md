# Task-Try_2

## Model

**Model repo:** bartowski/Qwen2.5-Coder-3B-Instruct-abliterated-GGUF  
**Source:** https://huggingface.co/bartowski/Qwen2.5-Coder-3B-Instruct-abliterated-GGUF  

### Download
Using `huggingface-cli`:
```bash
pip install -U "huggingface_hub[cli]"
huggingface-cli download bartowski/Qwen2.5-Coder-3B-Instruct-abliterated-GGUF \
  --include "*Q4_K_M*.gguf" \
  --local-dir ./models/qwen2.5-gguf \
  --local-dir-use-symlinks False
