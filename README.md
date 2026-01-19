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
```

### Train
```bash
python train_hier_softprompt_pref.py \
  --model_dir /root/Works-about-the-phd-task/Qwen2.5-Coder-1.5B-Instruct-HF \
  --data_jsonl /root/Works-about-the-phd-task/out/train_split.jsonl \
  --out_dir /root/Works-about-the-phd-task/ckpts_hier \
  --bandit_bin bandit \
  --bf16 \
  --steps 200 \
  --num_candidates 4 \
  --max_new_tokens 256
