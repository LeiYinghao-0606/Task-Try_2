#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
train_hier_softprompt_pref.py

Hierarchical (two-stage) cross-attention soft-prompt training with analyzer-guided preference.
- Stage A: soft prompt (Q) attends to security memory (K,V)
- Stage B: aligned prompt (Q) attends to context memory (K,V)
- LLM is frozen; only prompt-side parameters are trained.
- Preference signal is derived from:
    (1) Bandit risk reduction (lower is better)
    (2) functionality preservation proxy (lower drift is better)

Input JSONL format per line:
{
  "system_prompt": "...",
  "security_memory_text": "...",
  "context_text": "...",
  "file_rel": "...", (optional)
  ...
}

Run example:
python train_hier_softprompt_pref.py \
  --model_dir /root/Works-about-the-phd-task/Qwen2.5-Coder-1.5B-Instruct-HF \
  --data_jsonl /root/Works-about-the-phd-task/out/train_split.jsonl \
  --out_dir /root/Works-about-the-phd-task/ckpts_hier \
  --bandit_bin bandit \
  --bf16 \
  --steps 200 \
  --num_candidates 4 \
  --max_new_tokens 256
"""

from __future__ import annotations

import os
import re
import json
import time
import math
import ast
import uuid
import random
import shutil
import argparse
import tempfile
import subprocess
import difflib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModelForCausalLM


# -----------------------------
# Utilities
# -----------------------------

SEV_W = {"LOW": 1.0, "MEDIUM": 3.0, "HIGH": 5.0}

def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def read_jsonl(path: str) -> List[Dict[str, Any]]:
    data = []
    with open(path, "r", encoding="utf-8") as rf:
        for line in rf:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data

def ensure_hf_model_dir(model_dir: str):
    p = Path(model_dir).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"--model_dir not found: {p}")
    # GGUF quick reject
    if any(x.suffix.lower() == ".gguf" for x in p.glob("*.gguf")) or str(p).lower().endswith("gguf"):
        raise ValueError(
            f"--model_dir points to a GGUF directory: {p}\n"
            "This training requires a HuggingFace (PyTorch) model folder with config.json + safetensors/bin weights."
        )
    if not (p / "config.json").exists():
        raise FileNotFoundError(
            f"Missing config.json under: {p}\n"
            "Please pass a HF model directory created by `hf download ... --local-dir ...`."
        )
    return str(p)

def extract_code_from_context(context_text: str) -> Optional[str]:
    """
    Extract code from context_text fenced block: ```python ... ```
    """
    # try python fence
    m = re.search(r"```python\s*(.*?)```", context_text, flags=re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip("\n")
    # fallback: any fence
    m = re.search(r"```[\w+-]*\s*(.*?)```", context_text, flags=re.DOTALL)
    if m:
        return m.group(1).strip("\n")
    return None

def extract_unified_diff(text: str) -> Optional[str]:
    """
    Try to extract a unified diff patch from model output.
    """
    # locate first '--- ' line
    idx = text.find("--- ")
    if idx < 0:
        return None
    patch = text[idx:]
    # heuristics: stop before extra commentary if exists
    # but keep as much as possible; our applier ignores unknown trailing lines
    return patch.strip() + "\n"

def apply_unified_diff_to_code(old_code: str, patch_text: str) -> Tuple[bool, str]:
    """
    Apply a unified diff patch to a single-file code string.
    Returns (ok, new_code).
    Only supports one file diff with @@ hunks.
    """
    if not patch_text:
        return (False, old_code)

    old_lines = old_code.splitlines(keepends=True)
    diff_lines = patch_text.splitlines(keepends=False)

    # Find hunks
    hunk_re = re.compile(r"^@@\s*-(\d+)(?:,(\d+))?\s+\+(\d+)(?:,(\d+))?\s*@@")
    i = 0

    # Skip headers until first hunk
    while i < len(diff_lines) and not diff_lines[i].startswith("@@"):
        i += 1
    if i >= len(diff_lines):
        return (False, old_code)

    out_lines: List[str] = []
    old_idx = 0  # 0-based pointer into old_lines

    while i < len(diff_lines):
        line = diff_lines[i]
        if not line.startswith("@@"):
            # allow trailing junk after hunks
            i += 1
            continue

        m = hunk_re.match(line)
        if not m:
            return (False, old_code)
        old_start = int(m.group(1))
        # old_count = int(m.group(2) or "1")
        # new_start = int(m.group(3))
        # new_count = int(m.group(4) or "1")

        # copy unchanged part before hunk
        target_old_idx = max(old_start - 1, 0)
        if target_old_idx < old_idx:
            # overlapping hunks or mismatch
            return (False, old_code)
        out_lines.extend(old_lines[old_idx:target_old_idx])
        old_idx = target_old_idx

        i += 1
        # process hunk body
        while i < len(diff_lines):
            hl = diff_lines[i]
            if hl.startswith("@@"):
                break
            if hl.startswith("\\ No newline at end of file"):
                i += 1
                continue

            if hl.startswith(" "):
                # context line must match
                content = hl[1:] + "\n"
                if old_idx >= len(old_lines) or old_lines[old_idx] != content:
                    # try tolerate different newline handling
                    if old_idx < len(old_lines) and old_lines[old_idx].rstrip("\n") == hl[1:]:
                        out_lines.append(old_lines[old_idx])
                        old_idx += 1
                        i += 1
                        continue
                    return (False, old_code)
                out_lines.append(old_lines[old_idx])
                old_idx += 1
            elif hl.startswith("-"):
                # deletion: must match old
                content = hl[1:] + "\n"
                if old_idx >= len(old_lines) or old_lines[old_idx] != content:
                    if old_idx < len(old_lines) and old_lines[old_idx].rstrip("\n") == hl[1:]:
                        old_idx += 1
                        i += 1
                        continue
                    return (False, old_code)
                old_idx += 1
            elif hl.startswith("+"):
                # addition
                out_lines.append(hl[1:] + "\n")
            else:
                # unknown line, treat as failure
                return (False, old_code)
            i += 1

    # append remaining old lines
    out_lines.extend(old_lines[old_idx:])
    return (True, "".join(out_lines))


def run_bandit_on_code(bandit_bin: str, code: str, filename_hint: str = "temp.py") -> Tuple[float, int]:
    """
    Returns (risk_score, n_issues). risk_score is severity-weighted count.
    """
    if not code.strip():
        return (1e6, 999999)

    tmpdir = tempfile.mkdtemp(prefix="bandit_tmp_")
    try:
        fpath = os.path.join(tmpdir, filename_hint)
        with open(fpath, "w", encoding="utf-8") as wf:
            wf.write(code)

        cmd = [bandit_bin, "-f", "json", "-q", fpath]
        p = subprocess.run(cmd, capture_output=True, text=True)

        out = p.stdout.strip() or p.stderr.strip()
        if not out:
            return (1e6, 999999)

        try:
            obj = json.loads(out)
        except json.JSONDecodeError:
            return (1e6, 999999)

        results = obj.get("results", []) or []
        risk = 0.0
        for r in results:
            sev = (r.get("issue_severity") or "LOW").upper()
            risk += SEV_W.get(sev, 1.0)
        return (risk, len(results))
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def syntax_penalty_py(code: str) -> float:
    try:
        ast.parse(code)
        return 0.0
    except Exception:
        return 3.0  # hard penalty


def edit_drift(old: str, new: str) -> float:
    # 1 - similarity ratio
    ratio = difflib.SequenceMatcher(None, old, new).ratio()
    return max(0.0, 1.0 - ratio)


@torch.no_grad()
def mean_pool_llm_embedding(
    llm: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    text: str,
    device: torch.device,
    max_length: int = 1024,
) -> torch.Tensor:
    """
    Use frozen LLM hidden states as a semantic proxy embedding.
    Returns float32 vector [d].
    """
    enc = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding=False,
    )
    input_ids = enc["input_ids"].to(device)
    attn = enc["attention_mask"].to(device)

    # request hidden states
    out = llm(
        input_ids=input_ids,
        attention_mask=attn,
        output_hidden_states=True,
        use_cache=False,
    )
    hs = out.hidden_states[-1]  # [1, L, d]
    # mean pool with mask
    mask = attn.unsqueeze(-1).to(hs.dtype)  # [1,L,1]
    pooled = (hs * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
    return pooled.squeeze(0).float()


def cos_dist(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float()
    b = b.float()
    denom = (a.norm() * b.norm()).item()
    if denom <= 1e-12:
        return 1.0
    sim = (a @ b).item() / denom
    return float(max(0.0, 1.0 - sim))


# -----------------------------
# Model: cross-attn + prompt
# -----------------------------

class CrossAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.d = d_model
        self.h = n_heads
        self.dk = d_model // n_heads

        self.Wq = nn.Linear(d_model, d_model, bias=False)
        self.Wk = nn.Linear(d_model, d_model, bias=False)
        self.Wv = nn.Linear(d_model, d_model, bias=False)
        self.Wo = nn.Linear(d_model, d_model, bias=False)
        self.ln = nn.LayerNorm(d_model)

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        """
        Q: [B, M, d], K/V: [B, L, d]
        return: [B, M, d]
        """
        B, M, _ = Q.shape
        _, L, _ = K.shape

        # pre-norm (stabilizes training)
        Qn = self.ln(Q)
        Kn = self.ln(K)
        Vn = self.ln(V)

        q = self.Wq(Qn).view(B, M, self.h, self.dk).transpose(1, 2)  # [B,h,M,dk]
        k = self.Wk(Kn).view(B, L, self.h, self.dk).transpose(1, 2)  # [B,h,L,dk]
        v = self.Wv(Vn).view(B, L, self.h, self.dk).transpose(1, 2)  # [B,h,L,dk]

        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.dk)         # [B,h,M,L]
        attn = attn.softmax(dim=-1)
        out = attn @ v                                               # [B,h,M,dk]
        out = out.transpose(1, 2).contiguous().view(B, M, self.d)     # [B,M,d]
        out = self.Wo(out)
        return Q + out  # residual


class HierSoftPrompt(nn.Module):
    def __init__(self, d_model: int, prompt_len: int, n_heads: int):
        super().__init__()
        self.d = d_model
        self.m = prompt_len

        # trainable base prompt tokens P0
        self.P0 = nn.Parameter(torch.randn(1, prompt_len, d_model) * 0.02)

        self.attn_sec = CrossAttention(d_model, n_heads)   # P0 attends to security
        self.attn_ctx = CrossAttention(d_model, n_heads)   # P1 attends to context
        self.out_ln = nn.LayerNorm(d_model)

    def forward(self, S_emb: torch.Tensor, C_emb: torch.Tensor) -> torch.Tensor:
        """
        S_emb: [B, Ls, d]  security memory embeddings
        C_emb: [B, Lc, d]  context memory embeddings
        return: P*: [B, m, d]
        """
        B = S_emb.size(0)
        P = self.P0.expand(B, -1, -1)               # [B,m,d]
        P1 = self.attn_sec(P, S_emb, S_emb)         # security-aligned
        P2 = self.attn_ctx(P1, C_emb, C_emb)        # context-grounded
        return self.out_ln(P2)


# -----------------------------
# Generation with prefix embeddings (custom loop)
# -----------------------------

@torch.no_grad()
def sample_with_prefix(
    llm: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prefix_emb: torch.Tensor,          # [1, m, d]
    input_text: str,
    device: torch.device,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    eos_token_id: int,
) -> str:
    """
    Custom autoregressive sampling that supports prefix embeddings.
    """
    llm.eval()

    enc = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=4096)
    input_ids = enc["input_ids"].to(device)
    attn = enc["attention_mask"].to(device)

    # word embeddings
    emb_layer = llm.get_input_embeddings()
    word_emb = emb_layer(input_ids)  # [1, Lin, d]
    # concat prefix + words
    inputs_embeds = torch.cat([prefix_emb, word_emb], dim=1)  # [1, m+Lin, d]
    attn2 = torch.cat([torch.ones((1, prefix_emb.size(1)), device=device, dtype=attn.dtype), attn], dim=1)

    # first forward
    out = llm(
        inputs_embeds=inputs_embeds,
        attention_mask=attn2,
        use_cache=True,
    )
    past = out.past_key_values
    logits = out.logits[:, -1, :]  # [1, vocab]

    generated: List[int] = []

    for _ in range(max_new_tokens):
        # temperature + nucleus sampling
        if temperature <= 0:
            next_id = int(torch.argmax(logits, dim=-1).item())
        else:
            probs = F.softmax(logits / temperature, dim=-1)  # [1, vocab]
            # top-p
            sorted_probs, sorted_idx = torch.sort(probs, descending=True)
            cum = torch.cumsum(sorted_probs, dim=-1)
            mask = cum > top_p
            mask[..., 0] = False
            sorted_probs = sorted_probs.masked_fill(mask, 0.0)
            sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            sampled = torch.multinomial(sorted_probs, num_samples=1)  # [1,1]
            next_id = int(sorted_idx.gather(-1, sampled).item())

        if next_id == eos_token_id:
            break
        generated.append(next_id)

        out2 = llm(
            input_ids=torch.tensor([[next_id]], device=device, dtype=torch.long),
            use_cache=True,
            past_key_values=past,
        )
        past = out2.past_key_values
        logits = out2.logits[:, -1, :]

    return tokenizer.decode(generated, skip_special_tokens=True)


def compute_logp_with_prefix(
    llm: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prefix_emb: torch.Tensor,     # [1,m,d] requires grad via prompt_mod
    input_text: str,
    target_text: str,
    device: torch.device,
    max_target_tokens: int,
) -> torch.Tensor:
    """
    Compute log p(target_text | input_text, prefix_emb) (sum of token logprobs).
    This is differentiable w.r.t prefix_emb (and thus prompt_mod parameters).
    """
    llm.eval()

    enc_in = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=4096)
    in_ids = enc_in["input_ids"].to(device)        # [1, Lin]
    in_attn = enc_in["attention_mask"].to(device)  # [1, Lin]

    enc_t = tokenizer(
        target_text,
        return_tensors="pt",
        truncation=True,
        max_length=max_target_tokens,
        padding=False,
    )
    t_ids = enc_t["input_ids"].to(device)  # [1, Lt]
    if t_ids.numel() == 0:
        # empty output => very low likelihood
        return torch.tensor(-1e6, device=device)

    # full ids: [input, target]
    full_ids = torch.cat([in_ids, t_ids], dim=1)  # [1, Lin+Lt]
    emb_layer = llm.get_input_embeddings()
    full_emb = emb_layer(full_ids)                # [1, Lin+Lt, d]

    inputs_embeds = torch.cat([prefix_emb, full_emb], dim=1)  # [1, m+Lin+Lt, d]
    attn_full = torch.cat(
        [
            torch.ones((1, prefix_emb.size(1)), device=device, dtype=in_attn.dtype),
            torch.ones_like(full_ids, device=device, dtype=in_attn.dtype),
        ],
        dim=1
    )

    out = llm(inputs_embeds=inputs_embeds, attention_mask=attn_full, use_cache=False)
    logits = out.logits  # [1, m+Lin+Lt, vocab]

    m = prefix_emb.size(1)
    Lin = in_ids.size(1)
    Lt = t_ids.size(1)

    # logits positions that predict target tokens:
    # first target token predicted at position (m+Lin-1)
    start = m + Lin - 1
    end = start + Lt
    logits_t = logits[:, start:end, :]  # [1, Lt, vocab]

    logp = F.log_softmax(logits_t, dim=-1)  # [1,Lt,vocab]
    tgt = t_ids  # [1,Lt]
    token_logp = logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)  # [1,Lt]
    return token_logp.sum(dim=-1).squeeze(0)  # scalar


# -----------------------------
# Training
# -----------------------------

def train(args):
    ensure_hf_model_dir(args.model_dir)
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    # tokenizer + llm
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True, local_files_only=True, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # dtype
    torch_dtype = torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else torch.float32)

    llm = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype=torch_dtype if device.type == "cuda" else torch.float32,
        device_map=None,
    ).to(device)

    llm.eval()
    for p in llm.parameters():
        p.requires_grad_(False)

    model_dtype = next(llm.parameters()).dtype
    model_device = next(llm.parameters()).device

    hidden_size = llm.config.hidden_size

    # prompt module
    prompt_mod = HierSoftPrompt(d_model=hidden_size, prompt_len=args.prompt_len, n_heads=args.n_heads)

    # -------------------------
    # dtype FIX (方案 1)：让 prompt_mod 和 LLM 同 dtype/同 device
    # -------------------------
    prompt_mod = prompt_mod.to(device=model_device, dtype=model_dtype)

    # only optimize prompt-side params
    opt = torch.optim.AdamW(prompt_mod.parameters(), lr=args.lr, weight_decay=args.wd)

    # load data
    data = read_jsonl(args.data_jsonl)
    if len(data) == 0:
        raise ValueError(f"Empty dataset: {args.data_jsonl}")

    print(f"[INFO] loaded {len(data)} samples")
    n_trainable = sum(p.numel() for p in prompt_mod.parameters() if p.requires_grad)
    print(f"[INFO] trainable params (prompt-side): {n_trainable/1e6:.3f}M")
    print(f"[INFO] model dtype={model_dtype} device={model_device} bf16={args.bf16} fp16={args.fp16}")

    # main loop
    t0 = time.time()
    for step in range(1, args.steps + 1):
        rec = random.choice(data)
        system_prompt = rec.get("system_prompt", "").strip()
        sec_text = rec.get("security_memory_text", "").strip()
        ctx_text = rec.get("context_text", "").strip()
        file_rel = rec.get("file_rel", "temp.py")

        # build separate memories
        # S: security memory text
        # C: context text (task + code)
        sec_enc = tokenizer(sec_text, return_tensors="pt", truncation=True, max_length=args.max_mem_tokens)
        ctx_enc = tokenizer(ctx_text, return_tensors="pt", truncation=True, max_length=args.max_ctx_tokens)

        sec_ids = sec_enc["input_ids"].to(model_device)
        ctx_ids = ctx_enc["input_ids"].to(model_device)

        # frozen word embeddings as memory embeddings
        with torch.no_grad():
            emb_layer = llm.get_input_embeddings()
            S_emb = emb_layer(sec_ids).to(dtype=model_dtype)  # [1,Ls,d]
            C_emb = emb_layer(ctx_ids).to(dtype=model_dtype)  # [1,Lc,d]
            S_emb = S_emb.detach()
            C_emb = C_emb.detach()

        # produce grounded prompt embeddings
        P_star = prompt_mod(S_emb, C_emb)  # [1,m,d], dtype == model_dtype

        # build LLM input text (word tokens): system + security + context
        input_text = ""
        if system_prompt:
            input_text += system_prompt.strip() + "\n\n"
        if sec_text:
            input_text += sec_text.strip() + "\n\n"
        input_text += ctx_text.strip()

        # extract original code (for scoring)
        orig_code = extract_code_from_context(ctx_text) or ""
        if not orig_code.strip():
            # if cannot extract, skip this sample
            continue

        # semantic embedding for original (cached per-step)
        with torch.no_grad():
            try:
                orig_emb = mean_pool_llm_embedding(llm, tokenizer, orig_code, model_device, max_length=args.embed_max_len)
            except Exception:
                orig_emb = None

        # generate candidates (no grad)
        cand_texts: List[str] = []
        for _ in range(args.num_candidates):
            txt = sample_with_prefix(
                llm=llm,
                tokenizer=tokenizer,
                prefix_emb=P_star,
                input_text=input_text,
                device=model_device,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                eos_token_id=tokenizer.eos_token_id,
            )
            cand_texts.append(txt)

        # score candidates
        scored: List[Tuple[float, float, float, str, str]] = []
        # (utility, risk, drift, patch_text, new_code)
        for cand in cand_texts:
            patch = extract_unified_diff(cand)
            if patch is None:
                # not a patch -> heavy penalty
                utility = -1e6
                scored.append((utility, 1e6, 1e3, "", ""))
                continue

            ok, new_code = apply_unified_diff_to_code(orig_code, patch)
            if not ok or not new_code.strip():
                utility = -1e6
                scored.append((utility, 1e6, 1e3, patch, ""))
                continue

            # bandit risk
            risk, _n = run_bandit_on_code(args.bandit_bin, new_code, filename_hint=Path(file_rel).name)

            # functionality drift
            syn_pen = syntax_penalty_py(new_code) if args.language.lower() == "python" else 0.0
            ed = edit_drift(orig_code, new_code)

            sem = 0.0
            if orig_emb is not None:
                try:
                    with torch.no_grad():
                        new_emb = mean_pool_llm_embedding(llm, tokenizer, new_code, model_device, max_length=args.embed_max_len)
                    sem = cos_dist(orig_emb, new_emb)  # (1-cos)
                except Exception:
                    sem = 0.5

            drift = args.w_sem * sem + args.w_edit * ed + syn_pen

            utility = -risk - args.lambda_func * drift
            scored.append((utility, risk, drift, patch, new_code))

        # pick best/worst by utility
        scored.sort(key=lambda x: x[0], reverse=True)
        best = scored[0]
        worst = scored[-1]
        y_plus = best[3]   # patch text
        y_minus = worst[3]

        # if plus/minus invalid, skip
        if not y_plus.strip() or not y_minus.strip():
            continue
        if y_plus == y_minus:
            continue

        # compute DPO-like loss with grads (only prompt_mod)
        opt.zero_grad(set_to_none=True)

        logp_plus = compute_logp_with_prefix(
            llm=llm,
            tokenizer=tokenizer,
            prefix_emb=P_star,
            input_text=input_text,
            target_text=y_plus,
            device=model_device,
            max_target_tokens=args.max_target_tokens,
        )
        logp_minus = compute_logp_with_prefix(
            llm=llm,
            tokenizer=tokenizer,
            prefix_emb=P_star,
            input_text=input_text,
            target_text=y_minus,
            device=model_device,
            max_target_tokens=args.max_target_tokens,
        )

        # loss = -log sigmoid(beta * (logp_plus - logp_minus))
        diff = logp_plus - logp_minus
        loss = -F.logsigmoid(args.beta_dpo * diff)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(prompt_mod.parameters(), args.grad_clip)
        opt.step()

        if step % args.log_every == 0:
            elapsed = time.time() - t0
            print(
                f"[step {step:04d}/{args.steps}] "
                f"loss={loss.item():.4f} "
                f"logp+={logp_plus.item():.2f} logp-={logp_minus.item():.2f} "
                f"U+={best[0]:.2f} risk+={best[1]:.2f} drift+={best[2]:.2f} "
                f"U-={worst[0]:.2f} "
                f"elapsed={elapsed:.1f}s"
            )

        if args.save_every > 0 and step % args.save_every == 0:
            ckpt = {
                "step": step,
                "prompt_mod": prompt_mod.state_dict(),
                "args": vars(args),
            }
            ckpt_path = os.path.join(args.out_dir, f"prompt_mod_step{step}.pt")
            torch.save(ckpt, ckpt_path)
            print(f"[CKPT] saved: {ckpt_path}")

    # final save
    final_path = os.path.join(args.out_dir, "prompt_mod_final.pt")
    torch.save({"step": args.steps, "prompt_mod": prompt_mod.state_dict(), "args": vars(args)}, final_path)
    print(f"[DONE] saved final: {final_path}")


def build_argparser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", type=str, required=True, help="HF model directory (must contain config.json + safetensors/bin).")
    ap.add_argument("--data_jsonl", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--bandit_bin", type=str, default="bandit")

    ap.add_argument("--language", type=str, default="python")

    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--prompt_len", type=int, default=32)
    ap.add_argument("--n_heads", type=int, default=8)

    ap.add_argument("--num_candidates", type=int, default=4)
    ap.add_argument("--max_new_tokens", type=int, default=256)

    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.9)

    ap.add_argument("--beta_dpo", type=float, default=0.1)
    ap.add_argument("--lambda_func", type=float, default=1.0)

    ap.add_argument("--w_sem", type=float, default=1.0)
    ap.add_argument("--w_edit", type=float, default=1.0)
    ap.add_argument("--embed_max_len", type=int, default=768)

    ap.add_argument("--max_mem_tokens", type=int, default=256)
    ap.add_argument("--max_ctx_tokens", type=int, default=2048)
    ap.add_argument("--max_target_tokens", type=int, default=512)

    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--log_every", type=int, default=10)
    ap.add_argument("--save_every", type=int, default=50)

    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--cpu", action="store_true")

    return ap


def main():
    args = build_argparser().parse_args()
    train(args)


if __name__ == "__main__":
    main()
