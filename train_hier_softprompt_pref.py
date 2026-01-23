#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import os
import re
import json
import time
import math
import ast
import random
import shutil
import argparse
import tempfile
import subprocess
import difflib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM


# -----------------------------
# Heartbeat stats
# -----------------------------

@dataclass
class TrainStats:
    attempted: int = 0
    updates: int = 0

    skip_empty_code: int = 0
    skip_bad_gen: int = 0          # code extract / truncation / nonsense
    skip_apply_fail: int = 0       # (kept for compatibility; rarely used now)
    skip_bandit_fail: int = 0
    skip_no_preference: int = 0
    skip_other: int = 0

    last_loss: Optional[float] = None
    last_risk_in: Optional[float] = None
    last_risk_best: Optional[float] = None
    last_drift_best: Optional[float] = None

    t0: float = field(default_factory=time.time)
    last_step_secs: Optional[float] = None

    metrics_jsonl: Optional[str] = None

    def bump(self, name: str, n: int = 1):
        if hasattr(self, name):
            setattr(self, name, getattr(self, name) + n)
        else:
            self.skip_other += n

    def set_last_update(self, loss: float, risk_in: float, risk_best: float, drift_best: float):
        self.last_loss = float(loss)
        self.last_risk_in = float(risk_in)
        self.last_risk_best = float(risk_best)
        self.last_drift_best = float(drift_best)

    def summary_str(self) -> str:
        skips = (
            f"empty_code={self.skip_empty_code},"
            f"bad_gen={self.skip_bad_gen},"
            f"apply_fail={self.skip_apply_fail},"
            f"bandit_fail={self.skip_bandit_fail},"
            f"no_pref={self.skip_no_preference},"
            f"other={self.skip_other}"
        )
        last = "none" if self.last_loss is None else (
            f"loss={self.last_loss:.4f},rin={self.last_risk_in:.3f},"
            f"rbest={self.last_risk_best:.3f},drift={self.last_drift_best:.3f}"
        )
        elapsed = time.time() - self.t0
        sps = self.attempted / max(1e-9, elapsed)
        dt = "?" if self.last_step_secs is None else f"{self.last_step_secs:.2f}s"
        return (f"attempted={self.attempted} updates={self.updates} "
                f"skips[{skips}] last[{last}] step_time={dt} {sps:.2f} steps/s")

    def maybe_heartbeat(self, step: int, total_steps: int, every: int, flush: bool = False):
        if every <= 0:
            return
        if (self.attempted % every) != 0:
            return
        msg = f"[HB] step={step}/{total_steps} " + self.summary_str()
        print(msg)
        if flush:
            try:
                import sys
                sys.stdout.flush()
            except Exception:
                pass
        if self.metrics_jsonl:
            rec = {
                "step": step,
                "total_steps": total_steps,
                "attempted": self.attempted,
                "updates": self.updates,
                "skip_empty_code": self.skip_empty_code,
                "skip_bad_gen": self.skip_bad_gen,
                "skip_apply_fail": self.skip_apply_fail,
                "skip_bandit_fail": self.skip_bandit_fail,
                "skip_no_preference": self.skip_no_preference,
                "skip_other": self.skip_other,
                "last_loss": self.last_loss,
                "last_risk_in": self.last_risk_in,
                "last_risk_best": self.last_risk_best,
                "last_drift_best": self.last_drift_best,
                "last_step_secs": self.last_step_secs,
                "time": time.time(),
            }
            d = os.path.dirname(self.metrics_jsonl)
            if d:
                os.makedirs(d, exist_ok=True)
            with open(self.metrics_jsonl, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")


# -----------------------------
# Utilities
# -----------------------------

SEV_W = {"LOW": 1.0, "MEDIUM": 3.0, "HIGH": 5.0}

CODE_ONLY_RULES = (
    "You are given a Python file. Produce the COMPLETE updated file content.\n"
    "Requirements:\n"
    "- Output ONLY the updated file content (no explanation, no markdown fences).\n"
    "- Preserve ALL unchanged lines exactly.\n"
    "- Apply minimal changes needed to reduce security risk (e.g., avoid SQL injection).\n"
    "- Do NOT delete logic to game the analyzer.\n"
)

def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def read_jsonl(path: str) -> List[Dict[str, Any]]:
    data: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as rf:
        for line in rf:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data

def ensure_hf_model_dir(model_dir: str) -> str:
    p = Path(model_dir).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"--model_dir not found: {p}")
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
    m = re.search(r"```python\s*(.*?)```", context_text, flags=re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip("\n")
    m = re.search(r"```[\w+-]*\s*(.*?)```", context_text, flags=re.DOTALL)
    if m:
        return m.group(1).strip("\n")
    return None

def extract_full_code_from_model_output(text: str) -> Optional[str]:
    """
    Prefer fenced code; otherwise, drop leading prose until a likely code line.
    """
    if not text:
        return None
    t = text.replace("\r\n", "\n").replace("\r", "\n")

    # strip markdown fences if any
    m = re.search(r"```(?:python)?\s*(.*?)```", t, flags=re.DOTALL | re.IGNORECASE)
    if m:
        code = m.group(1).strip("\n")
        return code if code.strip() else None

    lines = t.splitlines()
    start = None
    for i, ln in enumerate(lines):
        if re.match(r"^\s*(#!/|from\s+|import\s+|def\s+|class\s+|#)", ln):
            start = i
            break
    if start is None:
        code = t.strip("\n")
        return code if code.strip() else None

    code = "\n".join(lines[start:]).strip("\n")
    return code if code.strip() else None

def compute_unified_diff(old: str, new: str, rel_path: str) -> str:
    rel_path = (rel_path or "temp.py").lstrip("./") or "temp.py"
    a = old.splitlines(keepends=True)
    b = new.splitlines(keepends=True)
    diff = difflib.unified_diff(
        a, b,
        fromfile=f"a/{rel_path}",
        tofile=f"b/{rel_path}",
        lineterm=""
    )
    return "\n".join(diff).rstrip() + "\n"

def run_bandit_on_code(bandit_bin: str, code: str, filename_hint: str = "temp.py") -> Tuple[float, int, bool]:
    if not code.strip():
        return (1e6, 999999, False)
    tmpdir = tempfile.mkdtemp(prefix="bandit_tmp_")
    try:
        fpath = os.path.join(tmpdir, filename_hint)
        with open(fpath, "w", encoding="utf-8") as wf:
            wf.write(code)

        cmd = [bandit_bin, "-f", "json", "-q", fpath]
        p = subprocess.run(cmd, capture_output=True, text=True)

        out = (p.stdout or "").strip() or (p.stderr or "").strip()
        if not out:
            return (1e6, 999999, False)

        try:
            obj = json.loads(out)
        except json.JSONDecodeError:
            return (1e6, 999999, False)

        results = obj.get("results", []) or []
        risk = 0.0
        for r in results:
            sev = (r.get("issue_severity") or "LOW").upper()
            risk += SEV_W.get(sev, 1.0)
        return (risk, len(results), True)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

def syntax_penalty_py(code: str) -> float:
    try:
        ast.parse(code)
        return 0.0
    except Exception:
        return 3.0

def edit_drift(old: str, new: str) -> float:
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
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length, padding=False)
    input_ids = enc["input_ids"].to(device)
    attn = enc["attention_mask"].to(device)
    out = llm(input_ids=input_ids, attention_mask=attn, output_hidden_states=True, use_cache=False)
    hs = out.hidden_states[-1]
    mask = attn.unsqueeze(-1).to(hs.dtype)
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

def dump_debug_fail(
    out_dir: str,
    step: int,
    file_rel: str,
    orig_code: str,
    input_text: str,
    raw_out: str,
    extracted_code: str,
    auto_diff: str,
    meta: Dict[str, Any],
    keep_first_n: int = 200,
):
    d = Path(out_dir) / "debug_fail"
    d.mkdir(parents=True, exist_ok=True)
    existing = len(list(d.glob("step*_meta.json")))
    if existing >= keep_first_n:
        return

    base = Path(file_rel).name
    (d / f"step{step:04d}_{base}.orig.py").write_text(orig_code, encoding="utf-8", errors="ignore")
    (d / f"step{step:04d}_{base}.input.txt").write_text(input_text or "", encoding="utf-8", errors="ignore")
    (d / f"step{step:04d}_{base}.raw.out.txt").write_text(raw_out or "", encoding="utf-8", errors="ignore")
    (d / f"step{step:04d}_{base}.extracted.py").write_text(extracted_code or "", encoding="utf-8", errors="ignore")
    (d / f"step{step:04d}_{base}.auto.diff").write_text(auto_diff or "", encoding="utf-8", errors="ignore")
    meta2 = dict(meta)
    meta2.update({
        "step": step,
        "file_rel": file_rel,
        "time": time.time(),
        "auto_diff_lines": len((auto_diff or "").splitlines()),
        "extracted_lines": len((extracted_code or "").splitlines()),
    })
    (d / f"step{step:04d}_{base}.meta.json").write_text(
        json.dumps(meta2, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


# -----------------------------
# Model
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
        B, M, _ = Q.shape
        _, L, _ = K.shape

        Qn = self.ln(Q)
        Kn = self.ln(K)
        Vn = self.ln(V)

        q = self.Wq(Qn).view(B, M, self.h, self.dk).transpose(1, 2)
        k = self.Wk(Kn).view(B, L, self.h, self.dk).transpose(1, 2)
        v = self.Wv(Vn).view(B, L, self.h, self.dk).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.dk)
        attn = attn.softmax(dim=-1)
        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(B, M, self.d)
        out = self.Wo(out)
        return Q + out

class HierSoftPrompt(nn.Module):
    def __init__(self, d_model: int, prompt_len: int, n_heads: int):
        super().__init__()
        self.P0 = nn.Parameter(torch.randn(1, prompt_len, d_model) * 0.02)
        self.attn_sec = CrossAttention(d_model, n_heads)
        self.attn_ctx = CrossAttention(d_model, n_heads)
        self.out_ln = nn.LayerNorm(d_model)

    def forward(self, S_emb: torch.Tensor, C_emb: torch.Tensor) -> torch.Tensor:
        B = S_emb.size(0)
        P = self.P0.expand(B, -1, -1)
        P1 = self.attn_sec(P, S_emb, S_emb)
        P2 = self.attn_ctx(P1, C_emb, C_emb)
        return self.out_ln(P2)


# -----------------------------
# Generation with prefix
# -----------------------------

@torch.no_grad()
def sample_with_prefix(
    llm: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prefix_emb: torch.Tensor,
    input_text: str,
    device: torch.device,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    eos_token_id: int,
    max_input_tokens: int,
) -> str:
    llm.eval()

    enc = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=max_input_tokens)
    input_ids = enc["input_ids"].to(device)
    attn = enc["attention_mask"].to(device)

    emb_layer = llm.get_input_embeddings()
    word_emb = emb_layer(input_ids)
    inputs_embeds = torch.cat([prefix_emb, word_emb], dim=1)
    attn2 = torch.cat([torch.ones((1, prefix_emb.size(1)), device=device, dtype=attn.dtype), attn], dim=1)

    out = llm(inputs_embeds=inputs_embeds, attention_mask=attn2, use_cache=True)
    past = out.past_key_values
    logits = out.logits[:, -1, :]

    generated: List[int] = []
    for _ in range(max_new_tokens):
        if temperature <= 0:
            next_id = int(torch.argmax(logits, dim=-1).item())
        else:
            probs = F.softmax(logits / temperature, dim=-1)
            sorted_probs, sorted_idx = torch.sort(probs, descending=True)
            cum = torch.cumsum(sorted_probs, dim=-1)
            mask = cum > top_p
            mask[..., 0] = False
            sorted_probs = sorted_probs.masked_fill(mask, 0.0)
            sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            sampled = torch.multinomial(sorted_probs, num_samples=1)
            next_id = int(sorted_idx.gather(-1, sampled).item())

        if eos_token_id is not None and next_id == eos_token_id:
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
    prefix_emb: torch.Tensor,
    input_text: str,
    target_text: str,
    device: torch.device,
    max_input_tokens: int,
    max_target_tokens: int,
) -> torch.Tensor:
    llm.eval()

    enc_in = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=max_input_tokens)
    in_ids = enc_in["input_ids"].to(device)
    in_attn = enc_in["attention_mask"].to(device)

    enc_t = tokenizer(target_text, return_tensors="pt", truncation=True, max_length=max_target_tokens, padding=False)
    t_ids = enc_t["input_ids"].to(device)
    if t_ids.numel() == 0:
        return torch.tensor(-1e6, device=device)

    full_ids = torch.cat([in_ids, t_ids], dim=1)
    emb_layer = llm.get_input_embeddings()
    full_emb = emb_layer(full_ids)

    inputs_embeds = torch.cat([prefix_emb, full_emb], dim=1)
    attn_full = torch.cat(
        [
            torch.ones((1, prefix_emb.size(1)), device=device, dtype=in_attn.dtype),
            torch.ones_like(full_ids, device=device, dtype=in_attn.dtype),
        ],
        dim=1,
    )

    out = llm(inputs_embeds=inputs_embeds, attention_mask=attn_full, use_cache=False)
    logits = out.logits

    m = prefix_emb.size(1)
    Lin = in_ids.size(1)
    Lt = t_ids.size(1)

    start = m + Lin - 1
    end = start + Lt
    logits_t = logits[:, start:end, :]

    logp = F.log_softmax(logits_t, dim=-1)
    token_logp = logp.gather(-1, t_ids.unsqueeze(-1)).squeeze(-1)
    return token_logp.sum(dim=-1).squeeze(0)


# -----------------------------
# Training
# -----------------------------

def train(args):
    model_dir = ensure_hf_model_dir(args.model_dir)
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True, local_files_only=True, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    torch_dtype = torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else torch.float32)

    llm = AutoModelForCausalLM.from_pretrained(
        model_dir,
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

    prompt_mod = HierSoftPrompt(d_model=hidden_size, prompt_len=args.prompt_len, n_heads=args.n_heads)
    prompt_mod = prompt_mod.to(device=model_device, dtype=model_dtype)

    opt = torch.optim.AdamW(prompt_mod.parameters(), lr=args.lr, weight_decay=args.wd)
    stats = TrainStats(metrics_jsonl=args.metrics_jsonl)

    data = read_jsonl(args.data_jsonl)
    if not data:
        raise ValueError(f"Empty dataset: {args.data_jsonl}")

    print(f"[INFO] loaded {len(data)} samples")
    n_trainable = sum(p.numel() for p in prompt_mod.parameters() if p.requires_grad)
    print(f"[INFO] trainable params (prompt-side): {n_trainable/1e6:.3f}M")
    print(f"[INFO] model dtype={model_dtype} device={model_device} bf16={args.bf16} fp16={args.fp16}")

    for step in range(1, args.steps + 1):
        step_t0 = time.time()
        stats.attempted += 1

        try:
            rec = random.choice(data)
            system_prompt = (rec.get("system_prompt") or "").strip()
            sec_text = (rec.get("security_memory_text") or "").strip()
            ctx_text = (rec.get("context_text") or "").strip()
            file_rel = rec.get("file_rel") or rec.get("file_name") or "temp.py"

            orig_code = extract_code_from_context(ctx_text) or ""
            if not orig_code.strip():
                stats.bump("skip_empty_code")
                continue

            # bandit baseline
            rin, _, ok_in = run_bandit_on_code(args.bandit_bin, orig_code, filename_hint=Path(file_rel).name)
            if not ok_in:
                rin = float(rec.get("meta", {}).get("risk_in", rin))

            # embed security/context tokens (frozen)
            sec_enc = tokenizer(sec_text, return_tensors="pt", truncation=True, max_length=args.max_mem_tokens)
            ctx_enc = tokenizer(ctx_text, return_tensors="pt", truncation=True, max_length=args.max_ctx_tokens)
            sec_ids = sec_enc["input_ids"].to(model_device)
            ctx_ids = ctx_enc["input_ids"].to(model_device)

            emb_layer = llm.get_input_embeddings()
            with torch.no_grad():
                S_emb = emb_layer(sec_ids).to(dtype=model_dtype).detach()
                C_emb = emb_layer(ctx_ids).to(dtype=model_dtype).detach()

            P_star = prompt_mod(S_emb, C_emb)

            # Build generation input
            input_text = CODE_ONLY_RULES + "\n"
            if system_prompt:
                input_text += system_prompt + "\n\n"
            if sec_text:
                input_text += sec_text + "\n\n"
            input_text += ctx_text

            orig_emb = None
            if args.w_sem > 0:
                try:
                    orig_emb = mean_pool_llm_embedding(llm, tokenizer, orig_code, model_device, max_length=args.embed_max_len)
                except Exception:
                    orig_emb = None

            # generate candidates
            candidates: List[Dict[str, Any]] = []
            raw_out_list: List[str] = []

            for k in range(args.num_candidates):
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
                    max_input_tokens=args.max_input_tokens,
                )
                raw_out_list.append(txt)

                new_code = extract_full_code_from_model_output(txt)
                if not new_code or not new_code.strip():
                    continue

                # evaluate
                risk, _, ok_b = run_bandit_on_code(args.bandit_bin, new_code, filename_hint=Path(file_rel).name)
                if not ok_b:
                    continue

                syn_pen = syntax_penalty_py(new_code) if args.language.lower() == "python" else 0.0
                ed = edit_drift(orig_code, new_code)

                sem = 0.0
                if args.w_sem > 0 and orig_emb is not None:
                    try:
                        new_emb = mean_pool_llm_embedding(llm, tokenizer, new_code, model_device, max_length=args.embed_max_len)
                        sem = cos_dist(orig_emb, new_emb)
                    except Exception:
                        sem = 0.5

                drift = args.w_sem * sem + args.w_edit * ed + args.w_syntax * syn_pen
                utility = -risk - args.lambda_func * drift

                candidates.append({
                    "utility": float(utility),
                    "risk": float(risk),
                    "drift": float(drift),
                    "code": new_code,
                    "auto_diff": compute_unified_diff(orig_code, new_code, file_rel),
                    "kind": "model",
                })

            # If no valid candidate, dump debug and skip
            if len(candidates) == 0:
                stats.bump("skip_bad_gen")
                if args.debug_keep_first_n > 0:
                    dump_debug_fail(
                        out_dir=args.out_dir,
                        step=step,
                        file_rel=file_rel,
                        orig_code=orig_code,
                        input_text=input_text[:4000],
                        raw_out="\n\n=====CANDIDATE=====\n\n".join(raw_out_list[: min(3, len(raw_out_list))]),
                        extracted_code="",
                        auto_diff="",
                        meta={"reason": "no_valid_candidate"},
                        keep_first_n=args.debug_keep_first_n,
                    )
                continue

            # Always include baseline (orig) to avoid no_pref explosion
            base_drift = 0.0
            base_utility = -rin - args.lambda_func * base_drift
            candidates.append({
                "utility": float(base_utility),
                "risk": float(rin),
                "drift": float(base_drift),
                "code": orig_code,
                "auto_diff": "",   # no-op
                "kind": "baseline",
            })

            candidates.sort(key=lambda x: x["utility"], reverse=True)
            best = candidates[0]
            worst = candidates[-1]

            # If utility gap too small, skip (optional)
            if (best["utility"] - worst["utility"]) < args.min_utility_gap:
                stats.bump("skip_no_preference")
                continue

            y_plus = best["code"]
            y_minus = worst["code"]
            if (not y_plus.strip()) or (not y_minus.strip()) or (y_plus == y_minus):
                stats.bump("skip_no_preference")
                continue

            opt.zero_grad(set_to_none=True)

            logp_plus = compute_logp_with_prefix(
                llm=llm,
                tokenizer=tokenizer,
                prefix_emb=P_star,
                input_text=input_text,
                target_text=y_plus,
                device=model_device,
                max_input_tokens=args.max_input_tokens,
                max_target_tokens=args.max_target_tokens,
            )
            logp_minus = compute_logp_with_prefix(
                llm=llm,
                tokenizer=tokenizer,
                prefix_emb=P_star,
                input_text=input_text,
                target_text=y_minus,
                device=model_device,
                max_input_tokens=args.max_input_tokens,
                max_target_tokens=args.max_target_tokens,
            )

            diff = logp_plus - logp_minus
            loss = -F.logsigmoid(args.beta_dpo * diff)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(prompt_mod.parameters(), args.grad_clip)
            opt.step()

            stats.updates += 1
            stats.set_last_update(loss.item(), rin, best["risk"], best["drift"])

            if args.log_every > 0 and (step % args.log_every == 0):
                print(
                    f"[step {step:04d}/{args.steps}] "
                    f"loss={loss.item():.4f} "
                    f"logp+={logp_plus.item():.2f} logp-={logp_minus.item():.2f} "
                    f"U+={best['utility']:.2f} (kind={best['kind']}) rin={rin:.2f} "
                    f"risk+={best['risk']:.2f} drift+={best['drift']:.2f} "
                    f"U-={worst['utility']:.2f} (kind={worst['kind']})"
                )
                if args.flush_stdout:
                    try:
                        import sys
                        sys.stdout.flush()
                    except Exception:
                        pass

            if args.save_every > 0 and (step % args.save_every == 0):
                ckpt = {"step": step, "prompt_mod": prompt_mod.state_dict(), "args": vars(args)}
                ckpt_path = os.path.join(args.out_dir, f"prompt_mod_step{step}.pt")
                torch.save(ckpt, ckpt_path)
                print(f"[CKPT] saved: {ckpt_path}")

        finally:
            stats.last_step_secs = time.time() - step_t0
            stats.maybe_heartbeat(step=step, total_steps=args.steps, every=args.heartbeat_every, flush=args.flush_stdout)

    final_path = os.path.join(args.out_dir, "prompt_mod_final.pt")
    torch.save({"step": args.steps, "prompt_mod": prompt_mod.state_dict(), "args": vars(args)}, final_path)
    print(f"[DONE] saved final: {final_path}")


# -----------------------------
# Args
# -----------------------------

def build_argparser():
    ap = argparse.ArgumentParser()

    ap.add_argument("--model_dir", type=str, required=True)
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

    ap.add_argument("--num_candidates", type=int, default=8)
    ap.add_argument("--max_new_tokens", type=int, default=768)

    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.9)

    ap.add_argument("--beta_dpo", type=float, default=0.1)
    ap.add_argument("--lambda_func", type=float, default=1.0)

    ap.add_argument("--w_sem", type=float, default=1.0)
    ap.add_argument("--w_edit", type=float, default=1.0)
    ap.add_argument("--w_syntax", type=float, default=1.0)
    ap.add_argument("--embed_max_len", type=int, default=768)

    ap.add_argument("--max_mem_tokens", type=int, default=512)
    ap.add_argument("--max_ctx_tokens", type=int, default=2048)
    ap.add_argument("--max_input_tokens", type=int, default=2048)     # NEW: bound prompt+ctx
    ap.add_argument("--max_target_tokens", type=int, default=768)     # NEW: bound full-file target

    ap.add_argument("--min_utility_gap", type=float, default=0.0)     # NEW: avoid tiny-diff preference

    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--log_every", type=int, default=10)
    ap.add_argument("--save_every", type=int, default=50)

    ap.add_argument("--heartbeat_every", type=int, default=1)
    ap.add_argument("--metrics_jsonl", type=str, default=None)
    ap.add_argument("--flush_stdout", action="store_true")

    ap.add_argument("--debug_keep_first_n", type=int, default=200)    # NEW: always dump early failures

    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--cpu", action="store_true")

    return ap

def main():
    args = build_argparser().parse_args()
    train(args)

if __name__ == "__main__":
    main()
