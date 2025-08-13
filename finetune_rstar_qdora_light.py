#!/usr/bin/env python3
# Lightweight SFT for Qwen2.5-Coder-7B-Instruct on rStar-Coder
# - Streaming dataset: no huge local downloads
# - QLoRA (bnb 4-bit) + LoRA+ (decoupled LR for A/B)
# - Defaults optimized for a single 48 GB GPU

import os
import re
import time
import random
import argparse
import logging
from typing import List, Dict, Optional, Tuple, Iterable

import torch
import torch.nn as nn

from datasets import load_dataset, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, get_peft_model

# -------------------------
# Logging & small helpers
# -------------------------
def setup_logging(verbosity=logging.INFO):
    fmt = "[%(asctime)s] %(levelname)s %(name)s: %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    logging.basicConfig(level=verbosity, format=fmt, datefmt=datefmt)
    for noisy in ("urllib3", "filelock", "fsspec", "huggingface_hub", "datasets"):
        logging.getLogger(noisy).setLevel(max(verbosity, logging.WARNING))
    logging.getLogger("transformers").setLevel(verbosity)

log = logging.getLogger("qdora_loraplus_light")

def count_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# -------------------------
# CLI
# -------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Low-resource SFT on rStar-Coder with QLoRA + LoRA+ (streaming).")

    # Model / IO
    p.add_argument("--model", default="Qwen/Qwen2.5-Coder-7B-Instruct", type=str)
    p.add_argument("--output", default="./qwen25coder7b-rstar-qdora-light", type=str)
    p.add_argument("--cache-dir", default=None, type=str)
    p.add_argument("--seed", default=42, type=int)
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--flash_attn", action="store_true", help="If set, force attn_implementation=flash_attention_2")
    p.add_argument("--trust_remote_code", action="store_true")
    p.add_argument("--compile", action="store_true", default=False)

    # Data (streaming)
    p.add_argument("--data-root", default="microsoft/rStar-Coder", type=str)
    p.add_argument("--sft-subsets", default="synthetic_sft,seed_sft", type=str,
                   help="Comma list of subset names (streaming ignores :index).")
    p.add_argument("--limit", default=4000, type=int, help="Total number of examples to pull (across subsets).")
    p.add_argument("--per-subset-limit", default=0, type=int,
                   help="Optional hard cap per subset; 0 = auto (limit/num_subsets).")
    p.add_argument("--min_chars", default=200, type=int)
    p.add_argument("--max_chars", default=32000, type=int)
    p.add_argument("--eval_holdout", default=400, type=int, help="Eval size taken from the top after shuffle.")
    p.add_argument("--shuffle_buffer", default=20000, type=int, help="Buffer to locally shuffle streaming samples.")

    # Tokenization & formatting
    p.add_argument("--max-seq-len", default=2048, type=int)
    p.add_argument("--pack", action="store_true", default=False, help="Enable only if you have Flash-Attn v2.")
    p.add_argument("--add_eos", action="store_true", default=True)

    # Optim
    p.add_argument("--per-device-train-batch-size", default=1, type=int)
    p.add_argument("--gradient-accumulation-steps", default=8, type=int)
    p.add_argument("--lr", default=1.5e-4, type=float)          # slightly lower by default
    p.add_argument("--lr_b_scale", default=8.0, type=float)      # LoRA+ B-matrix LR multiplier
    p.add_argument("--epochs", default=1, type=int)
    p.add_argument("--steps", default=1500, type=int)            # fewer steps by default
    p.add_argument("--warmup_ratio", default=0.03, type=float)
    p.add_argument("--save_steps", default=500, type=int)
    p.add_argument("--logging_steps", default=10, type=int)
    p.add_argument("--max_grad_norm", default=1.0, type=float)

    # LoRA/QDoRA
    p.add_argument("--target_modules", default="q_proj,v_proj,o_proj", type=str,
                   help="Fewer modules for smaller footprint; add k_proj,gate_proj,up_proj,down_proj if you can.")
    p.add_argument("--lora-r", default=16, type=int)
    p.add_argument("--lora-alpha", default=32, type=int)
    p.add_argument("--lora-dropout", default=0.05, type=float)
    p.add_argument("--gradient_checkpointing", action="store_true", default=True)

    # Dataloader workers
    p.add_argument("--dataloader_num_workers", default=1, type=int)

    # Verbosity
    p.add_argument("--verbosity", type=str, default="INFO", choices=["DEBUG","INFO","WARNING","ERROR"])
    return p.parse_args()

# -------------------------
# Data (streaming) helpers
# -------------------------
PROMPT_SYS = "You are an elite competitive programmer. Solve step-by-step, then output final, efficient code."
PROMPT_TMPL = """<|system|>
{sys}
</s>
<|user|>
{question}
</s>
<|assistant|>
{reasoning}

python
{code}
"""

RE_WS = re.compile(r"\s+")

def _string_len_ok(x: Optional[str], mn: int, mx: int) -> bool:
    if not x: return False
    n = len(x)
    return (n >= mn) and (n <= mx)

def _quality_ok(ex: Dict, args) -> bool:
    ok = True
    for k in ("verified", "is_passed"):
        if k in ex and ex[k] is not None:
            v = ex[k]
            if isinstance(v, list): v = any(bool(x) for x in v)
            try: v = bool(v)
            except Exception: v = False
            if not v: ok = False
    q = ex.get("question") or ex.get("seed_question") or ""
    reasoning = (ex.get("response") or "").strip()
    code = (ex.get("code") or "").strip()
    body = f"{reasoning}\n{code}".strip()
    if not _string_len_ok(q, 16, args.max_chars): ok = False
    if not _string_len_ok(body, args.min_chars, args.max_chars): ok = False
    return ok

def _format_text(ex: Dict) -> str:
    q = ex.get("question") or ex.get("seed_question") or ""
    reasoning = (ex.get("response") or "").strip()
    code = (ex.get("code") or "").strip().replace("\t", "    ")
    return PROMPT_TMPL.format(sys=PROMPT_SYS, question=q.strip(), reasoning=reasoning, code=code)

def _stream_subset(root: str, subset: str, cache_dir: Optional[str]) -> Iterable[Dict]:
    # streaming=True avoids downloading the full dataset to disk
    return load_dataset(root, subset, split="train", streaming=True, cache_dir=cache_dir)

def build_streaming_dataset(args) -> Tuple[Dataset, Dataset]:
    random.seed(args.seed)

    subsets = [s.strip() for s in args.sft_subsets.split(",") if s.strip()]
    per_subset_limit = args.per_subset_limit if args.per_subset_limit > 0 else max(1, args.limit // max(1, len(subsets)))
    total_cap = per_subset_limit * len(subsets)

    qkeys_seen = set()
    collected: List[Dict] = []

    log.info(f"Streaming mode: pulling up to {total_cap:,} samples "
             f"(~{per_subset_limit:,}/subset) without caching the full dataset to disk.")

    for name in subsets:
        count = 0
        try:
            stream = _stream_subset(args.data_root, name, args.cache_dir)
        except Exception as e:
            log.warning(f"Could not stream subset {name}: {e}")
            continue

        # simple local shuffle reservoir to avoid strong head-bias
        buf = []
        for ex in stream:
            if not _quality_ok(ex, args):
                continue
            q = ex.get("question") or ex.get("seed_question") or ""
            qkey = RE_WS.sub(" ", q.strip()).lower()
            if qkey in qkeys_seen:
                continue
            qkeys_seen.add(qkey)

            buf.append({"text": _format_text(ex), "subset_name": name})
            if len(buf) >= args.shuffle_buffer:
                random.shuffle(buf)
                while buf and count < per_subset_limit:
                    collected.append(buf.pop())
                    count += 1
                buf.clear()
                if count >= per_subset_limit:
                    break

        # flush any remainder
        if count < per_subset_limit and buf:
            random.shuffle(buf)
            for item in buf:
                if count >= per_subset_limit:
                    break
                collected.append(item)
                count += 1

        log.info(f"Subset '{name}': collected {count} examples.")

    # final cap + shuffle
    random.shuffle(collected)
    if len(collected) > total_cap:
        collected = collected[:total_cap]

    n = len(collected)
    if n < 10:
        raise RuntimeError(f"Too few samples collected ({n}). Increase --limit or relax filters.")

    eval_n = min(args.eval_holdout, max(100, n // 10))
    train = collected[eval_n:]
    eval_  = collected[:eval_n]

    log.info(f"Final streaming dataset sizes -> Train: {len(train):,} | Eval: {len(eval_):,}")

    # Create in-memory datasets (tiny; no giant on-disk cache)
    return Dataset.from_list(train), Dataset.from_list(eval_)

# -------------------------
# Model + Tokenizer
# -------------------------
def load_model_and_tokenizer(args):
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if args.bf16 else torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    tok_kwargs = dict(trust_remote_code=args.trust_remote_code)
    if args.cache_dir: tok_kwargs["cache_dir"] = args.cache_dir
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True, **tok_kwargs)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    mdl_kwargs = dict(
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float16,
        quantization_config=bnb_cfg,
        device_map="auto",
        trust_remote_code=args.trust_remote_code,
        low_cpu_mem_usage=True,
    )
    if args.cache_dir: mdl_kwargs["cache_dir"] = args.cache_dir
    model = AutoModelForCausalLM.from_pretrained(args.model, **mdl_kwargs)

    # Train-friendly flags
    model.config.use_cache = False  # avoid warning & excessive memory when checkpointing
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # Optional: Flash-Attn v2 (only if you have it)
    if args.flash_attn:
        try:
            # Works with recent transformers
            if hasattr(model.config, "attn_implementation"):
                model.config.attn_implementation = "flash_attention_2"
            elif hasattr(model, "set_default_attn_implementation"):
                model.set_default_attn_implementation("flash_attention_2")
            log.info("Using Flash Attention 2.")
        except Exception as e:
            log.warning(f"Could not set flash attention 2: {e}")

    if args.compile:
        try:
            model = torch.compile(model)
            log.info("torch.compile enabled.")
        except Exception as e:
            log.warning(f"torch.compile failed ({e}); continuing without compile.")

    # small speedup
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    return model, tokenizer

# -------------------------
# PEFT: QLoRA + LoRA+
# -------------------------
def add_qdora_loraplus(model, args):
    target_modules = [t.strip() for t in args.target_modules.split(",") if t.strip()]
    peft_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM",
        use_dora=True,  # QLoRA-friendly DoRA
    )
    peft_model = get_peft_model(model, peft_cfg)
    peft_model.print_trainable_parameters()

    # LoRA+ param groups — separate LR for lora_A vs lora_B
    a_params, b_params, other_params = [], [], []
    for n, p in peft_model.named_parameters():
        if not p.requires_grad:
            continue
        lname = n.lower()
        if "lora_a" in lname:
            a_params.append(p)
        elif "lora_b" in lname:
            b_params.append(p)
        else:
            other_params.append(p)

    lr = args.lr
    lr_b = args.lr * args.lr_b_scale

    opt_groups = []
    if a_params:      opt_groups.append({"params": a_params, "lr": lr})
    if b_params:      opt_groups.append({"params": b_params, "lr": lr_b})
    if other_params:  opt_groups.append({"params": other_params, "lr": lr})

    try:
        optimizer = torch.optim.AdamW(opt_groups, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0, fused=True)
        log.info("Using fused AdamW.")
    except TypeError:
        optimizer = torch.optim.AdamW(opt_groups, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0)
        log.info("Using standard AdamW (fused unavailable).")

    return peft_model, optimizer

# -------------------------
# Train
# -------------------------
def train(args):
    # Streaming build
    train_ds, eval_ds = build_streaming_dataset(args)

    model, tokenizer = load_model_and_tokenizer(args)
    peft_model, optimizer = add_qdora_loraplus(model, args)

    # TRL config — keep simple and light
    sft_cfg = SFTConfig(
        output_dir=args.output,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        bf16=args.bf16,
        learning_rate=args.lr,                       # base; groups override via param groups
        max_steps=args.steps if args.steps > 0 else -1,
        num_train_epochs=float(args.epochs) if args.steps <= 0 else 1.0,
        warmup_ratio=args.warmup_ratio,
        logging_steps=float(args.logging_steps),
        save_steps=float(args.save_steps),
        save_total_limit=2,
        gradient_checkpointing=args.gradient_checkpointing,
        report_to=None,
        max_grad_norm=args.max_grad_norm,
        dataset_text_field="text",
        dataset_kwargs={"add_special_tokens": args.add_eos},
        max_length=args.max_seq_len,                 # sequence length knob
        packing=args.pack,                           # keep False unless flash-attn available
        remove_unused_columns=True,
        dataloader_num_workers=args.dataloader_num_workers,
    )

    trainer = SFTTrainer(
        model=peft_model,
        args=sft_cfg,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,                  # modern TRL uses 'processing_class'
        optimizers=(optimizer, None),
    )

    t0 = time.time()
    log.info("Starting training…")
    trainer.train()
    dt = time.time() - t0
    log.info(f"Training finished in {dt/3600:.2f} h")

    trainer.save_model(args.output)
    tokenizer.save_pretrained(args.output)
    log.info(f"✅ Adapters saved to: {args.output}")

    # Diagnostics
    try:
        peft_model.eval()
        n_params = count_trainable_params(peft_model)
        log.info(f"Trainable parameters (PEFT): {n_params/1e6:.2f} M")
        sample_n = min(2, len(eval_ds))
        if sample_n > 0:
            batch = [eval_ds[i]["text"] for i in range(sample_n)]
            tok = tokenizer(batch, return_tensors="pt", truncation=True,
                            max_length=min(1024, args.max_seq_len), padding=True)
            tok = {k: v.to(peft_model.device) for k, v in tok.items()}
            with torch.no_grad():
                out = peft_model.generate(**tok, max_new_tokens=128, do_sample=True, top_p=0.9, temperature=0.2)
            print("\n--- Sample generations (tail) ---")
            for i, o in enumerate(out):
                print(f"\n[Sample {i}]")
                print(tokenizer.decode(o, skip_special_tokens=True)[-1000:])
    except Exception as e:
        log.warning(f"Generation smoke-test skipped: {e}")

# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    args = parse_args()
    setup_logging(getattr(logging, args.verbosity))
    torch.manual_seed(args.seed)
    os.makedirs(args.output, exist_ok=True)
    train(args)
