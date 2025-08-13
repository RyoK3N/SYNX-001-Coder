#!/usr/bin/env python3
# chat_SYNX001_lora.py
# Minimal chat REPL for Qwen2.5-Coder-7B-Instruct + PEFT (QLoRA) adapters.
# - Loads base model + LoRA adapters from disk
# - 4-bit quantization by default (fits well on a single 48GB L40S)
# - Uses tokenizer's chat_template if present (your checkpoint saved one)
# - Simple terminal chat loop with /clear and /exit

import os
import argparse
import logging
import json
from typing import List, Dict, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

LOG = logging.getLogger("chat_qwen_lora")

DEFAULT_SYSTEM = "You are a helpful coding assistant. Think step-by-step and return clean, efficient code when asked."

def setup_logging(verbosity: str = "INFO"):
    lvl = getattr(logging, verbosity.upper(), logging.INFO)
    logging.basicConfig(
        level=lvl,
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    # Quiet some chatty deps
    for noisy in ("urllib3", "filelock", "fsspec", "huggingface_hub", "datasets"):
        logging.getLogger(noisy).setLevel(max(lvl, logging.WARNING))
    logging.getLogger("transformers").setLevel(lvl)

def parse_args():
    ap = argparse.ArgumentParser(description="Chat with a Qwen model + LoRA adapters (low VRAM).")
    ap.add_argument("--base-model", type=str, default="Qwen/Qwen2.5-Coder-7B-Instruct",
                    help="Base model ID or local path.")
    ap.add_argument("--adapters", type=str, required=True,
                    help="Path to PEFT/LoRA adapter directory (e.g. ./qwen25coder7b-rstar-qdora-light).")
    ap.add_argument("--quantization", type=str, choices=["4bit", "8bit", "none"], default="4bit",
                    help="Quantization for the base model.")
    ap.add_argument("--bf16", action="store_true", help="Use bfloat16 where applicable.")
    ap.add_argument("--flash-attn", action="store_true", help="Try to enable Flash Attention 2.")
    ap.add_argument("--merge", action="store_true",
                    help="Merge LoRA into the base weights (requires non-4bit). Not recommended for low VRAM.")
    ap.add_argument("--system", type=str, default=DEFAULT_SYSTEM, help="System prompt.")
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--repetition-penalty", type=float, default=1.1)
    ap.add_argument("--verbosity", type=str, default="INFO", choices=["DEBUG","INFO","WARNING","ERROR"])
    return ap.parse_args()

def load_tokenizer(model_id: str, trust_remote_code: bool = True, cache_dir: Optional[str] = None):
    tok_kwargs = dict(trust_remote_code=trust_remote_code)
    if cache_dir:
        tok_kwargs["cache_dir"] = cache_dir
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True, **tok_kwargs)
    # Ensure a pad token exists
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

def load_model(base_model: str, adapters_path: str, quantization: str, bf16: bool, flash_attn: bool, merge: bool):
    dtype = torch.bfloat16 if bf16 else torch.float16

    bnb_cfg = None
    mdl_kwargs = dict(
        device_map="auto",
        torch_dtype=dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    if quantization == "4bit":
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
        )
        mdl_kwargs["quantization_config"] = bnb_cfg
    elif quantization == "8bit":
        bnb_cfg = BitsAndBytesConfig(load_in_8bit=True)
        mdl_kwargs["quantization_config"] = bnb_cfg

    LOG.info(f"Loading base model: {base_model} (quantization={quantization})")
    model = AutoModelForCausalLM.from_pretrained(base_model, **mdl_kwargs)

    # Disable KV cache during training-like phases; enable for inference
    model.config.use_cache = True

    # Optional: Flash-Attn 2 (only if installed & supported)
    if flash_attn:
        try:
            if hasattr(model.config, "attn_implementation"):
                model.config.attn_implementation = "flash_attention_2"
            elif hasattr(model, "set_default_attn_implementation"):
                model.set_default_attn_implementation("flash_attention_2")
            LOG.info("Flash Attention 2 enabled.")
        except Exception as e:
            LOG.warning(f"Could not enable Flash Attention 2: {e}")

    LOG.info(f"Loading LoRA adapters from: {adapters_path}")
    model = PeftModel.from_pretrained(model, adapters_path, is_trainable=False)

    if merge:
        if quantization == "4bit":
            raise ValueError("Merging LoRA into base weights is incompatible with 4-bit quantization. "
                             "Reload with --quantization none or 8bit, then use --merge.")
        LOG.info("Merging LoRA weights into the base model (this may take a while)…")
        model = model.merge_and_unload()

    model.eval()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    return model

def apply_chat_template(tokenizer, messages: List[Dict[str, str]]) -> str:
    """
    Use tokenizer's native chat template if available, else fall back to a simple format.
    """
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    except Exception:
        # Fallback: very simple prompt if no template is present
        sys = ""
        if messages and messages[0]["role"] == "system":
            sys = f"<|system|>\n{messages[0]['content']}\n</s>\n"
            msgs = messages[1:]
        else:
            msgs = messages
        parts = [sys]
        for m in msgs:
            if m["role"] == "user":
                parts.append(f"<|user|>\n{m['content']}\n</s>\n")
            elif m["role"] == "assistant":
                parts.append(f"<|assistant|>\n{m['content']}\n</s>\n")
        parts.append("<|assistant|>\n")
        return "".join(parts)

@torch.inference_mode()
def generate_reply(model, tokenizer, messages: List[Dict[str,str]],
                   max_new_tokens: int, temperature: float, top_p: float, repetition_penalty: float) -> str:
    prompt = apply_chat_template(tokenizer, messages)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    gen_cfg = dict(
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=max(0.0, temperature),
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=getattr(model.generation_config, "eos_token_id", tokenizer.eos_token_id),
    )
    output_ids = model.generate(**inputs, **gen_cfg)
    # We added generation prompt, so only decode the new text after the input length
    gen_text = tokenizer.decode(output_ids[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
    return gen_text.strip()

def main():
    args = parse_args()
    setup_logging(args.verbosity)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    tokenizer = load_tokenizer(args.base_model, trust_remote_code=True)
    model = load_model(args.base_model, args.adapters, args.quantization, args.bf16, args.flash_attn, args.merge)

    LOG.info("Ready! Type your message and press Enter. Commands: /clear, /save <path>, /exit")

    history: List[Dict[str, str]] = [{"role": "system", "content": args.system}]

    while True:
        try:
            user = input("\nYou > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n")
            break

        if not user:
            continue

        # Commands
        if user.lower() in ("/exit", "/quit"):
            break
        if user.lower().startswith("/clear"):
            history = [{"role": "system", "content": args.system}]
            print("History cleared.")
            continue
        if user.lower().startswith("/save"):
            parts = user.split(maxsplit=1)
            path = parts[1].strip() if len(parts) > 1 else "chat_history.json"
            try:
                with open(path, "w") as f:
                    json.dump(history, f, indent=2, ensure_ascii=False)
                print(f"Saved conversation to {path}")
            except Exception as e:
                print(f"Failed to save: {e}")
            continue

        # Normal turn
        history.append({"role": "user", "content": user})
        reply = generate_reply(
            model, tokenizer, history,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
        )
        history.append({"role": "assistant", "content": reply})
        print(f"\nAssistant > {reply}")

if __name__ == "__main__":
    main()
