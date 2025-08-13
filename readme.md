## Training code 
```bash
 ~ python finetune_rstar_qdora_light.py \
    --model Qwen/Qwen2.5-Coder-7B-Instruct \
    --data-root microsoft/rStar-Coder \
    --sft-subsets synthetic_sft,seed_sft \
    --limit 4000 --eval_holdout 400 \
    --max-seq-len 2048 \
    --steps 200 --per-device-train-batch-size 1 --gradient-accumulation-steps 8 \
    --lora-r 16 --lora-alpha 32 --lora-dropout 0.05 \
    --verbosity INFO
```

## Run Chat Infrerence after training
### 1) Activate the same env you trained in (has transformers, peft, bitsandbytes, torch)
### 2) Point to your adapter folder (the one you just trained)
```bash
python chat_qwen_lora.py \
  --base-model Qwen/Qwen2.5-Coder-7B-Instruct \
  --adapters ./qwen25coder7b-rstar-qdora-light \
  --quantization 4bit \
  --bf16 \
  --verbosity INFO
```
