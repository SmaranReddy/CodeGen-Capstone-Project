"""QLoRA supervised fine-tuning of Qwen2.5-Coder on ft-v1 (NOT RUN YET; written for Kaggle GPU).

    # Kaggle notebook, GPU (T4 / P100 16 GB). Install, then run:
    pip install -q "transformers>=4.46" "trl>=0.20" "peft>=0.13" "bitsandbytes>=0.44" "accelerate>=1.0" datasets
    python train_qlora.py --model Qwen/Qwen2.5-Coder-3B-Instruct --data /kaggle/input/ft-v1 --out /kaggle/working/qlora-3b
    python train_qlora.py --model Qwen/Qwen2.5-Coder-7B-Instruct --data /kaggle/input/ft-v1 --out /kaggle/working/qlora-7b --micro-batch 1 --grad-accum 16

Input: sft_train.prompt_completion.jsonl / sft_dev.prompt_completion.jsonl (conversational
prompt-completion; TRL computes the loss on the completion only). Hyperparameters follow
docs/fine_tuning_plan.md section 2.3. This script only trains and saves the LoRA adapter;
model selection is done by execution accuracy on the benchmark VALIDATION partition with the
existing scoring pipeline, never on dev loss alone. The test partition is not touched.
Pin the exact library versions you install in the run log (pip freeze) for reproducibility.
"""
import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-Coder-3B-Instruct")
    ap.add_argument("--data", required=True, help="folder holding the ft-v1 sft_*.prompt_completion.jsonl files")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--micro-batch", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--max-length", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=20261024)
    args = ap.parse_args()

    import torch
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    data = Path(args.data)
    ds = load_dataset("json", data_files={"train": str(data / "sft_train.prompt_completion.jsonl"),
                                          "dev": str(data / "sft_dev.prompt_completion.jsonl")})
    ds = ds.remove_columns(["id"])
    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                             bnb_4bit_compute_dtype=torch.bfloat16 if bf16 else torch.float16)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, quantization_config=bnb, device_map="auto",
                                                 torch_dtype=torch.bfloat16 if bf16 else torch.float16)
    lora = LoraConfig(r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.05, target_modules="all-linear",
                      task_type="CAUSAL_LM")
    cfg = SFTConfig(
        output_dir=args.out, num_train_epochs=args.epochs, learning_rate=args.lr,
        lr_scheduler_type="cosine", warmup_ratio=0.03, per_device_train_batch_size=args.micro_batch,
        per_device_eval_batch_size=1, gradient_accumulation_steps=args.grad_accum,
        gradient_checkpointing=True, bf16=bf16, fp16=not bf16, max_length=args.max_length,
        packing=False, completion_only_loss=True, eval_strategy="epoch", save_strategy="epoch",
        logging_steps=10, seed=args.seed, report_to="none")
    trainer = SFTTrainer(model=model, args=cfg, train_dataset=ds["train"], eval_dataset=ds["dev"],
                         processing_class=tok, peft_config=lora)
    trainer.train()
    trainer.save_model(args.out)
    Path(args.out, "run_config.json").write_text(json.dumps(vars(args), indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
