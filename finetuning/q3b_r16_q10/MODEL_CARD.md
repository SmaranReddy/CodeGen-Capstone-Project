# Checkpoint: q3b_r16_q10 (fine-tuned Qwen2.5-Coder-3B for NL → MongoDB)

## What it is
A **LoRA adapter** (not a full model) for `Qwen/Qwen2.5-Coder-3B-Instruct`. To use it, load the base model at
revision `488639f1ff808d1d3d0ba301aef8c11461451ec5` and apply the adapter (or merge it in); see `inference/generate.py`.

| | |
|---|---|
| Adapter files | `adapter_model.safetensors`, `adapter_config.json` (+ tokenizer files, `run_info.json`) |
| Adapter sha256 | `7096760ac2ed7e4a6c2eefdd7c0c249d4b3dba723f069982a499b876d434e09d` (`adapter_model.safetensors`) |
| Size | ~120 MB (≈30M trainable parameters, <1% of the 3B model) |
| Base for training | `unsloth/Qwen2.5-Coder-3B-Instruct-bnb-4bit` (4-bit QLoRA, Unsloth) |
| Base for inference | `Qwen/Qwen2.5-Coder-3B-Instruct` @ `488639f1…`, bfloat16, adapter merged (`merge_and_unload`) |

## Training
| | |
|---|---|
| Method | QLoRA (Unsloth), loss on the answer tokens only |
| LoRA | r 16, alpha 32, dropout 0, all 7 projections (q/k/v/o/gate/up/down) |
| Optimiser | adamw_8bit, lr 2e-4, cosine, warmup 3%, weight decay 0.01, seed 3407 |
| Batch | 2 × grad-accum 4 = 8 effective; max sequence length 2048 |
| Epochs | 1 (2,395 examples ≈ 300 steps) |
| Hardware / time | Kaggle, 1× T4, 30.7 min |
| Train loss | 0.048 |
| Dev loss (answer tokens, 100 dev rows) | 0.775 before → 0.0175 after |

Libraries: unsloth 2026.11.9, transformers 5.17.0, trl 1.15.0, peft 0.21.2, torch 2.11.0+cu130, bitsandbytes 0.50.2
(full list `training/pip_freeze.txt`). Full settings: `training/config.json`. Exact notebook: `training/codegen_finetune_sweep.ipynb`.
All 4 sweep runs: `training/sweep_summary.json` (sweep_20261122T0813Z).

## Data
`data/sft_train.prompt_completion.jsonl`: 2,395 NL→MQL pairs from the approved corpus v2.1 (Spider train, converted
and verified; AI-panel reviewed, not human-reviewed), on MQL-0010's 90 "build" databases.

**Leakage control for MQL-0010:** the 18 MQL-0010 holdout databases are removed entirely (none of the 716 holdout
questions or their databases were seen), as are 11 corpus questions that are near-duplicates of holdout questions
(MQL-0010's L3-L5 exclusion pairs). The remaining 2,901 rows are exactly MQL-0010's registered RAG retrieval view
(ids hash `cfde5e9c…`) minus those 11. Rebuild: `training/build_mql0010_view.py`. Details: `data/manifest.json`.

## How it was chosen
4 configs trained on the same data (baseline, 2 epochs, lr 1e-4, r 32). Picked by lowest dev loss on MQL-0010's 15
registered dev databases (never trained on). Dev losses: 0.0175 (this), 0.0180 (r32), 0.0209 (2 epochs), 0.0231
(lr 1e-4). The holdout was not used for selection.

## Evaluation
- **MQL-0010 holdout (716 questions, 18 unseen databases), zero-shot prompt (arm A0), execution accuracy on Atlas:
  ≈72%** (scored by Smaran with the MQL-0010 scorer; exact count in his scoring output).
- Generated with the unchanged MQL-0010 notebook/package, only the weights swapped (`eval/nl2mql_mql0010_holdout_FT.ipynb`,
  identity in `eval/FT_ADAPTER.json`): bf16, greedy, max 1,024 new tokens, batch 8. 715/716 outputs valid JSON.
- Caveats: one config, no fine-tuned + RAG arm; scored in a separate Atlas session from the base-model arms.

## Known differences between training and evaluation prompts
Same system prompt and template (`nl-mql-spider1-v1`), but training prompts list foreign keys and sample documents as
`(none)` and only the collections the corpus touched; evaluation prompts include them. See `PROMPT_FORMAT.md`.
