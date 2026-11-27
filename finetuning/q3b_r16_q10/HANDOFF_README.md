# Fine-tuned model handoff: q3b_r16_q10

Fine-tuned Qwen2.5-Coder-3B-Instruct (LoRA) for NL → MongoDB aggregation queries. ≈72% execution accuracy on the
MQL-0010 holdout (716 questions, 18 unseen databases, zero-shot prompt).

| You asked for | Where |
|---|---|
| Model checkpoint | `checkpoint/q3b_r16_q10/` (LoRA adapter + tokenizer + `run_info.json`; loads on top of the base model) |
| Training dataset | `data/` (train 2,395 / dev 506 / full 2,901 + `manifest.json`) |
| Training code and configuration | `training/` (notebook, `config.json`, dataset build scripts, `sweep_summary.json`, `pip_freeze.txt`) |
| Prompt format | `PROMPT_FORMAT.md` + `inference/system_prompt.txt`, `inference/example_user_prompt.txt` |
| Checkpoint details | `MODEL_CARD.md` |
| How it was evaluated | `eval/` (the MQL-0010 notebook with the fine-tuned weights + `FT_ADAPTER.json`) |
| Run it (demo) | `inference/generate.py` |

## Quick start
```bash
pip install torch transformers peft
python inference/generate.py --adapter checkpoint/q3b_r16_q10 \
    --system inference/system_prompt.txt --user inference/example_user_prompt.txt
# or a frozen MQL-0010 prompt:
python inference/generate.py --adapter checkpoint/q3b_r16_q10 --prompts <MQL-0010>/data/eval/prompts.jsonl --id spider_train_00439
# one merged model folder (loads with plain transformers, no peft):
python inference/generate.py --adapter checkpoint/q3b_r16_q10 --merge-to merged_q3b_r16_q10 --system ... --user ...
```
It downloads the base model (`Qwen/Qwen2.5-Coder-3B-Instruct` @ `488639f1…`, ~6 GB) on first run.

## Check the checkpoint is the right one
`sha256sum checkpoint/q3b_r16_q10/adapter_model.safetensors` must be
`7096760ac2ed7e4a6c2eefdd7c0c249d4b3dba723f069982a499b876d434e09d`. Only adapters ending in `_q10` are safe for
MQL-0010; the earlier `qwen25coder3b_lora_*` adapters were trained on the holdout databases.

## Reproduce
1. `training/build_mql0010_view.py --corpus corpus.jsonl --sft-dir approved_v2_1 --split split_registration.json --out data`
   rebuilds `data/` byte-for-byte (`build_pilot_sft.py` built `approved_v2_1` from the corpus).
2. `training/codegen_finetune_sweep.ipynb` on Kaggle (T4, Internet on, input = the `data/` files) trains it; the
   baseline entry of `SWEEP` is this run.
3. `eval/nl2mql_mql0010_holdout_FT.ipynb` with the MQL-0010 package + the adapter regenerates the scored predictions.
