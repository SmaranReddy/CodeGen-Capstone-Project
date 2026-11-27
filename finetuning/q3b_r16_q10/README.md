# Fine-tuning: adapter `q3b_r16_q10` (used as FT in MQL-0015)

QLoRA fine-tune of `Qwen/Qwen2.5-Coder-3B-Instruct`, trained with Unsloth on its 4-bit base, on a Kaggle T4. These are the source files of the adapter that MQL-0015 evaluated as **FT-R3V (59.8 %)**. **No weights are included** (`adapter_model.safetensors`, sha256 `7096760ac2ed7e4a6c2eefdd7c0c249d4b3dba723f069982a499b876d434e09d`).

| File | Content |
|---|---|
| `training/codegen_finetune_sweep.ipynb` | training notebook: unsloth `FastLanguageModel` 4-bit, `get_peft_model` (r 16, α 32, dropout 0, q/k/v/o/gate/up/down), TRL `SFTTrainer` on the chat-template text, **loss masking with `train_on_responses_only`** (only the assistant answer is supervised), adamw_8bit, cosine, warmup 0.03, 1 epoch, effective batch 8, max length 2,048 |
| `training/config.json` | exact hyper-parameters of this run; `training/sweep_summary.json` has the other sweep entries |
| `training/pip_freeze.txt` | exact training environment (unsloth 2026.11.9, trl 1.15.0, transformers 5.17.0, peft 0.21.2, ...) |
| `training/build_pilot_sft.py`, `training/build_mql0010_view.py` | data preparation: the approved corpus v2.1 rendered to prompt / completion rows; MQL-0010 holdout DBs removed |
| `PROMPT_FORMAT.md`, `inference/system_prompt.txt`, `inference/example_user_prompt.txt` | the training and inference prompt format (the example uses one MQL-0010 holdout question from Spider *train*; it is not a test-set item) |
| `inference/generate.py` | stand-alone inference with base + adapter |
| `eval/nl2mql_mql0010_holdout_FT.ipynb`, `eval/FT_ADAPTER.json` | holdout evaluation notebook used for the MQL-0010-FT1 side run (516 / 716) |
| `adapter_metadata/` | `adapter_config.json`, `run_info.json`, chat template and tokenizer config. The tokenizer itself is the base model's (`488639f1`) |
| `MODEL_CARD.md`, `HANDOFF_README.md` | model card and the original hand-off notes |

**How MQL-0015 applied the adapter.** It was merged into the bf16 base (`peft` `merge_and_unload`) inside the Kaggle runtime (`kaggle/nl2mql_mql0015_final_1000/src/mql0015_core.py`, `merge_adapter`). The merged weights were hash-recorded in the run manifest.

**Data not included:** `sft_train` / `sft_dev` prompt-completion files and the corpus. Rebuild them with the scripts above from the approved corpus.
