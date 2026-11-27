# Fine-tuning plan: Qwen2.5-Coder NL→MQL (ft-v1)

Status: plan only. **No training has been run.** The dataset this plan consumes is
`data/finetune/ft_v1/`, described in `docs/fine_tuning_dataset_expansion_report.md`.

Sections 1–2 separate **evidence** (what sources or our own measurements say) from
**recommendations** (what we propose).

---

## 1. Evidence

| Topic | Evidence | Source |
|---|---|---|
| QLoRA memory | 4-bit NF4 base weights plus LoRA adapters fine-tuned a 65B model on one 48 GB GPU without losing 16-bit finetuning quality. The paper gives no 7B training number. | Dettmers et al., NeurIPS 2023 ([paper](https://papers.neurips.cc/paper_files/paper/2023/file/1feb87871436031bdc0f2beaa62a049b-Paper-Conference.pdf)) |
| 7B VRAM (QLoRA vs LoRA-16bit) | Unsloth lists ≈5 GB as an absolute floor for 4-bit QLoRA of a 7B model and ≈19 GB for 16-bit LoRA. Third-party estimates for 7B QLoRA are 8–12 GB. Actual use scales with sequence length and batch size. | [Unsloth requirements](https://docs.unsloth.ai/get-started/beginner-start-here/unsloth-requirements); [VRLA Tech](https://vrlatech.com/how-much-vram-do-you-need-for-llm-fine-tuning-in-2026/) |
| Full fine-tuning cost | Rule of thumb, not measured here: mixed-precision AdamW needs about 16 bytes/parameter, i.e. ≈120 GB for 7.6B parameters before activations. | standard estimate (our arithmetic) |
| LoRA placement / LR | LoRA on all weight matrices, the MLP layers especially, beats attention-only LoRA. The optimal LoRA LR is ≈10× the full-FT LR. LoRA is more sensitive than full FT to large batch sizes. | [Thinking Machines, "LoRA Without Regret"](https://thinkingmachines.ai/blog/lora/); [TRL guide](https://huggingface.co/docs/trl/main/en/lora_without_regret) |
| LoRA trade-off | LoRA learns less, and forgets less, than full fine-tuning. | Biderman et al. 2024 ([arXiv 2405.09673](https://arxiv.org/pdf/2405.09673)) |
| Loss masking | TRL `SFTTrainer` computes loss on the completion only for prompt-completion datasets (`completion_only_loss` defaults to True). `assistant_only_loss=True` needs a chat template with `{% generation %}` markers. | [TRL SFT docs](https://huggingface.co/docs/trl/main/en/sft_trainer) |
| Model | Qwen2.5-Coder comes in 0.5/1.5/3/7/14/32B. The 7B Instruct model has 7.61B parameters, 28 layers, GQA (28 Q / 4 KV heads), and a 131K context. | [Qwen2.5-Coder report](https://arxiv.org/pdf/2409.12186); [model card](https://huggingface.co/unsloth/qwen2.5-coder-7b-instruct-bnb-4bit) |
| Free GPU | Kaggle gives about 30 GPU-h/week on one P100 (16 GB) or 2×T4 (2×16 GB). Session caps are reported as 9–12 h. These are second-hand figures; check the Kaggle UI. | [GeeksforGeeks](https://geeksforgeeks.org/how-to-use-gpu-in-kaggle); [Spheron](https://www.spheron.network/blog/google-colab-alternatives-8-gpu-clouds-compared-2026/) |
| Spider contamination | GPT-3.5 shows clear prior knowledge of Spider, which inflates zero-shot text-to-SQL scores. Older benchmarks such as Spider show the strongest leakage signals. | Ranaldi et al. 2024 ([arXiv 2402.08100](https://arxiv.org/pdf/2402.08100)) |
| Our baselines (n=455, Atlas execution) | Qwen-3B 217 (47.7%), Qwen-7B 287 (63.1%), gpt-oss-120b 385 (84.6%). On validation, Qwen-3B zero-shot scores 33/72 and RAG hybrid_k3 43/72. | `docs/spider1_expanded_benchmark.md`, `docs/week3_rag_report.md` |
| Our sequence lengths | Measured with the Qwen2.5 tokenizer on the ChatML-rendered records: prompt mean 1,489 / p95 2,649 / max 3,951 tokens; completion mean 67 / max 278; total max 4,053. Over 95% of each sequence is prompt, which the loss masks. | `scripts/finetune_v1/token_audit.py` → `data/finetune/ft_v1/token_audit.json` |
| Our hardware | No local GPU. Qwen runs on CPU via llama.cpp GGUF. Kaggle GPU runs already work (`scripts/kaggle_week3`). | project history |

---

## 2. Recommendations

### 2.1 Method: QLoRA, not full fine-tuning
- **Primary model:** Qwen2.5-Coder-**7B**-Instruct with QLoRA (4-bit NF4, double quantisation,
  bf16/fp16 compute) on one Kaggle 16 GB GPU.
- **Pilot model:** **3B** first, to validate the training loop, the loss mask and the eval
  wiring cheaply. The 3B pilot is also the closest comparison to the Week-3 RAG numbers.
- **Full fine-tuning: not recommended.** It needs about 120 GB of GPU memory for 7B, which we
  don't have. And with roughly 2–3K short examples it would raise the risk of forgetting
  general MQL knowledge (Biderman et al.).
- **16-bit LoRA:** only if a ≥24 GB GPU becomes available. It's an ablation, not the default.

### 2.2 Data format
- **Use** `data/finetune/ft_v1/sft_train.prompt_completion.jsonl`.
  - `prompt` = [system, user]. `completion` = [assistant], which holds the target MQL as
    compact JSON.
  - This is the benchmark's exact inference prompt: `prompt_spider1.SYSTEM` + `USER_TEMPLATE`,
    prompt version `nl-mql-spider1-v1`. Train-time and eval-time inputs therefore have the same
    shape.
- Apply Qwen2.5's own chat template. Loss falls on the completion only, via TRL's default for
  prompt-completion data. Don't depend on `assistant_only_loss`, because Qwen2.5's template has
  no generation markers.
- **Schema context:**
  - Spider examples get the full database: all collections, observed BSON type plus declared
    SQL type, FKs, and 2 frozen sample documents per collection.
  - Kaggle-derived examples get the target collection(s) plus 2 deterministic distractor
    collections, so the model still has to pick a collection. FKs are "(none)" because the
    source has none. Sample documents are frozen and generic, never the query's literals.
- Keep reference SQL, expected results and fixture metadata **out of the input**. The tests
  enforce this.

### 2.3 Training configuration (starting point; tune on validation only)
| Setting | 3B pilot | 7B main |
|---|---|---|
| Quantisation | 4-bit NF4, double quant | same |
| LoRA targets | `all-linear` (q,k,v,o,gate,up,down) | same |
| rank / alpha / dropout | r=16, α=32, 0.05 | r=16 (ablate 32), α=2r, 0.05 |
| LR / schedule | 2e-4, cosine, 3% warmup | 1e-4–2e-4, cosine, 3% warmup |
| Effective batch | 16 (micro 2 × grad-accum 8) | 16 (micro 1 × 16) |
| Epochs | 2 (select by validation execution accuracy) | 2–3 |
| max_seq_length | **4096**: the token audit (real Qwen tokenizer, `token_audit.json`) gives max 4,053 and p99 3,371 tokens, so no record is truncated | same |
| Precision | fp16 on T4/P100 (no bf16), gradient checkpointing on | same |
| Packing | off: keeps the completion-only mask simple and avoids cross-example attention | same |
| Seed | 3 seeds for the final configuration (variance estimate) | same |

The LR follows the "≈10× full-FT LR" evidence. Batch size is kept moderate because LoRA is
sensitive to large batches.

### 2.4 Data mix
The Kaggle-derived pool is mostly single-collection filter/sort/limit queries over a
synthetic schema. The Spider train examples carry the benchmark's real multi-collection
difficulty. To keep the simple, template-heavy pool from dominating:
- **Runs, compared on benchmark validation:**
  - (A) Spider-280 only
  - (B) Spider-280 + all Kaggle verified
  - (C) Spider-280 upweighted ×3 + Kaggle verified
  - (D) Kaggle only, as a transfer probe
- The near-duplicate cap (3 per question-template × SQL-skeleton cluster) is already applied.
  Don't uncap.
- Report per-hardness and per-pattern accuracy, so gains on easy patterns don't hide
  regressions on joins and grouping.

### 2.5 Evaluation and model selection
- **Model selection:** benchmark **validation** (72 items, 7 held-out databases).
  - Use the existing execution-based pipeline: `scripts/spider1/run_eval.py`, the read-only
    `eval_kit.validate_mql`, Atlas execution as `codegen_eval_readonly`, and
    `spider1_kit.exec_match`.
  - Use the same prompt version and generation settings as the stored baselines.
- **dev_internal** (cluster-grouped 5% of new data): loss and overfitting monitoring only.
  It is easier than the benchmark and must not drive model selection.
- **Baselines, under the same runtime and decoding:**
  - zero-shot base model
  - Week-3 RAG `hybrid_k3`
  - FT model
  - FT + RAG
  - Run every comparison in the same session or runtime: llama.cpp's KV reuse made
    cross-session reruns drift (Week 3).
  - If the FT model is evaluated as GGUF on CPU, the base model must be too. If both run on
    Kaggle GPU, generate both fresh there (clean-control design, `scripts/kaggle_week3`).
- **Statistics:** paired McNemar against the zero-shot and RAG baselines on the same 72 items,
  with Holm correction across the planned contrasts. The analysis plan is frozen before
  scoring, as in Week 3.
- **Test (65):** stays untouched. Run it exactly once, for the final selected configuration,
  and only when explicitly requested.

### 2.6 MQL syntax and safety
- **At inference:** keep the existing gate unchanged. Every generated query passes
  `eval_kit.validate_mql`, which:
  - allows only `aggregate`/`find`;
  - rejects `$out`, `$merge`, `$function`, `$accumulator` and `$where`;
  - restricts collections to the example's database;
  - checks `$unionWith` targets.
  Only then does it run, on the read-only Atlas user.
- **Training targets:** all of them already pass the same validator. The tests check that no
  forbidden stage occurs in `train.jsonl`.
- **Optional:** JSON-grammar-constrained decoding (llama.cpp GBNF) to remove format errors.
  Report it as a separate condition, not mixed into the main comparison.

### 2.7 Contamination and overfitting checks
- **Built into the dataset:**
  - Spider examples come from the train partition only.
  - Every Kaggle question is compared with all 137 held-out questions (trigram Jaccard) and
    with all Spider train/dev questions.
  - Collection namespaces cannot collide.
  - The RAG `leakage.check` and extra checks are run and recorded in
    `data/finetune/ft_v1/leakage_report.json`.
- **During training:**
  - Track the train/dev_internal loss gap.
  - Evaluate validation accuracy at each epoch checkpoint.
  - Stop at the best validation checkpoint.
- **Memorisation probe:** on validation, compare accuracy on items whose SQL pattern is
  over-represented in the Kaggle pool against the rest.
- **Pre-training contamination:** Qwen may have seen Spider (Ranaldi et al.). This affects
  base and FT equally; it is a caveat for absolute numbers, not for the FT-vs-base delta.

### 2.8 Hardware needs
| Run | GPU | Expected VRAM | Notes |
|---|---|---|---|
| 3B QLoRA pilot | Kaggle T4 or P100 16 GB | ≈5–8 GB | minutes per epoch at our size |
| 7B QLoRA main | Kaggle T4/P100 16 GB | ≈8–14 GB at the audited seq length | reduce micro-batch to 1 if OOM |
| 7B LoRA-16bit (ablation) | ≥24 GB (not available free) | ≈19 GB+ | optional |
| Inference/eval | Kaggle GPU (fresh, paired) or CPU GGUF Q4_K_M | — | same runtime for every compared model |

### 2.9 Exact next commands
```bash
# 1. rebuild + verify the dataset (deterministic; verify.py resumes)
python scripts/finetune_v1/extract.py
python scripts/finetune_v1/verify.py --workers 6
python scripts/finetune_v1/build_ft.py
python scripts/finetune_v1/replay_targets.py        # every target replayed from its text; must be 100%
python scripts/finetune_v1/verifier_selftest.py --sample 400
python -m pytest tests/test_finetune_v1.py -q

# 2. package for Kaggle (upload as a private Kaggle dataset "ft-v1"):
#    data/finetune/ft_v1/sft_train.prompt_completion.jsonl
#    data/finetune/ft_v1/sft_dev.prompt_completion.jsonl
#    data/finetune/ft_v1/build_manifest.json
#    scripts/finetune_v1/train_qlora.py

# 3. on Kaggle GPU (NOT RUN; written but untested). 3B pilot first, then 7B:
pip install -q "transformers>=4.46" "trl>=0.20" "peft>=0.13" "bitsandbytes>=0.44" "accelerate>=1.0" datasets
python train_qlora.py --model Qwen/Qwen2.5-Coder-3B-Instruct --data /kaggle/input/ft-v1 --out /kaggle/working/qlora-3b
python train_qlora.py --model Qwen/Qwen2.5-Coder-7B-Instruct --data /kaggle/input/ft-v1 --out /kaggle/working/qlora-7b --micro-batch 1 --grad-accum 16
pip freeze > /kaggle/working/pip_freeze.txt

# 4. evaluation (next task): in the same Kaggle session, generate VALIDATION (72) predictions
#    for the base model and the adapter with identical prompts/decoding (pattern of
#    scripts/kaggle_week3), then score locally with the existing Atlas pipeline
#    (read-only user) and McNemar/Holm as frozen in the analysis plan. Test (65) untouched.
```

`train_qlora.py` targets the TRL `SFTConfig` API as documented for v0.20+ (`max_length`,
`completion_only_loss`). Older TRL versions name the length argument differently. The script
has not been executed, so expect to adjust it to the versions Kaggle installs.

### 2.10 Licensing of the training mix
- The Kaggle-derived records are **CC BY-NC-SA 4.0**. The Spider-derived records are
  **CC BY-SA 4.0**.
- One combined adaptation cannot satisfy both ShareAlike terms at once.
- Every record carries its own `provenance.license`, so the two sources stay separable.
  `source_dataset` filters them.
- An adapter trained on the Kaggle-derived records inherits the **non-commercial**
  restriction. That is fine for this capstone; anything beyond academic use needs a licence
  review first.
