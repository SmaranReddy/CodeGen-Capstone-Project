# MQL-0010: Kaggle package and exact execution instructions

Date: 2026-11-19.

**Status: PACKAGE BUILT, NOT EXECUTED.** No prediction, Kaggle run, Atlas connection or provisioning has happened. Stopped for human review.

## 1. Deliverables

| Item | Path | sha256 |
|---|---|---|
| Upload-ready ZIP (38 files + manifest + sums; 1.1 MB) | `kaggle/nl2mql_mql0010_holdout_bundle.zip` | `5dfb8e19322414127aaaeb27945594e5f788d6692662086b7473bf83980df25f` |
| Package folder (identical content) | `kaggle/nl2mql_mql0010_holdout/` | — |
| Package manifest (per-file sha256 and bytes) | `kaggle/nl2mql_mql0010_holdout/package_manifest.json` | `87bea6957459f3ca02382a4400563d2da20a575dccc55047eccfef4d765bb164` |
| Checksums | `kaggle/nl2mql_mql0010_holdout/SHA256SUMS.txt` | — |
| Notebook | `kaggle/nl2mql_mql0010_holdout/nl2mql_mql0010_holdout.ipynb` | `5f330e4e…` |
| Package config | `kaggle/nl2mql_mql0010_holdout/configs/experiment.json` | `0f80cb87…` |
| Registration index (A4, unchanged) | `outputs/nl2mql_mql0010/preflight/registration_index.json` | `11b56c79…` |
| Atlas scoring and provisioning contract | `outputs/nl2mql_mql0010/scoring_contract/atlas_mapping.json` | `b14b3607…` |

**How the package was built:** `~/codegen_data/venv_tok/Scripts/python.exe -I -B scripts/nl2mql_mql0010/build_mql0010_package.py --tokenizer-dir ~/codegen_data/qwen3b_tok_488639f1`, using the pinned Qwen tokenizer.

**What is in the package:**
- registration copies;
- the 9 registered source modules, byte-identical to the A4-pinned hashes;
- 5 byte copies of the MQL-0009 package modules, and the new runtime core `src/mql0010_core.py`;
- approved corpus v2.1, unchanged; the view is enforced at runtime;
- 716 evaluation inputs (id, db_id, question only) and their frozen zero-shot prompts;
- the holdout schema context;
- the frozen plans A0, D0, E and EF.

**What it does not contain:** gold labels, credentials or a database driver.

## 2. What the notebook enforces before generating

1. Every package file matches the manifest, and no extra file is present.
2. The registration index matches the config pin. Every pinned document copy and all 9 registered source files match their A4 hashes.
3. **Retrieval view (R3):**
   - corpus hash;
   - the registered view id hash and count (2,912);
   - no document from the 18 holdout databases;
   - `planner.build_plan(forbidden_db_ids=…)`.
4. The evaluation ids hash (716 items, the 18 holdout databases), the frozen prompt hash and the template hash, and that no gold-like field is present.
5. Hardware and precision:
   - a Tesla T4 only, bf16 (emulated), or the run stops;
   - the pinned model revision is checked twice: download metadata and Hugging Face model info.
6. All four plans and every prompt are re-derived with the model's tokenizer. A single differing item stops the run, and the delta item list must equal the frozen 52.
7. The generation settings must equal A4: greedy, `max_new_tokens` 1024, batch 8 for first passes and batch 1 for the delta and retry passes.

## 3. Passes, sessions and outputs

| Session | Passes (in order) | Generations | ≈ T4-hours |
|---|---|---:|---:|
| 1 | `first:A0`, `first:D0`, `first:E` (batch 8), then `delta` (EP and EF, batch 1) | 2,148 + 104 | 7.6 |
| 2 | `retry:A0V`, `retry:D0V`, `retry:D0G`, `retry:FULL` (batch 1; only validator-flagged answers) | ≈ 620 | 2.4 |

**Checkpoints.** Every pass writes an append-only, fsynced, fingerprinted checkpoint under `passes/<pass>/`. A resumed run never regenerates a completed record and refuses a changed environment or config.

**At export the notebook:**
- assembles the 9 arms with the `driver.py` semantics;
- runs **`scoring.verify_exports`** as a gate: reuse is byte-exact, D0V and D0G retried the same items, and the matched delta and the prompt hashes check out;
- writes `predictions/<arm>/predictions.jsonl`, `plans/`, `MQL0010_PREDICTIONS_MANIFEST.json`, `MQL0010_STATUS.json`, `SHA256SUMS.txt` and `/kaggle/working/mql0010_predictions.zip`.

## 4. Exact steps (each needs your go-ahead)

**A. Kaggle generation**
1. In Kaggle, go to *Datasets → New dataset*. Upload `kaggle/nl2mql_mql0010_holdout_bundle.zip` and keep it private.
2. Session 1:
   - **Notebook:** create it from `nl2mql_mql0010_holdout.ipynb`.
   - **Settings:** Accelerator *GPU T4 x2*, Internet **on**, Persistence *Files only*, the package dataset attached, **no secrets**.
   - **Run:** keep `SESSION = 1` and choose *Save & Run All (Commit)*.
   - **Expected:** about 7.6 h, ending with "MQL-0010 STATUS: partial" (retries not yet run). That status is normal here.
   - **Afterwards:** download `mql0010_predictions.zip`, then pin the environment (*Settings → Environment → Pin to original environment*).
3. Session 2:
   - **New version:** create one and add the session-1 output as an input.
   - **Settings:** `SESSION = 2`, `RESUME_FROM = "auto"`.
   - **Run:** about 2.4 h, ending with "MQL-0010 STATUS: complete".
   - **Afterwards:** download `mql0010_predictions.zip` and put it at the repo root.
4. **If a session is cut short** (time budget, an error or a disconnect): attach that version's output, set `RESUME_FROM = "auto"` and run the same `SESSION` again.

**B. Atlas provisioning.** Separate approval; see `docs/mql0010_atlas_provisioning_plan.md`.

**C. Scoring** (local Windows, read-only; separate approval):
```
python -B scripts/nl2mql_mql0010/mql0010_scoring.py verify --predictions-zip mql0010_predictions.zip
python -B scripts/nl2mql_mql0010/mql0010_scoring.py score  --predictions-zip mql0010_predictions.zip --run-name run_final
```
`score` runs the following in one session:
- export verification;
- the gold reference, computed once from local SQLite and frozen by hash;
- one read-only full-parity Atlas preflight on `codegen_spider1_holdout_mql0010`;
- the unchanged registered evaluator, resumable, on all 9 arms;
- `scoring.analyze` in the registered A4 order: D0V vs D0 with the +1.0 pp criterion, then Holm over {FULL vs D0, D0V vs D0G}, then Holm over {E vs D0, EF vs EP}.

Outputs go to `outputs/nl2mql_mql0010/scoring_runs/run_final/`.

**D. Records:** a ledger entry (verified only after hash and recount checks) and presentation notes.

## 5. Verification done here (no inference)

`outputs/nl2mql_mql0010/test_results_package.txt`:
- **Package** (`tests/test_mql0010_package.py`):
  - manifest, SHA256SUMS and the ZIP are byte-identical to the tree;
  - registration A4 is preserved;
  - config = A4;
  - view and holdout exclusion in every plan;
  - no gold in the inputs;
  - four tamper cases are refused (data file; registration copy or registered source even with a re-hashed manifest; corpus bytes);
  - notebook structure, with no database or credential code;
  - provisioning guards.
- **Mocked notebook execution:** the real notebook cells, run with a fake model, a fake T4 probe and the **pinned tokenizer**:
  1. session 1 is killed after one batch, giving `partial`;
  2. session 1 is resumed from the hash-verified output: first passes and delta complete, with 104 delta generations;
  3. session 2 is resumed: `complete`, `verify_exports` ok, and D0V and D0G retried the same items.
- **Mocked scoring** (`cmd_score` with a fake client, preflight, reference and evaluator; it never reads gold results):
  - all 9 arms are scored;
  - the analysis is written;
  - a tampered export (one D0G record changed inside the zip) is refused.
- **Earlier suites:** components, corrections, registration and planner, including the 639/639 MQL-0009 reproduction.

## 6. Not done (deliberately)

- No predictions, no Kaggle launch, no Atlas connection or provisioning, no scoring.
- No ledger entry for MQL-0010. That is the next step after results, or as a dataset-audit entry if you ask.
- MQL-0008/0009 artifacts and approved corpus v2.1 are unchanged (tests assert the corpus and MQL-0009 package hashes).
