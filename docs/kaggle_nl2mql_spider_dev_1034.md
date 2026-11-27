# Run guide: MQL-0006, zero-shot Qwen2.5-Coder-3B-Instruct, NL→MQL on Spider dev (892 items)

**Status: COMPLETED and VERIFIED (2026-10-30).**
- **Generation:** session 1 produced 716 / 892. Session 2 resumed (§2b) and generated the other 176, 96 of them through the batch-size-1 OOM fallback.
- **Scoring:** done locally. 421 / 892 correct (47.2%).
- **Verification:** `all_checks_passed` and `complete`.
- Write-up: `docs/nl2mql_spider_dev_1034_model_comparison.md`.

**Scope (2026-10-30):** a single-model experiment, with **Qwen/Qwen2.5-Coder-3B-Instruct** as the only model. The
Groq-hosted `openai/gpt-oss-120b` comparison is **deferred** to a later experiment, so this run needs no Groq key
and makes no API call.

## Architecture: generate on Kaggle, score on Windows

```
Windows                         Kaggle (GPU, Internet)                     Windows
1. local Atlas preflight  ──►   2. Qwen generates predictions        ──►   3. verify + score on Atlas (read-only)
   (before any GPU use)            (no Atlas, no API, no secrets)          4. offline scoring verification
```

- **Kaggle only generates.** The notebook contains no Atlas client, URI, secret or pymongo, and no external
  model API, so no Kaggle secret is needed.
- **Atlas access happens only from your Windows machine**, using `MONGODB_URI` from your environment or `.env`.
  This is the read-only `codegen_eval_readonly` account that MQL-0001..0005 already used from this IP. `.env` is
  read, never written.
- **Evaluation set:** 892 items, 17 databases, 70 collections. It is Spider dev minus all 142 records on the 3
  databases of the held-out MQL test partition.
- **Bundle:** `kaggle/nl2mql_spider_dev_1034_bundle_v2.zip`, **unchanged and reused**:
  - sha256 `e9986dbef13cab8e1e047465b04ff8593d7b54f700cca4bcd1ed28f80d3b0760`, 32.0 MB;
  - manifest sha256 `3b587498443a10f3b385d5ceb595511d6b4fe4e67923a29f15a7f0a72100b4ac`.
  - Its bundled `code/dev1034_core.py` still contains an unused OpenAI-compatible client class from the earlier
    two-model plan. The Qwen-only notebook never imports or calls it, and leaving it there keeps the bundle hash
    unchanged.
- **Metric:** cross-engine execution accuracy, i.e. Qwen's MQL result on Atlas against the gold SQL result on
  SQLite. It is **not** native MQL exact match.

## 1. Windows, before Kaggle: preflight

```powershell
cd C:\Users\Asus\OneDrive\Desktop\CodeGen\codegen-capstone
$env:PYTHONDONTWRITEBYTECODE = "1"

python scripts/nl2mql/prepare_spider_dev_1034_bundle.py --verify kaggle/nl2mql_spider_dev_1034_bundle_v2

python scripts/nl2mql/local_atlas_scoring.py preflight
```

The first command should print
`OK kaggle/nl2mql_spider_dev_1034_bundle_v2: 892 records, 17 databases, 31 files; ...`.

The preflight checks:
1. The bundle's integrity.
2. All 892 gold SQL queries, re-executed on the bundled SQLite files and compared with the bundled reference.
3. Atlas:
   - ping and server version;
   - the account is **read-only**; it fails if any write or admin action is granted;
   - all **70** mapped collections are present;
   - exact counts;
   - a **full type-tagged row digest** of every collection against the bundled SQLite tables (`--quick` skips
     this).

The held-out databases are never read. The report goes to
`outputs/nl2mql_spider_dev_1034_model_comparison/preflight/atlas_preflight_<timestamp>.json`.

**Proceed only after it prints `PREFLIGHT OK`.**

## 2. Kaggle

1. **Datasets → New Dataset.** Upload `kaggle\nl2mql_spider_dev_1034_bundle_v2.zip` as **Private**. Do not
   upload `.env` or anything else; the notebook refuses to run if a `.env*` file is attached.
2. **Code → New Notebook → File → Import Notebook.** Choose
   `notebooks\kaggle\nl2mql_spider_dev_1034_model_comparison.ipynb`.
3. **Add Input** → the bundle dataset.
4. **Settings:**
   - Accelerator **GPU T4 x2** (one card is used);
   - **Internet on** (for the Hugging Face model download);
   - Persistence **Files only**;
   - **Pin to original environment**.
5. **No Groq API key and no MongoDB secret are required.** Add no secrets.
6. **Save Version → Save & Run All (Commit).**
7. In the version's log or `PREDICTIONS_STATUS.json`, confirm **`PREDICTIONS STATUS: complete`** with `"n": 892`
   and `"n_ok": 892`.
8. From the version's **Output** tab, download **`nl2mql_spider_dev_predictions.zip`**. Do not unzip it.

The notebook cells:

| Cell | What it does | What stops the run |
|---|---|---|
| env | Records package versions | A `.env*` file is attached |
| bundle | Verifies every hash, the 892 ids in order (`ids_sha256`), and that no record is on a held-out DB. **Kaggle decompresses nested `.gz` files on upload**: `data/reference.jsonl.gz` arrives as `data/reference.jsonl` (seen on the first run, 2026-10-30). The notebook copies the bundle to `/tmp/nl2mql_bundle_restored`, re-compresses the file as the builder did, and runs the strict check. If that zlib produces different gzip bytes, the file's content is checked against the 892 per-item digests and recomputed row hashes instead; every other file stays strictly hash-checked. `PREDICTIONS_STATUS.json` records which check was used (`bundle_check`, `bundle_restored_files`) | Anything differs, including tampered reference content |
| **preflight** | (1) Prompts re-derived from `(db_id, question)` equal the frozen `nl-mql-spider1-v1` prompts, and none contains gold SQL. (2) The model is `Qwen/Qwen2.5-Coder-3B-Instruct` at pinned revision `488639f1…`, with valid decoding limits. (3) A CUDA GPU is visible | Any failure → `PREDICTIONS_STATUS.json = preflight_failed`, and **no model is loaded** |
| qwen | Greedy decoding (`do_sample=False`), batch 8, `max_new_tokens` 1024. dtype comes from `QWEN_DTYPE = "auto"`: bf16 if `torch.cuda.is_bf16_supported()`, else fp16. **On Kaggle's T4 it selected bfloat16** (`run_config.json` `"dtype": "bfloat16"`); earlier versions of this guide wrongly said fp16. The checkpoint is written after every batch. **CUDA OOM:** the batch is released, then `gc.collect()` + `torch.cuda.empty_cache()`, and each of its prompts is retried alone with identical settings. A prompt that also runs out of memory alone is recorded as an error | A generation error is recorded and retried on resume. A crash keeps every checkpointed prediction. On resume, a configuration or fingerprint different from the resumed run stops the cell before anything is generated |
| package | Writes `PREDICTIONS_STATUS.json` (status, counts, hashes), `SHA256SUMS.txt` and the zip | — |

Each prediction record contains:
- `id`, `db_id`, `model`, `fingerprint`;
- `status`, `raw_text` (the MQL text) and `error`;
- `prompt_sha256`;
- tokens, latency and finish reason.

`run_config.json` holds the model id and revision, decoding settings, prompt and bundle hashes, dtype, GPU and
library versions.

**If the status is `partial`** (session timeout, crash or generation errors), resume as in §2b. Finished ids are
never regenerated. The fingerprint check refuses to mix runs that differ in settings, GPU type or library versions.

## 2b. Resume the partial run (716 done, 176 to generate)

The first session (2026-10-30) left 716 completed predictions and 176 CUDA-OOM errors. Its zip is
`C:\Users\Asus\Downloads\nl2mql_spider_dev_predictions.zip`: zip sha256 `9fae0d13…e4f9`, fingerprint
`184a7215a8a53394`, dtype `bfloat16`, Tesla T4, torch `2.11.0+cu128`, transformers `5.16.1`.

The updated notebook's resume path has been checked locally against an extracted copy of that zip:
- all hashes match and the fingerprint recomputes;
- 716 completed and 176 to generate;
- the checkpoint equals `predictions.jsonl`;
- the 716 lines round-trip byte-identically.

Steps:
1. Open the **same Kaggle notebook** that produced the partial run. **File → Import Notebook**, and choose the updated
   `notebooks\kaggle\nl2mql_spider_dev_1034_model_comparison.ipynb`.
2. Keep the private dataset `nl2mql-spider-dev-1034-bundle` attached. Do not re-upload it.
3. **Add Input → Notebook Output Files** → that notebook's partial version, the one whose output has
   `PREDICTIONS_STATUS.json` with `"n_ok": 716`.
4. In the **config** cell, change only these two lines:
   ```python
   RESUME_FROM = "auto"
   RESUME_EXPECTED_OK = 716
   ```
   `"auto"` finds the attached `nl2mql_spider_dev_predictions_v2/` under `/kaggle/input` and prints its path. If
   Kaggle also extracted the zip, the identical copy is accepted. If two *different* earlier outputs are attached,
   it refuses to guess: then set `RESUME_FROM` to the printed path of the right one.
5. Leave everything else unchanged:
   - `QWEN_MODEL_ID = "Qwen/Qwen2.5-Coder-3B-Instruct"`, `QWEN_REVISION = "488639f1ff808d1d3d0ba301aef8c11461451ec5"`;
   - `QWEN_DTYPE = "auto"`, which gives bf16 on the T4 as before;
   - `QWEN_BATCH_SIZE = 8`, `QWEN_MAX_NEW_TOKENS = 1024`.

   Settings: **GPU T4 x2**, Internet on, **Pin to original environment**, and no secrets. Torch, transformers,
   GPU type and dtype are all part of the fingerprint. If the environment changed, the qwen cell stops with
   `this session's configuration differs from the resumed run (earlier, now): {...}`. Do not edit anything to get
   past it; tell me what it lists.
6. **Save Version → Save & Run All (Commit).**
7. In the log, confirm:
   - `resume source verified: ... (fingerprint 184a7215a8a53394, 716 completed predictions kept, 176 to generate)`;
   - `Qwen: 176 of 892 to generate (...'dtype': 'bfloat16'...)`;
   - any `CUDA OOM on a batch of 8 ... retrying each prompt at batch size 1` lines;
   - at the end, `PREDICTIONS STATUS: complete`.

   In `PREDICTIONS_STATUS.json` → `models.qwen2.5-coder-3b-instruct.generation`, check:
   - `earlier_completed_kept_byte_identical: 716`;
   - `remaining_failures: 0`;
   - `completed_normal_batch` and `completed_batch_size_1_fallback`;
   - `oom_events_this_session`;
   - `generation_wall_s_total`.
8. From the new version's **Output** tab, download `nl2mql_spider_dev_predictions.zip`. Save it under a **new name**
   (for example `nl2mql_spider_dev_predictions_complete.zip`) so the partial zip in Downloads is kept unchanged.
   Then run §3 with that file.

**What to expect.**
- The 176 failed ids sit in the queue in their original order, so they form the same 22 batches of 8 as before.
  Many or all of those will run out of memory again. Up to 22 `CUDA OOM on a batch of 8` lines, and up to 176
  batch-size-1 predictions (about 20% of the set), are expected and do not mean anything is broken.
- The 716 earlier records are carried over byte for byte, so they do not carry the new per-record fields
  (`configured_batch_size`, `effective_batch_size`, `oom_recovery`, `oom_retries`). They were all generated in
  normal batches of 8, as `run_config.json` records, and the summary counts them as normal-batch.
- `oom_events_this_session` counts only the resumed session. The first session's 176 OOM error records are counted
  in `oom_error_records_earlier_sessions` and stay in `checkpoints/predictions.partial.jsonl`.
- When the results are written up, note that `mean_latency_s` mixes per-item latency from batches of 8 with
  single-prompt latency.

If a prompt runs out of memory even alone, the status stays `partial` with `remaining_failures` > 0. Do not score
it; report it, and it can be resumed again, or handled by a documented protocol change.

## 3. Windows, after Kaggle: verify and score

```powershell
python scripts/nl2mql/verify_kaggle_spider_dev_1034.py predictions --zip "$HOME\Downloads\nl2mql_spider_dev_predictions.zip"

python scripts/nl2mql/local_atlas_scoring.py score --predictions-zip "$HOME\Downloads\nl2mql_spider_dev_predictions.zip"

python scripts/nl2mql/local_atlas_scoring.py verify
```

**`predictions`** checks, offline:
- SHA-256 checksums;
- the bundle identity;
- exactly 892 expected ids, in original order, with no missing or duplicate ids;
- model id, revision, decoding and prompt hash;
- the run-config fingerprint;
- the prompt hash of every item;
- the recomputed status;
- only the Qwen model directory is present;
- no held-out database name, no credential, no test/train/split file.

**`score`** does the following:
1. Re-verifies the zip **before connecting to Atlas**, then imports it to `…/v1/predictions/`. Different content
   is never imported over existing content.
2. Re-checks the gold SQL on SQLite, plus Atlas privileges, coverage and counts. Add `--full-parity` to repeat the
   digest check.
3. Runs every prediction through:
   - the safety gate (`validate_mql` + `forbidden_anywhere` + cross-db, size and depth rules);
   - Atlas, read-only, with `maxTimeMS` 30 s and at most 25,000 rows;
   - the `exec_match` comparator, with the documented normalisation.
4. Writes `…/v1/scoring/`:
   - `models/qwen2.5-coder-3b-instruct/{evaluation_results.jsonl, prediction_rows.jsonl.gz, scoring_config.json,
     baseline_summary.json}`;
   - `RUN_STATUS.json`, `results_report.md` and `SHA256SUMS.txt`.
5. Runs the offline scoring verifier and writes `…/v1/scoring.verification.json`. The verifier covers:
   - checksums;
   - binding to these exact predictions;
   - gold SQL re-executed;
   - the safety gate re-run on every raw output;
   - DB mapping;
   - every verdict re-scored from the stored rows;
   - the summary recomputed;
   - no held-out content and no credentials.

Each outcome is reported separately:

| Outcome | Meaning |
|---|---|
| `correct` | Matches the gold result |
| `incorrect` | Executed but does not match |
| `invalid` | Format error, invalid MQL, or unsafe MQL |
| `timeout` | Exceeded the time limit |
| `failed` / model | Atlas rejected the query |
| `failed` / infrastructure | Generation error or Atlas unreachable |
| `not_comparable` | The reference cannot judge the result (tie at LIMIT, etc.) |

**Infrastructure failures are retried without regenerating predictions:** run the same `score` command again,
and only those items are re-executed.

Protection rules:
- A completely scored `v1/scoring/` is never scored over.
- A partial predictions zip is refused. `--allow-partial` scores it for the record under `v1/partial/<ts>/`, and
  it is never reported as a result.

**`verify`** re-runs the full offline verification of `v1/`.

## 4. Where the results go

- **Verified predictions:** `outputs/nl2mql_spider_dev_1034_model_comparison/v1/predictions/`
- **Scoring:** `outputs/nl2mql_spider_dev_1034_model_comparison/v1/scoring/`
  - `models/qwen2.5-coder-3b-instruct/baseline_summary.json` holds all metrics: correct/892, Wilson 95% CI,
    outcomes, validity and execution rates, difficulty breakdown, latency, tokens.
  - `results_report.md` is the presentation-ready table.
- **Verification:** `…/v1/scoring.verification.json` and `…/v1/predictions.verification.json`

After `verify` reports `all_checks_passed: true` and `complete: true`:
1. Copy the numbers into §8 of `docs/nl2mql_spider_dev_1034_model_comparison.md`.
2. Mark ledger MQL-0006 verified.
3. Update `docs/presentation_notes.md`.

## 5. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| Local preflight: `ServerSelectionTimeoutError` | Your current IP is not on the Atlas access list, or you are offline |
| `Atlas user is NOT read-only` | `MONGODB_URI` holds a writable account. Use `codegen_eval_readonly` |
| `<coll>: Atlas documents differ …` | Atlas data changed since 2026-10-20. Do not score; see §6 |
| Kaggle `no CUDA GPU is visible` | Set Accelerator to GPU T4 x2 |
| `predictions bundle failed verification` | The zip is not the notebook's untouched output. Re-download it |
| `predictions bundle is PARTIAL` | Resume on Kaggle (§2) until the status is `complete` |
| `different run fingerprint` | A setting, GPU type or library changed between versions. Use a fresh run |

## 6. Contingency: Atlas data missing (normally not needed)

**Never use `scripts/spider1/import_atlas.py`: it drops collections.**

```powershell
python scripts/nl2mql/seed_atlas_spider_dev.py                       # dry run, no network
$env:ATLAS_SEED_URI = "<a separate WRITABLE Atlas URI>"               # never the read-only MONGODB_URI
python scripts/nl2mql/seed_atlas_spider_dev.py --execute --only <missing collections>
Remove-Item Env:ATLAS_SEED_URI
```

The seed script:
- aborts if any target collection exists;
- never drops, overwrites or appends;
- verifies count and digest after inserting.

Re-run the local preflight afterwards.
