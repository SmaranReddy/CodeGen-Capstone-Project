# Kaggle run: zero-shot SQL baseline, Qwen2.5-Coder-3B-Instruct on Spider dev

**Status (2026-10-27):** the artifacts are prepared and validated locally. **The Kaggle run has not been done yet.**

This run evaluates the base model with no fine-tuning, on the full Spider dev set. The evaluation set is
`data/processed/spider_sql_v1/dev_full.jsonl`: 1,034 examples on 20 databases. The run does not train anything,
does not touch the 7B model, and does not use the test sets.

## Files

| Path | Role |
|---|---|
| `scripts/sql_v1/prepare_kaggle_dev_bundle.py` | Builds the private Kaggle input bundle from an explicit allowlist. It fails closed. |
| `scripts/sql_v1/spider_zs_sql.py` | Prompt, resumable generation log and scoring. It is copied into the bundle, and the notebook imports that copy. It also has a local CLI (`score`, `verify`). |
| `notebooks/kaggle/spider_sql_v1_zero_shot_3b.ipynb` | The Kaggle notebook. Its cells are tagged so the local test can run the non-GPU cells. |
| `scripts/sql_v1/make_kaggle_zs_notebook.py` | Regenerates the notebook with nbformat. Edit this file, not the `.ipynb`. |
| `tests/test_kaggle_sql_v1_baseline.py` | Local checks: no model, no GPU, no network. |
| `kaggle/spider_sql_v1_dev_bundle_v1/` and `kaggle/spider_sql_v1_dev_bundle_v1.zip` | The bundle you upload (disposable, rebuildable). |
| `data/raw/spider1/official_eval/LICENSE` | The Apache-2.0 text of taoyds/spider at commit `b7b5b8c`. It was added because redistributing the evaluator requires it. |

> `kaggle/` is not covered by `.gitignore`, and the bundle contains Spider data (106 MB of SQLite).
> Do not commit it.

### Bundle contents (30 files, 106.5 MB unpacked, 31.3 MB zip)

```
spider_sql_v1_dev_bundle_v1/
  bundle_manifest.json          sha256 + bytes + source of every file, counts, db_ids, provenance, exclusions
  ATTRIBUTION.md                Spider CC BY-SA 4.0 + evaluator Apache-2.0
  data/dev_full.jsonl           byte copy; sha256 1a9bacf3... = the spider-sql-v1 manifest value
  data/schemas_ddl_dev.json     CREATE TABLE statements of the 20 dev DBs (prompt schema)
  data/tables_dev.json          the 20 dev entries of the official tables.json (evaluator foreign-key maps)
  database/<db>/<db>.sqlite     the 20 dev databases, byte copies from spider_data/database/
  code/spider_zs_sql.py
  code/official_eval/{evaluation.py, process_sql.py, LICENSE, SOURCE_COMMIT.txt}
```

The bundle contains no train, val_internal, dev_clean, rejected or test files, no `test_database/`, no MQL data and
no Kaggle source data. The script never opens the test files. It refuses to run in any of these cases:

- a source hash differs from the pinned values or from the spider-sql-v1 manifest;
- the count is not 1,034, or the ids are not `spider_dev_0000..1033` in `dev.json` order;
- a dev database is missing;
- an output path matches the deny-list (`test*`, `train`, `val_internal`, `dev_clean`, `rejected`, `.env`);
- a credential-like string is found.

After writing, it compares the written tree and the zip against the allowlist.

## What the notebook does

**1. Environment.**
- Pins the run to GPU 0 and reports every GPU with its VRAM and compute capability.
- Picks bf16 only on capability 8.0 or higher. A T4 (7.5) gets **fp16**.
- Records package versions.
- Installs `transformers==4.46.3 accelerate==1.1.1 nltk==3.9.1` only if the image lacks transformers ≥ 4.43, accelerate ≥ 0.30 or nltk ≥ 3.8.

**2. Bundle.** It finds the bundle by its `bundle_manifest.json` under `/kaggle/input`, or uses `BUNDLE_DIR` if set. It then re-checks every file hash and the file set.

**3. Prompts.** Prompts are built from `{id, db_id, question}` and the schema DDL only. The notebook asserts that no gold SQL appears in any prompt, either verbatim or after whitespace and case normalisation, and then drops the gold records until scoring.

**3b. Scoring preflight (no gold, before the GPU is used).** This step exercises everything scoring needs on Kaggle's image and its read-only `/kaggle/input`. If any of it is broken, the run fails here rather than after generation.
- loads the official evaluator from the bundle;
- checks the nltk `punkt` data;
- reads the schema of all 20 DBs through the read-only shim;
- reads `wta_1` through the lenient decoder.

**4. Model.**
- `Qwen/Qwen2.5-Coder-3B-Instruct` at HF revision `488639f1ff808d1d3d0ba301aef8c11461451ec5`, with SDPA attention.
- The model's shipped `generation_config` (sampling defaults and `repetition_penalty` 1.05) is printed and then **replaced** with:
  - `do_sample=False, num_beams=1`;
  - `repetition_penalty=1.0`;
  - `max_new_tokens=256`, `max_time=240` s per batch;
  - EOS = `<|im_end|>` / `<|endoftext|>`.
- Seed 20261027.
- Left padding; batch size 8.

**5. Smoke test.** Runs the first 5 examples twice, writes `smoke/` and reports whether the two runs match. It stops if fewer than 4 of the 5 outputs are SELECT/WITH statements, which catches garbled fp16 output early.

**6. Generation.**
- Each batch is appended to `checkpoints/predictions.partial.jsonl` with flush + fsync.
- If a batch raises an error (for example OOM), it is retried one example at a time. An example that still fails gets a `status: "error"` record; failures are never dropped.
- Status per example:

  | Status | Meaning |
  |---|---|
  | `ok` | Generation finished normally. |
  | `empty` | No SQL could be extracted. |
  | `truncated` | Hit `max_new_tokens`. |
  | `timeout` | Hit `max_time`. |
  | `error` | Generation raised an error. |

- Each record also stores the raw output, extracted SQL, prompt and new token counts, latency (batch wall time / batch size), batch, session and run fingerprint.

**7. Scoring.** Happens after the model is freed. See *Metrics*.

**8. Summary.** Writes `baseline_summary.json`, `baseline_report.md` and `RUN_STATUS.json` (with output hashes).

### Prompt template (`spider-sql-zs-v1`, sha256 recorded in the outputs)

System:
```
You are an expert in SQLite. Given a database schema and a question, write one SQLite query that answers the question.

Rules:
- Use only the tables and columns that appear in the schema.
- Write a single read-only SELECT statement.
- Output only the SQL query: no explanation, no comments, no markdown.
```
User:
```
Database: {db_id}

Schema:
{CREATE TABLE statements from sqlite_master, separated by blank lines}

Question: {question}

SQL:
```
The tokenizer's chat template is applied with `add_generation_prompt=True`. SQL is extracted from the output in this order:
1. the first ```` ```sql ```` block, else the raw text;
2. a leading `SQL:` label is dropped;
3. the text is cut at the first `;` outside quotes;
4. newlines and tabs become spaces.

### Resume and run identity

- The **run fingerprint** hashes: model id + revision, dtype, prompt version + template hash, decoding settings, and the bundle manifest hash.
- `run_config.json` stores the fingerprint. Every log record carries it.
- On a re-run with a different configuration, the notebook refuses to continue and asks for a new `OUT_DIR`.
- `BATCH_SIZE` is not part of the fingerprint, so you can lower it after an OOM. Batch composition can shift greedy outputs slightly; each record stores its batch size.
- A torn last line (crash mid-write) is dropped with a warning. A duplicate id is an error.
- `predictions.jsonl` and the summary are written **only** when all 1,034 ids have a record. Otherwise `RUN_STATUS.json` says `incomplete` and lists the missing ids.

## Metrics

Every metric uses denominator **1,034**. Generation failures and empty outputs stay in the denominator and score 0.

| Metric | Definition |
|---|---|
| **Execution accuracy (official)** | `eval_exec_match` from taoyds/spider `evaluation.py` @ b7b5b8c, called per example exactly as its `evaluate()` loop does: parse, then the empty-SQL fallback, then `rebuild_sql_val/col`, then exec. |
| **Exact-set match (official)** | `Evaluator.eval_exact_match`, same loop. Values and DISTINCT are ignored, as in the official script. Cross-checked by running the official `evaluate(..., "match")` on the gold and prediction files; the two must agree to 3 decimals. |
| Official parser success | `process_sql.get_sql` accepts the prediction. |
| SQLite compile success | `EXPLAIN <pred>` succeeds on a read-only connection. |
| SQLite execution success | The prediction runs within 30 s and returns at most 1M rows. |
| Result-set match (**unofficial, secondary**) | Rows equal (ordered if the gold query has ORDER BY, otherwise as a multiset). |
| Generation | Status counts, failures, timeouts, truncations, mean latency per example (with its denominator), total generation wall time (summed over the batches that produced the final records; time spent on attempts superseded by a retry is not counted). |

**Wrappers around the official code.** These are disclosed in the report; the comparison logic is unchanged.

1. **SQLite connections.** The evaluator's `sqlite3.connect` is replaced with:
   - a **read-only** connection, because the original opens the DB read-write and a `DROP TABLE` prediction would damage it;
   - a **time limit**, because the original has none and one cross join on `wta_1` (104 MB) would hang;
   - a lenient UTF-8 decoder. `wta_1` contains non-UTF-8 bytes. Without the decoder, Python 3 fails **even gold-vs-gold** on `spider_dev_0455/0456`. The spider-sql-v1 builder uses the same decoder, and the test-suite evaluator uses `errors="ignore"`.
2. **Empty-prediction sentinel.** An empty or failed prediction is scored as `INVALID_EMPTY_PREDICTION`. The official `evaluate()` drops blank lines, which would misalign every later row. The sentinel neither parses nor executes.
3. **Skipped exec call.** The official exec call is skipped when our own timed run of the prediction already failed (error, timeout, or more than 1M rows). The official call would also return 0 in those cases: it catches errors, and a longer result cannot equal any gold result, since the largest gold result has 20,662 rows.

**Limitations.**
- The official EX at b7b5b8c is not the test-suite execution accuracy that most papers (including the Qwen2.5-Coder report) quote, so these numbers are not directly comparable to published ones.
- A correct query that the official parser cannot parse scores 0 on both EX and EM.
- The run is zero-shot: DDL only, no sample rows.
- Inference is batched with left padding.

## Local validation done (2026-10-27)

- The bundle was built and re-verified: 30 files, 1,034 records, 20 DBs. `dev_full.jsonl` is byte-identical to the
  processed file. The zip namelist equals the allowlist, and no `test`/`train`/`val`/`clean` names appear in it.
- **Gold-as-prediction oracle**, scored through the same code path: official EX, official EM, official parser,
  SQLite compile, SQLite exec and result-set match are each **1,034/1,034**.
  - The official `evaluate()` exact-match cross-check agrees (1.000).
  - The official hardness equals the stored `hardness` for all 1,034 records (easy 248 / medium 446 / hard 174 / extra 166).
- The notebook's non-GPU cells were run with a fake generator. Injected cases:
  - an empty output, an OOM on one batch (retried singly, then error, then retried on resume), a `max_time` stop, a `max_new_tokens` stop;
  - `DROP TABLE` (blocked by read-only; all DB hashes unchanged afterwards);
  - fenced multi-line output with tabs, non-SQL text, and a 3-way cross join on `wta_1` (time limit hit);
  - a simulated session loss after 12 batches, a torn log line, a configuration change (refused), and a resume.

  Result: exactly the 6 injected wrong answers score 0, every denominator is 1,034, `RUN_STATUS` is `complete`, and the `verify` CLI passes.
- The slowest gold query takes 1.24 s locally (`wta_1`), against the 30 s time limit.
- `pytest`: 229 passed, including 15 new tests.

## Step-by-step

### 1. Build the bundle on Windows

```powershell
cd C:\Users\Asus\OneDrive\Desktop\CodeGen\codegen-capstone
python scripts\sql_v1\prepare_kaggle_dev_bundle.py            # first time
# already built? either keep it, or rebuild:
python scripts\sql_v1\prepare_kaggle_dev_bundle.py --overwrite
python scripts\sql_v1\prepare_kaggle_dev_bundle.py --verify kaggle\spider_sql_v1_dev_bundle_v1
python -m pytest tests\test_kaggle_sql_v1_baseline.py -q
```
Expected: `records 1034, databases 20`, `OK ... 29 files` (plus the manifest), `15 passed`.
The upload file is `kaggle\spider_sql_v1_dev_bundle_v1.zip` (31.3 MB).

### 2. Create a private Kaggle Dataset

1. kaggle.com → **Create** → **New Dataset**.
2. Drag in `spider_sql_v1_dev_bundle_v1.zip`. Kaggle unpacks it; leave it unpacked.
3. Title: `spider-sql-v1-dev-bundle-v1`. Visibility: **Private** (the default; do not change it).
4. Click **Create**. Do not add or edit files in the dataset: the notebook rejects any file not in the manifest.

Use the zip through the web UI. `kaggle datasets create` on the folder would add a `dataset-metadata.json`, which
the hash check rejects.

### 3. Import the notebook

Kaggle → **Code** → **New Notebook** → **File** → **Import Notebook** → choose
`notebooks\kaggle\spider_sql_v1_zero_shot_3b.ipynb`. Set **Share** to Private.

### 4. Attach the dataset

Right panel → **Input** → **Add Input** → **Your Datasets** → `spider-sql-v1-dev-bundle-v1` → **Add**.
It appears under `/kaggle/input/spider-sql-v1-dev-bundle-v1/spider_sql_v1_dev_bundle_v1/`. The notebook finds it
on its own. If you attach a second copy, it stops; set `BUNDLE_DIR` to choose one.

### 5. GPU and internet

Right panel → **Settings**:

- **Accelerator: GPU T4 x2.** One card is used, in fp16. Avoid P100: current Kaggle PyTorch builds may not ship sm_60 kernels.
- **Internet: On.** This is needed to download the model (about 6 GB of safetensors) and possibly nltk `punkt_tab`. It requires a phone-verified Kaggle account.
- **Persistence: Files only.** Optional, for interactive sessions.

No secrets or HF token are needed: the model is public. Do not attach a `.env` file.

### 6. Run and save a version

1. Optional: run the first cells interactively, up to and including the smoke test, to check the setup.
2. Then use **Save Version** → **Save & Run All (Commit)**. Commit runs keep `/kaggle/working` as the version output.
3. Expected time on a T4 is roughly 20–45 min in total: download and load a few minutes, generation of 130 batches, scoring 1–2 min. The first run will give the real figure.
4. Afterwards, open the version and check the last cell's output for `RUN_STATUS: complete`.

**If the run stops partway:**
- **Same interactive session, or Persistence on:** run all cells again. Generation resumes from `checkpoints/predictions.partial.jsonl`.
- **Commit run:**
  1. Add the stopped version's output as input: **Add Input** → **Notebook Output** → this notebook.
  2. Set `RESUME_FROM = "/kaggle/input/<that-output>/spider_sql_v1_zs_qwen25coder3b_v1"`.
  3. Commit again.

  Keep every other setting unchanged, or the fingerprint check refuses to resume.

### 7. Bring the results back

1. Open the version → **Output** tab → **Download** (a zip of `/kaggle/working`).
2. Extract the zip into `codegen-capstone\outputs\spider_sql_v1_zs_3b\v1\`. `outputs/` is gitignored. Treat the zip as untrusted data and give it its own empty folder.
3. Verify against the same bundle:
   ```powershell
   python -I scripts\sql_v1\spider_zs_sql.py verify --bundle kaggle\spider_sql_v1_dev_bundle_v1 --outputs outputs\spider_sql_v1_zs_3b\v1\spider_sql_v1_zs_qwen25coder3b_v1
   ```
   The command checks that:
   - `RUN_STATUS` is `complete`;
   - the ids in both JSONL files equal the dev ids in order;
   - there is one fingerprint;
   - the hashes match;
   - the metrics recompute from `evaluation_results.jsonl`.

   It prints `"problems": []` on success.
4. Optional: re-score on Windows from the predictions alone, which is an independent check of the Kaggle scoring:
   ```powershell
   python -I scripts\sql_v1\spider_zs_sql.py score --bundle kaggle\spider_sql_v1_dev_bundle_v1 --predictions outputs\spider_sql_v1_zs_3b\v1\spider_sql_v1_zs_qwen25coder3b_v1\predictions.jsonl --out outputs\spider_sql_v1_zs_3b\v1\rescored_local
   ```
5. Copy the headline numbers into `docs/presentation_notes.md`.

### Expected paths

| What | Where |
|---|---|
| Bundle (Kaggle) | `/kaggle/input/<dataset-slug>/spider_sql_v1_dev_bundle_v1/` |
| Outputs (Kaggle) | `/kaggle/working/spider_sql_v1_zs_qwen25coder3b_v1/` |
| Outputs (repo) | `outputs/spider_sql_v1_zs_3b/v1/spider_sql_v1_zs_qwen25coder3b_v1/` |

The output folder holds:

| File or folder | Content |
|---|---|
| `predictions.jsonl` | One record per example, in dev order. |
| `evaluation_results.jsonl` | Per-example metrics and errors. |
| `baseline_summary.json` | Aggregate metrics and the configuration. |
| `baseline_report.md` | Readable summary and limitations. |
| `run_config.json`, `RUN_STATUS.json` | Run identity, completion status, output hashes. |
| `checkpoints/` | Append-only generation log and progress. |
| `official_eval/` | The official gold and prediction files, and the official printout. |
| `smoke/` | Smoke-test outputs (not part of the evaluation). |

## Troubleshooting

| Symptom | Fix |
|---|---|
| `expected one spider_sql_v1_dev_bundle ... found []` | The dataset is not attached (step 4), or it is still processing. |
| `found [a, b]` | Two copies are attached: remove one, or set `BUNDLE_DIR`. |
| `bundle file set differs` / `hash mismatch` | The dataset was edited or partially uploaded. Delete it and re-upload the zip. |
| `No CUDA GPU` | Settings → Accelerator → GPU T4 x2. |
| `no kernel image is available` | You are on a P100. Switch to T4. |
| HF download error / `LocalEntryNotFoundError` | Internet is off or the account is not phone-verified. Alternatively, attach an offline copy of the same model and set `LOCAL_MODEL_DIR`; the run identity then records the local path instead of the HF revision. |
| `LookupError: punkt_tab` | Turn internet on (nltk data download), then re-run. |
| `CUDA out of memory` | It is retried one example at a time automatically. If it repeats, set `BATCH_SIZE = 4` and re-run; that resumes. |
| `... belongs to another configuration` | You changed the model, dtype, prompt, decoding or bundle. Use a new `OUT_DIR`, or revert the setting. |
| `scoring preflight` error | The evaluator, nltk data or SQLite cannot run. Fix this before any GPU time is spent (usually: turn internet on for `punkt_tab`). |
| `smoke test: only k/5 outputs are SELECT` | Check the dtype line and the model files before spending GPU time. |
| `INCOMPLETE: N examples have no prediction` | The run stopped early. Re-run or resume (step 6). |
| `RUN_STATUS: needs_review` | The official `evaluate()` exact match disagrees with the per-example score. Do not report the numbers; keep `official_eval/` and investigate. |

## Final checklist

- [ ] `RUN_STATUS.json` → `"status": "complete"`, `n_predictions` = `n_evaluated` = **1034**
- [ ] `baseline_summary.json` → `complete: true`; every `metrics.*.denominator` = **1034**
- [ ] `generation.status_counts` sums to 1034; `generation_failures`, `timeouts` and `truncated_at_max_new_tokens` are noted
- [ ] `evaluation.official_cli_crosscheck.agree_to_3dp` = `true`; `evaluation.evaluator_errors` = 0, or each one is explained
- [ ] `config.model_revision` = `488639f1…`, `config.dtype` and `config.gpus` are recorded, `decoding.do_sample` = `false`
- [ ] `predictions.jsonl` has 1034 lines; every record has the same `fingerprint`
- [ ] The local `verify` command prints `"problems": []`
- [ ] The results are copied into `docs/presentation_notes.md`; the Kaggle dataset and notebook are still **Private**
