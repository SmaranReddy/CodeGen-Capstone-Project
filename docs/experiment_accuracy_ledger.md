# Experiment accuracy ledger

This is the permanent record of quantitative results: model evaluations, fine-tuning runs, RAG experiments and dataset audits. It was created on 2026-10-28.

**Rules**

- Add one entry per experiment. Give each a unique ID, never reuse an ID, and never delete or rewrite an earlier entry. If a result is corrected, add a dated note to the entry.
- Report every metric as `correct / total (percent)`, and name the dataset, split and evaluation protocol.
- Mark an entry **verified** only after its raw artifacts have been checked locally (hashes, record counts, metric recompute). Until then, mark it **provisional**.
- Keep results on different tasks, datasets or protocols in separate sections. Do not rank them against each other.
- The held-out `test_full` split (2,147) is evaluated once, at the end. Record a test result only when that run has actually happened.

## Current best results (directly comparable only)

| Task / dataset / split / protocol | Best entry | Official EX | Official EM |
|---|---|---|---|
| NL→SQL / spider-sql-v1 `dev_full` (1,034) / official Spider `evaluation.py` @ b7b5b8c | SQL-0001 (zero-shot 3B) | 446/1,034 (43.1%) | 378/1,034 (36.6%) |

Secondary metric, not used for ranking: SQL-0001 test-suite EX (e97acc5 defaults) is 630/1,034 (60.9%). See the SQL-0001 note (b).

Test-set (`test_full`) accuracy: **not evaluated yet.**
*Note 2026-11-25:* the NL→MQL test_full evaluation has now run: MQL-0015, the registered 1,000-item sample (see the NL→MQL table and the MQL-0015 entry). No NL→SQL test_full accuracy exists.

**Scope note 2026-10-29 (no entry, heading or metric changed).**
- The project's target task is **NL→MQL** (MongoDB). See `docs/target_language_scope_review.md`.
- The "primary"/"secondary" labels on sections A and B below reflect the 2026-10-27 framing and are kept as written.
- **SQL-0001 and every other section A result are SQL-only.** They are not MQL performance and are not comparable with any MQL result: different output language, metric, item count and runtime.
- The MQL zero-shot and RAG results (279 / 455 / Week 3) are still not back-filled into section B. Their source documents remain authoritative.

**NL→MQL current results (added 2026-10-29; each row is its own protocol, so rows are not ranked against each other or against section A).** Metric: strict execution accuracy on Atlas (`exec_match`), unanswered = wrong.

| Task / dataset / split / protocol | Entry | Qwen2.5-Coder-3B | Qwen2.5-Coder-7B | gpt-oss-120b |
|---|---|---|---|---|
| NL→MQL / v1 279 (Spider dev, authored gold MQL) / zero-shot, `nl-mql-spider1-v1`, Atlas exec vs gold result | MQL-0001 | 132/279 (47.3%) | 181/279 (64.9%) | 232/279 (83.2%) |
| NL→MQL / expanded 455 (Spider dev) / zero-shot, `nl-mql-spider1-v1`, Atlas MQL result vs SQLite gold-SQL result | MQL-0002 | 217/455 (47.7%) | 287/455 (63.1%) | 385/455 (84.6%) |
| NL→MQL / expanded train partition (318) / Qwen-3B GGUF q4_k_m on Kaggle T4, KV reset per item, fresh paired arms | MQL-0005 | zero-shot 160/318 (50.3%); random_k3 155/318 (48.7%); **hybrid_k3 172/318 (54.1%)** | — | — |
| NL→MQL / Spider dev minus the 3 held-out test DBs (892 items, 17 DBs) / zero-shot, `nl-mql-spider1-v1`, HF transformers bf16 on Kaggle T4, **cross-engine** (Atlas MQL result vs SQLite gold-SQL result), local Windows scoring (added 2026-10-30, verified) | MQL-0006 | 421/892 (47.2%), Wilson 95% 43.9–50.5% | — | — |
| NL→MQL / the same 892 items / Qwen-3B HF bf16 on Kaggle T4, paired arms generated in one run, cross-engine, local Windows scoring on Atlas (added 2026-11-17, verified) | MQL-0008 | zero-shot 415/892 (46.5%), Wilson 95% 43.3–49.8%; **hybrid RAG k=3 on approved corpus v2.1: 457/892 (51.2%)**, Wilson 95% 48.0–54.5%; exact McNemar p = 0.0018 | — | — |
| NL→MQL / 639 held-out Spider-train items (16 DBs disjoint from the corpus) / Qwen-3B HF bf16 on Kaggle T4, 8 paired runs in 3 sessions, cross-engine, local Windows scoring on Atlas `codegen_spider1_heldout` (added 2026-11-19, verified) | MQL-0009 | zero-shot 319/639 (49.9%); random k=3 (5 seeds) 334/335/334/333/328, mean 332.8 (52.1%); fixed k=3 334/639 (52.3%); **hybrid RAG k=3 on approved corpus v2.1: 357/639 (55.9%)**, Wilson 95% 52.0–59.7%; H1 hybrid − random mean +3.79 pp [1.28, 6.32], Holm p = 0.0079; H2 hybrid − fixed +3.60 pp, Holm p = 0.030 | — | — |
| NL→MQL / 716 held-out Spider-train items (18 corpus DBs, holdout view excludes them from retrieval) / Qwen-3B HF bf16 on Kaggle T4, 9 registered arms in 1 session, cross-engine, local Windows scoring on Atlas `codegen_spider1_holdout_mql0010` (added 2026-11-21, verified) | MQL-0010 | zero-shot A0 397/716 (55.4%); hybrid D0 413/716 (57.7%); **D0V (hybrid + schema-feedback retry) 430/716 (60.1%)**, D0V − D0 +2.37 pp [1.26, 3.63], p = 1.5e-5, +1 pp criterion met; **FULL 434/716 (60.6%)**, FULL − D0 +2.93 pp, Holm p = 0.011; generic retry D0G 414/716 | — | — |
| NL→MQL / the same 716 MQL-0010 holdout items, **secondary re-analysis (not a fresh test)** / Qwen-3B HF bf16 on Kaggle T4, R0V vs R3V regenerated in 1 matched session, cross-engine, Atlas `codegen_spider1_holdout_mql0010` (added 2026-11-23, verified) | MQL-0014 | R0 414/716 (57.8%); R0V 432/716 (60.3%); R3 438/716 (61.2%); **R3V (embedding-enhanced R3 + schema retry) 450/716 (62.8%)**, R3V − R0V +2.51 pp [0.42, 4.61], exact McNemar p = 0.030 (40/22) | — | — |
| NL→MQL / **Spider `test_full`, registered 1,000-item sample** (fresh test set, first use) / Qwen-3B HF bf16 on Kaggle T4, 2 sessions, cross-engine, Atlas `codegen_spider1_test_mql0015`, all arms scored in 1 session (added 2026-11-25, verified) | MQL-0015 | A0V 497/1000 (49.7%); R3V 544/1000 (54.4%); **FT-R3V (fine-tuned + R3 + schema retry) 598/1000 (59.8%)**, Wilson 95% 56.7–62.8%; FT-R3V − R3V +5.40 pp [2.7, 8.1], exact McNemar p = 0.00014 (125/71) | — | — |

MQL test-partition (65) accuracy for any new experiment: **not evaluated.** (The 455 aggregates in MQL-0002 predate the split and therefore include the 65 items; see MQL-0002.)

---

## A. NL→SQL (auxiliary track, SQL-only; headed "primary task" until 2026-10-29)

### SQL-0001: Qwen2.5-Coder-3B-Instruct, zero-shot, Spider SQL v1 dev baseline

**Status: verified (2026-10-28).** The local checks are listed below, and every one passed.

**Run**

| Item | Value |
|---|---|
| Run date | 2026-10-28 (run started 09:24:13 UTC, evaluation finished 09:35:43 UTC; from `run_config.json` and `baseline_summary.json`) |
| Platform | Kaggle notebook, batch run (`kaggle_run_type: Batch`) |
| Hardware | 1 × NVIDIA Tesla T4 (14.56 GB, compute capability 7.5) |
| Model | `Qwen/Qwen2.5-Coder-3B-Instruct` |
| Pinned revision | HF `488639f1ff808d1d3d0ba301aef8c11461451ec5` |
| dtype | float16 |
| Training | None (no fine-tuning) |
| Retrieval | None (no RAG, no few-shot examples) |
| Run fingerprint | `115bce1956f9389f` |
| Software | Python 3.13.15, torch 2.11.0+cu128, CUDA 12.8, transformers 5.16.1, accelerate 1.14.0, tokenizers 0.23.1, nltk 3.9.1, SQLite 3.45.1 |

**Data**

| Item | Value |
|---|---|
| Dataset | spider-sql-v1, `data/processed/spider_sql_v1/dev_full.jsonl` |
| Split | Development only (Spider 1.0 dev) |
| Size | 1,034 examples, 20 DBs |
| Hardness (official) | easy 248 / medium 446 / hard 174 / extra 166 |
| Bundle | `spider_sql_v1_dev_bundle_v1`, manifest sha256 `19a9f852…66d7c4` (matches the local bundle) |
| Held-out test | Not used, not packaged, not evaluated |

**Prompt and decoding**

| Item | Value |
|---|---|
| Prompt version | `spider-sql-zs-v1`, template sha256 `5ab3e34c…71d5f57` |
| Prompt content | System rules plus the user template `Database: {db_id}` / `Schema:` (CREATE TABLE DDL only, no sample rows) / `Question:` / `SQL:` |
| Chat template | Tokenizer default, `add_generation_prompt=True` |
| Decoding | Greedy: `do_sample=false`, `num_beams=1`, `repetition_penalty=1.0` (the shipped default of 1.05 is overridden) |
| Length limits | `max_new_tokens` 256; `max_time` 240 s per batch |
| EOS | `<|im_end|>` / `<|endoftext|>` (ids 151645, 151643) |
| Batching | Batch size 8, left padding |
| Seed | 20261027 |
| SQLite execution limit | 30 s per query, read-only connection |

The system prompt, verbatim:

> You are an expert in SQLite. Given a database schema and a question, write one SQLite query that answers the question.
>
> Rules:
> - Use only the tables and columns that appear in the schema.
> - Write a single read-only SELECT statement.
> - Output only the SQL query: no explanation, no comments, no markdown.

**Results (denominator 1,034 for every metric; dev split)**

| Metric | Result |
|---|---|
| **Execution accuracy (official `evaluation.py`)** | **446 / 1,034 (43.1%)** |
| **Exact-set match (official `evaluation.py`)** | **378 / 1,034 (36.6%)** |
| Official parser (`process_sql`) success | 573 / 1,034 (55.4%) |
| SQLite compile success (`EXPLAIN`) | 913 / 1,034 (88.3%) |
| SQLite execution success | 913 / 1,034 (88.3%) |
| Result-set match (unofficial, secondary) | 654 / 1,034 (63.2%) |
| Test accuracy | Not evaluated |

These are six different metrics. Only official EX and official EM are Spider accuracy. Parse, compile and execution success measure whether the query runs, not whether it is correct.

| Hardness | n | Official EX | Official EM |
|---|---:|---:|---:|
| easy | 248 | 188 / 248 (75.8%) | 187 / 248 (75.4%) |
| medium | 446 | 172 / 446 (38.6%) | 134 / 446 (30.0%) |
| hard | 174 | 62 / 174 (35.6%) | 52 / 174 (29.9%) |
| extra | 166 | 24 / 166 (14.5%) | 5 / 166 (3.0%) |

**Generation**

| Item | Value |
|---|---|
| Status | 1,034 / 1,034 `ok` |
| Generation failures | 0 |
| Timeouts | 0 |
| Truncated at `max_new_tokens` | 0 |
| Empty outputs | 0 |
| Finish reason | EOS for all 1,034 |
| SQL extraction | 674 raw, 360 from a markdown code fence (despite the "no markdown" rule) |
| Mean latency per example | 0.5518 s (batch wall time ÷ batch size, over 1,034 examples) |
| Total generation wall time | 570.56 s, in 130 batches, 1 session |
| Scoring time | 42.3 s |

**Verification (run locally on 2026-10-28)**

1. The documented command, from step 7 of `docs/kaggle_spider_sql_v1_baseline.md`, run from the repo root:

   ```powershell
   python -I scripts\sql_v1\spider_zs_sql.py verify --bundle kaggle\spider_sql_v1_dev_bundle_v1 --outputs outputs\spider_sql_v1_zs_3b\v1\spider_sql_v1_zs_qwen25coder3b_v1
   ```

   **Passed.** It printed `"problems": []` and exited with code 0. The metrics it recomputed equal those in `baseline_summary.json`.

2. `RUN_STATUS.json` reports `status: complete`, `n_expected = n_predictions = n_evaluated = 1034` and `problems: []`.
3. The SHA-256 of all 5 files in the manifest was recomputed: `predictions.jsonl`, `evaluation_results.jsonl`, `baseline_summary.json`, `baseline_report.md` and `run_config.json`. All 5 match.
4. All 13 downloaded files were copied through a staging folder. Their hashes are identical before and after the copy.
5. An extra, independent check script was run. These checks are not part of the repo verifier, and all of them passed:
   - **Record counts and ids**
     - There are 1,034 prediction records and 1,034 evaluation records.
     - No id is duplicated.
     - The ids are equal, in order, across predictions, evaluations and the dev file.
     - All ids are `spider_dev_*`, with no test, train or val ids.
     - `db_id` and gold SQL agree with the dev file.
     - The bundle's `dev_full.jsonl` is byte-identical to `data/processed/spider_sql_v1/dev_full.jsonl`.
   - **Run identity**
     - The fingerprint recomputes from `run_config.identity`.
     - Every prediction carries the same single fingerprint.
     - The bundle manifest hash and the prompt template hash match the local repo.
     - The system prompt and user template are identical to the repo's.
   - **Metrics**
     - All 6 aggregate counts recompute directly from the per-example fields.
     - The hardness breakdowns recompute.
     - Every EX=1 example also parses and executes.
     - There are 0 evaluator errors.
   - **Generation statistics**
     - The batch count (130), wall time (570.56 s) and mean latency (0.5518 s) recompute from `predictions.jsonl`.
   - **Official cross-check**
     - The official CLI printout (`official_eval/official_eval_match_stdout.txt`) gives overall exact match 0.366.
     - This agrees with the per-example value of 0.3656 (`agree_to_3dp: true`).
   - **Report**
     - Every count in `baseline_report.md` matches.
6. No discrepancy was found between the Kaggle notebook display, `baseline_summary.json`, `baseline_report.md` and the recomputed values.
7. The optional local re-score from predictions only (step 7.4 of the guide) was **not** run.

**Caveats**

- **Not comparable with published results.** Official EX at taoyds/spider b7b5b8c is not the test-suite execution accuracy reported in most papers, including the Qwen2.5-Coder report.
- **The official parser lowers EX and EM.**
  - 461/1,034 predictions fail `process_sql`, so they score 0 on official EX and EM.
  - 344 of those 461 execute, and 185 of them return the gold result set.
  - Known causes include column aliases (`AS Average_Age`) and table aliases without `AS` (`stadium s`). A rough regex count finds these in about 75 and 41 of the 344 executable failures. The full breakdown has not been analysed.
  - The unofficial result-set match (63.2%) is therefore well above official EX (43.1%). The unofficial match is secondary and is not a Spider metric.
- **Wrappers around the official evaluator.** These are disclosed in the guide and the report. The comparison logic is unchanged.
  - The SQLite connection is read-only and has a time limit.
  - A lenient UTF-8 decoder is used for `wta_1`.
  - An empty-prediction sentinel is used; it was needed 0 times in this run.
- **Many execution errors are wrong columns.** The 121 queries that fail to execute are exactly the 121 that fail `EXPLAIN`, so compile success equals execution success. The most common error is a wrong column reference, for example `no such column: T1.MakeId` (12 cases).
- **Batching can change outputs.** Batched greedy decoding with left padding can differ slightly from batch size 1.

**Artifacts** (`outputs/` is gitignored)

| What | Path |
|---|---|
| Run folder | `outputs/spider_sql_v1_zs_3b/v1/spider_sql_v1_zs_qwen25coder3b_v1/` |
| Predictions | `…/predictions.jsonl` |
| Per-example evaluation | `…/evaluation_results.jsonl` |
| Configuration | `…/run_config.json` |
| Summary | `…/baseline_summary.json` |
| Report | `…/baseline_report.md` |
| Run status and hashes | `…/RUN_STATUS.json` |
| Generation log, progress | `…/checkpoints/` (`predictions.partial.jsonl` is byte-identical to `predictions.jsonl`) |
| Official evaluator inputs and printout | `…/official_eval/` |
| Smoke test (not part of the evaluation) | `…/smoke/smoke_outputs.jsonl` |
| Kaggle model-download record | `outputs/spider_sql_v1_zs_3b/v1/__huggingface_repos__.json` (confirms commit `488639f1…`) |
| Guide | `docs/kaggle_spider_sql_v1_baseline.md` |

**Note 2026-10-28: local re-score and parser-rejection audit (diagnostic; the SQL-0001 metrics above are unchanged)**

Report: [`docs/sql_v1_zero_shot_parser_audit.md`](sql_v1_zero_shot_parser_audit.md). Dev data only; no inference was run.

- **Local re-score (Step 7.4) now run.** This supersedes item 7 of the verification list above.
  - All 6 metrics reproduce exactly from `predictions.jsonl`.
  - All 1,034 per-example records are identical on every field.
  - The only differences are CRLF line endings in `official_eval/*.txt` and nltk 3.9.3 locally vs 3.9.1 on Kaggle.
  - Output: `outputs/spider_sql_v1_zs_3b/v1/rescored_local/`.
- **Parser rejections.** The 461 rejections reproduce exactly (1,034/1,034 parse statuses), and all 1,034 gold queries parse. By cause:

  | Cause | n |
  |---|---:|
  | Genuine syntax error | 0 |
  | SQLite dialect incompatibility | 0 |
  | Invalid SQL for the schema (SQLite also rejects; mostly `no such column`) | 117 |
  | Alias / formatting (column alias, table alias without `AS`) | 94 |
  | Parser limitation (`INNER/LEFT JOIN` 217, multi-literal `IN` lists 26, others 7) | 250 |
  | Unclassified | 0 |

- **185 rejected predictions with a result-set match.** None is counted as correct.
  - On the 3 test-suite variants of the dev DBs, 24 return different rows from gold, so they are not equivalent.
  - 155 match on every variant.
  - 6 are on `wta_1` and cannot be checked.
  - For comparison, 36 of the 446 official EX hits are also distinguished on the variants.
- **Style sensitivity (diagnostic only, not a result).**
  - Some rewrites change only how the SQL is written: dropping column aliases, adding `AS` to table aliases, and `INNER JOIN` → `JOIN`.
  - Applied to rejected predictions, with the result rows checked to be unchanged, they would raise official EX from 446 to 604 (58.4%) and EM from 378 to 498 (48.2%).
  - A fine-tuning gain in official EX can therefore partly reflect learned Spider style. For SQL-0001 comparisons, report parser acceptance and a parser-independent execution metric alongside official EX and EM.
- **Audit artifacts:** `outputs/spider_sql_v1_zs_3b/v1/parser_audit/`, produced by `scripts/sql_v1/audit_zs_parser_rejections.py`.

**Note 2026-10-28 (b): test-suite execution accuracy added as a secondary metric (the SQL-0001 primary metrics above are unchanged)**

Report: [`docs/sql_v1_test_suite_eval.md`](sql_v1_test_suite_eval.md). Dev data only; the same 1,034 predictions (`eval_sql`); no inference was run.

**Status: verified (2026-10-28).**
- **Checks:**
  - gold as the prediction scores 1,034/1,034;
  - parity with the unmodified official evaluator;
  - a re-run gives byte-identical output;
  - all 695 DB files match the manifest sha256;
  - the SQL-0001 artifact hashes are unchanged.
- **Why this is a note, not a new ID.** It re-scores the SQL-0001 predictions under an added protocol; no new model run happened. This follows the parser-audit note above. A fine-tuned run gets its own entry and reports this metric there too.

| Metric (denominator 1,034) | Value | Protocol |
|---|---|---|
| **Test-suite EX** | **630 / 1,034 (60.9%)** | Official test-suite method, `taoyds/test-suite-sql-eval` @ e97acc5 with defaults (values predicted, DISTINCT removed); all 695 files of the 20 dev DBs |
| Test-suite EX, `--keep_distinct` | 603 / 1,034 (58.3%) | sensitivity only |
| TS EX by hardness: easy 248 / medium 446 / hard 174 / extra 166 | 208 (83.9%) / 286 (64.1%) / 86 (49.4%) / 50 (30.1%) | |

- **Does this reproduce the official test-suite methodology?** Yes for the decision rule: the official `result_eq`, `postprocess`, `remove_distinct` and the `value` → `1` step are imported unmodified. The disclosed deviations are:
  - read-only connections;
  - an enforced 60 s timeout (the official one cannot fire);
  - gold failures recorded rather than asserted.
  - None of them was triggered: 0 timeouts and 0 gold failures.
  - **Parity with the unmodified official `evaluation.py --etype exec`: matched.** It prints 0.839 / 0.641 / 0.494 / 0.301 / 0.609, which is exactly 208 / 286 / 86 / 50 / 630. One harness change was needed. On Windows, two earlier attempts stalled; one stack dump shows a blocked `accept()` inside a new `asyncio.run` event loop, and the root cause is not established. The bootstrap therefore reuses one event loop. The official files are unchanged. All 695 DB files are unchanged after its read-write run (`official_parity_run3/`).
- **Coverage.** 20/20 DBs and 695 SQLite files (20 originals + 675 variants; `wta_1` 33), all verified by sha256 manifest.
  - 1,034/1,034 examples are evaluable; 0 are missing.
  - Gold as the prediction scores 1,034/1,034.
  - A re-run gives byte-identical output.
- **Against official EX:**
  - 414 pass both;
  - 32 pass official EX only (31 coincidental single-DB matches; `spider_dev_0154` fails on the original DB once DISTINCT is removed, and on a variant with DISTINCT kept). All 32 fail under both settings;
  - 216 pass the test suite only (177 parser rejections and 39 parsed-but-EX-0);
  - 372 fail both.
  - Of the 185 parser-rejected predictions with a result-set match, 152 pass the test suite.
- **Role.** This is a secondary semantic metric, reported alongside official EX/EM. It does not replace them, and "Current best results" stays keyed to official EX/EM.
- **Artifacts:** `outputs/spider_sql_v1_zs_3b/v1/testsuite_eval_v1/` (`predictions/`, `predictions_keep_distinct/`, `gold_as_pred/`, `official_parity_run3/`, `comparison.json`; `official_parity/` and `official_parity_run2/` are stalled attempts), produced by `scripts/sql_v1/evaluate_test_suite_dev.py`.
- **Variant DBs:** extracted from `data/raw/spider1_testsuite_download/testsuite_database.zip` (sha256 `9ec24ea8…7f41da96`) to `%USERPROFILE%\codegen_data\spider1_testsuite_e97acc5\`, outside the repo.

---

## B. NL→MQL (primary task since 2026-10-29; headed "secondary task" until then; not comparable with section A)

The results here use different datasets and metrics (the Spider-derived MQL benchmarks of 279 / 455 examples, and the Week 3 RAG experiments). They have **not been back-filled** into this ledger yet. Until they are, the source documents remain authoritative:

- Week 2 model comparison: `docs/week2_report.md`, `docs/spider1_model_comparison.md`
- Week 3 RAG: `docs/week3_rag_report.md`

**Back-fill note 2026-10-29.**
- The project's primary task is NL→MQL (see `CLAUDE.md`, "Project scope"). On 2026-10-29 the section A and B headings were relabelled ("auxiliary, SQL-only" and "primary"); their 2026-10-28 wording is quoted inside each heading. This supersedes the scope note above, which said the headings were kept as written. No entry text was changed.
- Entries MQL-0001 to MQL-0005 below back-fill results that already existed. **No model was run, no output was regenerated and no Atlas query was executed for this back-fill.** The source documents remain the detailed write-ups.
- **What "verified" means for these entries.** On 2026-10-29 the following were recomputed locally from the raw artifacts, read-only:
  - record counts, unique ids, and id membership in the dataset or partition;
  - every `correct` count, from the per-item `result.correct` field in `executions.jsonl`;
  - dataset sha256 against the hash recorded at build time;
  - for MQL-0005, the Kaggle `SHA256SUMS.txt` (13/13 files OK).
  - Atlas re-execution of saved MQL was **not** rerun today. The entries cite the earlier re-execution logs and their dates. Under the ledger rule, the local checks above are what make an entry verified; the Atlas evidence is earlier-session evidence.
- **Two stale summary files** were found. Neither file was modified:
  - `outputs/evaluation_spider1/qwen2.5-coder-3b/summary.json` says 134/300. It covers the original 300-item sample, including 21 items later excluded from v1. On the 279 dataset ids, the same records give 132/279.
  - `outputs/evaluation_spider1_expanded/{qwen2.5-coder-3b,gpt-oss-120b}/summary.json` say 216 and 384. Both were written before the 2026-10-20 19:10 rescore (`rescore_log.json`), which changed `spider1_dev_0291` from `invalid_mql` to `correct` for both models (an empty pipeline is valid read-only MQL). The per-item records and `report.json` give 217 and 385, which are authoritative.

### MQL-0001: zero-shot Qwen-3B, Qwen-7B and gpt-oss-120b on the v1 279 benchmark

**Status: verified (2026-10-29 local recompute; Atlas re-execution 2026-10-15).**

| Item | Value |
|---|---|
| Run dates | Record timestamps: Qwen 3B and 7B 2026-10-10; gpt-oss-120b 2026-10-10 to 2026-10-15 (resumed for the last 68) |
| Evaluation method note | `exec_match` ignores column names and column order; row order is enforced only when the gold SQL has ORDER BY, with tie blocks allowed to permute |
| Task | NL→MQL, zero-shot (no retrieval, no training) |
| Dataset | `data/processed/spider1_nl_mql.jsonl`, 279 items, 20 Spider **dev** DBs, sha256 `e727b759…bdf7b0` (recomputed; matches `docs/week2_report.md`) |
| Split | Whole set. The 195/42/42 split in `data/splits/` is retrospective; every model had already seen all 279, so no held-out claim is made |
| Gold | Hand-authored gold MQL, validated on Atlas against the official gold-SQL result. Gold replay 279/279 |
| Models | Qwen2.5-Coder-3B-Instruct GGUF q4_k_m (HF rev `f74adce6`; GGUF sha256 `724fb256…30b7`); Qwen2.5-Coder-7B-Instruct GGUF q4_k_m (HF rev `13fb94bf`); `openai/gpt-oss-120b` via the Groq API (**no pinned revision available**: hosted model, version controlled by the provider) |
| Runtime / hardware | Qwen: llama-cpp-python on local CPU (no GPU). gpt-oss: Groq hosted |
| Prompt | `nl-mql-spider1-v1`, sha256 `7401bf14…c1be` (identical for all three) |
| Decoding | temperature 0, top_p 1.0, seed 42, `max_new_tokens` 1024 |
| Evaluation | `validate_mql` safety gate → execution on Atlas (`codegen_spider1`) → `exec_match` against the gold result (multiset; order checked only when the gold SQL has ORDER BY) |

| Metric (strict, n = 279) | Qwen-3B | Qwen-7B | gpt-oss-120b |
|---|---|---|---|
| **Execution accuracy** | **132/279 (47.3%)** | **181/279 (64.9%)** | **232/279 (83.2%)** |
| easy 71 / medium 126 / hard 44 / extra 38 | 47 / 74 / 7 / 4 | 58 / 98 / 16 / 9 | 64 / 107 / 37 / 24 |
| Malformed output | 2 | 3 | 8 |
| Execution failure | 15 | 4 | 5 |
| Result mismatch | 130 | 91 | 34 |

- Paired McNemar (from `week2_verification/verification.json`): 3B vs 7B p = 6.8e-10; 3B vs gpt-oss p = 4.5e-21; 7B vs gpt-oss p = 3.3e-08.
- The Qwen-3B hardness row was recomputed on 2026-10-29 from the 279 dataset ids. Its stored `summary.json` breakdown (47/75, 75/133, 8/48, 4/44) covers the 300-item file and must not be quoted.

**Verification**
- 2026-10-29, local:
  - 279 unique ids per model, equal to the dataset ids (3B: 300 records, of which exactly 279 are in the dataset);
  - correct counts recompute to 132 / 181 / 232;
  - one model id and one prompt hash per run.
- 2026-10-15, Atlas (`scripts/spider1/rescore.py --reexecute`, earlier session): every saved MQL was re-executed. Agreement is 279/279 for each model (`reexec_*.jsonl`, `agrees` field counted today), and gold replay is 279/279.
- 2026-10-20: re-verified as the read-only user (`codegen_eval_readonly`); see MQL-0002.

**Caveats**
- gpt-oss-120b is a large **open-weight** model on Groq, not a closed frontier model.
- Spider dev is public, and pre-training contamination of all three models is possible.
- Partial Gemini runs (`gemini-3.6-flash` 16–17 items, `gemini-3.8-flash` 9, `pilot_*` 12) were stopped by free-tier quotas. They are **not results**: they are excluded from all comparisons (`docs/spider1_model_comparison.md`).

**Artifacts:** `outputs/evaluation_spider1/{qwen2.5-coder-3b,qwen2.5-coder-7b,gpt-oss-120b,reference}/executions.jsonl`; `outputs/evaluation_spider1/week2_verification/`; `outputs/evaluation_spider1/comparison.{json,md}`; write-ups `docs/spider1_model_comparison.md`, `docs/week2_report.md`.

### MQL-0002: zero-shot Qwen-3B, Qwen-7B and gpt-oss-120b on the expanded 455 benchmark

**Status: verified (2026-10-29 local recompute; Atlas re-execution 2026-10-20).**

| Item | Value |
|---|---|
| Run dates | 279 v1 records reused (MQL-0001); 176 new items generated 2026-10-20 |
| Task | NL→MQL, zero-shot |
| Dataset | `data/processed/spider1_nl_mql_expanded.jsonl`, 455 items (279 v1 + 176 new), 20 Spider **dev** DBs, sha256 `59b6c1f3…d18134` (recomputed; matches `report.json`). Pool 493, 38 excluded (21 v1 adjudication + 17 NL/SQL audit) |
| Split | Whole set. The DB-disjoint 318/72/65 split (`data/splits_expanded/`) was made **after** these runs, so the 455 aggregates include the 65 test items. "Test partition unused" means no tuning and no new experiments on it; it does not mean these models never generated on those items |
| Gold | **No gold MQL** for the 176 new items. Scored as the model's MQL result on Atlas vs the gold-SQL result on SQLite, with per-table SQLite↔Atlas data parity as evidence (432 identical; 23 identical except 2 invalid-UTF-8 strings) |
| Models, runtime, prompt, decoding | As MQL-0001 (Qwen GGUF q4_k_m on local CPU; gpt-oss-120b on Groq; `nl-mql-spider1-v1`; temperature 0, seed 42, `max_new_tokens` 1024; Qwen `n_ctx` 8192) |
| Evaluation | `validate_mql` → Atlas → `exec_match`. Same comparator as MQL-0001 |

| Metric (strict, n = 455) | Qwen-3B | Qwen-7B | gpt-oss-120b |
|---|---|---|---|
| **Execution accuracy** | **217/455 (47.7%)** | **287/455 (63.1%)** | **385/455 (84.6%)** |
| easy 115 / medium 204 / hard 72 / extra 64 | 76 / 119 / 15 / 7 | 92 / 153 / 26 / 16 | 104 / 175 / 63 / 43 |
| v1 279 subset (consistency check) | 132 | 181 | 232 |

**Verification**
- 2026-10-29, local: 455 unique ids per model, equal to the dataset ids. The correct counts and hardness counts recompute from `executions.jsonl`, and the v1 subset reproduces MQL-0001 exactly.
- 2026-10-20, Atlas (`rescore_log.json`, earlier session):
  - full re-execution after the empty-pipeline scorer fix: 454/455 reproduced for 3B and gpt-oss (one change, `spider1_dev_0291` → correct) and 455/455 for 7B;
  - dry-run as `codegen_eval_readonly`: 455/455 for all three.
- The stale per-model `summary.json` values (216, 384) are explained in the back-fill note above.

**Artifacts:** `outputs/evaluation_spider1_expanded/{qwen2.5-coder-3b,qwen2.5-coder-7b,gpt-oss-120b}/executions.jsonl`, `report.{json,md}`, `rescore_log.json`, `reuse_manifest.json`; write-up `docs/spider1_expanded_benchmark.md`.

### MQL-0003: Week 3 RAG pilot, Qwen-3B on CPU, validation partition (72)

**Status: verified (2026-10-29 local recompute; Atlas re-execution 2026-10-23).**

| Item | Value |
|---|---|
| Run dates | 2026-10-20 / 21 |
| Task | NL→MQL with retrieved examples (RAG) |
| Data | `data/splits_expanded/validation.jsonl`, 72 items, 7 DBs (DB-disjoint from train and test) |
| Retrieval corpus | `data/rag/corpus_v1/` (sha256 `98dc8273…`): 280 examples from the train partition (**model-predicted MQL** that executed correctly; 265 gpt-oss, 15 Qwen-7B; not gold), 18 executed guide cards, 80 schema docs |
| Model / runtime | Qwen2.5-Coder-3B-Instruct GGUF q4_k_m (`f74adce6`), llama-cpp-python, local CPU |
| Prompt | `nl-mql-spider1-rag-v1` (sha256 `acb46214…`); zero-shot arm uses `nl-mql-spider1-v1` |
| Decoding | As MQL-0001 |
| Evaluation | As MQL-0002 |

| Arm (n = 72) | Correct |
|---|---|
| Stored zero-shot (from MQL-0002, validation ids) | 32/72 (44.4%) |
| `zs_rerun` (same session) | 33/72 (45.8%) |
| `random_k3` | 37/72 (51.4%) |
| **`hybrid_k3`** | **43/72 (59.7%)** |
| `hybrid_k3_g3` (+3 guide cards) | 41/72 (56.9%) |
| `hybrid_k3_fk` (exploratory; validation-informed) | 43/72 (59.7%) |

- hybrid_k3 vs zs_rerun: +13.9 points, McNemar p = 0.0525, Holm 0.16 (planned family). **Not significant.**
- llama.cpp reuses the KV prefix across calls, so outputs depend on run order. Compare only with same-session arms.

**Verification:** 2026-10-29, local: 72 unique ids per arm, all in the validation partition; correct counts recompute. 2026-10-23, Atlas (`outputs/rag_week3/rescore_check.json`): 72/72 reproduced for all five arms.

**Artifacts:** `outputs/rag_week3/{zs_rerun,random_k3,hybrid_k3,hybrid_k3_g3,hybrid_k3_fk}/executions.jsonl`, `outputs/rag_week3/report.{json,md}`; write-up `docs/week3_rag_report.md`.

### MQL-0004: Week 3 RAG, Qwen-3B on CPU, train partition (318)

**Status: verified for counts (2026-10-29 local recompute); the comparison it reports is superseded by MQL-0005.**

| Arm (n = 318) | Correct |
|---|---|
| Stored zero-shot (from MQL-0002, train ids) | 148/318 (46.5%) |
| `hybrid_k3` (completed 2026-10-23) | 171/318 (53.8%) |
| `random_k3` | **Incomplete: stopped at 143/318 records (65 correct) by the memory reaper. Not a result.** The user chose not to resume it |

- The reported +7.2 points (p = 0.003) compares a fresh RAG run with a **stored** zero-shot run from another session. The clean control (MQL-0005) shows that a fresh zero-shot run scores 160, so most of this gain was **baseline drift**. Do not quote +7.2 as the RAG effect.
- Atlas re-execution 2026-10-23: 316/318 reproduced for `hybrid_k3__train` (0667 wrong → correct, 0934 correct → wrong; the net count is unchanged). Both are ordering-instability items (see MQL-0005).
- Same model, runtime, prompt and corpus as MQL-0003. The train partition is a development partition: the hybrid weight was chosen on train leave-one-DB-out.

**Artifacts:** `outputs/rag_week3/hybrid_k3__train/`, `outputs/rag_week3/random_k3__train/` (partial), `outputs/rag_week3/report_train.{json,md}`.

### MQL-0005: Week 3 clean RAG control, Qwen-3B on Kaggle GPU, train partition (318)

**Status: verified (2026-10-29 local recompute + package hash check; Atlas re-execution 2026-10-24).**

| Item | Value |
|---|---|
| Run date | Generated 2026-10-23 (one Kaggle session); scored 2026-10-24 |
| Hardware / runtime | Kaggle, Tesla T4; llama-cpp-python 0.3.36, all layers on GPU, `n_ctx` 8192, **KV cache reset per item**, seed 42 |
| Model | Qwen2.5-Coder-3B-Instruct GGUF q4_k_m (same file as MQL-0003) |
| Data | `data/splits_expanded/train.jsonl`, 318 items |
| Package | `kaggle/week3_control_v1`, manifest sha256 `4b818cc4…8209`. Prompts were frozen on Windows and rebuilt identically on Kaggle (318/318 per arm) |
| Analysis plan | Frozen in the package before scoring. Primary contrast: hybrid_k3 vs random_k3. Confirmatory family of three, exact McNemar, Holm α = 0.05 |
| Evaluation | Generations scored locally on Atlas with the MQL-0002 harness |

| Arm (n = 318) | Correct | easy 84 / medium 140 / hard 45 / extra 49 |
|---|---|---|
| `zs_rerun` | 160/318 (50.3%) | 60 / 89 / 7 / 4 |
| `random_k3` | 155/318 (48.7%) | 58 / 88 / 8 / 1 |
| **`hybrid_k3`** | **172/318 (54.1%)** | 64 / 89 / 11 / 8 |

| Contrast | Diff | McNemar p | Holm p |
|---|---|---|---|
| hybrid_k3 vs random_k3 (primary) | +5.3 points | 0.021 | 0.064 |
| hybrid_k3 vs zs_rerun | +3.8 | 0.16 | 0.32 |
| random_k3 vs zs_rerun | −1.6 | 0.57 | 0.57 |

- **No contrast is significant after Holm.**
- Ordering sensitivity: `spider1_dev_0934` is a `$limit` tie. It is kept as correct, and with it counted wrong hybrid scores 171 (primary Holm 0.088). `spider1_dev_0667` sorts on a field that is missing from every document. With both counted wrong, hybrid scores 170 (Holm 0.12).
- Caveat: the train partition is not blind. The hybrid weight was tuned on it, and CPU results on it were already known.

**Verification**
- 2026-10-29, local: 318 unique ids per arm, all in the train partition; correct counts recompute; `SHA256SUMS.txt` in `import/week3_control_outputs/` checks 13/13 OK.
- 2026-10-24 (earlier session): `verification.json` passed (hashes, 5/5 determinism re-generations per arm, prompts rebuilt identically). Atlas re-execution reproduced 318 / 318 / 317 verdicts (the one change is 0934).

**Artifacts:** `outputs/rag_week3_clean/week3_control_v1/` (`{zs_rerun,random_k3,hybrid_k3}/executions.jsonl`, `report.{json,md}`, `verification.json`, `rescore_check.json`, `investigation_0934/`, `stability_audit/`, `import/`); write-up `docs/week3_rag_report.md` §7b.

### MQL-0006: zero-shot Qwen-3B (Kaggle GPU) vs Groq gpt-oss-120b on all 1,034 Spider dev questions

**Status: PLANNED: prepared 2026-10-29/30, not run. No metric exists.** Next unused ID confirmed: no MQL-0006 existed before this entry.

| Item | Value |
|---|---|
| Task / metric | NL→MQL. Cross-engine execution accuracy: generated MQL executed on Atlas `codegen_spider1` (read-only `codegen_eval_readonly`) vs Spider gold SQL executed on SQLite; denominator 1,034. **Not native MQL exact-match accuracy** |
| Data | `data/processed/spider_sql_v1/dev_full.jsonl`, 1,034 records (dev order), 20 DBs / 80 collections. Includes the 455-benchmark items and therefore the 65 MQL test-partition items (structural overlap; the partition file was not opened) |
| Models | `Qwen/Qwen2.5-Coder-3B-Instruct` @ HF `488639f1ff808d1d3d0ba301aef8c11461451ec5`, transformers on Kaggle GPU (fp16 on T4), greedy, `max_new_tokens` 1,024, batch 8. `openai/gpt-oss-120b` on Groq (**open-weight**, provider-hosted), temperature 0, top_p 1, seed 42, `max_tokens` 8,192, reasoning effort at the provider default |
| Prompt | `nl-mql-spider1-v1` (template sha256 `7401bf14…`), the same as MQL-0001..0005; zero-shot; frozen in the bundle (prompts sha256 `377cce08…`) |
| Evaluation | `validate_mql` + `forbidden_anywhere` (byte-copied harness) + extra cross-db/size rules → Atlas `maxTimeMS` 30 s, ≤ 25,000 rows → `spider1_kit.exec_match` unchanged. Outcomes: correct / incorrect / invalid / timeout / failed / not_comparable |
| Reference policy | Offline census: 988 `ok`, 44 `tie_at_limit` (mismatch → not_comparable), 2 `order_ties_unverified`, 0 `utf8_dependent`, 0 `reference_error` |
| Statistics (pre-specified) | Wilson 95% per model. Primary paired contrast frontier − Qwen on all 1,034: exact McNemar + paired bootstrap 95% CI (10,000, seed 20261029). Sensitivity: reference-ok 988; without the 38 prior-audit exclusions 996 |
| Package | `kaggle/nl2mql_spider_dev_1034_bundle_v1.zip` (sha256 `f6486f67…f3f2`, manifest `78edf259…9691`); notebook `notebooks/kaggle/nl2mql_spider_dev_1034_model_comparison.ipynb`; guide `docs/kaggle_nl2mql_spider_dev_1034.md`; write-up `docs/nl2mql_spider_dev_1034_model_comparison.md` |
| Artifacts (after the run) | `outputs/nl2mql_spider_dev_1034_model_comparison/v1/` (imported only by `scripts/nl2mql/verify_kaggle_spider_dev_1034.py --import` after every check passes) |

**Not comparable with:**
- MQL-0001 and MQL-0002: different item set, and CPU GGUF runtime for Qwen;
- MQL-0003 to MQL-0005;
- SQL-0001 (SQL-only).

**Open items before the run:**
- the Groq tier (about 1.67M tokens projected; the free tier needs about 8–9 daily sessions);
- the user's confirmation of the overlap with the 65 test-partition items;
- Atlas network access for Kaggle;
- the mentor's view on whether an open-weight model counts as "frontier".

**Note 2026-10-30: evaluation set corrected before any run (supersedes the Data, Reference policy, Statistics and Package rows above; nothing was run).**
- **Why.** The 1,034 dev records include the 65-item held-out MQL test partition, which must stay untouched until its one-time final evaluation.
- **How the held-out databases were identified.** The partition is database-disjoint. Its databases (`employee_hire_evaluation`, `pets_1`, `tvshow`) were read from the ID-only `data/splits_expanded/split_manifest.json`; `test.jsonl` was never opened. The manifest was cross-checked against the expanded dataset and mapped to dev ids through the pool's `dev_index`: all 65 map.
- **What is excluded.** At the user's choice, all **142** dev records on those databases. Dropping only the 65 questions would have kept 65 paraphrases with identical gold SQL.
- **Evaluation set: 892 records, 17 DBs, 70 collections**, original order.
  - Hardness: easy 214, medium 380, hard 148, extra 150.
  - Reference status: ok 846, tie_at_limit 44, order_ties_unverified 2.
  - Prior audit: 390 included, 34 excluded, 468 unaudited.
- **Primary metric:** correct / 892.
- **Sensitivity subsets:** reference-ok 846; without prior-audit exclusions 858.
- **Package:** `kaggle/nl2mql_spider_dev_1034_bundle_v2.zip` (sha256 `e9986dbe…0760`, manifest `3b587498…b4ac`). The v1 bundle, which contained the held-out DBs, was deleted unused.
- **Gold rows.** They are now shipped in the bundle (built with SQLite 3.40.1). Kaggle re-executes the gold queries with a tolerant re-check.
- **Network.** Atlas network access for Kaggle awaits user approval (run guide §0.1). M0 has no private endpoints or peering.
- `dev_full.jsonl`, the splits and every earlier artifact are unchanged.

**Note 2026-10-30 (b): local-scoring architecture (option C), before any run.** Status remains **PLANNED**.
- **Where each step runs:**
  - Kaggle generates predictions only: Qwen on the GPU, gpt-oss-120b on Groq. The notebook has no Atlas access.
  - SQLite-vs-Atlas scoring runs on Windows with `scripts/nl2mql/local_atlas_scoring.py`, as the read-only `codegen_eval_readonly`.
- **Commands:**
  - `preflight`: run before any Kaggle use;
  - `score`: verifies the downloaded `nl2mql_spider_dev_predictions.zip` first, imports it without overwriting, and scores it;
  - `verify`: offline.
- **Unchanged:** the evaluation set (892 / 17 DBs) and bundle v2 (sha256 `e9986dbe…0760`).
- **Results location:** `outputs/nl2mql_spider_dev_1034_model_comparison/v1/{predictions,scoring}/`.
- **When this entry may be marked verified:** only when `v1/scoring.verification.json` shows `all_checks_passed` and `complete`.

**Note 2026-10-30 (c): scope changed to a single model, before any run.** Status remains **PLANNED**.
- **Model:** MQL-0006 now evaluates only `Qwen/Qwen2.5-Coder-3B-Instruct` @ `488639f1…`, zero-shot, greedy, `max_new_tokens` 1,024, batch 8, on Kaggle GPU.
- **Deferred:** the Groq-hosted open-weight `openai/gpt-oss-120b` comparison moves to a later experiment, which gets its own ID. This supersedes the frontier rows, the paired-statistics row and the Groq open items above.
- **Notebook:** Qwen-only and generate-only. It needs no API key, no secret and no Atlas access.
- **Scoring:** local scoring is unchanged (`scripts/nl2mql/local_atlas_scoring.py`).
- **Unchanged:** the evaluation set (892 items / 17 DBs / 70 collections) and bundle v2 (sha256 `e9986dbe…0760`).
- **Primary metric:** cross-engine execution accuracy, correct / 892, with a Wilson 95% CI. There is no paired comparison.
- **Verification rule:** mark this entry verified only when `outputs/nl2mql_spider_dev_1034_model_comparison/v1/scoring.verification.json` has `all_checks_passed` and `complete`.

**Note 2026-10-30 (d): first Kaggle generation session was partial; dtype corrected.** Status remains **PLANNED**, and no metric exists.
- **Partial run:** `nl2mql_spider_dev_predictions.zip` (zip sha256 `9fae0d13…e4f9`; predictions sha256 `36ab616e…9fd8`; fingerprint `184a7215a8a53394`).
  - The predictions verifier passed all integrity checks, but the status is `partial`: 716 / 892 generated.
  - 176 are CUDA `OutOfMemoryError` on the 14.56 GiB T4: 22 batches of 8, mostly `dog_kennels` and `student_transcripts_tracking`.
  - Not scored; Atlas not contacted.
- **dtype correction:** the run used **bfloat16**, not fp16 as noted above. The notebook's `QWEN_DTYPE = "auto"` selected it because `torch.cuda.is_bf16_supported()` returned true on the T4. This is recorded in `run_config.json`. The resumed session keeps bf16, since it is part of the fingerprint.
- **Fix:** notebook only; bundle unchanged.
  - A batch that hits CUDA OOM is retried one prompt at a time with identical settings. A prompt that also runs out of memory alone is recorded as an error.
  - The resume source is verified before it is used, and the 716 completed predictions must be carried over byte-identically.
- **Disclosure:** the number of predictions generated through the batch-size-1 fallback goes in the final status and in this entry.

**Note 2026-10-30 (e): MQL-0006 COMPLETED and VERIFIED.** This note supersedes the PLANNED status above.

| Field | Value |
|---|---|
| Status | **verified**: `v1/predictions.verification.json` and `v1/scoring.verification.json` both show `all_checks_passed: true`, `complete: true`. Checks: checksums, binding to these predictions, gold SQL re-executed, safety gate re-run on every output, DB mapping, every verdict re-scored offline from the stored rows, summary recomputed, no held-out content or credentials |
| Model | `Qwen/Qwen2.5-Coder-3B-Instruct` @ HF `488639f1ff808d1d3d0ba301aef8c11461451ec5`, zero-shot |
| Data | Spider dev minus every record on the 3 held-out test DBs: **892 items, 17 DBs, 70 Atlas collections**; ids sha256 `21d1478b…0197`; bundle v2 `e9986dbe…0760` |
| Prompt / decoding | `nl-mql-spider1-v1` (template `7401bf14…`); greedy, `max_new_tokens` 1,024, configured batch 8, seed 42 |
| Hardware | Kaggle Tesla T4 (one card of T4 x2), **bfloat16**, torch 2.11.0+cu128, transformers 5.16.1, Python 3.13.15 |
| Generation | Two sessions, one fingerprint `184a7215a8a53394`. 716 in session 1. 176 in session 2, which resumed (the 716 carried over byte-identically): **796 in normal batches** (716 + 80), **96 through the batch-size-1 OOM fallback**; 12 batch OOM events, 0 single-prompt OOM, 0 remaining failures. Wall time 6,624.8 + 2,887.4 = **9,512.2 s**. Tokens in/out 1,109,483 / 95,891 |
| Scoring | Local Windows, Atlas `codegen_spider1` as read-only `codegen_eval_readonly`; `maxTimeMS` 30 s, 25,000-row cap; `exec_match`. Evaluation wall time 139.1 s; Atlas exec median 16 ms, p95 93 ms, max 30,047 ms |
| **Primary: cross-engine execution accuracy** | **421 / 892 (47.2%)**, Wilson 95% **43.9–50.5%**. Not native MQL exact match |
| Validity / execution | format valid 889/892 (99.66%); structurally valid 889/892 (99.66%); read-only validation passed 888/892 (99.55%); Atlas execution success 873/892 (97.87%); compared (correct + incorrect) 834/892 (93.5%); accuracy on compared 421/834 (50.48%) |
| Outcomes (sum 892) | correct 421 · incorrect 413 (wrong_row_count 230, wrong_column_count 111, wrong_values 71, order_mismatch 1) · invalid 4 (format_error 3, unsafe_mql 1) · timeout 2 · failed / model 13 (execution_error 10, plus 3 labelled `atlas_tier_restriction`, which are actually stages written without `$`, e.g. `{"match": …}`, rejected by Atlas with AtlasError 8000) · failed / infrastructure **0** · model generation failures **0** · not_comparable 39 (all `tie_at_limit` mismatches) |
| By hardness | easy 154/214 (71.96%, CI 65.6–77.6) · medium 225/380 (59.21%, 54.2–64.0) · hard 27/148 (18.24%, 12.9–25.2) · extra 15/150 (10.0%, 6.2–15.8) |
| Secondary (BSON-normalised) | 421/892 (47.2%), identical to the primary |
| Latency | per item, amortised over the batch: mean 10.43 s, median 9.42 s, p95 19.09 s, max 49.49 s. It mixes batch-8 and batch-1 items |
| Outcome by generation path (descriptive only, not a controlled comparison: the fallback items are concentrated on long-schema DBs) | session 1, batch 8: 348/716 · session 2, batch 8: 34/80 · batch-1 fallback: 39/96 |
| Artifacts | `outputs/nl2mql_spider_dev_1034_model_comparison/v1/predictions/` (imported from `nl2mql_spider_dev_predictions__.zip`, sha256 `86a8202a…5c98`); `v1/scoring/` (`models/qwen2.5-coder-3b-instruct/{baseline_summary.json, evaluation_results.jsonl, prediction_rows.jsonl.gz, scoring_config.json}`, `RUN_STATUS.json`, `results_report.md`, `SHA256SUMS.txt`); `v1/predictions.verification.json`; `v1/scoring.verification.json`. The partial first-session zip (`9fae0d13…e4f9`) was kept unmodified and not scored |
| Limitations | Cross-engine equivalence on one database instance; depends on the mapping and normalisation. Gold SQL not human-verified for every item; 468 items never audited. 96 items were generated at batch size 1, so minor numerical variation versus batch 8 is possible. Spider dev is public (contamination possible). `results_report.md` and `baseline_summary.json` give `generation_wall_s` = 2,887.4 s, which is **session 2 only**; the total is 9,512.2 s (`PREDICTIONS_STATUS.json` → `generation.generation_wall_s_total`) |
| Comparability | **Not directly comparable** with MQL-0001 (279, authored gold MQL, CPU GGUF), MQL-0002 (455, CPU GGUF, includes the 65 test items) or MQL-0005 (318 train partition, GGUF): different item sets, runtimes and in part reference type. The similar percentages (47.3% / 47.7% / 47.2%) are a coincidence of different protocols, not a replication. SQL-0001 is SQL-only and is not shown beside this result |

### MQL-0008: zero-shot vs hybrid RAG on the approved corpus v2.1, Qwen-3B (Kaggle GPU), 892 items

**Status: verified 2026-11-17** (export hashes, 892 unique records per arm, per-record recount, scoring checksums 13/13, analysis recomputed offline with identical output).

| Item | Value |
|---|---|
| Task / metric | NL→MQL. Cross-engine execution accuracy: generated MQL executed on Atlas `codegen_spider1` (server 8.0.37, read-only `codegen_eval_readonly`, no write or admin privilege) vs Spider gold SQL on SQLite; correct / 892, nothing excluded. **Not native MQL exact match** |
| Model | `Qwen/Qwen2.5-Coder-3B-Instruct` @ `488639f1ff808d1d3d0ba301aef8c11461451ec5`, bfloat16, greedy, max_new_tokens 1024, batch 8, seed 42 (`resolved_revision` is null in the run config: loaded from the pinned local snapshot) |
| Hardware / libraries | Kaggle Tesla T4; torch 2.11.0+cu130, transformers 5.20.2, accelerate 1.15.0, tokenizers 0.23.3. Generation 2026-11-16 18:51 to 2026-11-17 00:41 UTC (arm A 8,840 s, arm B 12,199 s) |
| Evaluation set | the 892 MQL-0006 items (17 databases); prompt `nl-mql-spider1-v1`; RAG prompt `nl-mql-spider1-rag-v1` |
| Arm B retrieval | approved corpus v2.1, 3,445 records, sha256 96552f061afba13792d75240c18001e87e70f4aebf9d0934dfe27c0a7dd199bf (AI-reviewed, not human-reviewed; labels never executed on Atlas). hybrid_masked: BM25 k1 1.5, b 0.75, pattern weight 1.0, K 3, 1,500-token cap, target database excluded; 3 examples for all 892 items |
| Analysis plan | registered 2026-11-16 before any prediction: sha256 985b2da6732c7222d20370c607c31b7b8e72b85b5bee2d2f2b2debc126604eb6 |

**Results (scored 2026-11-17 05:35 to 05:40 UTC)**

| Arm | correct / 892 | % | Wilson 95% | incorrect | invalid | execution_error | timeout | generation_failure | infrastructure_failure | not_comparable |
|---|---|---|---|---|---|---|---|---|---|---|
| A zero-shot | 415 / 892 | 46.52 | 43.27–49.81 | 414 | 4 | 13 | 2 | 0 | 0 | 44 |
| B hybrid RAG | 457 / 892 | 51.23 | 47.95–54.50 | 365 | 3 | 29 | 1 | 0 | 0 | 37 |

- **Primary comparison (B vs A, paired):** +42 items, +4.71 percentage points; paired bootstrap 95% interval +1.91 to +7.62 points (10,000 resamples, seed 20261029); RAG fixes 108 baseline errors and introduces 66; exact two-sided McNemar p = 0.0018. Hybrid RAG is better than zero-shot at the 0.05 level.
- **Exploratory (no claim):** by hardness A / B: easy 154 / 163 of 214, medium 221 / 243 of 380, hard 25 / 30 of 148, extra 15 / 21 of 150. Without the two near-duplicate items (890): +42, p = 0.0018. Reference status ok (846): 413 vs 448, p = 0.0083. Execution errors rise from 13 to 29 in arm B.
- **Zero-shot differs from MQL-0006 / MQL-0007 (421) with identical predictions.** Arm A's 892 outputs are text-identical to the MQL-0007 arm A outputs, but 6 items changed outcome (5 correct → not_comparable, 1 correct → incorrect). All 6 are `$sort` + `$limit` queries whose sort key has ties: Atlas returned a different tied row on 2026-11-17 than on 2026-11-07. This is server-side tie-order nondeterminism, not a scorer or model change. Both MQL-0008 arms were scored in one session with the same evaluator, so the paired comparison is unaffected in design; single-arm accuracies carry a few items of run-to-run noise.
- **Comparability:** A and B are directly comparable (same run, model, items, scorer). MQL-0006 (421) and MQL-0007 (hybrid RAG on corpus v1: 437) are historical, scored on another day; they are context, not arms of this experiment. MQL-0007 has no ledger entry of its own yet.
- **Artifacts:** export `outputs/mql0008_predictions.zip` (sha256 2e4774f2117e480eba3b997e33ac209475794376f483561c85db77017192f65c); run `outputs/nl2mql_mql0008_approved_corpus_v2_1/scoring_runs/run_20261117T053551Z/` (`input_validation_report.json`, `predictions/`, `scoring/results_report.md` sha256 1daa8f71eceeddab1cf011f5423bd70a4d0ae1638229ab223513a25bfb88bf47, `scoring/analysis.json`, `scoring/paired_comparison.jsonl`, `scoring/arms/<arm>/evaluation_results.jsonl`, `scoring/RUN_MANIFEST.json`); package `kaggle/nl2mql_mql0008_approved_corpus_v2_1/`; scorer `scripts/nl2mql_mql0008/`.
- **Not checked:** no executed Kaggle notebook or Kaggle log is saved locally, so model, GPU and library identity rest on the export's own run configs. Scoring was run once (no repeat to measure tie-order noise). Test partition (65): not evaluated.

### MQL-0009: retrieval-control experiment (zero-shot / random x5 / fixed / hybrid RAG), Qwen-3B, 639 held-out Spider-train items

**Status: PLANNED (registered 2026-11-18). The package is built and has not been executed. No prediction and no metric exist. Scoring is BLOCKED on Atlas data provisioning.** Next unused ID confirmed: MQL-0007 exists in outputs/ without its own entry, and MQL-0008 is the latest entry. No file used MQL-0009 before this entry.

| Item | Value |
|---|---|
| Question | Is the MQL-0008 hybrid-RAG gain due to relevant retrieval or to supplying any three demonstrations? |
| Runs | A_zero_shot; B_random_s1..s5 (unchanged `retrieval.Retriever(method="random")`, seeds 20261181..20261185); C_fixed (spider_train_02033, 01967, 03829 for every question); D_hybrid (the unchanged MQL-0008 hybrid_masked retriever). K=3 for B, C and D. 8 runs x 639 = 5,112 predictions |
| Model / decoding | `Qwen/Qwen2.5-Coder-3B-Instruct` @ `488639f1ff808d1d3d0ba301aef8c11461451ec5`, bfloat16 (emulated on a Tesla T4, compute capability 7.5, as in MQL-0006..0008; only a T4 is accepted), greedy, max_new_tokens 1024, batch 8, seed 42 |
| Evaluation set | spider-sql-v1 `val_internal` database-level split (16 Spider-train DBs, seed 20261027) intersected with spider-pool-v3 membership = 639 items; disjoint from all 123 corpus DBs and from Spider dev |
| Corpus | approved corpus v2.1, 3,445 records, sha256 96552f06...dd199bf, unmodified; the eligible view = the full corpus; 17 question-level exclusion pairs, applied identically in B, C and D |
| Prompt | `nl-mql-spider1-v1` / `nl-mql-spider1-rag-v1`; schema context derived offline from SQLite with the import_atlas.py mapping |
| Package | `kaggle/nl2mql_mql0009_retrieval_control_bundle.zip` sha256 2ed81c46874c828b139b0eac87ea087f9b0a2f97396fac9307fd9319a71515ce; manifest 2a4f22e0eb526540e70f62f561d0314b1b2faa5c12917196a953d9f98aecface |
| Analysis plan | `outputs/nl2mql_mql0009_retrieval_control/preflight/analysis_plan.json` sha256 ecb7479f7909a65f726fadce1e49dda4f6cc472cbcf9c2eaa3f2d59eb2bb4144. Primary: H1 D vs per-question mean of 5 random seeds (bootstrap + sign-flip), H2 D vs C (McNemar), Holm |
| Blocker | the 16 DBs (69 collections, 16,885 documents) are not on Atlas; scoring requires provisioning plus full parity with `outputs/nl2mql_mql0009_retrieval_control/scoring_contract/atlas_mapping.json` |
| Details | `docs/nl2mql_mql0009_retrieval_control.md` |

- **Not comparable** with MQL-0008 absolute accuracies (different item set). Only within-run paired comparisons are planned.
- 2026-11-18, status note (no metric): **PARTIAL, unscored.**
  - Kaggle session 1 ran as registered: A_zero_shot, D_hybrid and B_random_s1, 639/639 each, so 1,917 of 5,112
    predictions. Pinned revision, bf16 emulated on a T4, torch 2.11.0+cu130 / transformers 5.20.2, environment
    consistent, no problems. Export: `mql0009_predictions.zip`, predictions sha256 A `1ed94e2c…`, B_random_s1
    `c6ba66fa…`, D `e7536b6b…`.
  - Sessions 2 (C_fixed, B_random_s2/s3) and 3 (B_random_s4/s5) have not been run. The registered scorer refuses
    partial exports.
  - The Atlas provisioning blocker above is resolved: `codegen_spider1_heldout` holds 69 collections and 16,885
    documents (`outputs/nl2mql_mql0009_retrieval_control/provisioning/provision_20261118T040504.json`).
  - No accuracy is recorded until all 8 runs are scored. Continuation: `docs/mql0009_continuation_runbook.md`.

**Status update 2026-11-19: COMPLETE and VERIFIED.** The PLANNED / PARTIAL text above is historical and is left as written.

- **Generation:** Kaggle sessions 2 (C_fixed, B_random_s2/s3) and 3 (B_random_s4/s5) were run as registered. Exports are at the repo root:
  - `mql0009_predictions.zip` (session 1), sha256 e982a47d…;
  - `mql0009_predictions_session2.zip` (session 2), 2933da0a…;
  - `mql0009_predictionssession2.zip` (**session 3**, misnamed at download), bc3d2b9e….
- **Integrity checks:**
  - 8 × 639 = 5,112 records, each with one unique id per run, and every record `status: ok`.
  - `input_validation_report.json` reports `ok: true`.
  - The arms re-exported in later cumulative zips (A, B_s1 and D in sessions 2 and 3; B_s2/s3 and C in session 3) are byte-identical to their first export, so nothing was regenerated.
  - All 8 runs used the pinned revision 488639f1…: T4, bf16 emulated, torch 2.11.0+cu130, transformers 5.20.2.
  - Frozen hashes match: package manifest 2a4f22e0…, config 3f052e98…, analysis plan ecb7479f…, evaluation ids 6f1425f7….
- **Scoring:** registered command, `mql0009_scoring.py score --run-name run_final`.
  - The first attempt stopped at the Atlas preflight because the read-only user had no `find` on `codegen_spider1_heldout`. The user then added the grant and twice sent the registered command as a chat message; the shell prefix did not take effect either time. Claude took this as authorization and ran the command in-session on 2026-11-19, departing from the task brief's no-Atlas-scoring restriction.
  - There was one fail-closed full-parity preflight for all 8 runs: Atlas 8.0.37, 69 collections, 16,885 documents, digests match, read-only verified. `scoring/preflight/atlas_check.json` sha256 af4ae6f6….
  - The reference has 639 items, all `ok`, sha256 ce1a915b….
  - The per-arm counts and H1 were recomputed independently from `evaluation_results.jsonl` and `paired_per_question.jsonl` with identical results.

| Run | correct / 639 | % | incorrect | invalid | execution_error |
|---|---:|---:|---:|---:|---:|
| A_zero_shot | 319 | 49.92 | 303 | 4 | 13 |
| B_random_s1 | 334 | 52.27 | 285 | 3 | 17 |
| B_random_s2 | 335 | 52.43 | 280 | 5 | 19 |
| B_random_s3 | 334 | 52.27 | 283 | 4 | 18 |
| B_random_s4 | 333 | 52.11 | 282 | 3 | 21 |
| B_random_s5 | 328 | 51.33 | 281 | 5 | 25 |
| B_random mean | 332.8 | 52.08 | | | |
| C_fixed | 334 | 52.27 | 289 | 2 | 14 |
| D_hybrid | 357 | 55.87 | 263 | 2 | 17 |

No timeouts, generation failures, infrastructure failures or not_comparable outcomes in any run.

- **H1 (primary, registered):** D_hybrid − per-question mean of the 5 random seeds = +3.79 pp.
  - Question-level bootstrap 95% [+1.28, +6.32]; sign-flip p = 0.00395; Holm p = 0.0079.
  - Hybrid is better on 128 questions and the random mean is better on 80.
  - **Supported:** the hybrid gain is not explained by supplying any three demonstrations.
- **H2 (primary, registered):** D_hybrid − C_fixed = +3.60 pp, bootstrap 95% [+0.63, +6.73]; exact McNemar p = 0.0297 (D only 63, C only 40); Holm p = 0.030. **Supported.**
- **Secondary (unadjusted unless stated):**
  - D vs A: +5.95 pp, 86 fixes / 48 regressions, McNemar p = 0.0013.
  - Random mean vs A: +2.16 pp, bootstrap [−0.41, +4.73], sign-flip p = 0.108, **not significant**.
  - C vs A: +2.35 pp, p = 0.159, **not significant**.
  - D vs each seed: p = 0.033 / 0.033 / 0.031 / 0.026 / 0.0046. Holm over the 5 seeds leaves only s5 below 0.05 (adjusted 0.023; the others are 0.106).
  - Random-seed SD 0.43 pp, but 182 of 639 questions change outcome across seeds.
- **Exploratory:** set_op_or_subquery (91 items) is about 11% in every run; anti-join (18) is 33% zero-shot vs 10–11% with random or hybrid examples. See `docs/mql0009_post_experiment_analysis.md`.
- **Comparability:** all 8 runs are directly comparable (same items, model, scorer and Atlas snapshot). The absolute accuracies are **not** comparable with MQL-0008 (different items: Spider-train DBs, not Spider dev).
- **Artifacts:** `outputs/nl2mql_mql0009_retrieval_control/scoring_runs/run_final/`, containing:
  - `scoring/results_report.md` 8730abf3…;
  - `scoring/analysis.json` c0bc45ad…;
  - `scoring/paired_per_question.jsonl` cf594568…;
  - `scoring/RUN_MANIFEST.json` 352631ea…;
  - `scoring/arms/<run>/`.

### MQL-0010: schema-feedback retry, signature diversity and FK type notes (9 arms), Qwen-3B, 716 held-out Spider-train items

**Status: verified (2026-11-21).**

**Checks run:**
- archive verify (`mql0010_scoring.py verify`: SHA256SUMS, frozen package manifest 87bea695…, registration index 11b56c79…, frozen plans, `verify_exports`);
- full-parity read-only Atlas preflight;
- 0 infrastructure failures;
- recount of correct per arm equals `analysis.json`;
- 716 evaluation records per arm;
- prediction hashes equal `RUN_MANIFEST.json`;
- post-hoc configuration check 60/60 (`posthoc_config_check.json`):
  - every pass fingerprint, recomputed from the frozen package, matches;
  - config_sha256 0f80cb87… equals the frozen config;
  - D0G's generic text is verbatim R5;
  - batch size is 1 for delta and retries.

**Setup:**
- **Dataset file:**
  - gold from `data/processed/spider_sql_v1/train.jsonl` (sha256 40c1a7c1…);
  - split `kaggle/nl2mql_mql0010_holdout/registration/split_registration.json` (sha256 fd0c8e18…);
  - evaluation inputs `kaggle/nl2mql_mql0010_holdout/data/eval/eval_inputs.jsonl`.
- **Prompt:**
  - zero-shot `nl-mql-spider1-v1` (template sha256 7401bf14…);
  - RAG `nl-mql-spider1-rag-v1`;
  - frozen per-item prompts in `data/plans/plan_{A0,D0,E,EF}.jsonl` (plans sha256 d59f626f… / c489ce33… / a2b8c5d4… / ba196aaf…);
  - retry feedback: validator issue list (schema) or the R5 generic sentence;
  - retrieval corpus `data/corpus/approved_corpus_v2_1.jsonl` (96552f06…), view of 2,912 records.
- **Task:** NL→MQL, cross-engine (the Atlas MQL result is compared with the SQLite gold-SQL result), using the unchanged MQL-0006..0009 evaluator (`scripts/nl2mql/dev1034_core.py`).
- **Dataset:** 716 Spider-train items from 18 corpus DBs (holdout seed 20261119, after amendment A1). The retrieval view is corpus v2.1 minus the 18 holdout DBs (A4 R3).
- **Model:** Qwen/Qwen2.5-Coder-3B-Instruct @ 488639f1ff808d1d3d0ba301aef8c11461451ec5.
- **Decoding and precision:** HF bf16, emulated on a Kaggle T4; greedy; max_new_tokens 1024.
- **Batch size:** 8 for the first passes, with a recorded batch-1 OOM fallback (61 events). 1 for the matched delta and all retries.
- **Kaggle run:** all 8 passes in 1 Kaggle session (2 were planned).
- **Scoring:** local Windows, as `codegen_eval_readonly`, on Atlas `codegen_spider1_holdout_mql0010` (99 collections / 1,280 docs, server 8.0.39).
- **Gold reference:** 716/716 `ok`.

**Results (correct / 716, one metric: execution match):**

| Arm | Correct |
|---|---:|
| A0 | 397 (55.4%) |
| A0V | 415 (58.0%) |
| D0 | 413 (57.7%) |
| D0V | 430 (60.1%) |
| D0G | 414 (57.8%) |
| E | 424 (59.2%) |
| EP | 426 (59.5%) |
| EF | 421 (58.8%) |
| **FULL** | **434 (60.6%)** |

EP and EF differ from E only on the 52 matched-delta items. On those items: E 28/52, EP 28/52, EF 25/52.

**Registered A4 sequence** (exact McNemar test; bootstrap 95% CI, 10,000 resamples, seed 20261119):
- **Step 1 — D0V vs D0:** +2.37 pp [+1.26, +3.63], 17 / 0, p = 1.5e-5. **Rejects. The +1 pp practical criterion is met.**
- **Step 2 (Holm):**
  - FULL vs D0: +2.93 pp [+0.84, +5.17], Holm p = 0.011. **Rejects.**
  - D0V vs D0G: +2.23 pp [+1.26, +3.35], 16 / 0, Holm p = 6.1e-5. **Rejects.**
- **Step 3 (Holm):**
  - E vs D0: +1.54 pp [−0.28, +3.49], Holm p = 0.25. **Inconclusive (underpowered).**
  - EF vs EP: −0.70 pp [−1.54, 0.00], 1 / 6, Holm p = 0.25. **Descriptive.**
- **Descriptive comparisons:**
  - D0 vs A0: +2.23 pp, p = 0.26.
  - A0V vs A0: +2.51 pp, p = 1.2e-4.
  - D0G vs D0: +0.14 pp, p = 1.0.
- **Broken by a kept retry:** A0V 0, D0V 0, D0G 0, FULL 1.

**Noise (post hoc):**
- 10 reused byte-identical queries changed outcome on re-execution (tie/document-order nondeterminism). D0V vs D0 was unaffected.
- A sensitivity analysis that gives each reused record its source's outcome leaves every registered conclusion unchanged.

**Comparability:** the 9 arms are directly comparable with each other. They are **not** comparable with MQL-0008 (892) or MQL-0009 (639), because the items differ. "Current best results" is therefore not updated.

**Artifacts** (`outputs/nl2mql_mql0010/scoring_runs/run_final/`):
- `scoring/results_report.md` c98a1f7f…
- `scoring/analysis.json` 1ce743bc…
- `scoring/RUN_MANIFEST.json` 0844a9e8…
- `scoring/preflight/atlas_check.json` 0b323fee…
- `scoring/descriptive_recount_posthoc.json` b986b483…
- `scoring/posthoc_noise_sensitivity.json` e45233b2…
- `scoring/posthoc_config_check.json` 4bc24839…
- `scoring/arms/<arm>/`

**Post-hoc scripts:** `scripts/nl2mql_mql0010/analysis/posthoc_{descriptive_recount,noise_sensitivity,config_check}.py`.

**Input:** `mql0010_predictions.zip`, sha256 d18e7a85751151d0efff9b9432932dcbd37a62376be921cc0ec31945c0facebe.

**Report:** `docs/mql0010_final_results_analysis.md`.

### MQL-0011 stage 1: dense-retrieval offline screen (bge-small + embedded Qdrant + RRF), MQL-0010 dev split, 779 items

**Added:** 2026-11-22. **Status:** verified locally.
- The anchors reproduced exactly, and the artifact hashes were checked.
- Every gate value and arm metric was recomputed independently from `per_item.jsonl` by `scripts/nl2mql_mql0011/analysis/recompute_gate.py`, which does not import the screen code. 0 mismatches; output `screen/gate_recompute.json` 07148f56….

**What this is:** an **offline structural retrieval proxy. It is NOT execution accuracy.**
- No model generation and no Atlas query.
- Nothing here is comparable with any accuracy entry.

**Pre-registered verdict:** **FAIL. No stage 2.**

**Inputs:**
- **Retrieval pool:** approved corpus v2.1 (96552f06…) restricted to the registered MQL-0010 retrieval view.
  - 2,912 records, 105 DBs, ids sha256 cfde5e9c…, 18 holdout DBs absent.
  - Each query leaves its own DB out.
- **Questions:** the registered MQL-0010 **dev** split, 779 items in 15 DBs (ids sha256 b65ef62c…).
  - 679 items are attainable (their gold signature exists in the pool).
  - 507 have a corpus MQL, used for the skeleton metric.
- **Holdout:** the MQL-0010 holdout was **not read**.
- **Labels:** gold SQL and corpus MQL of dev items were used for metrics only.

**Embedding:**
- BAAI/bge-small-en-v1.5 at commit 5c38ec7c405ec4b44b94cc5a9bb96e735b38267a; weights sha256 3c9f3166….
- CLS pooling, L2-normalised, float32, 384 dimensions, CPU, batch 1.
- Embedded text: `masked_question`, the same text BM25 indexes.
- Re-encoding is byte-identical at 1 and at 4 threads.

**Vector store:** embedded Qdrant (qdrant-client 1.20.0, exact brute-force search) at `~/codegen_data/qdrant_mql0011`.
- Qdrant generates and filters the candidates.
- The ranking scores are re-computed from the artifact vectors (CanonicalScorer). Raw backend cosines differ by up to 3.1e-7, which flips 6-decimal ties.
- Backend parity on all 779 queries: 0 candidate-set mismatches, and 0 final-ranking mismatches between Qdrant and NumPy.

**Arms:** K = 3, top-3 of the ranking as in `retrieval_dev_proxy.py`.

| Arm | Retrieval |
|---|---|
| R0 | registered D0 hybrid |
| R1 | RRF(BM25) + Jaccard (matched control) |
| R2 | dense only |
| R3 | RRF(BM25, dense), k = 60, + Jaccard |
| R0E / R1E / R3E | the same with signature diversity |

**Results:**

| Arm | hit@3 | hit@3 attainable (679) | skeleton hit@3 (507) | mJ@3 | triples differing from R0 |
|---|---|---|---|---|---|
| R0 | 57.12 | 65.54 | 67.06 | 0.6564 | — |
| R1 | 56.74 | 65.10 | 66.07 | 0.6535 | 17.6% |
| R2 | 57.64 | 66.13 | 71.99 | 0.6354 | 96.9% |
| R3 | 59.44 | 68.19 | 69.43 | 0.6638 | 88.7% |
| R0E | 65.85 | 75.55 | 76.33 | 0.7121 | — |
| R3E | 69.32 | 79.53 | 78.70 | 0.7261 | 91.9% vs R0E |

The R0 and R0E anchors equal `outputs/nl2mql_mql0010/dev_evidence/retrieval_dev_proxy.json` exactly.

**Gate (paired bootstrap, 10,000 resamples, seed 20261122):**

| Check | Result | Outcome |
|---|---|---|
| G1: R3 − R0 on hit@3 attainable | +2.65 pp [0.29, 4.86] | **fails the +3.0 pp bar** |
| G2: R3 − R1 | +3.09 pp [1.03, 5.15] | passes |
| G3: R3 − R0 on skeleton hit@3 | +2.37 pp | passes |
| G4: the same checks on the 440 attainable non-twin items | R3 − R0 +1.59 pp [−1.59, 4.77]; R3 − R1 +2.05 pp [−0.68, 5.00] | **fails** |

A paraphrase twin is an item that has a candidate with an identical masked question or cosine ≥ 0.95 together with the same gold signature. 239 of the 779 items are twins.

**Descriptive:**
- The twin flag is partly defined by the dense channel: cosine ≥ 0.95 to a candidate with the same signature.
- The non-twin subset is therefore, by construction, the items where dense finds no close same-structure match. The −7.05 pp for dense alone (R2 − R0) on that subset reflects how the subset was selected; it is **not** a general property of embeddings.
- Twins come from other databases with different schemas, so they are **not leakage**. The deployed retriever would find them legitimately.
- G4 was registered as a robustness condition and is applied as written.
- The RRF rescaling by itself (R1 − R0) is −0.44 pp, so it explains nothing.

**Latency (embedded-local; not evidence of production-scale performance):**
- R0: 66 ms per query.
- R3: 414 ms per query (Qdrant local brute force).
- Artifact build: 48.5 s for 2,912 texts on CPU.

**Artifacts** (`outputs/nl2mql_mql0011/`):
- `screen/screen_results.json` fdf50ef6…
- `screen/per_item.jsonl` 59dc05ec…
- `screen/screen_summary.md` 16ff9d8d…
- `registration/screen_registration.json` 3ceaeb8f… (config sha256 989546b6…, plus source hashes)
- `preflight/{planner_equivalence f4f4ff1b…, backend_parity edfb054c…, ingest_report 73db5891…, artifact_verify_threads1 fc0ff975…, artifact_verify_threads4 06563f3c…}.json`
- `embeddings/mql0011__v2_1_view2912__bge_small_en_v1_5__5c38ec7c__masked_q/manifest.json` ff5f899b…

**Code and config:** `scripts/nl2mql_mql0011/`; config `scripts/nl2mql_mql0011/configs/mql0011_screen.json`.

**Limitation:** every metric is structural. The screen cannot detect non-structural accuracy effects, such as paraphrase or output-shape priming.

### MQL-0012: offline confirmation of X1 (neighbour-voted structure cues) and X2 (BM25 removed) against R3, fresh X0 split, 3,295 items

**Added:** 2026-11-22. **Status:** verified locally.
- Every confirmatory value and verdict was recomputed independently from `per_item.jsonl` by `scripts/nl2mql_mql0012/analysis/recompute_confirmatory.py`: 0 mismatches.
- MQL-0011 files and the Qdrant storage are byte-identical before and after the run.

**What this is:** an **offline structural retrieval proxy. It is NOT execution accuracy.** No model generation, no Atlas query.

**Pre-registered verdicts:**

| Arm | Verdict |
|---|---|
| X1 | **FAIL** (significantly worse than R3) |
| X2 | **FAIL** (non-inferiority not shown) |

**Data:**
- **X0:** all 90 MQL-0010 build DBs, 3,295 items (ids sha256 d31164f3…).
  - Same item rule as MQL-0010, plus spider-pool-v3 membership.
  - Disjoint from the MQL-0010 holdout, the MQL-0010/0011 dev split, MQL-0009, Spider dev and the expanded MQL splits.
  - 2,882 items are attainable; 2,405 have a corpus MQL (shape metric).
- **Retrieval:** the registered 2,912-record view, leaving out each query's own DB, with no pair exclusions.
  - The MQL-0011 embedding artifact and Qdrant collection were used read-only.
  - X0 query vectors come from the same pinned bge-small model.
- **Holdout:** not used.

**Registration:** `outputs/nl2mql_mql0012/registration/mql0012_registration.json` 2fcebcdc…
- Config sha256 5909a914…, registered 2026-11-22 09:28 UTC, before any X0 retrieval.
- Validation before registration: on all 779 MQL-0011 dev items, R3 and R0 reproduce their registered top-3, and the X1/X2 code with the regex cues reproduces R3's full rankings.

**Arms** (K = 3, RRF k = 60, structure weight 1.0):

| Arm | Retrieval |
|---|---|
| R3 | control: the unchanged MQL-0011 rrf_fused retriever |
| X1 | R3 with the regex cues replaced by a strict-majority (≥ 6 of 10) unweighted vote of the top-10 distinct dense neighbours' corpus labels, unioned with the unchanged schema-mention join rule and text_number rule |
| X2 | R3 with the BM25 term removed: L = 61/(60 + dense rank), plus the regex Jaccard. MQL-0011's `dense_only` mode was not used, because it has no Jaccard; this deviation is stated in the config |
| R0 | descriptive reference |

**Results:**

| Arm | structure_match (2,882) | pipeline_shape (2,405) | join (993) | hit@1 |
|---|---|---|---|---|
| R0 | 63.53 | 66.07 | 51.86 | 50.31 |
| R3 | 67.45 | 69.48 | 57.40 | 54.79 |
| X1 | 63.32 | 66.07 | 53.07 | 53.61 |
| X2 | 66.31 | 68.44 | 56.70 | 54.61 |

**Confirmatory tests** (paired bootstrap, 10,000 resamples, seed 20261123, numpy; exact McNemar; Holm over {p_X1, p_X2}):

| Comparison | Discordant pairs | Δ, pp | 95% CI | p |
|---|---|---|---|---|
| X1 − R3, structure | 164 / 283 | −4.13 | [−5.55, −2.71] | McNemar 2.0e-8 |
| X1 − R3, join subgroup | — | −4.33 | [−6.65, −2.11] | — |
| X2 − R3, structure | 102 / 135 | −1.15 | [−2.19, −0.10] | NI 0.618 |
| X2 − R3, shape | 81 / 106 | −1.04 | [−2.16, 0.08] | NI 0.508 |

- X1 fails criteria 1–3. The difference is significant, in the harmful direction.
- X2 fails both non-inferiority margins (CI lower bound must exceed −1.0).

**Descriptive** (not confirmatory):
- **R3 vs R0:** +3.92 pp [2.67, 5.17], McNemar 9e-10. This does not change the MQL-0011 gate, which stays FAIL.
- **X2 vs R3:** the structure CI excludes 0 (McNemar 0.037, unadjusted). This suggests BM25 adds about 1 pp, but it is a single unregistered test.
- **X1 vs R3 on shape:** −3.41 pp.
- **Subgroups (X1 vs R3):** group-by 49.8 vs 61.5 (n = 613); join 53.1 vs 57.4.
- **Twin split:** twins 84.2 vs 84.9 (n = 1,167); non-twins 49.2 vs 55.6 (n = 1,715). The twin definition is biased, so this is descriptive only.
- **Cue accuracy against gold signatures:** vote cues match the exact signature on 50.1% of items vs 40.2% for regex, and are more precise on most labels. They lose recall on aggregate (0.55 vs 0.82), having (0.27 vs 0.76), distinct (0.43 vs 0.67) and like (0.30 vs 0.69). They can never predict the four labels absent from the corpus.
- **Latency (embedded-local, per query):** R0 43 ms, R3 195 ms, X1 196 ms, X2 183 ms.

**Coverage (separate finding):**
- anti_join, intersect, nested and union never occur in approved corpus v2.1.
- 413 of 3,295 X0 items (12.5%) and 100 of 779 dev items are unattainable for every retriever. Details in `outputs/nl2mql_mql0012/coverage/coverage.json` 045f3365….

**Power, as registered:** X2 had a 3–16% chance of passing even if X2 and R3 were truly equal.

**Artifacts** (`outputs/nl2mql_mql0012/`):
- `results/per_item.jsonl` 2e66b2df…
- `results/results.json` a4ec7921…
- `results/summary.md` df0fcad4…
- `results/recompute_confirmatory.json` 0b4ac19d…
- `split/x0_split.json` a9d75ed3…
- `split/x0_items.jsonl` 90b08399…
- `preflight/validation.json` 766c4136…
- `query_cache_x0/manifest.json` 219a8dcf…

**Code:** `scripts/nl2mql_mql0012/`.

### MQL-0010-FT1 (side experiment): fine-tuned Qwen-3B LoRA q3b_r16_q10, A0 zero-shot prompts, MQL-0010 holdout (716)

**Added:** 2026-11-22. **Status:** provisional. Predictions (hashes, counts, ids, prompts) and the metric were checked locally and on Atlas; the fine-tuning training data was **not** checked (it exists only on Kaggle). Side experiment outside the registered MQL-0010 A4 sequence; the MQL-0010 entry and "Current best results" are unchanged.

- **Model:** Qwen/Qwen2.5-Coder-3B-Instruct @488639f1 bf16 with LoRA `q3b_r16_q10` (sha 7096760a…; r 16, alpha 32, all attn+MLP proj; 1 epoch, lr 2e-4, trained on the unsloth bnb-4bit base, merged into bf16). Training data reported by the export: 2,395 train / 506 dev rows, file `sft_train.prompt_completion.jsonl` (not available locally; the local `data/finetune/ft_v1` file of that name is a different set, 1,784 rows).
- **Generation:** unchanged MQL-0010 package (manifest 87bea695…), A0 zero-shot prompts (prompt_sha256 identical to MQL-0010 A0 for 716/716), greedy, max_new_tokens 1024, Kaggle T4 (bf16 emulated), 144 OOM batch-1 fallbacks.
- **Evaluation:** MQL-0010 holdout, 716 items / 18 DBs, Atlas `codegen_spider1_holdout_mql0010`, read-only `codegen_eval_readonly`, full-parity preflight OK; unchanged evaluator `dev1034_core` (sha 81b86f0b…) via `mql0010_scoring.py` (sha ba32e1c0…) functions, A0 only; frozen reference sha 1ff619f2… reused from run_final.
- **Execution accuracy (Atlas): 516 / 716 (72.07%).** incorrect 187, invalid 3 (2 unsafe_mql, 1 format_error), failed 10 (model execution_error), timeout 0. Model output: 716 stop, 1 unparseable.
- **Paired vs MQL-0010 (same 716 questions):** A0 397 → +16.62 pp (156/37, exact McNemar p 1.4e-18; same prompts, the clean comparison). Descriptive only (different prompts): D0 413 (+14.39 pp), D0V 430 (+12.01 pp), FULL 434 (+11.45 pp).
- **Leakage evidence (not a verification):** registered holdout retrieval view rebuilt exactly (2,912, ids sha cfde5e9c…); minus 11 L3–L5 near-duplicate ids = 2,901 = 2,395 + 506. 533/716 holdout items are corpus records, so leakage would matter. FT output equals the corpus gold MQL byte-for-byte on 273/532 corpus-member items (base 0), falling with hardness (easy 151/234 → extra 3/19), consistent with learned converter style.
- **Not run:** inspection of the actual training file for holdout DBs/ids; second-session re-score of base A0 on the same Atlas state.
- **Note 2026-11-22:** the line "No fine-tuned MQL model exists" under "Recorded but not ledger results" predates this entry.
- **Note 2026-11-23 (status → verified):** the friend's handoff `ft_handoff_q3b_r16_q10.zip` (repo root, sha 0103333c…; SHA256SUMS 25/25 OK; extracted to `outputs/finetuning_reuse/handoff_extract/`) supplies the adapter, training data, training notebook, config and prompt format. `adapter_model.safetensors` sha = 7096760a… (matches the scored run). Independent local leakage check (`outputs/finetuning_reuse/leakage_check/`, our own script, not the friend's builder):
  - rows 2,395 / 506 / 2,901 (full = train ∪ dev, 0 duplicates);
  - all ids are approved-corpus v2.1 records; 0 holdout ids; 0 rows from the 18 holdout DBs;
  - train = the 90 registered build DBs, dev = the 15 registered dev DBs, no DB overlap;
  - full = the 2,912 holdout retrieval view minus exactly the 11 L3–L5 near-duplicate ids;
  - questions and labels are byte-equal to the corpus (0 mismatches); one system prompt, identical to eval A0;
  - 0 exact, 0 normalised and 0 token-Jaccard ≥ 0.8 question matches against the 716 holdout questions;
  - positive control: the same detector flags 11/11 removed near-duplicates, and those 11 ids equal the registered L3–L5 pair corpus ids;
  - Spider `test_full` (MQL-0013, 40 DBs) shares 0 DBs with the 105 training DBs.

  Adapter selection used dev loss on the 15 dev DBs, not the holdout (4-config sweep, `training/sweep_summary.json`). Training config confirmed from the notebook code: Unsloth QLoRA, adamw_8bit, weight decay 0.01, max_seq_length 2048, response-only loss, no packing, LoRA dropout 0. Training prompts use the eval A0 layout but write `Foreign keys: (none)` / `Sample documents: (none)` and list only the touched collections. Remaining caveats: the holdout was reused (MQL-0010, MQL-0014, FT1) and FT1 was not pre-registered, so it is a secondary result; the base A0 was not re-scored in the same Atlas session; the Unsloth base-repo revision is not pinned.
- **Artifacts:** `outputs/nl2mql_mql0010_ft_side/ft_q3b_r16_q10_A0/` (REPORT.md, input zip sha 24118b11…, predictions sha a7996b83…, `scoring/arms/A0/evaluation_results.jsonl`, `scoring/comparison_vs_run_final.json`, `scoring/RUN_MANIFEST.json`).

### MQL-0014: R0V vs R3V execution re-check (BM25 hybrid vs embedding-enhanced R3 retrieval), Qwen-3B, MQL-0010 holdout (716), secondary re-analysis

**Added:** 2026-11-23. **Status:** verified. Export hashes and counts were checked; the 4 arms were assembled and checked by `verify_exports`; Atlas full-parity preflight OK; all metrics independently recomputed from the per-question evaluation files (counts, McNemar, numpy bootstrap CI [0.42, 4.61]). **This is a secondary re-analysis of existing evaluation data, not a fresh independent test.** The D0V recipe was selected on these items; R3 retrieval was developed on the MQL-0011 dev split and X0 and was never tuned on them. This entry does not change MQL-0011's failed stage-1 gate. MQL-0013 (Spider test_full) was registered but superseded and never run; test_full remains unused.

- **Arms:**
  - R0 = MQL-0010 D0 (planner_dense lexical `bm25`; the plan equals the frozen MQL-0010 `plan_D0` on 716/716 rows).
  - R3 = lexical `rrf_fused` (MQL-0011 R3: RRF_60(BM25) + RRF_60(bge-small @5c38ec7c dense) + regex-cue Jaccard; registered Qdrant collection read-only, byte-identical before and after).
  - Both use K=3, the 2,912-record view and the 13 registered L3–L5 exclusions. 0 examples came from a holdout DB or the question's own DB; 0 were dropped by the budget.
  - R0V / R3V add the MQL-0010 validator and one schema-feedback retry.
  - Prompts were identical on 58 items and differed on 658.
- **Generation:** Qwen/Qwen2.5-Coder-3B-Instruct @488639f1, bf16 emulated on a Tesla T4, greedy, max_new_tokens 1024, torch 2.11.0+cu130, transformers 5.20.2.
  - One Kaggle session.
  - Passes: shared (58), R0-only (658) and R3-only (658) at batch 8; distinct retries at batch 1.
  - Retried: R0V 131, R3V 126. Kept: 55 / 46.
- **Evaluation:** 716 items over 18 DBs on Atlas `codegen_spider1_holdout_mql0010` (server 8.0.39), read-only `codegen_eval_readonly`.
  - Unchanged evaluator `dev1034_core` (sha 81b86f0b…).
  - Reference = the frozen MQL-0010 reference (sha 1ff619f2…). A local recompute matched it on every field except the timing field `gold_ms`.
  - Each distinct (item, text) was executed once: 1,123 executions.
- **Execution accuracy (strict, /716):**

  | Arm | Correct | % | Incorrect | Invalid | Execution failure | Timeout | Generation failure | Infrastructure |
  |---|---:|---:|---:|---:|---:|---:|---:|---:|
  | R0 | 414 | 57.82 | 280 | 6 | 16 | 0 | 0 | 0 |
  | R0V | 432 | 60.34 | 260 | 5 | 19 | 0 | 0 | 0 |
  | R3 | 438 | 61.17 | 250 | 8 | 20 | 0 | 0 | 0 |
  | R3V | 450 | 62.85 | 239 | 6 | 21 | 0 | 0 | 0 |

- **Primary (registered):** R3V − R0V = **+2.51 pp**, paired bootstrap 95% CI [0.42, 4.61], exact McNemar p = **0.030**.
  - Both correct 410; both wrong 244; R3V-only 40; R0V-only 22.
  - H0 rejected at α 0.05. This was the single registered hypothesis.
- **Descriptive:**
  - R3 − R0 +3.35 pp [1.26, 5.45], p 0.0032 (43/19).
  - Retry effect: R0V − R0 +2.51 pp (18/0); R3V − R3 +1.68 pp (12/0).
  - 60% target: R3V 450 ≥ 430, met. R0V 432 is also just above it.
- **Breakdowns (R3V − R0V):**
  - Hardness: easy +1.67 (240), medium +4.41 (272), hard +0.85 (118), extra +1.16 (86).
  - Join +3.38 (237); no join +2.09 (479).
  - Flat +2.97 (639); nested or set-operation −1.30 (77). Nested/set-op accuracy stays about 16% in every arm, a known corpus coverage gap.
- **Reproducibility:** the original MQL-0010 D0V, rescored in the same session, gives 431/716. It was 430 in MQL-0010: `spider_train_06109` flipped incorrect→correct on re-execution of the same text (known tie-order noise). That text is shared by R0V and R3V, so the primary comparison is unaffected. The regenerated R0V text equals the old D0V text on 698/716 items; R0V vs old D0V is 1/0.
- **Artifacts:** `outputs/nl2mql_mql0010_r3_execution_recheck/`:
  - registration `registration/mql0014_registration.json` (f8378231…);
  - package `kaggle/nl2mql_mql0014_r3_recheck/` (manifest 76409a76…);
  - input `mql0014_predictions.zip` (c1b244f6…);
  - `scoring_runs/run_final/scoring/` (analysis.json, results_report.md, RUN_MANIFEST.json, arms/*/evaluation_results.jsonl);
  - `REPORT.md`.

### MQL-0015: A0V vs R3V vs FT-R3V, Qwen-3B, 1,000 Spider `test_full` items (registered final evaluation)

**Added:** 2026-11-25. **Status:** verified.
- **Checks run:**
  - Export SHA256SUMS OK. Status `complete` with no problems; the environment and weights were consistent across both sessions.
  - All six session-1 pass files are byte-identical in the session-2 export.
  - `verify` passed: 6 arms × 1,000, frozen prompts, weights per arm, adapter identity.
  - Atlas full-parity preflight OK: 180 collections.
  - Per-arm counts and the three McNemar tests were recomputed independently from `arms/*/evaluation_results.jsonl` and match the analysis.
- **This is the first and only `test_full` run.** Registration `fb5d07d0…`, amendment A1.

- **Arms:**
  - A0 = zero-shot; R3 = frozen R3 retrieval prompts (hybrid BM25 + bge-small dense, RRF, top 3); FT-R3 = the same R3 prompts on base + the verified LoRA, merged (adapter `7096760a…`).
  - The V arms add the schema validator and one schema-feedback retry.
- **Generation:** Qwen/Qwen2.5-Coder-3B-Instruct @488639f1, bf16 emulated on one Tesla T4 (Kaggle), greedy, max_new_tokens 1024, batch 8 for first answers and 1 for retries, torch 2.11.0+cu130, transformers 5.20.2, peft 0.21.2.
  - Session 1 (base passes): `mql0015_predictions.zip` (`bc3d49ca…`).
  - Session 2 (FT passes, resumed from the session-1 output attached as a dataset): `mql0015_predictions_session2.zip` (`2375aa69…`).
  - 0 generation failures. Retried / kept: A0V 152/85, R3V 154/83, FT-R3V 130/43.
- **Evaluation:**
  - Atlas `codegen_spider1_test_mql0015`, read-only `codegen_eval_readonly` via `mql0015_scoring_authorized.py`.
  - Unchanged evaluator `dev1034_core` (`81b86f0b…`), cross-engine against the SQLite gold, all arms in one session.
  - 3,144 unique executions; 0 infrastructure failures (no re-execution rounds).
- **Execution accuracy (strict, /1,000):**

  | Arm | Correct | % | Wilson 95% | Incorrect | Invalid | Execution failure | Timeout | Not comparable |
  |---|---:|---:|---|---:|---:|---:|---:|---:|
  | A0 | 471 | 47.1 | 44.0–50.2 | 447 | 14 | 16 | 22 | 30 |
  | **A0V** | 497 | 49.7 | 46.6–52.8 | 418 | 11 | 21 | 23 | 30 |
  | R3 | 515 | 51.5 | 48.4–54.6 | 397 | 3 | 35 | 21 | 29 |
  | **R3V** | 544 | 54.4 | 51.3–57.5 | 361 | 3 | 42 | 23 | 27 |
  | FT-R3 | 588 | 58.8 | 55.7–61.8 | 329 | 20 | 13 | 24 | 26 |
  | **FT-R3V** | 598 | 59.8 | 56.7–62.8 | 318 | 18 | 13 | 27 | 26 |

- **Primary (registered):** FT-R3V − R3V = **+5.40 pp**, paired bootstrap 95% CI [2.7, 8.1], cluster bootstrap [2.49, 8.3], exact McNemar p = **0.00014** (FT-R3V-only 125, R3V-only 71). H0 rejected.
- **Secondary (Holm):**
  - R3V − A0V +4.70 pp [1.7, 7.8], Holm p 0.0025 (140/93).
  - FT-R3V − A0V +10.10 pp [7.0, 13.3], Holm p 2.1e-9 (188/87).
  - Both rejected; all three stay rejected under Holm over three.
- **Descriptive:**
  - Retry effect: A0V − A0 +2.6 pp (26/0); R3V − R3 +2.9 pp (29/0); FT-R3V − FT-R3 +1.0 pp (11/1).
  - 60% target: FT-R3V 59.8% is just below it (fact only).
  - Weak spots: set operations (n 85) A0V 15 → R3V 8 → FT-R3V 4; anti-joins (n 35) 19 → 5 → 2. Extra-hard items: FT-R3V 40 vs R3V 42.
  - The primary result is stable under every registered sensitivity exclusion: +5.43 to +5.46 pp, p ≈ 0.00014.
- **Artifacts:** `outputs/nl2mql_final_1000/`:
  - `registration/`;
  - `scoring_runs/run_final/scoring/`: analysis.json `6f0f11de…`, results_report.md `afd6ce69…`, RUN_MANIFEST.json `d8fd6d00…`, `arms/*/evaluation_results.jsonl`;
  - input zips in the repo root.
- **Note:** the run's final console print failed after scoring (a Windows cp1252 `UnicodeEncodeError` on "−"). Every result file had already been written; nothing was re-run.

### Recorded but not ledger results

- **Spider 2.0-Lite 68** (`data/processed/nl_mql_dataset.jsonl`): only reference-oracle (gold replay) harness runs exist. In `outputs/evaluation_68/reference_baseline/`, 62/68 executed and matched and 6 hit execution errors. The 6 were re-run in `reference_slow6/` (2/6 correct, 3 execution errors) and again in `reference_slow6b/` (6/6 correct). **No model has been evaluated on it.**
- Partial Gemini runs and the CPU `random_k3__train` run: incomplete; see MQL-0001 and MQL-0004.
- No fine-tuned MQL model exists. No MQL test-partition (65) evaluation has been run for any RAG or fine-tuning experiment.

## C. Dataset audits

- 2026-10-28: SQL-0001 parser-rejection audit (an evaluation-protocol diagnostic). See the note under SQL-0001 and `docs/sql_v1_zero_shot_parser_audit.md`.
- 2026-10-28: SQL-0001 test-suite execution accuracy (an evaluation-protocol addition, 630/1,034). See note (b) under SQL-0001 and `docs/sql_v1_test_suite_eval.md`.

- 2026-10-30: RAG corpus eligibility audit for the 892-item NL→MQL evaluation set (planned MQL-0007; **verified**, local recompute by script).
  - Script `scripts/nl2mql/audit_rag_corpus_dev1034.py`; outputs `outputs/nl2mql_spider_dev_1034_rag/corpus_audit_v1/{candidates.jsonl,summary.json}`.
  - Accepted supervised pairs: **0 / 2,144 rows** (1,864 unique pairs).
    - corpus_v1 examples: 280/280 are evaluation items (by ID and question).
    - ft-v1 Spider rows: 280/280 are evaluation items.
    - ft-v1 Kaggle rows: 1,584/1,584 carry sql2mql-v1 labels, and v2 does not exist; 88 are known-wrong.
  - Guide cards: 4/18 excluded (`concert_singer` examples) and 14/18 pending a licence check.
  - MQL-0007 is **not** entered as planned: it is blocked and nothing was prepared. See `docs/nl2mql_spider_dev_1034_rag_experiment.md`.

- 2026-10-30: **sql2mql-v2 Phase 1 validation** (dataset audit; label validation measured LOCALLY on SQLite + MongoDB 8.0.34 Docker, never Atlas; **verified**: rates recomputed from the per-item JSONL by the scripts named). Write-up: `docs/sql2mql_v2_quality_assessment.md`. Code: `scripts/finetune_v2/`. Outputs: `outputs/sql2mql_v2/`. Source: `data/processed/spider_sql_v1/train.jsonl`, DEV sample only (500 ids; freeze `data/interim/finetune_v2/pilot_freeze.json`, sha `fb1affb9…`). No pilot id was used.
  - Adversarial suite: 139 cases, 110 converted, 110/110 correct under both NULL representations; **0 / 139 false accepts**.
  - Negative controls: 33/33 v1 wrong translations rejected.
  - Coverage matrix: 26/26 items with passing evidence.
  - Dev, verifier-2: coverage 378/500 (75.6%); accepted 361/500 (72.2%); 0 L1 or fixture mismatches; 14 tie-cut, 2 timeout, 1 non-discriminating. Gold SQLite results reproduce the recorded hashes 500/500.
  - **Mutant study (gate G4: Wilson upper bound ≤ 0.05) FAILED:**
    - iteration 1, 27/200 labels (13.5%, CI 9.5–18.9%);
    - iteration 2, pre-registered held-out, 7/161 (4.35%, CI 2.1–8.7%).
  - The secondary in-sample run of iteration 2 was stopped by the low-memory reaper and is not reported.
  - Status: **Phase 1 not passed**; no labels produced for training or RAG.
- 2026-10-30, note on the entry above it (RAG corpus eligibility audit): its text "v2 does not exist" was true when written. `sql2mql-v2` now exists in `scripts/finetune_v2/` but has not passed its gate, and the ft-v1 Kaggle labels remain sql2mql-v1 labels. The exclusion stands; `audit_rag_corpus_dev1034.py` now states the reason as "label from sql2mql-v1 (never re-derived and re-verified by a remediated converter)".
- 2026-10-31: **sql2mql-v2 verifier-3 and round-3 G4 re-measurement** (dataset audit; label validation measured LOCALLY on SQLite + MongoDB 8.0.34 Docker, never Atlas). **Verified:** an independent script recomputed every figure from `round3/records.jsonl` plus `round3_mutants/{mutants,no_mutants}.jsonl`, and all 19 frozen file hashes match.
  - **Sources and code:**
    - write-up: `docs/sql2mql_v2_round3_report.md`;
    - code: `scripts/finetune_v2/` (verifier3, fixtures3, run_v3, mutant_study_v3, round3_*);
    - outputs: `outputs/sql2mql_v2/round3/`, `round3_mutants/`, `round3_report.json`;
    - pre-registration: `data/interim/finetune_v2/round3_prereg.json` (body sha `944e9063…`, written before verifier-3 code existed) and `round3_code_freeze.json` (code hashes, 2026-10-31T16:57:34).
  - **Sample:** 740 records drawn from the 5,229 `exec_ok` train records outside PILOT/DEV, stratified J2 / OL / J1 / G with 130 / 130 / 130 / 350 records (population 446 / 751 / 1,090 / 2,942), seed 20261031. The iteration-2 holdout, the pilot, the 892-item evaluation set and the 65-item test partition were not used.
  - **Coverage and acceptance:** coverage 575/740 (77.7%); accepted 542/740 (73.2%); 28 tie-cut, 5 empty referenced table; 0 crashes. V8: 542/542.
  - **G4 (pre-registered: weighted Wilson upper bound ≤ 0.05):**
    - weighted, p_w 0.0238 over n_eff 502.2, Wilson 95% 0.0137–0.0412: **PASSED**;
    - unweighted, 15/542 (2.77%), Wilson 1.68–4.52%; this is not the population rate;
    - by stratum: J2 7/104 (6.73%, Wilson 3.30–13.25%); OL 0/78 (0.00%, 0–4.69%); J1 3/90 (3.33%, 1.14–9.35%); G 5/270 (1.85%, 0.79–4.26%).
  - **Mutants:** 2,741 in total. 1,969 killed; 21 false accepts across 15 labels; 751 unkilled.
  - **Verifier-3 on G2/G3:** G2 0 false accepts, with 108/110 correct accepted; G3 33/33 rejected.
  - **Pre-freeze checks** (in-sample, not gate evidence): the 200 diagnosis labels gave 3/200 under verifier-2 and 0/200 under verifier-3; the synthetic suite gave 0 for both.
  - **Interruptions:** the low-memory reaper stopped step 1 once and step 2 twice. Each was resumed with the frozen code, with 0 partial labels dropped (resume logs).
  - **Status:** G1–G3 and G5–G6 as frozen; G4 passes the user-specified, pre-registered round-3 criterion (unweighted form also ≤ 0.05). Phase 1 passed on that basis. Pilot not started; no labels produced for training or RAG.

- 2026-11-05: **Independent audit of `spider_pool.zip`** (dataset audit, SQL-side only; no model, no converter, no MongoDB; measured locally on SQLite 3.40.1; **verified**: counts recomputed from the archive and local files by script). Verdict: **CONDITIONAL PASS, `train.jsonl` only, after exclusions; the package as delivered is not a training artifact.**
  - **Sources:** write-up `docs/spider_pool_independent_audit.md`; outputs `outputs/spider_pool_audit_v1/`. Archive sha256 `31848241…`. Dataset: `train.jsonl` 5,514 + `val_internal.jsonl` 744 audited in full; `dev.jsonl` 899 and 135 dev rejects at ID level only.
  - **Full execution on official populated DBs:** 6,258 / 6,258 (100%), with 0 errors, 0 timeouts; hash match 6,258 / 6,258 (100%). This shows stability, not correctness, because Spider has no independent answer key.
  - **Parsers:** official 6,258 / 6,258, sqlglot 6,258 / 6,258. **Lineage:** 8,570 / 8,570 training-side rows match raw Spider.
  - **Leakage, train and val vs protected sets:** 0 IDs and 0 databases. **Package vs protected sets:** Spider dev 1,034 / 1,034 present; held-out MQL test 65 / 65 present (64 in `dev.jsonl`, 1 in `rejected.jsonl`).
  - **Label defects in kept rows:** Tier A confirmed defects 57 / 6,258 (0.91%; 53 train, 4 val); Tier B row-choice dependent 37 / 6,258 (0.59%); Tier C review 42 / 6,258 (0.67%). Hand sample (one LLM reader, not a gate): 3 / 80 clear errors (3.75%, Wilson 1.28–10.45%).
  - **Frozen V2 sets, membership only, no gate statistic:** in pool train, pilot 276 / 300, V2 dev 459 / 500, round-3 670 / 740.
  - **Not done:** dev rows not audited; question-text leakage against dev/test not re-measured; builder script absent, so cleaning not reproduced and R8 not replayed.
- 2026-11-05: **Reproducibility audit of `spider_pool.zip`** (dataset audit, SQL-side only; no model, no converter, no MongoDB; measured locally; **verified**: unmodified `build_spider_pool.py`, sha256 `45e1e28e…` = manifest `code_sha256`, run in isolated copies, outputs compared with the archive by script). Report: `docs/spider_pool_reproducibility_audit.md`. Evidence: `outputs/spider_pool_repro_audit_v1/`. Verdict: **CONDITIONAL PASS**.
  - **Linux, Python 3.12.3, SQLite 3.45.1:** data files byte-identical 4 / 4; record mismatches 0 / 9,604. Manifest differs in 1 key (`sqlite_version`); byte-identical after substituting it.
  - **Windows, SQLite 3.50.4:** same records; every file differs by CRLF. **Windows, SQLite 3.40.1:** train 5,518 vs 5,514 (4 R3 rejects move to kept); values differ in 56 train/val and 10 dev rows.
  - **Tiers in reproduced kept rows:** A 57 / 57, B 37 / 37, C 42 / 42; the builder handles none. 5,430 / 734 = Tier A + B removed, C kept; with C, 5,388 / 734.
  - **Protected, IDs only:** train + val hold 0 / 1,034 dev, 0 / 65 held-out, 0 / 2,147 test. Whole output holds 1,034 / 1,034 dev and 65 / 65 held-out, by design.
  - **Note on the earlier 2026-11-05 entry:** builder now obtained and R8 replayed (59 / 59 train and val rows). Its empty-table screen undercounted, 2 vs 4 rows; `spider_train_02208` should join Tier C for review.
  - **Not done:** SQLite 3.53.1 not run (download and compile blocked by session policy); the zip container has no generator and was not rebuilt; text-level leakage not measured; post-exclusion defect rate not measured.
- 2026-11-06: **spider-pool-v2 prepared** (dataset preparation, SQL-side only; no model, converter, verifier-3 or MongoDB; **verified**: 38 / 38 (100%) independent checks pass, 3 / 3 builds hash-identical, 5 / 5 (100%) corruption tests detected). Report: `docs/spider_pool_v2_preparation_report.md`. Artifact: `data/processed/spider_pool_v2/` (manifest sha256 `cb80a118…`). Source: `spider_pool.zip` train/val members only.
  - **Train (source 5,514):** retained **5,387 / 5,514 (97.70%)**; excluded 84 / 5,514 (1.52%: Tier A 53, B 31); quarantined 43 / 5,514 (0.78%: Tier C 42, `spider_train_02208` 1).
  - **Val (source 744):** retained **734 / 744 (98.66%)**; excluded 10 / 744 (1.34%: A 4, B 6); quarantined 0 / 744 (0.00%).
  - **Protected IDs in any output file:** dev 0 / 1,034 (0.00%), test 0 / 2,147 (0.00%), held-out 0 / 65 (0.00%). Train–val overlap: 0 IDs, 0 databases, 0 query groups.
  - **Not certified:** SQL labels are not shown correct; the post-exclusion error rate has not been measured.

- 2026-11-06: **sql2mql-v2 Phase-2 pilot, verifier-3, run01** (dataset audit; label validation measured LOCALLY on SQLite + MongoDB 8.0.34 Docker, digest `b9924c33…`, never Atlas; not a model metric; **verified**: counts recomputed from raw `records.jsonl`, all 19 frozen hashes match before and after the run, output hashes match `run_manifest.json`). Report: `docs/sql2mql_v2_pilot_v3_report.md`. Runner: `scripts/finetune_v2/run_pilot_v3.py` (sha `daf42dfa…`, frozen before execution in `data/interim/finetune_v2/pilot_v3_run01_runner_freeze.json`). Outputs: `outputs/sql2mql_v2/pilot_v3/run01/`. Sample: the 300 frozen pilot ids (`pilot_freeze.json`, sha `fb1affb9…`), source `data/processed/spider_sql_v1/train.jsonl`; converter sql2mql-v2 unchanged; verifier `sql2mql-v2-verifier-3`.
  - **Outcomes:** accepted **213 / 300 (71.00%**, Wilson 65.63–75.84%); converter rejection 76 / 300 (25.33%); verification failure 11 / 300 (3.67%, all `tie_cut`); conversion error, safety rejection, timeout, verifier error 0 / 300 (0.00%) each. Converted 224 / 300 (74.67%); accepted of converted 213 / 224 (95.09%). Semantic mismatches 0 / 224.
  - **By spider_pool_v2 membership:** train 207 / 271 (76.38%); quarantined 1 / 1 (not promoted); excluded 0 / 4 (0.00%); not in pool 5 / 24 (20.83%; 11 tie_cut).
  - **By hardness:** easy 69 / 72 (95.83%), medium 105 / 118 (88.98%), hard 30 / 64 (46.88%), extra 9 / 46 (19.57%).
  - **≥3-table rule** (J2 stratum OR ≥3 SQL tables OR ≥2 `$lookup`): 28 / 300 flagged, 20 / 28 accepted (71.43%); accepted ones quarantined from corpus candidacy, kept in all pilot counts.
  - **Gates:** P1 213 / 213 (100%) accepted labels carry every required check; P4 every failure classified (0 unclassified). **P2 and P3 PENDING** (independent human review; package `run01/review_package/`, 120 items, 0 judgements). **Pilot gate not passed.**
  - **Corpus candidacy (provisional, no corpus written):** 188 candidates pending P2; 19 quarantined (≥3 tables); 93 not candidates.
  - **Reproducibility:** second execution: strict comparison 278 / 300 (92.67%) identical (runner exit 1); all 22 differences are the informational `v6_determinism` order note; 300 / 300 (100%) identical outcomes, statuses and MQL.
  - **Tests:** 99 / 99 (100%) passed, 0 skipped (33 new + 66 existing V2).

- 2026-11-06: **AI cross-verification of the 120 pilot-v3 P2 review items** (dataset audit; AI review by Claude Opus 5.5, **not human**, does not satisfy P2; executions LOCAL on SQLite 3.40.1 + MongoDB 8.0.34 `spider_train_v2`, never Atlas; **verified**: counts recomputed from `cross_verified_review.csv` and `execution_evidence.jsonl` by script). Report: `outputs/sql2mql_v2/pilot_v3/run01/cross_review/cross_review_report.md`. Harness `cross_review_exec.py` (sha `483722b2…`, imports no verifier/converter code). Inputs: `review_package/review_items.jsonl` (120), AI sheet `review_inputs/review_sheet_ai_reviewed.csv` (input columns byte-identical to `review_sheet.csv`, 120 / 120).
  - **Execution:** full-result typed multiset equality 120 / 120 (100%); ORDER BY items tie-aware ordered equality 30 / 30 (100%; strict 28 / 30); ties at LIMIT cutoff 0 / 16; table row counts SQLite = Mongo for every referenced table.
  - **Cross decisions:** CORRECT 111 / 120 (92.50%); INCORRECT 6 / 120 (5.00%, Wilson 2.31–10.48%: R012, R084, R097, R101, R118, R119); UNCERTAIN 3 / 120 (2.50%: R009, R070, R111); BLOCKED 0 / 120.
  - **SQL→MQL non-equivalent:** 0 / 120 (0.00%, Wilson upper 3.10%). All 6 INCORRECT are NL→SQL defects in the Spider source SQL (confirmed in `train_spider.json`), reproduced faithfully by the MQL and invisible to verifier-3.
  - **Agreement with the AI static review:** 111 / 120 (92.50%) overall labels; 9 disagreements (4 defects the AI missed: R084, R097, R101, R118; 3 AI over-confident: R009, R070, R111; R012 U→INCORRECT; R090 U→CORRECT by data evidence).
  - **Storage-type sweep** (`cross_review_sweep.py`): 52 / 120 items use a column in an order, aggregate or range role (70 uses); text-stored numeric or non-ISO date columns flagged in 5 / 120 items (R070, R084, R101, R111; R106 text-vs-literal, resolved CORRECT); no new cases beyond those found by reading.
  - **Status:** P2/P3 still PENDING (human reviewer required); 93 accepted labels outside the sample unreviewed; no corpus written.

Earlier audits have not been back-filled yet. Sources: `docs/sql2mql_independent_audit.md`, `docs/current_dataset_inventory.md`, `docs/spider_sql_v1_dataset.md`.

- 2026-11-06: **Semantic quality audit of `spider_pool_v2/train.jsonl`** (dataset audit, SQL-side, **SQL-only**, no model metric; AI detectors + AI second pass + AI hand-read by Claude Sonnet 5.5, **not a human review**; local SQLite 3.50.4 / Python 3.13 / sqlglot 25.34.1, read-only; no MongoDB; **provisional**: recall limited, see below). Report: `docs/spider_pool_semantic_audit.md`; artifacts: `outputs/spider_pool_semantic_audit_v1/` (`audit_summary.json`, `record_audit.jsonl`).
  - **Statuses over 5,387 unique train records:** CONFIRMED_DEFECT 30 / 5,387 (0.56%); LIKELY_DEFECT 58 / 5,387 (1.08%); NEEDS_REVIEW 254 / 5,387 (4.72%); NO_ISSUE_DETECTED 5,045 / 5,387 (93.65%); BLOCKED 0 / 5,387 (0.00%). Counts are detector outputs, not an error-rate estimate.
  - **Execution/coverage:** parsed 5,387 / 5,387; executed 5,387 / 5,387; stored `expected_output` reproduced 5,387 / 5,387; independent re-derivation of CONFIRMED/LIKELY findings reproduced 89 / 89 (100%).
  - **Regression (previous 120-item review):** in-scope pilot-INCORRECT items flagged LIKELY or higher 5 / 5 (R012, R119 CONFIRMED; R084, R101, R118 LIKELY); R097 not in train (rejected earlier), caught as an out-of-scope control; UNCERTAIN R070, R111 NEEDS_REVIEW (latent), R009 missed (NO_ISSUE_DETECTED).
  - **Calibration (outside the 5,387):** recall on the 127 excluded/quarantined train rows: LIKELY or higher 31 / 127 (24.4%), REVIEW or higher 96 / 127 (75.6%). Pilot CORRECT items in scope: 105 / 110 clean, 5 NEEDS_REVIEW, 0 LIKELY or CONFIRMED (labels are AI-reviewed, not ground truth).
  - **Caveat:** low recall; the pilot's 6 / 120 (5.00%) wrong-label rate is the safer residual-risk guide. Corpus approval still needs the human P2 review. Nothing was modified or committed.
  - *Addendum 2026-11-06 (same entry, after an advisor-style self-review):*
    - **Counterexamples:** 28 / 30 CONFIRMED findings carry an executed answer-changing variant (direction flipped, operator swapped, quoted value substituted); the 2 truncated questions have no value to substitute; 2 operator findings (`flight_1` "at least 100000", `shop_membership` "more than 100") are *latent*: no stored row sits on the boundary, so both operators return the same answer on this database (still wrong mappings). Entity-count findings were demoted to LIKELY (15 records) because a human could accept counting history rows; this moved R118 from CONFIRMED to LIKELY.
    - **Method wording:** rules were specified before the first full run and tightened afterwards (not strictly pre-registered); added after the first run: quoted-value substitution rule and nested LIMIT-tie rule. The adversarial pass is independent only for direction/operator/order/literal findings; for numeric-text, non-ISO-date and entity-count findings it re-executes the engine's own corrected variant (confirms the answers differ, not that the correction is right).
    - **Data-protection boundary:** to explain why R097 is absent from `train.jsonl` the zip member `spider_pool/rejected.jsonl` (listed as not opened in the v2 manifest, holds dev rows) was opened and matched on one train ID, printing only id, database and reject reason; a repo-wide text search for that ID listed file names only. No dev/test question or SQL was read or used.
    - **Hardware / software:** local Windows 11 laptop CPU, Python 3.13 + SQLite 3.50.4 (profiles and all executions), sqlglot 25.34.1 in `outputs/spider_pool_semantic_audit_v1/deps/`; no GPU, no network model calls.
    - **Checks not run:** human review (pilot gate P2); re-execution on SQLite 3.40.1 and 3.53.1; independent recount by separate code was done once by hand (5,387 unique IDs = `train.jsonl`; statuses 30 / 58 / 254 / 5,045 / 0 from `record_audit.jsonl` and the four status files) but `s5_finalize.py` is the only generator; no audit of `val_internal.jsonl`; nested/set-operation queries only partly covered.
    - **Concentrations are single-column effects** (`Hotels.price_range` 8, `wrestler.Days_held` 8, `branch.membership_amount` 6): a column policy fits better than per-record exclusion.
    - **sha256:** `train.jsonl` `09407ae46f14…` (unchanged, equals manifest); `record_audit.jsonl` `0b77e529dfe6…`; `audit_summary.json` `d70b1846ffe2…`; scripts `common.py` `0beac0445a4c…`, `lex.py` `5668deaa795f…`, `engine.py` `d9e586c5a312…`, `s1_profile.py` `b4dd8d04ad98…`, `s2_run.py` `26c80aadeb9f…`, `s3_calibrate.py` `0f0b9fbdfd9d…`, `s4_adversarial.py` `857210ac5496…`, `s4b_evidence.py` `9bda1b1be388…`, `s5_finalize.py` `2819cc659738…`, `s6_report.py` `44b8e388102f…`, `s7_hashes.py` `492cce3a89de…`. Full hashes are in `audit_summary.json` (`sha256`).

### DA-2026-11-06-CORPUS1: provisional RAG corpus v1 (sql2mql-v2 + verifier-3 on the 5,045 audit-clean Spider train labels)

- **Status: verified** for counts and artifact hashes (13 tests pass; manifest hashes recomputed; derived files rebuilt byte-identically from `records.jsonl`). **Not** human-reviewed; not a model metric. Label-validation rate measured on **local** MongoDB 8.0.34 (container `codegen-mongo-8034`, 127.0.0.1:27018), not Atlas. Produced on 2026-11-06, CPU laptop.
- **Data:** `data/processed/spider_pool_v2/train.jsonl` (5,387), restricted to semantic-audit `NO_ISSUE_DETECTED` = 5,045. Pipeline: converter `sql2mql-v2` + frozen verifier-3, all 19 code hashes checked; runner `scripts/finetune_v2/run_corpus_v1.py`, freeze `data/interim/finetune_v2/corpus_v1_runner_freeze.json`.
- **Results (of 5,045):** converted 3,962 (78.53%); ACCEPTED 3,962 (78.53%); UNSUPPORTED 1,083 (21.47%); REJECTED, INVALID, TIMEOUT, EXECUTION_ERROR, VERIFICATION_ERROR all 0. Accepted but excluded from the corpus 373 (3+-table rule 342, SQL-hash overlap with an evaluation file 30, pilot cross-review UNCERTAIN 1). **Corpus size 3,589 (71.14%)**.
- **By hardness (accepted):** easy 1,332 / 1,349 (98.74%); medium 1,837 / 2,033 (90.36%); hard 555 / 1,014 (54.73%); extra 238 / 649 (36.67%). Subqueries (357) and set operations (355): 0 converted.
- **Why 0 verifier rejections:** the pilot's 11 tie-cut rejections are all outside the audit-clean set; the audit's `limit_ties` detector had removed them. Acceptance here is not comparable with the pilot's 71.0% (different population).
- **Leakage:** 0 database overlap with 11 evaluation files (84 eval DBs vs 123 corpus DBs); 0 question-hash hits; 30 SQL-text hash hits (generic SQL on different DBs) excluded conservatively. Re-run vs pilot run01 on 252 shared ids: 252 / 252 substantively identical, 236 / 252 strictly (16 differ in the known `v6_determinism` order note).
- **Caveats:** verifier-3 cannot see a wrong gold SQL; the pilot's AI cross-review found 6 / 120 (5%) wrong plus 3 uncertain. Converter-dev (415) and round-3 (621) slices are in-sample for the converter.
- **Checks not run:** human review (P2); second full re-execution (only the pilot-overlap comparison); Atlas execution.
- **Artifacts:** `outputs/sql2mql_v2/provisional_corpus_v1/` (`manifest.json`, `conversion_report.md`, `provisional_rag_corpus.jsonl` sha256 `61553d03fe42…`, `conversion_status.jsonl` `c367e7f9e5db…`, `records.jsonl` `b7526994d11f…`).

### DA-2026-11-06-MQL0007-PREFLIGHT: MQL-0007 near-duplicate audit, arm A compatibility and scoring-infrastructure validation

- **Status: verified** (artifacts and tests re-run locally). **No model metric and no MQL-0007 result exists**; no GPU inference was run. Local Windows CPU; fake read-only Atlas in tests (no real Atlas contact).
- **Near-duplicate audit (token Jaccard ≥ 0.8, threshold fixed before running):** evaluation set vs corpus 2 / 892 items (`spider_dev_0383`, `spider_dev_0824` vs `spider_train_00650`, Jaccard 0.80); exact-question hits 0 / 892; database overlap 0. Spider dev (1,034, incl. the 65 held-out): exact question 0, exact SQL 0, near-duplicate 0 beyond the pair. Spider test (2,147): exact question 0, exact SQL 0, database overlap 0; template-level pairs 16 / 2,147 (13 distinct corpus ids, Jaccard 0.80–1.00, ≤ 2 tokens differ, all different databases, 0 equal SQL). The protected-set gate is membership (exact/SQL/database); near pairs are reported, not gated (chosen after seeing the 16).
- **Exclusion (user decision 2026-11-06):** deterministic post-ranking exclusion in arms B and C. Plan diff vs unexcluded: C changes on exactly 1 item (`spider_dev_0383`), B on 0; supplementary arm D = that 1 item with the exclusion off. Corpus (3,589) unchanged.
- **Arm A:** static compatibility 15 / 15 checks (model revision, decoding, bf16, batch 8, prompts byte-identical, 892 / 892 per-item prompt hashes). Library versions unknown until Kaggle; reuse is conditional (`ZERO_SHOT_MODE=auto`), fresh arm A is the likely branch.
- **Tests:** `tests/test_nl2mql_mql0007_scoring.py` + package tests 56 passed / 5 skipped (no tokenizer); package tests with the pinned tokenizer 29 / 29 passed (notebook run end to end with a fake generator, kill-and-resume, auto/strict arm A). Statistics checked against hand values (exact McNemar b=10, c=2 → 158/4096).
- **Checks not run:** real Atlas scoring; GPU run; behaviour under a real Kaggle environment.
- **Artifacts:** `outputs/nl2mql_mql0007_rag_v1/preflight/` (`analysis_plan.json`, `arm_a_compatibility.json`, `near_duplicate_investigation.json`, `sensitivity_config.json`, `statistics_validation.json`, test result logs); package `kaggle/nl2mql_mql0007_rag_v1/`, zip sha256 `da944ad28fd7…`.

### DA-2026-11-12-P2-MULTIAGENT: multi-agent AI semantic audit of the 120 pilot_v3 run01 P2 review items

- **Status: verified** for artifacts and counts. Checks done:
  - 5 JSONL files × 120 ids validated;
  - 292 cited executor log entries re-executed: 253 had an identical result hash, and the other 39 had the same rows in a different order (`$group` without `$sort`);
  - 125 source hashes unchanged;
  - aggregates recomputed by `tools/aggregate.py`.

  This is an **AI-assisted audit (4 Claude Opus 5.5 subagents, same model). It is NOT the P2 human-review gate**, and the workbook's reviewer columns are still blank. Label-validation evidence comes from **local** SQLite (`mode=ro`) and MongoDB 8.0.34 (`codegen-mongo-8034`, `spider_train_v2`), not Atlas. It is not a model metric. Run on 2026-11-12, CPU laptop.
- **Data:** `outputs/sql2mql_v2/pilot_v3/run01/review_package/review_items.jsonl` (120 items, sha256 `f2ddda16…`); converter sql2mql-v2 with verifier-3. Protected 65-item test partition not opened.
- **Method:** Agent 1 (source SQL vs question), Agent 2 (MQL vs SQL), Agent 3 (adversarial, both), Agent 4 (independent pass, then adjudicated all 32 disputed items with fresh executor evidence; no majority vote).
- **Independence:** partial. Every agent's auto-loaded memory index showed the summary line "AI cross-review found 6/120 wrong source labels" (a count only, no item ids).
- **Results (final, of 120):**
  - source SQL wrong for its question: **10 / 120 (8.3%, Wilson 4.6–14.7%)**. All three source reviewers said NO on 4 / 120 (3.3%).
  - MQL ≠ SQL under the **strict** domain (any SQLite-storable value, including text in numeric columns): **20 / 120 (16.7%)**. These are 11 untyped `$gt`, 7 AVG/SUM over mixed types, and 2 Unicode `$regex` `i`.
  - MQL ≠ SQL under the **type-conformant** domain: **2 / 120 (1.7%, Wilson 0.5–5.9%)** (R040, R082).
  - None of the 20 differs on the real data.
  - both errors in one item: 0 / 120.
  - valid pairs: **90 / 120 strict; 108 / 120 type-conformant**.
  - unresolved: 0 / 120.
- **Agreement:** unanimous accept 88; unanimous reject 6; disagreement 26; adjudicated 32.
- **Checks not run:**
  - human review (P2);
  - Atlas execution;
  - per-item constructed counterexamples for the 88 undisputed items. A mechanical operator scan found no unflagged strict-domain pattern.
- **Artifacts:** `outputs/sql2mql_v2/pilot_v3/run01/manual_human_review/ai_multiagent_review/`:
  - `consensus_report.md`, `case_level_results.csv`, `review_methodology.md`, `agent_{1..3}_*.jsonl`, `agent_4_adjudication.jsonl`;
  - `_work/agent_N/exec_log.jsonl`, `tools/`.

### DA-2026-11-12-TYPE-CENSUS: plan-time type and fold-trap census (input to the Pilot V3 follow-up plan)

- **Status:** provisional. These are read-only measurements taken while writing `docs/pilot_v3_followup_implementation_plan.md` (§0, M1–M6). The census script was an ad-hoc command and is not committed; step S1 of the plan must reproduce these counts with `scripts/finetune_v2/type_contract.py`. Run 2026-11-12 on a CPU laptop. Sources: local SQLite (`mode=ro`), local MongoDB 8.0.34, and the stored Atlas mapping. **No Atlas query was run.**
- **Training databases** (139 databases used by spider_pool_v2 train and val_internal):
  - 3,824 columns;
  - 3,507 / 3,824 (91.7%) match their declared affinity;
  - 199 / 3,824 (5.2%) are homogeneous text under a numeric affinity, mostly dates (195);
  - 115 / 3,824 (3.0%) are mixed-class: baseball_1 90, bike_1 22, cre_Theme_park 1, formula_1 1, wine_1 1;
  - 3 have BLOB affinity;
  - columns holding U+212A or U+017F: 0.
- **Atlas evaluation mapping** (`kaggle/nl2mql_spider_dev_1034_bundle_v2/data/atlas_mapping.json`, types recorded 2026-10-20):
  - 70 collections, 383 fields;
  - 17 / 383 differ from their declared affinity, and 2 / 383 are mixed (`wta_1__players.birth_date`, `wta_1__rankings.player_id`).
- **LIKE semantics** (SQLite 3.40.1 vs MongoDB 8.0.34, `$documents`):
  - `$regex` with options `is` over-matches U+212A, U+017F and É on 4 / 7 patterns;
  - an ASCII character-class regex with options `s` equals SQLite on 7 / 7 patterns × 16 values, in both the `$regex` and `$regexMatch` forms.
- **Not a model metric.** No artifact was written apart from the plan document.
- **Correction (2026-11-12, same day).**
  - The LIKE line above should read "**5 / 7** patterns", not 4 / 7. The over-matching patterns were the Kaitlyn and Scanner patterns, `é`, `k` and `S`; only `a_b` and `%ix%` were exact.
  - **Additional measurement:** non-letter characters with Unicode case pairs (U+24B6/U+24D0, U+2160/U+2170) also over-match under `is`, and they pass sql2mql-v2's `isalpha()` pattern guard. The ASCII-class regex with options `s` matched SQLite on all of them.
  - None of the 138 spider_pool_v2 train LIKE patterns or the 124 corpus v1 `$regex` labels contains a non-ASCII character.

### DA-2026-11-13-TYPE-CONTRACT: TC-1 type contract and reproduction of the plan-time census (follow-up plan S1)

- **Status:** done 2026-11-13. Script `scripts/finetune_v2/type_contract.py` (build, build-eval, census, check-local); tests `tests/test_type_contract.py` (15 passed).
- **Reproduction of DA-2026-11-12-TYPE-CENSUS:** exact. Training DBs: 3,824 columns, 3,507 conforming, 199 homogeneous text under numeric affinity, 115 mixed, 3 BLOB. Atlas mapping: 383 fields, 17 differ, 15 text-in-numeric, 2 mixed (wta_1).
- **Manifests:** `data/interim/finetune_v2/type_contract_v1.json` (training; body sha256 6ffe0539…), `type_contract_eval_v1.json` (evaluation; body sha256 8a5b4696…).
- **Local check:** the local MongoDB import conforms to TC-1 on 728 / 728 collections × 2 representations (explicit null, missing). 16 databases had to be imported first (through verifier-3's RealDbs).
- **Not a model metric.**

### DA-2026-11-13-V21-REGRESSION: sql2mql-v2.1 + verifier-3.1 on the frozen 300-label pilot sample (follow-up plan S2–S6)

- **Status:** done 2026-11-13, **local MongoDB 8.0.34 only (not Atlas)**. Runner `scripts/finetune_v2/run_pilot_v4.py` (runner freeze sha256 1c105a1e…); code freeze `data/interim/finetune_v2/v2_1_code_freeze.json`.
- **Change:** LIKE emitted as ASCII character classes (`[kK]`) with `$regex` options `s` instead of `is`, in all three emission paths. Verifier-3.1 adds `fold_traps` / `fold_traps_dotted` fixture families (U+212A, U+017F; U+0130/U+0131 dotted family) for LIKE labels only.
- **Tests:** v2.1 LIKE unit tests 26 passed; engine tests 3 passed; verifier tests 4 passed; frozen-artifact tests 4 passed; the frozen v2 / round3 / pilot_v3 / corpus_v1 suites 112 passed.
- **Results (n = 300, same items as run01):**
  - accepted: run01 213, v2 control 213, v2.1 213;
  - v2 control vs run01: MQL identical 300 / 300, outcome identical 300 / 300;
  - C1 non-LIKE accepted labels: MQL byte-identical 205 / 205; non-LIKE outcomes identical 292 / 292;
  - C2 LIKE accepted labels: 8 / 8 differ only in the regex, same real-data row counts; fold-trap families agree (1 item degenerate: no substitutable letter);
  - C3 accepted / rejected sets equal (213 / 87).
- **Artifacts:** `outputs/sql2mql_v2_1/pilot_regression/{v21,v2_control}/records.jsonl`, `regression_report.json`.

### DA-2026-11-13-ATLAS-PREFLIGHT: live Atlas type-contract and LIKE checks (follow-up plan S7)

- **Status:** done 2026-11-13 on Atlas (server 8.0.37, read-only user). Script `scripts/nl2mql/atlas_type_contract_check.py` (12 tests passed).
- **Type contract:** PASS. 70 collections, 0 violations of `type_contract_eval_v1.json`; the only nonconformant fields are the 2 already recorded (`wta_1__players.birth_date`, `wta_1__rankings.player_id`). Report `outputs/pilot_v3_followup/preflight/atlas_type_contract_20261113T024032.json`.
- **LIKE semantics on Atlas:** PASS with `$documents` (ASCII-class regex, options `s`, equals SQLite; options `is` over-matches as locally). Report `atlas_like_check_20261113T024042.json`.
- **Not a model metric.** No evaluation question was executed.

### DA-2026-11-13-DETECTORS: pre-registered source-label detectors D1–D6 on spider_pool_v2 (follow-up plan S8)

- **Status:** done 2026-11-13. `scripts/source_label_audit_v1/{detectors.py,run_audit.py}`; pre-registration `data/interim/source_label_audit_v1/prereg.json` (sha256 59e96d21…), written before the pool scan. Detector parameters were tuned on the 120 run01 P2 items (development data) before pre-registration; tests 13 passed.
- **Recall on run01 (development, not held out):** all 9 must-detect items caught; 6 / 110 audit-accepted items flagged.
- **Pool scan (6,121 records, train + val_internal, local SQLite read-only):** 776 records with ≥ 1 quarantine hit (train 672, val_internal 104, recounted from hits.jsonl against the pool files; the per-split figures in `pool_summary.json` (676 + 104 = 780) are wrong). Hits: D1a 44, D1b 66, D2 3, D3a 225, D3b 96, D4_data 23, D5 39, D6a 303; report-only D1b_iso 121, D4_schema 33.
- **Siblings:** 9 sibling records of the run01 findings, all in train (R097 is not in the pool).
- **These are detector flags, not verified errors.** Precision is estimated by the AI panel (S9).

### DA-2026-11-16-AI-PANEL: AI review panel v1, calibration and G-AIR gate (follow-up plan S9, S10, section E)

- **Status:** done 2026-11-16. **AI-reviewed, not human-reviewed.** The frozen human P2 gate of run01 is unchanged: still pending, NOT passed. G-AIR is a separate, weaker gate that the user chose on 2026-11-12 in place of a new human review.
- **Scripts and inputs:** `scripts/ai_review_panel_v1/`; pre-registration `outputs/ai_review_panel_v1/prereg.json` (body sha256 977a2e8c...), written before any reviewer ran. User decisions of 2026-11-13: "all X" findings are quarantined; G-AIR n = 200; the gate bound applies to translation only; source labels are reported without a verdict.
- **Panel:** 4 reviewer roles per batch in isolated contexts (source, translation, adversarial, independent pass 1 then adjudication). Every non-YES vote was adjudicated with fresh execution evidence; no majority vote; unresolved items are quarantined. 578 main items in 6 batches, 116 stability items in 2 batches.
- **Calibration (planted errors, key hidden from reviewers):** source 60 / 60 caught, translation 60 / 60 caught, every planted class 1.0; clean controls 0 / 60 flagged (Wilson 95% upper 6.0%). Thresholds (>= 55 / 60, class rate >= 0.75) met.
- **Stability (repeat run):** source agreement 113 / 116 (kappa 0.931); translation 98 / 98 (kappa 1.0). Threshold kappa >= 0.6 met.
- **G-AIR (fresh sample of 200 accepted v2.1 labels from the pool, not run01):**
  - translation: 0 NO, 0 UNSURE of 200; Wilson 95% upper bound 1.88%; pre-registered bound 5%; **gate passed (AI-only)**;
  - source labels (report only): 6 NO + 1 UNSURE of 200 (NO rate 3.0%, Wilson 1.4%-6.4%); all 7 quarantined.
- **Detector precision (S9, panel source NO / sampled):** D1 26 / 40 including UNSURE, D2 0 / 1, D3 6 / 40, D4 4 / 21, D5 6 / 38, D6 8 / 40. Most detector hits are not confirmed errors; they stay in tier C (review), not tier A.
- **Main final labels (source / translation):** YES/YES 336, NO/YES 105, YES/NO 60, YES/NA 52, NO/NA 20, UNSURE/YES 3, NO/NO 2.
- **Evidence re-check (`respot.py`):** main 1,954 cited runs identical, 57 same rows in another order, 6 differ, 5 cite a logged error; stability 432 identical, 9 order-only, 1 differs. All 7 differing runs are `$group` + `$limit` without a sort.
- **Deviations (full text in `outputs/ai_review_panel_v1/deviations.jsonl`):**
  - DEV-1..3: one reviewer cited placeholder ids; mapped in the tools, tool hashes changed after pre-registration;
  - DEV-4: the dispute list includes the adjudicator's own pass-1 votes;
  - DEV-5, DEV-6: user stop/restart and a Docker/MongoDB outage; outage outputs archived and those roles rerun;
  - DEV-7: the last stability roles ran on a different model (Fable instead of Opus), so the stability kappa mixes model change with run-to-run variation;
  - VAL-1..4: some adversarial NO votes had no execution evidence (all adjudicated); two reviewers read the executor source (it holds no keys);
  - VAL-5: the TC-1 NULL rule was applied inconsistently in two adjudications; not relabelled.
- **Limits:**
  - all reviewers are instances of the same model family, so their agreement is not independent evidence and does not establish ground-truth accuracy;
  - a 200-item AI-only sample does not prove a real-world error rate below 5%;
  - planted classes are small (5-6 items each) and two error classes had no planted items (left_to_inner_all_x, untyped_gt_text);
  - isolation was partial: the memory index was not scrubbed, and every reviewer disclosed seeing its one-line summaries;
  - several source NO verdicts on text-stored-number sorting fail only on hypothetical data and are debatable.
- **Not a model metric.**

### DA-2026-11-16-POOL-V3: spider-pool-v3 (follow-up plan S10)

- **Status:** built 2026-11-16 at `data/processed/spider_pool_v3/`. spider_pool_v2 is untouched. No gold SQL was rewritten (`data/interim/source_label_adjudication_v1.jsonl`: 398 rows, `correction_applied: false` on all).
- **Counts:** train 5,387 -> 4,790 (597 removed); val_internal 734 -> 639 (95 removed).
- **Removed records by tier:** A (panel-adjudicated semantic defect) 26 (train 22, val 4); C (detector hit, awaiting review) 667 (train 575, val 91; 1 id, `spider_train_04071`, is not in the pool). 94 detector hits were cleared by the panel and stay in the pool.
- **Files:** `train.jsonl`, `val_internal.jsonl`, `quarantine.jsonl`, `exclusions.jsonl`, `exclusions_v3.jsonl`, `manifest.json`. IDs, provenance and exclusion reasons are preserved.
- **Not a model metric.**

### DA-2026-11-16-CORPUS-V2: provisional RAG corpus v2 and final checks (follow-up plan S6, S11)

- **Status:** built 2026-11-16, **local MongoDB 8.0.34 only (not Atlas)**; provisional, AI-reviewed, not human-reviewed. Corpus v1 is untouched. No training or RAG experiment was run.
- **Conversion (sql2mql-v2.1 + verifier-3.1, 5,045 candidates):** accepted 3,962; conversion_rejected 1,083; no infrastructure-error records. The run was interrupted several times (low memory, Docker outages) and resumed from its record file each time.
- **Corpus v2:** 3,447 records included; 515 accepted records excluded.
  - Exclusion reasons over all candidates (a record can have more than one): conversion rejected 1,083; pool-v3 tier C 386; three or more tables 505; SQL-hash leakage 36; run01 cross-review UNCERTAIN 1.
  - Against corpus v1 (3,589): 142 removed, 114 MQL changed (LIKE regex only; 0 unexpected non-LIKE changes).
- **Checks (`build_corpus_v2.py finalize`): pass.** No field outside the type-contract manifest; no manifest hash mismatch; no database overlap with Spider dev; 0 duplicate ids; every corpus id is in spider-pool-v3 train. 6 records reference a field recorded as nonconformant in TC-1 (`bike_1__weather.cloud_cover` 2, `cre_Theme_park__Hotels.price_range` 4); reported, not excluded.
- **Frozen artifacts:** `hash_baseline.py verify final` ok: 217 files unchanged, 1 protected file not opened, no file added inside a frozen directory.
- **Full test suite:** 578 passed, 3 failed.
  - `test_finetune_v2.py::test_no_atlas_credentials_are_read` fails on the word "environment" in a comment at `run_pilot_v4.py:278`; the file reads no environment variable and no Atlas URI. Not edited, because the runner hash is recorded in the code freeze.
  - `test_kaggle_sql_v1_baseline.py` (2 tests) fail on two stray `__pycache__` files dated 2026-10-29 inside `kaggle/spider_sql_v1_dev_bundle_v1/code/official_eval/`; older than this work and not touched.
- **Atlas:** only the S7 preflight (type contract, LIKE) ran on Atlas. No v2.1 label and no corpus record has been executed on Atlas; that gate is pending.
- **Not a model metric.**

### DA-2026-11-16-CORPUS-V2-SIGNOFF: pre-training sign-off of corpus v2 (tests, six-record adjudication, Atlas execution audit)

- **Status:** done 2026-11-16. **Outcome: BLOCKED; fine-tuning not authorized.** AI-reviewed, not human-reviewed. Report: `docs/corpus_v2_signoff_report_2026-11-16.md`.
- **Atlas execution audit** (`scripts/corpus_v2_signoff/atlas_exec_audit.py`, run `outputs/corpus_v2_atlas_audit/run_20261116T183448/`, live Atlas 8.0.37, read-only account with 0 write actions):
  - records discovered 3,447; attempted 0; validated 0; skipped 3,447 (collection absent on Atlas);
  - 0 of the 516 collections read by the corpus (123 Spider training databases) exist on Atlas, which holds only the 80 evaluation collections; the corpus is disjoint from them by design;
  - offline read-only safety gate: 3,447 / 3,447 labels pass;
  - TC-1 re-check of the 70 evaluation collections: 0 violations;
  - Atlas not modified; protected holdout not opened.
- **Protocol decision:** G-AIR is the operational review gate for this dataset version and passed under its AI-only procedure; the human P2 gate of run01 remains NOT PASSED. Recorded as a protocol amendment in `docs/g_air_protocol_decision_2026-11-16.md`. The 0 / 200 translation result does not establish a population error rate below 5%.
- **Six nonconformant-field records** (`data/interim/corpus_v2_adjudication_v1.jsonl`, by rule, AI-adjudicated): 2 quarantined (`spider_train_00145`, `spider_train_00146`: `$group` on `bike_1__weather.cloud_cover`); 4 included (`spider_train_05887`, `05888`, `05895`, `05896`: `cre_Theme_park__Hotels.price_range` projected only). No SQL or MQL rewritten.
- **Candidate corpus:** `outputs/sql2mql_v2_1/corpus_v2_1_candidate/` holds 3,445 records (corpus v2 minus the 2), status `CANDIDATE_NOT_APPROVED`. Provisional corpus v2 is unchanged.
- **667 detector-hit records:** kept in quarantine as a conservative exclusion; not confirmed defects; not re-reviewed.
- **Tests:** `python -B -m pytest tests -q -p no:cacheprovider -rs`: 591 collected, 591 passed, 0 failed, 0 skipped (before: 581 collected, 578 passed, 3 failed). This corrects the 3 failures listed in DA-2026-11-16-CORPUS-V2; `run_pilot_v4.py` was not edited.
- **Frozen artifacts:** `hash_baseline.py verify signoff` ok (217 files unchanged).
- **Not a model metric.** No training or RAG experiment was run.

### DA-2026-11-16-CORPUS-APPROVAL: approved training corpus v2.1 under the local validation policy

- **Status:** done 2026-11-16. **CONDITIONAL GO — APPROVED FOR TRAINING UNDER LOCAL VALIDATION POLICY.** AI-reviewed, not human-reviewed; human P2 NOT PASSED.
- **Owner decision (2026-11-16):** accept local MongoDB verification for training labels; reported evaluation metrics must come from Atlas. This lifts the BLOCKED status of DA-2026-11-16-CORPUS-V2-SIGNOFF. Training-corpus execution on Atlas was NOT PERFORMED (516 training collections absent from Atlas; accepted limitation, not a passed audit).
- **Artifact:** `outputs/sql2mql_v2_1/approved_corpus_v2_1/` — `corpus.jsonl` 3,445 records, sha256 96552f061afba13792d75240c18001e87e70f4aebf9d0934dfe27c0a7dd199bf; reproducible as provisional corpus v2 minus `spider_train_00145` and `spider_train_00146`.
- **Local validation (MongoDB 8.0.34, verifier 3.1):** 3,445 / 3,445 records with complete accepted verification records; live re-verification of a seeded sample 64 / 64.
- **Integrity:** 14 / 14 approval checks pass; 217 frozen files unchanged; holdout not opened.
- **Tests:** 594 collected, 594 passed, 0 failed, 0 skipped.
- **Note on DA-2026-11-16-AI-PANEL:** the 7 of 200 flagged source labels (6 NO, 1 UNSURE) are an observed sample result from an AI panel; they are not a reliable corpus-wide error-rate estimate.
- **Not a model metric.** No training was started.

### DA-2026-11-16-MQL0008-PACKAGE: MQL-0008 package (zero-shot vs hybrid RAG on approved corpus v2.1) and fresh leakage audit

- **Status: PACKAGE BUILT — NOT YET EXECUTED.** No model metric, no MQL-0008 result. Kaggle, GPU inference and Atlas scoring were not run. Local Windows CPU only.
- **Experiment:** MQL-0008, two arms only: `arm_a_zero_shot` and `arm_b_hybrid_rag`; Qwen/Qwen2.5-Coder-3B-Instruct @ 488639f1ff808d1d3d0ba301aef8c11461451ec5, bfloat16, greedy, max_new_tokens 1024, batch 8; the 892 MQL-0006 items.
- **Retrieval corpus:** `outputs/sql2mql_v2_1/approved_corpus_v2_1/corpus.jsonl`, 3,445 records, sha256 96552f061afba13792d75240c18001e87e70f4aebf9d0934dfe27c0a7dd199bf (AI-reviewed, not human-reviewed; never executed on Atlas).
- **Retrieval:** frozen MQL-0007 hybrid configuration, unchanged (BM25 k1 1.5, b 0.75, pattern weight 1.0, K 3, 1,500-token cap, target database excluded). Only the corpus changes.
- **Leakage audit (local recompute):** exact question 0 / 892; normalised question 0 / 892; database overlap 0; corpus SQL equal to an evaluation gold SQL 0 / 3,445; near-duplicates 2 pairs (`spider_train_00650` with `spider_dev_0383`, `spider_dev_0824`), excluded from retrieval for those two items; 0 evaluation items removed. Protected sets not opened (subset carry-over from corpus v1, 3,445 / 3,445 records identical).
- **Retrieval plan:** 892 / 892 items with 3 examples; largest block 997 tokens; 0 from the target database.
- **Tests:** project suite 594 / 594 passed. Package tests 31 / 31 passed without tokenizer; with the pinned tokenizer the run was stopped by the system (low memory) at 33 of 34 passed, 0 failed; the notebook kill-and-resume test was not completed. Validation is therefore **incomplete** until that test is re-run.
- **Artifacts:** `kaggle/nl2mql_mql0008_approved_corpus_v2_1/`, `kaggle/nl2mql_mql0008_approved_corpus_v2_1_bundle.zip` (sha256 afa0aa67e44d018f0c7c22756429d012928115da1f75dc1d2b730d48c1083551), `outputs/nl2mql_mql0008_approved_corpus_v2_1/preflight/` (`package_validation_report.md`, `leakage_report.json`, `retrieval_audit.json`).
- **Open:** a two-arm scoring adapter and analysis plan are still to be written before scoring.
- **Note (2026-11-16):** MQL-0007 was executed on 2026-11-07 on corpus v1 (A 421 / 892, C hybrid 437 / 892) but has no ledger entry yet; the statement in DA-2026-11-06-MQL0007-PREFLIGHT that no MQL-0007 result exists was true only at that time. Those figures are historical and are not MQL-0008 results.
- **Note (2026-11-16, later the same day): validation completed.** The package test suite was re-run in one pass with the pinned tokenizer: 34 collected, 34 passed, 0 failed, 0 skipped, including the notebook kill-and-resume test that had not finished. This supersedes the "incomplete" wording in the Tests bullet above. Package status: **PACKAGE READY — awaiting Kaggle execution** (still not executed; no MQL-0008 result).

### DA-2026-11-16-MQL0008-SCORING-READY: MQL-0008 scoring adapter and pre-registered analysis plan

- **Status: SCORING READY — AWAITING KAGGLE PREDICTIONS.** No MQL-0008 prediction or score exists. Kaggle was not run and Atlas was not contacted; all scoring tests use a fake read-only Atlas. Local Windows CPU.
- **Adapter:** `scripts/nl2mql_mql0008/mql0008_scoring.py` (verify / score / analyze), `scripts/nl2mql_mql0008/mql0008_stats.py`, `scripts/nl2mql_mql0008/README.md`. The evaluator, safety gate, limits (30 s, 25,000 rows), comparison and outcome categories are the unchanged MQL-0006 core (`scripts/nl2mql/dev1034_core.py`).
- **Analysis plan (registered before any prediction):** `outputs/nl2mql_mql0008_approved_corpus_v2_1/preflight/analysis_plan.json`, plan sha256 985b2da6732c7222d20370c607c31b7b8e72b85b5bee2d2f2b2debc126604eb6. Metric correct / 892, nothing excluded. One primary comparison (hybrid RAG vs zero-shot): exact two-sided McNemar, alpha 0.05, no multiplicity correction; Wilson 95% per arm; paired bootstrap 95% (10,000 resamples, seed 20261029). Secondary analyses are exploratory.
- **Input validation:** 892 unique ids per arm in order, same ids in both arms, experiment id, model and revision, approved corpus sha256 and count, package manifest sha256, frozen prompts, empty context in arm A, retrieval log and per-record retrieval metadata equal to the frozen plan in arm B, target-database exclusion, export checksums, no held-out database, no extra file or arm.
- **Tests:** `tests/test_nl2mql_mql0008_scoring.py` 32 / 32 passed. Full project suite 626 collected, 626 passed, 0 failed, 0 skipped.
- **Unchanged (hash-checked after the tests):** MQL-0008 package (19 files, manifest fb20c496fa20fbc5eecc60b1407ac3858a6370cd607df257709b3cea4e26fed8, bundle afa0aa67e44d018f0c7c22756429d012928115da1f75dc1d2b730d48c1083551), approved corpus, MQL-0007 package and results report.
- **Supersedes** the "Open" bullet of DA-2026-11-16-MQL0008-PACKAGE (the two-arm adapter and plan now exist).

### DA-2026-11-18-MQL0009-LEAKAGE: MQL-0009 evaluation split and leakage audit (held-out Spider-train vs approved corpus v2.1)

- **Status: verified.** Local recompute by script (`scripts/nl2mql_mql0009/build_mql0009_package.py`), re-checked by tests. No model was run, there was no database connection, and no gold SQL was executed.
- **Split:** 639 items, 16 DBs (797 → 744 → 734 → 639 funnel); all `exec_ok`. Registration: `outputs/nl2mql_mql0009_retrieval_control/preflight/eval_split_registration.json`.
- **Gates (registered before running):**

  | Gate | Hits |
  |---|---|
  | L1 DB overlap | 0 / 639 items |
  | L2 id overlap | 0 |
  | L2 source-index overlap | 0 |
  | L3 exact question | 4 pairs |
  | L4 normalised question | 5 pairs |
  | L5 token Jaccard ≥ 0.8 | 17 pairs |
  | → retrieval exclusions | 17 pairs on 13 items |
  | L6 normalised gold SQL equal to a corpus SQL (report only) | 7 / 639 items |
  | L7 masked SQL skeleton match (report only) | 383 / 639 items |

- **Fixed-example draw:** 0 skips for leakage.
- **Plans:** K=3 for every question of every demonstration run, with 0 over-budget drops.
- **Evidence:** `outputs/nl2mql_mql0009_retrieval_control/preflight/leakage_audit.json` and `fixed_selection.json`.

<!-- Release snapshot: entries after the completed MQL-0015 experiment (MQL-0016 and later dataset audits) are intentionally omitted from this branch. -->
