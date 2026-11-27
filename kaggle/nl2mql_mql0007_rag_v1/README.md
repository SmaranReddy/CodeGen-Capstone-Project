# MQL-0007 Kaggle package — RAG vs random vs zero-shot (generate only)

**Status: PREPARED, NOT RUN (package v1.1).** Nothing in this package has produced a model prediction. No MQL-0007 accuracy exists.

Question: does retrieving verified NL→MQL examples (`provisional_corpus_v1`, 3,589 entries, sql2mql-v2 / Verifier-3,
**provisional, not human-reviewed**) improve `Qwen/Qwen2.5-Coder-3B-Instruct` on the 892 MQL-0006 questions?

| arm | context | predictions |
|---|---|---|
| A_zero_shot | none | `ZERO_SHOT_MODE = "auto"` (default): MQL-0006's verified predictions are reused byte for byte **only if** the run environment equals MQL-0006's (torch 2.11.0+cu128, transformers 5.16.1, accelerate 1.14.0, tokenizers 0.23.1, Tesla T4) in addition to the identical model revision, prompt, decoding, bf16 and batch size; otherwise arm A is **generated fresh** in the same environment as B and C (about +2 h). `"reuse_mql0006"` = reuse or stop; `"fresh"` = always regenerate. |
| B_random_k3 | 3 random corpus examples, frozen seed 20261020 | generated |
| C_hybrid_k3 | MQL-0005 `hybrid_masked`: BM25 (k1 1.5, b 0.75) + 1.0 × pattern overlap, k=3, block ≤ 1,500 tokens, no guide cards, deterministic ties (score desc, doc_id), over-budget examples skipped, never truncated | generated |
| D_supp_allow_near_dup (supplementary) | arm C without the near-duplicate exclusion, only on the item(s) the exclusion changes (`spider_dev_0383`, 1 item) | generated automatically after C; sensitivity analysis only |

Identical across arms: model + revision `488639f1ff808d1d3d0ba301aef8c11461451ec5`, tokenizer, prompt base
(`nl-mql-spider1-v1` + one reference block before the final `Question:` line), **bfloat16, greedy, batch 8, max_new_tokens 1024, one GPU**.

## Near-duplicate handling (v1.1)

`spider_train_00650` is a template near-duplicate (Jaccard 0.80) of `spider_dev_0383` and `spider_dev_0824`. The corpus file is unchanged
(3,589). Arms **B and C** apply a deterministic evaluation-specific exclusion of that example for those two items **after ranking** (BM25
index, IDF and dedup untouched, so no other item's plan can change); the next-ranked examples take its place. Logged per item as
`excluded_by_rule` with the original rank. Proof (builder): the plan diff vs the unexcluded plan is exactly `{spider_dev_0383}` for C and empty
for B (for `spider_dev_0824` the example was only rank 10). Selection-level gate: no selected example has Jaccard ≥ 0.8 with its query.
The sensitivity arm D re-runs only `spider_dev_0383` with the exclusion off; the analysis plan (below) reports it separately.
Protected-set findings: `outputs/nl2mql_mql0007_rag_v1/preflight/near_duplicate_investigation.json`.

## What is (and is not) in the package

In: the complete 3,589-entry corpus (verbatim copy), the 892 evaluation inputs (`id`, `db_id`, `question` only), the frozen
zero-shot prompts, 17-database schema context, Spider `tables.json` subset for masking, frozen per-question retrieval plans
for arms B and C, the MQL-0006 zero-shot predictions (reuse), configs, code, tests, notebook, `package_manifest.json`
(sha256 of every file), `SHA256SUMS.txt`, `data/leakage_report.json`.

Never in: gold SQL, gold rows, SQLite databases, the held-out 65-item partition, `.env`/credentials, MQL-0006 scoring data.
No `.gz` files (Kaggle silently decompresses them). Atlas is **not** contacted from Kaggle; scoring runs later on Windows.

## Final launch checklist

1. Local: `python -m pytest kaggle/nl2mql_mql0007_rag_v1/tests tests/test_nl2mql_mql0007_scoring.py -q` and
   `python scripts/nl2mql/mql0007_preflight.py` (artifacts in `outputs/nl2mql_mql0007_rag_v1/preflight/`; analysis plan sha256 inside).
2. Local, BEFORE spending GPU time: Atlas read-only check of the evaluation data (MQL-0006 preflight, unchanged):
   `python scripts/nl2mql/local_atlas_scoring.py preflight` (needs the read-only `codegen_eval_readonly` connection string in your local environment; never put it on Kaggle).
3. Upload the ZIP as a **private** Kaggle dataset and build the notebook as below.
4. In the notebook set *Settings → Environment Preferences → **Pin to original environment*** right after the first run, so later versions and
   resumes keep the same libraries (the checkpoint fingerprint refuses a resume under different libraries). Pinning refers to the notebook's own
   first-run image, not to MQL-0006's; that is why arm A is auto-resolved.
5. Run arms one version at a time (recommended): version 1 `RUN_ARMS = ["A_zero_shot"]` (reuse is instant when the environment matches, else ~2 h),
   version 2 `["B_random_k3"]`, version 3 `["C_hybrid_k3"]` (also runs arm D, 1 item). **Use the same pinned environment for every version**;
   `scripts/nl2mql/mql0007_scoring.py` refuses arms whose torch/transformers/accelerate/tokenizers/GPU/dtype differ.
6. Download each version's `mql0007_predictions.zip`; score all of them together (below).

## Local scoring (Windows, read-only Atlas)

```
python scripts/nl2mql/mql0007_scoring.py verify --predictions-zip A.zip --predictions-zip B.zip --predictions-zip C.zip
python scripts/nl2mql/mql0007_scoring.py score  --predictions-zip A.zip --predictions-zip B.zip --predictions-zip C.zip --run-name v1
```
Offline verification first (no Atlas): each arm exactly once from a complete output (identical duplicates tolerated), exactly the 892 ids in order, no
missing/duplicate/unexpected/malformed record, prompt hashes equal to the package's frozen/planned prompts, model/dtype/decoding/batch size,
fingerprints, retrieval logs equal to the frozen plans (exclusion preserved), identical environment across arms, `MQL-0007` experiment id and package hash.
Then the read-only Atlas preflight (fails closed on any write privilege), then the unchanged MQL-0006 evaluator. Output:
`outputs/nl2mql_mql0007_rag_v1/<run-name>/` (`scoring/results_report.md`, `analysis.json`, per-arm `evaluation_results.jsonl`). Nothing under
`outputs/nl2mql_spider_dev_1034_model_comparison` is read or written. Per arm eight categories sum to 892: correct, incorrect, invalid,
execution_error, timeout, generation_failure, infrastructure_failure (Atlas outage; retried on re-run), not_comparable.
Statistics (pre-registered in `analysis_plan.json`): primary C vs A and C vs B, exact McNemar with Holm over the two; B vs A secondary; Wilson 95%;
paired bootstrap; breakdowns by hardness, database, gold-SQL pattern; sensitivity S1 (arm D swapped into C), S2 (890 items without the two
near-duplicate items), S3 (reference_status ok).

## Kaggle execution (exact steps)

1. **Dataset:** Kaggle → *Datasets → New Dataset* → upload `kaggle/nl2mql_mql0007_rag_v1_bundle.zip` (Kaggle extracts it; the
   package appears under `/kaggle/input/<dataset>/nl2mql_mql0007_rag_v1/`). Keep it **Private**. Title e.g. `nl2mql-mql0007-rag-v1`.
2. **Notebook:** *New Notebook → File → Import Notebook* → `nl2mql_mql0007_rag_v1.ipynb` (also inside the zip). *Add data* → your private dataset.
3. **Settings:** Accelerator **GPU T4 x2** (one card is used), Internet **On**, Persistence *Files only*. No secrets attached.
4. **Run:** *Save Version → Save & Run All (Commit)*. First cell block = config; leave defaults for the full experiment.
5. **Download** `/kaggle/working/mql0007_predictions.zip` (also `mql0007_outputs/`): per arm `models/<arm>/predictions.jsonl`,
   `run_config.json`, `retrieval_log.jsonl` (selected ids, scores, dropped-for-budget, block tokens, prompt hashes), plus
   `MQL0007_STATUS.json`, `preflight/preflight.json`, `SHA256SUMS.txt`.

**Time budget (estimate, not measured for RAG prompts).** MQL-0006 took ~48 min per 176 items (≈ 49 s per batch of 8, zero-shot,
~1k-token prompts). RAG prompts are ~2–3.2k tokens: expect slower batches and more batch-size-1 OOM fallbacks. Plan **~3–5 h per
generated arm**; B and C together may not fit one 12-hour session comfortably. Recommended: run `RUN_ARMS = ["B_random_k3"]`
in one version and `["C_hybrid_k3"]` in another (arm A is reused, no GPU time). Order-independent: arms have separate checkpoints. Arm D adds one prompt (effective batch 1).

**Resume.** If a version stops (timeout, OOM, disconnect), start a new version, *Add data → Notebook Output* of the stopped version,
set `RESUME_FROM = "auto"` and keep the same `RUN_ARMS`. The earlier output is hash-verified, its checkpoints are copied, and
only unfinished ids are generated (set `RUN_ARMS` as before) (completed records are never regenerated; a different configuration/fingerprint is refused).
Re-running in the same working directory also resumes from `checkpoints/`.

## What the notebook checks (and fails on)

GPU present, ≥ 14 GiB, bfloat16 supported (no precision fallback) · every package file's sha256, no extra files · corpus exactly 3,589
unique entries, evaluation exactly 892 ids / 17 DBs with the recorded ids hash · zero DB overlap, zero exact matches, only the
user-acknowledged template near-duplicate (below) · all corpus rows train/accepted/not excluded · pinned model revision · corpus
features re-derived == frozen · **retrieval plans re-derived with the real tokenizer == frozen plans** (any difference stops) · retrieval
never from the target database · arm A prompts byte-identical to the frozen zero-shot prompts · MQL-0006 reuse eligibility · library
versions vs MQL-0006 (printed, recorded) · no duplicate/missing/out-of-order ids, every prediction from its planned prompt.

## Known caveats (stated, not hidden)

- **Near-duplicate template pair (acknowledged 2026-11-06):** `spider_train_00650` ("…names of editors in ascending order of age")
  vs `spider_dev_0383` and `spider_dev_0824` (same template, one noun differs, different databases), token Jaccard exactly 0.80.
  The corpus is unmodified; the two items are flagged `near_duplicate_template_item` in the retrieval logs so results can be re-cut.
- Protected sets: membership checks (exact question, exact SQL, DB overlap) are zero for Spider dev (incl. the 65 held-out) and Spider
  test. Spider test has 16 *template-level* near-duplicate pairs (Jaccard ≥ 0.8) with the corpus; reported, not gated.
- Corpus is **provisional** (`human_review_status` NOT_REVIEWED for all 3,589); labels establish SQL→MQL result equivalence, not that the SQL answers the question.
- Deviations from MQL-0005 (see `configs/experiment.json`): HF tokenizer for budgeting (MQL-0005 used the GGUF tokenizer), HF bf16 instead of GGUF q4_k_m,
  corpus masking words from Spider `tables.json`, `text_number` flag from touched columns.
- **Scoring** uses `scripts/nl2mql/mql0007_scoring.py` (multi-arm adapter). The MQL-0006 scorer and its results are untouched.
- MQL-0006's 96 batch-1 OOM items and library versions: if the Kaggle image differs, arm A is regenerated automatically (see table).

## Verify the package locally

```
python -m pytest kaggle/nl2mql_mql0007_rag_v1/tests -q
# full notebook tests (fake generator, no GPU) additionally need `transformers` and the pinned tokenizer files:
MQL0007_TOKENIZER_DIR=<dir> python -m pytest kaggle/nl2mql_mql0007_rag_v1/tests -q
python -c "import sys; sys.path.insert(0,'kaggle/nl2mql_mql0007_rag_v1/src'); import mql0007_core as C; C.verify_package('kaggle/nl2mql_mql0007_rag_v1')"
```

Rebuild (needs the repo): `python scripts/nl2mql/build_mql0007_package.py --tokenizer-dir <dir>` then
`python scripts/nl2mql/make_mql0007_notebook.py` and `python scripts/nl2mql/build_mql0007_package.py --finalize`.
