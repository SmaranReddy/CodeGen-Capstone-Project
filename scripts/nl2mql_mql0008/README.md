# MQL-0008 scoring adapter and analysis plan

**Status: SCORING READY — AWAITING KAGGLE PREDICTIONS.** No MQL-0008 prediction exists and Atlas has not been contacted
for MQL-0008. Nothing here is a result.

- `mql0008_scoring.py`: validates the Kaggle export, scores both arms on Atlas (read-only), writes the analysis and report.
- `mql0008_stats.py`: the pre-registered analysis plan and the statistics.
- Analysis plan (registered 2026-11-16): `outputs/nl2mql_mql0008_approved_corpus_v2_1/preflight/analysis_plan.json`,
  sha256 of the plan body `985b2da6732c7222d20370c607c31b7b8e72b85b5bee2d2f2b2debc126604eb6`.
- Tests: `tests/test_nl2mql_mql0008_scoring.py` (fake Atlas; no network).

What is reused unchanged: the MQL-0006 evaluator (`scripts/nl2mql/dev1034_core.py`: safety gate, read-only execution,
30 s timeout, 25,000-row cap, result comparison, outcome classification, checkpoints, Atlas preflight), the read-only
client (`scripts/nl2mql/local_atlas_scoring.py`), and the outcome categories and statistics primitives used for MQL-0007.
No result-equivalence logic is implemented here.

## Input contract

One zip with both arms, or one zip per arm, as written by the MQL-0008 notebook (`mql0008_predictions.zip`):

```
mql0008_outputs/
  MQL0008_STATUS.json                 run manifest: experiment, package/config/ids hashes, corpus, model, per-arm prediction hashes
  SHA256SUMS.txt                      checksum of every other file
  preflight/preflight.json
  predictions/arm_a_zero_shot/        predictions.jsonl (892), run_config.json, retrieval_log.jsonl, checkpoints/predictions.partial.jsonl
  predictions/arm_b_hybrid_rag/       the same four files
```

The export is refused unless all of this holds:

1. Each arm has exactly 892 unique ids, in the evaluation order, every record `status: ok`.
2. Both arms cover the same ids; each arm appears once (a second copy must be byte-identical).
3. Experiment id `MQL-0008`; export status `complete`.
4. Model `Qwen/Qwen2.5-Coder-3B-Instruct` at `488639f1ff808d1d3d0ba301aef8c11461451ec5` in the status, the run config and every record;
   bfloat16, greedy, 1,024 new tokens, batch 8, seed 42.
5. Corpus sha256 `96552f061afba13792d75240c18001e87e70f4aebf9d0934dfe27c0a7dd199bf`, 3,445 records.
6. Package manifest sha256 `fb20c496fa20fbc5eecc60b1407ac3858a6370cd607df257709b3cea4e26fed8` (the local package is re-verified file by file).
7. Every prediction was generated from its frozen prompt (prompt hash).
8. Arm A: no retrieval metadata and an empty-context retrieval log.
9. Arm B: the retrieval log equals the frozen plan; each record's retrieved ids, ranks, scores and budget fields equal the plan;
   every retrieved example is in the approved corpus and not from the target database or any evaluation database.
10. Every file matches `SHA256SUMS.txt`, and each arm's prediction file matches the hash in the export status.
11. No record on a held-out database (`employee_hire_evaluation`, `pets_1`, `tvshow`) and no id outside the 892.
    The held-out partition itself is never opened.
12. No unknown arm folder and no extra file.
13. Same libraries and GPU in both arms (override only with `--allow-env-drift`, which is recorded).

Nothing is repaired, reordered or substituted.

## How to score the Kaggle export

Run on the local Windows machine, with the read-only Atlas connection string available in the local environment only
(the same setup as for MQL-0006 and MQL-0007).

```
# 1. offline validation, no Atlas
python scripts/nl2mql_mql0008/mql0008_scoring.py verify --predictions-zip mql0008_predictions.zip

# 2. score both arms on Atlas (read-only)
python scripts/nl2mql_mql0008/mql0008_scoring.py score --predictions-zip mql0008_predictions.zip

# if the arms were exported separately
python scripts/nl2mql_mql0008/mql0008_scoring.py score --predictions-zip A.zip --predictions-zip B.zip

# 3. resume after an interruption or an Atlas outage: same command with the run name that was printed
python scripts/nl2mql_mql0008/mql0008_scoring.py score --predictions-zip mql0008_predictions.zip --run-name run_<timestamp>

# 4. recompute the analysis and report from saved per-record results, no Atlas
python scripts/nl2mql_mql0008/mql0008_scoring.py analyze --run-dir outputs/nl2mql_mql0008_approved_corpus_v2_1/scoring_runs/run_<timestamp>
```

Exit code 0 = complete, 2 = partial (unresolved Atlas outages; re-run step 3), 1 = validation failed.

## Outputs

`outputs/nl2mql_mql0008_approved_corpus_v2_1/scoring_runs/run_<UTC timestamp>/`

| file | content |
|---|---|
| `input_validation_report.json` | every validation check and its result |
| `predictions/<arm>/` | the validated export files that were scored |
| `run_log.jsonl` | sessions, per-arm counts scored / already final, failures, resumes |
| `scoring/preflight/atlas_check.json` | server version, authenticated user, privileges (read-only proof), collection counts |
| `scoring/RUN_MANIFEST.json` | status, accounting, limits, input hashes, plan hash, code hashes |
| `scoring/arms/<arm>/evaluation_results.jsonl` | one record per item (892) |
| `scoring/arms/<arm>/summary.json`, `checkpoints/` | accounting; resumable checkpoints |
| `scoring/paired_comparison.jsonl` | per item: both outcomes, pair class, retrieved ids, ranks, scores |
| `scoring/analysis.json` | primary analysis and exploratory diagnostics |
| `scoring/results_report.md` | human-readable report |
| `scoring/SHA256SUMS.txt` | checksums of the scoring outputs |

## Metric and statistics (pre-registered)

- Metric: correct / 892 per arm. Outcome categories: correct, incorrect, invalid, execution_error, timeout,
  generation_failure (model failure), infrastructure_failure (Atlas outage, retried), not_comparable. Only `correct` counts;
  nothing is removed from the denominator.
- One primary comparison: arm B vs arm A, paired on the evaluation item. Exact two-sided McNemar test, alpha 0.05. No
  multiple-comparison correction.
- Reported: correct and accuracy per arm, difference in percentage points, Wilson 95% interval per arm, the 2×2 paired
  table, items RAG fixes, items RAG breaks, McNemar p, and a paired bootstrap 95% percentile interval of the difference
  (10,000 resamples, seed 20261029).
- Claim rule: RAG is called better or worse only if p < 0.05. Otherwise the report says there is no significant difference.
- Exploratory only: category counts and transitions, breakdowns by hardness, database and query pattern, retrieval
  coverage, changed items with retrieved ids and scores, and two sensitivity cuts (without the two near-duplicate items;
  reference status ok).

## Historical context (not MQL-0008 results)

| Experiment | Corpus | Accuracy |
|---|---|---:|
| MQL-0006 zero-shot | No retrieval | 47.20% |
| MQL-0007 hybrid RAG | Corpus v1, 3,589 records | 48.99% |
| MQL-0008 zero-shot | No retrieval | Pending |
| MQL-0008 hybrid RAG | Approved corpus v2.1, 3,445 records | Pending |
