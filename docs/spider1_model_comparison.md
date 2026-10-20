# Spider 1.0 NL→MQL: frontier-class model vs small Qwen2.5-Coder-Instruct

Generated reports: `outputs/evaluation_spider1/comparison.md` / `comparison.json` (three-way),
`comparison_qwen3b_vs_qwen7b_full.md` / `.json` (Qwen 3B vs 7B on all 279). Regenerate them
with the commands in §6; no model calls are made.

**Status (2026-10-15): complete on all 279 examples for all three models.**
- The benchmark contains **279 validated examples**.
- **All three models were evaluated on all 279**: Qwen2.5-Coder-3B, Qwen2.5-Coder-7B and
  gpt-oss-120b.
- gpt-oss-120b had been stopped at 211/279 on 2026-10-10. The project owner changed the
  decision, and the run was **resumed on 2026-10-15 for exactly the 68 missing ids**, on the
  Groq free tier. The 211 earlier results were kept unchanged.
- The **primary comparison is now the full paired 279** (§4.1). The earlier 211-example
  partial results are kept for audit in §4.3 and in
  `outputs/evaluation_spider1/gpt-oss-120b/history/`.

## 1. Models actually run

| Role | Model | Access | Settings |
|---|---|---|---|
| Baseline (mentor's "small Qwen") | **Qwen2.5-Coder-3B-Instruct**, official GGUF `q4_k_m` (HF rev `f74adce6`, sha256 verified) | local CPU, llama-cpp-python 0.3.36, 16 threads, no GPU | greedy (T=0, top_p=1, seed 42), max 1,024 new tokens (longest output was 273) |
| Baseline, second size | **Qwen2.5-Coder-7B-Instruct**, official GGUF `q4_k_m` (HF rev `13fb94bf`, sha256 verified) | same | same |
| Frontier-class | **openai/gpt-oss-120b** (120B open-weight MoE reasoning model) | Groq API (free tier), chosen by the project owner via `GROQ_MODEL` | T=0, top_p=1, seed 42, `reasoning_effort` at the provider default, max 8,192 tokens (reasoning + answer) |

### Frontier access history (evidence in `outputs/evaluation_spider1/`)

- **Anthropic (Claude Opus 5.5):** no API key was available, so it was never run. A provider
  is implemented and configured as `frontier_claude`.
- **Gemini 3.8 Flash** (the owner's first choice): the free tier allows 20 requests per day per
  model. 9 items ran before the quota stopped the run (`gemini-3.8-flash/`).
  - **Gemini 3.6 Flash:** same 20/day cap, 16 items (`gemini-3.6-flash/`).
  - **Gemini 3.1 Pro:** no free quota.
  - **Gemini 2.5 models:** unavailable to new users.
  - These partial Gemini runs are kept for transparency and **excluded from the comparison**.
    Mixing models across items would not be fair.
- **Groq gpt-oss-120b:** the free tier allows 200K tokens per day, as a refilling bucket.
  - The run went in stages (`gpt-oss-120b.log`, `.part1`–`.part5.log`), resuming from
    finished ids each time. While quota was exhausted it slept for the server-stated refill
    time. Wrong answers were never retried.
  - It was **stopped deliberately at 211/279** on 2026-10-10.
  - It was **resumed on 2026-10-15** with the same command, prompt, model id, settings and
    scorer (`run_eval.py --provider frontier_groq --label gpt-oss-120b`, log
    `gpt-oss-120b.part6.log`).
    - `run_eval.py` skips ids that already have a non-provider-error record, so only the 68
      missing ids were requested. No code change was needed.
    - All 68 finished in one session (19:44–19:56): no quota waits, 0 provider errors.
      Fifty-nine HTTP 429 rate-limit rejections were retried by the client. A 429 returns no
      answer, so no completed answer was re-requested.
  - **Audit of the final `gpt-oss-120b/executions.jsonl`** (279 lines):
    - 279 unique ids, exactly the dataset ids, in dataset order;
    - all from `openai/gpt-oss-120b` with the same prompt hash;
    - 0 duplicates, 0 unparseable lines, 0 provider errors;
    - the 211 earlier records are content-identical to the archived
      `gpt-oss-120b/history/executions.211_partial.jsonl`.
  - **Time gap.** The 211 and the 68 were generated 5 days apart from the same hosted model
    id. Groq does not expose a model revision, so a silent server-side change cannot be ruled
    out.

**Caveat on "frontier".** `gpt-oss-120b` is a **large open-weight model accessed through Groq,
not a closed frontier model** such as GPT-5, Claude Opus, or Gemini Pro. It was the strongest
model the available key could serve, and it stands in for the "frontier LM" in this study.

## 2. Fairness controls

- **Same examples.** All three models are scored on the 279 validated examples
  (`data/processed/spider1_nl_mql.jsonl`). Paired statistics use the items answered by every
  model.
- **Same prompt.** Every model gets the identical system and user text,
  `nl-mql-spider1-v1` (`scripts/spider1/prompt_spider1.py`); its SHA-256 is recorded in every
  run. The user text contains:
  - the full database schema: every collection, its fields, observed BSON types and declared
    SQL types;
  - foreign keys;
  - 2 frozen sample documents per collection;
  - the question.

  The prompt never includes gold SQL, gold MQL, or results.
- **Same execution and scoring.**
  - Parse with the same parser.
  - Validate with the same read-only check: no `$out`, `$merge`, `$where`, `$function`, or
    `$accumulator`, and only same-DB collections.
  - Execute on the same Atlas database `codegen_spider1` with a 30 s timeout.
  - Score with the same `exec_match` that accepted the gold MQL.
- **No second chances.** A wrong answer is never retried. Only transport and rate-limit errors
  are retried.
- **Decoding differences, documented rather than hidden.** gpt-oss is a reasoning model that
  produces hidden reasoning tokens. Qwen-Coder-Instruct has no reasoning mode. Both are run
  with T=0.

## 3. Metrics

- **Execution accuracy (EX).** The share of examples whose MQL result matches the gold under
  `exec_match`. Reported overall and by official Spider hardness, each with a 95% Wilson CI.
- **Robustness slices.** EX on unflagged items only, and on test-suite-backed items only.
- **Paired test.** McNemar exact test on the discordant pairs.
- **Failure taxonomy.** Run categories plus heuristic root-cause tags
  (`scripts/spider1/error_analysis.py`).
- **Latency and cost.** Latency is per request. For Qwen it is local CPU wall time. For
  gpt-oss it is the API round-trip.
  - The first 143 gpt-oss items include client-side rate-limit waits.
  - The remaining 136 (positions 144–211 and the 68 resumed items) record the request
    latency and the waits separately (`latency_s`, `wall_s`, `quota_wait_s`). Only those are
    used for the gpt-oss latency figure.
  - Cost is the token count × list price. All runs used free tiers, so actual spend was $0.

## 4. Results (final, 2026-10-15; `comparison.md` / `comparison.json`)

### 4.1 Primary: all three models on the same 279 examples

Strict execution accuracy: the generated MQL executes on Atlas and its result matches the
validated gold rows. Malformed, invalid or failing queries count as wrong. 95% Wilson CIs.

| Model | EX (n = 279) | 95% CI | easy (71) | medium (126) | hard (44) | extra (38) | median latency |
|---|---|---|---|---|---|---|---|
| Qwen2.5-Coder-3B | **132/279 = 47.3%** | 41.5–53.2% | 47 | 74 | 7 | 4 | 7.5 s (CPU) |
| Qwen2.5-Coder-7B | **181/279 = 64.9%** | 59.1–70.2% | 58 | 98 | 16 | 9 | 16.4 s (CPU) |
| **gpt-oss-120b** | **232/279 = 83.2%** | 78.3–87.1% | 64 | 107 | 37 | 24 | 1.33 s (Groq API) |

- **McNemar exact tests** (paired, n = 279):

  | Comparison | Only first model correct | Only second model correct | p |
  |---|---:|---:|---|
  | gpt-oss vs 3B | 113 (gpt-oss) | 13 (3B) | 4.5e-21 |
  | gpt-oss vs 7B | 69 (gpt-oss) | 18 (7B) | 3.3e-08 |
  | 7B vs 3B | 58 (7B) | 9 (3B) | 6.8e-10 |

- **Robustness slices** (3B / 7B / gpt-oss):
  - unflagged items (266): 49.2% / 66.5% / 85.7%;
  - test-suite-backed items (267): 47.6% / 66.3% / 82.4%.
- **Determinism.** Rerunning 20 items per Qwen model on an idle CPU gave identical outputs
  (20/20).
- **Latency.**
  - The idle-CPU Qwen rerun medians were 8.1 s and 17.2 s, versus 7.8 s and 16.7 s in the
    full runs.
  - The gpt-oss median is over its 136 cleanly timed requests; Groq server time was 0.86 s.
  - These are different deployments (hosted inference vs a laptop CPU), not just different
    models.
- **Hardness.** The gap grows with difficulty. On hard and extra items (82), gpt-oss scores
  61/82, versus 11/82 for 3B and 25/82 for 7B.
- **Tokens and cost.** All runs used free tiers, so actual spend was **$0**.
  - gpt-oss used 451,843 tokens over 279 items: 335,254 input and 116,589 output (including
    reasoning).
  - Qwen inference is local.
- **Atlas verification** (`docs/week2_report.md`). Every saved prediction of all three
  models was re-executed on Atlas on 2026-10-15. The verdicts agree with the recorded ones
  279/279 for each model, and the gold MQL replays 279/279.

### 4.2 The 68 examples added on 2026-10-15

| Model | EX on the 68 resumed ids |
|---|---|
| Qwen2.5-Coder-3B | 35/68 = 51.5% |
| Qwen2.5-Coder-7B | 48/68 = 70.6% |
| gpt-oss-120b | 56/68 = 82.4% |

- These ids come from dog_kennels (26), network_1 (16), orchestra (11), singer (8),
  real_estate_properties (1) and world_1 (6 of 34).
- By hardness they are 17 easy, 29 medium, 14 hard and 8 extra.
- The Qwen models score higher on these than on the first 211. gpt-oss scores about the same
  (82.4% vs 83.4%), so its full-dataset figure barely moved.

### 4.3 Historical: partial comparison on 211 ids (superseded, kept for audit)

Before 2026-10-15 gpt-oss-120b covered only dataset positions 1–211. Archived reports:
`outputs/evaluation_spider1/gpt-oss-120b/history/comparison.211_partial.{md,json}`.

| Model | EX on the paired 211 | 95% CI |
|---|---|---|
| Qwen2.5-Coder-3B | 97/211 = 46.0% | 39.4–52.7% |
| Qwen2.5-Coder-7B | 133/211 = 63.0% | 56.3–69.3% |
| gpt-oss-120b | 176/211 = 83.4% | 77.8–87.8% |

McNemar p on the 211: gpt-oss vs 3B 2.1e-18, gpt-oss vs 7B 3.4e-08, 7B vs 3B 2.1e-07. These
figures are superseded by §4.1 and should no longer be cited as the main result.

## 5. Error analysis (`error_analysis_<label>.json`, `slice_type_conversion[_paired].json`, `sensitivity_repair.json`)

All three models are analysed on all 279 examples.

| Root-cause tag (incorrect items; one item can have several tags) | Qwen-3B (147 wrong) | Qwen-7B (98 wrong) | gpt-oss (47 wrong) |
|---|---|---|---|
| extra output columns (e.g. keeps join arrays or grouping keys) | 46 | 22 | 4 |
| empty result (over-filtering, often due to type mismatch) | 38 | 26 | 7 |
| too many rows (missing `$group`, `$limit`, or distinct) | 25 | 14 | 10 |
| too few rows | 11 | 14 | 5 |
| wrong values, same shape | 18 | 12 | 7 |
| `$lookup` between differently typed keys (TEXT vs INT) | 11 | 13 | 1 |
| invalid stage or operator name (`{"match": ...}`, `$distinct`) | 10 | 0 | 0 |
| other MongoDB execution errors | 5 | 4 | 5 |
| malformed JSON | 2 | 3 | 8 |

The 5 gpt-oss execution errors all fall in the first 211; the 68 resumed items had none.
They break down as:
- 3 × `$in` used as a top-level query operator, where it needs to go inside `$expr`;
- 2 × `$convert` failing on the literal text `'null'` in car_1 Horsepower. These are items
  0167/0168, already flagged ambiguous for that data quirk.

### Main findings

1. **Type affinity is the MongoDB-specific trap.**
   - SQLite hides the fact that many Spider keys and years are stored as TEXT; MQL is
     type-strict.
   - The gold MQL needs an explicit conversion (`$toString`, `$toInt`, …) on 16 of the 279
     items.
   - On those 16: 3B **0/16**, 7B **1/16**, gpt-oss **13/16**.
   - On the other 263: 132/263, 180/263 and 219/263.
2. **The small models break the output contract.** They return extra fields, `$group` keys,
   or whole joined arrays. 3B's leading error is extra columns (46 items). The prompt asks
   for exactly the requested columns, and the comparator, like Spider's, requires the column
   count to match.
3. **3B makes outright syntax errors.** It writes stage names without `$` and invents
   operators such as `$distinct`. 7B and gpt-oss make none of these.
4. **gpt-oss has a formatting quirk.** All 8 of its format errors (6 in the first 211, 2 in
   the resumed 68: `0800`, `1002`) are the same defect: one redundant closing bracket
   (`...]]}`). The finish reason is always `stop`, so this is not truncation.
   - In a clearly separate sensitivity check (`sensitivity_repair.py`, same rule for every
     model), removing that one character makes 7 of the 8 correct. That would give
     239/279 = 85.7%.
   - The **primary metric stays strict at 232/279**.
   - The Qwen format errors (2 for 3B, 3 for 7B) are not this defect, so nothing in them is
     repairable.
5. **Difficulty scaling.** All models do well on easy items. The gap opens on hard and extra
   items, which need nested logic, set operations or correlated subqueries; there the small
   models mostly fail.

## 6. Reproduce the reports (no model calls)

```powershell
python scripts/spider1/compare.py qwen2.5-coder-3b qwen2.5-coder-7b gpt-oss-120b
python scripts/spider1/error_analysis.py qwen2.5-coder-3b qwen2.5-coder-7b gpt-oss-120b
python scripts/spider1/slice_type_conversion.py qwen2.5-coder-3b qwen2.5-coder-7b gpt-oss-120b
python scripts/spider1/slice_type_conversion.py --paired qwen2.5-coder-3b qwen2.5-coder-7b gpt-oss-120b
python scripts/spider1/sensitivity_repair.py qwen2.5-coder-3b qwen2.5-coder-7b gpt-oss-120b
# full-279 Qwen pair: run compare.py with the two Qwen labels, then copy comparison.* to comparison_qwen3b_vs_qwen7b_full.*
```

Re-running the models themselves (`run_eval.py --provider qwen|qwen7b|frontier_groq`) makes
new model calls and is **not** needed to regenerate these reports.

## 7. Status

- **Complete.**
  - The 279-example dataset is validated on Atlas; 267 items also have test-suite evidence.
  - The reference oracle scores 279/279 through the same model path, which validates the
    harness.
  - **Qwen-3B, Qwen-7B and gpt-oss-120b are all evaluated on all 279.**
  - Every verdict was re-verified by re-executing the saved MQL on Atlas on 2026-10-15
    (`scripts/spider1/rescore.py --reexecute`).
- **Audit trail.**
  - The 211-item partial state is archived in `outputs/evaluation_spider1/gpt-oss-120b/history/`.
    That covers executions, summary, comparison, error analysis, type-conversion slice and
    sensitivity outputs, plus the docs as of the partial stage.
  - The older `executions.pre_stop_backup.jsonl` is kept as well.
- **Not run.**
  - Claude: no API key.
  - Gemini: the free tier allows 20 requests per day per model. The partial Gemini runs are
    kept in `outputs/` and excluded from the comparison.
