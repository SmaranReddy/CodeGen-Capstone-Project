# Spider 1.0 NL→MQL: frontier-class model vs small Qwen2.5-Coder-Instruct

Generated reports: `outputs/evaluation_spider1/comparison.md` / `comparison.json` (three-way),
`comparison_qwen3b_vs_qwen7b_full.md` / `.json` (Qwen 3B vs 7B on all 279). Regenerate them
with the commands in §6; no model calls are made.

**Status (final for this week, 2026-10-10):**
- The benchmark contains **279 validated examples**.
- **Both Qwen models were evaluated on all 279.**
- The **gpt-oss-120b evaluation was intentionally stopped** after **211 verified** examples
  (dataset positions 1–211). The other **68 examples were not evaluated**. They are not
  counted as failures or as successes.
- The frontier comparison is therefore **partial**. It may be affected by processing order,
  because examples were run in Spider dev order, which is grouped by database.

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
  - It was **stopped deliberately at 211/279** (process stopped on 2026-10-10; no further Groq
    calls).
  - **Audit of `gpt-oss-120b/executions.jsonl`** (211 lines):
    - 211 unique ids, all in the dataset, all from `openai/gpt-oss-120b` with the same
      prompt hash;
    - 0 duplicates, 0 unparseable lines, 0 remaining provider errors;
    - the ids form a contiguous prefix (positions 1–211).
  - The earlier "168 vs 167" discrepancy was timing, not data. `compare.py` ran when 167 were
    done, and item 168 finished seconds later.

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
  - The last 68 record the request latency and the waits separately (`latency_s`, `wall_s`,
    `quota_wait_s`). Only those are used for the gpt-oss latency figure.
  - Cost is the token count × list price. All runs used free tiers, so actual spend was $0.

## 4. Results snapshot (2026-10-10, 05:00)

### 4.1 Complete comparison on all 279 examples: Qwen 3B vs 7B (`comparison_qwen3b_vs_qwen7b_full.md`)

| Model | EX | 95% CI | easy (71) | medium (126) | hard (44) | extra (38) | median latency |
|---|---|---|---|---|---|---|---|
| Qwen2.5-Coder-3B | **132/279 = 47.3%** | 41.5–53.2% | 47 | 74 | 7 | 4 | 7.5 s (CPU) |
| Qwen2.5-Coder-7B | **181/279 = 64.9%** | 59.1–70.2% | 58 | 98 | 16 | 9 | 16.4 s (CPU) |

- **Paired comparison.** 7B solves 58 items that 3B misses; 3B solves 9 that 7B misses.
  McNemar exact p = 6.8e-10.
- **Robustness slices.** The gap holds on the unflagged slice (49.2% vs 66.5%) and on the
  test-suite-backed slice (47.6% vs 66.3%).
- **Determinism.** Rerunning 20 items per model on an idle CPU gave identical outputs (20/20).
- **Latency.** The idle-CPU rerun medians were 8.1 s and 17.2 s, versus 7.8 s and 16.7 s in
  the full runs. So the agents sharing the CPU during the full runs did not materially
  distort latency.

### 4.2 gpt-oss-120b on its 211 evaluated examples (`comparison.md`, "own coverage")

| Model | Answered | Not evaluated | EX on answered | 95% CI |
|---|---|---|---|---|
| Qwen2.5-Coder-3B | 279 | 0 | 132/279 = 47.3% | 41.5–53.2% |
| Qwen2.5-Coder-7B | 279 | 0 | 181/279 = 64.9% | 59.1–70.2% |
| gpt-oss-120b | 211 | 68 | **176/211 = 83.4%** | 77.8–87.8% |

The rows have different denominators, so they are **not directly comparable**. Use §4.3.

### 4.3 Three-way paired comparison on the same 211 example ids (PARTIAL)

| Model | EX on the paired 211 | 95% CI | easy (54) | medium (97) | hard (30) | extra (30) |
|---|---|---|---|---|---|---|
| Qwen2.5-Coder-3B | 97/211 = 46.0% | 39.4–52.7% | 36 | 55 | 4 | 2 |
| Qwen2.5-Coder-7B | 133/211 = 63.0% | 56.3–69.3% | 44 | 74 | 11 | 4 |
| **gpt-oss-120b** | **176/211 = 83.4%** | 77.8–87.8% | 48 | 85 | 26 | 17 |

- **McNemar exact tests on the 211 paired ids**, recomputed and cross-checked against
  `scipy.stats.binomtest`:

  | Comparison | Only first model correct | Only second model correct | p |
  |---|---:|---:|---|
  | gpt-oss vs 3B | 86 (gpt-oss) | 7 (3B) | 2.1e-18 |
  | gpt-oss vs 7B | 53 (gpt-oss) | 10 (7B) | 3.4e-08 |
  | 7B vs 3B | 43 (7B) | 7 (3B) | 2.1e-07 |

- **Robustness slices** (paired 211):
  - unflagged items: 47.5% / 65.2% / 85.3%;
  - test-suite-backed items: 46.2% / 64.8% / 82.4% (3B / 7B / gpt-oss).
- **Processing-order caveat.**
  - The 211 are the first 211 dataset positions, which follow Spider dev order and so are
    grouped by database.
  - Not evaluated by gpt-oss: dog_kennels (26), network_1 (16), orchestra (11), singer (8),
    real_estate_properties (1), and 6 of world_1's 34 items.
  - By hardness, the 68 unevaluated items are 17 easy, 29 medium, 14 hard and 8 extra.
  - The unevaluated items look **easier** for the Qwen models. On them, 3B scores
    35/68 = 51.5% (vs 46.0% on the paired 211) and 7B 48/68 = 70.6% (vs 63.0%).
  - So the paired subset is not random. It may slightly understate everyone's
    full-benchmark accuracy, and gpt-oss's 83.4% must not be read as its 279-example
    accuracy.
- **Hardness.** The gap grows with difficulty. On hard and extra items (60 paired), gpt-oss
  scores 43/60, versus 6/60 for 3B and 15/60 for 7B.
- **Latency.**
  - gpt-oss-120b on Groq: median **1.19 s** per request, measured on the 68 cleanly timed
    requests; Groq server time 0.77 s.
  - Qwen runs locally on CPU: 7.5 s (3B) and 16.4 s (7B) median.
  - These are different deployments (hosted inference vs a laptop CPU), not just different
    models.
- **Tokens and cost.** All runs used free tiers, so actual spend was **$0**.
  - gpt-oss used 331,476 tokens over 211 items: 241,143 input and 90,333 output (including
    reasoning), about 1.57K per item.
  - Qwen inference is local.

## 5. Error analysis (`error_analysis_<label>.json`, `slice_type_conversion[_paired].json`, `sensitivity_repair.json`)

Each model is analysed on the examples it was evaluated on (denominators shown). Unevaluated
examples are excluded.

| Root-cause tag (incorrect items; one item can have several tags) | Qwen-3B (147 wrong / 279) | Qwen-7B (98 / 279) | gpt-oss (35 / 211) |
|---|---|---|---|
| extra output columns (e.g. keeps join arrays or grouping keys) | 46 | 22 | 4 |
| empty result (over-filtering, often due to type mismatch) | 38 | 26 | 5 |
| too many rows (missing `$group`, `$limit`, or distinct) | 25 | 14 | 9 |
| too few rows | 11 | 14 | 3 |
| `$lookup` between differently typed keys (TEXT vs INT) | 11 | 13 | 1 |
| invalid stage or operator name (`{"match": ...}`, `$distinct`) | 10 | 0 | 0 |
| other MongoDB execution errors | 5 | 4 | 5 |
| malformed JSON | 2 | 3 | 6 |

The 5 gpt-oss execution errors break down as:
- 3 × `$in` used as a top-level query operator, where it needs to go inside `$expr`;
- 2 × `$convert` failing on the literal text `'null'` in car_1 Horsepower. These are items
  0167/0168, already flagged ambiguous for that data quirk.

### Main findings

1. **Type affinity is the MongoDB-specific trap.**
   - SQLite hides the fact that many Spider keys and years are stored as TEXT; MQL is
     type-strict.
   - The gold MQL needs an explicit conversion (`$toString`, `$toInt`, …) on 16 of the 279
     items. Qwen-3B scores **0/16** on them and Qwen-7B **1/16**.
   - On the **paired** ids, 14 such items, the scores are 3B **0/14**, 7B **0/14** and
     gpt-oss **11/14**.
   - On the paired items that need no conversion: 97/197, 133/197 and 165/197.
2. **The small models break the output contract.** They return extra fields, `$group` keys,
   or whole joined arrays. 3B's leading error is extra columns (46 items). The prompt asks
   for exactly the requested columns, and the comparator, like Spider's, requires the column
   count to match.
3. **3B makes outright syntax errors.** It writes stage names without `$` and invents
   operators such as `$distinct`. 7B and gpt-oss make none of these.
4. **gpt-oss has a formatting quirk.** All 6 of its format errors are the same defect: one
   redundant closing bracket (`...]]}`). The finish reason is always `stop`, so this is not
   truncation. In a clearly separate sensitivity check (`sensitivity_repair.py`, same rule
   for every model), removing that one character makes 5 of the 6 correct. That would give
   181/211 = 85.8%. The **primary metric stays strict at 176/211**. The Qwen format errors
   (2 for 3B, 3 for 7B) are not this defect, so nothing in them is repairable.
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
  - Qwen-3B and Qwen-7B are evaluated on all 279.
- **Partial, stopped intentionally.**
  - gpt-oss-120b: 211/279 evaluated. The run was stopped on purpose; there are no more
    Groq calls and no paid tier.
  - The 68 unevaluated examples are neither failures nor successes.
  - The pre-stop file is preserved as `gpt-oss-120b/executions.pre_stop_backup.jsonl`; its
    SHA-256 is identical to `executions.jsonl`.
- **Not run.**
  - Claude: no API key.
  - Gemini: the free tier allows 20 requests per day per model. The partial Gemini runs are
    kept in `outputs/` and excluded from the comparison.
