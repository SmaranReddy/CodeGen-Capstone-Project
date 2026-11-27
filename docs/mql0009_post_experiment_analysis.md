# MQL-0009: post-experiment analysis

Date: 2026-11-19. Scope: the registered, scored MQL-0009 retrieval-control experiment.

**Evidence labels**

| Label | Meaning |
|---|---|
| **[F] fact** | Read from artifacts or code. |
| **[E] evidence** | Follows from [F] numbers. |
| **[H] hypothesis** | Not established by these artifacts. |
| **[DIAG] diagnostic** | Uses the evaluation items' gold SQL or outcomes to describe what happened. Retrospective only. No retrieval, prompt or threshold of the MQL-0010 design was fitted on it (see §8). |

**Reproduction.** Everything below is recomputed with read-only scripts from the scored artifacts, the frozen package and the approved corpus.

| Script | Output (in `outputs/nl2mql_mql0010/`) |
|---|---|
| `python -B scripts/nl2mql_mql0010/analysis/analyze_mql0009.py` | `mql0009_post_analysis/mql0009_post_analysis.json` |
| `python -B scripts/nl2mql_mql0010/analysis/offline_counterfactuals.py` | `mql0009_post_analysis/offline_counterfactuals.json` |
| `python -B scripts/nl2mql_mql0010/analysis/validator_evidence.py` | `dev_evidence/validator_evidence.json` |

Nothing under `outputs/nl2mql_mql0009_retrieval_control/` or `kaggle/nl2mql_mql0009_retrieval_control/` was written.

---

## 1. Executive summary

1. **[F] The experiment is complete and valid.**
   - 8 runs × 639 = 5,112 predictions.
   - One full-parity, read-only Atlas preflight covered all scoring.
   - Every registered hash matches.
   - The registered primary hypotheses both hold after Holm correction:
     - **H1 hybrid − random-seed mean: +3.79 pp [+1.28, +6.32], Holm p = 0.0079.**
     - **H2 hybrid − fixed: +3.60 pp [+0.63, +6.73], Holm p = 0.030.**
2. **[E] Relevance is real but modest.** The +5.95 pp hybrid gain over zero-shot splits into two parts:
   - about **+2.2 pp from "any three demonstrations"**: random mean − zero-shot, p = 0.108, not significant;
   - about **+3.8 pp from relevance**.
   The hybrid retriever is clearly better than random, but it is not a strong retriever:
   - its three examples carry the gold query's exact structure for 54% of questions [DIAG];
   - its regex structure cues agree exactly with gold SQL structure on 33% of Spider-train questions [F].
3. **[E] The largest error bucket is untouched by any retrieval.**
   - About 100 executable predictions per run return **zero rows**: A 104, random 99–110, C 101, D 100.
   - Together with the remaining wrong-row-count errors, this is 166–178 of each run's errors.
   - A schema-grounded static check flags **67 of D's 100 empty-result failures** and **137 of D's 282 wrong answers**, while flagging **0 of D's 357 correct answers** [DIAG].
   - Typical causes: a field read from a collection that does not have it; a joined field without its `$lookup` alias; an aggregation operator used inside `$match` without `$expr`.
   - The hybrid retriever cannot fix this; a generation-time check can.
4. **[F] Retrieval cannot help a fixed seventh of the task.**
   - The approved corpus has 0 subquery, set-operation or anti-join examples.
   - On the 91 evaluation items needing them, every run sits near 11%: A 12.1, random mean 10.3, C 12.1, D 11.0.
   - On anti-joins, demonstrations hurt: A 33% → D 11% [DIAG, n = 18].
5. **[E] About half of every run's fixes are output-shape priming, and random examples do it too.**
   - Of D's 86 fixes over zero-shot, 51 were zero-shot `wrong_column_count` and 44 changed only the projection.
   - Random seeds get about 30 such projection-only fixes each.
   - A deterministic `_id` post-processor is **not** a substitute: simulated on the stored result rows, it loses 10–17 correct answers per run (§6.4).

---

## 2. Experimental integrity (Phase 1)

| Requirement | Status | Evidence |
|---|---|---|
| 639 predictions per run | ✅ | Every `predictions.jsonl` has 639 unique ids and every record `status: ok` |
| 5,112 in total | ✅ | 8 × 639 |
| Manifests and SHA-256 | ✅ | `input_validation_report.json` `ok: true`; per-run hashes listed in the ledger (MQL-0009 status update) |
| Registered questions and runs | ✅ | Package manifest `2a4f22e0…`, evaluation ids `6f1425f7…`, config `3f052e98…`, analysis plan `ecb7479f…` |
| All 8 runs scored | ✅ | `scoring/arms/<run>/summary.json`: all `complete`, 0 infrastructure or generation failures |
| Same Atlas snapshot | ✅ | One preflight for all runs, `scoring/preflight/atlas_check.json` sha256 `af4ae6f6…` |
| Model revision and settings | ✅ | Every run config: revision `488639f1…`, T4, bf16 emulated, torch 2.11.0+cu130, transformers 5.20.2, greedy, batch 8 |
| No overwrite or regeneration | ✅ | See below |

**Atlas snapshot details.** The single preflight covered:
- Atlas 8.0.37, `codegen_spider1_heldout`;
- 69 collections and 16,885 documents, all digests matching;
- the read-only user `codegen_eval_readonly` with no write or admin action.

**Why nothing was regenerated.** The session exports are cumulative:
- session 2 re-exports A, B_s1 and D;
- session 3 re-exports A, B_s1, B_s2, B_s3, C and D.

Every re-exported copy is byte-identical to the first export. Offline re-derivation of every executed outcome from the stored rows with the registered evaluator gives **0 mismatches** in every run (`offline_counterfactuals.json`, `sanity_mismatches`).

**How scoring was run.**
- The first scoring attempt stopped at the Atlas preflight with `PreflightError: read-only user has no 'find' on database codegen_spider1_heldout`. No run was scored.
- The user then added the grant.
- The user then twice sent the registered command as a chat message, and the shell prefix did not take effect either time. **Claude took this as authorization and ran the unchanged registered command in-session.** That departed from the task brief's no-Atlas-scoring restriction.
- Scoring finished on 2026-11-19 at about 06:10–06:40 UTC.

**Known noise sources:**
- **Batch composition.** The OOM batch-1 fallback hit 72 or 80 items depending on the run. Greedy bf16-emulated outputs can depend on batch composition, so a few items per run may differ for that reason alone.
- **Tie order.** Atlas tie-order nondeterminism (MQL-0008) can move single items. All 8 runs were scored in one session against one snapshot, so the paired comparisons share the same evaluator state.

---

## 3. Accuracy and statistics

### 3.1 Per run [F]

| Run | correct / 639 | % | Wilson 95% | incorrect | invalid | execution_error | executable but incorrect |
|---|---:|---:|---|---:|---:|---:|---:|
| A zero-shot | 319 | 49.92 | 46.1–53.8 | 303 | 4 | 13 | 303 (47.4%) |
| B random s1 | 334 | 52.27 | 48.4–56.1 | 285 | 3 | 17 | 285 |
| B random s2 | 335 | 52.43 | 48.6–56.3 | 280 | 5 | 19 | 280 |
| B random s3 | 334 | 52.27 | 48.4–56.1 | 283 | 4 | 18 | 283 |
| B random s4 | 333 | 52.11 | 48.2–56.0 | 282 | 3 | 21 | 282 |
| B random s5 | 328 | 51.33 | 47.5–55.2 | 281 | 5 | 25 | 281 |
| B random, mean of 5 | 332.8 | 52.08 | — | | | | |
| C fixed | 334 | 52.27 | 48.4–56.1 | 289 | 2 | 14 | 289 |
| **D hybrid** | **357** | **55.87** | 52.0–59.7 | 263 | 2 | 17 | 263 (41.2%) |

There were no timeouts, generation failures, infrastructure failures or `not_comparable` outcomes. All 639 gold references have status `ok`.

**Invalid rates** (unparseable or unsafe):
- A 0.63%;
- random 0.47–0.78%;
- C 0.31%;
- D 0.31%.

**Execution-error rates:**
- A 2.03%;
- random 2.66–3.91%;
- C 2.19%;
- D 2.66%.

Demonstrations slightly *raise* execution errors. Most of the added ones come from copied corpus idioms (§5.4).

### 3.2 Registered tests [F] (recomputed independently; identical)

| Test | Difference | 95% CI | Test statistic | Holm p |
|---|---:|---|---|---:|
| **H1** D − per-question mean of the 5 seeds | **+3.79 pp** | [+1.28, +6.32] (bootstrap over questions; the 5 seed outcomes stay together) | sign-flip p = 0.00395; D better on 128 questions, random mean better on 80 | **0.0079** |
| **H2** D − C | **+3.60 pp** | [+0.63, +6.73] | exact McNemar p = 0.0297 (D only 63, C only 40) | **0.030** |

McNemar is not applied to the fractional seed mean.

### 3.3 Secondary comparisons [F] (unadjusted unless stated)

| Comparison | Difference | Detail |
|---|---:|---|
| D − A | +5.95 pp | 86 fixes / 48 regressions; McNemar p = 0.0013; CI [+2.5, +9.4] |
| Random mean − A | +2.16 pp | CI [−0.41, +4.79]; sign-flip p = 0.108; **not significant** |
| C − A | +2.35 pp | p = 0.159; **not significant** |

**D against each random seed:**

| Seed | D only / seed only | McNemar p | Holm-adjusted over the 5 seeds |
|---|---|---:|---:|
| s1 | 65 / 42 | 0.033 | 0.106 |
| s2 | 60 / 38 | 0.033 | 0.106 |
| s3 | 64 / 41 | 0.031 | 0.106 |
| s4 | 66 / 42 | 0.026 | 0.106 |
| s5 | 64 / 35 | 0.0046 | **0.023** |

Each seed alone is a weak comparator, which is why the registered primary test uses the 5-seed mean.

### 3.4 Decomposition [E]

D − A = **+5.95 pp**:
- **+2.16 pp** from demonstrations as such (random mean − A, not significant);
- **+3.79 pp** from relevance (D − random mean, significant).

Fixed examples (C, 52.27%) equal the random mean (52.08%): three arbitrary demonstrations are worth about 2 pp, whichever three they are.

### 3.5 Random-seed variability [F]

- Accuracy SD across seeds is **0.43 pp** (51.33–52.43).
- Item-level instability is large. Number of seeds correct per question:

| Seeds correct | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---:|---:|---:|---:|---:|---:|
| Questions | 220 | 45 | 33 | 48 | 56 | 237 |

- **182 of 639 questions (28%) change outcome depending only on which random examples were drawn.**
- Pairwise seed agreement is 0.84–0.88.
- Consequence: single-run comparisons of RAG configurations carry several points of item-level noise. Paired designs with fixed first-pass outputs are needed to attribute effects (adopted in MQL-0010).

---

## 4. Error transitions (zero-shot → hybrid)

### 4.1 By gold query family [DIAG]

| Gold family | n | Both correct | Fixed by D | Broken by D | Both wrong | Net | A % | Random mean % | C % | D % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Single table, no subquery or set operation | 353 | 228 | 43 | 25 | 57 | +18 | 71.7 | 73.1 | 74.5 | 76.8 |
| Join, no subquery or set operation | 195 | 38 | 38 | 17 | 102 | +21 | 28.2 | 33.4 | 30.8 | 39.0 |
| Subquery / set operation / anti-join | 91 | 5 | 5 | 6 | 75 | −1 | 12.1 | 10.3 | 12.1 | 11.0 |
| **All** | 639 | 271 | 86 | 48 | 234 | **+38** | 49.9 | 52.1 | 52.3 | 55.9 |

### 4.2 By gold SQL pattern [DIAG] (an item can carry several)

| Pattern | n | Fixed | Broken | Net | | Pattern | n | Fixed | Broken | Net |
|---|---:|---:|---:|---:|---|---|---:|---:|---:|---:|
| group_by | 134 | 41 | 11 | **+30** | | like | 30 | 5 | 0 | +5 |
| count | 166 | 39 | 12 | **+27** | | intersect | 24 | 4 | 2 | +2 |
| limit | 73 | 28 | 6 | **+22** | | nested | 48 | 1 | 0 | +1 |
| order_by | 120 | 30 | 11 | +19 | | aggregate | 115 | 3 | 6 | **−3** |
| join | 231 | 39 | 20 | +19 | | distinct | 42 | 2 | 5 | **−3** |
| where | 308 | 32 | 20 | +12 | | count_distinct | 15 | 0 | 4 | **−4** |
| having | 41 | 14 | 3 | +11 | | anti_join | 18 | 0 | 4 | **−4** |

**By hardness** (A / random mean / C / D):

| Hardness | n | A | Random mean | C | D |
|---|---:|---:|---:|---:|---:|
| easy | 181 | 75.7 | 76.0 | 77.9 | 77.9 |
| medium | 260 | 60.0 | 59.5 | 61.2 | 61.5 |
| **hard** | 111 | 18.0 | 27.4 | 23.4 | **42.3** (+15 pp over random) |
| extra | 87 | 6.9 | 11.5 | 9.2 | 10.3 |

**[E] Where retrieval works and where it doesn't.**
- Relevance pays off where a multi-stage *pattern* must be copied: group + sort + limit, having, count per group. That explains "hard".
- It does nothing on "extra", which is dominated by nested and set-operation queries the corpus cannot show.
- It mildly harms distinct, count-distinct, plain aggregates and anti-joins.

### 4.3 Fix mechanisms [DIAG]

**Fixes per run, and how many are projection-only:**

| Run | Fixes over A | of which A was `wrong_column_count` | only `$project` differs (same other stages) | A's result had an `_id` column |
|---|---:|---:|---:|---:|
| B s1 / s2 / s3 / s4 / s5 | 60 / 58 / 67 / 61 / 58 | 37 / 34 / 34 / 32 / 34 | 30 / 30 / 27 / 30 / 30 | 21 / 18 / 20 / 14 / 17 |
| C | 57 | 35 | 32 | 16 |
| **D** | **86** | 51 | 44 | 29 |

**`_id` and column-count errors per run:**

| | A | Random | C | D |
|---|---:|---:|---:|---:|
| Results carrying an `_id` column | 120 | 58–64 | 71 | 28 |
| `wrong_column_count` | 103 | 68–78 | 81 | 54 |

**Overlap between D's fixes and the other runs:**
- **72** of D's 86 fixes are also fixed by at least one random seed.
- Per seed the overlap is 43–46 (seed fixes: 58–67).
- 42 are also fixed by C.
- **13** are fixed only by D.

**[E] What the mechanisms are.**
- About half of every run's fixes are *result-contract priming*: dropping `_id` and returning exactly the asked columns. Any demonstrations produce this.
- D's extra fixes over random concentrate in multi-stage aggregation patterns (§4.2).

**Regressions [DIAG].** D's 48 regressions:
- 41 become `incorrect`, 7 become execution errors;
- concentrated in single-table (25) and join (17) items;
- they include the anti-join and count-distinct losses above.

---

## 5. Retrieval effectiveness

### 5.1 The retriever that ran [F]

The hybrid ranking was rebuilt offline from the package sources. It reproduces the registered D selections for **639/639** items (`reproduce_hybrid`), with 0 budget drops.

### 5.2 Score resolution and ties [F]

- **Score saturation.** The top score equals 2.0 on 116 items (BM25 maximum and a full Jaccard match) and 1.0 on 127.
- **Tie at the selection boundary.** 77 items tie at ranks 3/4; 20 have all top-10 scores tied.
- **The tie-break biases toward the start of the corpus.** Ties are broken by `doc_id`. On tied items, the selected documents sit at mean relative corpus position 0.331, against 0.482 overall.
- **Reuse.** 1,102 distinct documents are ever selected; the most reused (`spider_train_00016`) appears 12 times.
- **Redundancy.** In D, **330 of 639 triples (52%) show three examples with the same structure signature**, and 329 the same stage skeleton. With random examples, 3 of 639 do.

### 5.3 Does structural compatibility matter?

**Observational, D [DIAG].** Split by whether any top-3 example has exactly the gold query's structure signature:

| | n | A % | Random mean % | D % | D fixes / breaks |
|---|---:|---:|---:|---:|---|
| Structure matched in top 3 | 345 | 62.3 | 64.9 | 70.7 | 53 / 24 |
| Not matched | 294 | 35.4 | 37.0 | 38.4 | 33 / 24 |

- D beats the random mean by **+5.8 pp** where it matched the structure and by **+1.4 pp** where it did not.
- **How often each run shows the exact structure:**
  - D 54.0% of questions;
  - random seeds 10.5–15.2%;
  - C 11.6%.
- **Coverage gap.** The gold structure exists somewhere in the corpus for 538 questions; D found it for 345 (64%).
- Matched items are easier (A 62% vs 35%), so this table alone is not causal.

**Randomised, within question [DIAG].** In the random runs, which examples are drawn is random. For questions where some seeds showed a property and others did not, the within-question accuracy contrast estimates the property's causal effect, free of question difficulty (permutation test within question):

| Property of the drawn examples | Questions | Within-question effect | p (two-sided, unadjusted, 1 of 4) |
|---|---:|---:|---:|
| exact gold structure signature present | 258 | +2.36 pp | 0.19 |
| structure-label Jaccard ≥ 0.5 present | 403 | +2.09 pp | 0.13 |
| a join example shown for a single-table question | 335 | +0.55 pp | 0.71 |
| lexical (masked BM25) similarity above the question's median | 639 | +2.01 pp | 0.041 |

**[E] Interpretation:**
- Both structural and lexical similarity point in the expected direction with effects of about 2 pp per occurrence.
- Random draws rarely achieve strong matches, so this design has low power. No single property is established as *the* driver.
- The lexical p = 0.041 does not survive any multiplicity adjustment.
- Showing an irrelevant join is not measurably harmful on average.
- **Correlation vs causation.** The +5.8 vs +1.4 split is correlational. The randomised estimates are causal but imprecise.

### 5.4 Imitation of corpus idioms [F]

**Copy rates in D: how often the output uses an idiom when an example shows it, vs when none does:**

| Idiom | Use rate when shown | Use rate when not shown |
|---|---:|---:|
| `$lookup` | 51.3% | 10.6% |
| `$unwind` | 47.0% | 9.7% |
| machine join alias `__jN` | 46.0% | 0% |
| `$expr` | 35.9% | 5.5% |
| `$facet` count wrapper (shown on 209 items) | 0.5% | — |

- **Accuracy of outputs with these idioms** is close to the run average: `__jN` 36%, `$lookup` 37% in D vs 28% in A.
- **No run ever emits a type conversion** (`$toString`, `$toInt`, `$convert`). There is one in each of A, B_s1 and C, all wrong.
- **[E] Two consequences:**
  - The model copies surface form, including machine aliases, that it only half understands.
  - The verbose count wrapper, 798 records or 23% of the corpus, is almost never imitated. It costs context without transferring structure.

### 5.5 Regex structure cues (inference-time signal) [F]

Measured on all 7,773 Spider-train questions; no evaluation item is used.

- **Overall:** the cue set agrees exactly with gold SQL structure on **33.3%** (excluding `where`).
- **Per pattern:**

| Cue | Precision | Recall | Note |
|---|---:|---:|---|
| join | 0.70 | 0.45 | |
| group_by | 0.62 | 0.46 | |
| count_distinct | — | 0 | never predicted |
| union | 0.15 | — | fires on any "or" |
| anti_join | 0.78 | 0.98 | |

- The "needs an unsupported construct" cue has precision 0.63 and recall 0.51.
- **[E]** The structural half of the hybrid score is a weak signal. With weight 1.0 it is as heavy as the whole normalised BM25 score, which produces the plateaus in §5.2.

---

## 6. Root causes, ranked by consequence

### 6.1 [E] Schema-grounding errors dominate, and retrieval does not touch them

| Run | `wrong_row_count`, empty result | `wrong_row_count`, non-empty | `wrong_column_count` | `wrong_values` | execution_error | invalid / format |
|---|---:|---:|---:|---:|---:|---:|
| A | 104 | 68 | 103 | 28 | 13 | 4 |
| Random (range) | 99–110 | 64–71 | 68–78 | 34–42 | 17–25 | 3–5 |
| C | 101 | 70 | 81 | 34 | 14 | 2 |
| D | 100 | 66 | 54 | 41 | 17 | 2 |

**What demonstrations change and what they don't:**
- They cut `wrong_column_count` from 103 to 54.
- The empty-result bucket stays at about 100 whatever the demonstrations.

**Hand inspection** of a seeded sample of 14 D empty-result failures [DIAG] found:
- a field read from the start collection that lives in another collection (`city` on `hr_1__employees`, `Party` on `election`);
- a joined field without its alias (`commission_pct` after `$lookup … as "__j1"`);
- a nonexistent embedded path (`Customer.FirstName`);
- an aggregation expression inside a query operator (`{"Revenue": {"$gt": {"$avg": "$Revenue"}}}`), which MongoDB reads as a literal sub-document;
- `$unwind` destroying the unmatched rows an anti-join needs;
- swapped first and last name literals.

**The schema validator**, written from MongoDB semantics, checks field paths, collections and `$match` operands; see `docs/mql0010_rag_architecture_design.md` §3.4. It flags:

| | Count |
|---|---:|
| Verified corpus pipelines flagged (false alarms) | **0 of 3,445** |
| Wrong answers flagged, per run | 130–148 |
| Correct answers flagged, per run | 0–1 |
| D wrong answers flagged | 137 of 282 |
| D empty-result failures flagged | 67 of 100 |
| D correct answers flagged | 0 |

**Caveats:**
- These sizes are in-sample: the V3/V4 rules were written after reading sampled failures.
- The corpus check is weaker than it looks. Converter MQL never misnames fields, and its `$facet` wrapper makes the validator stop checking (opaque) after that stage.
- The best false-alarm evidence is the 0–1 flagged correct answers per run.

### 6.2 [F] Coverage ceiling

The corpus has **0** examples of nested, anti-join, intersect and union.

| | nested | anti-join | intersect | union |
|---|---:|---:|---:|---:|
| Spider train | 571 | 388 | 217 | 62 |
| 639 evaluation questions | 48 | 18 | 24 | 3 |

- The corpus has no `$lookup` with `let`/`pipeline` and no type conversions.
- It has 44 distinct stage skeletons and 65 distinct structure signatures.
- 217 corpus questions are duplicates after masking.
- The gold structure of **101 of 639** evaluation questions (16%) does not occur in the corpus at all.

### 6.3 [E] Low-resolution, redundant selection

- The structural signal is weak (§5.5), and scores are quantised (§5.2).
- Half of the triples repeat one structure three times (§5.2). When that structure is wrong, all three examples mislead together.
- On the MQL-0010 development split (§7), signature-diversified selection raises the share of questions whose correct structure is shown from 57.1% to 65.9% without changing the ranking.

### 6.4 [E] Output-shape priming inflates the "RAG gain", and cannot be replaced by a rule

- About half of every run's fixes are projection-only (§4.3).
- **Offline counterfactual.** Appending `{"$project": {"_id": 0}}` was simulated on the stored result rows.
  - Rule R1 (final inclusion projection that never mentions `_id`) never removes an `_id` column.
  - Rule R2 (drop `_id` unless the last stage is a group): net **−15 (A), −10 to −17 (random), −13 (C), −11 (D)**. `_id` is often the answer column.
- **Rejected.**

### 6.5 [F → E] Join-key type mismatches: smaller here than in MQL-0008

- The dormant MQL-0008 helper compares raw type sets. It would flag four evaluation DBs here, but three of the four are int vs double or a nullable key. MongoDB equality matches those numerically, so the notes would be false.
- Only `phone_market` (string ↔ int `Phone_ID`) is genuinely incompatible.
- 16 items join on any of the four pairs [DIAG]. Correct on them: A 4 to D 8; random 4–8.
- Contrast: MQL-0008 had 42 items on genuinely incompatible keys, and all 42 were wrong in both runs.

---

## 7. What this means for the architecture (summary; details in the design doc)

| Direction | Evidence | Verdict |
|---|---|---|
| **H. Generation-time schema validation and one feedback retry** | Flags about half of all wrong answers with 0–1 false alarms per run; targets the largest bucket, which retrieval cannot reach | **Implement, primary** |
| **E. Diversified selection** | 52% redundant triples; dev proxy hit@3 57.1 → 65.9; randomised exposure effect about +2 pp (not significant) | **Implement, test as hypothesis** |
| **F. Corpus coverage for set operations and subqueries** | About 14% of the task is unreachable by retrieval | **Proposal** (separate converter experiment) |
| D. Confidence gating for unsupported families | Ceiling about 1 item (11.0 vs 12.1%) | Not implemented |
| A/G. Schema-aware notes (type-incompatible join keys) | 4 items here, 42 in MQL-0008 | Implement behind a flag (numeric types unified), low priority |
| B/C. Learned structure / operator-aware scoring | Learned predictor beats the regex cues (exact 33.0 → 36.8%, macro-F1 0.52 → 0.59, train cross-validation) but *lowers* dev hit@3 alone (50.2 vs 57.1); adds at most 1.8 pt on top of diversity | **Evaluated, not adopted** |
| `_id` post-processor | −10 to −17 correct per run | **Rejected** |
| Tie-break changes | No dev-proxy effect | Not adopted |

---

## 8. Use of evaluation labels, and limitations

**What the 639 gold labels were used for.** They were used only for the [DIAG] tables. No MQL-0010 hyperparameter was fitted on them, but they did shape which components were *built*. The sampled empty-result failures motivated validator checks V3/V4. The 639 are therefore **design-informed** and must not serve as MQL-0010's confirmatory set. MQL-0010 uses a fresh, seeded holdout of 18 other Spider-train databases.

**Development evidence that touches no protected item:**
- regex-cue and structure-predictor accuracy, measured on Spider-train questions;
- the retrieval proxy, measured on the MQL-0010 dev split;
- validator false alarms, measured on corpus gold MQL.

**Remaining limitations:**
- No repair rate for the retry exists yet; it needs inference.
- The within-question tests are low-powered.
- The family labels come from SQL regexes, so they are approximate.
