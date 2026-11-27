# MQL-0010: pre-execution design audit

Date: 2026-11-19.

**Scope.** Read-only review of the registered MQL-0010 design:
- `scripts/nl2mql_mql0010/src/*`;
- `preflight/register.py`, `preflight/make_split.py`;
- `configs/mql0010_experiment.json` (sha256 `1c9e502e…`);
- `outputs/nl2mql_mql0010/preflight/*`, including amendments A1–A3;
- the three design documents;
- `CLAUDE.md`.

**What was not done.** No code, configuration, hypothesis or analysis plan was changed. No prediction, Kaggle run or Atlas connection was made.

**Checks that were run:**
- **Holdout integrity:** ids, databases and question text only. No holdout gold SQL or result was opened.
- **Power simulations,** with parameters taken from MQL-0009 development evidence only.

Both scripts are in the session scratchpad (`audit_checks.py`).

**Conflict of interest.** The same assistant session designed MQL-0010. The findings below are deliberately adversarial. Where the design documents overstate something, this audit says so.

---

## 1. Verdict

**Overall: CONDITIONAL GO.**

The core comparison (D0V vs D0, validation plus retry) is well isolated and will produce interpretable evidence. However, four issues must be fixed, or documented and pinned, before GPU time is spent:
- **R1:** amendment A3 is not pinned.
- **R2:** the type-note comparison is confounded by batch composition.
- **R3:** the holdout retrieval view is not enforced in code.
- **R4:** the primary test cannot tell a trivial effect from a meaningful one.

Each has a small fix (§7). None needs a redesign.

| Component | Arm comparison | Verdict | One-line reason |
|---|---|---|---|
| V: validation + one retry | D0V vs D0 | **VALID WITH CAVEATS** | Cleanly isolated and well powered. But it is nearly one-sided by construction, so p < 0.05 is a low bar (R4). It cannot separate *schema feedback* from *any second attempt* (no generic-retry control). |
| F: join-key type notes | EF vs E | **VALID WITH CAVEATS (descriptive only)** | Fresh inference with the note on changed prompts only, so the mechanism is genuinely tested. Confounded by batch composition (R2) and badly underpowered (52 changed items; power ≤ 0.71 even at +8 net). Cannot support an adoption claim. |
| E: diversified selection | E vs D0 | **VALID WITH CAVEATS** | Isolates selection only (same ranking, k and order). Power about 0.24–0.43 for a true +2 pp, and gated behind two earlier steps. A null result is inconclusive, not evidence of no effect. |
| Full architecture | FULL vs D0 | **VALID WITH CAVEATS** | A legitimate package effect. It cannot attribute the gain to components and includes E × V interaction. |
| C: subquery / set-op corpus coverage | none | **INSUFFICIENT EVIDENCE (not tested)** | No arm; MQL-0010 can say nothing about it. Correctly scoped as a proposal. |
| Holdout | — | **VALID WITH CAVEATS** | Database- and id-disjoint, drawn by seed, no model outcome seen. But it is not pristine (§5.3), and its retrieval view is enforced only by convention (R3). |

---

## 2. Arm-by-arm reconstruction from code

**Where the arms are defined.**
- Arms: `preflight/register.py:57–66`, copied into `configs/mql0010_experiment.json` `arms`.
- Flags: `planner.arm_settings` (`src/planner.py:43`), defaults at `planner.py:27–28`.

**How each kind of arm executes** (`src/driver.py`):

| Kind | Function | What it does |
|---|---|---|
| `generate` | `driver.first_pass`, lines 23–32 | one `generate(system, [user])` call per item |
| `prompt_delta` | `driver.prompt_delta`, lines 35–45 | generates only items whose plan `prompt_sha256` differs from the base arm (line 40); otherwise copies the base record (`source: reused:<arm>`) |
| `retry` | `driver.retry_arm`, lines 48–57 | calls `recovery.recover` on the source arm's `first_raw_text` |

**How prompts are built** (`planner.build_plan`, `planner.py:58–108`):
- ranking with `retriever_v2.StructureAwareRetriever`, baseline mode (`retriever_v2.py:46–49`, the unchanged MQL-0008 retriever);
- optional `select_diverse` over the top-50 candidates (`planner.py:79–83`, `retriever_v2.py:84–`);
- optional type notes (`planner.py:68–69` → `prompt_v2.fk_type_notes`, lines 23–37);
- the budget loop (`prompt_v2.budgeted_block`, 44–69);
- the unchanged `rag_block.insert_block`.

| Arm | Prompt | New inference? | Reuses | Hypothesis / estimand | What a result can legitimately support |
|---|---|---|---|---|---|
| **A0** | Frozen zero-shot `nl-mql-spider1-v1` (empty block ⇒ byte-identical; tested `test_zero_shot_arm_reproduces_frozen_prompts`) | Full pass (716) | — | Baseline level | Descriptive |
| **A0V** | First turn = A0. Flagged items get a second turn `[user P, assistant a1, user feedback + RETRY_SUFFIX]`, same system prompt (`prompt_v2.py:72–80`, `recovery.py:43–47`) | Only flagged items | A0's outputs | Effect of V without retrieval | Descriptive only (not in the test sequence) |
| **D0** | MQL-0008/0009 hybrid block: k = 3, `hybrid_masked`, w = 1.0, 13 L3–L5 exclusions | Full pass | — | Baseline | The comparator; also replicates the RAG effect against A0 (descriptive) |
| **D0V** | D0's first answer + the V turn for flagged items | Only flagged items (≈ 21% expected) | D0's outputs, by construction | **Step 1 (A3):** accuracy(D0 + V) − accuracy(D0) on the same first answers | "Flag-and-retry with schema feedback improves accuracy over the same first answers." **Not:** "schema feedback beats a generic second attempt" (no control). |
| **E** | D0 ranking; the 3 examples re-selected so each new structure signature is taken first; rank order kept; same k and budget | Full pass | — | **Step 3a:** effect of the selection rule alone | "Diversified selection changes accuracy by x [CI]". A non-rejection is inconclusive (§4.3). |
| **EF** | E's prompt plus a notes section on databases with a string-vs-number FK pair. The notes go to **every** question of that DB, joining or not. | Only items whose prompt hash differs from E's (the 2 holdout DBs with such pairs: race_track 30 + shop_membership 22 = 52 items) | E's records for all other items | **Step 3b:** effect of adding the notes on those DBs | Descriptive. A per-DB, all-questions estimand, not "on join questions". |
| **FULL** | EF's first answers + the V turn | Only flagged items | EF's outputs (E's for 664 items) | **Step 2:** package effect of E + F + V vs D0 | "The improved pipeline beats the baseline." No component attribution beyond steps 1/3. |

### B. The type-note arm, and the "reuse" ambiguity

The ambiguity resolves in favour of a real test. The notes enter the **prompt** of a **fresh** first-pass generation:
- `driver.prompt_delta` reuses a record only if the plan's prompt hash is unchanged (`driver.py:40`).
- For every item whose prompt contains a note, the hash differs, and `first_pass` is called with the new prompt (`driver.py:44`).
- No cached draft is fed into a second generation. Final predictions are not reused for changed prompts.
- The reuse is exact only because decoding is greedy *and* the prompt is byte-identical. That is tested: `test_fk_notes_flag_adds_only_genuine_incompatibilities` asserts an unaffected DB keeps D's prompt hash, and `test_driver_prompt_delta_generates_only_changed_prompts` covers the delta logic.

**Remaining confounder (R2).**
- E's full pass will run in batches of 8, as in MQL-0006..0009 on a T4 with emulated bf16. EF's 52 delta items will be generated in a separate pass with different batch neighbours and padding.
- Greedy outputs under padded batching are not guaranteed to be batch-invariant. MQL-0009's OOM fallback hit 72 vs 80 items depending on the run, so batch composition did vary.
- An EF − E difference on those 52 items therefore mixes the note effect with batch noise. Its size is unknown: no repo artifact measures batch invariance.

**Type classification is correct.**
- `prompt_v2.type_class` (lines 18–21) drops `null`/`missing`/`undefined` and maps int/long/double/decimal to one "number" class.
- A note is emitted only if the classes differ (line 34).
- On the MQL-0009 schemas, only `phone_market` fires (test). The raw helper would also flag int vs double (`company_employee`, `machine_repair`) and nullable keys (`chinook_1`). Those notes would be false, because MongoDB equality matches 1 and 1.0.
- **Edge case.** A field observed with mixed types, e.g. `["int", "string"]` vs `["int"]`, *is* flagged. That is directionally right: only the int rows will match. The note text ("an equality $lookup on them matches nothing") overstates it in that case. No such pair is known in the holdout schema census (§5.4); document only.

---

## 3. Retry (component V): critical questions

| Question | Answer from code |
|---|---|
| What triggers a retry? | `recovery.recover` (`recovery.py:35–55`): the first answer is unparseable by the registered evaluator's parser (`K9.normalize_prediction`, line 23), **or** `SchemaValidator.validate` returns ≥ 1 issue (V1 unknown collection, V2 unknown field path, V3 aggregation expression inside a query operator, V4 field-path string compared as a literal) |
| Deterministic and gold-independent? | Yes. The validator reads only the target DB's schema context (`validator.py:72–80`), the same context shown in the prompt. No execution, reference rows or gold SQL. Pure function; no randomness. |
| Does it receive the actual MongoDB error? | **No.** Validation is static. Runtime errors not covered by V1–V4 (unknown operator, bad `$group` accumulator, type errors) do not trigger a retry. MQL-0009 D had 17 execution errors; 6 were flagged statically. This is a design choice (no database on Kaggle) and should be stated as a scope limit. |
| Is only the failed query regenerated? | Only flagged *items* are regenerated, and their *whole* query is regenerated (not just the failing stage). The second turn contains the original prompt, the first answer and up to 6 issue lines (`validator.format_feedback`, line 289). |
| First vs final accuracy reported separately? | Yes, by construction: the first-attempt accuracy of D0V *is* D0, and the record keeps both `first_raw_text` and `raw_text` plus a trace (`driver.py:55`, `recovery.py:38–55`). The analysis plan lists fix/break-by-retry counts. |
| Can it improve without regressions? | Regressions are possible only on flagged items that were *correct*, and only if the retry has strictly fewer issues (`recovery.py:50–55`). On MQL-0009, 0–1 correct answers per run were flagged (D: 0 of 357). Regressions should be near zero, but the analysis must count them, not assume them. |
| Does the comparison isolate V? | It isolates "validator-triggered retry with schema feedback" on fixed first answers. It does **not** separate (a) the content of the feedback from (b) simply giving the model a second turn. That needs a control where the same flagged items get a generic message ("your query may be wrong; check it and return JSON"). |

### Power for step 1 (D0V vs D0)

Simulation (4,000 draws; flag rate 21.4% = MQL-0009 D 137/639; n = 716). "Break" is a deliberately pessimistic stress parameter: the probability that a *correct* answer is flagged and then broken. MQL-0009 suggests about 0.

| Repair rate among flagged | Breaks | Power (exact McNemar, α = 0.05) | Mean effect |
|---|---|---:|---:|
| 2% | 0 | 0.09 | +0.4 pp |
| 5% | 0 | **0.79** | **+1.1 pp** |
| 10% | 0 | 1.00 | +2.2 pp |
| 10% | 2% of correct | 0.24 | +1.0 pp |
| 20% | 0 / 2% | 1.00 / 0.95 | +4.3 / +3.2 pp |
| 30% | 0 | 1.00 | +6.4 pp |

**Findings:**
- **Power is not the problem; interpretation is (R4).** With near-zero breaks, McNemar rejects once there are about 6 fixes, i.e. about a 1 pp gain. Step 1 is therefore close to guaranteed whenever the model repairs even 5% of flagged answers. A "significant" result alone does not establish a practically meaningful improvement.
- The plan's statement "MDE ≈ 3 pp for D0V vs D0" (`docs/mql0010_experiment_plan.md` §7) is misleading. The detectable effect is about 1 pp when breaks are rare. The conclusion must rest on the size of the effect and its CI.
- Because step 1 gates everything else (A3), a near-automatic step 1 means steps 2–3 effectively get the full α. That is still valid fixed-sequence control (§4.4); it is just not much of a gate.

---

## 4. Statistical assessment

### 4.1 Tests

Exact two-sided McNemar on per-question correctness, with a question-level bootstrap CI, is appropriate for all paired binary comparisons here.
- No random-seed arms exist in MQL-0010, so the 5-seed issue from MQL-0009 does not arise.
- The registered plan correctly forbids pooling seeds.

### 4.2 Sample size and realistic minimum detectable effects (n = 716)

| Comparison | Discordance (dev-evidence basis) | Power at +2 pp | Power at +3 pp | Power at +4 pp |
|---|---|---:|---:|---:|
| E vs D0 | 8% (MQL-0009 seed-pair disagreement lower bound) | 0.43 | 0.79 | 0.97 |
| E vs D0 | 12% | 0.31 | 0.60 | 0.86 |
| E vs D0 | 16% (MQL-0009 D vs C) | 0.24 | 0.49 | 0.74 |
| EF vs E | only 52 changed items; 20–40% of them discordant | net +2: ≤ 0.05; net +5: ≤ 0.25; net +8: ≤ 0.71 | | |

**Findings:**
- **716 is adequate for V and for FULL** if the retry repairs at least about 10%. It is **inadequate for E** at its plausible effect size (≤ 2–3 pp).
- **F is unmeasurable** confirmatorily.
- **A practically meaningful 2 pp improvement from E cannot be detected reliably.**

### 4.3 Diversity (E): hypothesis, not result

**The supporting evidence is weak:**
- The dev proxy (hit@3 57.1 → 65.9) rewards "more distinct signatures in the top 3" almost by definition.
- The only randomised accuracy evidence (MQL-0009 within-question, exact structure shown) is +2.4 pp, p = 0.19.

**The experiment isolates E correctly:**
- same candidate ranking, same k, same budget, same rank order in the prompt;
- the top-1 example is always kept (tested).

**Missing:** a diversity control that is not structure-aware (e.g. pick by distinct source DB, or rank 1 + 2 random from the top 50). Without it, a positive E result cannot be attributed to *structural* diversity rather than diversity as such.

**What the result can support:** a positive E result can support "this selection rule helps". A null result cannot support "diversity does not help".

### 4.4 Amendment A3 and multiplicity

A3 (`preflight/amendment_A3_2026-11-19.json`) replaces the analysis plan's order with a fixed sequence at α = 0.05:
1. D0V vs D0;
2. FULL vs D0;
3. {E vs D0, EF vs E} with Holm.

**Is the procedure valid?** Yes. Fixed-sequence gatekeeping with Holm inside the last family controls the family-wise error rate at 0.05.

**Was there selection bias?** None from data. A3 was written before any MQL-0010 prediction, from power arguments that use MQL-0009 development evidence only.

**But A3 is not pinned (R1).**
- `configs/mql0010_experiment.json` pins `analysis_plan.json` (`analysis_plan_sha256`), which still names **FULL vs D0** as the primary H1.
- Neither the config nor any test references A3 or its hash, and no scorer exists yet.
- As things stand, two registered documents disagree about the primary hypothesis. A future scorer could implement either.

**A3 changes the headline claim.** The "primary" result is now V on the baseline, not the improved architecture. That is defensible (V is the best-evidenced component and the best-powered test), but every report must say so.

### 4.5 Sequential-order selection bias

None, provided the order is applied exactly as pinned and no step is reordered after unblinding. The gate structure means E and F are tested only if V and FULL succeed. Their unconditional power is lower than §4.2, which makes a null E even less informative.

---

## 5. Holdout integrity

### 5.1 Verified mechanically (`make_split.py`, re-checked by the audit script)

| Check | Result |
|---|---|
| Draw | `random.Random(20261119).sample(sorted 123 corpus DBs, 18)` (`make_split.py:84`); dev drawn with seed 20261120 from the rest (line 86) |
| Item set | 716 items, 18 DBs; ids sha256 equals the registered `ids_sha256` |
| DB overlap with the holdout retrieval view (corpus minus holdout DBs, 2,912 records) | 0 |
| Id overlap with the view | 0 |
| Holdout items whose own verified MQL is in the full corpus | **533 of 716.** Hidden only by the view and by the retriever's target-DB filter (`retrieval.py`, `rank_examples`: documents with `db_id == target` are skipped). Each holdout question's own record is in its own DB, so it is excluded even without the view. |
| DB, id and source-index overlap with the MQL-0009 639 | 0 / 0 / 0 |
| DB overlap with Spider dev 1034 | 0 |
| Exact / near-duplicate (token Jaccard ≥ 0.8) questions vs the 639 | 0 exact, 1 near (`spider_train_01507` ~ `spider_train_01796`, J = 0.889). Negligible. |
| L3–L5 near-duplicates vs the view | 13 pairs on 10 holdout items. These are **retrieval exclusions**, not evaluation exclusions: the 10 items stay in the 716 (sensitivity S2 drops them). Statements that "13 near-duplicates are excluded" are slightly misleading. |
| L6 identical gold SQL in the view | 14 items, report only |

### 5.2 Was it selected before inspection?

**Yes for the draw:** the seed and rule are in code that was written before the draw ran. **Since then, the designer has seen:**
- the 18 DB names and per-DB item counts;
- the holdout schemas, through a FK type census that printed race_track and shop_membership. This is schema-level inference-time information, not labels or outcomes.

Two further post-draw actions:
- **Amendment A1** (pool-v3 membership filter, 859 → 716) was applied after the per-DB counts were seen. It is a label-quality filter, applied for parity with how the 639 were built, and depends on no model outcome.
- **The L6 audit** read holdout gold SQL to compute identical-SQL matches. That is a leakage check, not used for design.

No component, threshold or parameter was chosen using holdout content:
- V's rules came from MQL-0009 failures;
- E's rule and pool size from the dev proxy;
- F was implemented before the holdout census ran.

### 5.3 Untouched, or merely unfitted?

**Untouched by model outcomes and by design decisions:** no MQL-0010 or other model output exists on these 716 questions.

**Not pristine:**
- The 18 DBs are corpus databases. Their questions went through the converter, verifier and AI review panel during corpus construction, and 533 have verified MQL in the approved corpus.
- That does not leak into any arm as long as the view is enforced (R3).
- But the holdout is a "corpus-construction-seen, model-unseen" set, not an externally fresh one.
- Spider train is public, so pretraining contamination of Qwen is possible. It affects absolute levels equally across arms, not the paired differences.

### 5.4 Contamination by the MQL-0009 analysis

**None at item level:** different databases, ids and source indices, and 1 near-duplicate question.

**Design-level only:** the 639 informed V's checks, and the confirmatory test runs on different databases. That is the intended firewall, and it holds.

### 5.5 View enforcement is by convention only (R3)

- `make_split.py` *computes* the view and its id hash, but nothing in `scripts/nl2mql_mql0010/src` *enforces* it. `planner.build_plan` ranks over whatever `docs` it is given.
- The Kaggle package builder is not written yet.
- **If the builder passed the full corpus:**
  - The holdout questions' own records would still be blocked by the target-DB filter.
  - But records of the *other 17* holdout DBs would become candidates.
  - The 13 L3–L5 exclusion pairs were computed only against the view, so near-duplicates from other holdout DBs would go unaudited.
- The risk is moderate, not catastrophic, but it is a silent failure mode.

---

## 6. Architectural hypotheses: what the design can and cannot conclude

| Hypothesis | Isolated? | Missing control / confounder | Cannot conclude |
|---|---|---|---|
| **V:** validation + one retry | Yes (D0V vs D0, same first answers) | Generic-retry control on the same flagged items; runtime errors are out of scope | That the *schema-specific feedback* is what helps; that V transfers to E's outputs (FULL includes that interaction but cannot separate it) |
| **F:** type notes | Mechanism yes (fresh inference on changed prompts) | Batch composition (R2); notes are shown to non-join questions too | Any confirmatory effect: 52 items, power ≤ 0.71 even at +8 net |
| **E:** diversity | Yes (selection only) | A non-structural diversity control; power | That diversity (structural or otherwise) does or does not help, if H3 is null |
| **C:** set-op coverage | Not tested | — | Anything |
| **Interactions** | Partly | FULL = E + F + V; no E + V arm without F (≈ FULL on 664 of 716 items) | Additivity of components |

---

## 7. Required corrections (smallest practical fixes)

These are described only; nothing was changed. Each needs authorization.

| # | Issue | Smallest fix |
|---|---|---|
| **R1** | A3 not pinned; two registered documents disagree on the primary hypothesis | Write `outputs/nl2mql_mql0010/preflight/registration_index.json`. It lists `analysis_plan.json`, A1, A2 and A3 with their sha256 and states "A3 supersedes analysis_plan.primary/secondary_gatekept". Add one test asserting those hashes. The future scorer reads the order from A3. |
| **R2** | EF vs E confounded by batch composition | In the EF delta pass, regenerate the **E prompts of the same 52 items in the same batches, with identical batch size and padding**, as E′. Register **EF vs E′** as the step-3b comparison (E′ reuses nothing). Cost: +52 generations (≈ 0.2 T4-h). Alternative: generate both at batch size 1. |
| **R3** | Holdout retrieval view not enforced | In the (not yet written) package builder, build `docs` only from the registered view. Assert `sha256(view ids) == split_registration.holdout_retrieval_view.ids_sha256` and `no doc.db_id ∈ holdout DBs`, with a unit test. The planner itself is unchanged. |
| **R4** | Step 1 is near one-sided, so p < 0.05 is achievable with about 1 pp | Add an interpretation rule by dated amendment (A4), without changing the test: report the effect and 95% CI, and call V "practically meaningful" only if the CI lower bound is ≥ +1.0 pp. Report regressions-by-retry explicitly. Correct the plan's "MDE ≈ 3 pp" statement for D0V. |

**Recommended, not required for GO:**
- **R5. Generic-retry control** (D0R): same flagged D0 items, second turn with a fixed message and no issue details. ≈ 150 generations (≈ 0.6 T4-h). Without it, the V conclusion must be worded as "flag-and-retry", not "schema feedback".
- **R6. Pre-state the E interpretation.** A non-rejection is inconclusive. Report the CI and the achieved discordance. Optionally add a non-structural diversity arm (rank 1 + top-50 picks from distinct source DBs) if E is a decision you need.
- **R7. Make F descriptive.** Mark EF vs E as descriptive in all reports. Its formal test can stay in the A3 family, but it will rarely be reached and is underpowered.
- **R8. Fix the wording** about "13 near-duplicates excluded" (retrieval exclusions; the items stay evaluated) and the mixed-type note text edge case.
- **R9. Process.** CLAUDE.md requires a ledger DA entry for quantitative dataset audits. The MQL-0010 split and leakage audit has none. Add it when authorised.

---

## 8. Decision

**CONDITIONAL GO.**
- Fix **R1–R4** before generation. They are documentation, pinning and package-builder obligations; no redesign is needed.
- Adopt **R5** if the project wants to claim that *schema-grounded* feedback, rather than a second attempt, is the mechanism.

**What this experiment can deliver:**
- a clean, well-powered estimate of the validation + retry effect;
- a valid package comparison;
- descriptive evidence on F.

**What it cannot deliver:**
- a decisive answer on diversity at n = 716;
- anything on corpus coverage.

Budget GPU time with those expectations.

---

## 9. Post-correction status (2026-11-19, after R1–R5 were authorised and implemented)

The user authorised R1–R5. They were implemented as **amendment A4**, written before any MQL-0010 prediction. The original config, analysis plan and A1–A3 are unchanged and still hash-pinned; A4 supersedes them where they overlap.

| Item | Implementation | Verification | Status |
|---|---|---|---|
| **R1** pin amendments | `preflight/amend_A4.py` (write-once) wrote `registration_index.json` (sha256 `11b56c79…`). It pins the config, analysis plan, split, A2, A3, A4 (`d463a399…`), all 9 source files and the corpus hash, and states the precedence A4 > A3 > plan > config. | `tests/test_mql0010_registration.py`: document hashes, source hashes, A4 ↔ `scoring.py` consistency (arms, retry sources, sequence, criterion, generic text, batch sizes); the original plan is intact | **Resolved** |
| **R2** batch-matched type notes | New control arm **EP**. `driver.matched_delta` generates E's prompt and EF's prompt for exactly the changed items in the same pass, one prompt per call (batch 1); unchanged items reuse E in both. The comparison is **EF vs EP**. | `test_matched_delta_generates_control_and_new_only_for_changed_items`; `scoring.verify_exports` checks sources and prompt hashes; dry run: 52 items → 104 calls | **Resolved.** The residual difference is only the note. Power is unchanged (≤ 0.71), so F stays descriptive. |
| **R3** view enforcement | `src/holdout_view.py` (`load_view`: corpus hash + registered view id hash + count + no holdout DB; `assert_docs_exclude`) and `planner.build_plan(forbidden_db_ids=…)` | Registration tests, including three tamper cases (wrong view hash, a holdout DB dropped from the list, modified corpus bytes); dry run on the real holdout: 2,912 docs, 0 holdout-DB docs | **Resolved** in code. The future package builder must call these; plan step 1 says so. |
| **R4** power claim + criterion | A4 withdraws "MDE ≈ 3 pp". Step 1 rejects at about +1 pp, so *practically meaningful* requires the CI lower bound ≥ +1.0 pp. Implemented in `scoring.analyze` (`practically_meaningful`). The experiment plan §7 carries a dated correction. | `test_practical_criterion_can_fail_while_test_rejects` (12 fixes on n = 2,000: p < 0.001 but not meaningful) | **Resolved** |
| **R5** generic-retry control | New arm **D0G**: same D0 first answers, trigger, acceptance rule, retry budget, system prompt, model, decoding and batch size as D0V. The second turn is the fixed `GENERIC_FEEDBACK` sentence (format errors get the same `FORMAT_FEEDBACK` in both). New hypothesis **D0V vs D0G** in step 2. | `test_generic_feedback_has_no_schema_content`, `test_generic_and_schema_share_trigger_and_format_message`, `test_retry_arms_schema_vs_generic_retry_the_same_items`; `verify_exports` requires identical retried item sets | **Resolved** |

**Test sequence after A4:**
1. D0V vs D0, plus the practical criterion.
2. If step 1 rejects: Holm over {FULL vs D0, D0V vs D0G}.
3. If both step-2 hypotheses reject: Holm over {E vs D0, EF vs EP}.

This is valid serial gatekeeping: each family is tested at the full α only if every hypothesis in the previous family rejected.

**Cost of the gate on E.** E is now also gated on the feedback-content test, which further lowers its already low unconditional power. That was accepted deliberately: E was already labelled inconclusive-if-null.

**Power of the new D0V vs D0G test** (Holm worst-case level 0.025; 3,000 simulations; flag rate 21.4%):

| Repair rate, schema vs generic | Generic repairs a subset of schema repairs | Independent repairs |
|---|---:|---:|
| 10% vs 5% | 0.66 | 0.21 |
| 20% vs 10% | 0.99 | 0.53 |
| 20% vs 5% | 1.00 | 0.96 |
| 30% vs 15% | 1.00 | 0.77 |
| 30% vs 20% | 0.99 | 0.36 |

**Interpretation.** A modest content advantage that is not nested is hard to detect. A non-rejection of D0V vs D0G therefore means "content effect not shown". It does not mean "content irrelevant".

**Compute** after A4 ≈ **10.0 T4-hours**:
- first passes: 7.2;
- matched delta, 104 calls: 0.4;
- retries for 4 arms, about 620 calls: 2.4.

**Tests:** 50 passed (`outputs/nl2mql_mql0010/test_results_A4.txt`).

**Mocked dry run:** passed end to end on the real holdout plans (`outputs/nl2mql_mql0010/dryrun/dry_run_report.json`).

### Updated decision

| Level | Decision |
|---|---|
| **Design** | **GO.** R1–R5 are resolved. R6–R9 remain documentation recommendations: E inconclusive if null; F descriptive; wording; ledger DA entry. |
| **Execution** | **Not yet.** It needs, each with separate authorisation: (1) building the Kaggle package and notebook from the tested components, with the pinned tokenizer; (2) Atlas provisioning of the 18 holdout DBs plus the read-only grant. See `docs/mql0010_final_execution_plan.md` §5. |

**Stopped for human review.**
