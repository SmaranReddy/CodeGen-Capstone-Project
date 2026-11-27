# MQL-0010: NL→MQL RAG architecture redesign

Date: 2026-11-19.

**Inputs.**
- `docs/mql0009_post_experiment_analysis.md`, cited as "PEA §x".
- Development evidence in `outputs/nl2mql_mql0010/dev_evidence/`.

**Implementation.** `scripts/nl2mql_mql0010/src/`, tested by `tests/test_mql0010_*.py` (35 passed).

**Status.** Designed, implemented and unit/integration-tested. **No MQL-0010 prediction or score exists.**

---

## 0. Executive summary: the three highest-impact improvements, ranked by evidence

1. **Schema-grounded validation + one controlled retry (V).**
   - **Evidence:** the static checker flags 137 of the hybrid run's 282 wrong MQL-0009 answers and 0 of its 357 correct ones. It covers 67 of the 100 empty-result failures, the largest error bucket, which no retrieval variant changed. It raises 0 false alarms on 3,445 verified corpus pipelines.
   - **Unknown:** the 3B model's repair rate. MQL-0010 tests it first (D0V vs D0, amendment A3), with clean attribution by reusing first-pass outputs.
2. **Signature-diversified example selection (E).**
   - **Evidence:** 52% of hybrid triples repeat one structure. Diversifying, without changing scores, raises the share of dev questions whose gold structure is shown from 57.1% to 65.9%.
   - **Caution:** the proxy partly rewards diversity by construction, and the only randomised accuracy evidence for exposure is about +2 pp (p = 0.19). E is a hypothesis under test (E vs D0, likely underpowered), not an expected gain.
3. **Corpus coverage for subqueries and set operations (proposal only).**
   - **Evidence:** about 14% of evaluation questions need constructs with 0 corpus examples, and they sit near 11% in every run.
   - **Plan:** a separately versioned candidate corpus via a copied, extended converter with verifier-3.1 checks (§8). Not built.

Below these: type-incompatible join-key notes (F) have a small, DB-dependent ceiling. Several plausible ideas were evaluated and rejected on evidence (§2): learned structure scoring, `_id` post-processing, tie-break changes and confidence gating.

---

## 1. Verdict on the current architecture

The MQL-0008/0009 system treats NL→MQL as "show three solved questions whose masked wording looks similar". MQL-0009 shows this is *better than random* (+3.8 pp), but the design caps its own gain:

1. **It optimises the wrong signal for the dominant failure.** About 100 of each run's roughly 280–320 errors are executable pipelines that return nothing, because they reference fields or collections the schema does not have at that point in the pipeline (PEA §6.1).
   - Retrieval is blind to this: the empty-result count is identical with zero-shot, random, fixed or hybrid examples.
   - The schema needed to catch it is already in the prompt; the system just never *checks* the output against it.
2. **Its structural signal is weak and coarse.**
   - Regex cues match gold structure exactly on 33% of training questions.
   - With weight 1.0 the cue Jaccard dominates and quantises the score: plateaus at 2.0 and 1.0, ties at ranks 3/4 on 12% of items, broken by `doc_id`.
3. **It concentrates rather than hedges.** 52% of selected triples repeat one structure. When that guess is wrong, all three demonstrations mislead.
4. **Its corpus cannot express about 14% of the task.** It has no subqueries, set operations, anti-joins, `let`/`pipeline` lookups or type conversions. On those questions retrieval is neutral or harmful.
5. **About half its measured gain is output-shape priming** (`_id`, exact columns) that random examples also deliver.

Statistically significant does not mean sufficient: hybrid RAG still answers 44% of these questions wrongly, and 93% of those wrong answers (263 of 282) execute without error.

---

## 2. Evidence-ranked design decisions

| # | Component | Decision | Evidence (strength) | Where |
|---|---|---|---|---|
| **V** | Schema-grounded static validation + one controlled retry | **Adopt (primary)** | Flags 137 of D's 282 wrong answers and 0 of its 357 correct; 0/3,445 false alarms on verified corpus MQL; covers 67 of D's 100 empty-result failures (strong for *detection*; repair rate unknown) | `validator.py`, `recovery.py`, `driver.py` |
| **E** | Signature-diversified example selection | **Adopt, test as a hypothesis** | Dev retrieval proxy hit@3 57.1 → 65.9 (attainable 65.5 → 75.6) with the ranking unchanged; randomised exposure effect about +2 pp (p = 0.19). Moderate: the proxy partly rewards diversity by construction | `retriever_v2.select_diverse`, `planner.py` |
| **F** | Notes for type-incompatible join keys (numeric types unified) | **Adopt behind a flag, low priority** | MQL-0008: 42 items on incompatible keys, 0% correct, 0 conversions emitted; MQL-0009: 1 genuine pair (4 items). The dormant helper's raw-type comparison would emit *false* notes for int vs double | `prompt_v2.fk_type_notes` |
| S | Learned structure predictor (naive Bayes) as the structural score | **Evaluated, not adopted** | Train-side cross-validation: exact signature 33.0 → 36.8%, macro-F1 0.517 → 0.586; but dev hit@3 drops 57.1 → 50.2 alone, and adds at most +1.8 on top of diversity (amendment A2) | `structure.py` (kept for evidence and the gate) |
| T | Lexical tie-break instead of `doc_id` | Not adopted | No change on the dev proxy | flag in `retriever_v2.py` |
| D | Confidence gating (no examples for unsupported families) | Not adopted | Ceiling about 1 item on MQL-0009; NB gate precision 0.77, recall 0.35 | — |
| P | `_id` post-processor | **Rejected** | Offline counterfactual −10 to −17 correct per run | `offline_counterfactuals.py` |
| C | Corpus coverage for subqueries and set operations | **Proposal** (§8) | About 14% of the task unreachable; needs converter + verifier work and a separately versioned corpus | — |
| G2 | Example-schema snippets, idiom canonicalisation | Proposal | Idiom copying is strong (`__jN` 46% vs 0%), but no accuracy evidence that it harms | — |

Not used, on purpose: embeddings, a vector database, an LLM reranker or an LLM-written query plan. Nothing in the evidence points at the semantic channel as the bottleneck. The lexical channel's randomised effect (+2 pp) is no larger than the structural one, and the dominant failure is not a retrieval failure at all.

---

## 3. Architecture

### 3.1 Diagram

```
 question q, target DB schema S (collections, observed field types, FKs, sample docs)
        │
        ▼
 ┌─────────────────────────┐        approved corpus v2.1 VIEW (target DB and holdout DBs removed;
 │ R  candidate ranking    │◄────── corpus file unchanged) → documents (masked question, SQL-derived
 │    UNCHANGED MQL-0008   │        structure labels, MQL)
 │    hybrid_masked score  │
 └──────────┬──────────────┘
            │ ranked candidates (top 50)
            ▼
 ┌─────────────────────────┐
 │ E  diversified selection│  best document of each NEW structure signature, k = 3
 └──────────┬──────────────┘
            ▼
 ┌─────────────────────────┐   F (flag): schema-derived notes, only for FK pairs whose
 │ prompt assembly         │◄─ type classes differ (string vs number), numeric types unified
 │ frozen zero-shot prompt │
 │ + unchanged rag_block   │
 └──────────┬──────────────┘
            ▼
     Qwen2.5-Coder-3B (greedy) ──► first answer a1
            │
            ▼
 ┌─────────────────────────┐  V1 collections exist      V3 no aggregation expressions inside
 │ V  static validator     │  V2 field paths resolve,      $match query operators without $expr
 │    (schema only,        │     tracked through stages V4 no "$field" strings compared as
 │     no DB access)       │                               literals in $match
 └──────────┬──────────────┘
     0 issues │   issues / unparseable
        ┌─────┴───────────────┐
        ▼                     ▼
     keep a1     one retry: [user: prompt, assistant: a1, user: issues + "fix and return JSON"],
                 same system prompt → a2; keep a2 only if it parses with FEWER issues
                                     │
                                     ▼
                              final query → (scoring on Atlas, separate step)
```

### 3.2 Component R: candidate ranking (unchanged)

`retrieval.Retriever(method="hybrid_masked", hybrid_weight=1.0)` is used through `retriever_v2.StructureAwareRetriever` in baseline mode:

  score(d) = BM25(mask(q), mask(d)) / max_d' BM25 + 1.0 · Jaccard(cues(q) \ {select, where}, labels(d) \ {select, where})

Ties are broken by `doc_id`. Documents are removed if they come from the target DB, or duplicate an id, an MQL or a normalised question.

It is kept unchanged because:
- it is the only retriever with a registered, significant relevance effect (MQL-0009 H1);
- its replacement candidates did not beat it on the dev proxy.

### 3.3 Component E: diversified selection

```
def select_diverse(ranked, k=3):                 # ranked: best-first, after exclusions
    chosen, seen, rest = [], set(), []
    for doc in ranked[:50]:
        s = signature(doc.labels)                 # SQL-derived labels minus select/text_number
        if s in seen: rest.append(doc); continue
        seen.add(s); chosen.append(doc)
        if len(chosen) == k: break
    chosen += rest[:k - len(chosen)]              # fewer than k signatures: fill in rank order
    return sorted(chosen, key=rank)               # keep the ranking order in the prompt
```

**Properties (tested):**
- The top-1 document is always kept.
- It never shows fewer distinct structures than the baseline.
- It is deterministic.
- It does not change the candidate scores.

**Rationale.** The retriever's top-1 structure is right for about 46% of dev questions. Spending ranks 2 and 3 on the *second and third most plausible structures*, rather than two more copies of the first, raises the chance the right structure is visible at all (dev hit@3 +8.7 pt). The 3B model then chooses among patterns, conditioned on the schema and question.

**Risk.** Weaker repetition when the top-1 guess is right. That is why E is an ablation arm with its own hypothesis (E vs D0), not an assumption.

### 3.4 Component V: static validation and controlled recovery

**Validator** (`validator.SchemaValidator`, about 300 lines, no dependencies). It tracks the set of field paths available to each stage:

| Stage | Effect on the available fields |
|---|---|
| start | the start collection's fields + `_id` |
| `$match` / `$sort` / `$limit` / `$skip` | unchanged (keys checked) |
| `$lookup` (equality form) | adds `alias` and `alias.<every foreign field>`; `from`, `localField`, `foreignField` are checked |
| `$lookup` (`let`/`pipeline`) | adds `alias` as opaque (anything under it accepted) |
| `$unwind` | path checked; shape unchanged |
| `$group` | `{_id, accumulators}`, plus `_id.<k>` for compound keys |
| `$project` / `$addFields` / `$set` / `$unset` | new shape; inclusion / exclusion semantics |
| `$facet`, `$replaceRoot`, `$bucket`, `$unionWith`, unknown stages | **opaque**: nothing after is checked, so no false alarms |

Issue codes:
- **V1** unknown collection, or unknown `$lookup.from`;
- **V2** unknown field path, with a suggestion naming the collection that has the field or the alias path available;
- **V3** aggregation-expression operator used as a query operand without `$expr`;
- **V4** `"$field"` string compared as a literal inside `$match`.

**Recovery** (`recovery.recover`). The decision rule was fixed before any MQL-0010 output:

```
a1 = generate(system, [user: P])
obj1, I1 = parse(a1), validate(obj1)            # parse = the registered evaluator's parser
if obj1 is not None and I1 == []:  return a1
feedback = FORMAT_MSG if obj1 is None else format_feedback(I1)   # at most 6 issues
a2 = generate(system, [user: P, assistant: a1, user: feedback + RETRY_SUFFIX])
return a2 if score(a2) < score(a1) else a1       # score = #issues, unparseable = +inf
```

Properties:
- One retry at most.
- No database execution and no gold information.
- The same system prompt, which carries the result contract.

**Why it should improve execution accuracy:**
- It targets the largest bucket directly.
- Its trigger is almost perfectly specific on MQL-0009: 0–1 correct answers flagged per run, so the "keep only if strictly better" rule rarely touches correct answers.
- The feedback is actionable: "field 'city' does not exist here; it exists in hr_1__locations.city; join that collection first".

**What is unknown** is the 3B model's repair rate. MQL-0010 measures it in isolation (D0V vs D0, the first test under amendment A3).

### 3.5 Component F: type-incompatible join-key notes

- **Rule.** For each declared FK (x → y), map the observed BSON types of both sides to classes:
  - {int, long, double, decimal} → number;
  - null and missing are ignored.
  
  Emit a note only if the classes differ.
- **Rendering.** The existing `NOTES_HEADER` text (MQL-0008 `rag_block`) is used verbatim. It recommends the `let`/`pipeline` `$lookup` form with `$toString`/`$toInt`.
- **Budget.** Notes have budget priority and are dropped only if they cannot fit at all.
- **Prompt impact.** Prompts change only on databases with such pairs. On the MQL-0010 holdout that is 2 of 18 DBs, from a schema-only census.

### 3.6 Data structures and interfaces

```
Document  {doc_id, kind="example", db_id, question, masked_question, patterns[], mql, content}   # mql0008_core.build_docs
Ranked    (doc, score, trace{score, lexical, structure})                                           # retriever_v2
PlanRow   {id, db_id, arm, selected_ids[], selected_scores[], selected_ranks[], selected_db_ids[],
           selected_traces[], ranked_top10[[id, score]], excluded_by_rule[], schema_notes[], dropped_budget[],
           context_tokens, prompt_tokens_est, empty_context, user_prompt_sha256, prompt_sha256, flags{}}
Issue     {code, stage, message, suggestion}
Prediction{id, db_id, arm, status, raw_text (scored), first_raw_text, source ("generated"|"reused:<arm>"),
           prompt_sha256, recovery{first_parsed, first_issues[], retried, feedback, retry_parsed,
           retry_issues[], kept}, experiment}
Arm flags {family, k, structure_scoring, tie_break, diversify, fk_type_notes, validate_retry, candidate_pool}
```

**Decision logging.** Every retrieval decision (selected ids, ranks, score traces, exclusions, notes, budget drops) and every recovery decision is written per item.

**Reproducibility.** `planner.user_text_for` rebuilds each prompt from the plan and refuses a hash mismatch.

---

## 4. End-to-end pseudocode

```
INPUT  question q, db_id, schema context S, frozen zero-shot prompt (system, base_user), arm flags
docs     = build_docs(corpus_view)                                   # view excludes target + holdout DBs
ranked   = Retriever.rank_examples(q, schema_info(S))                # unchanged hybrid score
ranked   = [r for r in ranked if r.id not in exclusions[q.id]]       # registered leakage exclusions
ordered  = select_diverse(ranked[:50], k) + rest   if flags.diversify == "signature"   else ranked
notes    = fk_type_notes(S, numeric_unified)        if flags.fk_type_notes            else []
chosen   = budget_loop(ordered, k, notes, max_ctx=1500, n_ctx=8192, max_new=1024)   # skip, never truncate
user     = insert_block(base_user, render_block(db_id, chosen, notes))              # empty ⇒ zero-shot prompt
a1       = LLM(system, [user])                                        # greedy
final, trace = recover(system, user, a1, SchemaValidator(S), LLM) if flags.validate_retry else (a1, None)
RETURN final, plan_row, trace
```

---

## 5. Latency and token overhead (estimates from MQL-0009 run configs, T4, bf16 emulated)

| Step | Cost |
|---|---|
| Ranking + diversification | < 50 ms per question on CPU (pure Python BM25 over about 3k documents) |
| Validation | < 5 ms per question |
| First pass | zero-shot 9.9 s/item; hybrid 12.8 s/item (mean input 1,661 tokens, of which 422 are the reference block; output 117 tokens) |
| Retry | only for flagged answers: about 21% of items on MQL-0009 (131–149 of 639); input ≈ first prompt + first answer + feedback ≈ 1.9k tokens; ≈ 13–15 s each |
| Expected total per item | about 12.8 + 0.21 × 14 ≈ **15.7 s** (+23% over hybrid); tokens ≈ +0.21 × 2.0k ≈ +420 input tokens per item on average |
| Diversification | no extra tokens (still 3 examples) |
| F | +1–3 lines on affected databases only |

---

## 6. Failure modes and safeguards

| Failure mode | Safeguard |
|---|---|
| Validator false alarm on a correct answer, followed by a worse retry | The retry is kept only if it has strictly fewer issues. Opaque stages stop checking. False alarms measured at 0/3,445 corpus pipelines and 0–1 per 639-item run. Reported per arm (fix vs break by retry). |
| Retry introduces a new, undetected error | Measured directly: D0V vs D0 discordance is attributable only to kept retries |
| Feedback leaks gold or evaluation information | The validator sees only the schema context already in the prompt. No execution, no reference rows. Tested. |
| Diversity removes a correct repeated pattern | E vs D0 is its own hypothesis; adoption only if it wins |
| Notes are false (int vs double) | Numeric unification, tested on synthetic and real schemas (only `phone_market` fires on the 639) |
| Non-determinism across batches | Derived arms reuse first-pass outputs by hash (§MQL-0010 plan), so V and F comparisons are free of batch noise |
| Silent drift of the frozen baseline | With every flag off the planner reproduces the registered MQL-0009 D plan for 639/639 items, byte-identical prompts (integration test) |
| Source drift after registration | Source hashes pinned in `configs/mql0010_experiment.json`; a test fails on drift |

---

## 7. Why each adopted component should raise execution accuracy

- **V** converts a detectable class of silent failures (empty results from impossible field paths, literal-compared expressions) into explicit, schema-grounded feedback. This is the only proposed component aimed at the largest error bucket.
  - **Upper bound**, if every flagged answer were repaired: about 137 items (21 pp) on MQL-0009. Not an expectation.
  - **Plausible range**, with a 15–40% repair rate for a 3B model: +3 to +9 pp.
- **E** raises the probability that the needed multi-stage pattern is on screen. That is the condition under which hybrid RAG gained most over random (+5.8 vs +1.4 pp, PEA §5.3). The expected effect is small (≤ 2–3 pp) and may be zero.
- **F** gives the only route to correct answers on incompatible-key joins. MQL-0008 showed 0/42 without it. The ceiling here is small and DB-dependent.

---

## 8. Proposed, not implemented: corpus coverage for subqueries and set operations

**Gap.** Spider train has 571 nested, 388 anti-join, 217 intersect and 62 union queries; the approved corpus has none. These families are 14% of the MQL-0009 evaluation, about 11% accurate in every run.

**Proposal: a separately versioned candidate corpus (`sql2mql_v2_2_candidate`).**
1. Extend a *copy* of the converter with patterns that have exact MongoDB equivalents:

   | SQL construct | MQL pattern |
   |---|---|
   | `NOT IN (subquery)` / `EXCEPT` on keys | `$lookup` + `$match {alias: {$size: 0}}` (no `$unwind`) |
   | `INTERSECT` on one column | `$group` + `$addToSet` per side + `$setIntersection` via `$facet` |
   | uncorrelated scalar subquery (`> (SELECT AVG …)`) | `$facet` / two-phase `$group` + `$expr` |
   | `UNION` | `$unionWith` + `$group` for de-duplication |

2. Verify every record with verifier-3.1 under TC-1 on the real databases and fixtures, exactly as v2.1.
3. Run the AI-panel review (G-AIR).
4. Measure coverage and verifier acceptance.
5. Test it in its own experiment: a corpus arm vs v2.1 with everything else fixed. **The approved v2.1 corpus and its converter stay unchanged.**

**Expected value.** Bounded by the base rate (≤ 14% of items) and by the 3B model's ability to follow these longer patterns. Worth doing after V, because V's empty-result checks also catch many broken set-op attempts.

---

## 9. Tradeoffs

- **Simplicity over sophistication.** Two new runtime components (E, V) and one optional note (F); no new dependency.
- **Generation cost.** V adds a second pass for about a fifth of the items; E and F add nothing.
- **Generality.** V is MongoDB-semantic, not dataset-tuned. Its checks hold for any schema.
- **Scope.** V cannot catch semantically wrong but schema-valid queries (wrong literal, wrong join direction, missing filter): the other roughly 50% of wrong answers.
