# Week 3: RAG baseline for NL → MQL (Qwen2.5-Coder-3B, expanded 455 benchmark)

**Bottom line.** On the 72-item validation partition, retrieving three structurally similar
solved examples from other databases raised Qwen2.5-Coder-3B execution accuracy from
**33/72 = 45.8% to 43/72 = 59.7%** (+13.9 points, paired bootstrap 95% CI [+1.4, +26.4]).
That compares with a same-session zero-shot rerun using the same model, decoding, validator,
Atlas executor and comparator.

The evidence is **suggestive, not conclusive**:
- exact McNemar p = 0.053 (16 items gained, 6 lost);
- Holm-adjusted across the three planned validation contrasts, p = 0.16;
- against a random-example control, the gain is +8.3 points, 95% CI [−2.8, +19.4].

Adding MongoDB guide cards did not help: 41/72, with prompts ~2.4× longer and ~55% slower.
An exploratory variant that adds notes on mismatched foreign-key types gave verdicts identical
to plain retrieval (43/72).
The test partition was not used.

**Three separate experiments are reported. They are not pooled.**

| experiment | partition | hardware | baseline | where |
|---|---|---|---|---|
| validation pilot | validation, 72 items | CPU, one session | same-session `zs_rerun` | §6 |
| earlier CPU train run | train, 318 items | CPU | stored Week 2 zero-shot (148/318) | §7 |
| **fresh clean control** | train, 318 items | Kaggle GPU (T4), one backend, KV cache reset per item | fresh `zs_rerun` and `random_k3`, generated in the same run | §7b |

**Clean control** (fresh on train, all three arms generated in one Kaggle run, analysis plan
frozen before any output existed):

| arm | correct | accuracy [Wilson 95%] |
|---|---:|---|
| zero-shot | 160/318 | 50.3% [44.8, 55.8] |
| random examples | 155/318 | 48.7% [43.3, 54.2] |
| relevant RAG (hybrid) | 172/318 | 54.1% [48.6, 59.5] |

- **Hybrid vs random** (pre-specified primary contrast): +5.3 points, bootstrap 95% CI
  [+1.3, +9.7], McNemar p = 0.021, **Holm p = 0.064**.
- **Hybrid vs zero-shot:** +3.8 points, CI [−1.3, +8.8], McNemar p = 0.16, **Holm p = 0.32**.

**Neither comparison is significant after Holm correction.** The CI and the nominal McNemar p
for the primary contrast are not adjusted for multiplicity. The train partition is **not a blind
test set**, and the 65-item **test partition remains untouched**.

The earlier CPU train figure (+7.2 points, p = 0.003) was measured against the *stored* Week 2
zero-shot baseline, 148/318. The fresh same-session zero-shot scores 160/318, so most of that
apparent gain was baseline drift: hybrid itself scored 171 on CPU and 172 on GPU.

Ordering stability: one hybrid prediction (0934) flips on re-execution, and one (0667) is
correct only because of document order. Counting both as wrong gives hybrid 170/318; hybrid vs
random is then +4.7, Holm p = 0.12. No conclusion changes (§7b).

Contents: 1 Research and design decisions · 2 Splits · 3 Corpus · 4 Pipeline · 5 Retrieval
selection (offline) · 6 Validation pilot · 7 Earlier CPU train run · 7b Clean Kaggle control
(train) · 8 Failure analysis · 9 Testing and review · 10 Limitations · 11 Week 4
recommendations · 12 Reproduction

---

## 1. Research and design decisions

**Advisor.** An advisor model reviewed the design twice: before implementation and before the
train-partition runs. An independent cold-context reviewer audited the code (§9). Each
recommendation adopted is noted where it applies.

**Sources consulted** (fetched 2026-10-20):
- MongoDB manual, version selector "9.0 (Current)"; the Atlas cluster runs 8.0.34. Pages: the
  SQL→aggregation mapping chart, `$lookup`, `$unwind`, `$group`, `$match`, `$project`, `$sort`,
  `$count`, `$toDouble`, `$convert`, `$setIntersection`, `$setDifference`, `$unionWith`, BSON
  comparison order, `$facet`, `$size`, `$in`, `$regexMatch`, `$regex`.
- Papers:
  - Gao et al., *Text-to-SQL Empowered by LLMs: A Benchmark Evaluation* (DAIL-SQL, arXiv
    2308.15363). Selects examples by masked-question and SQL-skeleton similarity, and removes
    cross-domain detail from examples.
  - Nan et al., *Enhancing Few-shot Text-to-SQL…* (arXiv 2305.12586). Selection by SQL-syntax
    similarity plus diversity beats random selection (Codex on Spider: 76.0 random vs 78.3
    similarity vs 80.3 similarity-diversity).
  - Chang & Fosler-Lussier, *How to Prompt LLMs for Text-to-SQL* (arXiv 2305.11853). With
    cross-domain demonstrations, gains level off after about 4 examples and long prompts can
    hurt. In-domain demonstrations help more.
  - Lewis et al. 2020 (RAG definition).
  - BEIR (Thakur et al. 2021): BM25 is a robust baseline.

| decision | choice | why (evidence) |
|---|---|---|
| retriever family | local sparse retrieval (BM25 / TF-IDF), no embeddings | BEIR: BM25 is a robust zero-shot baseline. No paid APIs, CPU-only, deterministic. A 280-document corpus needs no vector store. |
| question representation | DAIL-SQL-style masking (schema words → `sch`, numbers → `num`, literals → `val`) tested against raw text | every example comes from a different database than the query (database-disjoint split), so domain nouns cannot transfer; what transfers is the query structure |
| hybrid | masked BM25 (normalised) + w · Jaccard(question cues, example's SQL patterns) | Nan et al. and DAIL-SQL report that structural similarity drives example usefulness. Cues are computed from the question and schema only. |
| selection protocol | retriever and weight chosen **offline**, on train leave-one-database-out, with a structural relevance proxy and no generation | keeps validation generation runs for confirmation rather than search |
| k | 3 examples | Chang & Fosler-Lussier: cross-domain gains plateau around 4. Keeps the 3B prompt short (+344 tokens on average). |
| context placement | one block inserted before the final `Question:` line of the unchanged zero-shot prompt | the only difference from zero-shot is the context; k = 0 is byte-identical (tested) |
| schema documents | in the corpus, **not retrieved** | the full target schema is already in the zero-shot prompt verbatim; retrieving it again would add tokens without information |
| controls | same-session zero-shot rerun (k = 0) plus a seeded random-example control | the rerun separates "RAG" from run-to-run nondeterminism; the random control separates "relevant retrieval" from "any few-shot examples" (advisor recommendation) |

---

## 2. Splits (`data/splits_expanded/`)

The v1 split (`data/splits/`, 279 items, 195/42/42) does not cover the 176 items added in
the expansion. A new split was created **without touching the v1 files**. Each expanded
example inherits the v1 partition of its `db_id`. All 20 databases have a v1 assignment, so the
split is database-disjoint by construction. Every v1 item keeps its v1 partition, and no new
randomness or model output was used.

| partition | n | share | databases | easy / medium / hard / extra |
|---|---:|---:|---|---|
| train | 318 | 69.9% | car_1, concert_singer, cre_Doc_Template_Mgt, dog_kennels, flight_2, orchestra, poker_player, singer, student_transcripts_tracking, voter_1, world_1, wta_1 | 84 / 140 / 45 / 49 |
| validation | 72 | 15.8% | battle_death, course_teach, museum_visit, network_1, real_estate_properties | 15 / 34 / 15 / 8 |
| test | 65 | 14.3% | employee_hire_evaluation, pets_1, tvshow | 16 / 30 / 12 / 7 |

`python scripts/spider1/make_splits_expanded.py` re-derives and verifies the files, the
manifest, and the preservation of v1 partitions. **Retrospective caveat:** the stored zero-shot
predictions for all three models (3B 217, 7B 287, gpt-oss 385 of 455) were generated *before*
this split existed. Their per-partition scores are not untouched held-out results.

---

## 3. Retrieval corpus (`data/rag/corpus_v1/`, built by `scripts/rag/build_corpus.py`)

| kind | n | content | provenance / verification |
|---|---:|---|---|
| example | 280 | train question + MQL | a **saved zero-shot prediction**; no MQL was written by hand or newly generated. One per train item, taken from the first of gpt-oss-120b → Qwen-7B → Qwen-3B whose saved record is correct **and** whose raw output, re-executed at build time through the benchmark scorer (parse → read-only validation → Atlas → `exec_match` vs the SQLite gold result), is still correct. Stores the source file, a sha256 of the exact source JSONL line, the prompt version and the expected-rows hash. |
| guide | 18 | MongoDB pattern cards (`data/rag/sources/mongodb_guides.yaml`) | paraphrases of the cited manual pages. Each card's example pipeline is **executed at build time**: inline `$documents` input checked against a literal expected output, or a train-database collection checked with `exec_match` against an equivalent SQLite query. All 18 pass. |
| schema | 80 | one per Atlas collection (fields, observed BSON types, declared SQL types, FKs) | inference-time information; kept for completeness, never retrieved |

- **Example sources:** 265 gpt-oss-120b and 15 Qwen-7B. Most demonstrations were therefore
  written by a much stronger model; this is part of what "RAG" means here.
- **Skipped:** 38 train items have no example:
  - 14 are flagged `ambiguous` and skipped by policy;
  - 24 have no verified correct prediction.
- **Metadata:** each example stores its database, hardness, SQL-derived pattern labels (train
  gold SQL, used as metadata only), MQL operators and masked question.
- **Determinism:** `build_corpus.py --check` rebuilds in memory and compares byte for byte, and
  it reproduces (sha256 `98dc8273…`). The manifest records the hash of every input file.

**Leakage prevention** (`scripts/rag/leakage.py`, run at every build and in the tests). For
every retrievable document, the build fails if any of the following holds:
- it comes from a validation or test database, or from a held-out example id;
- it contains a held-out question or a held-out gold SQL;
- it names a held-out database or any held-out collection;
- it duplicates another document's id.

Schema documents may describe any database, but must not contain question, SQL or answer
text. The tests inject each leak type and assert the checker catches it.

At query time the retriever also drops documents from the target database. Validation queries
can never see their own database for that reason too, and train queries see only *other* train
databases.

---

## 4. Pipeline (`scripts/rag/`)

```
question + target schema
  → retrieval.py      Retriever.rank_examples / rank_guides  (deterministic; target-db docs removed;
                      duplicates by id, MQL or normalised question collapsed)
  → rag_prompt.py     build_messages: greedy by rank under max_context_tokens (1500) and
                      prompt + max_new_tokens + 64 ≤ n_ctx (8192); vocab-only GGUF tokenizer;
                      block inserted before "Question:"; empty selection ⇒ zero-shot prompt
  → providers.py      unchanged LlamaCppProvider (temperature 0, top_p 1, seed 42, 1024 new tokens)
  → run_eval.score_output   unchanged: parse → validate_mql + deep forbidden-operator scan
                      → Atlas aggregate (read-only user, maxTimeMS 30 s) → exec_match
  → executions.jsonl  per record: config and its hash, code hash, query text and cues, top-10
                      candidates (id, rank, score, db), selected and budget-dropped ids, the exact
                      context, raw output, MQL, result rows, verdict, retrieval / context /
                      generation / scoring / total latency, token counts
```

- **Configuration:** `configs/rag_week3.yaml` takes the dataset, model, generation and execution
  blocks from `configs/evaluation_spider1_expanded.yaml` unchanged. Each experiment is a named
  block, and its hash covers the config, model, generation settings, corpus hash and both prompt
  template hashes.
- **Resuming:** finished ids are never regenerated. A record with a different hash or partition
  is refused.
- **Test-partition guard:** the runner refuses the test partition without `--allow-test`.
- **Rescoring:** `rescore_rag.py` re-executes saved outputs without model calls.
- **Atlas access:** the connection authenticates as the read-only `codegen_eval_readonly` user.
  Only aggregates run, and only after validation; guide examples are fixed and checked in the
  repo.
- **Unchanged artifacts:** nothing in the zero-shot code paths, the 279 or 455 predictions, or
  the v1 split was modified.

---

## 5. Retrieval selection (offline, no model calls; `outputs/rag_week3/retrieval_eval.json`)

**Relevance proxy:** Jaccard overlap between the query's gold-SQL pattern set and each retrieved
example's pattern set, ignoring the generic `select` and `where`. Gold SQL is used only for this
scoring. **Selection rule, fixed in code before results:** best k = 3 mean Jaccard on train
leave-one-database-out (n = 318). Validation (n = 72) is reported as confirmation.

The numbers below come from the corrected index (review finding M1, §9): each train query's
BM25/TF-IDF statistics and hybrid normaliser exclude its own database. The first version
included it, and its train numbers were slightly higher (0.576 for the selected retriever).
The selection and every validation candidate list are unchanged by the fix.

| retriever | train-LODO k=3 mean Jaccard | hit@3 (exact pattern set) | validation k=3 mean Jaccard | validation hit@3 |
|---|---:|---:|---:|---:|
| random | 0.119 | 0.094 | 0.148 | 0.111 |
| tfidf_raw | 0.388 | 0.437 | 0.441 | 0.542 |
| tfidf_masked | 0.367 | 0.487 | 0.453 | 0.625 |
| bm25_raw | 0.390 | 0.434 | 0.456 | 0.542 |
| bm25_masked | 0.389 | 0.472 | 0.460 | 0.653 |
| hybrid_masked w=0.25 | 0.456 | 0.541 | 0.534 | 0.653 |
| hybrid_masked w=0.5 | 0.507 | 0.582 | 0.590 | 0.653 |
| **hybrid_masked w=1.0 (selected)** | **0.561** | 0.569 | **0.606** | 0.611 |

- BM25 and TF-IDF are equivalent here.
- Masking raises exact-pattern hits but not mean Jaccard.
- The pattern-cue hybrid is clearly better than either.
- The w = 0.5 setting has higher hit@3. The pre-declared criterion was kept rather than switched
  after seeing the results.
- Retrieval takes about 4 ms per query.

**Cue quality on train**, inference-time question heuristics vs gold-SQL labels, as precision /
recall:

| cue | precision / recall |
|---|---|
| count | 0.92 / 0.74 |
| order_by | 0.69 / 0.89 |
| anti_join | 0.66 / 1.00 |
| group_by | 0.72 / 0.61 |
| join | 0.68 / 0.58 |
| nested | 1.00 / 0.20 |
| union | 0.19 / 1.00 |

---

## 6. Validation pilot (72 items, Qwen2.5-Coder-3B q4_k_m, CPU)

Experiments are listed in the order run; every one is reported. `outputs/rag_week3/report.md`
has all tables.

| run | context | correct | accuracy [Wilson 95%] |
|---|---|---:|---|
| stored zero-shot (generated 10/10 and 10/20) | none | 32 | 44.4 [33.5, 55.9] |
| `zs_rerun` (k = 0, this session) — **baseline** | none | 33 | 45.8 [34.8, 57.3] |
| `random_k3` (control) | 3 random train examples | 37 | 51.4 [40.1, 62.6] |
| **`hybrid_k3` (primary)** | 3 retrieved examples | **43** | **59.7 [48.2, 70.3]** |
| `hybrid_k3_g3` | 3 examples + 3 guide cards | 41 | 56.9 [45.4, 67.7] |
| `hybrid_k3_fk` (validation-informed, exploratory) | 3 examples + FK type notes | 43 | 59.7 [48.2, 70.3] |

**Determinism.** `zs_rerun` reproduced 69/72 stored raw outputs byte for byte (70/72 verdicts).
The 3 differing items reflect CPU nondeterminism in llama.cpp across sessions:
- 0498 went from wrong to correct;
- 0872 changed from one error category to another;
- 1033 stayed wrong.

The same-session rerun is therefore the primary baseline. All saved validation outputs
(4 × 72) were re-executed on Atlas with `rescore_rag.py`: 288/288 verdicts reproduced.

**Paired comparisons** (same 72 items; McNemar exact two-sided on discordant pairs; paired
bootstrap with 10,000 resamples):

| contrast | both | only baseline | only RAG | neither | Δ points | bootstrap 95% CI | p | Holm p |
|---|---:|---:|---:|---:|---:|---|---:|---:|
| random_k3 vs zs_rerun | 28 | 5 | 9 | 30 | +5.6 | [−4.2, 15.3] | 0.42 | 0.42 |
| **hybrid_k3 vs zs_rerun** | 27 | 6 | 16 | 23 | **+13.9** | **[1.4, 26.4]** | 0.053 | 0.16 |
| hybrid_k3_g3 vs zs_rerun | 26 | 7 | 15 | 24 | +11.1 | [−1.4, 23.6] | 0.13 | 0.27 |
| hybrid_k3 vs random_k3 | 32 | 5 | 11 | 24 | +8.3 | [−2.8, 19.4] | 0.21 | — |
| hybrid_k3 vs stored zero-shot | 26 | 6 | 17 | 23 | +15.3 | [2.8, 27.8] | 0.035 | — |

**Outcome categories** (sum = 72). No run produced a malformed output, a validation rejection
or a timeout.

| run | correct | execution error | wrong row count | wrong column count | wrong values |
|---|---:|---:|---:|---:|---:|
| zs_rerun | 33 | 5 | 26 | 6 | 2 |
| random_k3 | 37 | 2 | 23 | 4 | 6 |
| hybrid_k3 | 43 | 0 | 19 | 6 | 4 |
| hybrid_k3_g3 | 41 | 0 | 21 | 8 | 2 |

The predicted new failure mode, copying collection names from another database's examples,
**never occurred** (0 in every run).

**By hardness** (accuracy %, n):

| run | easy (15) | medium (34) | hard (15) | extra (8) |
|---|---:|---:|---:|---:|
| zs_rerun | 66.7 | 50.0 | 26.7 | 25.0 |
| random_k3 | 66.7 | 67.6 | 20.0 | 12.5 |
| hybrid_k3 | 93.3 | 67.6 | 20.0 | 37.5 |
| hybrid_k3_g3 | 86.7 | 64.7 | 26.7 | 25.0 |

**By gold-SQL pattern** (accuracy %, n; an item can have several patterns; slices are small):

| run | count (25) | group_by (16) | join (30) | anti_join (8) | order_by (10) | having (6) |
|---|---:|---:|---:|---:|---:|---:|
| zs_rerun | 40.0 | 25.0 | 16.7 | 50.0 | 50.0 | 0.0 |
| random_k3 | 56.0 | 37.5 | 26.7 | 37.5 | 60.0 | 33.3 |
| hybrid_k3 | 68.0 | 37.5 | 26.7 | 75.0 | 50.0 | 33.3 |

The gains are concentrated in easy and medium questions and in count and anti-join patterns.
Hard questions do not improve.

**Latency and context** (mean; 16-thread CPU):

| run | retrieval | generation | total | context tokens | input tokens |
|---|---:|---:|---:|---:|---:|
| zs_rerun | 0 ms | 7.9 s | 8.0 s | 0 | 775 |
| random_k3 | 3.7 ms | 14.6 s | 14.6 s | 356 | 1132 |
| hybrid_k3 | 4.5 ms | 14.7 s | 14.8 s | 344 | 1120 |
| hybrid_k3_g3 | 4.4 ms | 22.9 s | 22.9 s | 827 | 1604 |

Retrieval cost is negligible. The cost of RAG is prompt processing: about 1.9× zero-shot
latency for k = 3 and 2.9× with guides. No run had an empty context or a budget drop.

---

## 7. Earlier CPU train run (pre-registered, superseded as evidence by §7b)

**Status.** The primary train run, `hybrid_k3 --partition train`, is complete: 318/318 items.
Claude Code stopped it once at 191 items because the machine was low on memory. It resumed on
2026-10-23 without regenerating any saved item; the experiment, dataset and code hashes all
matched.

- **`random_k3` on CPU:** stopped at 143/318 when the memory reaper killed it. By decision it was
  not resumed. The partial records stay in `outputs/rag_week3/random_k3__train/` for reference
  only and are not used in any comparison.
- **`hybrid_k3_fk` and `zs_rerun` on CPU:** not run.
- **Controls:** the train controls were instead run fresh on a GPU (§7b).
- **Baseline:** this section keeps the pre-registered fallback, the stored zero-shot run. Full
  tables: `outputs/rag_week3/report_train.{json,md}`.

| run | correct | accuracy [Wilson 95%] |
|---|---:|---|
| stored zero-shot 3B (Week 2) | 148/318 | 46.5 [41.1, 52.0] |
| `hybrid_k3` | 171/318 | 53.8 [48.3, 59.2] |

**Paired comparison vs stored zero-shot:**
- 40 items correct only with RAG, 17 only with zero-shot;
- +7.2 points, bootstrap 95% CI [+2.5, +11.9];
- exact McNemar p = 0.0032.

**Checks on the train effect:**
- **Without `concert_singer` and `singer`** (near-twin databases): +7.8 points [+2.8, +12.8], n = 281,
  p = 0.0032. The effect does not come from those two databases.
- **Near duplicates:** the mean raw-question Jaccard between a query and its top-1 retrieved
  example is 0.257 on train vs 0.272 on validation. 5.3% of train queries have a top-1 match ≥ 0.5,
  vs 6.9% on validation. Train retrieval is not closer to the query than validation retrieval.
- **By hardness**, RAG vs zero-shot: easy 75.0 vs 65.5, medium 63.6 vs 59.3, hard 26.7 vs 13.3,
  extra 14.3 vs 8.2. Unlike validation, hard and extra items improve on train. Validation had only
  15 hard and 8 extra items.
- **By pattern:** count_distinct 46.7 vs 20.0 (n = 15), anti_join 52.4 vs 33.3 (n = 21),
  order_by 43.9 vs 33.3, limit 41.2 vs 29.4. Join-heavy items barely move: join 18.6 vs 15.9
  (n = 113), nested 0 vs 0.
- **Categories:** wrong column count drops from 48 to 30, execution errors from 14 to 10. Wrong
  row count is unchanged (83 vs 82).
- **Latency:** mean generation 16.7 s per item, with 339 context tokens on average.

**Interpretation.** The train result points the same way as validation, but with a smaller
effect: +7.2 points vs +13.9. Given the overlapping CIs, the two are consistent. Three things
make it weaker evidence than a same-session baseline would be:
1. The baseline is the stored Week 2 run, not a same-session rerun. On validation, these two
   differed by 1 item, the noise floor described in M2.
2. The random-example control has not been run on train. The train result therefore cannot
   separate *retrieved* examples from *any* examples.
3. Train is in-sample for the retriever weight (§10).

**Re-execution caveat.** On re-execution, 316/318 verdicts reproduce. The two that flip do so in
opposite directions, so the total stays 171:
- **0667:** the model sorts on `Height` after projecting it away, so every row ties.
- **0934:** the model groups `Breeds` by name, so every count is 1.

In both, `$limit: 1` picks an arbitrary row from a tie, and Atlas does not guarantee an order for
ties. Both queries are wrong in substance. Treat ±1 item as execution noise for queries like these.

**Superseded.** The two weaknesses listed above, the stored baseline and the missing random
control, are addressed by the clean control in §7b. It does not reproduce the +7.2-point gain
over zero-shot.

---

## 7b. Clean Kaggle control (train, 318 items, fresh generation)

**Design.** All three arms were generated fresh in one Kaggle run on 2026-10-23:
- `zs_rerun` (k = 0), `random_k3` (seeded random train examples), `hybrid_k3` (the selected
  retriever);
- Qwen2.5-Coder-3B-Instruct q4_k_m GGUF (same file hash as the CPU runs) on llama-cpp-python
  0.3.36, all layers on one Tesla T4;
- temperature 0, seed 42, and the **llama.cpp KV cache reset before every item**, which removes
  the run-order dependence found in M2.

The prompts were built and frozen on Windows before upload, in package
`kaggle/week3_control_v1`. The analysis plan was frozen in the package manifest before any GPU
output existed. Primary contrast: `hybrid_k3` vs `random_k3`. Confirmatory family: the three
pairwise contrasts, each tested with exact McNemar, Holm-adjusted, α = 0.05.

The outputs were scored locally with the unchanged scorer (`run_eval.score_output`, read-only
Atlas user) and written to `outputs/rag_week3_clean/week3_control_v1/`. Nothing went to
`outputs/rag_week3/`.

**Import verification** (`verification.json`, passed):
- SHA256SUMS matched (13 files);
- 318 unique ids per arm, with no provider errors;
- every frozen prompt was rebuilt identically (318/318 per arm);
- the Kaggle determinism check regenerated 5 items per arm, and all were byte-identical.

| arm | correct | accuracy [Wilson 95%] | wrong rows | wrong columns | wrong values | exec error | malformed | timeout |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| `zs_rerun` | 160/318 | 50.3 [44.8, 55.8] | 75 | 48 | 21 (+1 order) | 11 | 2 | 0 |
| `random_k3` | 155/318 | 48.7 [43.3, 54.2] | 93 | 40 | 17 | 12 | 0 | 1 |
| `hybrid_k3` | 172/318 | 54.1 [48.6, 59.5] | 86 | 30 | 22 | 6 | 1 | 1 |

**Paired contrasts** (pre-specified; exact McNemar on discordant pairs; paired bootstrap, 10k
resamples, seed 20261020):

| contrast | both | only baseline | only treatment | Δ points | bootstrap 95% CI | McNemar p | Holm p |
|---|---:|---:|---:|---:|---|---:|---:|
| **hybrid_k3 vs random_k3 (primary)** | 139 | 16 | 33 | **+5.3** | [+1.3, +9.7] | 0.0213 | **0.0639** |
| hybrid_k3 vs zs_rerun | 135 | 25 | 37 | +3.8 | [−1.3, +8.8] | 0.1619 | 0.3238 |
| random_k3 vs zs_rerun | 133 | 27 | 22 | −1.6 | [−6.0, +2.8] | 0.5682 | 0.5682 |

**Reading.**
- **Hybrid vs random:** relevant examples beat random examples by 5.3 points. The unadjusted
  CI excludes 0, but after Holm correction over the frozen family the primary test **does not
  reject** at 0.05.
- **Hybrid vs zero-shot:** not significant.
- **Random vs zero-shot:** random examples do not help; the point estimate is negative.
- **What this suggests:** relevant examples may help where random few-shot examples do not.
  Whether RAG beats zero-shot at all is not established on this partition.
- **Why the CPU result differs:** the CPU train result in §7 (+7.2 vs stored zero-shot) does not
  replicate against a fresh baseline. The fresh zero-shot (160) is 12 items above the stored Week
  2 run (148), while hybrid is essentially unchanged (171 CPU vs 172 GPU).

By hardness (correct / n): easy 60 / 58 / 64 of 84; medium 89 / 88 / 89 of 140; hard 7 / 8 / 11
of 45; extra 4 / 1 / 8 of 49 (order: zero-shot / random / hybrid). Mean generation time on the
GPU was 2.4 / 2.8 / 2.8 s. This is not comparable with the CPU latencies.

**Agreement with earlier CPU outputs** (exploratory, listed in the frozen plan):
- `hybrid_k3`: 245/318 raw outputs are byte-identical to the CPU run, and verdicts agree on 309/318
  (4 correct only on CPU, 5 only on GPU).
- The 143 partial CPU `random_k3` records agree with the GPU verdicts on 133/143.

**Not blind.** Train is a development partition: the hybrid weight was selected on train-LODO
(§5). The CPU hybrid-vs-zero-shot result on train was known in advance, and 143 CPU `random_k3`
outputs had been seen. The comparison is pre-specified but not blind. **The 65-item test
partition was not used.** No test-split id appears in any Week 3 output file, CPU or Kaggle.

**Re-execution and ordering stability.** `score_kaggle_outputs.py` re-executed every saved verdict
on Atlas:
- `zs_rerun` 318/318 and `random_k3` 318/318 reproduced;
- `hybrid_k3` 317/318 reproduced (`rescore_check.json`).

**The flip: 0934.** The question is "What is the name of the breed with the most dogs?" The gold
answer is Bulldog (7 dogs vs 4 and 4), so the expected answer is unique and the item is valid.
- **What the query does:** it groups `Breeds` by name and never joins `Dogs`. Every breed
  therefore has count 1. **It does not count dogs by breed.**
- **Why the result varies:** `$sort` + `$limit: 1` then selects arbitrarily among the three
  tied breeds. Ten read-only re-executions returned Bulldog 5 times and Eskimo 5 times.
- **What was ruled out:** data changes (Atlas matches SQLite) and the comparator.
- **Verdict:** the saved `correct` was a coin flip that landed on the gold value.

Details: `outputs/rag_week3_clean/week3_control_v1/investigation_0934/`.

**Broader audit** (`stability_audit/`):
- **Scope:** all 65 saved *correct* predictions that use `$limit`, across the three arms (16 / 21
  / 28 in the order zero-shot / random / hybrid), plus `$first`/`$last` usage.
- **Method:** each was statically classified, and the sort key was checked for uniqueness at
  the cutoff. Suspicious items were re-executed 10 times, with tie-breaker or input-order
  variants.
- **Unique cutoff:** 61 of the 65 have a unique sort key at the cutoff.
- **Ties that cannot change the answer:** 0988 (zero-shot and random) ties at the cutoff, but
  every tied row has the same output value. 0093 (`$first`) is stable 10/10.
- **0934:** the one instability observed on re-execution.
- **0667 (hybrid), a latent instability:**
  - *What the query does:* it sorts on `Height` after projecting it away, since the field lives
    under `person.Height`. It is therefore sorting on a field missing from all 5 documents.
  - *What was observed:* 10/10 re-executions return the correct 68. Reversing the input order
    returns 154, which is wrong. The result is whichever document arrives first.

No other unstable correct predictions were found.

| variant (hybrid only; other arms unaffected) | hybrid_k3 | hybrid vs random: Δ / p / Holm | hybrid vs zero-shot: Δ / p / Holm |
|---|---:|---|---|
| saved verdicts (**primary, unchanged**) | 172 | +5.3 / 0.021 / 0.064 | +3.8 / 0.16 / 0.32 |
| 0934 counted wrong | 171 | +5.0 / 0.029 / 0.088 | +3.5 / 0.20 / 0.40 |
| 0934 and 0667 counted wrong | 170 | +4.7 / 0.040 / 0.12 | +3.1 / 0.25 / 0.49 |

- **Primary metric:** the saved verdicts are the primary execution-accuracy result. The frozen
  scorer assigns one verdict per execution, and nothing was relabelled.
- **Direction:** instability is reported separately here. It moves only against hybrid, and no
  conclusion changes.
- **Cited examples:** neither item is among the "helped" examples cited in `report.md`.
- **Audit limit:** wrong predictions with a tied or missing sort key were checked only with an
  `_id` tie-breaker (`tie_scan/`), so a latent baseline win of the 0667 kind cannot be ruled out.

**Report verification.** `stability_audit/verify_report.py` recomputes the counts, Wilson CIs,
discordant pairs, exact McNemar and Holm from the raw `executions.jsonl` files without the report
code. All match `report.json`. An independent bootstrap, with a different seed, agrees to within
0.4 points.

---

## 8. Failure analysis

Categories for `hybrid_k3` on validation (n = 72):

- 43 correct;
- 19 wrong row count, 6 wrong column count, 4 wrong values;
- 0 format errors, 0 invalid MQL, 0 execution errors, 0 timeouts.

No retrieved example's collection name was copied into an output (cross-database copies: 0).

**Where retrieval helped** (16 items wrong with zero-shot, correct with `hybrid_k3`; 6 the reverse):
- *Output contract.* Examples show `$project` with `_id: 0` and exactly the requested columns.
  Some gained items were fixed this way, but the net wrong-column-count is unchanged (6 for
  `zs_rerun`, 6 for `hybrid_k3`) because other items gained an extra column.
- *Net category shift vs `zs_rerun`:* execution errors 5 → 0, wrong row count 26 → 19, wrong
  values 2 → 4.
- *Field references.* Missing `$` prefixes in `$expr` / `$group` were fixed, mostly in `network_1`.
- *Patterns:* count distinct, anti-join via `$lookup` + `$size: 0`, and `$or` filters.
- By slice: `count` 68% vs 40%; `anti_join` 75% vs 50%; easy 93% vs 67%; medium 68% vs 50%.

**Where it hurt.** The model sometimes copied structure that did not fit the question:
- over-joining (0413);
- an extra projected column (0414);
- a `let` + `$toString` join pattern applied where types already matched (0405);
- a wrong projected field (0894), plus 0890 and 0504.

On hard items, accuracy was 20% vs 27% for zero-shot (n = 15). The examples did not help with
multi-step queries.

**Remaining 29 `hybrid_k3` failures.**
- 23 involve `$lookup`.
- 12 return zero rows. Two causes dominate:
  - *Join-key type mismatch* in `museum_visit`: `visit.visitor_ID` is a string and `visitor.ID` is
    an integer, so a plain `$lookup` matches nothing.
  - *Multi-hop joins* in `course_teach`: the model filters `course_arrange` on a field that only
    exists in `course`.

**Guide cards** (`hybrid_k3_g3`, 41/72) helped two items (0403, 0904) and hurt four (0415, 0495,
0503, 0888). In 0495 the word "each" falsely triggered the `group_by` cue. Cards also add about
480 context tokens and raise latency by 1.6x. They are not recommended in their current form.

**FK type notes** (`hybrid_k3_fk`, exploratory, designed after reading the failures above). Notes
were inserted for all 17 `museum_visit` items, the only validation database with an observed FK
type mismatch. 65 of the 72 raw outputs are identical to `hybrid_k3`; the item-level verdicts are
identical on all 72 items (43/72), and 9 of the 17 noted items are still wrong. Telling the 3B
model about the mismatch rarely made it emit a conversion. Only 1 of the 17 `fk` outputs uses
`$toString` / `$toInt` / `$convert` (0427, still zero rows), against 0 of 17 for `hybrid_k3`. The
notes were present in every `museum_visit` context and never dropped by the budget. Because this
variant was designed on validation, even a positive result would have been optimistic. The null
result stands.

All 7 raw outputs that differ from `hybrid_k3` are `museum_visit` items, whose prompts differ.
The other 55 items have byte-identical prompts and gave byte-identical outputs. Run-to-run noise
of the kind seen between sessions (M2, §10) did not show up within this pair of runs, which used
the same item order.

---

## 9. Testing and independent review

**Automated tests.** `python -m pytest -q`: **173 passed** (2026-10-24; the whole repository,
including `tests/test_rag_week3.py` and `tests/test_kaggle_week3.py`; 159 on 2026-10-21, before
the Kaggle package tests). The RAG tests cover:

- split reproduction and v1 partition inheritance;
- corpus provenance (each example's `source_record_sha256` matches the exact saved line) and
  byte-deterministic rebuild;
- leakage, including injected leaks: held-out question (also case and punctuation variants), SQL,
  collection, database, source partition, duplicate id, and a guide example starting from a
  held-out database;
- retrieval for every method: target-db exclusion, deduplication, determinism, empty corpus, and
  leave-one-database-out index and normaliser;
- prompt and budget: k = 0 byte-identical to zero-shot, block placement, budget drop, n_ctx
  overflow, missing marker, FK notes and their budget;
- config and experiment hashing, test-partition refusal, and resume refusal when a saved record's
  experiment hash, partition, dataset hash or code hash differs (before any model call or Atlas
  connection);
- scoring: malformed or forbidden outputs never reach Atlas (fake DB), execution errors,
  deterministic rescoring;
- report statistics: Wilson, paired McNemar and bootstrap, Holm;
- saved records: complete, and their contexts re-derive exactly under the current code.

**Re-execution.** `rescore_rag.py` re-ran every saved output on Atlas (read-only user) with no
model calls. The five validation runs each reproduced 72/72. The train `hybrid_k3` run reproduced
316/318: two items flip in opposite directions because of arbitrary tie order under `$limit`
(§7). See `outputs/rag_week3/rescore_check.json`. The clean Kaggle control reproduced 318/318,
318/318 and 317/318. The one flip is 0934, and the ordering audit is in §7b. The corpus
`--check` rebuild is byte-identical (sha256 `98dc8273…`).

**Advisor.** An advisor model was consulted before the approach was fixed and during the work. It
pushed toward the simplest defensible design: example retrieval with a random control, offline
retriever selection, and a validation-only pilot. Its advice shaped the protocol; it is not
claimed as approval of the results.

**Independent review.** A separate reviewer agent read the code, configs and outputs. It reported
no critical issues. Its findings and what was done about each:

| id | finding | action |
|---|---|---|
| M1 | BM25/TF-IDF statistics and the hybrid normaliser included the target database (train LODO not strict) | Fixed with a per-target index. The partial train run was archived to `outputs/rag_week3/history/superseded_non_lodo_index/`. Retrieval eval rerun; validation selections unchanged 72/72. |
| M2 | llama.cpp reuses the KV prefix across calls, so outputs depend on item order | Not changed, to keep zero-shot behaviour comparable. Documented as a noise floor (§10); Week 4 item. |
| M3 | Headline leaned on the stored-baseline p-value | Headline now uses exact McNemar vs `zs_rerun` with Holm, and is called suggestive. |
| minor | Guide-card pipelines were not validated like model output | They now go through the same read-only validator; all 18 pass. |
| minor | Leakage checker scanned only some fields, with exact-string matching | It scans every field the retriever or prompt sees, with normalised matching. |
| minor | Hard-coded execution config; report KeyError on a missing hardness level; misleading config comment; FK variant inside the Holm family | All fixed. The FK variant is excluded as exploratory. |
| minor | Guide ranking uses the top-k examples before the budget check | Harmless here (no budget drops in any run). Documented. |

---

## 10. Limitations

- **Power.** n = 72 from five databases. The primary effect is +13.9 points, with bootstrap 95% CI
  [1.4, 26.4] and exact McNemar p = 0.053 (Holm 0.16). It is suggestive, not significant. The
  difference between hybrid and random retrieval (+8.3, p = 0.21) is not established.
- **Noise floor.** Because of the KV prefix reuse (M2), the same prompt can give a different output
  depending on the previous call. `zs_rerun` matched the stored zero-shot on 69/72 raw outputs and
  70/72 verdicts. Differences of 1–3 items between runs are within noise.
- **Distillation.** 265 of the 280 corpus examples are verified outputs of gpt-oss-120b; 15 are
  from Qwen 7B. Part of the gain may be transfer from a stronger model's style rather than
  retrieval as such. The random control (+5.6) gives a rough size for the generic part.
- **Guide cards were written after zero-shot results existed.** They were written from MongoDB
  documentation, knowing the common zero-shot failure types. They did not improve results.
- **The FK variant is validation-informed.** It was designed after inspecting validation failures
  and is labelled exploratory.
- **Retrospective split.** The database-disjoint split was created after the Week 2 zero-shot
  predictions. The stored baselines were not tuned on it, but they predate it.
- **Train evidence is in-sample.** The retriever weight was selected on train-LODO, and train
  queries share near-duplicate questions across databases (`concert_singer`/`singer`; excluding
  them leaves the effect unchanged). The CPU train comparison (§7) uses the stored Week 2
  baseline. The clean control (§7b) has same-run baselines but is not blind, and neither of its
  RAG contrasts survives Holm correction.
- **Ordering instability.** Two saved hybrid `correct` verdicts on train depend on arbitrary tie
  or document order (0934, 0667). They are kept as primary and reported as a sensitivity (§7b).
- **Contamination.** Spider is public and likely in the pre-training data of every model used.
  Absolute accuracies are not comparable to unseen data; paired differences are less affected.
- **Scope.** One model (3B q4; CPU for §6–7, GPU for §7b), one prompt family, k = 3 only.
  Nothing was evaluated on the test partition.

---

## 11. Recommendations for Week 4 (fine-tuning)

1. **Fine-tune on the train partition only.** Use the 280 execution-verified NL→MQL pairs (train
   databases, provenance in the corpus), with validation for model selection and test held out
   until the end. Report gpt-oss-sourced and Qwen-sourced pairs separately (distillation, §10).
2. **Make the output contract part of training.** Most of the RAG gain is format and projection
   (`_id: 0`, exact columns, `$` prefixes). Fine-tuning can learn this directly, without spending
   about 350 prompt tokens per query.
3. **Teach join-key types explicitly.** Telling the 3B model about mismatched FK types in the
   prompt did not help (§8). Include training pairs that use `$toString` / `$toInt` inside
   `$lookup` pipelines, and put type information in the schema rendering.
4. **Execution-guided retry.** 12/29 failures return zero rows. One retry with the empty-result
   signal is cheap and uses no gold information. Report it separately from single-shot accuracy.
5. **Keep the noise floor fixed.** The Kaggle control (§7b) already resets the llama.cpp KV cache
   per item (`kv_reset_per_item: true`), and its fresh zero-shot (160/318 on train) is the
   baseline to beat. Use the same backend settings for Week 4. Keep a same-run zero-shot arm, and
   keep the item order fixed.
6. **Compare RAG + fine-tuning against fine-tuning alone** with the same paired protocol. Drop the
   guide cards. Evaluate Qwen 7B (run alone, never concurrently) as a stronger base.
7. **Run the test partition once,** at the end of Week 4, for the final selected configurations.

---

## 12. Reproduction

From `codegen-capstone/`, with `MONGODB_URI` for the read-only Atlas user set in `.env` (never
printed):

```bash
python scripts/spider1/make_splits_expanded.py          # verify the 318/72/65 database-disjoint split
python scripts/rag/build_corpus.py --check              # rebuild corpus_v1 and check it byte-for-byte
python scripts/rag/retrieval_eval.py                    # offline retriever selection (no model calls)
for e in zs_rerun random_k3 hybrid_k3 hybrid_k3_g3 hybrid_k3_fk; do
  python scripts/rag/run_rag_eval.py --experiment $e    # validation; resumes, skips finished items
done
for e in hybrid_k3 random_k3 hybrid_k3_fk zs_rerun; do   # pre-registered train runs (§7)
  python scripts/rag/run_rag_eval.py --experiment $e --partition train
done
python scripts/rag/rescore_rag.py                       # re-execute saved outputs, no model calls
python scripts/rag/rag_report.py                        # outputs/rag_week3/report.{json,md}
python scripts/rag/rag_report.py --partition train      # outputs/rag_week3/report_train.{json,md}
# clean Kaggle control (§7b): build package, run on Kaggle (scripts/kaggle_week3/KAGGLE_SETUP.md), then
python scripts/kaggle_week3/score_kaggle_outputs.py --zip <week3_control_outputs.zip>  # verify, score, rescore
python scripts/kaggle_week3/clean_report.py             # outputs/rag_week3_clean/week3_control_v1/report.{json,md}
D=outputs/rag_week3_clean/week3_control_v1/stability_audit
python $D/stability_audit.py > $D/stability_audit.json  # ordering audit (read-only, no model calls)
python $D/probe_0667_0093.py > $D/probe_0667_0093.json
python $D/verify_report.py > $D/verify_report.json      # independent check + sensitivities
python -m pytest -q
```

Run one model process at a time. The test partition is refused unless `--allow-test` is given; do
not pass it before Week 4's final evaluation. Because of M2, a regenerated run may differ from the
saved outputs on a few items. The saved `executions.jsonl` files are the record for this report.
