# Semantic quality audit of `spider_pool_v2/train.jsonl`

Audit date: 2026-11-06. SQL-side dataset audit (SQL-only track); no model metric. **Audit only: no dataset, converter, verifier-3, pilot output or MongoDB was modified, nothing was committed, and no conversion, RAG or fine-tuning was started.** Evidence: `outputs/spider_pool_semantic_audit_v1/` (start with `audit_summary.json`; per-record results in `record_audit.jsonl`).

**Reviewer.** All judgments here are automated detectors plus an AI second pass and an AI hand-read of every CONFIRMED and LIKELY item (Claude Sonnet 5.5). It is **not** a human review and does not satisfy pilot gate P2.

## 1. Result

Every one of the 5,387 unique training records has exactly one status.

| Status | Records | Share of 5,387 |
| --- | --- | --- |
| CONFIRMED_DEFECT | 30 | 0.56% |
| LIKELY_DEFECT | 58 | 1.08% |
| NEEDS_REVIEW | 254 | 4.72% |
| NO_ISSUE_DETECTED | 5,045 | 93.65% |
| BLOCKED | 0 | 0.0% |
| **Total** | **5,387** | 100% |

- These are **detector outputs, not an error-rate estimate.** NO_ISSUE_DETECTED means the implemented checks found nothing; it is not proof of correctness.
- **The detectors miss a lot** (section 6): on 127 known-defective excluded or quarantined rows they reached LIKELY or above on only 31 (24.4%), and the earlier 120-item pilot review found 5.0% wrong labels against 1.63% flagged CONFIRMED or LIKELY here. Treat the counts as a lower bound.

## 2. Final decision

1. **Confirmed source-label defects: 30 (0.56%).** 16 wrong superlative/ORDER BY direction, 6 comparison-operator boundary errors, 6 filter values not supported by (or contradicting) the question, 2 explicit ascending/descending contradictions.
2. **Likely defects: 58 (1.08%).** 39 records with numeric values stored as text and used numerically (answers change under numeric semantics), 15 with COUNT(\*) counting rows of a history/join table instead of the entities the question names, 2 missing DISTINCT, 2 non-ISO dates ordered as text.
3. **Need further review: 254 (4.72%).** By finding: 77 latent text-numeric storage, 68 question values the SQL never uses, 58 INNER JOINs that drop parents under each/every/all wording, 43 abbreviation-style or unsupported literals, 12 paraphrase siblings of a flagged record, plus smaller groups (see the detector table).
4. **Concentrated databases (CONFIRMED + LIKELY):** `wrestler` (8 of 40), `cre_Theme_park` (8 of 76), `shop_membership` (6 of 38), `wine_1` (5 of 78), `hospital_1` (5 of 76), `bike_1` (4 of 78), `customers_and_addresses` (4 of 83), `formula_1` (4 of 52). The top concentrations are single-column effects, so the remedy is a column policy, not per-record exclusion: `cre_Theme_park` mostly `Hotels.price_range` (8); `wrestler` mostly `wrestler.Days_held` (8); `shop_membership` mostly `branch.membership_amount` (6), `operator_mismatch` (1). Full table in section 4. 43 of 123 databases have no flag at all.
5. **Most common defect types (records):** data_type:numeric_text 115, nl_alignment:missing_filter 68, join:inner_vs_left 58, nl_alignment:unsupported_literal 51, ordering:direction 24, aggregation:count_vs_distinct 22.
6. **Can the retained dataset proceed to V2 SQL-to-MQL conversion?** **Not as an approved corpus. Conditionally yes for pipeline work.**
   - Apply the exclusions in section 9 first: remove the 30 confirmed, hold the 58 likely, hold the 254 needing review. That leaves 5,045 "no issue detected" records, which are an *unverified* pool, not a clean one.
   - Because recall is low, defects remain in that pool. The pilot review's 5.0% (6/120) is the better guide to residual risk than this audit's 1.6%.
   - The converter itself was clean in the pilot (0/120 non-equivalent), so conversion is not the risk. The labels are.
7. **Additional checks before approving a final corpus:** (a) a named human reviewer (pilot gate P2) over a fresh random sample of the post-exclusion pool, including the "no issue detected" stratum, to measure residual error; (b) human adjudication of the 312 flagged items; (c) a decision policy for text-typed numbers and non-ISO dates (convert in the MQL fixtures or exclude), because MongoDB will compare them the same lexicographic way; (d) a gold-answer check on 3+ table joins and nested/set-operation queries, which the detectors cover least; (e) rerun this audit on `val_internal.jsonl` before it is used for any model-selection decision.

## 3. Method

Scope lock: only `train.jsonl` rows, only their 123 databases opened read-only (`mode=ro`) from `data/raw/spider1/spider_data/database/<db>/<db>.sqlite`. No dev/test questions or gold SQL, no `tables.json`, no MongoDB. **One qualified exception to "no protected data":** to find out why review item R097 (`spider_train_04071`) is not in `train.jsonl` I opened the zip member `spider_pool/rejected.jsonl` (the v2 manifest lists it as not opened; it also holds dev rows) and matched that single train ID, printing only its id, database and reject reason. I also ran a repository-wide text search for that ID, which listed file names only. No dev/test question or SQL was read, and nothing from those rows was used as audit material. Python 3.13, SQLite 3.50.4, sqlglot 25.34.1 (isolated copy in `deps/`). Schemas come from `PRAGMA`, not from metadata files.

1. **Value profile** (`s1_profile.py`, `profiles/<db>.json`): per column, `typeof()` distribution of the stored values, numeric-looking text, date patterns (ISO, DD.MM.YYYY, M/D/YYYY, d Mon yyyy, ...), blank strings, NULLs, leading/trailing space, case variants, distinct count and actual uniqueness. Declared types are never trusted.
2. **AST analysis** (`engine.py`): each SQL is parsed (sqlite dialect), aliases and double-quoted string literals are resolved against the real schema, and joins, literals, operators, ORDER BY/LIMIT, aggregates, GROUP BY/HAVING and DISTINCT are extracted.
3. **Differential execution:** where a detector sees a risk it builds a corrected variant (numeric cast, ISO date conversion, COUNT DISTINCT, LEFT JOIN, group by key, ...) and runs both. A defect is only graded LIKELY or higher when the answers actually differ; when they coincide it is recorded as *latent* NEEDS_REVIEW.
4. **Question/SQL reasoning:** lexical rules, specified before the first full run and then tightened after reading flagged records (not "pre-registered" in a strict sense), for operators (at least / more than / after ...), explicit order words, superlatives (resolved against column kind: date, age, rank-like, numeric), literal support, truncated questions, quoted values missing from the SQL, antonym attributes.
5. **Adversarial second pass** (`s4_adversarial.py`): separate stdlib-only code (no sqlglot, no engine import) re-checks every CONFIRMED and LIKELY finding; 89 of 89 reproduced. **Limit of its independence:** for the numeric-text, non-ISO-date and entity-count findings (about two thirds of them) it re-executes the engine's own corrected `variant_sql`, so it confirms that the answers differ and that the stored values are text, not that the engine's correction is the right one. For direction, operator, explicit-order and literal findings it uses its own regex logic. Then I read every CONFIRMED and LIKELY record by hand; the false-positive mechanisms found that way ("at least" read as a minimum cue, `longest` on date columns, "phone number of ..." read as a count request, "count of each X" where X is the group key, comma-style joins, a sentence ending in "have") were fixed as general rules, with no record-specific logic.
6. **Sibling spread:** records sharing a `group` (same SQL, paraphrased question) inherit a REVIEW flag when a sibling is LIKELY or higher (12 added).

Grades: CONFIRMED = specific wrong mapping shown by data or an explicit wording rule; LIKELY = answer-changing evidence but a human could accept the SQL; REVIEW = ambiguous, latent or weak; INFO = recorded observation that never changes a status (e.g. GROUP BY on a non-unique name, 722 records, and join-key storage-type differences, 142). Record status = highest grade.

## 4. Results

### By database (databases with at least one flag; top 25 by CONFIRMED+LIKELY)

| Database | Records | Conf. | Likely | Review | Conf.+Likely | Any flag |
| --- | --- | --- | --- | --- | --- | --- |
| wrestler | 40 | 0 | 8 | 4 | 20.0% | 30.0% |
| cre_Theme_park | 76 | 0 | 8 | 1 | 10.5% | 11.8% |
| shop_membership | 38 | 1 | 5 | 10 | 15.8% | 42.1% |
| wine_1 | 78 | 4 | 1 | 4 | 6.4% | 11.5% |
| hospital_1 | 76 | 4 | 1 | 3 | 6.6% | 10.5% |
| bike_1 | 78 | 0 | 4 | 18 | 5.1% | 28.2% |
| customers_and_addresses | 83 | 2 | 2 | 6 | 4.8% | 12.0% |
| formula_1 | 52 | 0 | 4 | 2 | 7.7% | 11.5% |
| flight_1 | 89 | 3 | 0 | 4 | 3.4% | 7.9% |
| cre_Drama_Workshop_Groups | 77 | 0 | 3 | 3 | 3.9% | 7.8% |
| baseball_1 | 42 | 0 | 3 | 1 | 7.1% | 9.5% |
| department_store | 79 | 0 | 2 | 6 | 2.5% | 10.1% |
| apartment_rentals | 74 | 0 | 2 | 4 | 2.7% | 8.1% |
| medicine_enzyme_interaction | 44 | 2 | 0 | 3 | 4.5% | 11.4% |
| csu_1 | 55 | 2 | 0 | 2 | 3.6% | 7.3% |
| products_gen_characteristics | 75 | 2 | 0 | 2 | 2.7% | 5.3% |
| candidate_poll | 34 | 0 | 2 | 1 | 5.9% | 8.8% |
| document_management | 36 | 2 | 0 | 1 | 5.6% | 8.3% |
| company_office | 40 | 0 | 2 | 0 | 5.0% | 5.0% |
| cre_Doc_Tracking_DB | 88 | 0 | 2 | 0 | 2.3% | 2.3% |
| customer_complaints | 26 | 2 | 0 | 0 | 7.7% | 7.7% |
| e_government | 34 | 0 | 2 | 0 | 5.9% | 5.9% |
| entrepreneur | 40 | 2 | 0 | 0 | 5.0% | 5.0% |
| theme_gallery | 38 | 0 | 2 | 0 | 5.3% | 5.3% |
| university_basketball | 46 | 1 | 0 | 6 | 2.2% | 15.2% |

All 123 databases are in `audit_summary.json` (`by_database`).

### By SQL difficulty

| Hardness | CONFIRMED | LIKELY | REVIEW | NO_ISSUE | BLOCKED | total |
| --- | --- | --- | --- | --- | --- | --- |
| easy | 3 | 10 | 30 | 1349 | 0 | 1392 |
| extra | 11 | 4 | 33 | 649 | 0 | 697 |
| hard | 4 | 15 | 45 | 1014 | 0 | 1078 |
| medium | 12 | 29 | 146 | 2033 | 0 | 2220 |

### By query pattern

| Query class | CONFIRMED | LIKELY | REVIEW | NO_ISSUE | BLOCKED | total |
| --- | --- | --- | --- | --- | --- | --- |
| count | 0 | 6 | 20 | 717 | 0 | 743 |
| filter_complex | 1 | 0 | 14 | 90 | 0 | 105 |
| filter_simple | 2 | 4 | 18 | 836 | 0 | 860 |
| group | 0 | 13 | 15 | 514 | 0 | 542 |
| having | 0 | 0 | 8 | 146 | 0 | 154 |
| join | 13 | 9 | 95 | 1173 | 0 | 1290 |
| multi_join | 2 | 8 | 19 | 407 | 0 | 436 |
| nested | 1 | 6 | 6 | 340 | 0 | 353 |
| set_op | 3 | 0 | 23 | 365 | 0 | 391 |
| sort_limit | 8 | 12 | 36 | 457 | 0 | 513 |

Overall CONFIRMED+LIKELY rate is 1.6%. Lowest by query class: having 0.0%, filter_simple 0.7%, set_op 0.8%; highest: group 2.4%, sort_limit 3.9%. A low rate on set operations or nested queries would be a coverage gap rather than evidence of correctness, because most differential variants are built only for the top-level SELECT; the data do not show a clearly low rate for either (set_op 0.8%, nested 2.0%).

### By defect category (records; highest grade within the category)

| Category | Records | Confirmed | Likely | Review |
| --- | --- | --- | --- | --- |
| data_type:numeric_text | 115 | 0 | 40 | 75 |
| nl_alignment:missing_filter | 68 | 0 | 0 | 68 |
| join:inner_vs_left | 58 | 0 | 0 | 58 |
| nl_alignment:unsupported_literal | 51 | 6 | 0 | 45 |
| ordering:direction | 24 | 18 | 0 | 6 |
| aggregation:count_vs_distinct | 22 | 0 | 15 | 7 |
| nl_alignment:comparison_operator | 8 | 6 | 0 | 2 |
| aggregation:function_vs_question | 6 | 0 | 0 | 6 |
| data_type:non_iso_date | 4 | 0 | 2 | 2 |
| aggregation:missing_distinct | 3 | 0 | 2 | 1 |

A record can carry several categories, so rows do not sum to the status totals. INFO observations (never status-changing) are kept per record under `observations_info_only`.

### By detector and grade (findings)

| Detector | CONFIRMED | LIKELY | REVIEW |
| --- | --- | --- | --- |
| agg_wording | 0 | 0 | 6 |
| entity_count | 0 | 15 | 0 |
| explicit_order | 2 | 0 | 0 |
| inner_join_drop | 0 | 0 | 58 |
| literal_not_in_question | 6 | 0 | 43 |
| missing_distinct | 0 | 2 | 0 |
| non_iso_date | 0 | 2 | 2 |
| operator_mismatch | 6 | 0 | 0 |
| question_value_not_in_sql | 0 | 0 | 68 |
| sibling_spread | 0 | 0 | 12 |
| superlative_direction | 16 | 0 | 6 |
| text_numeric | 0 | 40 | 77 |

### The 30 confirmed defects

| Record | Database | Category | Clause |
| --- | --- | --- | --- |
| 00365 | flight_1 | nl_alignment:comparison_operator | salary > 100000 |
| 00431 | flight_1 | ordering:direction | COUNT(*) DESC LIMIT 1 |
| 00432 | flight_1 | ordering:direction | COUNT(*) DESC LIMIT 1 |
| 00969 | medicine_enzyme_interaction | nl_alignment:unsupported_literal | 'Protoporphyrinogen IX' |
| 00970 | medicine_enzyme_interaction | nl_alignment:unsupported_literal | 'Protoporphyrinogen IX' |
| 01006 | university_basketball | nl_alignment:comparison_operator | founded > 1850 |
| 01908 | school_finance | nl_alignment:comparison_operator | amount > 8.5 |
| 02293 | entrepreneur | ordering:direction | T1.Money_Requested |
| 02294 | entrepreneur | ordering:direction | T1.Money_Requested |
| 02358 | csu_1 | nl_alignment:comparison_operator | T2.year >= 2000 |
| 02359 | csu_1 | nl_alignment:comparison_operator | T2.year >= 2000 |
| 02798 | election | nl_alignment:unsupported_literal | 'Economic Matters' |
| 03948 | hospital_1 | ordering:direction | dateundergoes LIMIT 1 |
| 03949 | hospital_1 | ordering:direction | dateundergoes LIMIT 1 |
| 03970 | hospital_1 | ordering:direction | cost LIMIT 3 |
| 03971 | hospital_1 | ordering:direction | cost LIMIT 3 |
| 04509 | document_management | ordering:direction | COUNT(*) DESC LIMIT 1 |
| 04510 | document_management | ordering:direction | COUNT(*) DESC LIMIT 1 |
| 04685 | college_3 | nl_alignment:unsupported_literal | 'COMPUTER LITERACY' |
| 05407 | shop_membership | data_type:numeric_text, nl_alignment:comparison_operator | membership_amount >= 100 |
| 05556 | products_gen_characteristics | nl_alignment:unsupported_literal | 'sesame' |
| 05557 | products_gen_characteristics | nl_alignment:unsupported_literal | 'sesame' |
| 05810 | customer_complaints | ordering:direction | COUNT(*) LIMIT 5 |
| 05811 | customer_complaints | ordering:direction | COUNT(*) LIMIT 5 |
| 06131 | customers_and_addresses | data_type:numeric_text, ordering:direction | SUM(t1.order_quantity) |
| 06132 | customers_and_addresses | data_type:numeric_text, ordering:direction | SUM(t1.order_quantity) |
| 06517 | wine_1 | ordering:direction | Score LIMIT 1 |
| 06518 | wine_1 | ordering:direction | Score LIMIT 1 |
| 06519 | wine_1 | ordering:direction | SCORE LIMIT 1 |
| 06520 | wine_1 | ordering:direction | SCORE LIMIT 1 |

Examples: `wine_1` "highest score" answered with `ORDER BY Score LIMIT 1` (ascending); `hospital_1` "three most expensive procedures" with `ORDER BY cost LIMIT 3`; `products_gen_characteristics` question names "cumin" while the SQL filters `product_name = "sesame"`; `medicine_enzyme_interaction` question says product 'Heme' (the product of Ferrochelatase) while the SQL filters on 'Protoporphyrinogen IX'; `election` question is cut off after "and" while the SQL has a second committee literal; `flight_1` "at least 100000" written as `salary > 100000`.

## 5. Regression cases (previous 120-item pilot review)

Review IDs were resolved through `cross_verified_review.csv` (`item_id`), not guessed. Detectors were specified before these results were looked at. Rules were later changed after reading flagged records. **Rules added after the first full run:** the quoted-value substitution rule (question quotes a value the SQL never uses; this is what makes the cumin/sesame and Heme/Protoporphyrinogen pairs CONFIRMED) and the nested ORDER BY/LIMIT tie rule. **Rules tightened after the first run:** dangling-sentence endings, "at least" no longer a minimum cue, date-column duration cues, comma-style joins, join-key type mismatch demoted to INFO, GROUP BY on a non-unique column demoted to INFO, phone-number/each-X guards in the entity-count rule. One of those changes did move a regression case: entity-count findings were demoted from CONFIRMED to LIKELY (a human could accept counting history rows), which changed R118 from CONFIRMED to LIKELY. Results are reported as they came out.

| Review | Record | Database | Pilot decision | This audit | Detectors |
| --- | --- | --- | --- | --- | --- |
| R009 | spider_train_02184 | formula_1 | UNCERTAIN | NO_ISSUE_DETECTED | none |
| R012 | spider_train_04685 | college_3 | INCORRECT | CONFIRMED_DEFECT | literal_not_in_question:CONFIRMED |
| R070 | spider_train_02686 | party_host | UNCERTAIN | NEEDS_REVIEW | text_numeric:REVIEW |
| R084 | spider_train_02406 | candidate_poll | INCORRECT | LIKELY_DEFECT | non_iso_date:LIKELY |
| R097 | spider_train_04071 | student_1 | INCORRECT | OUT_OF_SCOPE: rejected earlier by pool rule zero_only_result, not in train.jsonl; calibration control status = CONFIRMED_DEFECT | zero_match_literal:CONFIRMED (calibration control) |
| R101 | spider_train_05405 | shop_membership | INCORRECT | LIKELY_DEFECT | text_numeric:LIKELY |
| R111 | spider_train_04709 | department_store | UNCERTAIN | NEEDS_REVIEW | text_numeric:REVIEW |
| R118 | spider_train_04227 | cre_Doc_Tracking_DB | INCORRECT | LIKELY_DEFECT | entity_count:LIKELY |
| R119 | spider_train_03948 | hospital_1 | INCORRECT | CONFIRMED_DEFECT | superlative_direction:CONFIRMED |

- **Caught:** R012, R084, R101, R118, R119 (all five in-scope pilot-INCORRECT items) and R097 as an out-of-scope control. R070 and R111 (UNCERTAIN, text-stored numbers whose answer is right only by digit-width coincidence) are NEEDS_REVIEW with the "latent" tag, as intended.
- **R118 is LIKELY, not CONFIRMED.** COUNT(\*) on a location-history table differs from COUNT(DISTINCT document), but a human could accept it. For the same reason all 15 entity-count findings are LIKELY.
- **Missed: R009** ("won" read as `wins = 1`): a pure natural-language judgment; no rule was added to catch it. NO_ISSUE_DETECTED.
- **R097** (`spider_train_04071`) is not in `train.jsonl`: the pool rule `zero_only_result` rejected it earlier. It was audited only as a control (the zero-match literal rule finds the swapped first/last-name filter).
- **Baseball fan-out:** in the 42 retained `baseball_1` records, 0 reference the `team` table in any position (FROM, JOIN or comma join), and the detectors produced 0 many-to-many or fan-out findings (all baseball findings: {'sibling_spread:REVIEW': 1, 'entity_count:LIKELY': 1, 'text_numeric:LIKELY': 2}). The earlier many-rows-per-team-name hazard is therefore not present in the retained rows. Statuses: {'NO_ISSUE_DETECTED': 38, 'NEEDS_REVIEW': 1, 'LIKELY_DEFECT': 3}. The 127 excluded/quarantined controls include the 14+ fan-out rows (section 6).

## 6. Calibration (kept out of the 5,387 counts)

Positive controls: the 84 excluded and 43 quarantined **train** rows (`calibration_positive_controls.jsonl`, read only from the zip's train member). One baseball query exceeded the 30 s timeout and is BLOCKED in the controls.

| Original tier | CONFIRMED | LIKELY | REVIEW | NO_ISSUE | BLOCKED | total |
| --- | --- | --- | --- | --- | --- | --- |
| A_confirmed_defect | 0 | 16 | 24 | 13 | 0 | 53 |
| B_engine_or_row_choice_dependent | 1 | 0 | 24 | 6 | 0 | 31 |
| C_review_before_use | 1 | 13 | 17 | 11 | 0 | 42 |
| C_review_added_by_reproducibility_audit | 0 | 0 | 0 | 1 | 0 | 1 |

- Recall at REVIEW or above: 96 / 127 (75.6%); at LIKELY or above: 31 / 127 (24.4%). Many of these rows were excluded for reasons my detectors do not model (engine-dependent row choice, many-to-many joins in nested queries).
- **Specificity on the pilot's CORRECT items** (labels are AI-reviewed, not ground truth): 105 of 110 in-scope CORRECT items are clean, 5 are NEEDS_REVIEW, **0 are LIKELY or CONFIRMED**.
- Pilot INCORRECT in scope: {'CONFIRMED_DEFECT': 2, 'LIKELY_DEFECT': 3, 'NOT_IN_TRAIN': 1}. Pilot UNCERTAIN: {'NO_ISSUE_DETECTED': 1, 'NEEDS_REVIEW': 2}.

## 7. Automated checks and coverage

- Records parsed by sqlglot: 5,387 / 5,387; executed on SQLite 3.50.4: 5,387 / 5,387; re-execution reproduced the stored `expected_output`: 5,387 / 5,387; internal detector errors: 0; BLOCKED: 0.
- Detectors run on every record: agg_wording, antonym_columns, entity_count, execution, fanout_differential, inner_vs_left, join_structure, literal_match, literal_vs_question, missing_distinct, non_iso_date, null_group_flags, operator, ordering_direction, text_numeric.
- **Not implemented / weak:** fan-out differential only for ungrouped aggregates over joins at the top level; join-key semantics (a plausible but wrong key that still overlaps) is checked only for type differences and near-zero overlap; superlatives need a single ORDER BY key or a single MIN/MAX; no check of selected-column correctness beyond antonym attributes; unions/intersections/exceptions and nested subqueries are only partly covered; abbreviation and synonym literals (LA/Los Angeles) are flagged REVIEW at best; no truth source for what the question author meant.

## 8. Limitations

- AI hand-read, not human review. The regression set was seen while rules were tuned for false positives.
- Recall is low (section 6); the pilot's 5% is the safer residual-risk estimate.
- Text-numeric and date findings are database-specific: they change the answer on the stored data, but a different data sample could hide or expose them.
- SQLite 3.50.4 was used because the pool was built on SQLite >= 3.45.1; all 5,387 stored outputs reproduced.
- Records are not assessed for correctness of the stored `expected_output` against the question, only against the SQL.

## 9. Recommended exclusions and quarantine

| Action | Records | Basis |
|---|---:|---|
| **Exclude from any training corpus** | 30 | `confirmed_defects.jsonl`; 28 of 30 findings carry an executed answer-changing counterexample (direction flipped, operator swapped or quoted value substituted); 2 truncated questions have no value to substitute, and 2 operator findings are latent (no stored row on the boundary, so both operators give the same answer on this database; they remain wrong mappings). Independent re-check: see section 3 |
| **Quarantine pending human decision** | 58 | `likely_defects.jsonl`; answer changes under correct semantics but acceptance is arguable |
| **Hold for review before use** | 254 | `needs_review.jsonl`; latent, ambiguous, or weak evidence. None is a recommended exclusion on its own |
| Retain, unverified | 5,045 | not proven correct |

Nothing was changed: every record keeps its original line in `train.jsonl`. These are recommendations for the next dataset version.

## 10. Reproduce

`py -3.13 -I outputs/spider_pool_semantic_audit_v1/scripts/` then `s1_profile.py`, `s2_run.py`, `s3_calibrate.py` (after `s2_run.py`), `s4_adversarial.py`, `s5_finalize.py`, `s6_report.py`. `s5_finalize.py` asserts 5,387 unique IDs equal to `train.jsonl` and that the four status files plus NO_ISSUE partition them.
