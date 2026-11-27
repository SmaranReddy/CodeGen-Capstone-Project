# Pilot V3 follow-up: implementation plan for converter, contract and dataset revision

**Date:** 2026-11-12
**Status:** PLAN ONLY. Nothing here has been implemented. No source code, dataset, frozen artifact, converter, verifier or corpus was changed while writing this plan.
**Input:** the multi-agent AI-assisted semantic audit of the 120 Pilot V3 run01 review items (`outputs/sql2mql_v2/pilot_v3/run01/manual_human_review/ai_multiagent_review/`, ledger entry `DA-2026-11-12-P2-MULTIAGENT`). That audit is **not** the P2 human-review gate, and P2 has not passed.
**Target contract (chosen by the user):** the **type-conformant domain**.
**Review mechanism (decided by the user, 2026-11-12):** semantic review is done by a **pre-registered multi-agent AI panel** (§E.7), not by human reviewers. Every result it produces is labelled "AI-reviewed", never "human-reviewed". The frozen P2 gate in `pilot_freeze.json` was defined as a human review, so replacing it with the AI panel is a recorded protocol change (§E.7.6). It is not presented as P2 having passed.

## Contents

- 0. Summary and evidence collected for this plan
- Part A. The type-conformant domain contract
- Part B. Converter fix for LIKE (R040, R082)
- Part C. The 10 source-label findings
- Part D. Targeted audit of the prepared Spider source pool
- Part E. Versioning, acceptance gates and the AI review panel
- F. Ordered implementation sequence
- G. Risks
- H. Established, AI recommendation, or needs approval

---

## 0. Summary and evidence collected for this plan

**The audit found two independent problems:**

1. **Source SQL is wrong for its question** in 10 of the 120 items.
   - 4 are high-confidence: R084, R097, R101 and R119.
   - 6 depend on how the question is read: R009, R010, R012, R090, R111 and R118.
   - The SQL→MQL verifier cannot detect these by design. It checks MQL against SQL, not SQL against the question.
2. **The MQL differs from the SQL** in 2 of 120 items under the type-conformant domain (R040, R082), and in 20 of 120 under the strict domain.
   - The extra 18 strict-domain items need text values stored in numeric columns.
   - The type-conformant contract excludes those values (Part A, §A.6).

**Measurements taken for this plan.** All were read-only and taken on 2026-11-12. "Local" means the local MongoDB 8.0.34 on port 27018 and the Spider SQLite files. Nothing was measured on Atlas.

| # | Measurement | Result |
|---|---|---|
| M1 | Storage-class census of every column in the 139 databases used by `data/processed/spider_pool_v2/{train,val_internal}.jsonl` (SQLite `typeof()`, file opened `mode=ro`) | 3,824 columns.<br>3,507 conform to their declared affinity.<br>199 are **homogeneous TEXT under a numeric affinity**: DATETIME 161, DATE 18, TIMESTAMP 16, BOOL 3, NUMERIC 1.<br>**115 are mixed-class** (integer+text and similar): baseball_1 90, bike_1 22, cre_Theme_park 1, formula_1 1, wine_1 1.<br>3 have BLOB affinity with one class. |
| M2 | Same census over the Atlas evaluation mapping `kaggle/nl2mql_spider_dev_1034_bundle_v2/data/atlas_mapping.json`: 70 collections, 383 fields, types recorded at the 2026-10-20 parity check | 17 fields differ from their declared affinity. 15 are homogeneous strings in DATETIME, bool or timestamp columns.<br>**2 are mixed:** `wta_1__players.birth_date` (int+string) and `wta_1__rankings.player_id` (int+string).<br>2 strings were stored with U+FFFD replacements. |
| M3 | Fold-trap census: text values containing U+212A (Kelvin sign) or U+017F (long s) in the same 139 databases | **0 columns.** 98 text columns contain some non-ASCII character. |
| M4 | SQLite 3.40.1 `LIKE` compared with MongoDB 8.0.34 `$regex` (options `is`) and with the proposed ASCII-class regex (options `s`), on an inline `$documents` set | The current translation over-matches U+212A (Kelvin sign) for `k`, U+017F (long s) for `s`, and `É` for `é`. The proposed translation matched SQLite exactly on all 7 patterns × 16 values, in both the `$regex` and `$regexMatch` forms (details in §B.4). |
| M5 | LIKE frequency | 8 of the 213 accepted pilot labels use LIKE, and 1 rejected pilot item does.<br>135 of the 5,387 `spider_pool_v2/train.jsonl` records contain ` LIKE `, with 138 LIKE patterns in all. None of those patterns, and none of the 124 `$regex` labels in corpus v1, contains a non-ASCII character. |
| M6 | Same-SQL paraphrase siblings of the 10 source findings in `spider_pool_v2` | 9 of the 10 findings have **exactly one sibling** with the same `db_id` and normalised SQL, and it is in `train.jsonl`.<br>R097 (`spider_train_04071`) is **not in spider_pool_v2** at all, neither retained nor excluded. The pilot population (`pilot_freeze.json`, drawn from the 7,773 train records) is therefore not the same population as spider_pool_v2. |

The census script is not committed, because this plan creates no code. Step S1 of §F turns it into `scripts/finetune_v2/type_contract.py`, which must reproduce M1 and M2 exactly.

---

## Part A. The type-conformant domain contract

### A.1 Why "matches the declared type" is the wrong definition

The audit described the type-conformant domain as "stored values match the declared column affinity". Measurements M1 and M2 show that this cannot be applied literally to Spider:
- Spider stores dates and booleans as text in DATETIME, DATE, TIMESTAMP and BOOL columns, all of which have NUMERIC affinity. There are 198 such columns in the training databases (195 date, 3 boolean) and 15 such fields in the evaluation set.
- None of them caused a translation error. Text compared with text, and sorted against text, behaves the same way in SQLite and MongoDB.
- A declared-affinity rule would reject them all and gain nothing.

What the converter actually relies on is the **observed storage class**. `schema_profile.Column.kind` returns `num`, `text`, `null` or `mixed`, and `Column.all_int` marks integer-only columns. The contract should state that assumption and enforce it.

### A.2 Definition: contract TC-1

For every collection `C = <db_id>__<table>` that a label is built on or a prediction is scored on, and every field `f`:

1. **Contract class.** `K(C,f)` is one of the following, taken from the SQLite snapshot's observed non-NULL storage classes:
   - `int`: SQLite INTEGER only;
   - `num`: INTEGER and/or REAL;
   - `text`: TEXT only;
   - `null`: no non-NULL value;
   - `nonconformant`: anything else (mixed classes, or BLOB).

   It is frozen in a versioned manifest. The declared SQL type and its affinity are recorded beside it as metadata, not as the contract.
2. **Allowed BSON types.** Every non-null value of `f` has a BSON type allowed for `K`. A field may also be explicit `null` or missing; both are already handled (F15).

   | K | Allowed BSON types |
   |---|---|
   | `int` | int, long |
   | `num` | int, long, double |
   | `text` | string |
   | `null` | null or missing only |

   `int` is kept separate from `num` because integer division (F8) depends on `all_int`.
3. **Forbidden types anywhere:** bool, date, timestamp, decimal128, objectId (other than `_id`), binary, array and embedded document.
4. **Field set.** No field outside the table's column set (plus `_id`). A column's field may be absent only in a document where that column's value is NULL; this is the `missing` representation used by `spider_train_v2_missing` (F15).
5. **Document count** equals the SQLite row count.
6. **Text encoding.** Strings are valid UTF-8. Any U+FFFD replacement is listed per field, and is currently known only for 2 evaluation strings.
7. **`nonconformant` fields** may appear in a label only where type does not matter: projection, or IS NULL / IS NOT NULL. The current converter already rejects type-sensitive use of `mixed` columns (comparison, aggregate, arithmetic, join key, LIKE). TC-1 makes that rule explicit and testable.

**TC-1 deliberately does not constrain:**
- **Values within a class.** A Kelvin sign in a `text` field is inside the contract, which is why R040 and R082 must be fixed in the converter (Part B).
- **Declared-vs-stored mismatches** such as text dates in DATETIME columns. These are recorded and feed detector D1 in Part D, but they are not violations.

### A.3 Where the contract must hold, and what proves it

| Place | Collections | Existing check | Gap | Proposed enforcement |
|---|---|---|---|---|
| Label construction (local) | `spider_train_v2` on local MongoDB (804 collections) | `mongo_import.parity()`: per-table multiset of type-tagged tuples, under both NULL forms | No frozen contract; the converter re-derives classes from SQLite on each run, and no record says which class each label assumed | New `type_contract_v1.json` manifest. Each converted label records the manifest sha256 and the contract classes of the fields it uses. |
| Model evaluation (Atlas) | `codegen_spider1`: 70 mapped collections (80 present) | `dev1034_core.atlas_preflight(full_parity=True)`: type-tagged row-multiset digest against the bundled SQLite, which implies type equality | (a) `local_atlas_scoring.py score` defaults to `full_parity=False`, so scoring checks counts only.<br>(b) `preflight --quick` skips the digest.<br>(c) The latest recorded type census (`parity_2026_10_20`) is three weeks old.<br>(d) There is no per-field contract check, and the 2 mixed `wta_1` fields are not flagged. | New mandatory read-only `$type` census at **every** scoring run, compared with the contract manifest; fail closed. The full digest stays as the deep check. |
| Training databases on Atlas | none (training labels are never executed on Atlas) | none | n/a | State this explicitly: local type parity proves **nothing** about Atlas. If training collections are ever uploaded, they must pass the same Atlas census first. |

**Do not infer that Atlas matches local.**
- The local `spider_train_v2` database and the Atlas `codegen_spider1` database hold different databases: train and dev respectively.
- Atlas conformance is established only by a census run against Atlas, with the read-only account, at scoring time.

### A.4 Proposed enforcement components (new files; nothing existing is edited)

1. **`scripts/finetune_v2/type_contract.py`** (converter-independent). It must not import `sql2mql_v2*`, `verifier*` or `mongo_import`.
   - `build`:
     - reads SQLite read-only;
     - writes `data/interim/finetune_v2/type_contract_v1.json` with per-field `declared`, `affinity`, `observed_classes`, `K`, `declared_mismatch` (bool) and `utf8_replacements`, plus the file's own sha256;
     - refuses to overwrite.
   - `check-local`: runs a `$type` census on the local Mongo `spider_train_v2` and `spider_train_v2_missing` databases (an aggregation of `$group` on `{$type: "$f"}` per field) and compares it with the manifest.
2. **`scripts/nl2mql/atlas_type_contract_check.py`** (new, read-only).
   - Uses the read-only evaluation URI and the existing `mongo_conn.tls_kwargs`.
   - Runs the same `$type` census over the mapped evaluation collections and compares it with a contract manifest built from the bundle's SQLite files.
   - Fails closed on:
     - any disallowed BSON type;
     - any extra field;
     - a count mismatch;
     - a missing collection.
   - Writes `outputs/<experiment>/preflight/atlas_type_contract_<ts>.json`.
   - **It does not modify `dev1034_core.py` or `local_atlas_scoring.py`**: their hashes pin MQL-0006 and MQL-0007. Each future scoring script calls the new check, or requires its report as an input, before scoring.
3. **Optional, needs mentor approval:**
   - An Atlas `$jsonSchema` validator with `bsonType` per field on each evaluation collection. This needs `collMod` from a separate admin account and is never done with the evaluation account.
   - It hardens against future writes but is not needed for correctness if the census runs at every scoring run.

### A.5 Evaluation-set consequence (decision needed, not retroactive)

M2 found 2 mixed fields in the evaluation set: `wta_1__players.birth_date` and `wta_1__rankings.player_id`.

**Recommendation:**
- For future experiments, report a "touches a nonconformant field in a type-sensitive position" slice separately.
- Do **not** drop or re-score these items in MQL-0006 or MQL-0007; those results stay as recorded.
- Excluding or slicing evaluation items changes the evaluation protocol, so it **needs mentor approval** before use.

### A.6 Why TC-1 excludes the 18 strict-domain-only discrepancies

**What the 18 are:**
- 11 untyped `$gt` filters, for example `{"age": {"$gt": 30}}`.
- 7 `$avg`/`$sum` aggregates over numeric columns.

All 18 differ from SQLite only when a **text value** sits in a column whose contract class is `num` or `int`:
- SQLite ranks any TEXT above every number, so `'unknown' > 30` is true. MongoDB's `$gt` compares within the numeric type bracket only.
- SQLite `AVG` reads `'abc'` as 0 and `SUM` reads a numeric prefix, while MongoDB `$avg`/`$sum` ignore strings.

**Under TC-1 such a value is a contract violation (rule A.2.2).** The census rejects the collection before any evaluation runs. Inside the contract the two engines agree exactly; the audit's execution evidence showed this on real data for all 18.

**Why not emulate SQLite instead.** That would mean `$cond`/`$type` wrappers on every comparison and a SQLite-compatible numeric-prefix parser inside `$avg`/`$sum`. It would:
- make the training labels much longer and less idiomatic, teaching the model MQL no MongoDB user writes;
- add a new emulation surface that would itself need verifying;
- protect only against data the preflight already refuses.

**Decision for v2.1:** no mixed-type emulation. Record the 18 as `correct under TC-1` and keep their strict-domain verdict in the audit files for traceability.

---

## Part B. Converter fix for LIKE (R040, R082)

### B.1 What R040 and R082 are

| ID | SQL | Current MQL `$match` |
|---|---|---|
| R040 (`spider_train_06342`, e_government) | `... WHERE line_1_number_building LIKE "%6862 Kaitlyn Knolls%"` | `{"line_1_number_building": {"$regex": "6862 Kaitlyn Knolls", "$options": "is"}}` |
| R082 (`spider_train_04935`, store_product) | `... WHERE product LIKE "%Scanner%"` | `{"product": {"$regex": "Scanner", "$options": "is"}}` |

Both return the right rows on the real data, because no Kelvin sign or long s exists anywhere (M3). Under TC-1 they are still wrong translations, because the contract allows any string value.

### B.2 The exact semantic difference

- **SQLite `LIKE`** (default, `PRAGMA case_sensitive_like = OFF`) folds case for the **26 ASCII letters only**. Every other character, including non-ASCII letters, matches only itself.
  - `%` matches any run of characters, and `_` matches exactly one character (one code point under UTF-8).
  - `%` and `_` both match newlines.
- **MongoDB `$regex` with option `i`** uses PCRE case-insensitive matching with Unicode case folding. That makes it wider than SQLite in two ways:
  - **Non-ASCII case pairs:** `É` matches `é`. The converter guards this by rejecting non-ASCII *letters* in the pattern (`ch.isalpha()`).
    **That guard has a hole** (found 2026-11-12 in review). Some characters have Unicode case pairs but are not letters to Python, so they pass the guard, and PCRE `i` still folds them. Examples: circled letters U+24B6/U+24D0 and Roman numerals U+2160/U+2170. Tested locally with SQLite 3.40.1 and MongoDB 8.0.34: SQLite `LIKE` matched only the identical character, while `$regex` with options `is` matched both cases. This is a second v2 bug, on the pattern side. v2.1 fixes it too, because without `i` a non-ASCII character matches only itself. On current data the impact is nil: no pool LIKE pattern contains a non-ASCII character (M5).
  - **Non-ASCII characters that fold to ASCII:**
    - U+212A KELVIN SIGN folds to `k`, so the pattern `k` (or `K`) also matches `"\u212a"`.
    - U+017F LATIN SMALL LETTER LONG S folds to `s`, so the pattern `s` also matches `"\u017f"`.

    The pattern is pure ASCII, so the pattern guard does not catch this.
- The current converter guards the second case only through `schema_profile.Column.fold_trap`, which looks for these characters in the **current data**. That guard is data-dependent. It protects one snapshot, not the contract.

**Why the verifier missed it.**
- The `case_variants` fixture family in `v2_fixtures.py` changes the case of ASCII text only.
- No fixture inserts a fold-trap character, so a translation that over-matches through Unicode folding passes every fixture.
- This is a verifier blind spot as well as a converter bug.

### B.3 Proposed minimal fix

**Idea:** stop asking the regex engine to fold case. Expand each ASCII letter in the pattern into a two-character class, and drop the `i` option.

```
SQLite LIKE pattern      current regex (options "is")      proposed regex (options "s")
%Scanner%                Scanner                            [sS][cC][aA][nN][nN][eE][rR]
%6862 Kaitlyn Knolls%    6862 Kaitlyn Knolls                6862 [kK][aA][iI][tT][lL][yY][nN] [kK][nN][oO][lL][lL][sS]
a_b                      ^a.b\z                             ^[aA].[bB]\z
```

**Rules for the new `like_regex_ascii(pattern)`:**
- `%` becomes `.*` and `_` becomes `.`. This is unchanged; the `s` option keeps newlines matching.
- An ASCII letter `c` becomes `[` + lower(c) + upper(c) + `]`.
- Any other character in `REGEX_META` (`\ . ^ $ | ? * + ( ) [ ] { } /`) is backslash-escaped. This set already includes `[` and `]`.
- Any other character is copied literally. Without `i` it matches only itself, which is SQLite's behaviour.
- Anchors stay as they are: `^` and `\z`, dropped next to a leading or trailing `%`.
- The emitted options become `"s"` in **all three** places in `Converter.like`:
  1. the positive query form `{path: {"$regex": rx, "$options": "s"}}`;
  2. the negated query form `{path: {"$not": {"$regex": rx, "$options": "s"}, "$ne": None}}`;
  3. the `$expr` form `{"$regexMatch": {"input": ..., "regex": rx, "options": "s"}}`, inside the existing `$type == "string"` guard.

**Behaviour preserved:**
- Every pattern accepted today is still accepted.
- On any data free of fold-trap characters, the matched set is the same.
- Patterns are still rejected exactly as today when:
  - the pattern is not a string literal;
  - LIKE is applied to a `num` or `mixed` column;
  - the pattern contains a non-ASCII letter.

**Guards:**
- The non-ASCII-letter pattern rejection is **kept** in v2.1, which keeps the change minimal. Under the new translation it is no longer needed for correctness, because `é` without `i` matches only `é`, exactly as in SQLite. Lifting it is optional future work with its own tests.
- The `fold_trap` rejection is **kept but becomes redundant**. It stays so that the set of accepted inputs is identical to v2. If it is removed later, that is a separate, tested change.

**Rejected alternatives:**
- **Keep `i` and add a guard excluding U+212A and U+017F.** This depends on knowing every PCRE fold to ASCII, a list that can change with the PCRE or ICU version.
- **`$toLower` on both sides.** MongoDB documents `$toLower` as well-defined only for ASCII, so its non-ASCII behaviour is not a contract to build on. It also it changes the pipeline shape (`$expr` everywhere, no index use) and does not handle `_` and `%`.
- **Reject every LIKE.** This loses 135 pool records (M5) for a defect that has a complete fix.

### B.4 Counterexamples and regression tests

**Counterexample set**, already executed for this plan as M4: SQLite 3.40.1 compared with MongoDB 8.0.34. Non-ASCII test values are written as Python escapes so they cannot be confused with ASCII letters. In particular, U+212A has a *canonical* decomposition to ASCII `K`, so even NFC normalisation, the common silent one in editors and tools, turns it into `K`. U+017F changes only under NFKC. This plan's own first draft lost the Kelvin sign this way. **Tests must build these strings from escapes, never from pasted glyphs.**

| Pattern | Values that must match (SQLite) | Values that must NOT match | Current `is` result | Proposed `s` + class result |
|---|---|---|---|---|
| `%6862 Kaitlyn Knolls%` | "6862 Kaitlyn Knolls", "6862 kaitlyn knolls" | `"6862 \u212aaitlyn \u212anolls"` | over-matches the U+212A row | exact |
| `%Scanner%` | "Scanner", "SCANNER" | `"\u017fcanner"` | over-matches the U+017F row | exact |
| `k` | "k" | `"\u212a"` | over-matches | exact |
| `S` | "s" | `"\u017f"` | over-matches | exact |
| `é` (converter rejects it today) | "é" | "É" | over-matches | exact (stays rejected in v2.1) |
| `%ix%` | "ix" | `"\u0130x"` | exact | exact |
| `a_b` | "a\nb", "axb" | | exact | exact |
| `\u24b6` (circled A; passes v2's guard) | `"\u24b6"` | `"\u24d0"` | over-matches | exact |
| `\u2170` (small Roman numeral one; passes v2's guard) | `"\u2170"` | `"\u2160"` | over-matches | exact |

**Regression tests.** These are new tests; `tests/test_finetune_v2.py::test_F13_like_regex` stays as it is for the frozen v2 module.

1. **`tests/test_finetune_v2_1_like.py`** (unit, no Mongo):
   - exact regex strings for at least 10 patterns, covering a leading or trailing `%`, `_`, every metacharacter, digits, spaces, an embedded `%`, an empty pattern, `%`, and `%%`;
   - options are `"s"` in all three emission paths;
   - the non-ASCII-pattern and `num`/`mixed` rejections still raise `Unsupported("like", ...)`.
   - a pattern containing U+24B6 or U+2170 is accepted exactly as in v2, because v2's guard lets it through, and is now translated without `i`.
2. **`tests/test_finetune_v2_1_like_engine.py`** (needs local Mongo, marked `mongo`):
   - for every pattern × value in the table above plus a random set (≥ 500 ASCII and Unicode strings, fixed seed), the SQLite `LIKE` result equals both MongoDB forms;
   - includes NOT LIKE, with SQLite's NULL behaviour: NULL is never returned.
3. **Verifier fixture family `fold_traps`**, in a new `scripts/finetune_v2/fixtures3_1.py` and `verifier3_1.py`:
   - for any label using LIKE, add copies of matching rows with U+212A and U+017F substituted for k/K and s/S, and copies with U+0130, U+0131 and É;
   - the v2 MQL for R040 and R082 must be **rejected** by verifier 3.1, a negative control showing the blind spot is closed;
   - the v2.1 MQL must be accepted.
4. **Atlas server check, before v2.1 labels are used in any Atlas-scored experiment:**
   - run counterexample tests 1 to 4 against the Atlas server version through the read-only account;
   - if `$documents` is refused, run the equivalent `$regexMatch` over `$literal` values in a `$project` on any existing evaluation collection, `$limit: 1`;
   - PCRE behaviour could in principle differ between server versions, so local 8.0.34 alone is not enough.

### B.5 Does anything need to be rejected instead?

**No.**
- Every currently supported LIKE (ASCII-letter patterns on `text` columns) has an exact translation.
- **Expected impact on existing outputs:**
  - none of the 139 training databases holds a fold-trap character (M3), so **no execution result changes** on current data;
  - only the pipeline text of LIKE labels changes;
  - in the pilot, that is the 8 accepted LIKE labels;
  - every other label must stay byte-identical (§E.4).

---

## Part C. The 10 source-label findings

**Principles:**
- No gold SQL is rewritten automatically.
- No authoritative source ranks above Spider's own gold for the training split. No upstream erratum is known to us for these train items, and none was looked for in this plan. So "correct from an authoritative source" mostly does not apply.
- **The only authoritative evidence available is the database contents.**
- **No corrected SQL in this revision.** With no human reviewer, a corrected gold SQL would be an AI-generated training label. The plan quarantines instead.
  - The cost is small: about 10 findings plus siblings, against 5,387 pool records.
  - The four simple fixes (the R097 name swap, the R119 DESC, LEFT JOIN for R010/R090, COUNT(DISTINCT) for R118) are listed as *candidate corrections* in `data/interim/source_label_adjudication_v1.jsonl`. None enters a dataset unless a later, separately approved step adds corrections.
- **Every finding is propagated to its paraphrase siblings** (M6): records with the same `db_id` and normalised SQL. Each sibling's question is then judged separately, since a sibling's wording may make the same SQL correct.

**Default action:** quarantine all 10 findings, plus their siblings pending review, in the new pool version (§E.2). In the tables below, "adjudication" means the AI panel's adjudication (§C.3).

### C.1 High-confidence findings (4)

These have concrete data-backed counterexamples, with all three agents rejecting or with the adjudicator ruling on execution evidence.

| ID / record | Question | Defect (AI finding) | Sibling (M6) | Recommendation |
|---|---|---|---|---|
| **R084** `spider_train_02406` candidate_poll | "List all people names in the order of their date of birth from old to young." | `ORDER BY date_of_birth` sorts DD.MM.YYYY **text** lexically, which is not chronological. | `spider_train_02407` ("ordered by their date of birth"): same defect, no direction stated. | **Quarantine (tier A after panel confirmation).** A correct SQL needs `substr` or date functions, which the converter rejects (F14). It cannot be corrected inside the converter's scope. |
| **R097** `spider_train_04071` student_1 | "How many students are taught by teacher TARRING LEIA?" | `firstname = "TARRING" AND lastname = "LEIA"`, but the data stores LastName TARRING, FirstName LEIA. The answer is 0 instead of 7. | **Not in spider_pool_v2** (M6). | **Quarantine; panel adjudication records the finding.** The swap is a candidate correction only, not applied (§C principles). **No pool-v3 action is needed**, because the record is not in the pool. The correction matters only for the pilot and corpus population (§E.2b). The reason it is absent is not known. Establish it by applying the pool builder's documented rules to this one train record: the result is `count = 0`, so check whether the builder drops degenerate results. **Do not open the zip's `rejected.jsonl`**, which spider_pool_v2's manifest lists under `members_not_opened` and which may contain dev records. |
| **R101** `spider_train_05405` shop_membership | "the 3 branches that have the most memberships" | `membership_amount` is TEXT, so `'91'` ranks above `'1418'`. | `spider_train_05404`: same SQL, same defect. | **Quarantine, tier A.** A correct SQL needs CAST, which is unsupported, so the item cannot be corrected inside the converter's scope. |
| **R119** `spider_train_03948` hospital_1 | "the patient who has the most recent undergoing treatment" | `ORDER BY dateundergoes LIMIT 1` is ASC, which gives the **oldest**. | `spider_train_03949` ("most recent"): same defect. | **Quarantine both records; panel adjudication.** `DESC` is a candidate correction only. If it is applied later, first confirm `dateundergoes` is ISO-ordered text. |

### C.2 Medium-confidence, interpretation-dependent findings (6)

| ID / record | Question | Concern (AI finding) | Sibling | Recommendation |
|---|---|---|---|---|
| **R009** `spider_train_02184` formula_1 | "...in position as standing and won?" | `wins = 1` reads "won" as exactly one win, versus `wins >= 1`. | `spider_train_02183`: same SQL. | **Quarantine; panel adjudication.** No detector reliably finds this class (§D.8). |
| **R010** `spider_train_06757` activity_1 | "Show all the faculty ranks and the number of students advised by each rank." | INNER JOIN drops ranks with no advisees, though "all" asks for them. Spider convention usually accepts this. | `spider_train_06758` ("each rank"): weaker "all". | **Quarantine; panel adjudication under the user's "all X" policy** (§H.3), applied to D3 hits pool-wide. A LEFT JOIN with COUNT(col) rewrite is a candidate correction only. |
| **R012** `spider_train_04685` college_3 | "...the instructor of course that has course name" (truncated) | The question omits the literal "COMPUTER LITERACY" used in the SQL. | `spider_train_04686`: full question. | **Quarantine.** Keep the sibling, which is a complete question with the same SQL. Never rewrite question text. |
| **R090** `spider_train_00515` allergy_1 | "Show all allergies with number of students affected." | `GROUP BY` over the junction table `Has_allergy`: there is **no join**, yet allergies with zero students in `Allergy_Type` are omitted. | `spider_train_00516` ("each different allergy"). | **Same "all X" policy as R010.** Detector D3 must cover child-table grouping, not only joins. |
| **R111** `spider_train_04709` department_store | "top three products purchased in the largest amount" | `total_amount_purchased` is TEXT and sorts lexically. It is correct on the current data only by luck. | `spider_train_04710`: same SQL. | **Quarantine (tier A-type), same class as R101.** It cannot be corrected without CAST. It is "medium" only because the current result is correct. |
| **R118** `spider_train_04227` cre_Doc_Tracking_DB | "the number of documents in that location" | `count(*)` over `Document_locations`, whose primary key is composite (document, location, date), counts stays rather than distinct documents. | `spider_train_04226`: same SQL. | **Quarantine; panel adjudication.** `COUNT(DISTINCT document_id)` is a candidate correction only. |

### C.3 Adjudication protocol (AI panel)

- **Panel.** The §E.7 panel reviews each finding and each sibling. Reviewers see the question, schema, SQL, executed result and the sibling's question. They do **not** see the earlier audit's verdicts: §C findings are re-reviewed blind.
- **Verdicts.** Each reviewer records Accept or Reject with logged execution evidence. The adjudicator rules on disagreements with fresh evidence, never by majority vote.
- **Asymmetric rule.** A record stays in the dataset only if the panel accepts it without any unresolved Reject or UNSURE. Anything else is quarantined, never accepted on a tie.
- **Log.** Outcomes go to `data/interim/source_label_adjudication_v1.jsonl`, append-only and never edited in place.

---

## Part D. Targeted audit of the prepared Spider source pool

**Scope:**
- `data/processed/spider_pool_v2/train.jsonl` (5,387 records) and `val_internal.jsonl` (734), with `quarantine.jsonl` (43) for reference.
- Read-only over the files and the SQLite databases.
- **Not** Spider dev or test, `data/splits_expanded/test.jsonl`, or the zip's `dev.jsonl`.

**Why the 120-item error rate does not extrapolate:**
- The 120 were sampled from the 213 **converter-accepted** pilot labels, out of 300 pilot records drawn from the 7,773-record train population of `pilot_freeze.json`.
- That is a different population from spider_pool_v2. It is filtered by convertibility, and at least one item (R097) is absent from the pool.
- Convertibility selects for simple SQL: no functions, no subqueries.
- The audit's labels are AI judgements on 120 items, with a Wilson 95% interval of roughly 4.6% to 14.7% for 10/120.
- **The audit therefore reports hit counts and measured detector precision only, never a projected pool error rate.**

**Common protocol for every detector:**
- **Freeze before running.** Store the code, thresholds and word lists in `data/interim/source_label_audit_v1/prereg.json` (sha256) **before** running on the pool.
- **Recall check.** First run each detector on the 120 review items. It must flag every "must detect" item listed for it. Its flags among the 110 audit-accepted items are reported as false positives under the audit's labels. The 120 are development data for the detectors, not test data.
- **Precision check.** For each detector, the §E.7 panel labels a seeded random sample of min(40, hits) pool hits. Report precision with a Wilson interval, labelled AI-reviewed.
- **Actions.** A hit means quarantine, pending AI-panel review. **No detector repairs SQL.** A non-hit is not a certification of correctness.
- **Output.** One row per (record, detector) in `outputs/source_label_audit_v1/hits.jsonl`, with evidence fields.

### D1. Numeric- or date-like values stored as TEXT, used in ordering or comparison

- **Detects:**
  - the SQL uses a column in `ORDER BY`, `MIN`/`MAX`, or `<`, `<=`, `>`, `>=`, `BETWEEN`;
  - and the column's contract class is `text` (TC-1 manifest);
  - and ≥ 90% of its non-null values parse as a number, or as a date in a recognised format.
- **Two subclasses:**
  - **D1a, order-unsafe now:** on the actual values, lexical order differs from numeric or chronological order. Examples: DD.MM.YYYY, M/D/YYYY, unpadded numbers.
  - **D1b, order-safe by luck or by format:** lexical order matches on current data. Examples: ISO-8601 dates, fixed-width numbers. Flag it, but at lower priority. ISO-8601 dates sort correctly by design; unpadded numbers are only accidentally safe.
- **Must detect:** R084 (D1a), R101 (D1a), R111 (D1b or D1a depending on values; it must be flagged either way).
- **False-positive risks:**
  - ISO date columns (very common in Spider: 195 text date columns in M1) are correct to sort lexically. Report them as D1b-ISO, not as defects.
  - Code-like identifiers such as zip codes or phone numbers, where text order is intended.
  - Equality comparisons are unaffected; exclude `=`, `IN` and LIKE.
- **Quarantine, never repair:** any D1a hit. The repair needs CAST or date functions, which the converter rejects. D1b-non-ISO goes to review; D1b-ISO is reported only.

### D2. Temporal wording that contradicts the sort direction

- **Detects:** the question contains a temporal superlative and the SQL sorts a date-like or age-like column in the opposite direction, usually with `LIMIT`.
- **Lexicon (frozen in the pre-registration):**
  - "most recent / latest / newest / last" expects DESC on a date;
  - "earliest / oldest / first" expects ASC on a date;
  - "youngest" expects **DESC on a birth date** but **ASC on age**;
  - "oldest" expects ASC on a birth date and DESC on age.
- **Column role:** decided from the column name (`date|time|year|dob|birth|created|_at`, versus `age`) and the TC-1 class.
- **Must detect:** R119. Control: R084's "from old to young" with ASC on `date_of_birth` is direction-correct, so D2 must **not** flag it. D1 catches its format problem.
- **False-positive risks:**
  - "first" used as ordinal ("first name");
  - negations;
  - questions that list in order without a superlative;
  - year columns stored as numbers, which are fine;
  - "last name".
- **Quarantine, never repair:** a direction flip changes the answer, so the record is quarantined. The column format must also be ISO-ordered (D1) before DESC is correct.

### D3. "All X" or "each X" questions that may omit entities

- **Detects:** the question contains "all", "each", "every", "for each" or "show … with number of …" (frozen list), and either:
  - **D3a:** an INNER JOIN between the entity table and a child table, with grouping on entity attributes; or
  - **D3b:** a GROUP BY over a child or junction table when an entity table exists that holds the grouped key as its primary key, with **no join**.
- **Evidence levels:**
  - data-level: number of entity keys absent from the child table on the actual database (> 0 means rows are really omitted now);
  - schema-level: absent keys are possible.
- **Must detect:** R010 (D3a) and R090 (D3b, no join; a join-only rule misses it).
- **False-positive risks:**
  - **High.** Spider conventionally accepts INNER JOIN for "each X".
  - Many questions mean "each X that has some Y".
  - The entity table may not exist in the schema.
  - D3a uses the data-level count to rank hits but must report both evidence levels.
- **Quarantine, never repair:** all hits go to panel review under the user's "all X" policy (§H.3). A LEFT JOIN rewrite changes the semantics; it is a candidate correction only.

### D4. Counting entities over composite keys or one-to-many tables

- **Detects:**
  - `COUNT(*)` with a GROUP BY or a single filter over table T;
  - where the question's counted noun ("number of documents / students / …") maps to a column `e` of T (by name or foreign key);
  - where `e` is not unique in T (composite PK containing `e`, or a one-to-many foreign key);
  - **and** on the actual data `COUNT(*) != COUNT(DISTINCT e)` for at least one group.
- **Must detect:** R118.
- **False-positive control:** run D4 on the 110 items the audit accepted, and report how many it flags. Each one is a false positive under the audit's labels. There is no specific known negative item for D4; R046 was a DISTINCT dispute, not a counting one.
- **False-positive risks:**
  - **High.** Questions may legitimately count rows (events, records, stays).
  - Noun-to-column mapping errors.
  - Equal counts on the data hide the issue; report these as schema-level only.
- **Quarantine, never repair:** any hit that is data-level different. A `COUNT(DISTINCT …)` rewrite is a candidate correction only.

### D5. Truncated or ambiguous questions

- **Detects:**
  - questions ending in a preposition, article, "named", "called", "that has …", or a dangling "with";
  - an unbalanced quote;
  - questions shorter than a threshold relative to the SQL's number of predicates;
  - or a question/`question_raw` mismatch beyond tokenisation.
- **Must detect:** R012.
- **False-positive risks:**
  - Spider's terse phrasing ("List the names of …").
  - Questions ending in a column name that is also a preposition.
  - Typos.
- **Quarantine, never repair:** never rewrite question text. If a same-SQL sibling has a complete question, keep that sibling and quarantine the truncated record.

### D6. Literal filter values the question does not support

**Check whether `scripts/spider1/audit_literals.py` can be reused.** It flags MQL literals absent from the SQL or question for the gold-MQL sets. Reuse its normalisation and ignore-list, and extend it for SQL literals against the question and the database.

- **Sub-detectors:**
  - **D6a.** A string or number literal in WHERE or HAVING is not in the question after case, quote and whitespace normalisation, and is not covered by a frozen implicit-mapping list (for example 1 or 'T' for "won", 'F' for "female").
  - **D6b.** A literal compared with column `c` matches **zero** rows of `c` but **does** occur in another column of the same table or a joined table. Example: firstname versus lastname.
  - **D6c.** The query returns 0 rows or a count of 0 because of the literal.
- **Must detect:** R097 (D6b and D6c) and R012 (D6a: "COMPUTER LITERACY" is not in the question).
- **False-positive risks:**
  - Implicit encodings (sex codes, booleans, ratings).
  - Spelled-out numbers ("three") versus digits.
  - Date formats.
  - Literals legitimately absent from the data (a 0 answer can be correct).
- **Quarantine, never repair:** D6b hits go to panel review. A swap is a candidate correction only, as for R097.

### D7. Expected detector coverage of the 10 findings

| Finding | D1 | D2 | D3 | D4 | D5 | D6 |
|---|---|---|---|---|---|---|
| R084 | ✔ (a) | must not flag | | | | |
| R097 | | | | | | ✔ (b, c) |
| R101 | ✔ (a) | | | | | |
| R119 | | ✔ | | | | |
| R009 | | | | | | weak (a) |
| R010 | | | ✔ (a) | | | |
| R012 | | | | | ✔ | ✔ (a) |
| R090 | | | ✔ (b) | | | |
| R111 | ✔ | | | | | |
| R118 | | | | ✔ | | |

### D8. Limits

- R009 ("won" read as `wins = 1`) is an interpretation error that **no** proposed detector reliably finds. Classes like this are found only by semantic review, here the AI panel of §E.7.
- So the detector audit is a **targeted filter, not a replacement** for a reviewed sample (§E.5).

---

## Part E. Versioning and acceptance gates

### E.1 New converter and verifier versions (new files only)

| Item | Path (new) | Notes |
|---|---|---|
| Converter `sql2mql-v2.1` | `scripts/finetune_v2/sql2mql_v2_1.py` | Imports the frozen `sql2mql_v2`, subclasses `Converter` to override `like()`, adds `like_regex_ascii()`, and sets `CONVERTER_VERSION = "sql2mql-v2.1"`. **`sql2mql_v2.py` and `schema_profile.py` are not edited**, so their hashes in `data/interim/finetune_v2/round3_code_freeze.json` and `pilot_v3_run01_runner_freeze.json` stay valid. If subclassing turns out to be impractical, fall back to a full copy under the new name, with a recorded diff against v2. |
| Verifier 3.1 | `scripts/finetune_v2/fixtures3_1.py`, `verifier3_1.py` | Adds the `fold_traps` fixture family (§B.4.3). Same independence rule as before: it must not import any converter module. |
| Type contract | `scripts/finetune_v2/type_contract.py`, `data/interim/finetune_v2/type_contract_v1.json` | §A.4.1 |
| Atlas contract check | `scripts/nl2mql/atlas_type_contract_check.py` | §A.4.2 |
| Source-label audit | `scripts/source_label_audit_v1/detectors.py`, `run_audit.py` | Part D; read-only over the pool. |
| Runner | `scripts/finetune_v2/run_pilot_v4.py` | Copies the structure of `run_pilot_v3.py`, but imports `sql2mql_v2_1` and `verifier3_1`, records the TC-1 manifest sha256 per label, and has its own runner freeze file. |
| Code freeze | `scripts/finetune_v2/freeze_v2_1_code.py` writing `data/interim/finetune_v2/v2_1_code_freeze.json` | Refuses to overwrite. Pins every new module plus the frozen v2 modules it imports. |

### E.2 New cleaned dataset version

- **Name:** `spider-pool-v3`, under `data/processed/spider_pool_v3/`.
- **Built by:** `scripts/spider_pool_v3/prepare_spider_pool_v3.py`, from `spider_pool_v2` files referenced by sha256 and **never modified**.
- **Rules:**
  - Retained records keep their original line bytes and source order, as in v2.
  - New exclusion tiers:
    - `A_semantic_defect_adjudicated`: records that the S10 AI-panel adjudication confirms as defects. Candidates are R084, R101 and R111, their siblings 02407, 05404 and 04710, and D1a hits. **Until adjudicated, every candidate sits in `C_semantic_review`.** The earlier audit's verdicts alone do not place a record in tier A;
    - `C_semantic_review`: the 6 medium findings, their siblings, and every D2 to D6 hit that the panel has not cleared. Under the asymmetric rule (§C.3), anything the panel does not clearly accept stays here.
  - No corrected labels in this revision (§C principles). Candidate corrections stay in the adjudication log only.
  - `manifest.json` records every input hash, the detector pre-registration hash, and counts per tier.
- `data/processed/spider_pool_v2/*`, `spider_pool.zip`, `outputs/spider_pool_audit_v1/` and the pilot run01 outputs stay untouched. They are verified by hash before and after each step (§E.6).

### E.2b The existing provisional corpus and its successor

**Current state of `outputs/sql2mql_v2/provisional_corpus_v1/`:**
- converter `sql2mql-v2`;
- 3,962 accepted;
- 3,589 in `provisional_rag_corpus.jsonl`;
- status PROVISIONAL;
- used as the retrieval corpus of MQL-0007.

**Affected records found there** (grep of the item ids, 2026-11-12):

| Record | Finding | In `provisional_rag_corpus.jsonl` |
|---|---|---|
| `spider_train_06342` | R040, LIKE | yes |
| `spider_train_04935` | R082, LIKE | yes |
| `spider_train_00515` | R090 | yes |
| `spider_train_00516` | sibling of R090 | yes |
| `spider_train_02183` | sibling of R009 | yes |
| `spider_train_02184` | R009 itself | no: already in `excluded_accepted.jsonl`, reason `pilot_cross_review_UNCERTAIN` |

The other findings and siblings are not in the corpus.

**Plan:**
- **`provisional_corpus_v1` is never modified, and MQL-0007's results stand as recorded.**
  - R040 and R082 return correct rows on the corpus's data (M3), so the retrieval examples they supplied were not wrong on that data.
  - Their MQL is only wrong under TC-1's wider value domain.
- **New `provisional_corpus_v2`**, at `outputs/sql2mql_v2_1/provisional_corpus_v2/`:
  - built by a new runner from spider-pool-v3 with converter v2.1 and verifier 3.1;
  - excludes every record in a quarantine tier, including `C_semantic_review`;
  - contains no corrected labels.
- **Its manifest records:** the v2.1 code freeze, the TC-1 manifest sha256, the spider-pool-v3 manifest sha256, the detector pre-registration sha256, and a diff of ids against corpus v1 with a reason per id (added, removed, MQL changed).
- **Expected diff:**
  - the LIKE records keep their ids, and only their regex text changes;
  - R090, its sibling 00516 and R009's sibling 02183 are removed pending adjudication;
  - other removals come from detector hits.
- **Status stays PROVISIONAL until G-P2' passes.**
- Any experiment that switches from corpus v1 to v2 is a new ledger entry, never a re-score of MQL-0007.

### E.3 New automated tests (all new files under `tests/`)

1. `test_finetune_v2_1_like.py` and `test_finetune_v2_1_like_engine.py`, described in §B.4.
2. `test_type_contract.py`. Fixtures:
   - a homogeneous int column gives `int`;
   - int+real gives `num`;
   - text in a DATETIME column gives `text` with `declared_mismatch=True`;
   - int+text gives `nonconformant`.

   It also checks that the manifest reproduces M1 counts on the real databases (marked slow), and that the census comparison fails on an injected string in a `num` field, a double in an `int` field, a bool, an extra field, and a count mismatch.
3. `test_atlas_type_contract_check.py`. It uses a fake client, as in the existing `test_nl2mql_dev1034.py` pattern; there is no network access. It must fail closed on each violation type and on a missing collection, and it must never print the URI.
4. `test_source_label_detectors.py`. It runs each detector on the 120 review items and asserts the §D7 table: every must-detect item is flagged, and R084 is not flagged by D2. It also reports, without asserting, each detector's flags among the 110 audit-accepted items.
5. `test_finetune_v2_1_regression.py`. Implements §E.4.
6. A non-overwrite test asserting that every frozen file still has its recorded hash: those listed in `round3_code_freeze.json`, `pilot_v3_run01_runner_freeze.json`, `corpus_v1_runner_freeze.json`, `pilot_freeze.json`, the spider_pool_v2 manifest (output files and `spider_pool.zip`) and `manual_human_review/source_hashes_before.json`, plus the contents of `outputs/sql2mql_v2/provisional_corpus_v1/` and `outputs/sql2mql_v2/pilot_v3/run01/`.

### E.4 Regression against the 120 review items and the pilot run

- **Run.** Convert and verify the 300 pilot records with v2.1 and verifier 3.1, writing to a **new** directory `outputs/sql2mql_v2_1/pilot_regression/`.
- **Pass criteria:**
  1. Every non-LIKE label among the 213 accepted is **byte-identical** to the v2 label: same collection, same pipeline JSON. This criterion is feasible because the run01 reproducibility re-run (`run01/reproducibility_diff_analysis.json`) found `mql_identical` 300/300, `outcome_identical` 300/300 and equal accepted sets. The only differences were the V6 row-order note in 22 records and timings. Compare v2.1 with run01 excluding timings and `verification.checks.v6_determinism`. As an environment control, also re-run frozen v2 in the same session; it must reproduce run01 the same way.
  2. The 8 LIKE labels differ only in regex text and options. Their results equal the v2 results on the real data, under both NULL representations.
  3. The accepted and rejected sets are identical to v2's: 213 accepted, 87 rejected. The one rejected pilot LIKE item, `spider_train_03583`, is rejected as `not_a_single_select` because it uses UNION. That reason is independent of LIKE, so no exception is needed.
  4. Verifier 3.1 **rejects** the v2 MQL of R040 and R082 and **accepts** their v2.1 MQL.
  5. The Kelvin and long-s counterexamples from §B.4 give identical SQLite and MongoDB results.
- **Report.** Output from the 120 audit items, with source-label status and translation status in **separate columns**:
  - translation under TC-1 is expected to go from 118/120 to 120/120;
  - the 10 source findings stay as they are, since a converter change cannot fix them.
- **Caution.** The 120 items were used to find these defects, so this regression proves the fixes. It is **not** an unbiased estimate of the v2.1 error rate. That needs a fresh sample (§E.5).

### E.5 Acceptance gates for v2.1 labels and spider-pool-v3

| Gate | Requirement | Evidence |
|---|---|---|
| **G-TC (contract)** | `type_contract_v1.json` reproduces M1. `check-local` passes on `spider_train_v2` and `_missing`. Every v2.1 label records the manifest sha256. | contract report |
| **G-LIKE** | §B.4 tests pass locally. The Atlas server check (§B.4.4) passes before any Atlas-scored use. | test logs, Atlas check JSON |
| **G-REG** | All §E.4 criteria pass. | regression report |
| **G-SRC (source labels)** | Detectors pre-registered. Recall 100% on §D7 must-detect items. Precision reported per detector with a Wilson interval from AI-panel-labelled samples. Every quarantined record traced to a detector or finding. | `hits.jsonl`, `prereg.json`, precision sample labels |
| **G-PRE (Atlas preflight)** | For every future Atlas-scored experiment: read-only privileges, collection coverage, exact counts, **TC-1 `$type` census pass at scoring time**, and full digest parity at least once per experiment. Required as an input to scoring. | `atlas_type_contract_<ts>.json`, `atlas_preflight_<ts>.json` |
| **G-AIR (fresh AI-panel review; replaces the human P2 for this revision)** | A **new** seeded, stratified sample of accepted v2.1 labels from spider-pool-v3. It excludes the 300 pilot records and the 500 DEV records, and is drawn before anyone sees its outputs. Planted defects are mixed in blind (§E.7.3). The §E.7 panel reviews both dimensions separately. | frozen review package, panel outputs, planted-defect key |

**Reporting for G-AIR:**
- Source-label accuracy and translation correctness are reported **separately**: `wrong_source / reviewed` and `wrong_translation / reviewed`, each with a Wilson 95% interval.
- **Proposed (user decision, §H.3):** apply the P2 threshold (Wilson upper bound ≤ 0.05) to **translation correctness**, as a converter gate. G-AIR also needs the panel's measured sensitivity to meet §E.7.3. Without that, a low AI-flagged error rate is not evidence.
- Source-label accuracy is a dataset-quality metric with its own threshold. That threshold is **for the user to set**, before the sample is drawn.
- P3 (no `discrepancy_type` ≥ 2) is reported for each dimension.
- Note: with n = 120, the Wilson upper bound is 0.031 for 0 errors, about 0.046 for 1 error, and about 0.059 for 2 errors. **P2 tolerates at most one wrong label in 120.**
- **Run01 under the AI review.** If the AI audit is taken as run01's review, run01 **fails** the frozen P2 threshold on translation alone: 2/120 gives a Wilson upper bound of 0.059, which is above 0.05. Counting source errors (10/120, upper bound 0.147), it fails by a wide margin. P3 also fails. This is why the revision is needed. The run01 AI audit is not re-labelled as a pass.
- **Sample size.** Because even 2 errors in 120 fail, G-AIR should use n ≥ 200. With n = 200, the Wilson upper bound is about 0.019 for 0 errors, 0.028 for 1, 0.036 for 2, and 0.043 for 3. Fix n before drawing the sample.

### E.6 Preservation of frozen artifacts (applies to every step)

- **Before and after every step**, re-hash:
  - the files in `round3_code_freeze.json`, `pilot_freeze.json`, `pilot_v3_run01_runner_freeze.json` and `corpus_v1_runner_freeze.json`;
  - `data/processed/spider_pool_v2/*` and `spider_pool.zip`;
  - `manual_human_review/source_hashes_before.json`;
  - `outputs/sql2mql_v2/provisional_corpus_v1/`;
  - `outputs/sql2mql_v2/pilot_v3/run01/`.

  Any change stops the work.
- No new output is written inside an existing frozen output directory.
- Never open `data/splits_expanded/test.jsonl`, Spider dev or test, or the zip's `dev.jsonl`.
- **Ledger.** Each quantitative step (the contract census, detector audit, regression and G-P2') gets its own entry in `docs/experiment_accuracy_ledger.md` under dataset audits, plus a line in `docs/presentation_notes.md`, as CLAUDE.md requires. Each entry states local versus Atlas.

---

### E.7 The AI review panel (replaces human review in this revision)

The user's decision is to use multi-agent AI review instead of human review. The run01 audit showed what such a panel can do: it found 10 source errors that the verifier cannot see. It also showed its weaknesses: the same model throughout, a leaked summary count, and no measured error rate for the reviewers themselves. This section turns the panel into a measured instrument instead of four agents producing confident-looking labels.

**E.7.1 Roles.** Same as run01: source auditor, translation auditor, adversarial end-to-end auditor, and an independent adjudicator who does a blind first pass. Every NO cites logged execution evidence through the read-only executor (`ai_multiagent_review/tools/ro_exec.py`, reused unchanged). No majority vote.

**E.7.2 Model diversity.**
- At least one reviewer per dimension should come from a **different model family** than the others, if one is available to the project. Earlier project reviews used Gemini.
- If no second family is available, record that, and weight the §E.7.3 sensitivity result accordingly.

**E.7.3 Planted-defect calibration (the core safeguard).**
- **Build the planted set** with `scripts/ai_review_panel_v1/plant_defects.py`. It takes correct accepted labels and plants one known defect per copy:
  - source-side: flipped sort direction; a literal moved to a sibling column; a dropped DISTINCT; INNER instead of LEFT JOIN on an "all X" question; COUNT(*) instead of COUNT(DISTINCT); a truncated question;
  - translation-side: reuse the existing mutant operators in `scripts/finetune_v2/mutants_v2.py`, imported read-only.
- **Hygiene.** Planted items are evaluation controls only. They are never written to any dataset or corpus, carry ids that cannot collide with real ones, and their key is kept out of every agent's reach.
- **Run blind.** Mix planted items into each review batch at about 15%, with the batch order seeded.
- **Pass criteria (proposed; thresholds are the user's decision, §H.3):**
  - per dimension, on at least 60 planted items, a Wilson 95% lower bound on sensitivity ≥ 0.80. With 60 items that means at least 55 caught; 54/60 gives 0.799;
  - no defect class below 0.75.
- **Coverage.** Translation-side calibration should also include the two classes the run01 audit found (Unicode regex folding, untyped `$gt` on text), as known positives.
- **Scope of the evidence.** Sensitivity measures planted defect types only. Natural defects outside those types stay unmeasured, and every report says so.

**E.7.4 Isolation, enforced and checked.**
- **Memory index.** Before each panel run, scrub the auto-loaded memory index of verdict counts or item ids. Run01 leaked "6/120" through it. Record the index hash used.
- **Prompts.** Prompt files are hashed in the pre-registration. No prompt contains another agent's output or an earlier verdict.
- **Disclosure.** Each agent reports a `context_disclosure`. Any disclosure of prior verdicts marks that agent's output as non-independent.
- **Workspaces.** Each agent writes only to its own `_work/agent_N/`. The orchestrator checks for cross-reads afterwards, where tool logs allow it.

**E.7.5 Evidence audit and stability.**
- Re-execute every log entry cited by a NO or by an adjudication, as `respot.py` did in run01.
- Re-run the full panel on a seeded 20% subset with a different item order, and report verdict agreement (Cohen's κ per dimension).
- Low stability (κ < 0.6) blocks G-AIR.

**E.7.6 Labelling and protocol record.**
- Every artifact says "AI-reviewed (multi-agent panel)".
- Add a ledger entry recording the protocol change: the human P2 is replaced by G-AIR for sql2mql-v2.1 and spider-pool-v3, with the user's decision date. The frozen `pilot_freeze.json` is not edited; the change is recorded beside it.
- Record the decision log (§H.3) in `data/interim/finetune_v2/v2_1_decisions.json`, append-only.

**E.7.7 Why this is weaker than human review, and what limits the damage.**
- **What remains:** errors the panel cannot recognise, and that the planted set does not represent, can still pass.
- **What limits the damage:**
  - the asymmetric rule turns panel uncertainty into quarantine, not acceptance;
  - the TC-1 contract and the verifier remove whole classes of translation error mechanically;
  - the planted-defect rate gives a measured, if partial, sensitivity.

---

## F. Ordered implementation sequence

| Step | Work | Depends on | New paths | Expected outputs | Acceptance |
|---|---|---|---|---|---|
| S0 | Hash baseline of every frozen artifact (§E.6) | none | `outputs/pilot_v3_followup/hash_baseline.json` | baseline JSON | all recorded hashes match |
| S1 | Type contract builder and local census | S0 | `scripts/finetune_v2/type_contract.py`, `tests/test_type_contract.py` | `data/interim/finetune_v2/type_contract_v1.json`, local census report | reproduces M1 (3,824 / 115 mixed / 199 text-in-numeric); `check-local` passes |
| S2 | Atlas contract check (code and fake-client tests first, live run second) | S1 | `scripts/nl2mql/atlas_type_contract_check.py`, its test | fake-client tests green; live report on `codegen_spider1` | live run reproduces M2 (2 mixed `wta_1` fields) or reports the difference; fails closed on injected faults in tests |
| S3 | Converter v2.1 LIKE fix and unit tests | S0 | `sql2mql_v2_1.py`, `tests/test_finetune_v2_1_like*.py` | test logs | §B.4 tests 1–2 pass; frozen v2 tests still pass unchanged |
| S4 | Verifier 3.1 with fold-trap fixtures | S3 (for the negative control) | `fixtures3_1.py`, `verifier3_1.py` | fixture tests | v2 R040/R082 MQL rejected; v2.1 MQL accepted |
| S5 | Code freeze v2.1 | S1–S4 | `freeze_v2_1_code.py`, `v2_1_code_freeze.json` | freeze file | refuses to overwrite; check passes |
| S6 | Pilot regression (§E.4) | S5 | `run_pilot_v4.py`, `outputs/sql2mql_v2_1/pilot_regression/` | regression report and ledger entry | all §E.4 criteria |
| S7 | Detector pre-registration, then run on the 120 items | S1 | `scripts/source_label_audit_v1/*`, `tests/test_source_label_detectors.py` | `prereg.json`, recall report | 100% must-detect recall (§D7) |
| S8 | Detectors on spider_pool_v2 and sibling propagation | S7 | `outputs/source_label_audit_v1/hits.jsonl` | hit counts per detector | every hit has evidence; no records modified |
| S8b | AI-panel calibration (§E.7.3): pre-register the protocol, build the planted-defect set, measure sensitivity | S7 | `scripts/ai_review_panel_v1/`, `outputs/ai_review_panel_v1/calibration/` | sensitivity per defect class with Wilson CI | §E.7.3 thresholds met before the panel's labels are used anywhere |
| S9 | Precision sampling of detector hits, labelled by the AI panel | S8, S8b | `outputs/source_label_audit_v1/precision_sample.jsonl` | precision per detector with Wilson CI | panel labels with evidence ids; user's "all X" policy applied |
| S10 | AI-panel adjudication of the §C items and their siblings (blind re-review) | S8b | `data/interim/source_label_adjudication_v1.jsonl` | adjudication log with candidate corrections, none applied | every item has panel verdicts with execution evidence; asymmetric rule applied |
| S11 | Build spider-pool-v3 | S8–S10 | `scripts/spider_pool_v3/prepare_spider_pool_v3.py`, `data/processed/spider_pool_v3/` | pool v3 and manifest | v2 untouched; counts traceable per tier |
| S11b | Build `provisional_corpus_v2` (§E.2b) | S5, S11 | `outputs/sql2mql_v2_1/provisional_corpus_v2/` and its runner | corpus, manifest, and id diff against corpus v1 | corpus v1 hashes unchanged; every removed or changed id has a reason |
| S12 | Fresh pilot on pool v3 with v2.1, then the G-AIR review | S6, S8b, S11 | `outputs/sql2mql_v2_1/pilot_v4/` | frozen review package, panel outputs, G-AIR report and ledger entry | sample and planted defects frozen before outputs; G-AIR computed |
| S13 | Atlas LIKE check and TC-1 census before any Atlas-scored use | S2, S6 | preflight outputs of that experiment | Atlas reports | G-LIKE and G-PRE |

**Parallelism:** S1–S2, S3–S4 and S7–S8 are independent tracks. No step needs a human reviewer. The user's decisions in §H.3 are needed before S9 (the "all X" policy) and S12 (thresholds and n).

---

## G. Risks

| Risk | Effect | Mitigation |
|---|---|---|
| Subclassing `Converter` reaches into private state that changes `like()` behaviour elsewhere | v2.1 differs from v2 outside LIKE | §E.4 criterion 1 (byte-identical non-LIKE labels over 300 records) catches this; fall back to a full copy |
| The PCRE version on Atlas differs from local | the LIKE fix is unverified on Atlas | §B.4.4 server check before Atlas use |
| The read-only Atlas user cannot run `$documents` | the check cannot run as written | `$literal` in `$project` fallback over an existing collection |
| Detector false positives quarantine many good records (D3, D4 especially) | training set shrinks | precision measured before quarantine is final; the user's "all X" policy; hits stay in `C_semantic_review`, not deleted |
| Detectors miss interpretation errors (R009-type) | residual source errors | G-AIR panel review with separate source-label reporting |
| **The AI panel shares blind spots: same model family, same training** | errors the panel cannot see pass as correct, and agreement looks stronger than it is | planted-defect sensitivity (§E.7.3); a second model family where available (§E.7.2); asymmetric rule; residual risk disclosed in every report |
| Earlier findings leak into the panel (as the MEMORY.md line did in the run01 audit) | anchoring | §E.7.4 isolation checks; a scrubbed memory index before each panel run; disclosure per agent |
| Prompt or threshold tuning after seeing panel results | inflated pass rate | protocol, prompts and thresholds hashed in the pre-registration before any panel run |
| Treating the 120 as a test set after using them for development | over-optimistic quality claims | §E.4 caution; G-P2' uses a fresh sample |
| The Atlas evaluation collections drift after 2026-10-20 | mismatch with SQLite gold | TC-1 census at every scoring run (G-PRE) |
| The provenance of R097's absence from spider_pool_v2 is unclear | pilot and pool populations not aligned | re-apply the pool builder's rules to that single record. Never open the zip's `rejected.jsonl` or `dev.jsonl`. |
| Reviewers report that AI review is not accepted as a substitute for the human P2 gate | the corpus quality claim is not accepted | every artifact is labelled AI-reviewed; the protocol change is recorded (§E.7.6); the planted-defect sensitivity is the evidence offered in its place |

---

## H. Established, AI recommendation, or needs approval

### H.1 Already established (reproduced execution or measurement evidence)

- **R040 and R082:** the MQL `$regex` with options `is` over-matches SQLite LIKE for values containing U+212A or U+017F. This was reproduced in the audit, and again here (M4) on SQLite 3.40.1 and MongoDB 8.0.34 locally.
- **v2's pattern guard has a second hole:** non-letter characters with case pairs (U+24B6/U+24D0, U+2160/U+2170) pass its `isalpha()` check and are over-matched under `i`. This was reproduced locally. No current pattern is affected.
- **The proposed ASCII-class translation** (options `s`, no `i`) matched SQLite on every pattern and value tested here, in both the `$regex` and `$regexMatch` forms (M4). This is local evidence only; it is not yet tested on the Atlas server.
- **Census results** (M1–M3, local SQLite and the stored Atlas mapping): 115 mixed-class columns in the training databases, 2 mixed fields in the evaluation mapping, 199 homogeneous-text numeric-affinity columns, and 0 fold-trap values.
- **Preflight behaviour:** the existing Atlas preflight digest is type-tagged. `local_atlas_scoring.py score` defaults to `full_parity=False`.
- **Frozen hashes:** `sql2mql_v2.py` and `schema_profile.py` are pinned by hash in the round-3 and pilot runner freeze files.
- **Pool membership:** 9 of the 10 source findings have one same-SQL sibling in spider_pool_v2. R097's record is not in spider_pool_v2.
- **Corpus membership:** `provisional_corpus_v1/provisional_rag_corpus.jsonl` contains R040, R082 and R090, plus the siblings 00516 (of R090) and 02183 (of R009).
- **Run01 reproduces:** its re-run had identical MQL for 300/300 records.
- **The 18 strict-domain-only discrepancies** need text values in numeric-class fields. On real data, all 18 MQL results equal the SQL results (audit execution evidence).

### H.2 AI recommendation only (not independently validated)

- **The 10 source-label findings.** They are AI audit judgements. Execution evidence backs the facts, for example R097 returns 0, but "wrong for the question" is a semantic judgement. They are confirmed or cleared only by the blind §C.3 panel re-review. Quarantine is the default, so an AI error here costs a few records, not correctness.
- **Whether the AI panel's verdicts are reliable at all.** This is measured, not assumed, through planted-defect sensitivity (§E.7.3). Defect types the planted set does not cover remain unmeasured.
- The TC-1 definition based on observed storage class, instead of declared affinity.
- The decision not to emulate SQLite mixed-type semantics.
- The LIKE fix design: subclass, new versions and keeping the guards.
- The per-item actions in Part C (quarantine, adjudicate or correct), the tier assignments, and the sibling propagation.
- The detector designs, word lists, thresholds and must-detect sets in Part D.
- The gate structure in Part E, including applying the P2 threshold to translation correctness only.

### H.3 Decisions the user makes (no human reviewer needed)

These are policy choices, not per-item review. Make them before the step that uses them, and record each with its date in the plan's decision log (§E.7.6).

1. **Adopt TC-1** as the project's official data contract, and record it in `docs/nl2mql_next_steps.md`.
2. **The "all X" policy** (R010, R090, and D3 pool-wide): include zero-count entities, or accept Spider's convention. If undecided, the asymmetric rule quarantines these records.
3. **Thresholds, fixed before sampling:**
   - the source-label accuracy threshold;
   - whether the P2 bound applies to translation only;
   - the G-AIR sample size;
   - the §E.7.3 panel-sensitivity thresholds.
4. **Replacing the human P2 gate with G-AIR.** This is a protocol change to a gate frozen on 2026-10-30. Record it as a dated deviation (§E.7.6). If the project's results are reported to a mentor or supervisor, tell them, because the gate they saw defined was a human review.
5. **Any change to evaluation scope**, for example slicing or excluding the 2 mixed `wta_1` fields. This changes reported model metrics, so the same disclosure applies. Earlier results (MQL-0006, MQL-0007) are never changed.
6. **An optional Atlas `$jsonSchema` validator.** It needs admin rights through a separate account.
7. **Whether candidate corrections are ever applied.** That would be a later, separate step, outside this revision.
