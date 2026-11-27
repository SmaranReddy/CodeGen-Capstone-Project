# Reproducibility audit of `spider_pool.zip`

Audit date: 2026-11-05. **Audit only: no dataset was cleaned, repaired or filtered, `spider_pool.zip` was not regenerated, and the converter, verifier-3 and frozen pilot artifacts were not touched. Nothing was committed.** These are SQL-side dataset-audit figures, measured locally. None is an NL→MQL model metric.

Follows `docs/spider_pool_independent_audit.md`. Evidence: `outputs/spider_pool_repro_audit_v1/` (start with `audit_summary.json`).

## 1. Summary

| Question | Answer |
|---|---|
| Is the cleaning logic correct? | **Correct for what it implements, but insufficient as cleaning.** Rules R1–R9 do what they say. It has no schema, date, semantic or split-integrity checks, and only partial nondeterminism and NULL checks |
| Are the data files reproducible? | **Yes.** All 4 data files are byte-identical; 0 record mismatches in 9,604 |
| Is the ZIP itself reproducible? | **No, not by this script.** It never creates a zip or the README. Only their contents could be checked |
| Is it portable? | **No.** Windows changes every file hash, and SQLite 3.40.1 changes the kept set (5,518 train rows) |
| Are Tiers A, B and C handled? | **No.** All 136 rows are still kept (57 + 37 + 42). The builder has no logic that could catch them |
| Is the output free of protected data? | **Train and val: yes, at ID and database level. Whole output: no.** It holds all 1,034 Spider dev records, including all 65 held-out items |
| What do 5,430 / 734 mean? | Tier A + Tier B removed, Tier C kept. These files do not exist anywhere; the builder produces 5,514 / 744 |
| Is the training source safe for V2 conversion? | **Yes, conditionally**: `train.jsonl` only, with the conditions in section 8 |

## 2. Which script was audited

**The file named in the request is not the builder.**

| File | sha256 | Finding |
|---|---|---|
| `build_spider_pool.py` (repo root, untracked) | `45e1e28e50c56b91…298c6a` | **The builder.** Its full hash equals the delivered manifest's `code_sha256` |
| `scripts/spider1/build_pool.py` (tracked since October) | `9ac842c5…fc8163` | **A different script.** It builds the dev candidate pool and the 300-item sample. It parses `dev.json` gold SQL and writes to `data/interim/spider1` by default, so it was read but **not run** |

**Safety check before running** (`stage1_static_safety.json`): standard-library imports only, no process, network or delete calls, no hard-coded record IDs, read-only database connections, and writes only under `--out`.

**Configuration and dependencies:** the only configuration is the command line (`--raw`, `--sqlv1`, `--out`, `--timeout`). There is no requirements file. The real dependencies are the Python `sqlite3` module and the OS's text-mode line endings, and neither is pinned.

## 3. Reproduction

The unmodified script ran on private copies of the original inputs, with explicit temporary output folders. All input checks passed, and on Linux the database copies still matched the originals afterwards.

| Environment | train / val / dev / rejected | Data files vs delivered | Record differences |
|---|---|---|---|
| **Linux (WSL2), Python 3.12.3, SQLite 3.45.1** | 5,514 / 744 / 899 / 2,447 | **4 of 4 byte-identical** | **0** |
| Windows, Python 3.13.7, SQLite 3.50.4 | 5,514 / 744 / 899 / 2,447 | Identical only after converting CRLF to LF | 0 |
| Windows, Python 3.10.11, SQLite 3.40.1 (the project default) | **5,518** / 744 / 899 / **2,443** | Different | 4 rows change side; values differ in 56 train/val and 10 dev rows |

**Comparison with the delivered files, Linux run:**

| File | sha256 (delivered = reproduced) | Delivered | Reproduced | Only delivered | Only reproduced | Mismatches |
|---|---|---:|---:|---:|---:|---:|
| `train.jsonl` | `7443cd8e…3c37b` | 5,514 | 5,514 | 0 | 0 | 0 |
| `val_internal.jsonl` | `5972f2f8…04b30` | 744 | 744 | 0 | 0 | 0 |
| `dev.jsonl` | `3a6ed708…4b11f` | 899 | 899 | 0 | 0 | 0 (ID and line hash only) |
| `rejected.jsonl` | `b04ed7f6…cf4a1b` | 2,447 | 2,447 | 0 | 0 | 0 |

- **Checked field by field for train and val:** ID, split, database, question, raw question, SQL, hardness, class, features, group, order, limit, tie blocks, columns, row count, expected output, result hash and source. Row order also matches. Per-record results are in `records/record_level_comparison.jsonl` (8,705 train, val and rejected rows, all matching).
- **Exclusions:** every rejected row has the same ID, split, database and `reject_reason`. The funnel recomputed from the files equals the manifest.
- **`manifest.json`:** exactly 1 key differs, `sqlite_version` (3.53.1 delivered, 3.45.1 here). Substituting that one value makes the file byte-identical to the delivered one, which also confirms all input, code and 159 database hashes.
- **Repeat runs:** the script ran 6 times on Linux, across 3 WSL sessions. 5 of those runs were confirmed identical to the delivered data files. The sixth run's check output was not displayed, so it is not counted.

**What could not be reproduced or checked**

- **SQLite 3.53.1 was not run.** Downloading and compiling it was blocked by this session's permission policy. So the recorded version is unconfirmed, and the delivered manifest was only matched by substitution. This does not affect the data: the delivered files equal the 3.45.1 output byte for byte.
- **The ZIP's own hash** (`31848241…`) has no generator: the builder imports no zip library and writes no README. Regenerating it was also out of bounds. The zip's timestamps show the README was added about 36 seconds after the data files were written.
- **The README was written separately.** Its tables match the reproduced numbers, but nothing ties it to the build.
- **Running the author's script is not an independent derivation.** It shows the delivered files came from this code and these inputs, not that they are right. The earlier audit's separate re-implementation of R1 and R4–R7 is what supports those rules; the R8 replay here reuses the builder's own function.

**SQLite version sensitivity**

- The 4 rows that move on 3.40.1 are the R3 rejects `spider_train_00547`, `00548` (`store_1`), `05935`, `05936` (`cre_Theme_park`). This supports the README's explanation that they are version differences, not data errors.
- The other differences are floating-point noise of at most 1.49e-08 in train and val, with no change of row count or non-numeric value. The 10 dev rows were compared by line hash only, so their cause was not examined.
- Versions 3.45.1 and 3.50.4 give the same records as the delivered files. Versions in between were not tested.

## 4. Audit of the cleaning logic

Assessed from the code and from 20 synthetic-database tests of the real functions (`stage4_logic_tests.json`; all behaved as read from the code). No test used project data.

| Area | What the code really does | Assessment |
|---|---|---|
| **Duplicates** | R9 only: same database and normalised question, first kept wins. Paraphrases of the same SQL are kept on purpose | **Correct but narrow.** It rejected 2 rows. A hyphen variant slips through the key, and non-ASCII letters are stripped |
| **Conflicting labels** | **No check.** R9 never compares SQL, so a real conflict would be settled by file order | **Gap, with no harm here.** Of 8,570 train/val inputs, 1 question had differing SQL: `spider_train_03525` and `03463`, with the same result. No two kept rows conflict |
| **SQL validity** | Runs the query on a read-only connection (R2). No parser, no SELECT check | **Adequate.** 0 rejections; write statements fail |
| **Schema checks** | **None.** `tables.json` is hashed and never parsed | **Absent.** The earlier audit's 0 broken references is what covers this |
| **Empty databases** | R1 trusts the `tier` field from spider-sql-v1 | **Correct on this data** (1,744 confirmed earlier), but not recomputed by the builder |
| **NULL handling** | R4 rejects empty or all-NULL results; R5 rejects a single 0 | **Correct but narrow.** Partly-NULL results pass, and blank strings count as values |
| **Dates** | **No logic at all** | **Absent.** Text date subtraction passes (Tier A: 6 rows) |
| **Nondeterminism: top-level** | R7 rejects a tie at a top-level `ORDER BY … LIMIT`. R8 rejects order it cannot verify | **Correct and conservative.** All 1,246 kept ordered rows have verified tie blocks |
| **Nondeterminism: other** | **Not checked:** nested LIMIT, bare `GROUP BY` columns, aggregate plus bare column, `LIMIT` without `ORDER BY` | **Gap.** This is why Tier B survives. `LIMIT` without `ORDER BY` has 0 kept rows at top level |
| **Split integrity** | **No check.** Disjointness is inherited from spider-sql-v1 | **Holds, but unenforced.** 0 ID, 0 database and 0 query overlap between train, val and dev |
| **Answer correctness** | **Nothing.** Hashes compare the SQL with itself | **Out of its reach**, which is why Tier A survives |

**R8 is now replayed**, closing a gap in the earlier audit: all 59 train/val rows are confirmed (40 `DISTINCT`, 19 where the keyed query returned different rows). The 2 dev rows were left alone.

**Claims that do not hold as written**

| Claim | Finding |
|---|---|
| Inputs "both hash-checked, the build aborts on any mismatch" / "checks every input against the pinned hashes" | **Overstated.** Hashes are compared only with the manifest in the input folder. A forged input folder with a matching manifest was accepted (synthetic test). SQLite files are recorded, never checked. The `spider_data.zip` hash is only a literal written to the output, though I confirmed it matches the local zip |
| "Output is byte-for-byte reproducible" | **True only on an LF platform with a compatible SQLite.** Text-mode writes produce CRLF on Windows |
| Reproduce command: `python scripts/build_spider_pool.py` | **Wrong path here.** The script is at the repo root |
| "Rules, applied in this order" | **Minor.** The code tests R8 before R7. This cannot change a reason, because a tie is only found when the analysis succeeds |
| "The test split is never read" | **True for content.** `test.json` and `test_tables.json` are hashed only |

Also harmless on this data: `group` folds case and spacing inside string literals, which could merge different queries. 0 groups have more than one result, and the 2 groups with two spellings differ only in letter case.

## 5. Tier A, B and C

**None of the tiers is handled by the builder.** They were identified after the zip was built, and the script contains no exclusion list and no rule that could catch them.

| Tier | Rows | Train / val | Still kept in reproduced output | Rejected | Why the builder misses them |
|---|---:|---:|---:|---:|---|
| **A: confirmed defects** | 57 | 53 / 4 | **57** | 0 | They run and return stable, non-trivial answers. No rule compares SQL with the question |
| **B: row-choice dependent** | 37 | 31 / 6 | **37** | 0 | R7 only looks at the top-level `ORDER BY`. Bare columns and nested ties are not examined |
| **C: human review** | 42 | 42 / 0 | **42** | 0 | Many-to-many joins, blank strings, unmatched literals and empty tables are not examined |

- The tiers do not overlap, and every row's recorded split and database match the reproduced output. Row status: `records/tier_status_in_reproduced_output.jsonl`.
- The list is a recommendation from one audit and is not ground truth. Nothing has been applied to any file.

**Correction to my earlier audit: Tier C is incomplete by one row.** The earlier empty-table screen missed rows that repeat an SQL string. The most likely cause, not directly confirmed, is SQLite's statement cache suppressing the authorizer callback it relied on (3,876 earlier records have no captured reads). A fresh-connection recheck found 4 rows rather than 2 (`stage9_empty_table_recheck.json`):

- `spider_train_02208` (`formula_1`) has the same SQL as Tier C row `02207`, but is in no tier. It should get the same human decision.
- `spider_train_02945` was missed by the screen but is already in Tier C for another reason.
- This was the only check using that capture. Tiers A and B have no paraphrase left outside the tier, so 5,430 / 734 are unaffected.

## 6. Reconciling 5,430 and 734

Recounted from the reproduced files, not only by subtraction (`stage5_tiers_and_data_checks.json`):

| Exclusions applied | Train | Val |
|---|---:|---:|
| None (what the builder and the ZIP contain) | **5,514** | **744** |
| Tier A | 5,461 (−53) | 740 (−4) |
| **Tier A + Tier B (the reported figures)** | **5,430** (−31) | **734** (−6) |
| Tier A + B + C | 5,388 (−42) | 734 (−0) |

**5,430 and 734 account for Tier A and Tier B only. Tier C is not removed from them.** Removing Tier C changes only train, giving 5,388 (5,387 if `02208` follows `02207`). These are arithmetic results from a recommendation; no such files exist.

## 7. Protected data

ID-level checks only. Protected questions and gold queries were not viewed.

| Set | Size | Train + val kept | Whole output |
|---|---:|---:|---:|
| Spider dev | 1,034 | **0** | **1,034** (899 `dev.jsonl`, 135 `rejected.jsonl`) |
| Held-out MQL test | 65 | **0** | **65** (64 `dev.jsonl`, 1 `rejected.jsonl`) |
| Spider test | 2,147 | **0** | **0** (0 shared databases) |

- **The dev rows are there by design.** `SPLITS` includes `dev_full.jsonl`, and any faithful re-run recreates this exposure. So the reproduced output, like the ZIP, fails "contains no protected data".
- **Handling in this audit:** step 4 meant the builder itself had to parse dev data and execute dev gold SQL. Output was limited to counts. Dev rows were compared by ID, reason and line hash only, and no reproduced file was copied to the repo. All temporary output folders and input copies have been deleted: the WSL temp folder by the audit script, the Windows copy by hand. For the Spider test check, my script parsed `test_full.jsonl` in memory and kept only `id` and `db_id`.
- **Limit:** text-level overlap with dev and test was not measured, as that would mean reading protected fields. The zero database overlap is the strongest available evidence.

## 8. Remaining issues and V2 readiness

**Provenance**

| Item | Status |
|---|---|
| Builder missing (H1 in the earlier audit) | **Closed:** script obtained, hash matches, output reproduced |
| Builder not preserved | **Open:** `build_spider_pool.py` and `spider_pool.zip` are untracked. Neither is in version control or carries an author record |
| Build environment | **Open:** nothing pins the SQLite version or line endings, and the project's own Python 3.10 gets a different result |
| Input folder trust | **Open:** spider-sql-v1 is accepted on its own manifest. The hashes in the delivered manifest do match, and all 8,570 training-side rows were traced to raw Spider in the earlier audit |
| "145-example eval set" (M5) | **Open:** still unexplained, and it uses held-out `pets_1` |

**Data quality**

- The 57 Tier A and 37 Tier B rows remain in the delivered and reproduced files.
- The rate of defects **after** exclusions is unmeasured. The earlier hand sample (3 clear errors in 80, Wilson 95% 1.28–10.45%) describes the pool before exclusions, and came from one LLM reader.
- Screens have blind spots, and more wrong labels may exist. Tier B was not read row by row against its questions.
- SQL labels are not MQL labels. This audit says nothing about conversion quality.

**Is the training source safe for V2 conversion after exclusions?** **Yes, as a SQL source, under these conditions:**

1. Take only `train.jsonl`, and `val_internal.jsonl` for checkpoint selection. No dev-bearing file or row may reach any pipeline.
2. Exclude Tier A and Tier B, giving 5,430 / 734, through a versioned ID list applied at intake.
3. Have a person decide Tier C, including `02208`.
4. Use the delivered files or an LF build with a tested SQLite (3.45.1 or 3.50.4 for the records, as shown here). Do not regenerate on SQLite 3.40.1.
5. Keep treating converted labels as unverified until the pilot's checks and human review have run.

**This does not start or clear the pilot.** The earlier open items stand: your decision on the frozen 300 (276 are in the pool; I still recommend keeping the set frozen), a verifier-3 pilot runner, a named human reviewer, and your go-ahead.

## 9. Audit conduct

- **Run:** the unmodified builder, in isolated copies. **Not run:** `scripts/spider1/build_pool.py`, the converter, verifier-3, the pilot, G4, RAG, fine-tuning, MongoDB.
- **Round-3 records:** no gate statistic was computed on the 740 records.
- **The ZIP is unchanged:** sha256 `31848241861988ebb0619e1802c38fb5d90cfb62f818a65a53cf601340888a6b` before and after. Existing uncommitted changes are as they were.
- **Files changed outside the audit folder:** this report, plus appended entries in `docs/experiment_accuracy_ledger.md` and `docs/presentation_notes.md`, which `CLAUDE.md` requires. No earlier entry was edited.

## Verdict

**CONDITIONAL PASS**
