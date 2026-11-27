# spider-pool-v2 preparation report

Date: 2026-11-06. **Dataset preparation only.** No SQL→MQL conversion, verifier-3, pilot, fine-tuning, MongoDB, Atlas or Docker work was done. `spider_pool.zip` is unchanged (sha256 `31848241…0888a6b`), and nothing was committed.

**Do not read retention as approval.** A record kept in this dataset is not shown to have a correct SQL label. It only survived the audit's screens. Correctness still has to be shown through conversion verification and independent review.

## 1. Artifact

`data/processed/spider_pool_v2/` (version `spider-pool-v2`):

| File | Records | sha256 | Content |
|---|---:|---|---|
| `train.jsonl` | **5,387** | `09407ae4…495a7` | Retained training records |
| `val_internal.jsonl` | **734** | `3a93a3cd…dd13e` | Retained internal-validation records |
| `quarantine.jsonl` | **43** | `56735364…2182f` | Records awaiting a human decision; not for training |
| `exclusions.jsonl` | **137** | `c58f81f0…0b647` | One entry per removed record (94 excluded, 43 quarantined) |
| `manifest.json` | n/a | `cb80a118…38053` | Source hashes, exclusion provenance, rules, counts and output hashes |

**How records are stored**
- Retained and quarantined records are the **original bytes of their source lines**, in source order. No field was added, removed, normalised or repaired. Questions, SQL, IDs, `db_id`, `group`, `source` and expected outputs are exactly as delivered.
- Each `exclusions.jsonl` entry holds the ID, split, database, group, disposition (`excluded` or `quarantined`), tier, the audit's reason **verbatim**, the evidence file, and the source zip member, line number and line sha256.

**Code**
- Builder: `scripts/spider_pool_v2/prepare_spider_pool_v2.py` (sha256 `6ec696e4…9b433`). It refuses to run if the output folder exists, writes to a temporary folder and then renames it, and stops if the zip, the tier list or the recheck evidence differs from its pinned hash.
- Verifier: `scripts/spider_pool_v2/verify_spider_pool_v2.py`. It is separate code that does not import the builder.
- Verification result: `outputs/spider_pool_v2_prep/verification_report.json`.

**Sources and evidence**

| Input | sha256 | Use |
|---|---|---|
| `spider_pool.zip` | `31848241…0888a6b` | Only the `train.jsonl` and `val_internal.jsonl` members, each matched to the zip manifest's hash. `dev.jsonl` and `rejected.jsonl` are never opened by the builder |
| `outputs/spider_pool_audit_v1/records/recommended_exclusions.jsonl` | `008cae9c…0d5b3` | All 136 tiers and reasons, used as recorded |
| `outputs/spider_pool_repro_audit_v1/stage9_empty_table_recheck.json` | `6b633c3e…3a762` | The one added quarantine row, `spider_train_02208` |
| Both audit reports and `audit_summary.json` | in the manifest | Hashes recorded at build time |

## 2. Filtering rules

| Rule | Disposition | Train | Val | Total |
|---|---|---:|---:|---:|
| Tier A, confirmed defect | **Excluded** | 53 | 4 | 57 |
| Tier B, engine or row-choice dependent | **Excluded** | 31 | 6 | 37 |
| Tier C, review before use | **Quarantined** | 42 | 0 | 42 |
| `spider_train_02208`, added by the reproducibility audit | **Quarantined** | 1 | 0 | 1 |
| Every other source record | **Retained** | 5,387 | 734 | 6,121 |

**No other filter was applied.** The R1–R9 rejects were already absent from the source.

**Validation:** its 10 Tier A and B rows are excluded, and it has no Tier C rows.

**`02208` has its own tier label**, `C_review_added_by_reproducibility_audit`. Its reason says it has the same SQL as Tier C `02207` (reads the empty table `pitstops`) and that it was missed by the earlier screen.

## 3. Exact counts (recalculated)

| | Train | Val |
|---|---:|---:|
| Source (`spider_pool.zip`, matches the zip manifest) | 5,514 | 744 |
| Excluded | −84 | −10 |
| Quarantined | −43 | −0 |
| **Retained** | **5,387** | **734** |
| Databases / distinct queries retained | 123 / 3,084 | 16 / 435 |
| Hardness easy / medium / hard / extra | 1,392 / 2,220 / 1,078 / 697 | 186 / 315 / 128 / 105 |

**Reconciliation with earlier figures**
- 5,430 = 5,514 − 57 Tier A − 37 Tier B, counted per split (the earlier A + B figure).
- 5,388 = 5,430 − 42 Tier C.
- **5,387** = 5,388 − `02208`, recounted from the written files.
- Val is 734 in every case, since none of the quarantined rows are validation rows.
- The 43 quarantined rows come from 15 databases. The most affected are `bike_1` (10), `baseball_1` (6), `hospital_1` (4) and `insurance_and_eClaims` (4).

## 4. Verification

The verifier ran 38 checks and **all passed**. It re-derives every expected set from the zip and the evidence files rather than reusing the builder's logic.

| Area | What was checked |
|---|---|
| Identity | Exact file set; output hashes and counts equal the manifest; LF line endings |
| Uniqueness | Unique IDs in each file; no ID in two record files; every output ID exists in source train or val |
| Separation | Train/val overlap: 0 IDs, 0 databases, 0 query groups; split fields consistent |
| Dispositions | Every one of the 6,258 source records has exactly one disposition. The retained set equals source minus tiers; excluded and quarantine sets are exact |
| Reasons | Every removed record has a tier, a reason and evidence. Tier and reason are verbatim; line hashes match the zip |
| Content | Retained and quarantined records equal their source as parsed JSON **and** byte for byte; source order kept |
| Duplicates | None introduced. No query group is split between retained and removed |
| Protected data | 0 of 1,034 Spider dev IDs, 0 of 2,147 Spider test IDs, 0 of 65 held-out IDs; only `spider_train_` IDs; no dev database |
| Counts | Source = excluded + quarantined + retained for each split; tier counts per split equal the audit's |

**Testing the verifier**
- **Determinism:** two scratch builds and the final build produced identical hashes for all 5 files.
- **Overwrite protection:** a second run into an existing folder was refused.
- **Corruption tests:** on scratch copies, five corruptions were each detected. These were an edited question, a dropped exclusion entry, a Tier A row re-added, a duplicated val row and an injected dev ID.
- **Exit codes:** with the final verifier (sha256 `c1512ce4…`), the clean build exits 0 and each corruption exits 1. An earlier version crashed on the injected dev ID instead of failing cleanly; that was fixed. Results: `outputs/spider_pool_v2_prep/corruption_tests.json`.

**How protected IDs were checked:** the verifier reads `dev.jsonl` and the dev rows of `rejected.jsonl` from inside the zip, and `test_full.jsonl`, keeping only IDs and dev database names. No protected question or SQL was viewed.

## 5. Quarantine decisions needed

All 43 rows are listed in `exclusions.jsonl` with their reasons. For each, a reviewer decides whether to release it to train or exclude it.

| Reason | Rows |
|---|---:|
| Many-to-many join, possible fan-out | 18 |
| Compare on a column mixing numbers and blank strings | 12 |
| `=` literal never matches, so an OR or UNION branch never fires | 9 |
| Reads an empty table | 3 (2 Tier C, plus `02208`) |
| Nested `LIMIT` tie with the same returned value | 2 |

The reasons add up to 44 for 43 rows, because one Tier C row has two. Exact per-row reasons are in the file.

**Reviewer note on `spider_train_02945` (`sakila_1`):** its recorded reason is the unmatched literal. The reproducibility audit's recheck found that it also reads the empty table `staff`, like its paraphrase `02944`. Its `exclusions.jsonl` reason was kept verbatim, so the reviewer should weigh both.

**Releasing rows** should be done by adding a reviewed decision file and building a **new** version, such as `spider_pool_v3`. Never edit `spider_pool_v2` in place, and do not move quarantined rows into conversion before that decision.

## 6. Effect on frozen V2 sets (IDs only, no gate statistic)

| Frozen set | Size | In v2 train | Quarantined | Excluded | Not in pool |
|---|---:|---:|---:|---:|---:|
| Pilot | 300 | 271 | 1 (04586) | 4 (01259, 02225, 06102, 06104) | 24 |
| V2 dev | 500 | 450 | 1 (03935) | 8 | 41 |

The frozen sets were not changed. Whether the pilot stays frozen at 300 or is re-registered on v2 is still your decision. I still recommend keeping it frozen and reporting v2 membership as a slice.

## 7. Remaining limitations

- **Labels are not certified.** The tiers come from automated screens and one LLM reader. The earlier hand sample found 3 clear errors in 80 (Wilson 1.28–10.45%) **before** exclusions, and the error rate after exclusions has not been measured. More wrong labels may remain.
- **Tier B was not read row by row**, and the screens have blind spots, as the missed `02208` shows.
- **No text-level leakage check** against dev or test, because that would require reading protected fields. The evidence is 0 shared IDs and 0 shared databases.
- **SQL is not MQL.** `LIKE`, `NOT IN`, integer-vs-text joins and text/number compares will need careful conversion and will produce converter rejections.
- **Expected outputs come from SQLite.** On SQLite 3.40.1, 56 train/val outputs differ by up to 1.49e-08. Compare with a tolerance and do not regenerate them on that version.
- **Not version-controlled:** `.gitignore` excludes `*.jsonl` and the whole `outputs/` folder. So neither the v2 data files nor the evidence the builder pins (`recommended_exclusions.jsonl`, `stage9_empty_table_recheck.json`) would be tracked, even if committed.
- **If that evidence is lost or edited, v2 cannot be rebuilt:** the builder stops on its pinned hash. The hashes in `manifest.json`, and this report, are the record. Backing up those two files with the zip would protect the rebuild.
- **Provenance items still open:** the builder and zip are untracked, and the README's 145-example Spider-DK eval set is unexplained.

## 8. Verdict

**`data/processed/spider_pool_v2/train.jsonl` (5,387) and `val_internal.jsonl` (734) are safe to pass to the V2 conversion pipeline as its SQL source,** on these conditions:

- Only those two files are used. `quarantine.jsonl` and `exclusions.jsonl` must not be loaded as training data.
- Converted labels are treated as unverified until verifier-3 and independent human review accept them.

"Safe" here means:
- the data holds no protected data;
- every audited defect is out of the data;
- every record is traceable to its source;
- the artifact is reproducible.

It does **not** mean the SQL labels are semantically correct. This does not start or authorise the pilot, which still needs:
- your pilot-set decision;
- a verifier-3 pilot runner;
- a named human reviewer;
- your go-ahead.
