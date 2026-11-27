# MQL-0009: safe Atlas provisioning of the 16 held-out databases (prepared, NOT executed)

**Status (2026-11-18):**

| Item | State |
|---|---|
| Wrapper | Built and tested against mocked clients |
| Dry run | Passed offline |
| Atlas / MongoDB | Not contacted |
| Data | Nothing was seeded |
| MQL-0009 package | Unchanged (manifest `2a4f22e0…`) |
| Existing importer and seeder | Unchanged |

## 1. What the existing tools can modify

| Tool | Database written | Operations | Repo files written | Verdict |
|---|---|---|---|---|
| `scripts/spider1/import_atlas.py` | `codegen_spider1` (hard-coded `K.ATLAS_DB`) | **`drop_collection`** then `insert_many` | **rewrites `data/interim/spider1/atlas_collections.json`** (read by `prompt_spider1` and MQL-0009 `readiness`) | **Unsafe for this task: never use** |
| `scripts/nl2mql/seed_atlas_spider_dev.py` CLI | `--target-db`, **default `codegen_spider1`** | `list_collection_names`, `insert_many(ordered=True)`, `find` (digest) | none | Safe operations, but unsafe default target; the CLI accepts only the Spider-dev bundle |
| `seed_atlas_spider_dev.seed()` | whatever database the client and `target_db` give it | the same three operations; aborts if any target collection exists; post-insert count and digest check | none | Good base: insert-only. Not idempotent: a rerun after a partial failure aborts on the clash. Its clash check is a snapshot |
| `seed_atlas_spider_dev.sqlite_docs()` | none (read-only SQLite) | `SELECT … ORDER BY rowid` (fallback: natural order) | none | Re-used unchanged |
| `dev1034_core.atlas_collection_digest` | none | `find({}, {"_id": 0})` | none | Re-used unchanged |

## 2. The wrapper: `scripts/nl2mql_mql0009/provision_heldout_atlas.py`

**Target and inputs:**
- The target is the constant **`codegen_spider1_heldout`**.
- Input is the pinned provisioning contract `outputs/nl2mql_mql0009_retrieval_control/scoring_contract/atlas_mapping.json`
  (sha256 `f8500804…`). It holds per-collection fields, counts, type-tagged row digests and SQLite hashes.
- Sources are the repo's Spider SQLite files.

**Re-used unchanged:** `seed()`, `sqlite_docs()`, `atlas_collection_digest`, `sqlite_table_digest` and
`privileges_report`, all imported rather than copied.

**Safety layers.** Each layer fails closed, before any write.
1. **Destination:**
   - `--target-db` must equal `codegen_spider1_heldout`, and `--execute` also needs `--confirm-target codegen_spider1_heldout`.
   - `codegen_spider1`, `admin`, `local` and `config` are always refused.
2. **Contract pin and sources:**
   - The contract sha256 is pinned.
   - Every SQLite file's hash is checked, and every table is re-read to the contract count and digest.
3. **Explicit mapping:** logical `(db_id, table)` maps to physical `codegen_spider1_heldout.<db_id>__<table>`. The checks:
   - the name equals `<db_id>__<table>`, and the db_id is one of the 16;
   - no exact or case-insensitive collision;
   - the database prefix is unambiguous (no db_id is a `__`-prefix of another's name);
   - MongoDB name rules hold, and the namespace is at most 235 bytes;
   - no clash with the 80 Spider-dev collection names;
   - every foreign key of the 16 DBs (59 pairs) maps to two collections of the **same** logical database.
4. **Guarded client:**
   - It structurally exposes one database and only `list_collection_names`, `find`, `count_documents` and `insert_many`.
   - `insert_many` works only for collections that were absent at plan time, and takes no options.
   - Everything else raises `DestructiveOperationRefused` before reaching the driver. That covers other databases,
     `drop_database`, `drop_collection`, `drop`, `delete_*`, `update_*`, `replace_*`, `rename`, `create_index`,
     `bulk_write`, `aggregate`/`$out`/`$merge` and raw `command`.
5. **Credentials and privileges:**
   - The URI comes from `MQL0009_PROVISION_URI` only.
   - `MONGODB_URI`, `MONGODB_ATLAS_URI` and `ATLAS_SEED_URI` are refused by name, and a URI equal to `MONGODB_URI` is
     refused. The URI is never printed; errors are redacted.
   - After connecting, `connectionStatus` must show:
     - `insert`, `find` and `listCollections` on the target;
     - **no** write or admin action on any other database, on any-database, or on the cluster;
     - **no** destructive action on the target: `remove`, `update`, `dropCollection`, `dropDatabase`, rename, index changes
       and similar. In practice the built-in `readWrite` role is refused.
6. **Idempotent preflight.** Every contract collection is classified before writing:

   | State | Meaning | Action |
   |---|---|---|
   | `absent` | not on the target | insert |
   | `verified_existing` | count, digest and field set all equal the contract | skip |
   | `conflict` | partial, extra documents, a changed value or extra fields | **abort; nothing written** |

   Any collection on the target that is not in the contract also aborts.
7. **Insert and verify:**
   - `seed()` inserts only the absent subset through the guarded client, with ordered batches.
   - Each collection is then verified by count and digest.
   - Finally the target must hold exactly the 69 contract collections and 16,885 documents, all digest-equal.
   - The operations actually used are checked against the allowed set.
8. **Report:** written only to `outputs/nl2mql_mql0009_retrieval_control/provisioning/`. The repo inventory file is never
   touched.

**Failure behaviour:**
- A crash mid-run leaves completed collections in place.
- A rerun verifies them, skips them and inserts the rest.
- A collection that failed mid-insert is partial, so it is a **conflict**. The tool aborts and never deletes or
  overwrites it; a human must inspect and remove it.

## 3. Dry-run report

Command: `python -B scripts/nl2mql_mql0009/provision_heldout_atlas.py`

Report: `outputs/nl2mql_mql0009_retrieval_control/provisioning/dry_run_20261118T020554.json`

- Target: `codegen_spider1_heldout`.
- 16 logical databases, 69 collections, 16,885 documents.
- 69/69 SQLite sources verified against the contract.
- 59 foreign-key (`$lookup`) pairs checked.
- No network or database connection.

**Planned operations, per collection:**
- If absent: `insert_many(ordered=True)` in one batch (every table has fewer than 20,000 rows), then a count and digest check.
- If present and identical: verify only.
- If present and different: ABORT.

| Logical DB → `codegen_spider1_heldout.<db>__<table>` | Collections | Docs | Tables |
|---|---:|---:|---|
| chinook_1 | 11 | 15,607 | Album, Artist, Customer, Employee, Genre, Invoice, InvoiceLine, MediaType, Playlist, PlaylistTrack, Track |
| company_employee | 3 | 31 | company, employment, people |
| cre_Docs_and_Epenses | 7 | 55 | Accounts, Documents, Documents_with_Expenses, Projects, Ref_Budget_Codes, Ref_Document_Types, Statements |
| election_representative | 2 | 12 | election, representative |
| epinions_1 | 4 | 31 | item, review, trust, useracct |
| gymnast | 2 | 17 | gymnast, people |
| hr_1 | 7 | 216 | countries, departments, employees, job_history, jobs, locations, regions |
| inn_1 | 2 | 610 | Reservations, Rooms |
| machine_repair | 4 | 37 | machine, repair, repair_assignment, technician |
| manufactory_1 | 2 | 17 | Manufacturers, Products |
| phone_market | 3 | 17 | market, phone, phone_market |
| ship_1 | 2 | 16 | Ship, captain |
| small_bank_1 | 3 | 20 | ACCOUNTS, CHECKING, SAVINGS |
| swimming | 4 | 37 | event, record, stadium, swimmer |
| tracking_share_transactions | 7 | 97 | Investors, Lots, Purchases, Ref_Transaction_Types, Sales, Transactions, Transactions_Lots |
| tracking_software_problems | 6 | 65 | Problem_Category_Codes, Problem_Log, Problem_Status_Codes, Problems, Product, Staff |
| **total** | **69** | **16,885** | |

**Why `$lookup` still works:**
- All 16 logical databases share one physical database, and every name carries its `<db_id>__` prefix, so `$lookup.from`
  names are exactly the names in the MQL-0009 prompts. A test checks that they equal the package context.
- The evaluator's safety gate forbids a `db` key in `$lookup`, so nothing can reach outside `codegen_spider1_heldout`.

## 4. Permission requirements (minimum, database-scoped)

**Provisioning account** (new, temporary, for example `codegen_heldout_provision`):
- One custom role on database `codegen_spider1_heldout` only, with actions `find`, `insert`, `listCollections` and
  `createCollection`. With `insert`, the collection is created implicitly; `createCollection` is optional.
- No `remove`, `update`, `dropCollection` or `dropDatabase`.
- No privilege on any other database and no cluster actions. **Do not use the built-in `readWrite` or `readWriteAnyDatabase`
  roles**: the wrapper refuses them.
- Supply its URI only as `$MQL0009_PROVISION_URI`. Delete or disable the user after provisioning.

**Read-only scoring account** (`codegen_eval_readonly`, existing):
- Add `find`, `listCollections` and `killCursors` on database **`codegen_spider1_heldout`** (the whole database, collection
  ""). Do this by extending its custom role or by adding a second read-only custom role.
- It must keep **no** write or admin action anywhere.
- `mql0009_scoring.py score` enforces both points: `atlas_preflight` requires `find` on `codegen_spider1_heldout.` and
  aborts on any write or admin action.
- Credentials stay separate: the scoring account never needs write access, and the provisioning account never scores.

## 5. Mocked test results

`tests/test_mql0009_provisioning.py`: **20/20 passed.** It uses a fake MongoDB client that records every operation and
synthetic SQLite databases whose contract is built by the real builder derivation.

| Requirement | Tests |
|---|---|
| Destination enforcement | other, protected and near-miss names; `--confirm-target`; `provision` and CLI refuse `codegen_spider1` |
| No destructive operations | 15 destructive calls blocked by the guard; zero destructive operations recorded in every scenario |
| Collision detection | case-insensitive, ambiguous prefix, wrong physical name, a foreign key to an unmapped table, a dev-name clash |
| All 16 logical DBs | real contract offline (pymongo blocked): 16 DBs, 69 collections, 16,885 docs, names equal the package context, 59 FK pairs |
| Document-count verification | a lost document on insert stops the run |
| Digest verification | a corrupted value on insert stops the run; a rerun then refuses to overwrite (conflict) |
| Conflicting collection | wrong value, partial, extra document, extra field: abort with nothing inserted and the data untouched |
| Unexpected collection on target | abort |
| Idempotency | a full rerun writes nothing; a pre-existing correct collection is skipped |
| Failure handling | an insert error mid-run, then a rerun verifies the finished collections and completes |
| Privileges | `readWrite`-like, writes on `codegen_spider1`, any-database, cluster `dropDatabase`, missing `insert`: all abort before writing |
| Credentials | refused env names, missing URI, URI equal to `MONGODB_URI` |
| Contract and sources | contract tamper and SQLite digest mismatch are refused |
| Readiness | a complete provisioning report (right target, 69 collections, 16,885 docs) unblocks `readiness`; an incomplete one does not |
| Untouched tools | hashes of `seed_atlas_spider_dev.py`, `import_atlas.py` and the MQL-0009 package manifest are asserted |

## 6. Is provisioning safe to authorize?

**Yes, with conditions.** The wrapper makes the destructive paths structurally impossible from this code: drops, overwrites,
writes to another database and inventory rewrites. Conflicting data aborts rather than being repaired. The remaining risks
are operational.

1. **The role must be the minimal custom role**, not `readWrite`. The wrapper checks this, but only through
   `connectionStatus`. An Atlas admin creating the user is the real control.
2. **Concurrent writers.** The window between preflight and insert is protected only by `seed()`'s second clash check and
   the post-insert digest. Nobody else should write to `codegen_spider1_heldout` during the run.
3. **Partial collections after a crash are never auto-cleaned.** Removing one needs a human with a separate admin role.
   That is deliberate.
4. **Atlas-side behaviour has not been exercised.** Driver error codes, implicit collection creation under `insert`-only,
   and the tier quota (a few MB against 512 MB, about 54 MB used) are untested here. The first real run should be watched;
   it is about 17k documents and seconds of work.
5. **After provisioning:**
   - grant the read-only role access to the new database;
   - run `mql0009_scoring.py readiness`. It accepts the wrapper's `provision_*.json` report (complete, 69 collections,
     16,885 documents, matching target), so the repo's dev inventory file never has to be edited. The authoritative check
     is still `score`'s live full-parity preflight;
   - then score only after the Kaggle predictions exist.

**To execute later** (user action):

```
$env:MQL0009_PROVISION_URI = "<provisioning-user URI>"
python -B scripts/nl2mql_mql0009/provision_heldout_atlas.py --execute --target-db codegen_spider1_heldout --confirm-target codegen_spider1_heldout
Remove-Item Env:MQL0009_PROVISION_URI
```
