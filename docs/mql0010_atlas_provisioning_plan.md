# MQL-0010: Atlas provisioning plan (prepared, NOT executed)

Date: 2026-11-19.

**Status.** Code and contract are ready, and the offline dry run passed. **Nothing has been written to Atlas, and no connection has been made.** Execution needs your explicit approval.

## 1. What will be created

| Item | Value |
|---|---|
| Target database | **`codegen_spider1_holdout_mql0010`** (new; separate from `codegen_spider1` and from MQL-0009's `codegen_spider1_heldout`) |
| Source | the 18 holdout databases' Spider SQLite files (`data/raw/spider1/spider_data/database/<db>/<db>.sqlite`), mapped exactly as `import_atlas.py` / MQL-0009 do |
| Contract | `outputs/nl2mql_mql0010/scoring_contract/atlas_mapping.json`, sha256 `b14b3607…`, pinned by the frozen package config (`scoring_contract.mapping_sha256`) |
| Size | 18 databases, **99 collections, 1,280 documents**; about 1 MB, far below the 512 MB tier limit |
| Holdout databases | allergy_1, behavior_monitoring, cre_Theme_park, customers_and_addresses, e_learning, game_1, insurance_and_eClaims, local_govt_and_lot, mountain_photos, news_report, program_share, race_track, railway, school_player, shop_membership, store_product, student_1, workshop_paper |

## 2. Tool

`scripts/nl2mql_mql0010/provision_holdout_atlas.py`. It reuses the **unchanged, audited** MQL-0009 wrapper (`scripts/nl2mql_mql0009/provision_heldout_atlas.py`) and only points it at the MQL-0010 target, contract, URI variable and report folder.

**Safety properties, inherited and tested:**

| Property | How it works |
|---|---|
| Write firewall | Only ping, buildInfo, connectionStatus, list_collection_names, find, count_documents and insert_many exist. Every drop, delete, update, replace, rename, index, `$out`/`$merge` or command raises `DestructiveOperationRefused`. Inserts are allowed only into contract collections that were **absent** at preflight, once each. |
| Single database | Any other database name is refused. `codegen_spider1`, `codegen_spider1_heldout`, admin, local and config are protected. `--execute` also needs `--confirm-target` repeating the name. |
| Preflight before any write | Each contract collection is classified as absent (→ insert), verified_existing (→ skip, an idempotent re-run) or conflict (→ ABORT, nothing written). Unexpected collections in the target also abort. |
| Privilege check | Insert, find and listCollections on the target. No write or admin action on any other resource. No destructive action on the target: readWrite is refused. |
| Verification | After inserting, every collection's document count, type-tagged row digest and field set must equal the contract. |
| Credentials | The URI is read from `$MQL0010_PROVISION_URI` only. `MONGODB_URI`, `MONGODB_ATLAS_URI`, `ATLAS_SEED_URI` and `MQL0009_PROVISION_URI` are refused, and the URI must differ from the read-only scoring URI. It is never printed. |
| Report | `outputs/nl2mql_mql0010/provisioning/<dry_run_or_provision>_<time>.json` (the only file written) |

**Dry run already done (offline):** `outputs/nl2mql_mql0010/provisioning/dry_run_20261119T153634.json`.
- 18 databases, 99 collections, 1,280 documents;
- 99 SQLite tables re-read to the contract digests;
- 88 foreign-key `$lookup` pairs checked inside their logical databases;
- no collision with the Spider-dev collection names.

**Tests** (`tests/test_mql0010_package.py::test_provisioning_dry_run_and_target_guards`):
- the dry run;
- refusal of the three protected targets;
- refusal of `--execute` without confirmation;
- refusal of the scoring and MQL-0009 URI variables;
- the guard refusing another database, `drop_collection`, and inserts into non-absent collections.

## 3. Steps that need you (in order)

1. **Create a provisioning user in the Atlas UI** with a custom role scoped to `codegen_spider1_holdout_mql0010` only.
   - Allowed actions: `find`, `insert`, `listCollections`, `createCollection`.
   - **Not** `readWrite`, and no other database.
2. **Grant the existing read-only scoring user `codegen_eval_readonly`** these actions on `codegen_spider1_holdout_mql0010`:
   - `find`;
   - `listCollections`;
   - `killCursors`.
   
   Missing `find` is exactly what blocked MQL-0009 scoring. Keep no write or admin action.
3. **Put the provisioning URI in your shell** as `MQL0010_PROVISION_URI`. Never use `MONGODB_URI`.
4. **Re-run the dry run, then execute:**
   ```
   python -B scripts/nl2mql_mql0010/provision_holdout_atlas.py
   python -B scripts/nl2mql_mql0010/provision_holdout_atlas.py --execute --target-db codegen_spider1_holdout_mql0010 --confirm-target codegen_spider1_holdout_mql0010
   ```
5. **Check the report:** `status: complete`, 99 collections and 1,280 documents verified, and operations limited to `{ping, connectionStatus, list_collection_names, find, count_documents, insert_many}`.
6. **Later, at scoring:** `mql0010_scoring.py score` runs the read-only full-parity preflight against the same contract. It fails closed if any collection, count or digest differs, or if the scoring user holds any write action.

## 4. Failure handling

- **Conflict found:** the tool aborts before writing. It never drops or overwrites. Inspect the database and remove the offending collections manually yourself, then re-run.
- **Interrupted insert:** re-run. Completed collections verify as `verified_existing` and are skipped. A partial collection is a conflict and needs manual cleanup.
- **The contract changes** (any package rebuild): its hash changes. Provisioning and scoring then both refuse the old pin until the package is rebuilt and reviewed.

## 5. What this plan does not do

- It does not create Atlas users or roles (that is a manual UI step).
- It does not touch `codegen_spider1`, `codegen_spider1_heldout` or any other database.
- It does not change credentials or environment variables.
- It does not run scoring.
