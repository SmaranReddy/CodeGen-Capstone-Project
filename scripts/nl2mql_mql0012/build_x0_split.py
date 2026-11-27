"""MQL-0012 X0: fresh database-level confirmation split (reads inputs only; computes no retrieval or outcome).

Registered rule (fixed here before any X0 retrieval exists):
  * Database universe for X0 = the 90 MQL-0010 "build" databases (split_registration.json["build_dbs"]): the corpus
    databases that are neither MQL-0010 holdout (18) nor MQL-0010/MQL-0011 dev (15).
  * ALL 90 build databases are included (no sampling: "use as many legitimate questions as possible"). The seed
    20261123 is recorded for the record and used only by the bootstrap; it selects nothing.
  * Items: spider-sql-v1 train.jsonl rows of those databases with the MQL-0010 item rule (tier == exec_ok, no
    empty_result / all_null_result flag) AND spider-pool-v3 train membership (amendment A1) -- identical to how the
    MQL-0010 dev/holdout items were built.
  * Hard stops: any database overlap with the MQL-0010 holdout or dev, the MQL-0009 evaluation DBs, Spider dev
    (dev_full), or the expanded MQL splits (train/validation/test, Spider-dev derived); any item-id overlap with the
    MQL-0010 dev or holdout items.
  * Retrieval candidates: the registered MQL-0010 retrieval view (2,912 records) = the MQL-0011 embedding artifact
    and Qdrant collection, leave-target-database-out per query. No question-level exclusion pairs (same policy as
    the MQL-0011 dev screen, so R3 is run exactly as registered); near-duplicate exposure is reported descriptively.
Writes outputs/nl2mql_mql0012/split/{x0_split.json, x0_items.jsonl}. Refuses to overwrite.
"""
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
import mql0012_common as M   # noqa: E402

C = M.C11
SEED = 20261123


def main():
    if M.SPLIT.exists():
        raise SystemExit(f"{M.SPLIT} exists; the split is never overwritten")
    reg = C.load_registration()
    build, hold, dev = set(reg["build_dbs"]), set(reg["holdout"]["dbs"]), set(reg["dev"]["dbs"])
    if len(build) != 90 or build & hold or build & dev:
        raise SystemExit("build DBs are not the registered 90 disjoint databases")
    protected = {
        "mql0009_eval_dbs": {r["db_id"] for r in C.read_jsonl(M.MQL0009_EVAL)},
        "spider_dev_full_dbs": {r["db_id"] for r in C.read_jsonl(M.SPIDER_DEV_FULL)},
        "expanded_mql_split_dbs": {r["db_id"] for p in M.EXPANDED_SPLITS for r in C.read_jsonl(p)},
        "mql0010_holdout_dbs": hold, "mql0011_dev_dbs": dev}
    overlap = {k: sorted(build & v) for k, v in protected.items()}
    if any(overlap.values()):
        raise SystemExit(f"X0 databases overlap a protected set: {overlap}")
    pool = {r["id"] for r in C.read_jsonl(M.POOL_V3_TRAIN)}
    train = C.read_jsonl(C.SPIDER_TRAIN)
    before_a1 = [r for r in train if r["db_id"] in build and M.eligible_item(r)]
    items = [r for r in before_a1 if r["id"] in pool]
    used_ids = {r["id"] for r in C.read_jsonl(C.DEV_ITEMS)} | {
        r["id"] for r in C.read_jsonl(C.ROOT / "outputs/nl2mql_mql0010/preflight/holdout_items.jsonl")}
    if {r["id"] for r in items} & used_ids:
        raise SystemExit("X0 item ids overlap MQL-0010 dev/holdout items")
    corpus_ids = {r["id"] for r in C.read_jsonl(C.CORPUS)}
    view_rows, _ = C.load_view_rows()
    ids = [r["id"] for r in items]
    rep = {
        "experiment": "MQL-0012", "role": "X0 fresh database-level confirmation split", "rules": __doc__,
        "seed": SEED, "seed_note": "recorded; selects nothing (all 90 build databases included); used by the bootstrap",
        "included_dbs": sorted(build), "n_included_dbs": len(build),
        "n_dbs_with_items": len({r["db_id"] for r in items}),
        "excluded_dbs": {"mql0010_holdout": sorted(hold), "mql0011_dev": sorted(dev)},
        "protected_overlap_checked": {k: v for k, v in overlap.items()},
        "n_items": len(items), "n_items_before_pool_v3_filter": len(before_a1),
        "n_items_that_are_corpus_records": sum(i in corpus_ids for i in ids),
        "items_by_db": dict(sorted(Counter(r["db_id"] for r in items).items())),
        "ids_sha256": C.ids_sha256(ids),
        "retrieval_view": {"n_records": len(view_rows), "ids_sha256": C.ids_sha256([r["id"] for r in view_rows])},
        "inputs_sha256": {"spider_sql_v1_train": C.sha256_file(C.SPIDER_TRAIN), "pool_v3_train": C.sha256_file(M.POOL_V3_TRAIN),
                          "split_registration_mql0010": C.sha256_file(C.REGISTRATION), "corpus": C.sha256_file(C.CORPUS)},
    }
    C.write_json(M.SPLIT, rep)
    C.write_jsonl(M.X0_ITEMS, [{"id": r["id"], "db_id": r["db_id"], "question": r["question"]} for r in items])
    print(json.dumps({k: rep[k] for k in ("n_items", "n_items_before_pool_v3_filter", "n_dbs_with_items",
                                          "n_items_that_are_corpus_records", "ids_sha256", "retrieval_view")}, indent=1))


if __name__ == "__main__":
    main()
