"""Restore the source collections required to execute the 68-example benchmark.

Imports only the collections each example needs, grouped by source database, into the
dedicated pilot database (`codegen_capstone_pilot`). Reuses the Week-1 conversion kit's
read-only SQLite access and column/type rules. Existing collections are left untouched.

Derived helper collections used by two reference conversions (`local062__costs_keyed`,
`local285__whsle_keyed`) are rebuilt with the same key expressions the conversion used.

The set of collections created *by this evaluation* is tracked in
`outputs/evaluation_68/restored_collections.json`; only those are ever dropped.

Usage:
    python scripts/evaluation/restore_data.py --plan
    python scripts/evaluation/restore_data.py --import --db IPL
    python scripts/evaluation/restore_data.py --import-all
    python scripts/evaluation/restore_data.py --drop --db IPL
    python scripts/evaluation/restore_data.py --drop-all
"""
import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

TRANSIENT = ("resolution lifetime expired", "ServerSelectionTimeout", "AutoReconnect",
             "Connection reset", "connection closed", "Temporary failure", "network",
             "timed out", "No address associated", "getaddrinfo failed", "WinError")


def _retry(fn, *, tries=5, delay=8.0, label=""):
    last = None
    for attempt in range(1, tries + 1):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001
            last = exc
            msg = str(exc)
            if attempt >= tries or not any(k in msg for k in TRANSIENT):
                raise
            print(f"  [retry {attempt}/{tries - 1}] {label}: {msg.splitlines()[0][:140]}")
            time.sleep(delay * attempt)
    raise last

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_kit import ROOT, get_client, imports_for, load_artifact, load_dataset  # noqa: E402

PILOT_DB = "codegen_capstone_pilot"
OUT_DIR = ROOT / "outputs/evaluation_68"
RESTORE_LOG = OUT_DIR / "restored_collections.json"

_BALL_KEY = {"$concat": [{"$toString": "$match_id"}, "|", {"$toString": "$over_id"}, "|",
                         {"$toString": "$ball_id"}, "|", {"$toString": "$innings_no"}]}

# Fields/indexes the Week-1 conversions added *after* import (not captured in the artifact's
# `imports`). Reproduced verbatim so the reference pipelines execute as they did originally.
# Sources: scripts/conversions/conv_local{021,025,026,096}.py
POST_IMPORT = {
    "local021__batsman_scored": {"add": {"join_key": _BALL_KEY}, "index": "join_key"},
    "local025__batsman_scored": {"add": {"join_key": _BALL_KEY}, "index": "join_key"},
    "local025__extra_runs": {"add": {"join_key": _BALL_KEY}, "index": "join_key"},
    "local026__batsman_scored": {"add": {"join_key": _BALL_KEY}, "index": "join_key"},
    "local026__extra_runs": {"add": {"join_key": _BALL_KEY}, "index": "join_key"},
    "local096__M_Cast": {"add": {"pid_norm": {"$trim": {"input": "$PID"}}}, "index": "MID"},
    "local096__Person": {"index": "PID"},
}

DERIVED = {
    "local062__costs_keyed": {
        "source": "local062__costs",
        "key": {"$concat": [{"$toString": "$prod_id"}, "|", {"$toString": "$time_id"}, "|",
                            {"$toString": "$channel_id"}, "|", {"$toString": "$promo_id"}]},
        "keep": ["unit_cost", "unit_price"],
    },
    "local285__whsle_keyed": {
        "source": "local285__veg_whsle_df",
        "key": {"$concat": [{"$toString": "$item_code"}, "|", "$whsle_date"]},
        "keep": ["whsle_px_rmb-kg"],
    },
}


def load_restore_log():
    if RESTORE_LOG.exists():
        return json.loads(RESTORE_LOG.read_text(encoding="utf-8"))
    return {"evaluation_created": [], "groups": {}}


def save_restore_log(log):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RESTORE_LOG.write_text(json.dumps(log, indent=2, default=str), encoding="utf-8")


def build_plan():
    rows = load_dataset()
    groups = defaultdict(lambda: {"examples": [], "imports": {}, "derived": []})
    for r in rows:
        iid = r["instance_id"]
        art = load_artifact(iid)
        db = (r.get("db") or "unknown")
        g = groups[db]
        g["examples"].append(iid)
        for imp in imports_for(iid, art):
            if not imp.get("source_table") or not imp.get("fields"):
                continue
            key = imp["collection"]
            g["imports"][key] = {
                "collection": key, "source_table": imp["source_table"], "fields": imp["fields"],
                "filter": imp.get("filter"), "indexes": imp.get("indexes") or [], "iid": iid,
            }
        for d in (art.get("mongodb") or {}).get("derived") or []:
            g["derived"].append(d["collection"])
    return groups


def plan(verbose=True):
    from conversion_kit import manifest
    man = manifest()
    groups = build_plan()
    by_iid = {r["instance_id"]: r for r in load_dataset()}
    client = get_client()
    try:
        existing = set(client[PILOT_DB].list_collection_names())
    finally:
        client.close()
    total_missing_docs = 0
    for db, g in sorted(groups.items()):
        missing = [c for c in g["imports"] if c not in existing]
        derived_missing = [c for c in g["derived"] if c not in existing]
        docs = 0
        for c in missing:
            iid = c.split("__", 1)[0]
            imp = g["imports"][c]
            try:
                docs += _estimate_rows(man[iid], imp)
            except Exception:
                pass
        total_missing_docs += docs
        if verbose:
            print(f"{db:24s} examples={len(g['examples']):3d} collections={len(g['imports']):3d} "
                  f"missing={len(missing):3d} derived_missing={len(derived_missing)} est_rows={docs:,}")
    print(f"TOTAL missing collections={sum(len([c for c in g['imports'] if c not in existing]) for g in groups.values())} "
          f"est_rows={total_missing_docs:,}")
    return groups, existing


def _estimate_rows(rec, imp):
    from conversion_kit import sqlite_ro
    sq = sqlite_ro(rec)
    sql = f'SELECT COUNT(*) FROM "{imp["source_table"]}"' + (f" WHERE {imp['filter']}" if imp.get("filter") else "")
    return sq.execute(sql).fetchone()[0]


def import_group(db, log):
    from conversion_kit import manifest, sqlite_ro
    man = manifest()
    groups = build_plan()
    g = groups.get(db)
    if not g:
        print(f"no group '{db}'")
        return 0
    client = get_client()
    created = 0
    try:
        target = client[PILOT_DB]
        existing = set(target.list_collection_names())
        for coll, imp in g["imports"].items():
            if coll in existing:
                print(f"  [skip] {coll} (already present)")
                continue
            iid = imp.get("iid") or coll.split("__", 1)[0]
            rec = man[iid]
            n = _retry(lambda: _import_collection(target, rec, coll, imp["source_table"],
                                                  imp["fields"], imp.get("filter"),
                                                  imp.get("indexes") or []),
                       label=f"import {coll}")
            print(f"  [import] {coll} docs={n:,}")
            spec = POST_IMPORT.get(coll)
            if spec:
                _retry(lambda: _apply_post_import(target, coll, spec), label=f"post-import {coll}")
                print(f"  [post]   {coll} -> {spec}")
            if coll not in log["evaluation_created"]:
                log["evaluation_created"].append(coll)
            created += 1
        for dcoll in g["derived"]:
            if dcoll in existing:
                print(f"  [skip] {dcoll} (already present)")
                continue
            spec = DERIVED.get(dcoll)
            if not spec:
                print(f"  [warn] no derived builder for {dcoll}; skipped")
                continue
            _retry(lambda: _build_derived(target, dcoll, spec), label=f"derive {dcoll}")
            print(f"  [derive] {dcoll}")
            if dcoll not in log["evaluation_created"]:
                log["evaluation_created"].append(dcoll)
            created += 1
        log["groups"].setdefault(db, {})
        log["groups"][db]["examples"] = g["examples"]
        log["groups"][db]["collections"] = list(g["imports"]) + list(g["derived"])
        save_restore_log(log)
    finally:
        client.close()
    return created


def _import_collection(target, rec, coll, table, columns, where, indexes):
    from conversion_kit import sqlite_ro
    sq = sqlite_ro(rec)
    cols = ", ".join(f'"{c}"' for c in columns)
    sql = f'SELECT {cols} FROM "{table}"' + (f" WHERE {where}" if where else "")
    cur = sq.execute(sql)
    names = [d[0] for d in cur.description]
    batch = []
    n = 0
    if coll in target.list_collection_names():
        target.drop_collection(coll)
    while True:
        rows = cur.fetchmany(20000)
        if not rows:
            break
        batch = [dict(zip(names, r)) for r in rows]
        target[coll].insert_many(batch, ordered=False)
        n += len(batch)
    for f in indexes:
        target[coll].create_index(f)
    return n


def _apply_post_import(target, coll, spec):
    c = target[coll]
    if spec.get("add"):
        c.update_many({}, [{"$set": spec["add"]}])
    if spec.get("index"):
        c.create_index(spec["index"])


def _build_derived(target, dcoll, spec):
    if dcoll in target.list_collection_names():
        target.drop_collection(dcoll)
    proj = {"_id": 0, "key": spec["key"]}
    for f in spec["keep"]:
        proj[f] = 1
    target[spec["source"]].aggregate([{"$project": proj}, {"$out": dcoll}])
    target[dcoll].create_index("key")


def drop_group(db, log):
    client = get_client()
    dropped = 0
    try:
        target = client[PILOT_DB]
        for coll in list(log.get("groups", {}).get(db, {}).get("collections", [])):
            if coll in log["evaluation_created"] and coll in target.list_collection_names():
                target.drop_collection(coll)
                log["evaluation_created"].remove(coll)
                dropped += 1
        log.get("groups", {}).pop(db, None)
        save_restore_log(log)
    finally:
        client.close()
    return dropped


def drop_all(log):
    client = get_client()
    dropped = 0
    try:
        target = client[PILOT_DB]
        for coll in list(log["evaluation_created"]):
            if coll in target.list_collection_names():
                target.drop_collection(coll)
                dropped += 1
        log["evaluation_created"] = []
        log["groups"] = {}
        save_restore_log(log)
    finally:
        client.close()
    return dropped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--import", dest="do_import", action="store_true")
    ap.add_argument("--import-all", action="store_true")
    ap.add_argument("--drop", action="store_true")
    ap.add_argument("--drop-all", action="store_true")
    ap.add_argument("--db", default=None)
    args = ap.parse_args()
    log = load_restore_log()
    if args.plan:
        plan()
    elif args.import_all:
        from conversion_kit import manifest  # noqa: F401
        groups = build_plan()
        for db in sorted(groups):
            print(f"== import group {db} ==")
            import_group(db, log)
    elif args.do_import:
        print(f"== import group {args.db} ==")
        import_group(args.db, log)
    elif args.drop_all:
        print(f"dropped {drop_all(log)} collections")
    elif args.drop:
        print(f"dropped {drop_group(args.db, log)} collections in {args.db}")
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
