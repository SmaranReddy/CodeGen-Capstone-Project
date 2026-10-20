"""Validate hand-authored gold MQL against the gold SQL result (dev database on Atlas).

Gold MQL lives in data/interim/spider1/gold_mql/<db_id>.json (the original 300-item sample)
and data/interim/spider1/gold_mql_expansion/<db_id>.json (the 193 pool items added by the
expansion); load_gold merges both and refuses an id present in both. Format:
    {"<id>": {"collection": "<db_id>__<table>", "pipeline": [...], "notes": "..."}}

For each entry: structural read-only check -> run on Atlas `codegen_spider1` ->
exec_match against the gold SQL rows from sample.jsonl (order/tie aware).

Usage:
    python scripts/spider1/validate_gold.py --db concert_singer [--show]
    python scripts/spider1/validate_gold.py --all [--out data/interim/spider1/gold_validation.jsonl]
    python scripts/spider1/validate_gold.py --items data/interim/spider1/pool.jsonl --db wta_1
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import spider1_kit as K  # noqa: E402

GOLD_DIR = K.INTERIM / "gold_mql"
GOLD_EXP_DIR = K.INTERIM / "gold_mql_expansion"
GOLD_DIRS = (GOLD_DIR, GOLD_EXP_DIR)
FORBIDDEN = {"$out", "$merge", "$where", "$function", "$accumulator", "$documents",
             }


def walk_keys(o):
    if isinstance(o, dict):
        for k, v in o.items():
            yield k
            yield from walk_keys(v)
    elif isinstance(o, list):
        for v in o:
            yield from walk_keys(v)


def refs(o):
    out = set()
    if isinstance(o, dict):
        for k, v in o.items():
            if k in ("from", "coll") and isinstance(v, str):
                out.add(v)
            elif k == "$unionWith" and isinstance(v, str):
                out.add(v)
            else:
                out |= refs(v)
    elif isinstance(o, list):
        for v in o:
            out |= refs(v)
    return out


def structural(entry, db_id, existing):
    coll, pipe = entry.get("collection"), entry.get("pipeline")
    if not isinstance(coll, str) or not isinstance(pipe, list):
        return "missing collection/pipeline"
    bad = {k for k in walk_keys(pipe) if k in FORBIDDEN}
    if bad:
        return f"forbidden operators {sorted(bad)}"
    for c in {coll} | refs(pipe):
        if not c.startswith(db_id + "__") or c not in existing:
            return f"collection {c} not in {db_id}"
    return None


def validate(mdb, item, entry, existing):
    err = structural(entry, item["db_id"], existing)
    if err:
        return {"pass": False, "reason": "structural: " + err}
    try:
        docs = list(mdb[entry["collection"]].aggregate(entry["pipeline"], maxTimeMS=120000,
                                                        allowDiskUse=True))
    except Exception as e:  # noqa: BLE001
        return {"pass": False, "reason": f"execution: {type(e).__name__}: {str(e)[:300]}"}
    cols, rows = K.docs_to_rows(docs)
    ok, why = K.exec_match(rows, item["gold_rows"], item.get("tie_blocks"), item["order_by"])
    return {"pass": ok, "reason": why, "pred_columns": cols, "pred_rows": [list(r) for r in rows[:20]],
            "n_pred_rows": len(rows)}


def load_gold(db_id):
    out = {}
    for d in GOLD_DIRS:
        p = d / f"{db_id}.json"
        if not p.exists():
            continue
        entries = json.loads(p.read_text(encoding="utf-8"))
        dup = set(out) & set(entries)
        if dup:
            raise ValueError(f"gold MQL ids authored twice ({p}): {sorted(dup)}")
        out.update(entries)
    return out


def load_issues():
    out = {}
    for d in GOLD_DIRS:
        for p in d.glob("*.issues.json"):
            entries = json.loads(p.read_text(encoding="utf-8"))
            dup = set(out) & set(entries)
            if dup:
                raise ValueError(f"issues recorded twice ({p}): {sorted(dup)}")
            out.update(entries)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--ids", nargs="*")
    ap.add_argument("--show", action="store_true", help="print gold vs pred rows on failure")
    ap.add_argument("--out")
    ap.add_argument("--items", default=str(K.INTERIM / "sample.jsonl"),
                    help="candidate items (default: the frozen 300 sample; pool.jsonl for the expansion)")
    args = ap.parse_args()

    sample = K.read_jsonl(args.items)
    if args.db:
        sample = [s for s in sample if s["db_id"] == args.db]
    if args.ids:
        sample = [s for s in sample if s["id"] in args.ids]
    client = K.mongo_client()
    results, npass = [], 0
    try:
        mdb = client[K.ATLAS_DB]
        existing = set(mdb.list_collection_names())
        gold_cache = {}
        for item in sample:
            gold = gold_cache.setdefault(item["db_id"], load_gold(item["db_id"]))
            entry = gold.get(item["id"])
            if entry is None:
                res = {"pass": False, "reason": "no gold MQL authored"}
            else:
                res = validate(mdb, item, entry, existing)
            npass += res["pass"]
            results.append({"id": item["id"], "db_id": item["db_id"], **res})
            flag = "PASS" if res["pass"] else "FAIL"
            print(f"{flag} {item['id']} [{item['hardness']}] {res['reason']}")
            if not res["pass"] and args.show and entry is not None:
                print("   Q:", item["question"])
                print("   SQL:", item["gold_sql"])
                print("   gold:", item["gold_rows"][:8])
                print("   pred:", res.get("pred_columns"), res.get("pred_rows", [])[:8])
    finally:
        client.close()
    print(f"\n{npass}/{len(sample)} passed")
    if args.out:
        K.write_jsonl(args.out, results)


if __name__ == "__main__":
    main()
