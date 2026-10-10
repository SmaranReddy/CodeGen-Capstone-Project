"""Cross-check gold MQL on test-suite database variants (coincidence-proof evidence).

For each sampled item with gold MQL and each imported variant k: run the gold SQL on the
variant SQLite and the gold MQL on Atlas `codegen_spider1_ts<k>`, then compare with the
same `exec_match`. A variant whose ORDER BY ... LIMIT cutoff falls in a tie, or whose
tie structure cannot be verified, is recorded as `inconclusive` (not pass, not fail).

Usage: python scripts/spider1/testsuite_check.py [--db DB] [--ids ...] [--out PATH]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import spider1_kit as K  # noqa: E402
from validate_gold import load_gold  # noqa: E402


def check_item(client, item, entry, variants):
    out = []
    for k, v in enumerate(variants):
        rec = {"variant": v}
        try:
            _, rows = K.run_sql(item["db_id"], item["gold_sql"], v)
        except Exception as e:  # noqa: BLE001
            rec.update(status="inconclusive", reason=f"gold SQL error on variant: {e}")
            out.append(rec)
            continue
        ties = K.tie_analysis(item["db_id"], item["gold_sql"], rows, v)
        if ties["order_by"] and (ties["status"] != "ok" or ties.get("tie_at_limit")):
            rec.update(status="inconclusive", reason=f"order/tie: {ties['status']} tie_at_limit={ties.get('tie_at_limit')}")
            out.append(rec)
            continue
        try:
            docs = list(client[f"{K.ATLAS_DB}_ts{k + 1}"][entry["collection"]].aggregate(
                entry["pipeline"], maxTimeMS=120000, allowDiskUse=True))
        except Exception as e:  # noqa: BLE001
            rec.update(status="fail", reason=f"MQL execution error: {str(e)[:200]}")
            out.append(rec)
            continue
        _, prow = K.docs_to_rows(docs)
        ok, why = K.exec_match(prow, rows, ties["blocks"], ties["order_by"])
        rec.update(status="pass" if ok else "fail", reason=why, gold_rows=len(rows),
                   nonempty=bool(rows))
        if not ok:
            rec.update(gold_preview=[list(r) for r in rows[:5]], pred_preview=[list(r) for r in prow[:5]])
        out.append(rec)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db")
    ap.add_argument("--ids", nargs="*")
    ap.add_argument("--out")
    args = ap.parse_args()
    variants = json.loads((K.INTERIM / "testsuite_variants.json").read_text(encoding="utf-8"))
    sample = K.read_jsonl(K.INTERIM / "sample.jsonl")
    if args.db:
        sample = [s for s in sample if s["db_id"] == args.db]
    if args.ids:
        sample = [s for s in sample if s["id"] in args.ids]
    client = K.mongo_client()
    results = []
    try:
        for item in sample:
            entry = load_gold(item["db_id"]).get(item["id"])
            vs = variants.get(item["db_id"])
            if entry is None or not vs:
                results.append({"id": item["id"], "db_id": item["db_id"], "checks": [],
                                "summary": "not_checked"})
                continue
            checks = check_item(client, item, entry, vs)
            st = [c["status"] for c in checks]
            summary = "fail" if "fail" in st else ("pass" if "pass" in st else "inconclusive")
            results.append({"id": item["id"], "db_id": item["db_id"], "checks": checks, "summary": summary})
            print(f"{summary.upper():12s} {item['id']} {[c['status'] for c in checks]}"
                  + ("" if summary != "fail" else "  " + "; ".join(c["reason"] for c in checks if c["status"] == "fail")))
    finally:
        client.close()
    if args.out:
        K.write_jsonl(args.out, results)


if __name__ == "__main__":
    main()
