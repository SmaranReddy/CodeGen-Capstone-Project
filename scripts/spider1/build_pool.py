"""Build the Spider 1.0 dev candidate pool and the frozen stratified sample.

Pre-registered rules (applied to all 1,034 dev questions BEFORE any MQL is written):
  1. gold SQL parses with the official process_sql and executes on the official SQLite DB
  2. unique by (db_id, normalised gold SQL) -- first question in dev order is kept
  3. gold result is non-empty and contains at least one non-NULL value
  4. gold result has <= 1000 rows
  5. ORDER BY ... LIMIT golds whose LIMIT cutoff falls inside a tie are excluded
  6. ORDER BY golds whose tie structure cannot be verified are excluded
Then a seed-42 sample of N (default 300), stratified proportionally by official hardness.
The sample is frozen: items that later fail conversion are reported as attrition and
never replaced by easier items.

Usage: python scripts/spider1/build_pool.py [--n 300] [--seed 42]
"""
import argparse
import hashlib
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import spider1_kit as K  # noqa: E402


def norm_sql(sql):
    s = re.sub(r"\s+", " ", sql.strip().rstrip(";")).lower()
    return re.sub(r"\s*([(),=<>])\s*", r"\1", s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    dev = K.load_json(K.SPIDER / "dev.json")
    pool, seen, excl = [], set(), Counter()
    for idx, ex in enumerate(dev):
        db, sql, q = ex["db_id"], ex["query"], ex["question"]
        rec = {"id": f"spider1_dev_{idx:04d}", "dev_index": idx, "db_id": db, "question": q,
               "gold_sql": sql}
        key = (db, norm_sql(sql))
        if key in seen:
            excl["duplicate_sql"] += 1
            continue
        seen.add(key)
        try:
            hard, _ = K.hardness(db, sql)
        except Exception:  # noqa: BLE001
            excl["unparseable"] += 1
            continue
        try:
            cols, rows = K.run_sql(db, sql)
        except Exception:
            excl["sql_error"] += 1
            continue
        if not rows or all(v is None for r in rows for v in r):
            excl["empty_or_null_result"] += 1
            continue
        if len(rows) > 1000:
            excl["too_many_rows"] += 1
            continue
        ties = K.tie_analysis(db, sql, rows)
        if ties["order_by"] and ties["status"] != "ok":
            excl["order_unverifiable"] += 1
            continue
        if ties.get("tie_at_limit"):
            excl["tie_at_limit"] += 1
            continue
        rec.update(hardness=hard, gold_columns=cols, gold_rows=[list(r) for r in rows],
                   n_rows=len(rows), order_by=ties["order_by"], limit=ties["limit"],
                   tie_blocks=ties["blocks"])
        pool.append(rec)

    by_h = {}
    for r in pool:
        by_h.setdefault(r["hardness"], []).append(r)
    rng = random.Random(args.seed)
    total = len(pool)
    quotas = {h: round(args.n * len(v) / total) for h, v in by_h.items()}
    diff = args.n - sum(quotas.values())
    for h in sorted(by_h, key=lambda h: -len(by_h[h]))[:abs(diff)]:
        quotas[h] += 1 if diff > 0 else -1
    sample = []
    for h in sorted(by_h):
        items = sorted(by_h[h], key=lambda r: r["dev_index"])
        sample += rng.sample(items, quotas[h])
    sample.sort(key=lambda r: r["dev_index"])

    out = K.INTERIM
    K.write_jsonl(out / "pool.jsonl", pool)
    K.write_jsonl(out / "sample.jsonl", sample)
    dev_sha = hashlib.sha256((K.SPIDER / "dev.json").read_bytes()).hexdigest()
    summary = {
        "source": "Spider 1.0 dev.json (official spider_data.zip, Google Drive id 1403EGqzIDoHMdQF4c9Bkyl7dZLZ5Wt6J)",
        "dev_json_sha256": dev_sha,
        "dev_questions": len(dev),
        "excluded": dict(excl),
        "eligible_pool": total,
        "pool_by_hardness": {h: len(v) for h, v in sorted(by_h.items())},
        "seed": args.seed,
        "sample_size": len(sample),
        "sample_by_hardness": dict(sorted(Counter(r["hardness"] for r in sample).items())),
        "sample_by_db": dict(sorted(Counter(r["db_id"] for r in sample).items())),
    }
    (out / "pool_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
