"""Print sampled items for one db_id with the Atlas schema (observed BSON types),
foreign keys from tables.json, and two sample documents per collection.

Usage: python scripts/spider1/show_items.py <db_id> [--ids ...]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import spider1_kit as K  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("db_id")
    ap.add_argument("--ids", nargs="*")
    ap.add_argument("--no-schema", action="store_true")
    ap.add_argument("--items", default=str(K.INTERIM / "sample.jsonl"))
    a = ap.parse_args()
    if not a.no_schema:
        man = json.loads((K.INTERIM / "atlas_collections.json").read_text(encoding="utf-8"))
        client = K.mongo_client()
        mdb = client[K.ATLAS_DB]
        for name, info in man.items():
            if info["db_id"] != a.db_id:
                continue
            print(f"\n## {name}  ({info['documents']} docs)")
            for f, t in info["fields"].items():
                print(f"   {f}: {'/'.join(t)}   (declared {info['declared'].get(f)})")
            for d in mdb[name].find({}, {"_id": 0}).limit(2):
                print("   e.g.", json.dumps(d, default=str)[:300])
        client.close()
        tj = {t["db_id"]: t for t in K.load_json(K.SPIDER / "tables.json")}[a.db_id]
        cols = tj["column_names_original"]
        tabs = tj["table_names_original"]
        print("\nforeign keys:")
        for x, y in tj["foreign_keys"]:
            print(f"   {tabs[cols[x][0]]}.{cols[x][1]} -> {tabs[cols[y][0]]}.{cols[y][1]}")
    print("\n# items")
    for r in K.read_jsonl(a.items):
        if r["db_id"] != a.db_id or (a.ids and r["id"] not in a.ids):
            continue
        print(f"\n{r['id']} [{r['hardness']}] order_by={r['order_by']} limit={r['limit']} rows={r['n_rows']}")
        print("  Q:  ", r["question"])
        print("  SQL:", r["gold_sql"])
        print("  gold:", json.dumps(r["gold_rows"][:6], default=str)[:400])


if __name__ == "__main__":
    main()
