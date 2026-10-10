"""Flag literals in gold MQL that do not occur in the gold SQL or question (hard-coding check).

Structural constants (0, 1, -1, booleans, null, short format strings) and field/collection
names are ignored; everything else is reported for manual review.
Usage: python scripts/spider1/audit_literals.py
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import spider1_kit as K  # noqa: E402
from validate_gold import GOLD_DIR  # noqa: E402

IGNORE_NUM = {0, 1, -1, 2, 10, 15, 100}


def literals(o, out, key=None):
    if isinstance(o, dict):
        for k, v in o.items():
            if k in ("from", "localField", "foreignField", "as", "coll", "$unionWith"):
                continue
            literals(v, out, k)
    elif isinstance(o, list):
        for v in o:
            literals(v, out, key)
    elif isinstance(o, bool) or o is None:
        return
    elif isinstance(o, (int, float)):
        if o not in IGNORE_NUM:
            out.add(o)
    elif isinstance(o, str):
        if not o.startswith("$") and len(o) > 1:
            out.add(o)


def main():
    sample = {r["id"]: r for r in K.read_jsonl(K.INTERIM / "sample.jsonl")}
    man = json.loads((K.INTERIM / "atlas_collections.json").read_text(encoding="utf-8"))
    fields = {f for v in man.values() for f in v["fields"]}
    n = 0
    for p in sorted(GOLD_DIR.glob("*.json")):
        if p.name.endswith(".issues.json"):
            continue
        for iid, e in json.loads(p.read_text(encoding="utf-8")).items():
            item = sample[iid]
            text = (item["gold_sql"] + " " + item["question"]).lower()
            lits = set()
            literals(e["pipeline"], lits)
            sus = []
            for x in lits:
                s = str(x).lower()
                if isinstance(x, str) and (x in fields or re.fullmatch(r"[a-z_]+", s) and len(s) < 4):
                    continue
                if isinstance(x, float) and str(int(x)) in text:
                    continue
                core = s.strip("^$.*%").replace("\\", "")
                if core and core in text:
                    continue
                sus.append(x)
            if sus:
                n += 1
                print(f"{iid} [{item['db_id']}] {sus[:8]}")
    print(f"\n{n} entries with literals not found in SQL/question (review manually)")


if __name__ == "__main__":
    main()
