"""Accuracy sliced by whether the gold MQL needs explicit type conversion (an MQL-specific
difficulty: Spider stores many numeric keys/values as TEXT, which SQL hides via affinity).

Each model is scored on the ids it answered (provider errors / never-evaluated ids excluded).
With --paired, every model is restricted to the ids answered by ALL given models.
Writes outputs/evaluation_spider1/slice_type_conversion[_paired].json.

Usage: python scripts/spider1/slice_type_conversion.py [--paired] <label> [...]"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import spider1_kit as K  # noqa: E402

CONV = ("$toString", "$toInt", "$toDouble", "$toLong", "$convert", "$toDecimal")


def needs_conversion(row):
    return any(c in json.dumps(row["gold_mql"]) for c in CONV)


def main():
    args = sys.argv[1:]
    paired = "--paired" in args
    labels = [a for a in args if a != "--paired"]
    data = {d["id"]: d for d in K.read_jsonl(K.ROOT / "data/processed/spider1_nl_mql.jsonl")}
    needs = {i for i, d in data.items() if needs_conversion(d)}
    runs = {}
    for lab in labels:
        runs[lab] = {r["id"]: r for r in K.read_jsonl(K.ROOT / "outputs/evaluation_spider1" / lab / "executions.jsonl")
                     if r["id"] in data and r["result"]["category"] != "provider_error"}
    scope = set(data)
    if paired:
        for recs in runs.values():
            scope &= set(recs)
    out = {"paired": paired, "n_scope": len(scope), "n_needs_conversion_in_scope": len(needs & scope),
           "n_other_in_scope": len(scope - needs), "runs": {}}
    print(f"scope: {'paired ids' if paired else 'each model on its answered ids'}; "
          f"gold needs explicit type conversion: {len(needs & scope)}/{len(scope)}")
    for lab, recs in runs.items():
        ids = [i for i in recs if i in scope]
        a = [recs[i]["result"]["correct"] for i in ids if i in needs]
        b = [recs[i]["result"]["correct"] for i in ids if i not in needs]
        out["runs"][lab] = {"needs_conversion": [sum(a), len(a)], "other": [sum(b), len(b)]}
        print(f"{lab:28s} needs-conversion {sum(a)}/{len(a)} = {sum(a) / max(1, len(a)):.1%}   "
              f"other {sum(b)}/{len(b)} = {sum(b) / max(1, len(b)):.1%}")
    name = "slice_type_conversion_paired.json" if paired else "slice_type_conversion.json"
    (K.ROOT / "outputs/evaluation_spider1" / name).write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
