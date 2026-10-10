"""Sensitivity check (NOT the primary metric): how many format errors are a single redundant
closing bracket (e.g. `...}]]}`), and would those queries be correct if that one character were
removed? Same rule for every model; repaired queries go through the same validator, Atlas
execution and exec_match. Writes outputs/evaluation_spider1/sensitivity_repair.json.

Usage: python scripts/spider1/sensitivity_repair.py <label> [<label> ...]
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "evaluation"))
import spider1_kit as K  # noqa: E402
from eval_kit import validate_mql  # noqa: E402


def repair(text):
    t = (text or "").strip()
    if t.startswith("```"):
        t = t.strip("`").removeprefix("json").strip()
    for bad, good in (("]]}", "]}"),):
        if t.endswith(bad):
            try:
                return json.loads(t[: -len(bad)] + good)
            except Exception:  # noqa: BLE001
                return None
    return None


def main():
    data = {d["id"]: d for d in K.read_jsonl(K.ROOT / "data/processed/spider1_nl_mql.jsonl")}
    client = K.mongo_client()
    mdb = client[K.ATLAS_DB]
    colls = mdb.list_collection_names()
    report = {}
    try:
        for label in sys.argv[1:]:
            recs = [r for r in K.read_jsonl(K.ROOT / "outputs/evaluation_spider1" / label / "executions.jsonl")
                    if r["id"] in data and r["result"]["category"] == "format_error"]
            rep = {"format_errors": len(recs), "single_extra_bracket": 0, "correct_after_repair": 0, "ids": {}}
            for r in recs:
                parsed = repair(r["raw_text"])
                if parsed is None:
                    continue
                rep["single_extra_bracket"] += 1
                ex = data[r["id"]]
                parsed.setdefault("method", "aggregate")
                ok, why, norm = validate_mql(parsed, {c for c in colls if c.startswith(ex["db_id"] + "__")})
                verdict = "invalid"
                if ok:
                    try:
                        docs = list(mdb[norm["collection"]].aggregate(norm["pipeline"], maxTimeMS=30000))
                        _, rows = K.docs_to_rows(docs)
                        verdict = "correct" if K.exec_match(rows, ex["gold_rows"], ex.get("tie_blocks"),
                                                            ex["order_by"])[0] else "incorrect"
                    except Exception as e:  # noqa: BLE001
                        verdict = f"execution_error: {type(e).__name__}"
                rep["ids"][r["id"]] = verdict
                rep["correct_after_repair"] += verdict == "correct"
            report[label] = rep
            print(label, {k: v for k, v in rep.items() if k != "ids"})
    finally:
        client.close()
    (K.ROOT / "outputs/evaluation_spider1/sensitivity_repair.json").write_text(json.dumps(report, indent=1),
                                                                              encoding="utf-8")


if __name__ == "__main__":
    main()
