"""Heuristic error analysis of model predictions on the validated Spider 1.0 MQL dataset.

For every incorrect prediction, tag likely root causes (a prediction can get several tags):
  extra_columns / missing_columns  -- column count above/below gold
  too_many_rows / too_few_rows     -- row count mismatch (e.g. missing $group/$limit, over-filtering)
  empty_result                     -- query ran but returned nothing
  type_mismatch_join               -- $lookup localField/foreignField with different stored types
  type_mismatch_filter             -- $match literal of a different BSON type than the field
  agg_op_in_match                  -- aggregation operator ($avg, $max, ...) used in $match without $expr
  invalid_collection_or_stage      -- rejected by the read-only validator
  execution_error:<kind>           -- MongoDB server error, grouped by error code/name
  malformed_json                   -- output not parseable
  order_only                       -- right rows, wrong order
Writes outputs/evaluation_spider1/error_analysis_<label>.json and prints a summary.

Usage: python scripts/spider1/error_analysis.py <label> [<label> ...]
"""
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import spider1_kit as K  # noqa: E402

OUT = K.ROOT / "outputs/evaluation_spider1"
AGG_OPS = {"$avg", "$max", "$min", "$sum", "$size", "$count"}


def field_types(db_id):
    man = json.loads((K.INTERIM / "atlas_collections.json").read_text(encoding="utf-8"))
    return {(c, f): set(t) - {"null"} for c, v in man.items() if v["db_id"] == db_id
            for f, t in v["fields"].items()}


def py_bson(v):
    if isinstance(v, bool):
        return "bool"
    if isinstance(v, int):
        return "int"
    if isinstance(v, float):
        return "double"
    if isinstance(v, str):
        return "string"
    return None


def walk_match(o, coll, types, tags):
    if isinstance(o, dict):
        for k, v in o.items():
            if not k.startswith("$"):
                ft = types.get((coll, k))
                lits = v if not isinstance(v, dict) else [x for op, x in v.items() if op in
                                                          ("$eq", "$gt", "$gte", "$lt", "$lte", "$ne", "$in", "$nin")]
                flat = []
                for x in (lits if isinstance(lits, list) else [lits]):
                    flat += x if isinstance(x, list) else [x]
                for x in flat:
                    bt = py_bson(x)
                    if ft and bt and not (bt in ft or {bt, *ft} <= {"int", "double", "long"}):
                        tags.add("type_mismatch_filter")
                if isinstance(v, dict) and set(v) & AGG_OPS:
                    tags.add("agg_op_in_match")
                for op, x in (v.items() if isinstance(v, dict) else []):
                    if isinstance(x, dict) and set(x) & AGG_OPS:
                        tags.add("agg_op_in_match")
            elif k in ("$and", "$or", "$nor"):
                for sub in v:
                    walk_match(sub, coll, types, tags)


def tag(rec, ex):
    tags = set()
    res = rec["result"]
    cat = res["category"]
    if cat == "format_error":
        return {"malformed_json"}
    if cat == "invalid_mql":
        return {"invalid_collection_or_stage"}
    if cat == "provider_error":
        return {"provider_error"}
    mql = rec.get("mql") or {}
    types = field_types(ex["db_id"])
    coll = mql.get("collection")
    for st in mql.get("pipeline", []):
        if not isinstance(st, dict):
            continue
        if "$match" in st and isinstance(st["$match"], dict):
            walk_match(st["$match"], coll, types, tags)
        lk = st.get("$lookup")
        if isinstance(lk, dict) and "localField" in lk:
            lt = types.get((coll, lk["localField"].split(".")[-1])) or set()
            if not lt:  # localField may come from an earlier unwound join: search all collections
                lt = set().union(*[t for (c, f), t in types.items() if f == lk["localField"].split(".")[-1]] or [set()])
            ft = types.get((lk.get("from"), lk.get("foreignField"))) or set()
            if lt and ft and not (lt & ft) and not (lt | ft) <= {"int", "double", "long"}:
                tags.add("type_mismatch_join")
    if cat == "execution_error" and "is not allowed or the syntax is incorrect" in (res.get("reason") or ""):
        # Atlas proxy wording for an unknown stage/operator, e.g. {"match": ...} or "$distinct"
        tags.add("invalid_stage_or_operator_name")
    elif cat == "execution_error":
        m = re.search(r"'codeName': '(\w+)'", res.get("reason") or "") or \
            re.search(r"^(\w+)", res.get("reason") or "")
        tags.add("execution_error:" + (m.group(1) if m else "other"))
    elif cat == "order_mismatch":
        tags.add("order_only")
    elif cat in ("wrong_row_count", "wrong_column_count", "wrong_values"):
        n_pred = res.get("n_rows") or 0
        gold_n = len(ex["gold_rows"])
        if n_pred == 0:
            tags.add("empty_result")
        elif n_pred > gold_n:
            tags.add("too_many_rows")
        elif n_pred < gold_n:
            tags.add("too_few_rows")
        pc = len(rec.get("pred_columns") or [])
        gc = len(ex["gold_rows"][0])
        if n_pred and pc > gc:
            tags.add("extra_columns")
        elif n_pred and pc < gc:
            tags.add("missing_columns")
        if cat == "wrong_values" and not tags:
            tags.add("wrong_values_same_shape")
    return tags or {"unclassified"}


def evaluated_records(records, data):
    """Records that were actually scored: in the dataset, not provider errors, first per id.
    Never-evaluated ids and provider errors are excluded, not counted as wrong."""
    seen, out = set(), []
    for r in records:
        if r["id"] in data and r["id"] not in seen and r["result"]["category"] != "provider_error":
            seen.add(r["id"])
            out.append(r)
    return out


def main():
    data = {d["id"]: d for d in K.read_jsonl(K.ROOT / "data/processed/spider1_nl_mql.jsonl")}
    for label in sys.argv[1:]:
        recs = evaluated_records(K.read_jsonl(OUT / label / "executions.jsonl"), data)
        wrong = [r for r in recs if not r["result"]["correct"]]
        per = {r["id"]: sorted(tag(r, data[r["id"]])) for r in wrong}
        cnt = Counter(t for ts in per.values() for t in ts)
        by_db = Counter(data[i]["db_id"] for i in per)
        out = {"label": label, "n": len(recs), "n_not_evaluated": len(data) - len(recs), "wrong": len(wrong), "tag_counts": dict(cnt.most_common()),
               "wrong_by_db": dict(by_db.most_common()), "per_item": per}
        (OUT / f"error_analysis_{label}.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
        print(f"== {label}: {len(wrong)}/{len(recs)} wrong ({len(data) - len(recs)} not evaluated)")
        for t, c in cnt.most_common():
            print(f"   {t:32s} {c}")


if __name__ == "__main__":
    main()
