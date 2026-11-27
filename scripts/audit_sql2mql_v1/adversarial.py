"""Run the adversarial suite: python -B scripts/audit_sql2mql_v1/adversarial.py

For each case: hand-specified expectation vs (a) sqlsem's SQLite rendering (cross-check),
(b) the converter's MQL on MongoDB with NULL stored as explicit null, (c) the same MQL with
NULL stored as an absent field, and (d) what the EXISTING verifier (verify.verify_pair:
its own fixtures + comparator) concludes for the same SQL. Writes
outputs/audit_sql2mql_v1/adversarial.jsonl and adversarial_udf.json.
"""
import hashlib
import json

import common as K
from adversarial_cases import CASES, MONGO_DATE_CHECKS, SCHEMA, UDF_CHECKS


def judge(case, rows, expected):
    if rows is None:
        return False
    if case.get("subset_k"):
        base = [(r["id"],) for r in next(iter(case["tables"].values()))]
        g = [tuple(K.norm(x) for x in r) for r in rows]
        return len(g) == case["subset_k"] and len(set(g)) == len(g) and all(
            r in [tuple(K.norm(x) for x in b) for b in base] for r in g)
    if case.get("any_of"):
        return any(K.same(rows, a, True) for a in case["any_of"])
    return K.same(rows, expected, case.get("ordered", False))


def run_case(case, N):
    tables = case["tables"](N) if callable(case["tables"]) else case["tables"]
    case = dict(case, tables=tables)
    expected = case.get("expect")
    expected = expected(N) if callable(expected) else expected
    rec = {"id": case["id"], "feature": case["feature"], "sql": case["sql"], "expected": expected,
           "any_of": case.get("any_of"), "subset_k": case.get("subset_k"), "dialect": case.get("dialect")}
    mql, ast, an, reason = K.convert(case["sql"], SCHEMA)
    # SQLite cross-check via sqlsem's MySQL-semantics rendering (the pipeline's gold engine)
    try:
        sq = K.sqlite_rows(ast, tables, SCHEMA, N)
        rec["sqlite"] = sq
        rec["sqlite_agrees"] = judge(case, sq, expected)
    except Exception as e:  # noqa: BLE001
        rec["sqlite"] = f"ERROR {type(e).__name__}: {e}"
        rec["sqlite_agrees"] = None
    if mql is None:
        rec.update(converter="unsupported", reason=reason)
        return rec
    rec["converter"] = "converted"
    rec["mql"] = mql
    base = an.base
    cols = K.out_columns(mql, SCHEMA, base)
    for rep, missing in (("explicit_null", False), ("missing_field", True)):
        docs, err = K.run_mongo(mql, tables, missing=missing)
        rows = None if docs is None else K.rows_of(docs, cols)
        rec[rep] = {"rows": rows, "error": err, "pass": judge(case, rows, expected)}
    # what the existing verifier concludes on its own fixtures
    pid = "nk_" + hashlib.sha256(case["sql"].encode()).hexdigest()[:12]
    try:
        vr = K.V.verify_pair({"pair_id": pid, "question": case["id"], "sql": case["sql"]}, SCHEMA)
        rec["existing_verifier"] = {"status": vr["status"], "reasons": vr.get("reasons", [])[:3]}
    except Exception as e:  # noqa: BLE001
        rec["existing_verifier"] = {"status": "error", "reasons": [f"{type(e).__name__}: {str(e)[:200]}"]}
    return rec


def udf_checks():
    import datetime as dt
    udfs = K.S.make_udfs(dt.datetime(2026, 10, 25, 12, 0, 0))
    out = []
    for name, args, want in UDF_CHECKS:
        got = udfs[name][1](*args)
        out.append({"udf": name, "args": args, "expected_mysql": want, "got": got, "pass": got == want})
    for expr, want in MONGO_DATE_CHECKS:
        d = list(K.V.mongo_db().aggregate([{"$documents": [{}]}, {"$project": {
            "_id": 0, "v": {"$dateToString": {"format": "%Y-%m-%d", "date": expr}}}}]))[0]["v"]
        out.append({"mongo": expr, "expected_mysql": want, "got": d, "pass": d == want})
    return out


def main():
    N = K.server_clock()
    recs = [run_case(c, N) for c in CASES]
    K.write_jsonl(K.OUT / "adversarial.jsonl", recs)
    u = udf_checks()
    (K.OUT / "adversarial_udf.json").write_text(json.dumps(u, indent=1, default=str), encoding="utf-8")
    for r in recs:
        if r["converter"] != "converted":
            print(f"{r['id']:5} UNSUPPORTED  sqlite_ok={r['sqlite_agrees']}  verifier=-  {r['reason'][:70]}")
            continue
        e, m = r["explicit_null"], r["missing_field"]
        print(f"{r['id']:5} null={'PASS' if e['pass'] else 'FAIL'} missing={'PASS' if m['pass'] else 'FAIL'} "
              f"sqlite_ok={r['sqlite_agrees']} verifier={r['existing_verifier']['status']}"
              + (f"  got={e['rows'] if e['rows'] is not None else e['error']}" if not e["pass"] else ""))
    print("UDF/date checks:", sum(x["pass"] for x in u), "/", len(u), [x for x in u if not x["pass"]])


if __name__ == "__main__":
    main()
