"""`adversarial_v2.py suite`: sql2mql-v2 converter + verifier on every adversarial case.

Per case:
  * SQLite cross-check: the case SQL run natively vs the (resolved) hand expectation;
  * converter: MQL or a rejection code;
  * TRUTH for the MQL: its result on the case database (explicit null and absent-field NULL),
    judged against the hand expectation, independently of the verifier;
  * verifier verdict.
FALSE ACCEPT = verifier accepted an MQL that is wrong under either NULL representation.
Writes outputs/sql2mql_v2/adversarial/suite.jsonl and suite.summary.json.
"""
import json
import sqlite3

import adversarial_v2 as A
import v2_common as C
import comparator as CMP
import mongo_import as MI
import schema_profile as SP
import sql2mql_v2 as M
import verifier as V

# hand-derived SQLite expectations replacing MySQL-semantics audit expectations
OVERRIDES = {
    "N1": [(1, 3), (2, None), (3, None), (4, 3)],
    "N1b": [(1, 3), (3, None), (4, 3)],   # audit expectation 3.5 is MySQL; no dialect note there
    "N3": [],
}


def build_db(case):
    if "schema" not in case:
        return A.build_db(case)
    path = A.db_path(A.db_id(case))
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    con = sqlite3.connect(path)
    for t, cols in case["schema"].items():
        con.execute(f'CREATE TABLE "{t}" (' + ", ".join(f'"{c}" {d}' for c, d in cols.items()) + ")")
        data = case["tables"].get(t, [])
        con.executemany(f'INSERT INTO "{t}" VALUES ({", ".join("?" for _ in cols)})',
                        [tuple(r.get(c) for c in cols) for r in data])
    con.commit()
    con.close()
    return path


def expected_case(case):
    if case["id"] in OVERRIDES:
        case = dict(case, expect=OVERRIDES[case["id"]], any_of=None, subset_k=None)
    return case


def run(client):
    real = V.RealDbs(client, path_of=A.db_path)
    out = []
    for case in A.all_cases():
        case = expected_case(case)
        dbid = A.db_id(case)
        build_db(case)
        rec = {"id": case["id"], "origin": case["origin"], "feature": case.get("feature"),
               "covers": case.get("covers", []), "sql": case["sql"],
               "expect_reject": bool(case.get("expect_reject")),
               "resolution": A.RESOLUTIONS.get(case["id"], (None, None))[1] or
               ("expectation overridden: SQLite integer division" if case["id"] in OVERRIDES else None)}
        if not rec["expect_reject"]:
            sq, err = A.sqlite_native(case)
            rec["sqlite_agrees"] = None if sq is None else A.judge(case, sq)
            rec["sqlite_error"] = err
        prof = SP.Profile(dbid, A.db_path(dbid))
        try:
            mql, feats = M.convert(case["sql"], prof)
        except M.Unsupported as e:
            rec.update(converter="rejected", code=e.code, detail=e.detail)
            out.append(rec)
            continue
        except Exception as e:  # noqa: BLE001
            rec.update(converter="crash", code=type(e).__name__, detail=str(e)[:200])
            out.append(rec)
            continue
        rec.update(converter="converted", mql=mql, features=feats)
        tables = MI.read_tables(A.db_path(dbid))
        cols = V.out_columns(mql)
        for missing in (False, True):
            mdb = client["sql2mql_v2_adv" + ("_missing" if missing else "")]
            MI.load(mdb, dbid, tables, missing)
            try:
                docs = list(mdb[mql["collection"]].aggregate(mql["pipeline"], maxTimeMS=C.TIMEOUT_MS))
                got = CMP.docs_to_rows(docs, cols)
                ok = None if rec["expect_reject"] else A.judge(case, got)
                rec["missing" if missing else "explicit_null"] = {"rows": got, "correct": ok}
            except Exception as e:  # noqa: BLE001
                rec["missing" if missing else "explicit_null"] = {"error": str(e)[:200], "correct": False}
        v = V.verify({"id": case["id"], "db_id": dbid, "sql": case["sql"]}, mql, real)
        rec["verifier"] = v["status"]
        rec["verifier_detail"] = v.get("detail")
        out.append(rec)
    C.write_jsonl(A.OUT / "suite.jsonl", out)
    conv = [r for r in out if r["converter"] == "converted"]
    correct = [r for r in conv if not r["expect_reject"] and r["explicit_null"]["correct"]
               and r["missing"]["correct"]]
    wrong = [r for r in conv if not r["expect_reject"] and not (r["explicit_null"]["correct"]
                                                                 and r["missing"]["correct"])]
    false_acc = [r["id"] for r in wrong if r["verifier"] == "accepted"]
    reject_violations = [r["id"] for r in conv if r["expect_reject"]]
    summ = {
        "cases": len(out), "audit_cases": sum(r["origin"] == "audit_2026_10_25" for r in out),
        "v2_cases": sum(r["origin"] == "v2" for r in out),
        "converter": {"converted": len(conv), "rejected": sum(r["converter"] == "rejected" for r in out),
                      "crash": [r["id"] for r in out if r["converter"] == "crash"]},
        "converted_correct_both_null_representations": len(correct),
        "converted_wrong": [r["id"] for r in wrong],
        "wrong_and_rejected_by_verifier": [r["id"] for r in wrong if r["verifier"] != "accepted"],
        "FALSE_ACCEPTS": false_acc,
        "expect_reject_but_converted": reject_violations,
        "correct_but_verifier_rejected": {r["id"]: r["verifier"] for r in correct if r["verifier"] != "accepted"},
        "sqlite_disagrees_with_expectation": [r["id"] for r in out if r.get("sqlite_agrees") is False],
        "sqlite_error_cases": [r["id"] for r in out if r.get("sqlite_error")],
    }
    (A.OUT / "suite.summary.json").write_text(json.dumps(summ, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(summ, indent=1))
    return summ
