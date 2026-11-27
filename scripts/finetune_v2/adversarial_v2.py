"""Adversarial regression suite for sql2mql-v2 (V1, G2/G3 of the frozen criteria).

Cases: the 103 hand-specified cases of the 2026-10-25 audit (scripts/audit_sql2mql_v1/
adversarial_cases.py, imported as DATA only; no audit code that touches MongoDB is run)
plus the v2 cases in adversarial_cases_v2.py. Each case gets its own SQLite file
(outputs/sql2mql_v2/adversarial/dbs/adv_<id>/adv_<id>.sqlite) whose column affinities follow
the case schema; the verifier imports it into local MongoDB like a Spider database.

Expectation per case = the audit's hand expectation, unless the case is MySQL-specific: then
the expectation is re-derived by hand from documented SQLite behaviour and recorded in
RESOLUTIONS (never silently replaced by SQLite's output). Clock functions (NOW(), CURDATE(),
INTERVAL) do not exist in SQLite: the correct v2 outcome is rejection.

Subcommands
  negative   v1 converter MQL recorded for the audit's wrong cases (outputs/audit_sql2mql_v1/
             adversarial.jsonl) -> v2 verifier; every one must be rejected (G3)
  suite      v2 converter on every case -> v2 verifier; judge results against the expectation;
             a FALSE ACCEPT is a verifier-accepted MQL whose result differs from it (G2)
"""
import argparse
import datetime as dt
import json
import sqlite3
import sys

import v2_common as C
import comparator as CMP

sys.dont_write_bytecode = True
sys.path.insert(0, str(C.ROOT / "scripts/audit_sql2mql_v1"))
import adversarial_cases as AC  # noqa: E402  (pure data + datetime)

OUT = C.OUT / "adversarial"
DBS = OUT / "dbs"
CLOCK = dt.datetime(2026, 10, 30, 12, 0, 0)
DDL_TYPE = {"int": "INTEGER", "float": "REAL", "text": "TEXT", "bool": "BOOLEAN", "date": "DATE",
            "datetime": "DATETIME"}

# Cases whose audit expectation follows MySQL where SQLite differs; expectation by hand from
# the SQLite documentation (https://sqlite.org/datatype3.html, lang_expr.html, lang_aggfunc.html).
RESOLUTIONS = {
    "W16": ("[1]", "SQLite LIKE is ASCII-case-insensitive, but `=` uses BINARY collation: case-sensitive"),
    "A10": ("'Bob'", "BINARY collation: 'B' (0x42) < 'a' (0x61)"),
    "O11": ("binary order", "ORDER BY text uses BINARY collation (byte order)"),
    "N1": ("7/2 = 3, x/0 = NULL", "integer / integer is integer division in SQLite; division by zero is NULL"),
}


def schema_of(case):
    return AC.SCHEMA


def tables_of(case):
    t = case["tables"]
    return t(CLOCK) if callable(t) else t


def db_id(case):
    return "adv_" + case["id"].replace("@", "_")


def db_path(dbid):
    return DBS / dbid / f"{dbid}.sqlite"


def build_db(case):
    path = db_path(db_id(case))
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    con = sqlite3.connect(path)
    for t, rows in tables_of(case).items():
        cols = list(AC.SCHEMA[t])
        con.execute(f'CREATE TABLE "{t}" (' + ", ".join(f'"{c}" {DDL_TYPE[AC.SCHEMA[t][c]]}' for c in cols) + ")")
        con.executemany(f'INSERT INTO "{t}" VALUES ({", ".join("?" for _ in cols)})',
                        [tuple(r.get(c) for c in cols) for r in rows])
    con.commit()
    con.close()
    return path


def expectation(case):
    e = case.get("expect")
    return e(CLOCK) if callable(e) else e


def judge(case, rows):
    """rows: positional result rows; -> bool against the case expectation."""
    if rows is None:
        return False
    rows = CMP.norm_rows(rows)
    if case.get("subset_k"):
        base = [(r["id"],) for r in next(iter(tables_of(case).values()))]
        return len(rows) == case["subset_k"] and len(set(rows)) == len(rows) and all(r in base for r in rows)
    if case.get("any_of"):
        return any(len(rows) == len(a) and all(CMP.row_eq(x, tuple(y)) for x, y in zip(rows, a))
                   for a in case["any_of"])
    exp_rows = [tuple(r) for r in expectation(case)]
    if case.get("ordered"):
        return len(rows) == len(exp_rows) and all(CMP.row_eq(x, y) for x, y in zip(rows, exp_rows))
    return CMP.multiset_eq(rows, exp_rows)


def sqlite_native(case):
    con = sqlite3.connect(f"file:{db_path(db_id(case)).as_posix()}?mode=ro", uri=True)
    try:
        return [tuple(r) for r in con.execute(case["sql"])], None
    except sqlite3.Error as e:
        return None, f"{type(e).__name__}: {e}"
    finally:
        con.close()


def rename(mql, case):
    """v1 MQL named collections nl2sqlkaggle__<t>; the v2 harness uses adv_<id>__<t>."""
    s = json.dumps(mql).replace('"nl2sqlkaggle__', f'"{db_id(case)}__')
    return json.loads(s)


def all_cases():
    import adversarial_cases_v2 as A2
    return [dict(c, origin="audit_2026_10_25") for c in AC.CASES] + [dict(c, origin="v2") for c in A2.CASES]


def cmd_negative(client):
    import verifier as V
    recs = [json.loads(x) for x in open(C.ROOT / "outputs/audit_sql2mql_v1/adversarial.jsonl", encoding="utf-8")]
    cases = {c["id"]: c for c in AC.CASES}
    wrong = [r for r in recs if r.get("converter") == "converted"
             and (not r["explicit_null"]["pass"] or not r["missing_field"]["pass"])]
    real = V.RealDbs(client, path_of=db_path)
    out = []
    for r in wrong:
        case = cases[r["id"]]
        build_db(case)
        mql = rename(r["mql"], case)
        res = V.verify({"id": r["id"], "db_id": db_id(case), "sql": case["sql"]}, mql, real)
        out.append({"id": r["id"], "v1_wrong_under": [k for k in ("explicit_null", "missing_field")
                                                     if not r[k]["pass"]],
                    "v1_mql": mql, "v2_verifier": res["status"], "detail": res.get("detail"),
                    "checks": res["checks"]})
    C.write_jsonl(OUT / "negative_controls_v1_mql.jsonl", out)
    acc = [o["id"] for o in out if o["v2_verifier"] == "accepted"]
    summ = {"controls": len(out), "rejected": len(out) - len(acc), "false_accepts": acc,
            "by_status": {s: sum(o["v2_verifier"] == s for o in out) for s in sorted({o["v2_verifier"] for o in out})}}
    (OUT / "negative_controls_v1_mql.summary.json").write_text(json.dumps(summ, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(summ, indent=1))
    return summ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("negative", "suite"))
    a = ap.parse_args()
    client = C.local_client()
    C.check_server(client)
    if a.cmd == "negative":
        cmd_negative(client)
    else:
        import suite_v2
        suite_v2.run(client)


if __name__ == "__main__":
    main()
