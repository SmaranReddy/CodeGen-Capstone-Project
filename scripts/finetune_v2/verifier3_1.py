"""VERIFIER SIDE. sql2mql verifier 3.1 (2026-11-13) = verifier-3 (verifier3.py, imported unchanged) + the
fold-trap fixture families of fixtures3_1 (plan §B.4.3).

A strict SUPERSET of verifier-3: verifier3.verify runs first with the same arguments; any verifier-3
rejection is returned unchanged (only the version field is re-labelled). For an accepted pair whose SQL uses
LIKE, the fold_traps / fold_traps_dotted families then run under both NULL representations; a mismatch
rejects with status `fixture_mismatch` (detail prefixed `v31_fold_traps`).

Fixture collections live in databases of their own (sql2mql_v31_fixtures[_missing]).
Must not import the converter (sql2mql_v2*.py, schema_profile.py).
"""
import time

import v2_common as C
import verifier3 as V3
import fixtures3_1 as FX
import mongo_import as MI
from verifier import Reject, run_mql, out_columns, order_plan, sqlite_truth, compare, witnesses

VERIFIER_VERSION = "sql2mql-v2-verifier-3.1"
BASE_VERIFIER_VERSION = V3.VERIFIER_VERSION
FIXTURE_DB = "sql2mql_v31_fixtures"
RealDbs = V3.RealDbs


def verify(item, mql, real_dbs, seed=20261030, fixtures=True):
    rec = V3.verify(item, mql, real_dbs, seed=seed, fixtures=fixtures)
    rec["base_verifier"] = rec.get("verifier")
    rec["verifier"] = VERIFIER_VERSION
    if rec.get("status") != "accepted" or not fixtures:
        return rec
    if not FX.uses_like(item["sql"]):
        rec["checks"]["v31_fold_traps"] = "not applicable (no LIKE)"
        return rec
    t0 = time.time()
    try:
        rec["checks"]["v31_fold_traps"] = _fold_traps(item, mql, real_dbs, seed)
    except Reject as e:
        rec["status"], rec["detail"] = e.status, e.detail
    rec["seconds"] = round(rec.get("seconds", 0) + time.time() - t0, 2)
    return rec


def _fold_traps(item, mql, real_dbs, seed):
    db = item["db_id"]
    path, real, info = real_dbs.get(db)
    cols = out_columns(mql)
    sql_tables = FX.referenced_tables(item["sql"], info)
    plan = order_plan(item["sql"])
    con = C.sqlite_ro(path)
    try:
        wit = witnesses(con, item["sql"], info)
    finally:
        con.close()
    builder = FX.make_builder(info, sql_tables, item["sql"], f"{seed}:{item['id']}:v31", wit)
    out, used = {}, 0
    for fam in FX.FOLD_FAMILIES:
        fcon, stored, problems = builder.materialize(fam)
        try:
            if problems:
                out[fam] = "skipped: " + problems[0]
                continue
            try:
                ftruth = sqlite_truth(fcon, item["sql"], plan)
            except Reject as e:
                out[fam] = f"skipped: {e.status}"
                continue
        finally:
            fcon.close()
        for missing in (False, True):
            mdb = real_dbs.client[FIXTURE_DB + ("_missing" if missing else "")]
            MI.load(mdb, db, stored, missing)
            docs = run_mql(mdb, mql)
            ok, why, _ = compare(docs, cols, ftruth, plan)
            key = fam + ("@missing" if missing else "")
            out[key] = why if ok else f"FAIL {why}"
            if not ok:
                raise Reject("fixture_mismatch", f"v31_fold_traps {key}: {why}")
            used += 1
    if used == 0:
        out["note"] = "no fold-trap family applicable (degenerate); verifier-3 verdict stands"
    return out
