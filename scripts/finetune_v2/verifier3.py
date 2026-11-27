"""VERIFIER SIDE. sql2mql-v2 verifier, iteration 3 (2026-10-31). A strict SUPERSET of verifier-2
(verifier.py, unchanged): every verifier-2 check runs first, with the same code (imported), and
a pair rejected by verifier-2 is rejected here. Added checks:

  v3_reorder     the real database again with every referenced table stored in REVERSED rowid
                 order (NULL as explicit null) and in a seeded SHUFFLED order (NULL as absent
                 field). The SQL truth is order-free except for tie order (tie-aware comparison
                 as before); MQL whose answer depends on natural document order is rejected.
  fixtures       every verifier-2 family, plus fixtures3 families: reversed / shuffled row order
                 for six families, in-place dangling join keys (one per join-key column, value
                 and NULL variants), consistent per-join-group key shifts, NULL join keys on both
                 sides of every edge (also as absent fields).
  v3_lookup_keys every $lookup localField / foreignField must name a column (exact case)
                 of the collection it is read from: a lookup on a column the table does not have
                 (e.g. swapped localField/foreignField) is rejected structurally.

Fixture / reorder collections live in databases of their own (sql2mql_v3_*), so a verifier-3
run never shares collections with a verifier-2 run.
Must not import the converter (sql2mql_v2.py, schema_profile.py).
"""
import random
import time

import v2_common as C
import verifier as V2
import fixtures3 as FX
import mongo_import as MI
import index_real as IX
from verifier import Reject, run_sql, run_mql, out_columns, mql_collections, order_plan, \
    sqlite_truth, compare, witnesses  # noqa: F401  (verifier-2 code, unchanged)

VERIFIER_VERSION = "sql2mql-v2-verifier-3"
IMPORT_VERSION = V2.IMPORT_VERSION
FIXTURE_DB = "sql2mql_v3_fixtures"
REORDER_DB = "sql2mql_v3_reorder"
MISSING_FAMILIES = V2.MISSING_FAMILIES + ("null_keys_both",)
BIG = 2000


class RealDbs(V2.RealDbs):
    """verifier-2 real-database cache + key indexes on large collections (result-invariant)
    + reversed / shuffled copies for v3_reorder."""

    def __init__(self, client, path_of=C.sqlite_path):
        super().__init__(client, path_of)
        self.indexed, self.reordered = set(), {}

    def get(self, db_id):
        got = super().get(db_id)
        if db_id not in self.indexed:
            for dbname in (C.REAL_DB, C.REAL_DB + "_missing"):
                self._index(self.client[dbname], db_id, got[1])
            self.indexed.add(db_id)
        return got

    def _index(self, mdb, db_id, tables):
        for t, (_, rows) in tables.items():
            if len(rows) > BIG:
                for c in IX.key_columns(db_id, t):
                    mdb[C.coll(db_id, t)].create_index([(c, 1)])

    def reorder(self, db_id, tables):
        """Load `tables` of db_id reversed (explicit null) / shuffled (missing), once per db."""
        key = (db_id, tuple(sorted(tables)))
        if self.reordered.get(db_id) == key:
            return
        _, real, _ = self.get(db_id)
        rev = {t: (real[t][0], list(reversed(real[t][1]))) for t in tables}
        shuf = {}
        for t in tables:
            r = list(real[t][1])
            random.Random(f"reorder:{db_id}:{t}").shuffle(r)
            shuf[t] = (real[t][0], r)
        for missing, data in ((False, rev), (True, shuf)):
            mdb = self.client[REORDER_DB + ("_missing" if missing else "")]
            MI.load(mdb, db_id, data, missing)
            self._index(mdb, db_id, data)
        self.reordered[db_id] = key


def verify(item, mql, real_dbs, seed=20261030, fixtures=True):
    rec = {"id": item["id"], "db_id": item["db_id"], "verifier": VERIFIER_VERSION, "checks": {}}
    t0 = time.time()
    try:
        _verify(item, mql, real_dbs, seed, fixtures, rec)
        rec["status"] = "accepted"
    except Reject as e:
        rec["status"], rec["detail"] = e.status, e.detail
    rec["seconds"] = round(time.time() - t0, 2)
    return rec


def _lookup_keys(mql, real):
    """v3_lookup_keys: each $lookup localField must be (exactly, case-sensitive like MongoDB) a
    column of the start collection's table or `<as>.<column>` of an earlier lookup; each
    foreignField must be a column of the `from` table. A field that exists nowhere makes
    MongoDB match null/missing to null/missing (a silent cross join), so it is never accepted."""
    fields = set(real[mql["collection"].split("__", 1)[1]][0])
    alias_cols = {}
    for st in mql["pipeline"]:
        lk = st.get("$lookup") if isinstance(st, dict) else None
        if not isinstance(lk, dict) or "localField" not in lk:
            continue
        fcols = set(real[lk["from"].split("__", 1)[1]][0])
        lf = lk["localField"]
        head, _, rest = lf.partition(".")
        if not (lf in fields or (rest and rest in alias_cols.get(head, set()))):
            raise Reject("v3_lookup_keys", f"localField {lf!r} is not a field of the joined documents")
        if lk["foreignField"] not in fcols:
            raise Reject("v3_lookup_keys", f"foreignField {lk['foreignField']!r} is not a column of the from table")
        alias_cols[lk["as"]] = fcols
    return "ok"


def _verify(item, mql, real_dbs, seed, fixtures, rec):
    db = item["db_id"]
    if not isinstance(mql, dict) or not isinstance(mql.get("collection"), str):
        raise Reject("structure", "not an MQL object")
    path, real, info = real_dbs.get(db)
    allowed = {C.coll(db, t) for t in real}
    ok, why, _ = V2.validate_mql(mql, allowed)
    rec["checks"]["v8_validate_mql"] = why
    if not ok:
        raise Reject("unsafe_mql", why)
    cols = out_columns(mql)
    try:
        sql_tables = FX.referenced_tables(item["sql"], info)
    except Exception as e:  # noqa: BLE001
        raise Reject("sql_parse", str(e)[:200]) from e
    mql_tables = {c.split("__", 1)[1] for c in mql_collections(mql)}
    if not mql_tables <= sql_tables:
        raise Reject("structure", f"MQL reads tables the SQL does not: {sorted(mql_tables - sql_tables)}")
    rec["checks"]["v3_lookup_keys"] = _lookup_keys(mql, real)
    for t in sql_tables:
        if not real[t][1]:
            raise Reject("empty_referenced_table", t)
        if any(C.has_bad_text(v) for r in real[t][1] for v in r):
            raise Reject("bad_text", f"invalid UTF-8 in {t}")
    plan = order_plan(item["sql"])
    rec["checks"]["order"] = {k: plan[k] for k in ("ordered", "limit", "offset", "keyable")}

    con = C.sqlite_ro(path)
    try:
        truth = sqlite_truth(con, item["sql"], plan)
    finally:
        con.close()
    if truth.get("tie_cut"):
        raise Reject("tie_cut", "ORDER BY ... LIMIT cuts through a tie on the real database")
    rec["checks"]["real_rows"] = len(truth["rows"])
    for rep, dbname in (("explicit_null", C.REAL_DB), ("missing", C.REAL_DB + "_missing")):
        docs = run_mql(real_dbs.client[dbname], mql)
        ok, why, _ = compare(docs, cols, truth, plan)
        rec["checks"][f"l1_{rep}"] = why
        if not ok:
            raise Reject(f"l1_mismatch_{rep}", why)
        if rep == "explicit_null":
            again = run_mql(real_dbs.client[dbname], mql)
            ok2, why2, _ = compare(again, cols, truth, plan)
            if not ok2:
                raise Reject("nondeterministic", f"second execution differs: {why2}")
            rec["checks"]["v6_determinism"] = "identical documents" if again == docs else \
                "same result, different document order"
    # v3_reorder: real data in reversed / shuffled storage order
    real_dbs.reorder(db, sql_tables)
    for rep, dbname in (("reversed", REORDER_DB), ("shuffled_missing", REORDER_DB + "_missing")):
        docs = run_mql(real_dbs.client[dbname], mql)
        ok, why, _ = compare(docs, cols, truth, plan)
        rec["checks"][f"v3_reorder_{rep}"] = why
        if not ok:
            raise Reject("v3_reorder_mismatch", f"{rep}: {why}")
    if not fixtures:
        return
    nonempty = bool(truth["rows"])
    con = C.sqlite_ro(path)
    try:
        wit = witnesses(con, item["sql"], info)
    finally:
        con.close()
    rec["checks"]["witness_rows"] = {t: len(v) for t, v in wit.items()}
    builder = FX.make_builder(info, sql_tables, item["sql"], f"{seed}:{item['id']}", wit)
    families = FX.FAMILIES + FX.extra_families(builder)
    fam_rec = {}
    for fam in families:
        fcon, stored, problems = builder.materialize(fam)
        try:
            if problems:
                fam_rec[fam] = "skipped: " + problems[0]
                continue
            try:
                ftruth = sqlite_truth(fcon, item["sql"], plan)
            except Reject as e:
                if e.status == "limit_without_order":
                    raise
                fam_rec[fam] = f"skipped: {e.status}"
                continue
        finally:
            fcon.close()
        miss = fam in MISSING_FAMILIES or fam.startswith("dangle_null")
        for missing in ((False, True) if miss else (False,)):
            mdb = real_dbs.client[FIXTURE_DB + ("_missing" if missing else "")]
            MI.load(mdb, db, stored, missing)
            docs = run_mql(mdb, mql)
            ok, why, _ = compare(docs, cols, ftruth, plan)
            key = fam + ("@missing" if missing else "")
            fam_rec[key] = why if ok else f"FAIL {why}"
            if not ok:
                rec["checks"]["fixtures"] = fam_rec
                raise Reject("fixture_mismatch", f"{key}: {why}")
        nonempty = nonempty or bool(ftruth["rows"])
    rec["checks"]["fixtures"] = fam_rec
    used = [k for k, v in fam_rec.items() if not v.startswith("skipped")]
    if len(used) < 4:
        raise Reject("too_few_fixtures", f"only {len(used)} fixture runs")
    if not nonempty:
        raise Reject("non_discriminating", "SQL result empty on the real DB and every fixture")
