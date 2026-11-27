"""VERIFIER SIDE. Independent differential verifier for (SQL, candidate MQL) pairs.

Oracle: the ORIGINAL SQL text executed verbatim by SQLite. The verifier never translates SQL;
it only (a) runs SQL on SQLite, (b) runs the candidate MQL on the local MongoDB 8.0.34
(127.0.0.1:27018), (c) compares results with comparator.py. sqlglot is used only to list the
referenced tables, collect literals for fixtures, and derive ORDER BY keys / LIMIT for the
tie analysis (the derived key query is itself checked against the verbatim SQL).

A pair is ACCEPTED only if every check below passes:
  structure   MQL is {"collection", "pipeline"}, ends with an inclusion $project (the answer
              columns, in order), references only collections of tables the SQL references
  V8 safety   eval_kit.validate_mql(mql, collections of the database) passes unchanged
  data        every referenced table is non-empty in the real database; no invalid UTF-8
  L1 (V4)     real database, NULL as explicit null AND as an absent field
  ties (V5)   if ORDER BY ... LIMIT/OFFSET cuts through a tie block on the real database:
              rejected (`tie_cut`); elsewhere ordering is compared tie-aware
  V6          the real-database MQL run is repeated and must return identical documents
  fixtures    every family in fixtures.FAMILIES that materialises without storage-class drift
              (explicit null; null_mix/all_null/sample_a also as absent fields)
  discrimin.  if the real result is empty, at least one fixture must give a non-empty result
LIMIT without ORDER BY is rejected unless the unlimited result already fits in the limit.

Must not import the converter (sql2mql_v2.py, schema_profile.py).
"""
import sqlite3
import sys
import time

from sqlglot import exp, parse_one

import v2_common as C
import comparator as CMP
import v2_fixtures as FX
import mongo_import as MI

sys.path.insert(0, str(C.ROOT / "scripts/evaluation"))
sys.path.insert(0, str(C.ROOT / "scripts/spider1"))
from eval_kit import validate_mql  # noqa: E402

VERIFIER_VERSION = "sql2mql-v2-verifier-2"   # -1: iteration 1 (2026-10-30), failed G4; see docs
IMPORT_VERSION = "import-1"                  # representation contract of mongo_import.py
WITNESS_MAX = 40
MISSING_FAMILIES = ("null_mix", "all_null", "sample_a", "all_null_keep", "null_atoms_even", "null_keys_even")
SQLITE_TIMEOUT_S = 30


class Reject(Exception):
    def __init__(self, status, detail=""):
        super().__init__(f"{status}: {detail}")
        self.status, self.detail = status, detail


# ---------------------------------------------------------------- execution
def run_sql(con, sql):
    deadline = time.time() + SQLITE_TIMEOUT_S
    con.set_progress_handler(lambda: 1 if time.time() > deadline else 0, 10000)
    try:
        cur = con.execute(sql)
        rows = [tuple(r) for r in cur.fetchall()]
    except sqlite3.OperationalError as e:
        raise Reject("sqlite_error", str(e)[:200]) from e
    finally:
        con.set_progress_handler(None, 0)
    for r in rows:
        for v in r:
            if C.has_bad_text(v):
                raise Reject("bad_text", "invalid UTF-8 in SQLite result")
    return rows


def run_mql(mdb, mql):
    from pymongo.errors import ExecutionTimeout, OperationFailure
    try:
        return list(mdb[mql["collection"]].aggregate(mql["pipeline"], maxTimeMS=C.TIMEOUT_MS))
    except ExecutionTimeout as e:
        raise Reject("mongo_timeout", str(e)[:200]) from e
    except OperationFailure as e:
        raise Reject("mongo_error", f"{e.code}: {str(e)[:200]}") from e


# ---------------------------------------------------------------- structure
def _walk(x):
    if isinstance(x, dict):
        yield x
        for v in x.values():
            yield from _walk(v)
    elif isinstance(x, list):
        for v in x:
            yield from _walk(v)


def mql_collections(mql):
    out = {mql["collection"]}
    for d in _walk(mql["pipeline"]):
        if "$lookup" in d and isinstance(d["$lookup"], dict) and "from" in d["$lookup"]:
            out.add(d["$lookup"]["from"])
        if "$unionWith" in d:
            u = d["$unionWith"]
            if isinstance(u, str):
                out.add(u)
            elif isinstance(u, dict) and u.get("coll"):
                out.add(u["coll"])
    return out


def out_columns(mql):
    pipe = mql.get("pipeline")
    if not isinstance(pipe, list) or not pipe or not isinstance(pipe[-1], dict) or list(pipe[-1]) != ["$project"]:
        raise Reject("structure", "pipeline must end with a $project")
    proj = pipe[-1]["$project"]
    if proj.get("_id", 1) not in (0, False):
        raise Reject("structure", "final $project must exclude _id")
    cols = []
    for k, v in proj.items():
        if k == "_id":
            continue
        if v in (0, False):
            raise Reject("structure", "final $project must be an inclusion projection")
        cols.append(k)
    if not cols:
        raise Reject("structure", "final $project has no answer column")
    return cols


# ---------------------------------------------------------------- ordering
def _int_arg(node):
    if node is None:
        return None
    e = node.expression if isinstance(node, (exp.Limit, exp.Offset)) else node
    if isinstance(e, exp.Literal) and not e.is_string and e.this.isdigit():
        return int(e.this)
    raise Reject("order_plan", f"non-literal LIMIT/OFFSET {node.sql()}")


def order_plan(sql):
    tree = parse_one(sql, read="sqlite")
    limit, offset = _int_arg(tree.args.get("limit")), _int_arg(tree.args.get("offset"))
    order = tree.args.get("order")
    plan = {"ordered": order is not None, "limit": limit, "offset": offset, "keyable": False}
    nolimit = tree.copy()
    nolimit.set("limit", None)
    nolimit.set("offset", None)
    plan["nolimit_sql"] = nolimit.sql(dialect="sqlite")
    if order is None or not isinstance(tree, exp.Select):
        return plan
    sel = list(tree.expressions)
    aliases = {e.alias.lower(): e.this for e in sel if isinstance(e, exp.Alias)}
    keys = []
    for o in order.expressions:
        e = o.this
        if isinstance(e, exp.Literal) and not e.is_string:
            item = sel[int(e.this) - 1]
            e = item.this if isinstance(item, exp.Alias) else item
        elif isinstance(e, exp.Column) and not e.table and e.name.lower() in aliases:
            e = aliases[e.name.lower()]
        keys.append(e)
    k = nolimit.copy()
    if k.args.get("distinct"):
        bare = [(s.this if isinstance(s, exp.Alias) else s) for s in sel]
        pos = []
        for e in keys:
            hit = [i for i, b in enumerate(bare) if b == e]
            if not hit:
                return plan
            pos.append(hit[0])
        plan.update(keyable=True, key_positions=pos, key_sql=None)
        return plan
    k.set("expressions", list(k.expressions) + [exp.alias_(e.copy(), f"__k{i}") for i, e in enumerate(keys)])
    plan.update(keyable=True, key_sql=k.sql(dialect="sqlite"), n_keys=len(keys))
    return plan


def sqlite_truth(con, sql, plan):
    """-> dict(rows=verbatim result, keyed=full ordered rows with keys | None, n_cols)."""
    rows = run_sql(con, sql)
    truth = {"rows": rows}
    if plan["limit"] is not None or plan["offset"] is not None or plan["ordered"]:
        full = run_sql(con, plan["nolimit_sql"])
        truth["full"] = full
    if plan["ordered"]:
        if not plan["keyable"]:
            raise Reject("order_plan", "ORDER BY keys not derivable")
        if plan.get("key_positions") is not None:
            keyed = [r + tuple(r[i] for i in plan["key_positions"]) for r in truth["full"]]
            n = len(truth["full"][0]) if truth["full"] else 0
        else:
            keyed = run_sql(con, plan["key_sql"])
            n = len(keyed[0]) - plan["n_keys"] if keyed else 0
            if not CMP.multiset_eq(CMP.norm_rows([r[:n] for r in keyed]), CMP.norm_rows(truth["full"])):
                raise Reject("order_plan", "key query disagrees with the verbatim SQL")
        ok, why, cut = CMP.match_ordered(CMP.norm_rows(rows), CMP.norm_rows(keyed), n,
                                         plan["offset"], plan["limit"])
        if not ok:
            raise Reject("order_plan", f"verbatim result not a window of the keyed order ({why})")
        truth.update(keyed=keyed, n_cols=n, tie_cut=cut)
    elif plan["limit"] is not None and len(truth["full"]) > plan["limit"] + (plan["offset"] or 0):
        raise Reject("limit_without_order", "LIMIT without ORDER BY selects arbitrary rows")
    return truth


def compare(docs, cols, truth, plan):
    """-> (ok, reason, tie_cut)."""
    try:
        got = CMP.docs_to_rows(docs, cols)
    except CMP.BadValue as e:
        return False, f"bad value: {e}", False
    exp_rows = CMP.norm_rows(truth["rows"])
    if exp_rows and len(exp_rows[0]) != len(cols):
        return False, f"column count {len(cols)} != {len(exp_rows[0])}", False
    if plan["ordered"]:
        return CMP.match_ordered(got, CMP.norm_rows(truth["keyed"]), truth["n_cols"],
                                 plan["offset"], plan["limit"])
    return CMP.multiset_eq(got, exp_rows), "multiset", False


def witnesses(con, sql, info):
    """Real rows (per table) that take part in the query's joins and WHERE: the query is rewritten
    to `SELECT DISTINCT <alias>.rowid` without GROUP BY / HAVING / ORDER BY / LIMIT. Tables
    without rowid, or rewrites SQLite refuses, give no witnesses (sampling still applies)."""
    tree = parse_one(sql, read="sqlite")
    out = {}
    if not isinstance(tree, exp.Select):
        return out
    for t in tree.find_all(exp.Table):
        real = info.table_of(t.name)
        if not real or real in out:
            continue
        q = tree.copy()
        for k in ("group", "having", "order", "limit", "offset"):
            q.set(k, None)
        q.set("distinct", exp.Distinct())
        q.set("expressions", [exp.column("rowid", table=t.alias_or_name)])
        try:
            ids = [r[0] for r in run_sql(con, q.sql(dialect="sqlite") + f" LIMIT {WITNESS_MAX}")]
            cols = info.tables[real][0]
            qc = ", ".join(f'"{c}"' for c in cols)
            ph = ", ".join("?" for _ in ids)
            rows = [tuple(r) for r in con.execute(f'SELECT {qc} FROM "{real}" WHERE rowid IN ({ph}) ORDER BY rowid',
                                                  ids)] if ids else []
        except (Reject, sqlite3.Error):
            rows = []
        out[real] = rows
    return out


# ---------------------------------------------------------------- real database cache
class RealDbs:
    def __init__(self, client, path_of=C.sqlite_path):
        self.client, self.cache, self.imported, self.path_of = client, {}, {}, path_of

    def get(self, db_id):
        if db_id not in self.cache:
            if len(self.cache) > 6:
                self.cache.pop(next(iter(self.cache)))
            path = self.path_of(db_id)
            tables = MI.read_tables(path)
            self.cache[db_id] = (path, tables, FX.DbInfo(path, tables))
        if db_id not in self.imported:
            meta = self.client[C.REAL_DB]["_import_meta"]
            sha = C.sha256_file(self.path_of(db_id))
            m = meta.find_one({"_id": db_id})
            if not m or m.get("sha256") != sha or m.get("import") != IMPORT_VERSION:
                rep = MI.import_real(self.client, db_id, self.path_of(db_id))
                if rep["explicit_null"] or rep["missing"]:
                    raise RuntimeError(f"import parity failed for {db_id}: {rep}")
                meta.replace_one({"_id": db_id}, {"_id": db_id, "sha256": sha,
                                                  "import": IMPORT_VERSION, "tables": rep["tables"]},
                                 upsert=True)
            self.imported[db_id] = True
        return self.cache[db_id]


# ---------------------------------------------------------------- main entry
def verify(item, mql, real_dbs, seed=20261030, families=FX.FAMILIES, fixtures=True):
    """item: {id, db_id, sql}. -> evidence record with `status` ('accepted' or a reject)."""
    rec = {"id": item["id"], "db_id": item["db_id"], "verifier": VERIFIER_VERSION, "checks": {}}
    t0 = time.time()
    try:
        _verify(item, mql, real_dbs, seed, families, fixtures, rec)
        rec["status"] = "accepted"
    except Reject as e:
        rec["status"], rec["detail"] = e.status, e.detail
    rec["seconds"] = round(time.time() - t0, 2)
    return rec


def _verify(item, mql, real_dbs, seed, families, fixtures, rec):
    db = item["db_id"]
    if not isinstance(mql, dict) or not isinstance(mql.get("collection"), str):
        raise Reject("structure", "not an MQL object")
    path, real, info = real_dbs.get(db)
    allowed = {C.coll(db, t) for t in real}
    ok, why, _ = validate_mql(mql, allowed)
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
            # V6: a second execution must give the same result under the item's own rule
            # (multiset, or tie-aware order); document order of an unordered result may differ
            again = run_mql(real_dbs.client[dbname], mql)
            ok2, why2, _ = compare(again, cols, truth, plan)
            if not ok2:
                raise Reject("nondeterministic", f"second execution differs: {why2}")
            rec["checks"]["v6_determinism"] = "identical documents" if again == docs else \
                "same result, different document order"
    if not fixtures:
        return
    nonempty = bool(truth["rows"])
    con = C.sqlite_ro(path)
    try:
        wit = witnesses(con, item["sql"], info)
    finally:
        con.close()
    rec["checks"]["witness_rows"] = {t: len(v) for t, v in wit.items()}
    builder = FX.Builder(info, sql_tables, item["sql"], f"{seed}:{item['id']}", wit)
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
        for missing in ((False, True) if fam in MISSING_FAMILIES else (False,)):
            mdb = real_dbs.client[C.FIXTURE_DB + ("_missing" if missing else "")]
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
