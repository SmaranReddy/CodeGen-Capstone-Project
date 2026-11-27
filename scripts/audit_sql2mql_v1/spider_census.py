"""Original-database census: sql2mql-v1 on Spider 1.0 train_spider + train_others.

    python -B scripts/audit_sql2mql_v1/spider_census.py [--limit N] [--workers 6]

Ground truth = the original SQL executed on the ORIGINAL Spider SQLite database (read-only).
MongoDB side = the converter's MQL over the same database's rows, shipped inline with
`$documents` exactly as SQLite stores them (types preserved; NULL as explicit null, and again
as absent fields). Spider dev / the 455 benchmark (and its 65-item test partition) are never
read: the script asserts that no train db_id occurs in data/splits_expanded/{validation,test}.

Identifier case: SQLite resolves table/column names case-insensitively; MongoDB field names
are case-sensitive and the converter copies names verbatim. Names are canonicalised to the
schema's spelling before conversion; how many queries needed it is reported separately.
Evidence label: `audit_original_db` (Spider-train distribution; not an estimate for Kaggle).
"""
import argparse
import collections
import concurrent.futures as cf
import json
import sqlite3

from sqlglot import exp

import common as K

SP = K.ROOT / "data/raw/spider1/spider_data"
OUT = K.OUT / "spider_census.jsonl"
MAX_PAYLOAD = 12_000_000  # bytes of JSON rows per query ($documents rides in one command)


def guard_held_out(dbs):
    held = set()
    for f in ("test", "validation"):
        for line in open(K.ROOT / f"data/splits_expanded/{f}.jsonl", encoding="utf-8"):
            held.add(json.loads(line)["db_id"])
    dev = {x["db_id"] for x in json.load(open(SP / "dev.json", encoding="utf-8"))}
    assert not (dbs & held), f"held-out db overlap: {dbs & held}"
    assert not (dbs & dev), f"dev db overlap: {dbs & dev}"


def items():
    seen, out = set(), []
    for f in ("train_spider.json", "train_others.json"):
        for i, x in enumerate(json.load(open(SP / f, encoding="utf-8"))):
            key = (x["db_id"], x["query"].strip())
            if key not in seen:
                seen.add(key)
                out.append({"id": f"{f.split('.')[0]}:{i}", "db": x["db_id"], "sql": x["query"].strip(),
                            "question": x["question"]})
    return out


def map_type(decl):
    d = (decl or "").upper()
    if "INT" in d and "POINT" not in d:
        return "int"
    if any(k in d for k in ("REAL", "FLOA", "DOUB", "DEC", "NUMERIC", "NUMBER")):
        return "float"
    if "BOOL" in d:
        return "bool"
    if "DATETIME" in d or "TIMESTAMP" in d:
        return "datetime"
    if "DATE" in d:
        return "date"
    return "text"


_db_cache = {}


def load_db(db):
    if db in _db_cache:
        return _db_cache[db]
    path = SP / "database" / db / f"{db}.sqlite"
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    con.text_factory = lambda b: b.decode("utf-8", errors="replace")
    schema, rows = {}, {}
    for (t,) in con.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"):
        cols = con.execute(f'PRAGMA table_info("{t}")').fetchall()
        schema[t] = {c[1]: map_type(c[2]) for c in cols}
        cur = con.execute(f'SELECT * FROM "{t}"')
        names = [d[0] for d in cur.description]
        rows[t] = [dict(zip(names, r)) for r in cur.fetchall()]
    con.close()
    if len(_db_cache) > 12:  # items are processed grouped by db; keep memory bounded
        _db_cache.pop(next(iter(_db_cache)))
    _db_cache[db] = (path, schema, rows)
    return _db_cache[db]


def canonicalise(ast, schema):
    """Rewrite table and column names to the schema's spelling, resolving each column in its
    own table (qualifier -> alias -> table; unqualified -> the single table holding it).
    -> (ast, changed)."""
    tmap = {t.lower(): t for t in schema}
    changed = [False]
    alias = {}
    for t in ast.find_all(exp.Table):
        real = tmap.get(t.name.lower())
        if real:
            alias[t.name.lower()] = real
            if t.alias:
                alias[t.alias.lower()] = real
    in_query = set(alias.values())

    def col_in(table, name):
        hits = [c for c in schema[table] if c.lower() == name.lower()]
        return hits[0] if hits else None

    def tr(n):
        if isinstance(n, exp.Table) and n.name.lower() in tmap and n.name != tmap[n.name.lower()]:
            changed[0] = True
            n.set("this", exp.to_identifier(tmap[n.name.lower()]))
        elif isinstance(n, exp.Column) and not isinstance(n.this, exp.Star):
            if n.table:
                t = alias.get(n.table.lower())
                real = col_in(t, n.name) if t else None
            else:
                hits = {col_in(t, n.name) for t in in_query} - {None}
                real = next(iter(hits)) if len(hits) == 1 else None
            if real and real != n.name:
                changed[0] = True
                n.set("this", exp.to_identifier(real))
        return n
    return ast.transform(tr), changed[0]


def diagnostics(an, rows):
    """Data-level properties that make SQLite and MongoDB disagree regardless of the converter."""
    d = {}
    def types(t, c):
        return sorted({type(r.get(c)).__name__ for r in rows.get(t, []) if r.get(c) is not None})
    mixed = []
    for t, cols in an.columns.items():
        for c in cols:
            ty = types(t, c)
            if len(ty) > 1:
                mixed.append(f"{t}.{c}:{'/'.join(ty)}")
    d["mixed_type_columns"] = mixed
    jk = []
    for j in an.joins:
        l, r = j["on"]
        def tc(x):
            q, _, c = x.rpartition(".")
            return an.aliases.get(q, an.base) if q else an.base, c
        (lt, lc), (rt, rc) = tc(l), tc(r)
        if types(lt, lc) != types(rt, rc):
            jk.append(f"{lt}.{lc}:{types(lt, lc)} vs {rt}.{rc}:{types(rt, rc)}")
    d["join_key_type_mismatch"] = jk
    return d


def real_db_rows(path, sql):
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    con.text_factory = lambda b: b.decode("utf-8", errors="replace")
    try:
        cur = con.execute(sql)
        return [tuple(r) for r in cur.fetchall()]
    finally:
        con.close()


def judge(ast, an, mql, path, tables, schema, missing, gold_sql):
    gold = real_db_rows(path, gold_sql)  # the ORIGINAL Spider SQL text on the original database
    docs, err = K.run_mongo(mql, tables, missing=missing)
    if docs is None:
        return "mongo_error", err
    cols = K.out_columns(mql, schema, an.base)
    g = [tuple(K.norm(x) for x in r) for r in K.rows_of(docs, cols)]
    e = [tuple(K.norm(x) for x in r) for r in gold]
    if e and g and len(e[0]) != len(g[0]):
        return "fail", f"column count {len(g[0])} != {len(e[0])}"
    ordered = ast.args.get("order") is not None
    if an.limit is not None and not ordered:
        m = ast.copy()
        m.set("limit", None)
        m.set("offset", None)
        sup = [tuple(K.norm(x) for x in r) for r in real_db_rows(path, m.sql(dialect="sqlite"))]
        ok = len(g) == len(e) and all(g.count(r) <= sup.count(r) for r in set(g))
        return ("pass" if ok else "fail"), "limit-without-order superset"
    if len(g) != len(e):
        return "fail", f"rows {len(g)} != {len(e)}"
    if not ordered:
        return ("pass" if sorted(g, key=repr) == sorted(e, key=repr) else "fail"), "multiset"
    if g == e:
        return "pass", "sequence"
    # tie-aware: sort keys from the real DB
    sk = K.S.sort_key_sql(ast)
    if sk is None:
        return "inconclusive", "order differs; sort keys unavailable"
    m = ast.copy()
    keys_sql = sk[0]
    try:
        krows = real_db_rows(path, keys_sql)
    except Exception:  # noqa: BLE001
        return "inconclusive", "sort-key query failed"
    keys = [tuple(r[-sk[1]:]) for r in krows]
    if an.limit is not None:
        m.set("limit", None)
        m.set("offset", None)
        sk2 = K.S.sort_key_sql(m)
        allk = [tuple(r[-sk2[1]:]) for r in real_db_rows(path, sk2[0])]
        end, start = (an.offset or 0) + an.limit, an.offset or 0
        if (end < len(allk) and allk[end - 1] == allk[end]) or (0 < start < len(allk) and allk[start - 1] == allk[start]):
            return "inconclusive", "tie at LIMIT boundary"
    blocks, b, prev = [], -1, object()
    for k in keys:
        if k != prev:
            b += 1
            prev = k
        blocks.append(b)
    if len(blocks) != len(e):
        return "inconclusive", "sort-key rows misaligned"
    for bb in set(blocks):
        ix = [i for i, x in enumerate(blocks) if x == bb]
        if sorted((g[i] for i in ix), key=repr) != sorted((e[i] for i in ix), key=repr):
            return "fail", "tie-aware sequence"
    return "pass", "tie-aware sequence"


MAX_DB_BYTES = 20_000_000


def audit_one(it):
    rec = dict(it)
    dbfile = SP / "database" / it["db"] / f"{it['db']}.sqlite"
    if dbfile.stat().st_size > MAX_DB_BYTES:
        return dict(rec, status="skipped_payload", reason=f"database file {dbfile.stat().st_size} bytes")
    path, schema, rows = load_db(it["db"])
    ast, dialect, err = K.S.parse_sql(it["sql"])
    if ast is None:
        return dict(rec, status="parse_error", reason=err)
    ast, changed = canonicalise(ast, schema)
    rec["identifier_case_rewritten"] = changed
    an = K.S.analyze(ast)
    if an.unsupported:
        return dict(rec, status="unsupported", reason=";".join(an.unsupported))
    if any(t not in schema for t in an.columns):
        return dict(rec, status="unsupported", reason="unknown table after canonicalisation")
    sch = {t: schema[t] for t in an.columns}
    try:
        mql = K.M.convert(ast, an, sch)
    except K.M.Unsupported as e:
        return dict(rec, status="unsupported", reason=f"converter:{e}")
    tables = {t: rows[t] for t in an.columns}
    size = len(json.dumps(tables, default=str))
    if size > MAX_PAYLOAD:
        return dict(rec, status="skipped_payload", reason=f"{size} bytes")
    rec["mql"] = mql
    rec["features"] = sorted(an.features)
    rec["diagnostics"] = diagnostics(an, tables)
    for rep, missing in (("explicit_null", False), ("missing_field", True)):
        try:
            rec[rep] = judge(ast, an, mql, path, tables, sch, missing, it["sql"])
        except Exception as e:  # noqa: BLE001
            rec[rep] = ("gold_error", f"{type(e).__name__}: {str(e)[:200]}")
    rec["status"] = "tested"
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    its = items()
    guard_held_out({x["db"] for x in its})
    done = {x["id"] for x in K.read_jsonl(OUT)}
    todo = [x for x in its if x["id"] not in done][: a.limit]
    print("unique (db, sql):", len(its), "todo:", len(todo))
    # group by db so each worker loads a database once
    todo.sort(key=lambda x: x["db"])

    def work(it):
        try:
            return audit_one(it)
        except Exception as e:  # noqa: BLE001
            return dict(it, status="error", reason=f"{type(e).__name__}: {str(e)[:300]}")

    n = 0
    with cf.ThreadPoolExecutor(a.workers) as ex:
        for rec in ex.map(work, todo):
            K.append_jsonl(OUT, rec)
            n += 1
            if n % 250 == 0:
                print(n, flush=True)
    recs = K.read_jsonl(OUT)
    print(collections.Counter(r["status"] for r in recs))
    print("explicit", collections.Counter(r["explicit_null"][0] for r in recs if r["status"] == "tested"))
    print("missing", collections.Counter(r["missing_field"][0] for r in recs if r["status"] == "tested"))


if __name__ == "__main__":
    main()
