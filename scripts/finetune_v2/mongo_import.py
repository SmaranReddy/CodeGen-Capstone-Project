"""VERIFIER SIDE. SQLite rows -> local MongoDB documents, and SQLite<->MongoDB parity.

Representation contract (identical to the benchmark's Atlas import, scripts/spider1/
import_atlas.py): one collection per table named `<db_id>__<table>`; every column is a field,
in table column order; values keep their SQLite storage class (INTEGER -> int32/int64,
REAL -> double, TEXT -> string, BLOB -> binary, NULL -> explicit null); documents are inserted
in rowid order. The `missing` representation is the same with every NULL field omitted
(F15: labels are verified under both).

Must not import the converter (sql2mql_v2.py, schema_profile.py).
"""
import sqlite3
from collections import Counter

import v2_common as C


def read_tables(path, tables=None):
    """-> {table: (columns, rows)} as SQLite stores them (rowid order where available)."""
    con = C.sqlite_ro(path)
    try:
        names = [r[0] for r in con.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name")]
        out = {}
        for t in names:
            if tables is not None and t not in tables:
                continue
            cols = [r[1] for r in con.execute(f'PRAGMA table_info("{t}")')]
            q = ", ".join(f'"{c}"' for c in cols)
            try:
                rows = con.execute(f'SELECT {q} FROM "{t}" ORDER BY rowid').fetchall()
            except sqlite3.OperationalError:
                rows = con.execute(f'SELECT {q} FROM "{t}"').fetchall()
            out[t] = (cols, [tuple(r) for r in rows])
        return out
    finally:
        con.close()


def to_doc(cols, row, missing=False):
    d = {}
    for c, v in zip(cols, row):
        if v is None and missing:
            continue
        d[c] = v
    return d


def load(mdb, db_id, tables, missing=False):
    """Replace collections <db_id>__<table> in `mdb` with `tables` ({t: (cols, rows)})."""
    for t, (cols, rows) in tables.items():
        name = C.coll(db_id, t)
        mdb.drop_collection(name)
        mdb.create_collection(name)
        docs = [to_doc(cols, r, missing) for r in rows]
        for i in range(0, len(docs), 20000):
            mdb[name].insert_many(docs[i:i + 20000], ordered=True)


def _key(v):
    if v is None:
        return ("null",)
    if isinstance(v, bool):
        return ("bool", v)
    if isinstance(v, int):
        return ("int", v)
    if isinstance(v, float):
        return ("float", v)
    if isinstance(v, (bytes, bytearray)):
        return ("bytes", bytes(v))
    return ("str", v)


def parity(mdb, db_id, tables, missing=False):
    """Per-table multiset parity of (type, value) tuples between SQLite rows and MongoDB."""
    problems = []
    for t, (cols, rows) in tables.items():
        got = Counter()
        for d in mdb[C.coll(db_id, t)].find({}, {"_id": 0}):
            extra = set(d) - set(cols)
            if extra:
                problems.append(f"{t}: unexpected fields {sorted(extra)}")
            if not missing and set(d) != set(cols):
                problems.append(f"{t}: document lacks fields")
            got[tuple(_key(d.get(c)) for c in cols)] += 1
        want = Counter(tuple(_key(v) for v in r) for r in rows)
        if got != want:
            problems.append(f"{t}: multiset differs ({sum(want.values())} sqlite vs {sum(got.values())} mongo)")
    return problems


def import_real(client, db_id, path=None):
    """Import one database under both NULL representations; -> parity report."""
    tables = read_tables(path or C.sqlite_path(db_id))
    rep = {}
    for missing, dbname in ((False, C.REAL_DB), (True, C.REAL_DB + "_missing")):
        mdb = client[dbname]
        load(mdb, db_id, tables, missing)
        rep["missing" if missing else "explicit_null"] = parity(mdb, db_id, tables, missing)
    rep["tables"] = {t: len(r) for t, (_, r) in tables.items()}
    return rep
