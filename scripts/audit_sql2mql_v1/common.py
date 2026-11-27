"""Shared helpers for the independent sql2mql-v1 audit (read-only with respect to ft-v1).

The audited modules (scripts/finetune_v1) are imported, never modified. Run every script with
`python -B` so no bytecode is written next to them. Nothing here calls verify.main() or
build_ft.py (both write project artifacts).

Independence: the comparator below (`norm`, `rows_of`, `same`) is written for the audit and
does not use spider1_kit.exec_match / fixtures.compare. MongoDB execution reuses
verify.inline_pipeline (pure plumbing: prefixes `$documents`, inlines `$lookup.from`).
"""
import json
import math
import sys
import threading
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FT = ROOT / "scripts/finetune_v1"
OUT = ROOT / "outputs/audit_sql2mql_v1"
for p in (FT, ROOT / "scripts/spider1", ROOT / "scripts/evaluation"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import nk_common as C  # noqa: E402
import sql2mql as M  # noqa: E402
import sqlsem as S  # noqa: E402
import verify as V  # noqa: E402
from pymongo.errors import OperationFailure  # noqa: E402

_lock = threading.Lock()


def convert(sql, schema):
    """-> (mql | None, ast, analysis, reason)."""
    ast, dialect, err = S.parse_sql(sql)
    if ast is None:
        return None, None, None, f"parse:{err}"
    an = S.analyze(ast)
    if an.unsupported:
        return None, ast, an, "unsupported:" + ";".join(an.unsupported)
    sch = {t: schema[t] for t in an.columns}
    try:
        return M.convert(ast, an, sch), ast, an, None
    except M.Unsupported as e:
        return None, ast, an, f"converter_unsupported:{e}"


def drop_nulls(tables):
    """Missing-field representation: a SQL NULL is an absent field, not an explicit null."""
    return {t: [{k: v for k, v in r.items() if v is not None} for r in rows] for t, rows in tables.items()}


def run_mongo(mql, tables, missing=False):
    """-> (docs, error). Executes on Atlas with the rows inline (`$documents`)."""
    if missing:
        tables = drop_nulls(tables)
    docs = {C.coll(t): rows for t, rows in tables.items()}
    try:
        return list(V.mongo_db().aggregate(V.inline_pipeline(mql, docs), maxTimeMS=60000)), None
    except OperationFailure as e:
        return None, f"{e.code}:{str(e)[:200]}"


def out_columns(mql, schema, base):
    """Answer columns in order, read from the final $project (star -> schema order)."""
    last = mql["pipeline"][-1] if mql["pipeline"] else {}
    proj = last.get("$project", {})
    cols = [k for k, v in proj.items() if k != "_id" and v != 0 and not k.startswith("__s")]
    return cols if cols else list(schema[base])


def rows_of(docs, cols):
    return [tuple(d.get(c) for c in cols) for d in docs]


def norm(v):
    if v is None:
        return None
    if isinstance(v, bool):
        return float(v)
    if isinstance(v, (int, float)):
        f = float(v)
        return None if math.isnan(f) else round(f, 4) + 0.0
    if isinstance(v, (dict, list)):
        return "OBJ:" + json.dumps(v, sort_keys=True, default=str)
    return str(v)


def same(got, exp, ordered):
    g = [tuple(norm(x) for x in r) for r in got]
    e = [tuple(norm(x) for x in r) for r in exp]
    if ordered:
        return g == e
    return sorted(g, key=repr) == sorted(e, key=repr)


def sqlite_rows(sql_ast, tables, schema, clock):
    con = S.sqlite_db(tables, {t: schema[t] for t in tables}, clock)
    try:
        return S.run_sqlite(con, S.to_sqlite(sql_ast))[1]
    finally:
        con.close()


def server_clock():
    return V.server_clock()


def write_jsonl(path, recs):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")


def append_jsonl(path, rec):
    path.parent.mkdir(parents=True, exist_ok=True)
    with _lock, open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")


def read_jsonl(path):
    if not path.exists():
        return []
    return [json.loads(x) for x in open(path, encoding="utf-8") if x.strip()]


def wilson(k, n, z=1.96):
    if n == 0:
        return (None, None)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (round((c - h) / d, 4), round((c + h) / d, 4))
