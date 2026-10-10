"""Shared plumbing for the Spider 1.0 (Yale, dev split) -> MongoDB/MQL track.

Plumbing only: read-only SQLite access to the official Spider databases, official
hardness labels (taoyds/spider `evaluation.py`), gold-SQL execution, ORDER BY tie
analysis, a Spider-1-style execution comparator, and Atlas import into the dedicated
`codegen_spider1` database. The Mongo URI comes from .env and is never printed.

Comparator semantics (`exec_match`) -- the same function scores gold MQL and models:
  * a result is a list of rows; each MQL output document is one row
  * an ObjectId `_id` (document identity) or an `_id` that is null in every row (a
    `$group: {_id: null}` key) is dropped; any other `_id` is a value column;
    nested objects are flattened (`_id.a`, `_id.b`)
  * the column COUNT must equal the gold's; column names and order are ignored
    (any column permutation may match)
  * numbers compare after rounding to 4 decimals (int 3 == float 3.0); strings exact;
    NULL == missing field
  * unordered gold (no top-level ORDER BY): rows compared as a multiset
  * ordered gold: rows compared in sequence, except that rows tied on the ORDER BY key
    in the gold may appear in any order among themselves (tie blocks)
"""
import importlib.util
import itertools
import json
import math
import os
import re
import sqlite3
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data/raw/spider1"
SPIDER = RAW / "spider_data"
DB_DIR = SPIDER / "database"
OFFICIAL = RAW / "official_eval"
INTERIM = ROOT / "data/interim/spider1"
ATLAS_DB = "codegen_spider1"

_OFF = None


def official():
    """Import the official process_sql + evaluation modules from data/raw (read-only)."""
    global _OFF
    if _OFF is None:
        sys.path.insert(0, str(OFFICIAL))
        try:
            ps = importlib.import_module("process_sql")
            ev = importlib.import_module("evaluation")
        finally:
            sys.path.remove(str(OFFICIAL))
        _OFF = (ps, ev)
    return _OFF


TS_DIR = RAW / "testsuite"


def db_path(db_id, variant=None):
    """Official dev DB, or a test-suite variant (data/raw/spider1/testsuite/<db>/<variant>.sqlite)."""
    if variant:
        return TS_DIR / db_id / f"{variant}.sqlite"
    return DB_DIR / db_id / f"{db_id}.sqlite"


def sqlite_ro(db_id, variant=None):
    con = sqlite3.connect("file:" + db_path(db_id, variant).as_posix() + "?mode=ro", uri=True)
    con.text_factory = lambda b: b.decode("utf-8", errors="replace")
    return con


def run_sql(db_id, sql, variant=None):
    con = sqlite_ro(db_id, variant)
    try:
        cur = con.execute(sql)
        cols = [d[0] for d in cur.description]
        return cols, [tuple(r) for r in cur.fetchall()]
    finally:
        con.close()


def hardness(db_id, sql):
    ps, ev = official()
    schema = ps.Schema(ps.get_schema(str(db_path(db_id))))
    parsed = ps.get_sql(schema, sql)
    return ev.Evaluator().eval_hardness(parsed), parsed


# ---------------------------------------------------------------- top-level clause split
def _top_level_positions(sql, keyword):
    """Positions of `keyword` (case-insensitive, word-bounded) at parenthesis depth 0,
    outside string literals."""
    out, depth, i, n, q = [], 0, 0, len(sql), None
    kw = keyword.lower()
    low = sql.lower()
    while i < n:
        ch = sql[i]
        if q:
            if ch == q:
                q = None
        elif ch in ("'", '"'):
            q = ch
        elif ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif depth == 0 and low.startswith(kw, i):
            before = low[i - 1] if i else " "
            after = low[i + len(kw)] if i + len(kw) < n else " "
            if not (before.isalnum() or before == "_") and not (after.isalnum() or after == "_"):
                out.append(i)
        i += 1
    return out


def order_info(sql):
    """Return dict(order_by: bool, limit: int|None, order_expr: str|None, select_list, from_rest)
    for a simple top-level `SELECT .. FROM .. ORDER BY e [LIMIT k]` (no set operations)."""
    s = sql.strip().rstrip(";")
    info = {"order_by": False, "limit": None, "order_expr": None, "analyzable": False}
    if any(_top_level_positions(s, k) for k in ("union", "intersect", "except")):
        info["set_op"] = True
    ob = _top_level_positions(s, "order by")
    if not ob:
        lim = _top_level_positions(s, "limit")
        if lim:
            info["limit"] = int(re.findall(r"\d+", s[lim[-1]:])[0])
        return info
    info["order_by"] = True
    tail = s[ob[-1] + len("order by"):]
    lim = _top_level_positions(tail, "limit")
    if lim:
        info["limit"] = int(re.findall(r"\d+", tail[lim[-1]:])[0])
        tail = tail[:lim[-1]]
    expr = re.sub(r"\s+(asc|desc)\s*$", "", tail.strip(), flags=re.I)
    info["order_expr"] = expr
    sel = _top_level_positions(s, "select")
    frm = _top_level_positions(s, "from")
    if sel and frm and sel[0] == 0 and not info.get("set_op"):
        info["select_list"] = s[len("select"):frm[0]].strip()
        info["from_rest"] = s[frm[0]:ob[-1]].strip()
        info["direction"] = "desc" if re.search(r"\bdesc\s*$", tail.strip(), re.I) else "asc"
        info["analyzable"] = True
    return info


def tie_analysis(db_id, sql, gold_rows, variant=None):
    """For an ORDER BY gold: compute tie-block ids per gold row and whether a LIMIT cutoff
    falls inside a tie. Verified by re-deriving the gold rows from the keyed query."""
    info = order_info(sql)
    res = {"order_by": info["order_by"], "limit": info["limit"], "blocks": None,
           "tie_at_limit": None, "status": "no_order"}
    if not info["order_by"]:
        return res
    if not info["analyzable"]:
        res["status"] = "unanalyzable"
        return res
    sel = info["select_list"]
    distinct = sel.lower().startswith("distinct ")
    keyed = (f"SELECT {sel}, {info['order_expr']} {info['from_rest']} "
             f"ORDER BY {info['order_expr']} {info['direction']}")
    try:
        _, rows = run_sql(db_id, keyed, variant)
    except Exception as e:  # noqa: BLE001
        res["status"] = f"keyed_query_failed: {e}"
        return res
    k = info["limit"]
    head = rows[:k] if k is not None else rows
    if [r[:-1] for r in head] != list(gold_rows) and not distinct:
        res["status"] = "keyed_rows_differ"
        return res
    if distinct:
        res["status"] = "distinct_unverified"
        return res
    keys = [r[-1] for r in head]
    blocks, b = [], 0
    for i, key in enumerate(keys):
        if i and key != keys[i - 1]:
            b += 1
        blocks.append(b)
    res["blocks"] = blocks
    if k is not None and len(rows) > k:
        res["tie_at_limit"] = rows[k][-1] == rows[k - 1][-1]
    else:
        res["tie_at_limit"] = False
    res["status"] = "ok"
    return res


# ---------------------------------------------------------------- comparator
def _norm(v):
    if v is None:
        return None
    if isinstance(v, bool):
        return float(v)
    if isinstance(v, (int, float)):
        f = float(v)
        if math.isnan(f):
            return None
        return round(f, 4) + 0.0
    if isinstance(v, (bytes, bytearray)):
        return v.decode("utf-8", errors="replace")
    return str(v)


def _flatten(doc, prefix=""):
    out = {}
    for k, v in doc.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, key + "."))
        else:
            out[key] = v
    return out


def docs_to_rows(docs):
    """MQL output documents -> (columns, rows). Drops an ObjectId `_id`."""
    from bson import ObjectId

    flat = []
    for d in docs:
        if not isinstance(d, dict):
            flat.append({"value": d})
            continue
        d = dict(d)
        if isinstance(d.get("_id"), ObjectId):
            d.pop("_id")
        flat.append(_flatten(d))
    # a `$group: {_id: null}` key is never an answer value: drop an all-null `_id` column
    if flat and all(d.get("_id", None) is None for d in flat):
        for d in flat:
            d.pop("_id", None)
    cols = []
    for d in flat:
        for k in d:
            if k not in cols:
                cols.append(k)
    rows = [tuple(d.get(c) for c in cols) for d in flat]
    return cols, rows


def exec_match(pred_rows, gold_rows, blocks=None, ordered=False):
    """Spider-1-style execution match. Returns (match: bool, reason: str)."""
    gold = [tuple(_norm(v) for v in r) for r in gold_rows]
    pred = [tuple(_norm(v) for v in r) for r in pred_rows]
    if len(pred) != len(gold):
        return False, f"row count {len(pred)} != gold {len(gold)}"
    if not gold:
        return True, "both empty"
    ng, np_ = len(gold[0]), len(pred[0]) if pred else 0
    if any(len(r) != np_ for r in pred):
        return False, "ragged prediction rows"
    if np_ != ng:
        return False, f"column count {np_} != gold {ng}"
    gold_cols = [Counter(r[i] for r in gold) for i in range(ng)]
    cand = []
    for j in range(np_):
        pc = Counter(r[j] for r in pred)
        cand.append([i for i in range(ng) if gold_cols[i] == pc])
    if any(not c for c in cand):
        return False, "some predicted column matches no gold column"
    for perm in itertools.product(*cand):
        if len(set(perm)) != ng:
            continue
        inv = [perm.index(i) for i in range(ng)]
        aligned = [tuple(r[inv[i]] for i in range(ng)) for r in pred]
        if not ordered:
            if Counter(aligned) == Counter(gold):
                return True, "multiset match"
            continue
        if blocks is None:
            if aligned == gold:
                return True, "ordered match"
            continue
        ok = True
        for b in sorted(set(blocks)):
            idx = [i for i, x in enumerate(blocks) if x == b]
            if Counter(aligned[i] for i in idx) != Counter(gold[i] for i in idx):
                ok = False
                break
        if ok:
            return True, "ordered match (tie-aware)"
    return False, "values/rows differ" + (" (order-sensitive)" if ordered else "")


# ---------------------------------------------------------------- Atlas
def mongo_client():
    import certifi
    from dotenv import load_dotenv
    from pymongo import MongoClient

    load_dotenv(ROOT / ".env")
    return MongoClient(os.environ["MONGODB_URI"], serverSelectionTimeoutMS=30000,
                       tlsCAFile=certifi.where())


def collection_name(db_id, table):
    return f"{db_id}__{table}"


def tables(db_id, variant=None):
    con = sqlite_ro(db_id, variant)
    try:
        return [r[0] for r in con.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name")]
    finally:
        con.close()


def table_schema(db_id, table, variant=None):
    con = sqlite_ro(db_id, variant)
    try:
        return [(r[1], r[2]) for r in con.execute(f'PRAGMA table_info("{table}")').fetchall()]
    finally:
        con.close()


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, default=str, ensure_ascii=False) + "\n")
    return path
