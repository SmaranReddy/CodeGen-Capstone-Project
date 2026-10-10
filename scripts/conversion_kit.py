"""Shared conversion kit for the Spider 2.0-Lite -> MongoDB/MQL batch.

This is plumbing only (NOT a SQL->MQL translator): read-only SQLite access, schema
introspection, explicit-column import into the dedicated pilot database, the OFFICIAL
Spider comparator (source extracted verbatim from evaluate.py), a registry that tracks
every collection created during this run, and artifact/status/dataset writers.

All per-example table->collection mapping and MQL are written by hand in each
`scripts/conversions/conv_<id>.py`. The Mongo URI comes from .env and is never printed.
"""
import ast
import json
import math
import os
import sqlite3
from pathlib import Path

import certifi
import pandas as pd
from dotenv import load_dotenv
from pymongo import MongoClient

ROOT = Path(__file__).resolve().parents[1]
PILOT_DB = "codegen_capstone_pilot"
CONV_DIR = ROOT / "data/interim/conversions"
PILOT_DIR = ROOT / "data/interim/pilot"
STATUS_PATH = ROOT / "data/interim/conversion_status.jsonl"
DATASET_PATH = ROOT / "data/processed/nl_mql_dataset.jsonl"
REGISTRY_PATH = CONV_DIR / "_atlas_collections.json"

# the 7 already-validated examples from the prototype round
EXISTING_VALIDATED = ["local041", "local163", "local038", "local022", "local039", "local197", "local075"]

_MANIFEST = None
_PATTERNS = None


def manifest():
    global _MANIFEST
    if _MANIFEST is None:
        _MANIFEST = {}
        with open(ROOT / "data/interim/spider2_local_manifest.jsonl", encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                _MANIFEST[d["instance_id"]] = d
    return _MANIFEST


def patterns():
    global _PATTERNS
    if _PATTERNS is None:
        _PATTERNS = {}
        with open(ROOT / "data/interim/spider2_local_query_patterns.jsonl", encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                _PATTERNS[d["instance_id"]] = d
    return _PATTERNS


def manifest_record(iid):
    return manifest()[iid]


def pattern_record(iid):
    return patterns()[iid]


# ---------------------------------------------------------------- SQLite (read-only)
def sqlite_ro(rec):
    return sqlite3.connect("file:" + (ROOT / rec["database_path"]).as_posix() + "?mode=ro", uri=True)


def sqlite_df(rec, sql):
    return pd.read_sql_query(sql, sqlite_ro(rec))


def table_columns(rec, table):
    """Declared columns and SQLite declared types for a table."""
    sq = sqlite_ro(rec)
    return [(r[1], r[2]) for r in sq.execute(f'PRAGMA table_info("{table}")').fetchall()]


def column_storage_classes(rec, table, columns):
    """Distinct typeof() per column (SQLite is dynamically typed)."""
    sq = sqlite_ro(rec)
    out = {}
    for c in columns:
        rows = sq.execute(f'SELECT DISTINCT typeof("{c}") FROM "{table}"').fetchall()
        out[c] = sorted(r[0] for r in rows)
    return out


def count(rec, table, where=None):
    sq = sqlite_ro(rec)
    sql = f'SELECT COUNT(*) FROM "{table}"' + (f" WHERE {where}" if where else "")
    return sq.execute(sql).fetchone()[0]


# ---------------------------------------------------------------- official comparator
def _official_compare():
    src = (ROOT / "data/raw/spider2/evaluation_suite/evaluate.py").read_text(encoding="utf-8")
    fn = next(n for n in ast.parse(src).body
              if isinstance(n, ast.FunctionDef) and n.name == "compare_pandas_table")
    ns = {"pd": pd, "math": math}
    exec(compile(ast.Module([fn], []), "evaluate.py", "exec"), ns)
    return ns["compare_pandas_table"]


compare_pandas_table = _official_compare()


def score(pred, rec):
    """Official semantics. Returns (per-gold verdicts, overall pass = any gold)."""
    cfg = rec["gold_eval_config"]
    cc = cfg["condition_cols"]
    verdicts = {}
    for i, p in enumerate(rec["gold_result_paths"]):
        gold = pd.read_csv(ROOT / p)
        cols = cc[i] if cc and isinstance(cc[0], list) else cc
        verdicts[Path(p).name] = {
            "gold_columns": list(gold.columns),
            "gold_rows": len(gold),
            "condition_cols": cols,
            "score": int(compare_pandas_table(pred, gold, cols or None, cfg["ignore_order"])),
        }
    return verdicts, any(v["score"] == 1 for v in verdicts.values())


def gold_frames(rec):
    return [(Path(p).name, pd.read_csv(ROOT / p)) for p in rec["gold_result_paths"]]


def matched_reason(verdicts, ok):
    if not ok:
        return "no gold matched"
    return "matched " + ", ".join(k for k, x in verdicts.items() if x["score"] == 1)


# ---------------------------------------------------------------- Mongo + registry
def mongo_client():
    load_dotenv(ROOT / ".env")
    return MongoClient(os.environ["MONGODB_URI"], serverSelectionTimeoutMS=30000, tlsCAFile=certifi.where())


class Registry:
    """Tracks every collection created during this run so cleanup only ever touches ours."""

    def __init__(self, db):
        self.db = db
        self.created = []
        if REGISTRY_PATH.exists():
            self.created = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))

    def _flush(self):
        CONV_DIR.mkdir(parents=True, exist_ok=True)
        REGISTRY_PATH.write_text(json.dumps(sorted(set(self.created)), indent=2), encoding="utf-8")

    def dropped_first(self, name):
        """Drop a collection only if we created it earlier in this run."""
        if name in self.created and name in self.db.list_collection_names():
            self.db.drop_collection(name)

    def note(self, name):
        if name not in self.created:
            self.created.append(name)
            self._flush()


def import_table(reg, rec, iid, table, columns, where=None, index=None):
    """Copy `columns` of `table` (all rows unless `where`) into `<iid>__<table>`.

    Values keep their SQLite storage class; NULL -> explicit null. No dedup, default ObjectId.
    Drops a same-named collection only if this run created it before."""
    name = f"{iid}__{table}"
    reg.dropped_first(name)
    sq = sqlite_ro(rec)
    cols = ", ".join(f'"{c}"' for c in columns)
    sql = f'SELECT {cols} FROM "{table}"' + (f" WHERE {where}" if where else "")
    rows = sq.execute(sql).fetchall()
    docs = [dict(zip(columns, r)) for r in rows]
    if name in reg.db.list_collection_names():
        reg.db.drop_collection(name)
    for i in range(0, len(docs), 20000):
        reg.db[name].insert_many(docs[i:i + 20000], ordered=False)
    n = reg.db[name].count_documents({})
    expected = sq.execute(f'SELECT COUNT(*) FROM "{table}"' + (f" WHERE {where}" if where else "")).fetchone()[0]
    assert n == len(docs) == expected, (name, n, len(docs), expected)
    reg.note(name)
    info = {"collection": name, "source_table": table, "fields": columns, "documents": n, "filter": where}
    if index:
        for fld in index:
            reg.db[name].create_index(fld)
            info.setdefault("indexes", []).append(fld)
    return info


def materialize_view(reg, rec, iid, view):
    """Run SELECT * FROM <view> and import the rows as `view_<name>` (materialized view rule)."""
    name = f"{iid}__view_{view}"
    reg.dropped_first(name)
    sq = sqlite_ro(rec)
    cur = sq.execute(f'SELECT * FROM "{view}"')
    cols = [d[0] for d in cur.description]
    docs = [dict(zip(cols, r)) for r in cur.fetchall()]
    if name in reg.db.list_collection_names():
        reg.db.drop_collection(name)
    if docs:
        for i in range(0, len(docs), 20000):
            reg.db[name].insert_many(docs[i:i + 20000], ordered=False)
    reg.note(name)
    return {"collection": name, "source_view": view, "fields": cols,
            "documents": reg.db[name].count_documents({}), "materialized_view": True}


# ---------------------------------------------------------------- writers
def write_conversion(iid, artifact):
    CONV_DIR.mkdir(parents=True, exist_ok=True)
    path = CONV_DIR / f"{iid}.json"
    path.write_text(json.dumps(artifact, indent=2, default=str, ensure_ascii=False), encoding="utf-8")
    return path


def load_status():
    if not STATUS_PATH.exists():
        return {}
    out = {}
    for line in open(STATUS_PATH, encoding="utf-8"):
        d = json.loads(line)
        out[d["instance_id"]] = d
    return out


def save_status(status):
    STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(STATUS_PATH, "w", encoding="utf-8") as f:
        for iid in sorted(status):
            f.write(json.dumps(status[iid], default=str, ensure_ascii=False) + "\n")
