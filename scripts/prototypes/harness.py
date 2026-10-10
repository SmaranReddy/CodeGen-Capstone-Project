"""Shared plumbing for the conversion prototypes (NOT a converter).

Provides only: read-only SQLite access, import of explicitly listed columns into the dedicated pilot database,
the OFFICIAL Spider comparator (source extracted from evaluate.py), and artifact writing.
All table->collection mapping and all MQL lives in each per-example script.
The Mongo URI comes from .env and is never printed or stored.
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

ROOT = Path(__file__).resolve().parents[2]
PILOT_DB = "codegen_capstone_pilot"
PILOT_DIR = ROOT / "data/interim/pilot"


def manifest_record(iid):
    for line in open(ROOT / "data/interim/spider2_local_manifest.jsonl", encoding="utf-8"):
        d = json.loads(line)
        if d["instance_id"] == iid:
            return d
    raise KeyError(iid)


def sqlite_ro(rec):
    return sqlite3.connect("file:" + (ROOT / rec["database_path"]).as_posix() + "?mode=ro", uri=True)


def _official_compare():
    src = (ROOT / "data/raw/spider2/evaluation_suite/evaluate.py").read_text(encoding="utf-8")
    fn = next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == "compare_pandas_table")
    ns = {"pd": pd, "math": math}
    exec(compile(ast.Module([fn], []), "evaluate.py", "exec"), ns)
    return ns["compare_pandas_table"]


compare_pandas_table = _official_compare()


def mongo_db():
    load_dotenv(ROOT / ".env")
    cl = MongoClient(os.environ["MONGODB_URI"], serverSelectionTimeoutMS=30000, tlsCAFile=certifi.where())
    return cl[PILOT_DB]


def import_table(db, rec, iid, table, columns, where=None):
    """Copy `columns` of `table` (all rows unless `where`) into collection `<iid>__<table>`.
    Values keep their SQLite storage class; NULL -> None (explicit null). No dedup, default ObjectId _id.
    Only collections that start with '<iid>__' are ever dropped (our own earlier run)."""
    name = f"{iid}__{table}"
    if name in db.list_collection_names():
        db.drop_collection(name)
    sq = sqlite_ro(rec)
    cols = ", ".join(f'"{c}"' for c in columns)
    sql = f'SELECT {cols} FROM "{table}"' + (f" WHERE {where}" if where else "")
    rows = sq.execute(sql).fetchall()
    docs = [dict(zip(columns, r)) for r in rows]
    for i in range(0, len(docs), 20000):
        db[name].insert_many(docs[i:i + 20000], ordered=False)
    n = db[name].count_documents({})
    assert n == len(docs) == sq.execute(f'SELECT COUNT(*) FROM "{table}"' + (f" WHERE {where}" if where else "")).fetchone()[0]
    return {"collection": name, "source_table": table, "fields": columns, "documents": n, "filter": where}


def sqlite_df(rec, sql):
    sq = sqlite_ro(rec)
    return pd.read_sql_query(sql, sq)


def score(pred, rec):
    """Official semantics. Returns per-gold verdicts and overall pass (any gold)."""
    cfg = rec["gold_eval_config"]
    cc = cfg["condition_cols"]
    verdicts = {}
    for i, p in enumerate(rec["gold_result_paths"]):
        gold = pd.read_csv(ROOT / p)
        cols = cc[i] if cc and isinstance(cc[0], list) else cc
        verdicts[Path(p).name] = {
            "gold_columns": list(gold.columns), "gold_rows": len(gold),
            "condition_cols": cols, "score": int(compare_pandas_table(pred, gold, cols or None, cfg["ignore_order"]))}
    return verdicts, any(v["score"] == 1 for v in verdicts.values())


def write_artifact(iid, artifact):
    PILOT_DIR.mkdir(parents=True, exist_ok=True)
    path = PILOT_DIR / f"{iid}.json"
    path.write_text(json.dumps(artifact, indent=2, default=str, ensure_ascii=False), encoding="utf-8")
    return path
