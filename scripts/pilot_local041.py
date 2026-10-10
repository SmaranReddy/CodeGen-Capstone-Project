"""One-off pilot: Spider2-Lite local041 (SQLite) -> MongoDB/MQL -> validate vs gold.

NOT a generic converter. Reads raw Spider files read-only; writes only to the
dedicated MongoDB database `codegen_capstone_pilot` and to data/interim/pilot/.
The connection URI is read from .env and never printed or written.
"""
import ast
import json
import os
import sqlite3
import sys
from pathlib import Path

import certifi
import pandas as pd
from dotenv import load_dotenv
from pymongo import MongoClient

ROOT = Path(__file__).resolve().parent.parent
IID = "local041"
PILOT_DB = "codegen_capstone_pilot"
COLL = "local041_trees"

# ---- source record -------------------------------------------------------
manifest = next(json.loads(l) for l in open(ROOT / "data/interim/spider2_local_manifest.jsonl", encoding="utf-8")
                if json.loads(l)["instance_id"] == IID)
cfg = manifest["gold_eval_config"]

# ---- official comparison function, extracted verbatim from evaluate.py ----
src = (ROOT / "data/raw/spider2/evaluation_suite/evaluate.py").read_text(encoding="utf-8")
fn = next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == "compare_pandas_table")
ns = {"pd": pd, "math": __import__("math")}
exec(compile(ast.Module([fn], []), "evaluate.py", "exec"), ns)
compare_pandas_table = ns["compare_pandas_table"]

# ---- 1. read relational data (read-only) ---------------------------------
db_uri = "file:" + (ROOT / manifest["database_path"]).as_posix() + "?mode=ro"
sq = sqlite3.connect(db_uri, uri=True)
REFERENCE_SQL = ("SELECT 100.0 * SUM(CASE WHEN health = 'Good' THEN 1 ELSE 0 END) / COUNT(*) AS PCT_GOOD "
                 "FROM trees WHERE boroname = 'Bronx'")
sql_result = sq.execute(REFERENCE_SQL).fetchall()
rows = sq.execute("SELECT tree_id, boroname, health FROM trees").fetchall()
docs = [{"tree_id": t, "boroname": b, "health": h} for t, b, h in rows]  # NULL health kept as null
print("sqlite rows:", len(docs), "| reference SQL result:", sql_result)

# ---- 2. import into the dedicated pilot namespace ------------------------
load_dotenv(ROOT / ".env")
client = MongoClient(os.environ["MONGODB_URI"], serverSelectionTimeoutMS=20000, tlsCAFile=certifi.where())
coll = client[PILOT_DB][COLL]
if coll.count_documents({}) != 0:
    sys.exit(f"{PILOT_DB}.{COLL} not empty; refusing to touch it")
for i in range(0, len(docs), 20000):
    coll.insert_many(docs[i:i + 20000], ordered=False)
n_imported = coll.count_documents({})
print("imported:", n_imported)
assert n_imported == len(docs)

# ---- 3. MQL (aggregation pipeline) ---------------------------------------
PIPELINE = [
    {"$match": {"boroname": "Bronx"}},
    {"$group": {"_id": None,
                "good": {"$sum": {"$cond": [{"$eq": ["$health", "Good"]}, 1, 0]}},
                "total": {"$sum": 1}}},
    {"$project": {"_id": 0, "PCT_GOOD": {"$multiply": [100, {"$divide": ["$good", "$total"]}]}}},
]
mongo_result = list(coll.aggregate(PIPELINE))
print("mongo result:", mongo_result)

# ---- 4. validate with the official semantics -----------------------------
pred = pd.DataFrame(mongo_result)
verdicts = {}
for p in manifest["gold_result_paths"]:
    gold = pd.read_csv(ROOT / p)
    verdicts[Path(p).name] = {
        "gold": gold.to_dict(orient="records"),
        "score": int(compare_pandas_table(pred, gold, cfg["condition_cols"] or None, cfg["ignore_order"])),
    }
matched = any(v["score"] == 1 for v in verdicts.values())  # any one gold suffices (compare_multi_pandas_table)
print("verdicts:", {k: v["score"] for k, v in verdicts.items()}, "| matched:", matched)

# ---- 5. artifact ---------------------------------------------------------
artifact = {
    "instance_id": IID,
    "source": {"db": manifest["db"], "sqlite_path": manifest["database_path"],
               "question": manifest["question"], "external_knowledge": manifest["external_knowledge"],
               "original_gold_sql": None,
               "reference_sql_written_by_us": REFERENCE_SQL,
               "reference_sql_result_on_sqlite": sql_result[0][0]},
    "relational_schema_used": {"table": "trees", "columns_needed": ["boroname", "health"],
                               "full_ddl_columns": ["idx", "tree_id", "tree_dbh", "stump_diam", "status", "health",
                                                    "spc_latin", "spc_common", "address", "zipcode", "borocode",
                                                    "boroname", "nta_name", "state", "latitude", "longitude"]},
    "mongodb": {
        "database": PILOT_DB, "collection": COLL, "documents_imported": n_imported,
        "mapping": {"table->collection": "trees -> local041_trees",
                    "row->document": "one document per SQLite row (690,626, duplicates preserved)",
                    "fields_kept": {"tree_id": "carried for traceability (NOT unique, so not used as _id)",
                                    "boroname": "filter field", "health": "tested field; SQL NULL -> BSON null"},
                    "fields_dropped": "13 other columns: not needed to answer the question",
                    "_id": "default ObjectId"},
        "example_document": docs[0],
    },
    "mql": {"method": "aggregate", "collection": COLL, "pipeline": PIPELINE},
    "mongo_result": mongo_result,
    "gold": verdicts,
    "eval_config": cfg,
    "validation": {"semantics": "official compare_pandas_table (evaluate.py), any-one-gold, abs tol 0.01, "
                                "ignore_order, column names ignored",
                   "matched_any_gold": matched},
    "assumptions": [
        "Denominator = all Bronx rows (COUNT(*)), NULL health counted as not Good; this is what gold _a encodes.",
        "Duplicate tree_id rows are kept (gold counts rows, not distinct trees).",
        "Output column named PCT_GOOD to mirror gold _a (names are ignored by the evaluator anyway).",
    ],
}
out = ROOT / "data/interim/pilot"
out.mkdir(parents=True, exist_ok=True)
(out / f"{IID}.json").write_text(json.dumps(artifact, indent=2, default=str), encoding="utf-8")
print("wrote", out / f"{IID}.json")
