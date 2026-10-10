"""local163 - single-table aggregate: faculty whose salary is closest to their rank's average (ties kept)."""
import sys; sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import pandas as pd
from harness import *

IID = "local163"
rec = manifest_record(IID)
gold_sql = (ROOT / rec["gold_sql_path"]).read_text(encoding="utf-8")

# 0. reference: shipped gold SQL on SQLite, scored against the gold CSV
ref = sqlite_df(rec, gold_sql)
ref_v, ref_ok = score(ref, rec)

# 1. data/mapping: one table, 4 needed columns, all rows
db = mongo_db()
m = import_table(db, rec, IID, "university_faculty", ["FacRank", "FacFirstName", "FacLastName", "FacSalary"])
coll = db[m["collection"]]

# 2. MQL: AVG per rank + ABS diff + MIN diff per rank, keep rows equal to the min (ties kept, as SQL's equi-join does)
PIPELINE = [
    {"$setWindowFields": {"partitionBy": "$FacRank", "output": {"AvSalary": {"$avg": "$FacSalary"}}}},
    {"$addFields": {"Diff": {"$abs": {"$subtract": ["$FacSalary", "$AvSalary"]}}}},
    {"$setWindowFields": {"partitionBy": "$FacRank", "output": {"MinDiff": {"$min": "$Diff"}}}},
    {"$match": {"$expr": {"$eq": ["$Diff", "$MinDiff"]}}},
    {"$project": {"_id": 0, "FacRank": 1, "FacFirstName": 1, "FacLastName": 1, "Salary": "$FacSalary"}},
]
res = list(coll.aggregate(PIPELINE))
pred = pd.DataFrame(res, columns=["FacRank", "FacFirstName", "FacLastName", "Salary"])
v, ok = score(pred, rec)
print(IID, "ref_sql_ok", ref_ok, "| mql_ok", ok, v, "\n", pred)

write_artifact(IID, {
    "instance_id": IID, "db": rec["db"], "question": rec["question"], "pattern": "aggregate + per-group extremum with ties",
    "gold_sql": gold_sql, "gold_sql_reproduces_gold_csv": ref_ok,
    "relational_semantics": "AVG(FacSalary) per FacRank; ABS(salary-avg); MIN of that per rank; return ALL rows whose diff equals the min (equi-join keeps ties).",
    "mongodb": {"database": PILOT_DB, "imports": [m], "_id": "default ObjectId",
                "null_handling": "FacSalary has no NULLs in this table; nulls would be kept as null", "indexes": []},
    "mql": {"collection": m["collection"], "method": "aggregate", "pipeline": PIPELINE},
    "mongo_result": res, "gold": v, "validation_passed": ok,
    "rules_discovered": [
        "Per-group extremum 'keep rows equal to the min' -> $setWindowFields (partition) twice, then $match $eq; ties are preserved (gold has 2 ASSC rows).",
        "Float equality of computed values is safe here only because salaries are integers; for REAL columns compare with a tolerance or reuse the identical expression.",
    ]})
