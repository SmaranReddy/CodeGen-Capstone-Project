"""local018 - percentage-point change in the share of the top 2021 collision factor.

No shipped gold SQL: a labelled reference SQL is authored and verified against the gold
CSV first. The year is taken from `collision_date` (case_ids.db_year only covers 2017+).
Identify the primary_collision_factor violation category with the most incidents in 2021,
then report its share of all incidents in 2011 minus its share in 2021 (percentage points).
"""
import pandas as pd

from conversion_kit import import_table, manifest_record, matched_reason, pattern_record, score, sqlite_df

IID = "local018"

REFERENCE_SQL = """
WITH y AS (SELECT CAST(substr(collision_date,1,4) AS INT) AS yr, pcf_violation_category AS cat FROM collisions),
tot AS (SELECT yr, COUNT(*) AS t FROM y GROUP BY yr),
cnt AS (SELECT yr, cat, COUNT(*) AS c FROM y GROUP BY yr, cat),
top2021 AS (SELECT cat FROM cnt WHERE yr = 2021 ORDER BY c DESC LIMIT 1)
SELECT
  (SELECT 100.0 * c.c / t.t FROM cnt c JOIN tot t ON t.yr = c.yr WHERE c.yr = 2011 AND c.cat = (SELECT cat FROM top2021))
- (SELECT 100.0 * c.c / t.t FROM cnt c JOIN tot t ON t.yr = c.yr WHERE c.yr = 2021 AND c.cat = (SELECT cat FROM top2021)) AS v
"""

PIPELINE = [
    {"$match": {"collision_date": {"$ne": None}}},
    {"$set": {"yr": {"$toInt": {"$substrCP": ["$collision_date", 0, 4]}}}},
    {"$match": {"yr": {"$in": [2011, 2021]}}},
    {"$group": {"_id": {"yr": "$yr", "cat": "$pcf_violation_category"}, "c": {"$sum": 1}}},
    {"$setWindowFields": {"partitionBy": "$_id.yr",
                          "output": {"year_total": {"$sum": "$c", "window": {"documents": ["unbounded", "unbounded"]}}}}},
    {"$set": {"share": {"$multiply": [100, {"$divide": ["$c", "$year_total"]}]}}},
    {"$group": {"_id": "$_id.cat",
                "share2011": {"$sum": {"$cond": [{"$eq": ["$_id.yr", 2011]}, "$share", 0]}},
                "share2021": {"$sum": {"$cond": [{"$eq": ["$_id.yr", 2021]}, "$share", 0]}}}},
    {"$sort": {"share2021": -1}},
    {"$limit": 1},
    {"$project": {"_id": 0, "v": {"$subtract": ["$share2011", "$share2021"]}}},
]


def run(reg):
    rec = manifest_record(IID)
    ref = sqlite_df(rec, REFERENCE_SQL)
    ref_v, ref_ok = score(ref, rec)

    m = import_table(reg, rec, IID, "collisions", ["pcf_violation_category", "collision_date"])
    coll = reg.db[m["collection"]]

    res = list(coll.aggregate(PIPELINE, allowDiskUse=True))
    pred = pd.DataFrame(res, columns=["v"])
    v, ok = score(pred, rec)

    artifact = {
        "pattern": "top-category-by-year then percentage-point share delta via $setWindowFields",
        "gold_sql": None, "gold_sql_reproduces_gold_csv": "n/a (no shipped gold SQL)",
        "reference_sql": REFERENCE_SQL, "reference_sql_matches_gold": ref_ok,
        "relational_semantics": ("Derive the incident year from collision_date; find the pcf_violation_category most "
                                 "common in 2021; report its share (%) of 2011 incidents minus its share of 2021 incidents."),
        "source_schema": {"collisions": ["pcf_violation_category", "collision_date"]},
        "mongodb": {"database": "codegen_capstone_pilot", "imports": [m], "_id": "default ObjectId"},
        "mql": {"collection": coll.name, "method": "aggregate", "options": {"allowDiskUse": True}, "pipeline": PIPELINE},
        "mongo_result": res,
        "validation": {"per_gold": v, "matched_any_gold": ok},
        "assumptions": [
            "The reporting year is the first 4 characters of collision_date (case_ids.db_year only covers 2017+).",
            "A category's share in a year = its incident count / all incidents that year * 100.",
        ],
    }
    return ("passed" if ok else "failed"), matched_reason(v, ok), artifact
