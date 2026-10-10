"""local068 - April/May/June city insert counts 2021-2023 with cumulative totals and YoY growth.

No shipped gold SQL: a labelled reference SQL is authored and verified against the gold
CSV first. 2021 is the growth baseline and is excluded from the output.
"""
import pandas as pd

from conversion_kit import import_table, manifest_record, matched_reason, pattern_record, score, sqlite_df

IID = "local068"

REFERENCE_SQL = """
WITH base AS (
  SELECT CAST(substr(insert_date,1,4) AS INTEGER) AS year,
         CAST(substr(insert_date,6,2) AS INTEGER) AS month,
         COUNT(*) AS monthly_total
  FROM cities
  WHERE CAST(substr(insert_date,1,4) AS INTEGER) IN (2021,2022,2023)
    AND CAST(substr(insert_date,6,2) AS INTEGER) IN (4,5,6)
  GROUP BY year, month),
w AS (
  SELECT year, month, monthly_total,
         SUM(monthly_total) OVER (PARTITION BY month ORDER BY year ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS running_total
  FROM base),
g AS (
  SELECT year, month, monthly_total, running_total,
         100.0 * (monthly_total - LAG(monthly_total) OVER (PARTITION BY month ORDER BY year)) / LAG(monthly_total) OVER (PARTITION BY month ORDER BY year) AS monthly_total_yoy_growth,
         100.0 * (running_total - LAG(running_total) OVER (PARTITION BY month ORDER BY year)) / LAG(running_total) OVER (PARTITION BY month ORDER BY year) AS running_total_yoy_growth
  FROM w)
SELECT * FROM g WHERE year >= 2022 ORDER BY year, month
"""

PIPELINE = [
    {"$addFields": {"year": {"$toInt": {"$substrCP": ["$insert_date", 0, 4]}},
                    "month": {"$toInt": {"$substrCP": ["$insert_date", 5, 2]}}}},
    {"$match": {"year": {"$in": [2021, 2022, 2023]}, "month": {"$in": [4, 5, 6]}}},
    {"$group": {"_id": {"year": "$year", "month": "$month"}, "monthly_total": {"$sum": 1}}},
    {"$setWindowFields": {
        "partitionBy": "$_id.month", "sortBy": {"_id.year": 1},
        "output": {"running_total": {"$sum": "$monthly_total", "window": {"documents": ["unbounded", "current"]}}}}},
    {"$setWindowFields": {
        "partitionBy": "$_id.month", "sortBy": {"_id.year": 1},
        "output": {
            "prev_mt": {"$shift": {"output": "$monthly_total", "by": -1, "default": None}},
            "prev_rt": {"$shift": {"output": "$running_total", "by": -1, "default": None}}}}},

    {"$addFields": {
        "monthly_total_yoy_growth": {"$multiply": [100, {"$divide": [{"$subtract": ["$monthly_total", "$prev_mt"]}, "$prev_mt"]}]},
        "running_total_yoy_growth": {"$multiply": [100, {"$divide": [{"$subtract": ["$running_total", "$prev_rt"]}, "$prev_rt"]}]}}},
    {"$match": {"_id.year": {"$gte": 2022}}},
    {"$project": {"_id": 0, "year": "$_id.year", "month": "$_id.month", "monthly_total": 1,
                  "running_total": 1, "monthly_total_yoy_growth": 1, "running_total_yoy_growth": 1}},
    {"$sort": {"year": 1, "month": 1}},
]

COLS = ["year", "month", "monthly_total", "running_total", "monthly_total_yoy_growth", "running_total_yoy_growth"]


def run(reg):
    rec = manifest_record(IID)
    ref = sqlite_df(rec, REFERENCE_SQL)
    ref_v, ref_ok = score(ref, rec)

    cities = import_table(reg, rec, IID, "cities", ["city_id", "insert_date"])
    coll = reg.db[cities["collection"]]
    res = list(coll.aggregate(PIPELINE, allowDiskUse=True))
    pred = pd.DataFrame(res, columns=COLS)
    v, ok = score(pred, rec)

    artifact = {
        "pattern": "date-part grouping + partitioned running total and $shift lag growth rates",
        "gold_sql": None, "gold_sql_reproduces_gold_csv": "n/a (no shipped gold SQL)",
        "reference_sql": REFERENCE_SQL, "reference_sql_matches_gold": ref_ok,
        "relational_semantics": ("Count cities inserted per month (Apr/May/Jun) per year 2021-2023; running total per month "
                                 "across years; YoY growth for monthly and running totals; output 2022+."),
        "source_schema": {"cities": ["city_id", "insert_date"]},
        "mongodb": {"database": "codegen_capstone_pilot", "imports": [cities], "_id": "default ObjectId"},
        "mql": {"collection": coll.name, "method": "aggregate", "options": {"allowDiskUse": True}, "pipeline": PIPELINE},
        "mongo_result": res,
        "validation": {"per_gold": v, "matched_any_gold": ok},
        "assumptions": ["insert_date is 'YYYY-MM-DD'; year/month extracted positionally.",
                        "Output columns follow gold variant _a."],
    }
    return ("passed" if ok else "failed"), matched_reason(v, ok), artifact
