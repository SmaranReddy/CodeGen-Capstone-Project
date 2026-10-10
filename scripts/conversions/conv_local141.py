"""local141 - annual sales vs quota difference per salesperson (AdventureWorks).

No shipped gold SQL: a labelled reference SQL is authored and verified against the gold
CSV first. Sales = SUM(totaldue) of a salesperson's orders in a year; annual quota =
SUM(SalesQuota) for that salesperson-year. The inner join (sales people that also have a
quota) reproduces gold variant `_b` (58 rows).
"""
import pandas as pd

from conversion_kit import import_table, manifest_record, matched_reason, pattern_record, score, sqlite_df

IID = "local141"

REFERENCE_SQL = """
WITH sales AS (
  SELECT salespersonid AS sp, CAST(substr(orderdate,1,4) AS INTEGER) AS yr, SUM(totaldue) AS ts
  FROM salesorderheader GROUP BY salespersonid, yr),
quota AS (
  SELECT BusinessEntityID AS sp, CAST(substr(QuotaDate,1,4) AS INTEGER) AS yr, SUM(SalesQuota) AS q
  FROM SalesPersonQuotaHistory GROUP BY BusinessEntityID, yr)
SELECT s.sp AS salespersonid, s.yr AS year, s.ts AS total_sales, q.q AS annual_quota,
       s.ts - q.q AS difference
FROM sales s JOIN quota q ON s.sp = q.sp AND s.yr = q.yr
ORDER BY s.sp, s.yr
"""

PIPELINE_TEMPLATE = [
    {"$addFields": {"yr": {"$toInt": {"$substrCP": ["$orderdate", 0, 4]}}}},
    {"$group": {"_id": {"sp": "$salespersonid", "yr": "$yr"}, "ts": {"$sum": "$totaldue"}}},
    {"$lookup": {"from": "__QUOTA__", "let": {"sp": "$_id.sp", "yr": "$_id.yr"},
                 "pipeline": [
                     {"$match": {"$expr": {"$and": [
                         {"$eq": ["$BusinessEntityID", "$$sp"]},
                         {"$eq": [{"$toInt": {"$substrCP": ["$QuotaDate", 0, 4]}}, "$$yr"]}]}}},
                     {"$group": {"_id": None, "q": {"$sum": "$SalesQuota"}}}],
                 "as": "qq"}},
    {"$unwind": "$qq"},
    {"$project": {"_id": 0, "salespersonid": "$_id.sp", "year": "$_id.yr", "total_sales": "$ts",
                  "annual_quota": "$qq.q", "difference": {"$subtract": ["$ts", "$qq.q"]}}},
    {"$sort": {"salespersonid": 1, "year": 1}},
]

COLS = ["salespersonid", "year", "total_sales", "annual_quota", "difference"]


def run(reg):
    rec = manifest_record(IID)
    ref = sqlite_df(rec, REFERENCE_SQL)
    ref_v, ref_ok = score(ref, rec)

    orders = import_table(reg, rec, IID, "salesorderheader", ["salespersonid", "orderdate", "totaldue"])
    quota = import_table(reg, rec, IID, "SalesPersonQuotaHistory", ["BusinessEntityID", "QuotaDate", "SalesQuota"],
                         index=["BusinessEntityID"])

    coll = reg.db[orders["collection"]]
    PIPELINE = []
    for s in PIPELINE_TEMPLATE:
        if "$lookup" in s:
            s = {"$lookup": {**s["$lookup"], "from": quota["collection"]}}
        PIPELINE.append(s)
    res = list(coll.aggregate(PIPELINE, allowDiskUse=True))
    pred = pd.DataFrame(res, columns=COLS)
    v, ok = score(pred, rec)

    artifact = {
        "pattern": "per-entity-period sum + indexed grouped sub-pipeline $lookup join + difference",
        "gold_sql": None, "gold_sql_reproduces_gold_csv": "n/a (no shipped gold SQL)",
        "reference_sql": REFERENCE_SQL, "reference_sql_matches_gold": ref_ok,
        "relational_semantics": ("Annual sales (SUM totaldue) per salesperson compared to annual quota (SUM SalesQuota); "
                                 "difference = sales - quota; inner join reproduces gold variant _b."),
        "source_schema": {"salesorderheader": ["salespersonid", "orderdate", "totaldue"],
                          "SalesPersonQuotaHistory": ["BusinessEntityID", "QuotaDate", "SalesQuota"]},
        "mongodb": {"database": "codegen_capstone_pilot", "imports": [orders, quota], "_id": "default ObjectId"},
        "mql": {"collection": coll.name, "method": "aggregate", "options": {"allowDiskUse": True}, "pipeline": PIPELINE},
        "mongo_result": res,
        "validation": {"per_gold": v, "matched_any_gold": ok},
        "assumptions": ["Year is the calendar year of orderdate / QuotaDate.",
                        "Multiple quota rows in a year are summed.",
                        "The inner join (58 rows) is validated against gold variant _b."],
    }
    return ("passed" if ok else "failed"), matched_reason(v, ok), artifact
