"""local064 - months of 2020 with the most/fewest customers holding a positive month-end balance.

No shipped gold SQL: a labelled reference SQL is authored and verified against the gold
CSV first. Month-end balance = deposits - withdrawals per customer (purchases ignored);
customers with no activity in a month have balance 0. Averages use all distinct customers.
"""
import pandas as pd

from conversion_kit import import_table, manifest_record, matched_reason, pattern_record, score, sqlite_df

IID = "local064"

REFERENCE_SQL = """
WITH b AS (
  SELECT customer_id, CAST(substr(txn_date,6,2) AS INTEGER) AS m,
         SUM(CASE WHEN txn_type='deposit' THEN txn_amount
                  WHEN txn_type='withdrawal' THEN -txn_amount ELSE 0 END) AS bal
  FROM customer_transactions GROUP BY customer_id, m),
cust AS (SELECT DISTINCT customer_id FROM customer_transactions),
mon(m) AS (SELECT 1 UNION SELECT 2 UNION SELECT 3 UNION SELECT 4 UNION SELECT 5 UNION SELECT 6
           UNION SELECT 7 UNION SELECT 8 UNION SELECT 9 UNION SELECT 10 UNION SELECT 11 UNION SELECT 12),
full AS (SELECT c.customer_id, mo.m, COALESCE(b.bal,0) AS bal
         FROM cust c CROSS JOIN mon mo LEFT JOIN b ON b.customer_id=c.customer_id AND b.m=mo.m),
agg AS (SELECT m, SUM(CASE WHEN bal>0 THEN 1 ELSE 0 END) AS pos, AVG(bal) AS avgb FROM full GROUP BY m),
hi AS (SELECT * FROM agg ORDER BY pos DESC, m ASC LIMIT 1),
lo AS (SELECT * FROM agg ORDER BY pos ASC, m ASC LIMIT 1)
SELECT hi.m AS highest_month, hi.pos AS highest_positive_count, hi.avgb AS highest_avg_balance,
       lo.m AS lowest_month, lo.pos AS lowest_positive_count, lo.avgb AS lowest_avg_balance,
       hi.avgb - lo.avgb AS average_difference
FROM hi, lo
"""

GRID = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

PIPELINE_TEMPLATE = [
    {"$addFields": {"m": {"$toInt": {"$substrCP": ["$txn_date", 5, 2]}}}},
    {"$group": {"_id": {"c": "$customer_id", "m": "$m"},
                "bal": {"$sum": {"$switch": {"branches": [
                    {"case": {"$eq": ["$txn_type", "deposit"]}, "then": "$txn_amount"},
                    {"case": {"$eq": ["$txn_type", "withdrawal"]}, "then": {"$multiply": ["$txn_amount", -1]}}],
                    "default": 0}}}}},
    {"$group": {"_id": "$_id.m", "pos": {"$sum": {"$cond": [{"$gt": ["$bal", 0]}, 1, 0]}}, "sumBal": {"$sum": "$bal"}}},
    {"$group": {"_id": None, "rows": {"$push": {"m": "$_id", "pos": "$pos", "sumBal": "$sumBal"}}}},
    {"$project": {"_id": 0, "rows": {"$map": {"input": {"$range": [1, 13]}, "as": "mm", "in": {
        "$let": {"vars": {"hit": {"$arrayElemAt": [{"$filter": {"input": "$rows", "as": "r", "cond": {"$eq": ["$$r.m", "$$mm"]}}}, 0]}},
                 "in": {"m": "$$mm", "pos": {"$ifNull": ["$$hit.pos", 0]}, "sumBal": {"$ifNull": ["$$hit.sumBal", 0]}}}}}}}},
    {"$unwind": "$rows"},
    {"$addFields": {"avg": {"$divide": ["$rows.sumBal", "__N__"]}}},
    {"$facet": {
        "hi": [{"$sort": {"rows.pos": -1, "rows.m": 1}}, {"$limit": 1}],
        "lo": [{"$sort": {"rows.pos": 1, "rows.m": 1}}, {"$limit": 1}]}},
    {"$project": {"_id": 0,
                  "highest_month": {"$arrayElemAt": ["$hi.rows.m", 0]},
                  "highest_positive_count": {"$arrayElemAt": ["$hi.rows.pos", 0]},
                  "highest_avg_balance": {"$arrayElemAt": ["$hi.avg", 0]},
                  "lowest_month": {"$arrayElemAt": ["$lo.rows.m", 0]},
                  "lowest_positive_count": {"$arrayElemAt": ["$lo.rows.pos", 0]},
                  "lowest_avg_balance": {"$arrayElemAt": ["$lo.avg", 0]},
                  "average_difference": {"$subtract": [{"$arrayElemAt": ["$hi.avg", 0]}, {"$arrayElemAt": ["$lo.avg", 0]}]}}},
]

COLS = ["highest_month", "highest_positive_count", "highest_avg_balance",
        "lowest_month", "lowest_positive_count", "lowest_avg_balance", "average_difference"]


def run(reg):
    rec = manifest_record(IID)
    ref = sqlite_df(rec, REFERENCE_SQL)
    ref_v, ref_ok = score(ref, rec)

    txn = import_table(reg, rec, IID, "customer_transactions", ["customer_id", "txn_date", "txn_type", "txn_amount"])
    coll = reg.db[txn["collection"]]
    n = len(coll.distinct("customer_id"))

    PIPELINE = []
    for s in PIPELINE_TEMPLATE:
        if "$addFields" in s and "avg" in s["$addFields"]:
            s = {"$addFields": {"avg": {"$divide": ["$rows.sumBal", n]}}}
        PIPELINE.append(s)

    res = list(coll.aggregate(PIPELINE, allowDiskUse=True))
    pred = pd.DataFrame(res, columns=COLS)
    v, ok = score(pred, rec)

    artifact = {
        "pattern": "per-entity-period net balance + full grid + argmax/argmin counts + difference",
        "gold_sql": None, "gold_sql_reproduces_gold_csv": "n/a (no shipped gold SQL)",
        "reference_sql": REFERENCE_SQL, "reference_sql_matches_gold": ref_ok,
        "relational_semantics": ("Per customer/month net balance (deposits - withdrawals); fill missing months with 0; "
                                 "months with most/fewest positive-balance customers; average across all customers; difference."),
        "source_schema": {"customer_transactions": ["customer_id", "txn_date", "txn_type", "txn_amount"]},
        "mongodb": {"database": "codegen_capstone_pilot", "imports": [txn], "_id": "default ObjectId",
                    "injected": {"N": n}},
        "mql": {"collection": coll.name, "method": "aggregate", "options": {"allowDiskUse": True}, "pipeline": PIPELINE},
        "mongo_result": res,
        "validation": {"per_gold": v, "matched_any_gold": ok},
        "assumptions": ["Purchases do not affect the balance (only deposits/withdrawals).",
                        "Missing customer-months have balance 0; averages divide by total distinct customers.",
                        "Output columns follow gold variant _a."],
    }
    return ("passed" if ok else "failed"), matched_reason(v, ok), artifact
