"""local193 - share of lifetime sales in the first 7 / 30 days (exact elapsed time).

No shipped gold SQL: a labelled reference SQL is authored and verified against the gold
CSV first. Elapsed time uses julianday differences (exact seconds); customers with zero
lifetime sales are excluded.
"""
import pandas as pd

from conversion_kit import (import_table, manifest_record, matched_reason, pattern_record, score, sqlite_df)

IID = "local193"

REFERENCE_SQL = """
WITH f AS (SELECT customer_id, MIN(payment_date) AS first_date FROM payment GROUP BY customer_id),
ltv AS (SELECT customer_id, SUM(amount) AS tot FROM payment GROUP BY customer_id),
joined AS (SELECT l.customer_id, l.tot, f.first_date FROM ltv l JOIN f ON f.customer_id = l.customer_id WHERE l.tot > 0),
per AS (
    SELECT j.customer_id, j.tot,
           SUM(CASE WHEN julianday(p.payment_date) - julianday(j.first_date) <= 7 THEN p.amount ELSE 0 END) AS s7,
           SUM(CASE WHEN julianday(p.payment_date) - julianday(j.first_date) <= 30 THEN p.amount ELSE 0 END) AS s30
    FROM joined j JOIN payment p ON p.customer_id = j.customer_id
    GROUP BY j.customer_id
)
SELECT 100.0 * AVG(s7 / tot) AS avg_pct_ltv_7_days,
       100.0 * AVG(s30 / tot) AS avg_pct_ltv_30_days,
       AVG(tot) AS avg_ltv
FROM per
"""


def run(reg):
    rec = manifest_record(IID)
    ref = sqlite_df(rec, REFERENCE_SQL)
    ref_v, ref_ok = score(ref, rec)

    m = import_table(reg, rec, IID, "payment", ["customer_id", "amount", "payment_date"])
    coll = reg.db[m["collection"]]

    fmt = "%Y-%m-%d %H:%M:%S.%L"

    def days_expr(limit):
        return {"$lte": [{"$divide": [{"$dateDiff": {
            "startDate": {"$dateFromString": {"dateString": "$$fd", "format": fmt}},
            "endDate": {"$dateFromString": {"dateString": "$payment_date", "format": fmt}},
            "unit": "second"}}, 86400]}, limit]}

    inner = [
        {"$match": {"$expr": {"$eq": ["$customer_id", "$$cid"]}}},
        {"$project": {
            "in7": {"$cond": [days_expr(7), "$amount", 0]},
            "in30": {"$cond": [days_expr(30), "$amount", 0]}}},
        {"$group": {"_id": None, "s7": {"$sum": "$in7"}, "s30": {"$sum": "$in30"}}},
    ]
    PIPELINE = [
        {"$group": {"_id": "$customer_id", "tot": {"$sum": "$amount"}, "first": {"$min": "$payment_date"}}},
        {"$match": {"tot": {"$gt": 0}}},
        {"$lookup": {"from": m["collection"], "let": {"cid": "$_id", "fd": "$first"},
                     "pipeline": inner, "as": "agg"}},
        {"$unwind": "$agg"},
        {"$addFields": {"p7": {"$divide": ["$agg.s7", "$tot"]}, "p30": {"$divide": ["$agg.s30", "$tot"]}}},
        {"$group": {"_id": None,
                    "avg_pct_ltv_7_days": {"$avg": {"$multiply": [100, "$p7"]}},
                    "avg_pct_ltv_30_days": {"$avg": {"$multiply": [100, "$p30"]}},
                    "avg_ltv": {"$avg": "$tot"}}},
        {"$project": {"_id": 0, "avg_pct_ltv_7_days": 1, "avg_pct_ltv_30_days": 1, "avg_ltv": 1}},
    ]
    res = list(coll.aggregate(PIPELINE, allowDiskUse=True))
    pred = pd.DataFrame(res, columns=["avg_pct_ltv_7_days", "avg_pct_ltv_30_days", "avg_ltv"])
    v, ok = score(pred, rec)

    artifact = {
        "pattern": "per-customer min date + windowed share via $dateDiff + averages",
        "gold_sql": None, "gold_sql_reproduces_gold_csv": "n/a (no shipped gold SQL)",
        "reference_sql": REFERENCE_SQL, "reference_sql_matches_gold": ref_ok,
        "relational_semantics": ("Per customer: lifetime sales (sum of payments) and first payment timestamp; share of sales within 7/30 exact days; "
                                 "average shares and average lifetime sales, excluding zero-LTV customers."),
        "source_schema": {"payment": ["customer_id", "amount", "payment_date"]},
        "mongodb": {"database": "codegen_capstone_pilot", "imports": [m], "_id": "default ObjectId"},
        "mql": {"collection": coll.name, "method": "aggregate", "options": {"allowDiskUse": True}, "pipeline": PIPELINE},
        "mongo_result": res,
        "validation": {"per_gold": v, "matched_any_gold": ok},
        "assumptions": [
            "Elapsed time is measured in exact seconds via $dateDiff (<= 7 and <= 30 days), matching SQLite julianday differences.",
            "payment_date uses the '%Y-%m-%d %H:%M:%S.%L' format.",
        ],
    }
    return ("passed" if ok else "failed"), matched_reason(v, ok), artifact
