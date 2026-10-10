"""local031 - peak monthly delivered volume within the weakest year (2016-2018).

No shipped gold SQL: a labelled reference SQL is authored and verified against the gold
CSV first. The month is taken from the customer delivery date, not the purchase date.
"""
import pandas as pd

from conversion_kit import (import_table, manifest_record, matched_reason, pattern_record, score, sqlite_df)

IID = "local031"

REFERENCE_SQL = """
WITH d AS (
    SELECT substr(order_delivered_customer_date, 1, 4) AS y,
           substr(order_delivered_customer_date, 6, 2) AS m
    FROM olist_orders
    WHERE order_status = 'delivered' AND order_delivered_customer_date IS NOT NULL
),
years AS (SELECT y, COUNT(*) AS c FROM d WHERE y IN ('2016','2017','2018') GROUP BY y),
lowyear AS (SELECT y FROM years ORDER BY c ASC LIMIT 1)
SELECT MAX(cnt) AS HIGHEST_MONTHLY_VOLUME
FROM (SELECT m, COUNT(*) AS cnt FROM d JOIN lowyear ON d.y = lowyear.y GROUP BY m)
"""


def run(reg):
    rec = manifest_record(IID)
    ref = sqlite_df(rec, REFERENCE_SQL)
    ref_v, ref_ok = score(ref, rec)

    m = import_table(reg, rec, IID, "olist_orders",
                     ["order_id", "order_status", "order_purchase_timestamp", "order_delivered_customer_date"])
    coll = reg.db[m["collection"]]

    PIPELINE = [
        {"$match": {"order_status": "delivered", "order_delivered_customer_date": {"$ne": None}}},
        {"$addFields": {"y": {"$substrCP": ["$order_delivered_customer_date", 0, 4]},
                        "m": {"$substrCP": ["$order_delivered_customer_date", 5, 2]}}},
        {"$match": {"y": {"$in": ["2016", "2017", "2018"]}}},
        {"$group": {"_id": {"y": "$y", "m": "$m"}, "c": {"$sum": 1}}},
        {"$group": {"_id": "$_id.y", "total": {"$sum": "$c"},
                    "months": {"$push": {"m": "$_id.m", "c": "$c"}}}},
        {"$sort": {"total": 1}},
        {"$limit": 1},
        {"$unwind": "$months"},
        {"$sort": {"months.c": -1}},
        {"$limit": 1},
        {"$project": {"_id": 0, "HIGHEST_MONTHLY_VOLUME": "$months.c"}},
    ]
    res = list(coll.aggregate(PIPELINE))
    pred = pd.DataFrame(res, columns=["HIGHEST_MONTHLY_VOLUME"])
    v, ok = score(pred, rec)

    artifact = {
        "pattern": "year/month grouping + weakest-year selection + per-year max",
        "gold_sql": None, "gold_sql_reproduces_gold_csv": "n/a (no shipped gold SQL)",
        "reference_sql": REFERENCE_SQL, "reference_sql_matches_gold": ref_ok,
        "relational_semantics": ("Delivered orders grouped by delivery month; among 2016-2018 pick the year with the fewest deliveries; "
                                 "return that year's highest monthly delivered volume (2016 -> 205)."),
        "source_schema": {"olist_orders": ["order_id", "order_status", "order_purchase_timestamp", "order_delivered_customer_date"]},
        "mongodb": {"database": "codegen_capstone_pilot", "imports": [m], "_id": "default ObjectId"},
        "mql": {"collection": coll.name, "method": "aggregate", "pipeline": PIPELINE},
        "mongo_result": res,
        "validation": {"per_gold": v, "matched_any_gold": ok},
        "assumptions": [
            "The month/year come from order_delivered_customer_date (the gold's counts match the delivery-date grouping, not purchase date).",
            "Null delivery dates are excluded (8 rows), matching the SQL IS NOT NULL filter.",
        ],
    }
    return ("passed" if ok else "failed"), matched_reason(v, ok), artifact
