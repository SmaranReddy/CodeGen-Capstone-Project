"""local028 - delivered orders pivoted by month for 2016, 2017 and 2018.

No shipped gold SQL: a labelled reference SQL is authored and verified against the gold
CSV first. Months are taken from the customer delivery date. Only the three year columns
are compared (condition_cols=[1,2,3]).
"""
import pandas as pd

from conversion_kit import (import_table, manifest_record, matched_reason, pattern_record, score, sqlite_df)

IID = "local028"

REFERENCE_SQL = """
WITH d AS (
    SELECT substr(order_delivered_customer_date, 1, 4) AS y,
           substr(order_delivered_customer_date, 6, 2) AS m
    FROM olist_orders
    WHERE order_status = 'delivered' AND order_delivered_customer_date IS NOT NULL
)
SELECT m AS aMonth,
       SUM(CASE WHEN y = '2016' THEN 1 ELSE 0 END) AS a2016,
       SUM(CASE WHEN y = '2017' THEN 1 ELSE 0 END) AS a2017,
       SUM(CASE WHEN y = '2018' THEN 1 ELSE 0 END) AS a2018
FROM d GROUP BY m ORDER BY m
"""


def _year_sum(col):
    return {"$reduce": {"input": "$ys", "initialValue": 0, "in": {"$add": [
        "$$value", {"$cond": [{"$eq": ["$$this.y", col]}, "$$this.c", 0]}]}}}


def run(reg):
    rec = manifest_record(IID)
    ref = sqlite_df(rec, REFERENCE_SQL)
    ref_v, ref_ok = score(ref, rec)

    m = import_table(reg, rec, IID, "olist_orders",
                     ["order_id", "order_status", "order_delivered_customer_date"])
    coll = reg.db[m["collection"]]

    PIPELINE = [
        {"$match": {"order_status": "delivered", "order_delivered_customer_date": {"$ne": None}}},
        {"$addFields": {"y": {"$substrCP": ["$order_delivered_customer_date", 0, 4]},
                        "m": {"$substrCP": ["$order_delivered_customer_date", 5, 2]}}},
        {"$match": {"y": {"$in": ["2016", "2017", "2018"]}}},
        {"$group": {"_id": {"m": "$m", "y": "$y"}, "c": {"$sum": 1}}},
        {"$group": {"_id": "$_id.m", "ys": {"$push": {"y": "$_id.y", "c": "$c"}}}},
        {"$project": {"_id": 0, "aMonth": "$_id",
                      "a2016": _year_sum("2016"),
                      "a2017": _year_sum("2017"),
                      "a2018": _year_sum("2018")}},
    ]
    res = list(coll.aggregate(PIPELINE, allowDiskUse=True))
    pred = pd.DataFrame(res, columns=["aMonth", "a2016", "a2017", "a2018"])
    v, ok = score(pred, rec)

    artifact = {
        "pattern": "group by year/month + conditional sums pivot (array reduce)",
        "gold_sql": None, "gold_sql_reproduces_gold_csv": "n/a (no shipped gold SQL)",
        "reference_sql": REFERENCE_SQL, "reference_sql_matches_gold": ref_ok,
        "relational_semantics": ("Delivered orders counted by delivery month and year; pivot so each of 2016/2017/2018 is a column and each "
                                 "month a row."),
        "source_schema": {"olist_orders": ["order_id", "order_status", "order_delivered_customer_date"]},
        "mongodb": {"database": "codegen_capstone_pilot", "imports": [m], "_id": "default ObjectId"},
        "mql": {"collection": coll.name, "method": "aggregate", "options": {"allowDiskUse": True}, "pipeline": PIPELINE},
        "mongo_result": res,
        "validation": {"per_gold": v, "matched_any_gold": ok},
        "assumptions": [
            "Months use order_delivered_customer_date; nulls excluded.",
            "condition_cols=[1,2,3] compares only the three year columns, so the aMonth label is not scored.",
        ],
    }
    return ("passed" if ok else "failed"), matched_reason(v, ok), artifact
