"""local004 - top-3 customers by average payment per order (with order count / lifespan).

customers -> orders (customer_id) -> order_payments (order_id); group by
customer_unique_id: COUNT(DISTINCT order_id), SUM(payment_value), MIN/MAX timestamp;
AOV = round(total/orders, 2). condition_cols=[0,1,2] compare id, order_count, AOV.
"""
import pandas as pd

from conversion_kit import (import_table, manifest_record, matched_reason, pattern_record, score, sqlite_df)

IID = "local004"


def run(reg):
    rec = manifest_record(IID)
    gold_sql = open(rec["gold_sql_path"], encoding="utf-8").read()
    ref = sqlite_df(rec, gold_sql)
    ref_v, ref_ok = score(ref, rec)

    m_cust = import_table(reg, rec, IID, "customers", ["customer_id", "customer_unique_id"],
                          index=["customer_id"])
    m_ord = import_table(reg, rec, IID, "orders", ["order_id", "customer_id", "order_purchase_timestamp"],
                         index=["customer_id"])
    m_pay = import_table(reg, rec, IID, "order_payments", ["order_id", "payment_value"],
                         index=["order_id"])
    coll = reg.db[m_cust["collection"]]

    PIPELINE = [
        {"$lookup": {"from": m_ord["collection"], "localField": "customer_id",
                     "foreignField": "customer_id", "as": "o"}},
        {"$unwind": "$o"},
        {"$lookup": {"from": m_pay["collection"], "localField": "o.order_id",
                     "foreignField": "order_id", "as": "pay"}},
        {"$unwind": "$pay"},  # inner join: an order with no payment row is dropped, as in the shipped SQL
        {"$group": {"_id": "$customer_unique_id",
                    "order_ids": {"$addToSet": "$o.order_id"},
                    "total_payment": {"$sum": "$pay.payment_value"},
                    "first_ts": {"$min": "$o.order_purchase_timestamp"},
                    "last_ts": {"$max": "$o.order_purchase_timestamp"}}},
        {"$project": {"order_count": {"$size": "$order_ids"}, "total_payment": 1,
                      "first_ts": 1, "last_ts": 1}},
        {"$addFields": {"AOV": {"$round": [{"$divide": ["$total_payment", "$order_count"]}, 2]}}},
        {"$addFields": {"days": {"$dateDiff": {
            "startDate": {"$dateFromString": {"dateString": "$first_ts", "format": "%Y-%m-%d %H:%M:%S"}},
            "endDate": {"$dateFromString": {"dateString": "$last_ts", "format": "%Y-%m-%d %H:%M:%S"}},
            "unit": "millisecond"}}}},
        {"$addFields": {"ACL": {"$cond": [{"$lt": [{"$divide": ["$days", 86400000]}, 7]}, 1,
                                          {"$divide": [{"$divide": ["$days", 86400000]}, 7]}]}}},
        {"$sort": {"AOV": -1}},
        {"$limit": 3},
        {"$project": {"_id": 0, "customer_unique_id": "$_id", "order_count": 1, "AOV": 1, "ACL": 1}},
    ]
    res = list(coll.aggregate(PIPELINE, allowDiskUse=True))
    pred = pd.DataFrame(res, columns=["customer_unique_id", "order_count", "AOV", "ACL"])
    v, ok = score(pred, rec)

    artifact = {
        "pattern": "two-hop join + group-by with distinct count / sum / min / max + top-3",
        "gold_sql": gold_sql, "gold_sql_reproduces_gold_csv": ref_ok,
        "reference_sql": gold_sql, "reference_sql_matches_gold": ref_ok,
        "relational_semantics": ("Per customer_unique_id: orders = COUNT(DISTINCT order_id), total = SUM(payment_value), "
                                 "first/last purchase; AOV = round(total/orders,2); lifespan weeks = (last-first)/7 (or 1 if <7d); top 3 by AOV."),
        "source_schema": {"customers": ["customer_id TEXT", "customer_unique_id TEXT"],
                          "orders": ["order_id TEXT", "customer_id TEXT", "order_purchase_timestamp TEXT"],
                          "order_payments": ["order_id TEXT", "payment_value REAL"]},
        "mongodb": {"database": "codegen_capstone_pilot", "imports": [m_cust, m_ord, m_pay], "_id": "default ObjectId"},
        "mql": {"collection": coll.name, "method": "aggregate", "options": {"allowDiskUse": True}, "pipeline": PIPELINE},
        "mongo_result": res,
        "validation": {"per_gold": v, "matched_any_gold": ok},
        "assumptions": [
            "order_count uses COUNT(DISTINCT order_id) via $addToSet + $size (payment fan-out must not inflate it).",
            "Lifespan weeks is not compared (condition_cols=[0,1,2]); it is kept for faithfulness.",
            "Timestamps are ISO text parsed with an explicit format; days then divided by 7 as in the SQL.",
        ],
    }
    return ("passed" if ok else "failed"), matched_reason(v, ok), artifact
