"""local030 - averages of the 5 lowest-paying cities' delivered-order totals.

No shipped gold SQL: a labelled reference SQL is authored and verified against the gold
CSV first. For each city: total payments and delivered-order count; take the 5 cities with
the smallest total payment, then average their totals and their order counts.
"""
import pandas as pd

from conversion_kit import (import_table, manifest_record, matched_reason, pattern_record, score, sqlite_df)

IID = "local030"

REFERENCE_SQL = """
WITH delivered AS (
    SELECT c.customer_city AS city, o.order_id AS oid
    FROM olist_orders o JOIN olist_customers c ON o.customer_id = c.customer_id
    WHERE o.order_status = 'delivered'
),
pay AS (
    SELECT d.city, SUM(p.payment_value) AS totp, COUNT(DISTINCT d.oid) AS ocount
    FROM delivered d JOIN olist_order_payments p ON d.oid = p.order_id
    GROUP BY d.city
),
low AS (SELECT * FROM pay ORDER BY totp ASC LIMIT 5)
SELECT AVG(totp) AS AVG_TOTAL_PAYMENT, AVG(ocount) AS AVG_DELIVERED_ORDER_COUNT FROM low
"""


def run(reg):
    rec = manifest_record(IID)
    ref = sqlite_df(rec, REFERENCE_SQL)
    ref_v, ref_ok = score(ref, rec)

    m_o = import_table(reg, rec, IID, "olist_orders", ["order_id", "customer_id", "order_status"])
    m_c = import_table(reg, rec, IID, "olist_customers", ["customer_id", "customer_city"], index=["customer_id"])
    m_p = import_table(reg, rec, IID, "olist_order_payments", ["order_id", "payment_value"], index=["order_id"])
    coll = reg.db[m_o["collection"]]

    PIPELINE = [
        {"$match": {"order_status": "delivered"}},
        {"$lookup": {"from": m_c["collection"], "localField": "customer_id",
                     "foreignField": "customer_id", "as": "c"}},
        {"$unwind": "$c"},
        {"$lookup": {"from": m_p["collection"], "localField": "order_id",
                     "foreignField": "order_id", "as": "p"}},
        {"$unwind": "$p"},
        {"$group": {"_id": {"city": "$c.customer_city", "oid": "$order_id"},
                    "op": {"$sum": "$p.payment_value"}}},
        {"$group": {"_id": "$_id.city", "totp": {"$sum": "$op"}, "ocount": {"$sum": 1}}},
        {"$sort": {"totp": 1}},
        {"$limit": 5},
        {"$group": {"_id": None,
                    "AVG_TOTAL_PAYMENT": {"$avg": "$totp"},
                    "AVG_DELIVERED_ORDER_COUNT": {"$avg": "$ocount"}}},
        {"$project": {"_id": 0, "AVG_TOTAL_PAYMENT": 1, "AVG_DELIVERED_ORDER_COUNT": 1}},
    ]
    res = list(coll.aggregate(PIPELINE, allowDiskUse=True))
    pred = pd.DataFrame(res, columns=["AVG_TOTAL_PAYMENT", "AVG_DELIVERED_ORDER_COUNT"])
    v, ok = score(pred, rec)

    artifact = {
        "pattern": "filtered joins + two-level group (order then city) + lowest-N + averages",
        "gold_sql": None, "gold_sql_reproduces_gold_csv": "n/a (no shipped gold SQL)",
        "reference_sql": REFERENCE_SQL, "reference_sql_matches_gold": ref_ok,
        "relational_semantics": ("Delivered orders only; per city total payments and distinct delivered-order count; five cities with the "
                                 "smallest total payment; average of their totals and of their counts."),
        "source_schema": {"olist_orders": ["order_id", "customer_id", "order_status"],
                          "olist_customers": ["customer_id", "customer_city"], "olist_order_payments": ["order_id", "payment_value"]},
        "mongodb": {"database": "codegen_capstone_pilot", "imports": [m_o, m_c, m_p], "_id": "default ObjectId",
                    "indexes": ["olist_customers.customer_id", "olist_order_payments.order_id"]},
        "mql": {"collection": coll.name, "method": "aggregate", "options": {"allowDiskUse": True}, "pipeline": PIPELINE},
        "mongo_result": res,
        "validation": {"per_gold": v, "matched_any_gold": ok},
        "assumptions": [
            "An order can have several payment rows; payments are summed per order before per-city aggregation.",
            "The inner join in SQL drops delivered orders without payments, mirrored by $unwind.",
        ],
    }
    return ("passed" if ok else "failed"), matched_reason(v, ok), artifact
