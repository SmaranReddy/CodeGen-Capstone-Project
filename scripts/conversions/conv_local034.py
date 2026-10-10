"""local034 - average per-category max payments for the category's preferred method.

No shipped gold SQL: a labelled reference SQL is authored and verified against the gold
CSV first. A payment is identified by (order_id, payment_sequential); the preferred method
per category is the payment type with the most such payments, and the metric averages that
per-category maximum over all product categories.
"""
import pandas as pd

from conversion_kit import (import_table, manifest_record, matched_reason, pattern_record, score, sqlite_df)

IID = "local034"

REFERENCE_SQL = """
WITH cp AS (
    SELECT pr.product_category_name AS cat, p.payment_type AS pt,
           COUNT(DISTINCT p.order_id || '-' || p.payment_sequential) AS c
    FROM olist_order_payments p
    JOIN olist_orders o ON p.order_id = o.order_id
    JOIN olist_order_items oi ON o.order_id = oi.order_id
    JOIN olist_products pr ON oi.product_id = pr.product_id
    GROUP BY cat, pt
),
mx AS (SELECT cat, MAX(c) AS m FROM cp GROUP BY cat)
SELECT AVG(m) AS avg_top_payment_count FROM mx
"""


def run(reg):
    rec = manifest_record(IID)
    ref = sqlite_df(rec, REFERENCE_SQL)
    ref_v, ref_ok = score(ref, rec)

    m_pay = import_table(reg, rec, IID, "olist_order_payments",
                         ["order_id", "payment_sequential", "payment_type"])
    m_ord = import_table(reg, rec, IID, "olist_orders", ["order_id"], index=["order_id"])
    m_itm = import_table(reg, rec, IID, "olist_order_items", ["order_id", "product_id"], index=["order_id"])
    m_prd = import_table(reg, rec, IID, "olist_products", ["product_id", "product_category_name"], index=["product_id"])
    coll = reg.db[m_pay["collection"]]

    key = {"$concat": ["$order_id", "-", {"$toString": "$payment_sequential"}]}
    PIPELINE = [
        {"$lookup": {"from": m_ord["collection"], "localField": "order_id",
                     "foreignField": "order_id", "as": "o"}},
        {"$unwind": "$o"},
        {"$lookup": {"from": m_itm["collection"], "localField": "order_id",
                     "foreignField": "order_id", "as": "oi"}},
        {"$unwind": "$oi"},
        {"$lookup": {"from": m_prd["collection"], "localField": "oi.product_id",
                     "foreignField": "product_id", "as": "pr"}},
        {"$unwind": "$pr"},
        {"$group": {"_id": {"cat": "$pr.product_category_name", "pt": "$payment_type"},
                    "keys": {"$addToSet": key}}},
        {"$addFields": {"c": {"$size": "$keys"}}},
        {"$group": {"_id": "$_id.cat", "m": {"$max": "$c"}}},
        {"$group": {"_id": None, "avg_top_payment_count": {"$avg": "$m"}}},
        {"$project": {"_id": 0, "avg_top_payment_count": 1}},
    ]
    res = list(coll.aggregate(PIPELINE, allowDiskUse=True))
    pred = pd.DataFrame(res, columns=["avg_top_payment_count"])
    v, ok = score(pred, rec)

    artifact = {
        "pattern": "category/method grouping with distinct payment keys + per-category max + average",
        "gold_sql": None, "gold_sql_reproduces_gold_csv": "n/a (no shipped gold SQL)",
        "reference_sql": REFERENCE_SQL, "reference_sql_matches_gold": ref_ok,
        "relational_semantics": ("Payments per (category, payment type) counted as distinct (order_id, payment_sequential); preferred method per "
                                 "category = highest count; average that maximum across categories."),
        "source_schema": {"olist_order_payments": ["order_id", "payment_sequential", "payment_type"],
                          "olist_orders": ["order_id"], "olist_order_items": ["order_id", "product_id"],
                          "olist_products": ["product_id", "product_category_name"]},
        "mongodb": {"database": "codegen_capstone_pilot", "imports": [m_pay, m_ord, m_itm, m_prd],
                    "_id": "default ObjectId",
                    "indexes": ["olist_orders.order_id", "olist_order_items.order_id", "olist_products.product_id"]},
        "mql": {"collection": coll.name, "method": "aggregate", "options": {"allowDiskUse": True}, "pipeline": PIPELINE},
        "mongo_result": res,
        "validation": {"per_gold": v, "matched_any_gold": ok},
        "assumptions": [
            "A payment row is identified by order_id + payment_sequential; joining order items must not double count it, hence $addToSet.",
            "Category labels use the Portuguese product_category_name (gold _a).",
        ],
    }
    return ("passed" if ok else "failed"), matched_reason(v, ok), artifact
