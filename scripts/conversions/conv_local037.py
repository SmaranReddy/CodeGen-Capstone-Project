"""local037 - top 3 categories by their most-used payment type's payment count.

No shipped gold SQL: a labelled reference SQL is authored and verified against the gold
CSV first. A payment is identified by (order_id, payment_sequential); per category keep
the payment type with the most payments, then take the three largest counts.
"""
import pandas as pd

from conversion_kit import (import_table, manifest_record, matched_reason, pattern_record, score, sqlite_df)

IID = "local037"

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
mx AS (SELECT cat, MAX(c) AS m FROM cp GROUP BY cat),
best AS (SELECT cp.cat, cp.pt, cp.c FROM cp JOIN mx ON mx.cat = cp.cat AND mx.m = cp.c)
SELECT cat AS product_category_name, pt AS payment_type, c AS payment_count
FROM best ORDER BY c DESC LIMIT 3
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
        {"$sort": {"_id.cat": 1, "c": -1}},
        {"$group": {"_id": "$_id.cat", "pt": {"$first": "$_id.pt"}, "c": {"$first": "$c"}}},
        {"$sort": {"c": -1}},
        {"$limit": 3},
        {"$project": {"_id": 0, "product_category_name": "$_id", "payment_type": "$pt", "payment_count": "$c"}},
    ]
    res = list(coll.aggregate(PIPELINE, allowDiskUse=True))
    pred = pd.DataFrame(res, columns=["product_category_name", "payment_type", "payment_count"])
    v, ok = score(pred, rec)

    artifact = {
        "pattern": "category/method distinct-payment counts + per-category argmax + top-N",
        "gold_sql": None, "gold_sql_reproduces_gold_csv": "n/a (no shipped gold SQL)",
        "reference_sql": REFERENCE_SQL, "reference_sql_matches_gold": ref_ok,
        "relational_semantics": ("Distinct payments per (category, payment type); per category the most-used type and its count; three largest counts."),
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
            "Payments are counted as distinct (order_id, payment_sequential) values.",
            "Category labels use the Portuguese product_category_name (gold _a shows the same values as gold _b translations).",
        ],
    }
    return ("passed" if ok else "failed"), matched_reason(v, ok), artifact
