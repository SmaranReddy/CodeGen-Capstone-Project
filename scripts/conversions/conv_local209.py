"""local209 - top store by order count and its delivered-order ratio.

No shipped gold SQL: a labelled reference SQL is authored and verified against the gold
CSV first. Orders are joined to deliveries on delivery_order_id; an order counts as
delivered when its delivery row has status 'DELIVERED'. The store with the most orders is
selected (ties broken by store_id) and its delivered/total ratio is reported. Reproduces
gold variant `_a` (5 columns).
"""
import pandas as pd

from conversion_kit import import_table, manifest_record, matched_reason, score, sqlite_df

IID = "local209"

REFERENCE_SQL = """
WITH per_store AS (
  SELECT o.store_id AS store_id, COUNT(*) AS total_orders
  FROM orders o GROUP BY o.store_id),
top AS (SELECT store_id, total_orders FROM per_store ORDER BY total_orders DESC, store_id ASC LIMIT 1)
SELECT s.store_id AS store_id, s.store_name AS store_name, t.total_orders AS TOTAL_ORDERS,
  (SELECT COUNT(DISTINCT o.order_id) FROM orders o JOIN deliveries d ON d.delivery_order_id = o.delivery_order_id
     WHERE o.store_id = t.store_id AND d.delivery_status = 'DELIVERED') AS DELIVERED_ORDERS
FROM top t JOIN stores s ON s.store_id = t.store_id
"""


def _pipeline(deliveries_coll, stores_coll):
    return [
        {"$lookup": {"from": deliveries_coll, "localField": "delivery_order_id",
                     "foreignField": "delivery_order_id", "as": "d"}},
        {"$addFields": {"is_delivered": {"$cond": [{"$in": ["DELIVERED", "$d.delivery_status"]}, 1, 0]}}},
        {"$group": {"_id": "$store_id", "TOTAL_ORDERS": {"$sum": 1},
                    "DELIVERED_ORDERS": {"$sum": "$is_delivered"}}},
        {"$sort": {"TOTAL_ORDERS": -1, "_id": 1}},
        {"$limit": 1},
        {"$lookup": {"from": stores_coll, "localField": "_id", "foreignField": "store_id", "as": "s"}},
        {"$unwind": "$s"},
        {"$project": {"_id": 0, "store_id": "$_id", "store_name": "$s.store_name",
                      "TOTAL_ORDERS": 1, "DELIVERED_ORDERS": 1,
                      "RATIO": {"$divide": ["$DELIVERED_ORDERS", "$TOTAL_ORDERS"]}}},
    ]


COLS = ["store_id", "store_name", "TOTAL_ORDERS", "DELIVERED_ORDERS", "RATIO"]


def run(reg):
    rec = manifest_record(IID)
    ref = sqlite_df(rec, REFERENCE_SQL)
    ref["RATIO"] = ref["DELIVERED_ORDERS"] / ref["TOTAL_ORDERS"]
    ref_v, ref_ok = score(ref, rec)

    orders = import_table(reg, rec, IID, "orders", ["order_id", "store_id", "delivery_order_id"])
    deliveries = import_table(reg, rec, IID, "deliveries", ["delivery_order_id", "delivery_status"],
                              index=["delivery_order_id"])
    stores = import_table(reg, rec, IID, "stores", ["store_id", "store_name"], index=["store_id"])

    coll = reg.db[orders["collection"]]
    pipeline = _pipeline(deliveries["collection"], stores["collection"])
    res = list(coll.aggregate(pipeline, allowDiskUse=True))
    pred = pd.DataFrame(res, columns=COLS)
    v, ok = score(pred, rec)

    artifact = {
        "pattern": "join orders->deliveries via $lookup, per-store aggregation, $sort+$limit top store, store lookup",
        "gold_sql": None, "gold_sql_reproduces_gold_csv": "n/a (no shipped gold SQL)",
        "reference_sql": REFERENCE_SQL, "reference_sql_matches_gold": ref_ok,
        "relational_semantics": ("Orders grouped by store; the store with the most orders (tie -> lowest store_id); "
                                 "among its orders, the ratio whose delivery_order_id maps to a DELIVERED delivery."),
        "source_schema": {"orders": ["order_id", "store_id", "delivery_order_id"],
                          "deliveries": ["delivery_order_id", "delivery_status"],
                          "stores": ["store_id", "store_name"]},
        "mongodb": {"database": "codegen_capstone_pilot", "imports": [orders, deliveries, stores]},
        "mql": {"collection": coll.name, "method": "aggregate", "options": {"allowDiskUse": True}, "pipeline": pipeline},
        "mongo_result": res,
        "validation": {"per_gold": v, "matched_any_gold": ok},
        "assumptions": ["Delivered order = an order whose delivery_order_id links to a delivery with status 'DELIVERED'.",
                        "Counts are over orders (one row per order_id), so the delivery join does not multiply rows.",
                        "Top store ties broken by ascending store_id.",
                        "Validated against gold variant _a (5 columns)."],
    }
    return ("passed" if ok else "failed"), matched_reason(v, ok), artifact
