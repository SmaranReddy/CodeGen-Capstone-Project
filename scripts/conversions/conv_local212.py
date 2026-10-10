"""local212 - top 5 drivers by average daily deliveries.

No shipped gold SQL: a labelled reference SQL is authored and verified against the gold
CSV first. Each delivery is linked to its order (on delivery_order_id) to obtain the order
calendar day; per driver the number of distinct active days and total deliveries are
counted, and the average daily deliveries = total / active days. Top 5 by average.
Reproduces gold variant `_c` (driver_id / AVG_DAILY_DELIVERIES).
"""
import pandas as pd

from conversion_kit import import_table, manifest_record, matched_reason, score, sqlite_df

IID = "local212"


REFERENCE_SQL = """
SELECT dv.driver_id AS driver_id, COUNT(*) AS total_deliveries,
  COUNT(DISTINCT printf('%04d-%02d-%02d', o.order_created_year, o.order_created_month, o.order_created_day)) AS active_days
FROM deliveries dv JOIN orders o ON o.delivery_order_id = dv.delivery_order_id
WHERE dv.driver_id IS NOT NULL
GROUP BY dv.driver_id
"""


def _pipeline(orders_coll):
    return [
        {"$lookup": {"from": orders_coll, "localField": "delivery_order_id",
                     "foreignField": "delivery_order_id", "as": "o"}},
        {"$unwind": "$o"},
        {"$match": {"driver_id": {"$ne": None}}},
        {"$addFields": {"day": {"$concat": [{"$toString": "$o.order_created_year"}, "-",
                                            {"$toString": "$o.order_created_month"}, "-",
                                            {"$toString": "$o.order_created_day"}]}}},
        {"$group": {"_id": {"driver": "$driver_id", "day": "$day"}, "c": {"$sum": 1}}},
        {"$group": {"_id": "$_id.driver", "total_deliveries": {"$sum": "$c"},
                    "active_days": {"$sum": 1}}},
        {"$addFields": {"AVG_DAILY_DELIVERIES": {"$divide": ["$total_deliveries", "$active_days"]}}},
        {"$sort": {"AVG_DAILY_DELIVERIES": -1, "_id": 1}},
        {"$limit": 5},
        {"$project": {"_id": 0, "driver_id": "$_id", "AVG_DAILY_DELIVERIES": 1}},
    ]


COLS = ["driver_id", "AVG_DAILY_DELIVERIES"]


def run(reg):
    rec = manifest_record(IID)
    ref = sqlite_df(rec, REFERENCE_SQL)
    ref["AVG_DAILY_DELIVERIES"] = ref["total_deliveries"] / ref["active_days"]
    ref = ref.sort_values("AVG_DAILY_DELIVERIES", ascending=False).head(5)
    ref_v, ref_ok = score(ref, rec)

    deliveries = import_table(reg, rec, IID, "deliveries", ["delivery_order_id", "driver_id"])
    orders = import_table(reg, rec, IID, "orders",
                          ["delivery_order_id", "order_created_year", "order_created_month", "order_created_day"],
                          index=["delivery_order_id"])

    coll = reg.db[deliveries["collection"]]
    pipeline = _pipeline(orders["collection"])
    res = list(coll.aggregate(pipeline, allowDiskUse=True))
    pred = pd.DataFrame(res, columns=COLS)
    v, ok = score(pred, rec)

    artifact = {
        "pattern": "join deliveries->orders via $lookup, two-stage $group (driver,day)->driver, average, top-5",
        "gold_sql": None, "gold_sql_reproduces_gold_csv": "n/a (no shipped gold SQL)",
        "reference_sql": REFERENCE_SQL, "reference_sql_matches_gold": ref_ok,
        "relational_semantics": ("Per driver: average deliveries per active day = total deliveries / distinct order days; "
                                 "top 5 drivers by that average."),
        "source_schema": {"deliveries": ["delivery_order_id", "driver_id"],
                          "orders": ["delivery_order_id", "order_created_year", "order_created_month", "order_created_day"]},
        "mongodb": {"database": "codegen_capstone_pilot", "imports": [deliveries, orders]},
        "mql": {"collection": coll.name, "method": "aggregate", "options": {"allowDiskUse": True}, "pipeline": pipeline},
        "mongo_result": res,
        "validation": {"per_gold": v, "matched_any_gold": ok},
        "assumptions": ["An active day is a distinct (year, month, day) among a driver's linked orders.",
                        "Deliveries without a matching order or driver are ignored.",
                        "Average = total deliveries / active days (not a plain mean of per-day counts).",
                        "Validated against gold variant _c (driver_id / AVG_DAILY_DELIVERIES)."],
    }
    return ("passed" if ok else "failed"), matched_reason(v, ok), artifact
