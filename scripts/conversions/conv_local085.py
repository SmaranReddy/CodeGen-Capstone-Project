"""local085 - the three employees (among those with >50 orders) with the highest late-order %.

No shipped gold SQL: a labelled reference SQL is authored and verified against the gold
CSV first. Late = shippeddate >= requireddate (ISO text comparisons).
"""
import pandas as pd

from conversion_kit import (import_table, manifest_record, matched_reason, pattern_record, score, sqlite_df)

IID = "local085"

REFERENCE_SQL = """
WITH per_emp AS (
    SELECT employeeid,
           COUNT(*) AS total_orders,
           SUM(CASE WHEN shippeddate >= requireddate THEN 1 ELSE 0 END) AS late_orders
    FROM Orders
    GROUP BY employeeid
)
SELECT employeeid AS EMPLOYEE_ID,
       late_orders AS LATE_ORDERS,
       (100.0 * late_orders / total_orders) AS LATE_ORDER_PERCENTAGE
FROM per_emp
WHERE total_orders > 50
ORDER BY LATE_ORDER_PERCENTAGE DESC
LIMIT 3
"""


def run(reg):
    rec = manifest_record(IID)
    ref = sqlite_df(rec, REFERENCE_SQL)
    ref_v, ref_ok = score(ref, rec)

    m = import_table(reg, rec, IID, "Orders", ["orderid", "employeeid", "requireddate", "shippeddate"])
    coll = reg.db[m["collection"]]

    PIPELINE = [
        {"$group": {"_id": "$employeeid",
                    "total_orders": {"$sum": 1},
                    "late_orders": {"$sum": {"$cond": [{"$gte": ["$shippeddate", "$requireddate"]}, 1, 0]}}}},
        {"$match": {"total_orders": {"$gt": 50}}},
        {"$addFields": {"LATE_ORDER_PERCENTAGE": {"$divide": [{"$multiply": [100, "$late_orders"]}, "$total_orders"]}}},
        {"$sort": {"LATE_ORDER_PERCENTAGE": -1}},
        {"$limit": 3},
        {"$project": {"_id": 0, "EMPLOYEE_ID": "$_id", "LATE_ORDERS": "$late_orders",
                      "LATE_ORDER_PERCENTAGE": 1}},
    ]
    res = list(coll.aggregate(PIPELINE))
    pred = pd.DataFrame(res, columns=["EMPLOYEE_ID", "LATE_ORDERS", "LATE_ORDER_PERCENTAGE"])
    v, ok = score(pred, rec)

    artifact = {
        "pattern": "group + conditional sum + derived ratio + top-N",
        "gold_sql": None, "gold_sql_reproduces_gold_csv": "n/a (no shipped gold SQL)",
        "reference_sql": REFERENCE_SQL, "reference_sql_matches_gold": ref_ok,
        "relational_semantics": ("Late = shippeddate >= requireddate; per-employee late count and percentage; keep employees with "
                                 ">50 orders; top 3 by percentage."),
        "source_schema": {"Orders": ["orderid INT", "employeeid INT", "requireddate TEXT", "shippeddate TEXT (nullable)"]},
        "mongodb": {"database": "codegen_capstone_pilot", "imports": [m], "_id": "default ObjectId"},
        "mql": {"collection": coll.name, "method": "aggregate", "pipeline": PIPELINE},
        "mongo_result": res,
        "validation": {"per_gold": v, "matched_any_gold": ok},
        "assumptions": [
            "Dates compared as ISO text (SQLite DATE affinity); NULL shippeddate never counts as late.",
            "Percentage returned unrounded; the comparator uses abs tolerance 0.01.",
        ],
    }
    return ("passed" if ok else "failed"), matched_reason(v, ok), artifact
