"""local056 - customer with the highest average monthly change in payment amounts.

No shipped gold SQL: a labelled reference SQL is authored and verified against the gold
CSV first. Monthly payment totals per customer; average absolute month-over-month change.
"""
import pandas as pd

from conversion_kit import import_table, manifest_record, matched_reason, pattern_record, score, sqlite_df

IID = "local056"

REFERENCE_SQL = """
WITH monthly AS (SELECT customer_id, substr(payment_date, 1, 7) AS ym, SUM(amount) AS m
                 FROM payment GROUP BY customer_id, ym),
d AS (SELECT customer_id, ABS(m - LAG(m) OVER (PARTITION BY customer_id ORDER BY ym)) AS chg FROM monthly)
SELECT UPPER(c.first_name || ' ' || c.last_name) AS CUSTOMER_FULL_NAME, AVG(d.chg) AS avg_chg
FROM d JOIN customer c ON c.customer_id = d.customer_id
GROUP BY d.customer_id ORDER BY avg_chg DESC LIMIT 1
"""

PIPELINE_TEMPLATE = [
    {"$set": {"ym": {"$substrCP": ["$payment_date", 0, 7]}}},
    {"$group": {"_id": {"cid": "$customer_id", "ym": "$ym"}, "m": {"$sum": "$amount"}}},
    {"$sort": {"_id.cid": 1, "_id.ym": 1}},
    {"$group": {"_id": "$_id.cid", "ms": {"$push": "$m"}}},
    {"$project": {"avg_chg": {"$avg": {"$map": {
        "input": {"$range": [1, {"$size": "$ms"}]},
        "as": "i",
        "in": {"$abs": {"$subtract": [{"$arrayElemAt": ["$ms", "$$i"]},
                                      {"$arrayElemAt": ["$ms", {"$subtract": ["$$i", 1]}]}]}}}}}}},
    {"$sort": {"avg_chg": -1}},
    {"$limit": 1},
    {"$lookup": {"from": "__CUSTOMER__", "localField": "_id", "foreignField": "customer_id", "as": "c"}},
    {"$unwind": "$c"},
    {"$project": {"_id": 0, "CUSTOMER_FULL_NAME": {"$toUpper": {"$concat": ["$c.first_name", " ", "$c.last_name"]}}}},
]


def run(reg):
    rec = manifest_record(IID)
    ref = sqlite_df(rec, REFERENCE_SQL)
    ref_v, ref_ok = score(ref, rec)

    pay = import_table(reg, rec, IID, "payment", ["customer_id", "amount", "payment_date"], index=["customer_id"])
    cust = import_table(reg, rec, IID, "customer", ["customer_id", "first_name", "last_name"], index=["customer_id"])

    coll = reg.db[pay["collection"]]
    PIPELINE = []
    for s in PIPELINE_TEMPLATE:
        if "$lookup" in s:
            s = {"$lookup": {**s["$lookup"], "from": cust["collection"]}}
        PIPELINE.append(s)
    res = list(coll.aggregate(PIPELINE, allowDiskUse=True))
    pred = pd.DataFrame(res, columns=["CUSTOMER_FULL_NAME"])
    v, ok = score(pred, rec)

    artifact = {
        "pattern": "monthly rollup per customer + array window (absolute consecutive diff) + top-1",
        "gold_sql": None, "gold_sql_reproduces_gold_csv": "n/a (no shipped gold SQL)",
        "reference_sql": REFERENCE_SQL, "reference_sql_matches_gold": ref_ok,
        "relational_semantics": ("Per customer sum payments per calendar month (YYYY-MM); average the absolute change between "
                                 "consecutive months; return the customer with the highest such average, name uppercased."),
        "source_schema": {"payment": ["customer_id", "amount", "payment_date"], "customer": ["customer_id", "first_name", "last_name"]},
        "mongodb": {"database": "codegen_capstone_pilot", "imports": [pay, cust], "_id": "default ObjectId"},
        "mql": {"collection": coll.name, "method": "aggregate", "options": {"allowDiskUse": True}, "pipeline": PIPELINE},
        "mongo_result": res,
        "validation": {"per_gold": v, "matched_any_gold": ok},
        "assumptions": [
            "Months are keyed by the first 7 characters of payment_date (YYYY-MM).",
            "'Average monthly change' is the mean absolute difference between consecutive monthly totals.",
        ],
    }
    return ("passed" if ok else "failed"), matched_reason(v, ok), artifact
