"""local198 - median country sales among countries with more than 4 customers.

No shipped gold SQL: a labelled reference SQL is authored and verified against the gold
CSV first. The median averages the two middle values when the count is even
((195.10 + 303.96) / 2 = 249.53).
"""
import pandas as pd

from conversion_kit import (import_table, manifest_record, matched_reason, pattern_record, score, sqlite_df)

IID = "local198"

REFERENCE_SQL = """
WITH ct AS (
    SELECT c.Country, SUM(i.Total) AS tot
    FROM customers c JOIN invoices i ON c.CustomerId = i.CustomerId
    GROUP BY c.Country
),
big AS (SELECT c.Country FROM customers c GROUP BY c.Country HAVING COUNT(*) > 4),
v AS (SELECT ct.tot FROM ct JOIN big ON ct.Country = big.Country)
SELECT AVG(tot) AS Median_total_sales
FROM (SELECT tot, ROW_NUMBER() OVER (ORDER BY tot) AS rn, COUNT(*) OVER () AS n FROM v)
WHERE rn IN ((n + 1) / 2, (n + 2) / 2)
"""


def run(reg):
    rec = manifest_record(IID)
    ref = sqlite_df(rec, REFERENCE_SQL)
    ref_v, ref_ok = score(ref, rec)

    m_cus = import_table(reg, rec, IID, "customers", ["CustomerId", "Country"])
    m_inv = import_table(reg, rec, IID, "invoices", ["CustomerId", "Total"], index=["CustomerId"])
    coll = reg.db[m_cus["collection"]]

    n_ = {"$size": "$tots"}
    lo = {"$floor": {"$divide": [{"$subtract": ["$$n", 1]}, 2]}}
    hi = {"$ceil": {"$divide": [{"$subtract": ["$$n", 1]}, 2]}}
    PIPELINE = [
        {"$lookup": {"from": m_inv["collection"], "localField": "CustomerId",
                     "foreignField": "CustomerId", "as": "inv"}},
        {"$unwind": "$inv"},
        {"$group": {"_id": "$Country", "tot": {"$sum": "$inv.Total"},
                    "customers": {"$addToSet": "$CustomerId"}}},
        {"$addFields": {"nc": {"$size": "$customers"}}},
        {"$match": {"nc": {"$gt": 4}}},
        {"$sort": {"tot": 1}},
        {"$group": {"_id": None, "tots": {"$push": "$tot"}}},
        {"$project": {"_id": 0, "Median_total_sales": {"$let": {
            "vars": {"a": "$tots", "n": n_},
            "in": {"$avg": [{"$arrayElemAt": ["$$a", lo]}, {"$arrayElemAt": ["$$a", hi]}]}}}}},
    ]
    res = list(coll.aggregate(PIPELINE, allowDiskUse=True))
    pred = pd.DataFrame(res, columns=["Median_total_sales"])
    v, ok = score(pred, rec)

    artifact = {
        "pattern": "per-key total + membership filter + even/odd median via sorted array",
        "gold_sql": None, "gold_sql_reproduces_gold_csv": "n/a (no shipped gold SQL)",
        "reference_sql": REFERENCE_SQL, "reference_sql_matches_gold": ref_ok,
        "relational_semantics": ("Per country total invoice sales; retain countries with > 4 customers; median across those totals, "
                                 "averaging the two middle values for an even count."),
        "source_schema": {"customers": ["CustomerId", "Country"], "invoices": ["CustomerId", "Total"]},
        "mongodb": {"database": "codegen_capstone_pilot", "imports": [m_cus, m_inv], "_id": "default ObjectId",
                    "indexes": ["invoices.CustomerId"]},
        "mql": {"collection": coll.name, "method": "aggregate", "options": {"allowDiskUse": True}, "pipeline": PIPELINE},
        "mongo_result": res,
        "validation": {"per_gold": v, "matched_any_gold": ok},
        "assumptions": [
            "The country customer count uses customers (not invoices) and the > 4 filter is on that count.",
            "Even-sized sample uses the average of the two middle values; floor/ceil of (n-1)/2 coincide for odd n.",
        ],
    }
    return ("passed" if ok else "failed"), matched_reason(v, ok), artifact
