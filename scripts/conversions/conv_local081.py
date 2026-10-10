"""local081 - 1998 customer spending by threshold group (count + percentage).

No shipped gold SQL: a labelled reference SQL is authored and verified against the gold
CSV first. Spend = sum(unitprice*quantity) excluding discounts, over orders placed in 1998;
ranges are inclusive on both ends.
"""
import pandas as pd

from conversion_kit import (import_table, manifest_record, matched_reason, pattern_record, score, sqlite_df)

IID = "local081"

REFERENCE_SQL = """
WITH cust AS (
    SELECT o.customerid AS cid, SUM(od.unitprice * od.quantity) AS spent
    FROM orders o JOIN order_details od ON o.orderid = od.orderid
    WHERE o.orderdate LIKE '1998%'
    GROUP BY o.customerid
)
SELECT g.groupname,
       COUNT(c.cid) AS CUSTOMER_COUNT,
       ROUND(100.0 * COUNT(c.cid) / (SELECT COUNT(*) FROM cust), 2) AS PERCENTAGE
FROM customergroupthreshold g
LEFT JOIN cust c ON c.spent >= g.rangebottom AND c.spent <= g.rangetop
GROUP BY g.groupname
"""


def run(reg):
    rec = manifest_record(IID)
    ref = sqlite_df(rec, REFERENCE_SQL)
    ref_v, ref_ok = score(ref, rec)

    m_ord = import_table(reg, rec, IID, "orders", ["orderid", "customerid", "orderdate"])
    m_det = import_table(reg, rec, IID, "order_details", ["orderid", "unitprice", "quantity"], index=["orderid"])
    m_thr = import_table(reg, rec, IID, "customergroupthreshold", ["groupname", "rangebottom", "rangetop"])
    coll = reg.db[m_ord["collection"]]

    PIPELINE = [
        {"$match": {"orderdate": {"$regex": "^1998"}}},
        {"$lookup": {"from": m_det["collection"], "localField": "orderid",
                     "foreignField": "orderid", "as": "od"}},
        {"$unwind": "$od"},
        {"$group": {"_id": "$customerid", "spent": {"$sum": {"$multiply": ["$od.unitprice", "$od.quantity"]}}}},
        {"$lookup": {"from": m_thr["collection"], "let": {"s": "$spent"},
                     "pipeline": [{"$match": {"$expr": {"$and": [
                         {"$gte": ["$$s", "$rangebottom"]}, {"$lte": ["$$s", "$rangetop"]}]}}}],
                     "as": "g"}},
        {"$unwind": "$g"},
        {"$group": {"_id": "$g.groupname", "CUSTOMER_COUNT": {"$sum": 1}}},
        {"$setWindowFields": {"output": {"total": {"$sum": "$CUSTOMER_COUNT"}}}},
        {"$addFields": {"PERCENTAGE": {"$round": [{"$multiply": [100, {"$divide": ["$CUSTOMER_COUNT", "$total"]}]}, 2]}}},
        {"$project": {"_id": 0, "groupname": "$_id", "CUSTOMER_COUNT": 1, "PERCENTAGE": 1}},
    ]
    res = list(coll.aggregate(PIPELINE))
    pred = pd.DataFrame(res, columns=["groupname", "CUSTOMER_COUNT", "PERCENTAGE"])
    v, ok = score(pred, rec)

    artifact = {
        "pattern": "filtered join + group + bucketed range join + windowed percentage",
        "gold_sql": None, "gold_sql_reproduces_gold_csv": "n/a (no shipped gold SQL)",
        "reference_sql": REFERENCE_SQL, "reference_sql_matches_gold": ref_ok,
        "relational_semantics": ("1998 customers' spend (no discounts); each customer bucketed by inclusive [rangebottom, rangetop]; "
                                 "count and share of the 1998 customer population per group."),
        "source_schema": {"orders": ["orderid", "customerid", "orderdate TEXT"], "order_details": ["orderid", "unitprice", "quantity"],
                          "customergroupthreshold": ["groupname", "rangebottom", "rangetop"]},
        "mongodb": {"database": "codegen_capstone_pilot", "imports": [m_ord, m_det, m_thr], "_id": "default ObjectId",
                    "indexes": ["order_details.orderid"]},
        "mql": {"collection": coll.name, "method": "aggregate", "pipeline": PIPELINE},
        "mongo_result": res,
        "validation": {"per_gold": v, "matched_any_gold": ok},
        "assumptions": [
            "orderdate is filtered as text matching '1998%' (SQLite DATE stored as 'YYYY-MM-DD').",
            "Threshold ranges are inclusive at both ends; PERCENTAGE uses 2-decimal rounding and each group's share of all 1998 buyers.",
        ],
    }
    return ("passed" if ok else "failed"), matched_reason(v, ok), artifact
