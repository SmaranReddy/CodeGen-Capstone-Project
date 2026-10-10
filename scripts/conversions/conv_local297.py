"""local297 -> MongoDB/MQL.

Question: monthly net cash flow per customer (deposits +, withdrawals -), cumulative
closing balance, growth rate of the most recent month vs the previous one, and the
percentage of customers whose most recent month grew by more than 5%.

IMPORTANT (ambiguous gold): this single question has TWO official accepted answers.
  * variant A = 44.869215  -> group ALL transactions (purchases contribute 0).
  * variant B = 53.002070  -> restrict to deposits/withdrawals BEFORE grouping.
The prompt says "group all deposits and withdrawals", which reads as variant B; we
implement B. Both variants reproduce and both score against the eval harness.
"""
import pandas as pd

from conversion_kit import import_table, manifest_record, score, sqlite_df

IID = "local297"

REFERENCE_SQL_B = """
WITH mn AS (
    SELECT customer_id, substr(txn_date, 1, 7) || '-01' AS month,
           SUM(CASE WHEN txn_type = 'deposit' THEN txn_amount ELSE -txn_amount END) AS net
    FROM customer_transactions
    WHERE txn_type IN ('deposit', 'withdrawal')
    GROUP BY customer_id, month
),
cl AS (
    SELECT customer_id, month, SUM(net) OVER (PARTITION BY customer_id ORDER BY month) AS closing
    FROM mn
),
pr AS (
    SELECT customer_id, month, closing,
           LAG(closing) OVER (PARTITION BY customer_id ORDER BY month) AS prev
    FROM cl
),
rn AS (
    SELECT customer_id, closing, prev,
           ROW_NUMBER() OVER (PARTITION BY customer_id ORDER BY month DESC) AS rn
    FROM pr
)
SELECT 100.0 * SUM(CASE WHEN prev IS NULL THEN 0
                        WHEN prev = 0 THEN CASE WHEN closing * 100 > 5 THEN 1 ELSE 0 END
                        ELSE CASE WHEN (closing - prev) * 100.0 / prev > 5 THEN 1 ELSE 0 END END)
       / SUM(CASE WHEN prev IS NULL THEN 0 ELSE 1 END) AS PERCENTAGE_CUSTOMERS
FROM rn WHERE rn = 1;
"""


# Variant A (reproduces local297_a.csv = 44.869215): keeps ALL transactions with
# purchases contributing 0, so purchase-only months are retained (497 customers).
REFERENCE_SQL_A = """
WITH mn AS (
  SELECT customer_id,
         substr(txn_date, 1, 7) || '-01' AS month,
         SUM(CASE WHEN txn_type = 'deposit' THEN txn_amount
                  WHEN txn_type = 'withdrawal' THEN -txn_amount
                  ELSE 0 END) AS net
  FROM customer_transactions
  GROUP BY customer_id, month
),
cl AS (
  SELECT customer_id, month,
         SUM(net) OVER (PARTITION BY customer_id ORDER BY month) AS closing
  FROM mn
),
pr AS (
  SELECT customer_id, month, closing,
         LAG(closing) OVER (PARTITION BY customer_id ORDER BY month) AS prev
  FROM cl
),
rn AS (
  SELECT customer_id, closing, prev,
         ROW_NUMBER() OVER (PARTITION BY customer_id ORDER BY month DESC) AS rn
  FROM pr
)
SELECT 100.0 * SUM(CASE WHEN prev IS NULL THEN 0
                    WHEN prev = 0 THEN CASE WHEN closing * 100 > 5 THEN 1 ELSE 0 END
                    ELSE CASE WHEN (closing - prev) * 100.0 / prev > 5 THEN 1 ELSE 0 END END)
       / SUM(CASE WHEN prev IS NULL THEN 0 ELSE 1 END) AS pct_customers_growth_gt_5
FROM rn WHERE rn = 1;
"""


def run(reg):
    rec = manifest_record(IID)
    tx = import_table(reg, rec, IID, "customer_transactions",
                      ["customer_id", "txn_date", "txn_type", "txn_amount"])
    coll = reg.db[tx["collection"]]

    pipeline = [
        {"$match": {"txn_type": {"$in": ["deposit", "withdrawal"]}}},
        {"$set": {"month": {"$concat": [{"$substrCP": ["$txn_date", 0, 7]}, "-01"]}}},
        {"$group": {
            "_id": {"c": "$customer_id", "m": "$month"},
            "net": {"$sum": {"$cond": [{"$eq": ["$txn_type", "deposit"]},
                                       "$txn_amount", {"$multiply": [-1, "$txn_amount"]}]}}}},
        {"$sort": {"_id.c": 1, "_id.m": 1}},
        {"$setWindowFields": {
            "partitionBy": "$_id.c",
            "sortBy": {"_id.m": 1},
            "output": {
                "closing": {"$sum": "$net", "window": {"documents": ["unbounded", "current"]}},
                "pos": {"$documentNumber": {}}}}},
        {"$set": {"prev": {"$cond": [{"$eq": ["$pos", 1]}, None,
                                     {"$subtract": ["$closing", "$net"]}]}}},
        {"$setWindowFields": {
            "partitionBy": "$_id.c",
            "sortBy": {"_id.m": -1},
            "output": {"rn": {"$documentNumber": {}}}}},
        {"$match": {"rn": 1}},
        {"$group": {
            "_id": None,
            "num": {"$sum": {"$cond": [
                {"$eq": [{"$ifNull": ["$prev", "__NONE__"]}, "__NONE__"]}, 0, 1]}},
            "hits": {"$sum": {"$cond": [
                {"$eq": [{"$ifNull": ["$prev", "__NONE__"]}, "__NONE__"]}, 0,
                {"$cond": [
                    {"$eq": ["$prev", 0]},
                    {"$cond": [{"$gt": [{"$multiply": ["$closing", 100]}, 5]}, 1, 0]},
                    {"$cond": [{"$gt": [{"$divide": [
                        {"$multiply": [{"$subtract": ["$closing", "$prev"]}, 100]},
                        "$prev"]}, 5]}, 1, 0]}]}]}}}},
        {"$project": {"_id": 0,
                      "PERCENTAGE_CUSTOMERS": {"$multiply": [100, {"$divide": ["$hits", "$num"]}]}}},
    ]
    res = list(coll.aggregate(pipeline))
    pred = pd.DataFrame(res, columns=["PERCENTAGE_CUSTOMERS"])
    verdicts, ok = score(pred, rec)

    ref_b = sqlite_df(rec, REFERENCE_SQL_B)
    ref_a = sqlite_df(rec, REFERENCE_SQL_A)
    ref_b_ok = score(ref_b, rec)[1]
    ref_a_ok = score(ref_a, rec)[1]
    artifact = {
        "instance_id": IID,
        "reference_sql": REFERENCE_SQL_B,
        "reference_sql_value": ref_b.to_dict(orient="records"),
        "reference_sql_matches_gold": ref_b_ok,
        "alternate_interpretation": {
            "reference_sql": REFERENCE_SQL_A,
            "value": ref_a.to_dict(orient="records"),
            "matches_gold": ref_a_ok,
            "difference": "group ALL transactions (purchases contribute 0) instead of "
                          "filtering to deposit/withdrawal first",
        },
        "shipped_gold_sql": None,
        "collections": {tx["collection"]: tx["documents"]},
        "mql": {"method": "aggregate", "collection": tx["collection"], "pipeline": pipeline},
        "result": res,
        "score": verdicts,
        "notes": "two official accepted answers exist for this question; variant B implemented",
    }
    return ("passed" if ok else "failed"), ("matched gold" if ok else "no match"), artifact
