"""local298 - total prior-month user balance (negative balances floored at zero) per month.

No shipped gold SQL: a labelled reference SQL is authored and verified against the gold
CSV first. Balance as of the 1st of a month = cumulative signed transactions strictly
before that date; each customer's balance is floored at 0 before summing.
"""
import pandas as pd

from conversion_kit import import_table, manifest_record, matched_reason, pattern_record, score, sqlite_df

IID = "local298"

REFERENCE_SQL = """
WITH signed AS (SELECT customer_id, txn_date, CASE WHEN txn_type='deposit' THEN txn_amount ELSE -txn_amount END AS amt FROM customer_transactions),
cust AS (SELECT DISTINCT customer_id FROM customer_transactions),
months(m) AS (VALUES('2020-02-01'),('2020-03-01'),('2020-04-01'))
SELECT m AS month, SUM(MAX(0, bal)) AS total_balance_prev_month FROM (
  SELECT mo.m AS m, cu.customer_id,
         COALESCE((SELECT SUM(s.amt) FROM signed s WHERE s.customer_id = cu.customer_id AND s.txn_date < mo.m), 0) AS bal
  FROM months mo CROSS JOIN cust cu) t
GROUP BY m ORDER BY m
"""

MONTHS = ["2020-02-01", "2020-03-01", "2020-04-01"]


def _bal_expr(idx):
    return {"$sum": {"$map": {
        "input": "$txns", "as": "t",
        "in": {"$cond": [{"$lt": ["$$t.d", MONTHS[idx]]}, "$$t.a", 0]}}}}


PIPELINE = [
    {"$set": {"amt": {"$cond": [{"$eq": ["$txn_type", "deposit"]}, "$txn_amount",
                                 {"$multiply": [-1, "$txn_amount"]}]}}},
    {"$group": {"_id": "$customer_id", "txns": {"$push": {"d": "$txn_date", "a": "$amt"}}}},
    {"$project": {"b0": {"$max": [0, _bal_expr(0)]},
                  "b1": {"$max": [0, _bal_expr(1)]},
                  "b2": {"$max": [0, _bal_expr(2)]}}},
    {"$group": {"_id": None, "s0": {"$sum": "$b0"}, "s1": {"$sum": "$b1"}, "s2": {"$sum": "$b2"}}},
    {"$project": {"_id": 0, "rows": [
        {"month": MONTHS[0], "total_balance_prev_month": "$s0"},
        {"month": MONTHS[1], "total_balance_prev_month": "$s1"},
        {"month": MONTHS[2], "total_balance_prev_month": "$s2"}]}},
    {"$unwind": "$rows"},
    {"$replaceWith": "$rows"},
    {"$sort": {"month": 1}},
]


def run(reg):
    rec = manifest_record(IID)
    ref = sqlite_df(rec, REFERENCE_SQL)
    ref_v, ref_ok = score(ref, rec)

    m = import_table(reg, rec, IID, "customer_transactions", ["customer_id", "txn_date", "txn_type", "txn_amount"])
    coll = reg.db[m["collection"]]

    res = list(coll.aggregate(PIPELINE, allowDiskUse=True))
    pred = pd.DataFrame(res, columns=["month", "total_balance_prev_month"])
    v, ok = score(pred, rec)

    artifact = {
        "pattern": "per-customer signed ledger + cumulative pre-month balance floored at 0 + monthly totals",
        "gold_sql": None, "gold_sql_reproduces_gold_csv": "n/a (no shipped gold SQL)",
        "reference_sql": REFERENCE_SQL, "reference_sql_matches_gold": ref_ok,
        "relational_semantics": ("deposit is positive, withdrawal/purchase negative; for each month sum each customer's "
                                 "cumulative balance strictly before the 1st (floor at 0) and total it."),
        "source_schema": {"customer_transactions": ["customer_id", "txn_date", "txn_type", "txn_amount"]},
        "mongodb": {"database": "codegen_capstone_pilot", "imports": [m], "_id": "default ObjectId"},
        "mql": {"collection": coll.name, "method": "aggregate", "options": {"allowDiskUse": True}, "pipeline": PIPELINE},
        "mongo_result": res,
        "validation": {"per_gold": v, "matched_any_gold": ok},
        "assumptions": [
            "Signed amount = +txn_amount for deposits, -txn_amount otherwise.",
            "A customer's balance as of a month start uses transactions strictly before that date and is floored at 0.",
            "Months are Feb/Mar/Apr 2020; January 2020 is only the baseline.",
        ],
    }
    return ("passed" if ok else "failed"), matched_reason(v, ok), artifact
