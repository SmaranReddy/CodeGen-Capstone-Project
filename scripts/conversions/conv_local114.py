"""local114 - web sales report per region with top sales representative(s).

No shipped gold SQL: a labelled reference SQL is authored and verified against the gold
CSV first. Orders join account -> sales rep -> region; region totals (order count, sales)
and per-rep sales are computed; the rep(s) tied at the region's maximum sales are kept.
Reproduces the split-column gold variant `_d`.
"""
import pandas as pd

from conversion_kit import import_table, manifest_record, matched_reason, score, sqlite_df

IID = "local114"

REFERENCE_SQL = """
WITH region_tot AS (
  SELECT r.id AS region_id, r.name AS region_name, COUNT(o.id) AS num_orders, SUM(o.total_amt_usd) AS total_sales
  FROM web_orders o
  JOIN web_accounts a ON a.id = o.account_id
  JOIN web_sales_reps s ON s.id = a.sales_rep_id
  JOIN web_region r ON r.id = s.region_id
  GROUP BY r.id, r.name),
rep_tot AS (
  SELECT r.id AS region_id, s.name AS rep_name, SUM(o.total_amt_usd) AS rep_sales
  FROM web_orders o
  JOIN web_accounts a ON a.id = o.account_id
  JOIN web_sales_reps s ON s.id = a.sales_rep_id
  JOIN web_region r ON r.id = s.region_id
  GROUP BY r.id, s.id, s.name),
top AS (SELECT region_id, MAX(rep_sales) AS mx FROM rep_tot GROUP BY region_id)
SELECT rt.region_name AS REGION_NAME, rt.num_orders AS TOTAL_REGION_ORDERS,
       rt.total_sales AS TOTAL_REGION_SALES, rp.rep_name AS TOP_SALES_REP_NAME,
       rp.rep_sales AS TOP_SALES_REP_SALES
FROM region_tot rt
JOIN top t ON t.region_id = rt.region_id
JOIN rep_tot rp ON rp.region_id = rt.region_id AND rp.rep_sales = t.mx
ORDER BY rt.total_sales DESC
"""

PIPELINE = [
    {"$lookup": {"from": "__ACC__", "localField": "account_id", "foreignField": "id", "as": "a"}},
    {"$unwind": "$a"},
    {"$lookup": {"from": "__REPS__", "localField": "a.sales_rep_id", "foreignField": "id", "as": "s"}},
    {"$unwind": "$s"},
    {"$lookup": {"from": "__REG__", "localField": "s.region_id", "foreignField": "id", "as": "r"}},
    {"$unwind": "$r"},
    {"$group": {"_id": {"region_id": "$r.id", "region_name": "$r.name", "rep": "$s.name"},
                "rep_sales": {"$sum": "$total_amt_usd"}, "orders": {"$sum": 1}}},
    {"$setWindowFields": {"partitionBy": "$_id.region_id",
                          "output": {"mx": {"$max": "$rep_sales"},
                                     "total_orders": {"$sum": "$orders"},
                                     "total_sales": {"$sum": "$rep_sales"}}}},
    {"$match": {"$expr": {"$eq": ["$rep_sales", "$mx"]}}},
    {"$project": {"_id": 0, "REGION_NAME": "$_id.region_name", "TOTAL_REGION_ORDERS": "$total_orders",
                  "TOTAL_REGION_SALES": "$total_sales", "TOP_SALES_REP_NAME": "$_id.rep",
                  "TOP_SALES_REP_SALES": "$rep_sales"}},
    {"$sort": {"TOTAL_REGION_SALES": -1}},
]

COLS = ["REGION_NAME", "TOTAL_REGION_ORDERS", "TOTAL_REGION_SALES", "TOP_SALES_REP_NAME", "TOP_SALES_REP_SALES"]


def run(reg):
    rec = manifest_record(IID)
    ref = sqlite_df(rec, REFERENCE_SQL)
    ref_v, ref_ok = score(ref, rec)

    orders = import_table(reg, rec, IID, "web_orders", ["id", "account_id", "total_amt_usd"], index=["account_id"])
    acc = import_table(reg, rec, IID, "web_accounts", ["id", "sales_rep_id"], index=["id"])
    reps = import_table(reg, rec, IID, "web_sales_reps", ["id", "name", "region_id"], index=["id"])
    regn = import_table(reg, rec, IID, "web_region", ["id", "name"], index=["id"])

    mapping = {"__ACC__": acc["collection"], "__REPS__": reps["collection"], "__REG__": regn["collection"]}
    coll = reg.db[orders["collection"]]
    pipeline = [{**s, "$lookup": {**s["$lookup"], "from": mapping[s["$lookup"]["from"]]}} if "$lookup" in s else s
                for s in PIPELINE]
    res = list(coll.aggregate(pipeline, allowDiskUse=True))
    pred = pd.DataFrame(res, columns=COLS)
    v, ok = score(pred, rec)

    artifact = {
        "pattern": "multi-hop $lookup (order->account->rep->region) + per-rep group + window max/totals + tie filter",
        "gold_sql": None, "gold_sql_reproduces_gold_csv": "n/a (no shipped gold SQL)",
        "reference_sql": REFERENCE_SQL, "reference_sql_matches_gold": ref_ok,
        "relational_semantics": ("Per region: order count and total sales (SUM total_amt_usd) across the region's reps; "
                                 "the representative(s) with the region's maximum sales are returned (ties included)."),
        "source_schema": {"web_orders": ["id", "account_id", "total_amt_usd"],
                          "web_accounts": ["id", "sales_rep_id"],
                          "web_sales_reps": ["id", "name", "region_id"],
                          "web_region": ["id", "name"]},
        "mongodb": {"database": "codegen_capstone_pilot", "imports": [orders, acc, reps, regn], "_id": "default ObjectId"},
        "mql": {"collection": coll.name, "method": "aggregate", "options": {"allowDiskUse": True}, "pipeline": pipeline},
        "mongo_result": res,
        "validation": {"per_gold": v, "matched_any_gold": ok},
        "assumptions": ["An order's salesperson is resolved via account.sales_rep_id, and the region via the rep.",
                        "Ties for top sales rep are all returned.",
                        "Validated against the split-column gold variant _d."],
    }
    return ("passed" if ok else "failed"), matched_reason(v, ok), artifact
