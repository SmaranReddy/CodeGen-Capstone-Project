"""local062 - Italian customers bucketed into 10 equal profit intervals (Dec 2021).

No shipped gold SQL: a labelled reference SQL is authored and verified against the gold
CSV first. The shipped `profits` view (costs x sales on prod/time/channel/promo) is
replicated in Mongo. Sales are pre-filtered to Italy + 2021-12 and joined to `costs` via a
composite key to stay memory-safe; per-customer profit = SUM(quantity_sold*(unit_price-
unit_cost)); the global [min,max] range is split into 10 equal intervals and each bucket
reports its customer count and min/max profit. Reproduces gold variant `_a`.
"""
import pandas as pd

from conversion_kit import import_table, manifest_record, matched_reason, score, sqlite_df

IID = "local062"

REFERENCE_SQL = """
WITH pc AS (
  SELECT p.cust_id, SUM(p.quantity_sold*(p.unit_price-p.unit_cost)) AS profit
  FROM profits p
  JOIN times t ON t.time_id = p.time_id
  JOIN customers c ON c.cust_id = p.cust_id
  JOIN countries co ON co.country_id = c.country_id
  WHERE co.country_name = 'Italy' AND t.calendar_month_desc = '2021-12'
  GROUP BY p.cust_id),
rng AS (SELECT MIN(profit) mn, MAX(profit) mx FROM pc),
b AS (
  SELECT c.cust_id, c.profit,
         CASE WHEN r.mx = r.mn THEN 1
              ELSE MIN(10, CAST((c.profit - r.mn)/((r.mx - r.mn)/10) AS INTEGER) + 1) END AS bucket
  FROM pc c, rng r)
SELECT bucket AS BUCKET_NUMBER, COUNT(*) AS CUSTOMER_COUNT,
       MIN(profit) AS MIN_PROFIT_IN_BUCKET, MAX(profit) AS MAX_PROFIT_IN_BUCKET
FROM b GROUP BY bucket ORDER BY bucket
"""

SALES_WHERE = ("time_id IN (SELECT time_id FROM times WHERE calendar_month_desc = '2021-12') "
               "AND cust_id IN (SELECT c.cust_id FROM customers c "
               "JOIN countries co ON co.country_id = c.country_id WHERE co.country_name = 'Italy')")

KEY_EXPR = {"$concat": [{"$toString": "$prod_id"}, "|", {"$toString": "$time_id"}, "|",
                        {"$toString": "$channel_id"}, "|", {"$toString": "$promo_id"}]}

PIPELINE = [
    {"$addFields": {"key": KEY_EXPR}},
    {"$lookup": {"from": "__COSTS__", "localField": "key", "foreignField": "key", "as": "c"}},
    {"$unwind": "$c"},
    {"$addFields": {"profit": {"$multiply": ["$quantity_sold", {"$subtract": ["$c.unit_price", "$c.unit_cost"]}]}}},
    {"$group": {"_id": "$cust_id", "profit": {"$sum": "$profit"}}},
    {"$setWindowFields": {"output": {"mn": {"$min": "$profit"}, "mx": {"$max": "$profit"}}}},
    {"$addFields": {"bucket": {"$cond": [
        {"$eq": ["$mx", "$mn"]}, 1,
        {"$min": [10, {"$add": [{"$floor": {"$divide": [
            {"$subtract": ["$profit", "$mn"]},
            {"$divide": [{"$subtract": ["$mx", "$mn"]}, 10]}]}}, 1]}]}]}}},
    {"$group": {"_id": "$bucket", "CUSTOMER_COUNT": {"$sum": 1},
                "MIN_PROFIT_IN_BUCKET": {"$min": "$profit"}, "MAX_PROFIT_IN_BUCKET": {"$max": "$profit"}}},
    {"$project": {"_id": 0, "BUCKET_NUMBER": "$_id", "CUSTOMER_COUNT": 1,
                  "MIN_PROFIT_IN_BUCKET": 1, "MAX_PROFIT_IN_BUCKET": 1}},
    {"$sort": {"BUCKET_NUMBER": 1}},
]

COLS = ["BUCKET_NUMBER", "CUSTOMER_COUNT", "MIN_PROFIT_IN_BUCKET", "MAX_PROFIT_IN_BUCKET"]


def run(reg):
    rec = manifest_record(IID)
    ref = sqlite_df(rec, REFERENCE_SQL)
    ref_v, ref_ok = score(ref, rec)

    sales = import_table(reg, rec, IID, "sales",
                         ["prod_id", "cust_id", "time_id", "channel_id", "promo_id", "quantity_sold"],
                         where=SALES_WHERE)
    costs = import_table(reg, rec, IID, "costs",
                         ["prod_id", "time_id", "promo_id", "channel_id", "unit_cost", "unit_price"])

    keyed = f"{IID}__costs_keyed"
    reg.dropped_first(keyed)
    reg.db[costs["collection"]].aggregate([
        {"$project": {"_id": 0, "key": KEY_EXPR, "unit_cost": 1, "unit_price": 1}},
        {"$out": keyed},
    ])
    reg.db[keyed].create_index("key")
    reg.note(keyed)

    coll = reg.db[sales["collection"]]
    pipeline = [{**s, "$lookup": {**s["$lookup"], "from": keyed}} if "$lookup" in s else s for s in PIPELINE]
    res = list(coll.aggregate(pipeline, allowDiskUse=True))
    pred = pd.DataFrame(res, columns=COLS)
    v, ok = score(pred, rec)

    artifact = {
        "pattern": "view replication (costs x sales) + pre-filtered import + composite-key $lookup + window min/max + equal-width bucketing",
        "gold_sql": None, "gold_sql_reproduces_gold_csv": "n/a (no shipped gold SQL)",
        "reference_sql": REFERENCE_SQL, "reference_sql_matches_gold": ref_ok,
        "relational_semantics": ("For Italian customers in 2021-12, profit per customer = SUM(quantity_sold*(unit_price-unit_cost)); "
                                 "the [min,max] profit range is split into 10 equal-width buckets; each bucket returns the "
                                 "customer count and its min/max profit."),
        "source_schema": {"profits (view)": ["cust_id", "time_id", "quantity_sold", "unit_cost", "unit_price"],
                          "times": ["time_id", "calendar_month_desc"],
                          "customers": ["cust_id", "country_id"], "countries": ["country_id", "country_name"]},
        "mongodb": {"database": "codegen_capstone_pilot", "imports": [sales, costs],
                    "derived": [{"collection": keyed, "method": "$out of composite prod|time|channel|promo key"}]},
        "mql": {"collection": coll.name, "method": "aggregate", "options": {"allowDiskUse": True}, "pipeline": pipeline},
        "mongo_result": res,
        "validation": {"per_gold": v, "matched_any_gold": ok},
        "assumptions": ["The `profits` view is replicated as costs JOIN sales on (prod_id, time_id, channel_id, promo_id).",
                        "Italy = countries.country_name='Italy'; the month is times.calendar_month_desc='2021-12'.",
                        "Bucket number 1..10; value equal to max falls in the last bucket.",
                        "Validated against gold variant _a (cols [1,2,3])."],
    }
    return ("passed" if ok else "failed"), matched_reason(v, ok), artifact
