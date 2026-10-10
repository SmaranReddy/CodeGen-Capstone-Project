"""local197 - top-10 customers, monthly totals, month-over-month difference (LAG window), max, rounded."""
import sys; sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import re
import pandas as pd
from harness import *

IID = "local197"
rec = manifest_record(IID)
gold_sql = (ROOT / rec["gold_sql_path"]).read_text(encoding="utf-8")
ref = sqlite_df(rec, gold_sql); ref_v, ref_ok = score(ref, rec)
sq = sqlite_ro(rec)

ADJ = """WITH m AS (SELECT customer_id c, strftime('%Y-%m',payment_date) ym, SUM(amount) a FROM payment GROUP BY 1,2),
 top AS (SELECT customer_id c FROM payment GROUP BY 1 ORDER BY SUM(amount) DESC LIMIT 10),
 l AS (SELECT m.c, ym, a, LAG(a) OVER (PARTITION BY m.c ORDER BY ym) pa, LAG(ym) OVER (PARTITION BY m.c ORDER BY ym) pym FROM m JOIN top ON top.c=m.c)
 SELECT c, ym, ROUND(ABS(a-pa),2) d FROM l WHERE pa IS NOT NULL
   AND CAST(substr(ym,1,4) AS INT)*12+CAST(substr(ym,6,2) AS INT) = CAST(substr(pym,1,4) AS INT)*12+CAST(substr(pym,6,2) AS INT)+1
 ORDER BY d DESC LIMIT 3"""
dates = [r[0] for r in sq.execute("SELECT payment_date FROM payment")]
pre = {
  "payment_date_all_iso_prefix": all(re.match(r"^\d{4}-\d{2}-\d{2} ", d) for d in dates),
  "distinct_months": [r[0] for r in sq.execute("SELECT DISTINCT strftime('%Y-%m',payment_date) FROM payment ORDER BY 1")],
  "top_customers_rank9_to_12": sq.execute("SELECT customer_id, SUM(amount) FROM payment GROUP BY 1 ORDER BY 2 DESC LIMIT 4 OFFSET 8").fetchall(),
  "amount_storage_classes": sq.execute("SELECT typeof(amount), COUNT(*) FROM payment GROUP BY 1").fetchall(),
  "sqlite_calendar_adjacent_top3": sq.execute(ADJ).fetchall(),
}
assert pre["top_customers_rank9_to_12"][1][1] != pre["top_customers_rank9_to_12"][2][1], "tie at the top-10 boundary"

db = mongo_db()
imports = [
  import_table(db, rec, IID, "payment", ["customer_id", "amount", "payment_date"]),
  import_table(db, rec, IID, "customer", ["customer_id", "first_name", "last_name"]),
]
C = lambda t: f"{IID}__{t}"
db[C("payment")].create_index("customer_id"); db[C("customer")].create_index("customer_id")
indexes = [{"collection": C("payment"), "key": "customer_id"}, {"collection": C("customer"), "key": "customer_id"}]

IDX = {"$add": [{"$multiply": [{"$toInt": {"$substrCP": ["$ym", 0, 4]}}, 12]}, {"$toInt": {"$substrCP": ["$ym", 5, 2]}}]}
PIPELINE = [
  {"$group": {"_id": "$customer_id", "total_payments": {"$sum": "$amount"}}},
  {"$sort": {"total_payments": -1, "_id": 1}}, {"$limit": 10},                    # top10_customer
  {"$lookup": {"from": C("payment"), "localField": "_id", "foreignField": "customer_id",
               "pipeline": [{"$group": {"_id": {"$substrCP": ["$payment_date", 0, 7]}, "amount": {"$sum": "$amount"}}}],   # strftime('%Y-%m')
               "as": "months"}},
  {"$unwind": "$months"},
  {"$project": {"_id": 0, "customer_id": "$_id", "ym": "$months._id", "amount": "$months.amount"}},
  {"$addFields": {"midx": IDX}},
  {"$setWindowFields": {"partitionBy": "$customer_id", "sortBy": {"midx": 1},
                        "output": {"prev_amount": {"$shift": {"output": "$amount", "by": -1}},
                                   "prev_midx": {"$shift": {"output": "$midx", "by": -1}}}}},
  {"$match": {"$expr": {"$eq": ["$prev_midx", {"$subtract": ["$midx", 1]}]}}},   # immediately preceding CALENDAR month only
  {"$addFields": {"difference": {"$round": [{"$abs": {"$subtract": ["$amount", "$prev_amount"]}}, 2]}}},
  {"$sort": {"difference": -1, "customer_id": 1, "ym": 1}}, {"$limit": 1},
  {"$lookup": {"from": C("customer"), "localField": "customer_id", "foreignField": "customer_id", "as": "cu"}},
  {"$unwind": "$cu"},
  {"$project": {"_id": 0, "customer_id": 1, "customer_name": {"$concat": ["$cu.first_name", " ", "$cu.last_name"]}, "month": "$ym", "difference": 1}},
]
res = list(db[C("payment")].aggregate(PIPELINE))
v, ok = score(pd.DataFrame(res), rec)
print(IID, "ref_sql_ok", ref_ok, "| mql_ok", ok, res, v)

write_artifact(IID, {
  "instance_id": IID, "db": rec["db"], "question": rec["question"], "pattern": "window function (LAG) over per-customer monthly totals + top-N + max",
  "gold_sql": gold_sql, "gold_sql_reproduces_gold_csv": ref_ok,
  "ambiguity_documented_not_silently_resolved": {
     "issue": "Gold SQL uses LAG(...) OVER (PARTITION BY customer_id) with NO ORDER BY, and groups by strftime('%m') only, so its 'previous row' depends on SQLite scan order and can pair non-adjacent months (e.g. 2006-02 then 2005-05).",
     "choice": "MQL orders by month and only compares a month with the immediately preceding CALENDAR month, matching the question text.",
     "evidence_choice_is_safe_here": "calendar-adjacent SQLite reference gives the same winner as the shipped SQL (customer 148, 2005-07, 77.83; runner-up 76.85); no tie at max and none at the top-10 boundary.",
     "residual_risk": "On other data the unordered-LAG gold SQL and the adjacent-month reading could differ; the CSV is the authority."},
  "prechecks": pre,
  "relational_semantics": "Top 10 customers by SUM(amount); monthly SUM per customer; difference to the preceding month; global max difference with customer and month; round to 2 decimals.",
  "mongodb": {"database": PILOT_DB, "imports": imports, "indexes": indexes, "_id": "default ObjectId",
              "type_note": "amount is stored as REAL for 16,025 rows and INTEGER for 24; imported as stored (double / int)"},
  "mql": {"collection": C("payment"), "method": "aggregate", "pipeline": PIPELINE},
  "mongo_result": res, "gold": v, "validation_passed": ok,
  "rules_discovered": [
    "LAG/LEAD -> $setWindowFields with an explicit sortBy and $shift(by:-1/+1). An unordered SQL window has no defined meaning: always add the order key the question implies and record it.",
    "A 'month-over-month' difference must be restricted to calendar-adjacent months (month index = 12*year+month), otherwise gaps pair non-adjacent months.",
    "strftime('%Y-%m', ts) on ISO text == the first 7 characters ($substrCP 0,7) provided every value has the ISO prefix (pre-check verified); no date parsing needed.",
    "Top-N then detail: drive from the group-by-customer stage, $limit, then $lookup with a sub-pipeline back to the fact table (needs an index on the foreign field).",
    "Mixed INTEGER/REAL values in one column import as int/double and $sum handles the mix like SQLite's SUM.",
    "$round is half-to-even; here the value 77.83000000000001 is not on a .5 boundary so it is safe (see schema_notes.md section 6 for the general rule).",
  ]})
