"""local075 - multi-table analytical report: conditional aggregation + EXISTS / NOT EXISTS on the same table."""
import sys; sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import pandas as pd
from harness import *

IID = "local075"
rec = manifest_record(IID)
gold_sql = (ROOT / rec["gold_sql_path"]).read_text(encoding="utf-8")
ref = sqlite_df(rec, gold_sql); ref_v, ref_ok = score(ref, rec)
sq = sqlite_ro(rec)
pre = {
  "gold_sql_rows": len(ref),
  "product_pages": sq.execute("SELECT COUNT(*) FROM shopping_cart_page_hierarchy WHERE product_id IS NOT NULL").fetchone()[0],
  "event_types": sq.execute("SELECT event_type, COUNT(*) FROM shopping_cart_events GROUP BY 1").fetchall(),
  "pages_with_null_product_id": [r[0] for r in sq.execute("SELECT page_id FROM shopping_cart_page_hierarchy WHERE product_id IS NULL")],
}
assert pre["gold_sql_rows"] == pre["product_pages"], "the CTE inner joins in the gold SQL would drop pages; plan must replicate"

db = mongo_db()
imports = [
  import_table(db, rec, IID, "shopping_cart_events", ["visit_id", "page_id", "event_type"]),
  import_table(db, rec, IID, "shopping_cart_page_hierarchy", ["page_id", "page_name", "product_id"]),
]
C = lambda t: f"{IID}__{t}"
ev = lambda t: {"$eq": ["$event_type", t]}
PIPELINE = [
  # EXISTS / NOT EXISTS (visit has an event_type=3 row) -> window max over the whole visit partition
  {"$setWindowFields": {"partitionBy": "$visit_id",
      "output": {"visit_purchased": {"$max": {"$cond": [ev(3), 1, 0]}, "window": {"documents": ["unbounded", "unbounded"]}}}}},
  {"$lookup": {"from": C("shopping_cart_page_hierarchy"), "localField": "page_id", "foreignField": "page_id", "as": "h"}},
  {"$unwind": "$h"},                                                    # INNER JOIN
  {"$match": {"h.product_id": {"$ne": None}, "page_id": {"$nin": [1, 2, 12, 13]}}},   # IS NOT NULL; NOT IN (...)
  {"$group": {"_id": {"page_id": "$page_id", "page_name": "$h.page_name"},
              "number_of_product_being_viewed": {"$sum": {"$cond": [ev(1), 1, 0]}},
              "number_added_to_the_cart": {"$sum": {"$cond": [ev(2), 1, 0]}},
              "without_being_purchased_in_cart": {"$sum": {"$cond": [{"$and": [ev(2), {"$eq": ["$visit_purchased", 0]}]}, 1, 0]}},
              "count_of_actual_purchases": {"$sum": {"$cond": [{"$and": [ev(2), {"$eq": ["$visit_purchased", 1]}]}, 1, 0]}}}},
  {"$sort": {"_id.page_id": 1}},
  {"$project": {"_id": 0, "page_id": "$_id.page_id", "page_name": "$_id.page_name", "number_of_product_being_viewed": 1,
                "number_added_to_the_cart": 1, "without_being_purchased_in_cart": 1, "count_of_actual_purchases": 1}},
]
res = list(db[C("shopping_cart_events")].aggregate(PIPELINE))
v, ok = score(pd.DataFrame(res), rec)
print(IID, "ref_sql_ok", ref_ok, "| mql_ok", ok, len(res), "rows", v)

write_artifact(IID, {
  "instance_id": IID, "db": rec["db"], "question": rec["question"],
  "pattern": "multi-table analytical report: CASE-sum pivots + EXISTS/NOT EXISTS subquery + join of three CTEs",
  "gold_sql": gold_sql, "gold_sql_reproduces_gold_csv": ref_ok, "prechecks": pre,
  "relational_semantics": "For product pages only (page_id not in 1,2,12,13): views (event 1), cart adds (event 2), adds in visits with a purchase event (3) = purchases, adds in visits without = abandoned.",
  "mongodb": {"database": PILOT_DB, "imports": imports, "indexes": [], "_id": "default ObjectId",
              "join_strategy": "page_hierarchy has 13 rows; looked up from events (32,734) by page_id"},
  "mql": {"collection": C("shopping_cart_events"), "method": "aggregate", "pipeline": PIPELINE},
  "mongo_result": res, "gold": v, "validation_passed": ok,
  "rules_discovered": [
    "EXISTS / NOT EXISTS correlated on the same table -> compute a per-key flag with $setWindowFields ($max over an unbounded partition); no self-$lookup needed.",
    "SUM(CASE WHEN cond THEN 1 ELSE 0 END) -> $sum + $cond (pivot-style counters in one $group).",
    "IS NOT NULL -> {$ne: null} (matches explicit null and missing); NOT IN -> $nin.",
    "Several CTEs over the same fact rows that are INNER JOINed back together equal a single $group only if every key appears in every CTE; pre-check (gold SQL row count == number of product pages) proves it for this data.",
    "Gold columns are matched by value vectors, so output column names/order are free; condition_cols select which gold columns must be matched (here views/adds/purchases/abandoned, not ids).",
  ]})
