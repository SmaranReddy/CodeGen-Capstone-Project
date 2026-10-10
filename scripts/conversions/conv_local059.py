"""local059 - 2021 overall average quantity sold of each division's top-3 products.

No shipped gold SQL: a labelled reference SQL is authored and verified against the gold
CSV first. Calendar year 2021 (the `date` column) totals sold_quantity per product, takes
each division's top 3 products by total, and averages those three totals. Reproduces gold
variant `_a`.
"""
import pandas as pd

from conversion_kit import import_table, manifest_record, matched_reason, score, sqlite_df

IID = "local059"

REFERENCE_SQL = """
WITH sums AS (
  SELECT dp.division, s.product_code, SUM(s.sold_quantity) AS q
  FROM hardware_fact_sales_monthly s
  JOIN hardware_dim_product dp ON dp.product_code = s.product_code
  WHERE substr(s.date, 1, 4) = '2021'
  GROUP BY dp.division, s.product_code),
t3 AS (SELECT division, q, ROW_NUMBER() OVER (PARTITION BY division ORDER BY q DESC) AS rn FROM sums)
SELECT division, AVG(q) AS OVERALL_AVG_QUANTITY_SOLD
FROM t3 WHERE rn <= 3 GROUP BY division
"""

PIPELINE = [
    {"$lookup": {"from": "__DIM__", "localField": "product_code", "foreignField": "product_code", "as": "d"}},
    {"$unwind": "$d"},
    {"$group": {"_id": {"division": "$d.division", "pc": "$product_code"}, "q": {"$sum": "$sold_quantity"}}},
    {"$sort": {"_id.division": 1, "q": -1}},
    {"$group": {"_id": "$_id.division", "top": {"$push": "$q"}}},
    {"$project": {"_id": 0, "division": "$_id",
                  "OVERALL_AVG_QUANTITY_SOLD": {"$avg": {"$slice": ["$top", 3]}}}},
]

COLS = ["division", "OVERALL_AVG_QUANTITY_SOLD"]


def run(reg):
    rec = manifest_record(IID)
    ref = sqlite_df(rec, REFERENCE_SQL)
    ref_v, ref_ok = score(ref, rec)

    dim = import_table(reg, rec, IID, "hardware_dim_product", ["product_code", "division"], index=["product_code"])
    sales = import_table(reg, rec, IID, "hardware_fact_sales_monthly", ["product_code", "date", "sold_quantity"],
                         where="substr(date, 1, 4) = '2021'")

    coll = reg.db[sales["collection"]]
    pipeline = [{**s, "$lookup": {**s["$lookup"], "from": dim["collection"]}} if "$lookup" in s else s for s in PIPELINE]
    res = list(coll.aggregate(pipeline, allowDiskUse=True))
    pred = pd.DataFrame(res, columns=COLS)
    v, ok = score(pred, rec)

    artifact = {
        "pattern": "per-product totals + partitioned top-3 + average of slice",
        "gold_sql": None, "gold_sql_reproduces_gold_csv": "n/a (no shipped gold SQL)",
        "reference_sql": REFERENCE_SQL, "reference_sql_matches_gold": ref_ok,
        "relational_semantics": ("For calendar 2021 (date year), sum sold_quantity per product within each division, take "
                                 "the top 3 products per division, and average their three totals."),
        "source_schema": {"hardware_fact_sales_monthly": ["product_code", "date", "sold_quantity"],
                          "hardware_dim_product": ["product_code", "division"]},
        "mongodb": {"database": "codegen_capstone_pilot", "imports": [dim, sales], "_id": "default ObjectId"},
        "mql": {"collection": coll.name, "method": "aggregate", "options": {"allowDiskUse": True}, "pipeline": pipeline},
        "mongo_result": res,
        "validation": {"per_gold": v, "matched_any_gold": ok},
        "assumptions": ["Calendar year 2021 is taken from the `date` column (not fiscal_year).",
                        "Division comes from hardware_dim_product.",
                        "Validated against gold variant _a."],
    }
    return ("passed" if ok else "failed"), matched_reason(v, ok), artifact
