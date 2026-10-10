"""local058 - hardware segments: unique sales products 2020, ordered by 2020->2021 growth.

No shipped gold SQL: a labelled reference SQL is authored and verified against the gold
CSV first. Unique products = DISTINCT product_code appearing in the big fact table
hardware_fact_sales_monthly per fiscal year; counted for fiscal 2020 per segment and
ordered by the percentage increase 2020 -> 2021. Reproduces gold variant `_a`.
"""
import pandas as pd

from conversion_kit import import_table, manifest_record, matched_reason, score, sqlite_df

IID = "local058"

REFERENCE_SQL = """
WITH d AS (SELECT DISTINCT product_code, fiscal_year FROM hardware_fact_sales_monthly),
agg AS (
  SELECT dp.segment,
         COUNT(DISTINCT CASE WHEN d.fiscal_year = 2020 THEN d.product_code END) AS c2020,
         COUNT(DISTINCT CASE WHEN d.fiscal_year = 2021 THEN d.product_code END) AS c2021
  FROM d JOIN hardware_dim_product dp ON dp.product_code = d.product_code
  GROUP BY dp.segment)
SELECT segment, c2020 AS unique_product_count_2020
FROM agg
ORDER BY (c2021 - c2020) * 1.0 / c2020 DESC
"""

PIPELINE = [
    {"$group": {"_id": {"pc": "$product_code", "fy": "$fiscal_year"}}},
    {"$lookup": {"from": "__DIM__", "localField": "_id.pc", "foreignField": "product_code", "as": "d"}},
    {"$unwind": "$d"},
    {"$group": {"_id": "$d.segment",
                "c2020": {"$sum": {"$cond": [{"$eq": ["$_id.fy", 2020]}, 1, 0]}},
                "c2021": {"$sum": {"$cond": [{"$eq": ["$_id.fy", 2021]}, 1, 0]}}}},
    {"$addFields": {"pct": {"$multiply": [{"$divide": [{"$subtract": ["$c2021", "$c2020"]}, "$c2020"]}, 100]}}},
    {"$sort": {"pct": -1, "_id": 1}},
    {"$project": {"_id": 0, "segment": "$_id", "unique_product_count_2020": "$c2020"}},
]

COLS = ["segment", "unique_product_count_2020"]


def run(reg):
    rec = manifest_record(IID)
    ref = sqlite_df(rec, REFERENCE_SQL)
    ref_v, ref_ok = score(ref, rec)

    dim = import_table(reg, rec, IID, "hardware_dim_product", ["product_code", "segment"], index=["product_code"])
    sales = import_table(reg, rec, IID, "hardware_fact_sales_monthly", ["product_code", "fiscal_year"])

    coll = reg.db[sales["collection"]]
    pipeline = [{**s, "$lookup": {**s["$lookup"], "from": dim["collection"]}} if "$lookup" in s else s for s in PIPELINE]
    res = list(coll.aggregate(pipeline, allowDiskUse=True))
    pred = pd.DataFrame(res, columns=COLS)
    v, ok = score(pred, rec)

    artifact = {
        "pattern": "distinct-fact-key group + dim lookup + per-segment year-over-year count growth",
        "gold_sql": None, "gold_sql_reproduces_gold_csv": "n/a (no shipped gold SQL)",
        "reference_sql": REFERENCE_SQL, "reference_sql_matches_gold": ref_ok,
        "relational_semantics": ("Unique products per segment = DISTINCT product_code in the sales fact table for a fiscal "
                                 "year; output the fiscal-2020 count, ordered by the 2020->2021 percentage increase."),
        "source_schema": {"hardware_fact_sales_monthly": ["product_code", "fiscal_year"],
                          "hardware_dim_product": ["product_code", "segment"]},
        "mongodb": {"database": "codegen_capstone_pilot", "imports": [dim, sales], "_id": "default ObjectId"},
        "mql": {"collection": coll.name, "method": "aggregate", "options": {"allowDiskUse": True}, "pipeline": pipeline},
        "mongo_result": res,
        "validation": {"per_gold": v, "matched_any_gold": ok},
        "assumptions": ["The fact table key is (product_code, fiscal_year); fiscal_year defines the year.",
                        "Segment comes from hardware_dim_product.",
                        "Validated against gold variant _a."],
    }
    return ("passed" if ok else "failed"), matched_reason(v, ok), artifact
