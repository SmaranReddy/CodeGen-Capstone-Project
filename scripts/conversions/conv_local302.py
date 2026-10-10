"""local302 - attribute type with the worst average sales % change (12 weeks around 2020-06-15).

No shipped gold SQL: a labelled reference SQL is authored and verified against the gold
CSV first. For each attribute value, % change = (after - before)/before over the 12-week
windows before/after 2020-06-15; each attribute type is averaged across its values.
"""
import pandas as pd

from conversion_kit import import_table, manifest_record, matched_reason, pattern_record, score, sqlite_df

IID = "local302"

REFERENCE_SQL = """
WITH u AS (
  SELECT 'region' AS att, region AS v, sales,
         CASE WHEN week_date >= '2020-03-23' AND week_date < '2020-06-15' THEN 'b'
              WHEN week_date >= '2020-06-15' AND week_date <= '2020-08-31' THEN 'a' END AS w
  FROM cleaned_weekly_sales
  UNION ALL SELECT 'platform', platform, sales,
         CASE WHEN week_date >= '2020-03-23' AND week_date < '2020-06-15' THEN 'b'
              WHEN week_date >= '2020-06-15' AND week_date <= '2020-08-31' THEN 'a' END FROM cleaned_weekly_sales
  UNION ALL SELECT 'age_band', age_band, sales,
         CASE WHEN week_date >= '2020-03-23' AND week_date < '2020-06-15' THEN 'b'
              WHEN week_date >= '2020-06-15' AND week_date <= '2020-08-31' THEN 'a' END FROM cleaned_weekly_sales
  UNION ALL SELECT 'demographic', demographic, sales,
         CASE WHEN week_date >= '2020-03-23' AND week_date < '2020-06-15' THEN 'b'
              WHEN week_date >= '2020-06-15' AND week_date <= '2020-08-31' THEN 'a' END FROM cleaned_weekly_sales
  UNION ALL SELECT 'customer_type', customer_type, sales,
         CASE WHEN week_date >= '2020-03-23' AND week_date < '2020-06-15' THEN 'b'
              WHEN week_date >= '2020-06-15' AND week_date <= '2020-08-31' THEN 'a' END FROM cleaned_weekly_sales),
s AS (SELECT att, v, SUM(CASE WHEN w = 'b' THEN sales END) AS sb, SUM(CASE WHEN w = 'a' THEN sales END) AS sa
      FROM u WHERE w IS NOT NULL GROUP BY att, v),
r AS (SELECT att, AVG(100.0 * (sa - sb) / sb) AS avg_pct FROM s GROUP BY att)
SELECT att AS ATTRIBUTE_NAME, avg_pct AS AVG_PCT_CHANGE FROM r ORDER BY avg_pct ASC LIMIT 1
"""

WINDOW_MATCH = {"$or": [
    {"$and": [{"$gte": ["$week_date", "2020-03-23"]}, {"$lt": ["$week_date", "2020-06-15"]}]},
    {"$and": [{"$gte": ["$week_date", "2020-06-15"]}, {"$lte": ["$week_date", "2020-08-31"]}]}]}

PIPELINE = [
    {"$match": {"$expr": WINDOW_MATCH}},
    {"$project": {"sales": 1, "before": {"$lt": ["$week_date", "2020-06-15"]},
                  "items": [{"att": "region", "v": "$region"}, {"att": "platform", "v": "$platform"},
                            {"att": "age_band", "v": "$age_band"}, {"att": "demographic", "v": "$demographic"},
                            {"att": "customer_type", "v": "$customer_type"}]}},
    {"$unwind": "$items"},
    {"$group": {"_id": {"att": "$items.att", "v": "$items.v"},
                "sb": {"$sum": {"$cond": ["$before", "$sales", 0]}},
                "sa": {"$sum": {"$cond": ["$before", 0, "$sales"]}}}},
    {"$addFields": {"pct": {"$multiply": [100, {"$divide": [{"$subtract": ["$sa", "$sb"]}, "$sb"]}]}}},
    {"$group": {"_id": "$_id.att", "avg_pct": {"$avg": "$pct"}}},
    {"$sort": {"avg_pct": 1}},
    {"$limit": 1},
    {"$project": {"_id": 0, "ATTRIBUTE_NAME": "$_id", "AVG_PCT_CHANGE": "$avg_pct"}},
]


def run(reg):
    rec = manifest_record(IID)
    ref = sqlite_df(rec, REFERENCE_SQL)
    ref_v, ref_ok = score(ref, rec)

    m = import_table(reg, rec, IID, "cleaned_weekly_sales",
                     ["region", "platform", "age_band", "demographic", "customer_type", "sales", "week_date"])
    coll = reg.db[m["collection"]]

    res = list(coll.aggregate(PIPELINE, allowDiskUse=True))
    pred = pd.DataFrame(res, columns=["ATTRIBUTE_NAME", "AVG_PCT_CHANGE"])
    v, ok = score(pred, rec)

    artifact = {
        "pattern": "windowed before/after sales comparison fanned out over 5 attribute types + argmin",
        "gold_sql": None, "gold_sql_reproduces_gold_csv": "n/a (no shipped gold SQL)",
        "reference_sql": REFERENCE_SQL, "reference_sql_matches_gold": ref_ok,
        "relational_semantics": ("For each attribute value compare summed sales in the 12 weeks before vs after 2020-06-15 "
                                 "as a percentage; average within each attribute type and return the most negative."),
        "source_schema": {"cleaned_weekly_sales": ["region", "platform", "age_band", "demographic", "customer_type", "sales", "week_date"]},
        "mongodb": {"database": "codegen_capstone_pilot", "imports": [m], "_id": "default ObjectId"},
        "mql": {"collection": coll.name, "method": "aggregate", "options": {"allowDiskUse": True}, "pipeline": PIPELINE},
        "mongo_result": res,
        "validation": {"per_gold": v, "matched_any_gold": ok},
        "assumptions": [
            "Before window = 2020-03-23..2020-06-08; after window = 2020-06-15..2020-08-31 (weekly Monday dates).",
            "An attribute value's % change = (after - before)/before * 100; attribute type = mean over values.",
            "Output columns match gold variant _a.",
        ],
    }
    return ("passed" if ok else "failed"), matched_reason(v, ok), artifact
