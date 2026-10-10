"""local077 - monthly top interest composition, 3-month rolling average, and lag names/values.

No shipped gold SQL: a labelled reference SQL is authored and verified against the gold
CSV first. Per interest-month the metric is AVG(composition / index_value); each month's
top interest is kept; rolling window and lags use $setWindowFields.
"""
import pandas as pd

from conversion_kit import import_table, manifest_record, matched_reason, pattern_record, score, sqlite_df

IID = "local077"

MONTHS = ['09-2018', '10-2018', '11-2018', '12-2018', '01-2019', '02-2019',
          '03-2019', '04-2019', '05-2019', '06-2019', '07-2019', '08-2019']

REFERENCE_SQL = """
WITH m AS (
  SELECT im.interest_id, im.month_year, AVG(im.composition * 1.0 / im.index_value) AS val
  FROM interest_metrics im WHERE im.month_year IN ('09-2018','10-2018','11-2018','12-2018','01-2019','02-2019','03-2019','04-2019','05-2019','06-2019','07-2019','08-2019')
  GROUP BY im.interest_id, im.month_year),
ranked AS (SELECT m.*, ROW_NUMBER() OVER (PARTITION BY month_year ORDER BY val DESC, interest_id) AS rn FROM m),
champ AS (SELECT month_year, interest_id, val FROM ranked WHERE rn = 1),
champ2 AS (SELECT c.month_year, mp.interest_name, c.val,
                  substr(c.month_year,4) || '-' || substr(c.month_year,1,2) AS ym
           FROM champ c JOIN interest_map mp ON CAST(mp.id AS INTEGER) = CAST(c.interest_id AS INTEGER)),
seq AS (SELECT *, ROW_NUMBER() OVER (ORDER BY ym) AS rn FROM champ2)
SELECT s.ym AS MONTH_DATE, s.interest_name, s.val AS MAX_INDEX_COMPOSITION,
       (SELECT AVG(x.val) FROM seq x WHERE x.rn <= s.rn AND x.rn > s.rn - 3) AS ROLLING_AVG_3MONTH,
       p1.interest_name AS PREV_MONTH_INTEREST_NAME, p1.val AS PREV_MONTH_MAX_COMPOSITION,
       p2.interest_name AS PREV_2MONTH_INTEREST_NAME, p2.val AS PREV_2MONTH_MAX_COMPOSITION
FROM seq s
LEFT JOIN seq p1 ON p1.rn = s.rn - 1 LEFT JOIN seq p2 ON p2.rn = s.rn - 2
ORDER BY s.rn
"""

PIPELINE_TEMPLATE = [
    {"$match": {"month_year": {"$in": MONTHS}}},
    {"$group": {"_id": {"iid": "$interest_id", "my": "$month_year"},
                "val": {"$avg": {"$divide": ["$composition", "$index_value"]}}}},
    {"$addFields": {"mk": {"$concat": [{"$substrCP": ["$_id.my", 3, 4]}, "-", {"$substrCP": ["$_id.my", 0, 2]}]}}},
    {"$sort": {"mk": 1, "val": -1, "_id.iid": 1}},
    {"$group": {"_id": "$mk", "interest_id": {"$first": "$_id.iid"}, "val": {"$first": "$val"}}},
    {"$lookup": {"from": "__MAP__", "let": {"iid": "$interest_id"},
                 "pipeline": [{"$match": {"$expr": {"$eq": ["$id", "$$iid"]}}}, {"$project": {"interest_name": 1}}],
                 "as": "mp"}},
    {"$unwind": "$mp"},
    {"$addFields": {"interest_name": "$mp.interest_name"}},
    {"$setWindowFields": {
        "sortBy": {"_id": 1},
        "output": {
            "ROLLING_AVG_3MONTH": {"$avg": "$val", "window": {"documents": [-2, 0]}},
            "p1n": {"$shift": {"output": "$interest_name", "by": -1, "default": None}},
            "p1v": {"$shift": {"output": "$val", "by": -1, "default": None}},
            "p2n": {"$shift": {"output": "$interest_name", "by": -2, "default": None}},
            "p2v": {"$shift": {"output": "$val", "by": -2, "default": None}}}}},
    {"$project": {"_id": 0, "MONTH_DATE": "$_id", "interest_name": 1, "MAX_INDEX_COMPOSITION": "$val",
                  "ROLLING_AVG_3MONTH": 1, "PREV_MONTH_INTEREST_NAME": "$p1n",
                  "PREV_MONTH_MAX_COMPOSITION": "$p1v", "PREV_2MONTH_INTEREST_NAME": "$p2n",
                  "PREV_2MONTH_MAX_COMPOSITION": "$p2v"}},
    {"$sort": {"MONTH_DATE": 1}},
]

COLS = ["MONTH_DATE", "interest_name", "MAX_INDEX_COMPOSITION", "ROLLING_AVG_3MONTH",
        "PREV_MONTH_INTEREST_NAME", "PREV_MONTH_MAX_COMPOSITION",
        "PREV_2MONTH_INTEREST_NAME", "PREV_2MONTH_MAX_COMPOSITION"]


def run(reg):
    rec = manifest_record(IID)
    ref = sqlite_df(rec, REFERENCE_SQL)
    ref_v, ref_ok = score(ref, rec)

    metrics = import_table(reg, rec, IID, "interest_metrics", ["interest_id", "month_year", "composition", "index_value"])
    imap = import_table(reg, rec, IID, "interest_map", ["id", "interest_name"], index=["id"])

    coll = reg.db[metrics["collection"]]
    PIPELINE = []
    for s in PIPELINE_TEMPLATE:
        if "$lookup" in s:
            s = {"$lookup": {**s["$lookup"], "from": imap["collection"]}}
        PIPELINE.append(s)
    res = list(coll.aggregate(PIPELINE, allowDiskUse=True))
    pred = pd.DataFrame(res, columns=COLS)
    v, ok = score(pred, rec)

    artifact = {
        "pattern": "per-interest-month average ratio + monthly argmax + windowed rolling avg and $shift lags",
        "gold_sql": None, "gold_sql_reproduces_gold_csv": "n/a (no shipped gold SQL)",
        "reference_sql": REFERENCE_SQL, "reference_sql_matches_gold": ref_ok,
        "relational_semantics": ("For each interest-month average composition/index_value; keep each month's maximum; "
                                 "report a 3-month rolling average and the previous one/two months' top interest and value."),
        "source_schema": {"interest_metrics": ["interest_id", "month_year", "composition", "index_value"],
                          "interest_map": ["id", "interest_name"]},
        "mongodb": {"database": "codegen_capstone_pilot", "imports": [metrics, imap], "_id": "default ObjectId"},
        "mql": {"collection": coll.name, "method": "aggregate", "options": {"allowDiskUse": True}, "pipeline": PIPELINE},
        "mongo_result": res,
        "validation": {"per_gold": v, "matched_any_gold": ok},
        "assumptions": [
            "The metric per interest-month is AVG(composition / index_value).",
            "month_year 'MM-YYYY' is re-keyed to 'YYYY-MM' for chronological sort.",
            "Rolling average uses up to the current and two prior months; lags via $shift.",
        ],
    }
    return ("passed" if ok else "failed"), matched_reason(v, ok), artifact
