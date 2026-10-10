"""local301 - percentage change in weekly sales around June 15 for 2018/2019/2020.

Targets gold _a (condition col = percentage_change). For each year, delta_weeks is
computed against that year's June 15 (JULIANDAY differenced by 7 days, +1); the four
weeks before [delta -3..0] and after [delta 1..4] are summed and the percentage change
computed. Three copies of the pipeline are combined with $unionWith.
"""
import pandas as pd

from conversion_kit import (import_table, manifest_record, matched_reason, pattern_record, score, sqlite_df)

IID = "local301"


def _year_block(coll, year):
    base = f"{year}-06-15"
    return [
        {"$addFields": {"wd": {"$dateFromString": {"dateString": "$week_date"}}}},
        {"$addFields": {"days": {"$dateDiff": {"startDate": {"$dateFromString": {"dateString": base}},
                                               "endDate": "$wd", "unit": "day"}}}},
        {"$addFields": {"delta": {"$add": [{"$round": [{"$divide": ["$days", 7]}, 0]}, 1]}}},
        {"$group": {"_id": None,
                    "before": {"$sum": {"$cond": [{"$and": [{"$gte": ["$delta", -3]}, {"$lte": ["$delta", 0]}]}, "$sales", 0]}},
                    "after": {"$sum": {"$cond": [{"$and": [{"$gte": ["$delta", 1]}, {"$lte": ["$delta", 4]}]}, "$sales", 0]}}}},
        {"$project": {"_id": 0, "sale_year": {"$literal": year},
                      "percentage_change": {"$multiply": [{"$subtract": [{"$divide": ["$after", "$before"]}, 1]}, 100]}}},
    ]


def run(reg):
    rec = manifest_record(IID)
    gold_sql = open(rec["gold_sql_path"], encoding="utf-8").read()
    ref = sqlite_df(rec, gold_sql)
    ref_v, ref_ok = score(ref, rec)

    m = import_table(reg, rec, IID, "cleaned_weekly_sales", ["week_date", "sales"])
    coll = reg.db[m["collection"]]

    first = _year_block(coll.name, 2018)
    PIPELINE = first + [
        {"$unionWith": {"coll": coll.name, "pipeline": _year_block(coll.name, 2019)}},
        {"$unionWith": {"coll": coll.name, "pipeline": _year_block(coll.name, 2020)}},
        {"$sort": {"sale_year": 1}},
    ]
    res = list(coll.aggregate(PIPELINE))
    pred = pd.DataFrame(res)
    v, ok = score(pred, rec)

    artifact = {
        "pattern": "date arithmetic (julian-day style) + conditional sums + UNION ALL of per-year blocks",
        "gold_sql": gold_sql, "gold_sql_reproduces_gold_csv": ref_ok,
        "reference_sql": gold_sql, "reference_sql_matches_gold": ref_ok,
        "prechecks": {"week_date_format": "YYYY-MM-DD (ISO text)", "sales_type": "INTEGER",
                      "rounding_note": "days/7 is never a .5 boundary here, so $round (half-even) equals SQLite round"},
        "relational_semantics": ("For each year block, delta_weeks = round((week_date - YEAR-06-15)/7)+1; "
                                 "before = sum(sales) over delta in [-3,0]; after = sum(sales) over [1,4]; "
                                 "pct = (after/before - 1)*100. All rows participate; other years fall outside the ranges."),
        "source_schema": {"cleaned_weekly_sales": ["week_date TEXT", "sales INTEGER"]},
        "mongodb": {"database": "codegen_capstone_pilot", "imports": [m], "_id": "default ObjectId"},
        "mql": {"collection": coll.name, "method": "aggregate", "pipeline": PIPELINE},
        "mongo_result": res,
        "validation": {"per_gold": v, "matched_any_gold": ok, "targeted_gold": "local301_a.csv (percentage_change)"},
        "assumptions": [
            "Target gold _a; golds _b/_c use different pre-window definitions and are not compared.",
            "All rows are summed (as the SQL does); the year-of-each-row is not filtered because the delta window already excludes other years.",
        ],
    }
    return ("passed" if ok else "failed"), matched_reason(v, ok), artifact
