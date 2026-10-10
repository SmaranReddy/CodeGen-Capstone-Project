"""local017 - the year whose two most common collision causes are unique to it.

Targets gold _b (2001, integer). The shipped SQL yields 2001 too, but as TEXT because
STRFTIME returns a string; the comparator treats a number as never equal to a string,
so the MQL returns an integer year. Category percentages are ranked per year; the top-2
categories are concatenated and years with a unique pair are returned.
"""
import pandas as pd

from conversion_kit import (import_table, manifest_record, matched_reason, pattern_record, score, sqlite_df)

IID = "local017"


def run(reg):
    rec = manifest_record(IID)
    gold_sql = open(rec["gold_sql_path"], encoding="utf-8").read()
    ref = sqlite_df(rec, gold_sql)
    ref_v, ref_ok = score(ref, rec)

    m = import_table(reg, rec, IID, "collisions", ["case_id", "collision_date", "pcf_violation_category"])
    coll = reg.db[m["collection"]]

    PIPELINE = [
        {"$addFields": {"year": {"$toInt": {"$substrCP": ["$collision_date", 0, 4]}}}},
        {"$group": {"_id": {"year": "$year", "cat": "$pcf_violation_category"}, "sub": {"$sum": 1}}},
        {"$setWindowFields": {"partitionBy": "$_id.year", "output": {"ann": {"$sum": "$sub"}}}},
        {"$addFields": {"pct": {"$round": [{"$multiply": [{"$divide": ["$sub", "$ann"]}, 100]}, 1]}}},
        {"$setWindowFields": {"partitionBy": "$_id.year", "sortBy": {"pct": -1},
                              "output": {"r": {"$documentNumber": {}}}}},
        {"$match": {"r": {"$lte": 2}}},
        {"$sort": {"_id.year": 1, "r": 1}},
        {"$group": {"_id": "$_id.year", "cats": {"$push": "$_id.cat"}}},
        {"$addFields": {"top2": {"$reduce": {"input": "$cats", "initialValue": "",
            "in": {"$cond": [{"$eq": ["$$value", ""]},
                             {"$ifNull": ["$$this", ""]},
                             {"$concat": ["$$value", ", ", {"$ifNull": ["$$this", ""]}]}]}}}}},
        {"$group": {"_id": "$top2", "years": {"$push": "$_id"}}},
        {"$match": {"$expr": {"$eq": [{"$size": "$years"}, 1]}}},
        {"$project": {"_id": 0, "year": {"$arrayElemAt": ["$years", 0]}}},
    ]
    res = list(coll.aggregate(PIPELINE))
    pred = pd.DataFrame(res)
    v, ok = score(pred, rec)

    artifact = {
        "pattern": "percentage ranking per year + string aggregation + uniqueness filter",
        "gold_sql": gold_sql, "gold_sql_reproduces_gold_csv": ref_ok,
        "reference_sql": gold_sql, "reference_sql_matches_gold": ref_ok,
        "note_on_gold_sql": "Shipped SQL returns '2001' as TEXT; the comparator requires a number, so the MQL casts the year to int.",
        "relational_semantics": ("Per year, rank pcf_violation_category by its share of collisions (1 decimal); take the top 2; "
                                 "concatenate them; keep years whose pair is unique; return those years."),
        "source_schema": {"collisions": ["case_id REAL", "collision_date TEXT 'YYYY-MM-DD'",
                                         "pcf_violation_category TEXT (nullable)"]},
        "mongodb": {"database": "codegen_capstone_pilot", "imports": [m], "_id": "default ObjectId"},
        "mql": {"collection": coll.name, "method": "aggregate", "pipeline": PIPELINE},
        "mongo_result": res,
        "validation": {"per_gold": v, "matched_any_gold": ok, "targeted_gold": "local017_b.csv (2001)"},
        "assumptions": [
            "Target gold _b (2001); gold _a (2021) is a competing interpretation.",
            "ROW_NUMBER with pct ties is order-sensitive; ties at ranks 1-2 could change the concatenation (checked against the shipped SQL result).",
        ],
    }
    return ("passed" if ok else "failed"), matched_reason(v, ok), artifact
