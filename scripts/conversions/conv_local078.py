"""local078 - top-10 and bottom-10 interest categories by their peak composition.

interest_metrics JOIN interest_map (interest_id = id; interest_id is REAL, id INTEGER,
so it must be cast to int in MQL or equality fails by BSON type). RANK() within each
interest_name by composition DESC; keep rank 1; top 10 / bottom 10 by composition;
UNION (dedup). condition_cols=[1,2] compares interest_name + composition.
"""
import pandas as pd

from conversion_kit import (import_table, manifest_record, matched_reason, pattern_record, score, sqlite_df)

IID = "local078"


def run(reg):
    rec = manifest_record(IID)
    gold_sql = open(rec["gold_sql_path"], encoding="utf-8").read()
    ref = sqlite_df(rec, gold_sql)
    ref_v, ref_ok = score(ref, rec)

    m1 = import_table(reg, rec, IID, "interest_metrics",
                      ["month_year", "interest_id", "composition"])
    m2 = import_table(reg, rec, IID, "interest_map", ["id", "interest_name"])
    coll = reg.db[m1["collection"]]

    PIPELINE = [
        {"$match": {"month_year": {"$ne": None}}},  # WHERE month_year IS NOT NULL
        {"$addFields": {"interest_id_int": {"$toInt": "$interest_id"}}},  # REAL -> INT to match interest_map.id
        {"$lookup": {"from": m2["collection"], "localField": "interest_id_int",
                     "foreignField": "id", "as": "m"}},
        {"$unwind": "$m"},
        {"$setWindowFields": {"partitionBy": "$m.interest_name", "sortBy": {"composition": -1},
                              "output": {"interest_rank": {"$rank": {}}}}},
        {"$match": {"interest_rank": 1}},
        {"$project": {"_id": 0, "time": "$month_year", "interest_name": "$m.interest_name",
                      "composition": 1}},
        {"$facet": {
            "top": [{"$sort": {"composition": -1}}, {"$limit": 10}],
            "bottom": [{"$sort": {"composition": 1}}, {"$limit": 10}]}},
        {"$project": {"rows": {"$concatArrays": ["$top", "$bottom"]}}},
        {"$unwind": "$rows"},
        {"$replaceRoot": {"newRoot": "$rows"}},
        {"$group": {"_id": {"time": "$time", "interest_name": "$interest_name",
                            "composition": "$composition"}}},  # UNION dedup
        {"$project": {"_id": 0, "time": "$_id.time", "interest_name": "$_id.interest_name",
                      "composition": "$_id.composition"}},
    ]
    res = list(coll.aggregate(PIPELINE))
    pred = pd.DataFrame(res)
    v, ok = score(pred, rec)

    artifact = {
        "pattern": "RANK window + top-N/bottom-N + UNION dedup",
        "gold_sql": gold_sql, "gold_sql_reproduces_gold_csv": ref_ok,
        "reference_sql": gold_sql, "reference_sql_matches_gold": ref_ok,
        "prechecks": {"interest_id_type": "REAL in interest_metrics vs INTEGER id in interest_map -> $toInt cast required"},
        "relational_semantics": ("Peak (max composition) month per interest category; top 10 and bottom 10 by that peak; "
                                 "UNION removes duplicates."),
        "source_schema": {"interest_metrics": ["month_year TEXT", "interest_id REAL", "composition REAL"],
                          "interest_map": ["id INT", "interest_name TEXT"]},
        "mongodb": {"database": "codegen_capstone_pilot", "imports": [m1, m2], "_id": "default ObjectId"},
        "mql": {"collection": coll.name, "method": "aggregate", "pipeline": PIPELINE},
        "mongo_result": res,
        "validation": {"per_gold": v, "matched_any_gold": ok},
        "assumptions": [
            "interest_id (REAL) is cast with $toInt before the $lookup; without the cast no document matches (BSON number type mismatch).",
            "RANK with ties keeps every tied peak (SQL RANK semantics).",
            "Top/bottom sets are de-duplicated with a $group to reproduce UNION (comparator requires equal column lengths).",
        ],
    }
    return ("passed" if ok else "failed"), matched_reason(v, ok), artifact
