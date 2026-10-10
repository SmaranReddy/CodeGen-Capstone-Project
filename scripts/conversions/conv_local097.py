"""local097 - the 10-year window (inclusive of the start year) containing the most films.

No shipped gold SQL: a labelled reference SQL is authored and verified against the gold
CSV first. The year is the last 4 characters of Movie.year; every distinct year is a
candidate start and the window covers [start, start+9].
"""
import pandas as pd

from conversion_kit import (import_table, manifest_record, matched_reason, pattern_record, score, sqlite_df)

IID = "local097"

REFERENCE_SQL = """
WITH yc AS (
    SELECT CAST(substr(year, -4) AS INT) AS y, COUNT(*) AS n
    FROM Movie GROUP BY y
)
SELECT a.y AS start_year,
       (SELECT SUM(b.n) FROM yc b WHERE b.y BETWEEN a.y AND a.y + 9) AS total_movies
FROM yc a
ORDER BY total_movies DESC, a.y ASC
LIMIT 1
"""


def run(reg):
    rec = manifest_record(IID)
    ref = sqlite_df(rec, REFERENCE_SQL)
    ref_v, ref_ok = score(ref, rec)

    m = import_table(reg, rec, IID, "Movie", ["MID", "year"])
    coll = reg.db[m["collection"]]

    year_expr = {"$toInt": {"$substrCP": ["$year", {"$subtract": [{"$strLenCP": "$year"}, 4]}, 4]}}
    PIPELINE = [
        {"$addFields": {"y": year_expr}},
        {"$group": {"_id": "$y", "n": {"$sum": 1}}},
        {"$sort": {"_id": 1}},
        {"$group": {"_id": None, "years": {"$push": {"y": "$_id", "n": "$n"}}}},
        {"$project": {"cands": {"$map": {"input": "$years", "as": "a", "in": {
            "start_year": "$$a.y",
            "total_movies": {"$reduce": {"input": "$years", "initialValue": 0, "in": {"$add": [
                "$$value",
                {"$cond": [{"$and": [{"$gte": ["$$this.y", "$$a.y"]},
                                     {"$lte": ["$$this.y", {"$add": ["$$a.y", 9]}]}]}, "$$this.n", 0]}]}}}}}}}},
        {"$unwind": "$cands"},
        {"$sort": {"cands.total_movies": -1, "cands.start_year": 1}},
        {"$limit": 1},
        {"$replaceRoot": {"newRoot": "$cands"}},
        {"$project": {"_id": 0, "start_year": 1, "total_movies": 1}},
    ]
    res = list(coll.aggregate(PIPELINE, allowDiskUse=True))
    pred = pd.DataFrame(res, columns=["start_year", "total_movies"])
    v, ok = score(pred, rec)

    artifact = {
        "pattern": "per-year counts + all-pairs windowed sum (array map/reduce) + top-1",
        "gold_sql": None, "gold_sql_reproduces_gold_csv": "n/a (no shipped gold SQL)",
        "reference_sql": REFERENCE_SQL, "reference_sql_matches_gold": ref_ok,
        "relational_semantics": ("Year = last 4 chars of Movie.year as a number; count films per year; for every distinct start year "
                                 "sum the counts in [start, start+9]; return the max (ties by smallest start year)."),
        "source_schema": {"Movie": ["MID TEXT", "year TEXT"]},
        "mongodb": {"database": "codegen_capstone_pilot", "imports": [m], "_id": "default ObjectId"},
        "mql": {"collection": coll.name, "method": "aggregate", "options": {"allowDiskUse": True}, "pipeline": PIPELINE},
        "mongo_result": res,
        "validation": {"per_gold": v, "matched_any_gold": ok},
        "assumptions": [
            "Window is inclusive of the start year and spans 10 distinct years ([start, start+9]); a 9-year span reproduces the gold.",
            "condition_cols=[1] compares only total_movies; start_year is returned for completeness.",
        ],
    }
    return ("passed" if ok else "failed"), matched_reason(v, ok), artifact
