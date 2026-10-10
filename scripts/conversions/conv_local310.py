"""local310 - the three years with the smallest (best driver + best constructor) points.

No shipped gold SQL: a labelled reference SQL is authored and verified against the gold
CSV first. Points are summed from results per driver and per constructor, per season.
"""
import pandas as pd

from conversion_kit import (import_table, manifest_record, matched_reason, pattern_record, score, sqlite_df)

IID = "local310"

REFERENCE_SQL = """
WITH dp AS (SELECT r.year, res.driver_id, SUM(res.points) AS p
            FROM results res JOIN races r ON res.race_id = r.race_id
            GROUP BY r.year, res.driver_id),
     cp AS (SELECT r.year, res.constructor_id, SUM(res.points) AS p
            FROM results res JOIN races r ON res.race_id = r.race_id
            GROUP BY r.year, res.constructor_id),
     dm AS (SELECT year, MAX(p) AS m FROM dp GROUP BY year),
     cm AS (SELECT year, MAX(p) AS m FROM cp GROUP BY year)
SELECT dm.year
FROM dm JOIN cm ON dm.year = cm.year
ORDER BY dm.m + cm.m ASC
LIMIT 3
"""


def run(reg):
    rec = manifest_record(IID)
    ref = sqlite_df(rec, REFERENCE_SQL)
    ref_v, ref_ok = score(ref, rec)

    m_res = import_table(reg, rec, IID, "results", ["race_id", "driver_id", "constructor_id", "points"])
    m_races = import_table(reg, rec, IID, "races", ["race_id", "year"], index=["race_id"])
    coll = reg.db[m_res["collection"]]

    PIPELINE = [
        {"$lookup": {"from": m_races["collection"], "localField": "race_id",
                     "foreignField": "race_id", "as": "rc"}},
        {"$unwind": "$rc"},
        {"$facet": {
            "d": [{"$group": {"_id": {"y": "$rc.year", "d": "$driver_id"}, "p": {"$sum": "$points"}}},
                  {"$group": {"_id": "$_id.y", "m": {"$max": "$p"}}}],
            "c": [{"$group": {"_id": {"y": "$rc.year", "c": "$constructor_id"}, "p": {"$sum": "$points"}}},
                  {"$group": {"_id": "$_id.y", "m": {"$max": "$p"}}}]}},
        {"$unwind": "$d"},
        {"$unwind": "$c"},
        {"$match": {"$expr": {"$eq": ["$d._id", "$c._id"]}}},
        {"$addFields": {"total": {"$add": ["$d.m", "$c.m"]}}},
        {"$sort": {"total": 1}},
        {"$limit": 3},
        {"$project": {"_id": 0, "year": "$d._id"}},
    ]
    res = list(coll.aggregate(PIPELINE, allowDiskUse=True))
    pred = pd.DataFrame(res, columns=["year"])
    v, ok = score(pred, rec)

    artifact = {
        "pattern": "facet (driver/constructor maxima) + join on year + smallest-N",
        "gold_sql": None, "gold_sql_reproduces_gold_csv": "n/a (no shipped gold SQL)",
        "reference_sql": REFERENCE_SQL, "reference_sql_matches_gold": ref_ok,
        "relational_semantics": ("Per season: highest driver season points + highest constructor season points (both summed from results); "
                                 "the three seasons with the smallest sum (ties ordered arbitrarily)."),
        "source_schema": {"results": ["race_id", "driver_id", "constructor_id", "points"], "races": ["race_id", "year"]},
        "mongodb": {"database": "codegen_capstone_pilot", "imports": [m_res, m_races], "_id": "default ObjectId",
                    "indexes": ["races.race_id"]},
        "mql": {"collection": coll.name, "method": "aggregate", "options": {"allowDiskUse": True}, "pipeline": PIPELINE},
        "mongo_result": res,
        "validation": {"per_gold": v, "matched_any_gold": ok},
        "assumptions": [
            "Points come only from the results table (season standings are not used).",
            "The three smallest sums are distinct enough that tie ordering does not change the returned years.",
        ],
    }
    return ("passed" if ok else "failed"), matched_reason(v, ok), artifact
