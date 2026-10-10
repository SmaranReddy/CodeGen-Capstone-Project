"""local335 - constructors whose drivers scored the fewest points among point-scorers, per season.

No shipped gold SQL: a labelled reference SQL is authored and verified against the gold
CSV first. A driver-season total is the sum of results.points; a season is counted for
every constructor that driver raced for that year (distinct years per constructor).
"""
import pandas as pd

from conversion_kit import import_table, manifest_record, matched_reason, pattern_record, score, sqlite_df

IID = "local335"

REFERENCE_SQL = """
WITH dy AS (SELECT r.driver_id, ra.year, SUM(r.points) AS pts
            FROM results r JOIN races ra ON ra.race_id = r.race_id
            WHERE ra.year >= 2001 GROUP BY r.driver_id, ra.year),
sc AS (SELECT * FROM dy WHERE pts > 0),
miny AS (SELECT year, MIN(pts) AS mp FROM sc GROUP BY year),
low AS (SELECT sc.driver_id, sc.year FROM sc JOIN miny ON miny.year = sc.year AND miny.mp = sc.pts)
SELECT c.name AS CONSTRUCTOR_NAME, COUNT(DISTINCT low.year) AS SEASONS_WITH_LOWEST
FROM low JOIN results r ON r.driver_id = low.driver_id
     JOIN races ra ON ra.race_id = r.race_id AND ra.year = low.year
     JOIN constructors c ON c.constructor_id = r.constructor_id
GROUP BY c.name ORDER BY SEASONS_WITH_LOWEST DESC, c.name LIMIT 5
"""

PIPELINE_TEMPLATE = [
    {"$lookup": {"from": "__RACES__", "localField": "race_id", "foreignField": "race_id", "as": "ra"}},
    {"$unwind": "$ra"},
    {"$group": {"_id": {"d": "$driver_id", "y": "$ra.year"}, "pts": {"$sum": "$points"},
                "cons": {"$addToSet": "$constructor_id"}}},
    {"$match": {"pts": {"$gt": 0}}},
    {"$group": {"_id": "$_id.y", "minpts": {"$min": "$pts"}, "rows": {"$push": "$$ROOT"}}},
    {"$unwind": "$rows"},
    {"$match": {"$expr": {"$eq": ["$rows.pts", "$minpts"]}}},
    {"$project": {"y": "$_id", "cons": "$rows.cons"}},
    {"$unwind": "$cons"},
    {"$lookup": {"from": "__CONSTRUCTORS__", "localField": "cons", "foreignField": "constructor_id", "as": "c"}},
    {"$unwind": "$c"},
    {"$group": {"_id": "$c.name", "years": {"$addToSet": "$y"}}},
    {"$addFields": {"SEASONS_WITH_LOWEST": {"$size": "$years"}}},
    {"$sort": {"SEASONS_WITH_LOWEST": -1, "_id": 1}},
    {"$limit": 5},
    {"$project": {"_id": 0, "CONSTRUCTOR_NAME": "$_id", "SEASONS_WITH_LOWEST": 1}},
]


def run(reg):
    rec = manifest_record(IID)
    ref = sqlite_df(rec, REFERENCE_SQL)
    ref_v, ref_ok = score(ref, rec)

    res_t = import_table(reg, rec, IID, "results", ["driver_id", "race_id", "constructor_id", "points"], index=["race_id"])
    races = import_table(reg, rec, IID, "races", ["race_id", "year"], where="year >= 2001", index=["race_id"])
    cons = import_table(reg, rec, IID, "constructors", ["constructor_id", "name"], index=["constructor_id"])

    coll = reg.db[res_t["collection"]]
    repl = {"__RACES__": races["collection"], "__CONSTRUCTORS__": cons["collection"]}
    PIPELINE = []
    for s in PIPELINE_TEMPLATE:
        if "$lookup" in s:
            s = {"$lookup": {**s["$lookup"], "from": repl[s["$lookup"]["from"]]}}
        PIPELINE.append(s)
    res = list(coll.aggregate(PIPELINE, allowDiskUse=True))
    pred = pd.DataFrame(res, columns=["CONSTRUCTOR_NAME", "SEASONS_WITH_LOWEST"])
    v, ok = score(pred, rec)

    artifact = {
        "pattern": "driver-season point totals + per-season minimum + constructor-year distinct count",
        "gold_sql": None, "gold_sql_reproduces_gold_csv": "n/a (no shipped gold SQL)",
        "reference_sql": REFERENCE_SQL, "reference_sql_matches_gold": ref_ok,
        "relational_semantics": ("Since 2001: per driver-year total points; keep point-scoring drivers; per season find the "
                                 "lowest total; count distinct seasons per constructor the lowest driver raced for; top 5."),
        "source_schema": {"results": ["driver_id", "race_id", "constructor_id", "points"],
                          "races": ["race_id", "year"], "constructors": ["constructor_id", "name"]},
        "mongodb": {"database": "codegen_capstone_pilot", "imports": [res_t, races, cons], "_id": "default ObjectId"},
        "mql": {"collection": coll.name, "method": "aggregate", "options": {"allowDiskUse": True}, "pipeline": PIPELINE},
        "mongo_result": res,
        "validation": {"per_gold": v, "matched_any_gold": ok},
        "assumptions": [
            "A driver-season totals results.points; only drivers with a positive total are point-scorers.",
            "A lowest driver-season is credited to every constructor they raced for that year; seasons are distinct.",
            "races is imported only for year >= 2001.",
        ],
    }
    return ("passed" if ok else "failed"), matched_reason(v, ok), artifact
