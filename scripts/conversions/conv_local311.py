"""local311 - constructor-year top combined points (best driver + constructor standings).

No shipped gold SQL: a labelled reference SQL is authored and verified against the gold
CSV first. combined_points = the constructor's final standings points + its best driver's
final standings points in that year; return the top 3, output name and year (gold _a).
"""
import pandas as pd

from conversion_kit import import_table, manifest_record, matched_reason, pattern_record, score, sqlite_df

IID = "local311"

REFERENCE_SQL = """
WITH cy AS (SELECT cs.constructor_id, ra.year, MAX(cs.points) AS cpts
            FROM constructor_standings cs JOIN races ra ON ra.race_id = cs.race_id GROUP BY cs.constructor_id, ra.year),
dy AS (SELECT ds.driver_id, ra.year, MAX(ds.points) AS dpts
       FROM driver_standings ds JOIN races ra ON ra.race_id = ds.race_id GROUP BY ds.driver_id, ra.year),
dc AS (SELECT DISTINCT r.driver_id, r.constructor_id, ra.year FROM results r JOIN races ra ON ra.race_id = r.race_id),
combined AS (SELECT cy.constructor_id, cy.year, cy.cpts, MAX(dy.dpts) AS bdpts
             FROM cy JOIN dc ON dc.constructor_id = cy.constructor_id AND dc.year = cy.year
             JOIN dy ON dy.driver_id = dc.driver_id AND dy.year = cy.year
             GROUP BY cy.constructor_id, cy.year, cy.cpts)
SELECT c.name AS name, cb.year, (cb.cpts + cb.bdpts) AS combined_points
FROM combined cb JOIN constructors c ON c.constructor_id = cb.constructor_id
ORDER BY combined_points DESC LIMIT 3
"""

PIPELINE_TEMPLATE = [
    {"$lookup": {"from": "__RACES__", "localField": "race_id", "foreignField": "race_id", "as": "ra"}},
    {"$unwind": "$ra"},
    {"$group": {"_id": {"d": "$driver_id", "c": "$constructor_id", "y": "$ra.year"}}},
    {"$lookup": {"from": "__DS__", "let": {"d": "$_id.d", "y": "$_id.y"},
                 "pipeline": [
                     {"$match": {"$expr": {"$eq": ["$driver_id", "$$d"]}}},
                     {"$lookup": {"from": "__RACES__", "localField": "race_id", "foreignField": "race_id", "as": "rr"}},
                     {"$unwind": "$rr"},
                     {"$match": {"$expr": {"$eq": ["$rr.year", "$$y"]}}},
                     {"$group": {"_id": None, "m": {"$max": "$points"}}}],
                 "as": "ds"}},
    {"$lookup": {"from": "__CS__", "let": {"c": "$_id.c", "y": "$_id.y"},
                 "pipeline": [
                     {"$match": {"$expr": {"$eq": ["$constructor_id", "$$c"]}}},
                     {"$lookup": {"from": "__RACES__", "localField": "race_id", "foreignField": "race_id", "as": "rr"}},
                     {"$unwind": "$rr"},
                     {"$match": {"$expr": {"$eq": ["$rr.year", "$$y"]}}},
                     {"$group": {"_id": None, "m": {"$max": "$points"}}}],
                 "as": "cs"}},
    {"$match": {"$and": [{"ds": {"$ne": []}}, {"cs": {"$ne": []}}]}},
    {"$project": {"c": "$_id.c", "y": "$_id.y", "dpts": {"$arrayElemAt": ["$ds.m", 0]}, "cpts": {"$arrayElemAt": ["$cs.m", 0]}}},
    {"$group": {"_id": {"c": "$c", "y": "$y"}, "bdpts": {"$max": "$dpts"}, "cpts": {"$first": "$cpts"}}},
    {"$addFields": {"combined_points": {"$add": ["$bdpts", "$cpts"]}}},
    {"$sort": {"combined_points": -1, "_id.y": 1}},
    {"$limit": 3},
    {"$lookup": {"from": "__CONS__", "localField": "_id.c", "foreignField": "constructor_id", "as": "cn"}},
    {"$unwind": "$cn"},
    {"$project": {"_id": 0, "name": "$cn.name", "year": "$_id.y"}},
]


def run(reg):
    rec = manifest_record(IID)
    ref = sqlite_df(rec, REFERENCE_SQL)
    ref_v, ref_ok = score(ref[["name", "year"]], rec)

    results = import_table(reg, rec, IID, "results", ["driver_id", "constructor_id", "race_id"], index=["race_id"])
    races = import_table(reg, rec, IID, "races", ["race_id", "year"], index=["race_id"])
    ds = import_table(reg, rec, IID, "driver_standings", ["driver_id", "race_id", "points"], index=["driver_id"])
    cs = import_table(reg, rec, IID, "constructor_standings", ["constructor_id", "race_id", "points"], index=["constructor_id"])
    cons = import_table(reg, rec, IID, "constructors", ["constructor_id", "name"], index=["constructor_id"])

    coll = reg.db[results["collection"]]
    repl = {"__RACES__": races["collection"], "__DS__": ds["collection"], "__CS__": cs["collection"], "__CONS__": cons["collection"]}
    PIPELINE = []
    for s in PIPELINE_TEMPLATE:
        if "$lookup" in s:
            spec = s["$lookup"]
            new = {**spec, "from": repl[spec["from"]]}
            if "pipeline" in spec:
                new["pipeline"] = []
                for st in spec["pipeline"]:
                    if "$lookup" in st:
                        st = {"$lookup": {**st["$lookup"], "from": repl[st["$lookup"]["from"]]}}
                    new["pipeline"].append(st)
            s = {"$lookup": new}
        PIPELINE.append(s)
    res = list(coll.aggregate(PIPELINE, allowDiskUse=True))
    pred = pd.DataFrame(res, columns=["name", "year"])
    v, ok = score(pred, rec)

    artifact = {
        "pattern": "driver+constructor standings maxima joined to driver-team-year, top-3 by combined points",
        "gold_sql": None, "gold_sql_reproduces_gold_csv": "n/a (no shipped gold SQL)",
        "reference_sql": REFERENCE_SQL, "reference_sql_matches_gold": ref_ok,
        "relational_semantics": ("For each constructor-year take its final constructor-standings points plus the final "
                                 "standings points of its best-scoring driver that year; return the top 3."),
        "source_schema": {"results": ["driver_id", "constructor_id", "race_id"], "races": ["race_id", "year"],
                          "driver_standings": ["driver_id", "race_id", "points"], "constructor_standings": ["constructor_id", "race_id", "points"],
                          "constructors": ["constructor_id", "name"]},
        "mongodb": {"database": "codegen_capstone_pilot", "imports": [results, races, ds, cs, cons], "_id": "default ObjectId"},
        "mql": {"collection": coll.name, "method": "aggregate", "options": {"allowDiskUse": True}, "pipeline": PIPELINE},
        "mongo_result": res,
        "validation": {"per_gold": v, "matched_any_gold": ok},
        "assumptions": [
            "Final standings points per driver/constructor-year = MAX(points) across that year's races.",
            "A constructor's best driver is the one with the highest driver-standings points among its race entries.",
            "Output columns name and year match gold variant _a.",
        ],
    }
    return ("passed" if ok else "failed"), matched_reason(v, ok), artifact
