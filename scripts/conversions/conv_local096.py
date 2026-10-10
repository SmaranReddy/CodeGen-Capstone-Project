"""local096 - yearly percentage of films whose entire credited cast is female.

No shipped gold SQL: a labelled reference SQL is authored and verified against the gold
CSV first. M_Cast.PID carries a trailing space, so a normalized `pid_norm` field is added
before the $lookup to Person. The year is the last 4 characters of Movie.year.
"""
import pandas as pd

from conversion_kit import import_table, manifest_record, matched_reason, pattern_record, score, sqlite_df

IID = "local096"

REFERENCE_SQL = """
WITH cast_norm AS (SELECT MID, TRIM(PID) AS PID FROM M_Cast),
fg AS (SELECT m.MID, CAST(substr(m.year, -4) AS INT) AS y,
              COUNT(c.PID) AS ncast,
              SUM(CASE WHEN p.Gender = 'Female' THEN 1 ELSE 0 END) AS nfemale
       FROM Movie m
       LEFT JOIN cast_norm c ON c.MID = m.MID
       LEFT JOIN Person p ON p.PID = c.PID
       GROUP BY m.MID, y)
SELECT y AS YEAR, COUNT(*) AS TOTAL_MOVIES,
       ROUND(100.0 * SUM(CASE WHEN ncast > 0 AND nfemale = ncast THEN 1 ELSE 0 END) / COUNT(*), 2) AS PERCENTAGE_EXCLUSIVELY_FEMALE
FROM fg GROUP BY y ORDER BY y
"""

YEAR_EXPR = {"$toInt": {"$substrCP": ["$year", {"$subtract": [{"$strLenCP": "$year"}, 4]}, 4]}}

PIPELINE_TEMPLATE = [
    {"$set": {"yr": YEAR_EXPR}},
    {"$lookup": {"from": "__CAST__", "localField": "MID", "foreignField": "MID", "as": "cast"}},
    {"$unwind": {"path": "$cast", "preserveNullAndEmptyArrays": True}},
    {"$lookup": {"from": "__PERSON__", "localField": "cast.pid_norm", "foreignField": "PID", "as": "p"}},
    {"$unwind": {"path": "$p", "preserveNullAndEmptyArrays": True}},
    {"$group": {"_id": "$MID", "yr": {"$first": "$yr"},
                "ncast": {"$sum": {"$cond": [{"$ifNull": ["$cast.PID", False]}, 1, 0]}},
                "nfemale": {"$sum": {"$cond": [{"$eq": ["$p.Gender", "Female"]}, 1, 0]}}}},
    {"$group": {"_id": "$yr", "TOTAL_MOVIES": {"$sum": 1},
                "excl": {"$sum": {"$cond": [{"$and": [{"$gt": ["$ncast", 0]}, {"$eq": ["$nfemale", "$ncast"]}]}, 1, 0]}}}},
    {"$addFields": {"PERCENTAGE_EXCLUSIVELY_FEMALE": {"$round": [{"$multiply": [100, {"$divide": ["$excl", "$TOTAL_MOVIES"]}]}, 2]}}},
    {"$project": {"_id": 0, "YEAR": "$_id", "TOTAL_MOVIES": 1, "PERCENTAGE_EXCLUSIVELY_FEMALE": 1}},
    {"$sort": {"YEAR": 1}},
]


def run(reg):
    rec = manifest_record(IID)
    ref = sqlite_df(rec, REFERENCE_SQL)
    ref_v, ref_ok = score(ref, rec)

    movie = import_table(reg, rec, IID, "Movie", ["MID", "year"], index=["MID"])
    cast = import_table(reg, rec, IID, "M_Cast", ["MID", "PID"])
    person = import_table(reg, rec, IID, "Person", ["PID", "Gender"], index=["PID"])
    cast_coll = reg.db[cast["collection"]]
    cast_coll.update_many({}, [{"$set": {"pid_norm": {"$trim": {"input": "$PID"}}}}])
    cast_coll.create_index("MID")

    coll = reg.db[movie["collection"]]
    repl = {"__CAST__": cast["collection"], "__PERSON__": person["collection"]}
    PIPELINE = []
    for s in PIPELINE_TEMPLATE:
        if "$lookup" in s:
            s = {"$lookup": {**s["$lookup"], "from": repl[s["$lookup"]["from"]]}}
        PIPELINE.append(s)
    res = list(coll.aggregate(PIPELINE, allowDiskUse=True))
    pred = pd.DataFrame(res, columns=["YEAR", "TOTAL_MOVIES", "PERCENTAGE_EXCLUSIVELY_FEMALE"])
    v, ok = score(pred, rec)

    artifact = {
        "pattern": "cast-gender rollup per film then per-year percentage (trim + $round)",
        "gold_sql": None, "gold_sql_reproduces_gold_csv": "n/a (no shipped gold SQL)",
        "reference_sql": REFERENCE_SQL, "reference_sql_matches_gold": ref_ok,
        "relational_semantics": ("Per film determine cast size and female count; a film is exclusively female when it has "
                                 "at least one actor and all actors are Female; report yearly share (%), 2 dp."),
        "source_schema": {"Movie": ["MID", "year"], "M_Cast": ["MID", "PID"], "Person": ["PID", "Gender"]},
        "mongodb": {"database": "codegen_capstone_pilot", "imports": [movie, cast, person], "_id": "default ObjectId"},
        "mql": {"collection": coll.name, "method": "aggregate", "options": {"allowDiskUse": True}, "pipeline": PIPELINE},
        "mongo_result": res,
        "validation": {"per_gold": v, "matched_any_gold": ok},
        "assumptions": [
            "M_Cast.PID is normalized with $trim before the Person join (it stores a trailing space).",
            "The film year is the last 4 characters of Movie.year; films with no cast are not female-only.",
        ],
    }
    return ("passed" if ok else "failed"), matched_reason(v, ok), artifact
