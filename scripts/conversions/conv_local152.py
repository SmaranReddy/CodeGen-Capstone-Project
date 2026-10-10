"""local152 - top 9 directors by movie count, with rating/vote/duration stats.

No shipped gold SQL: a labelled reference SQL is authored and verified against the gold
CSV first. "Average inter-movie duration" is the mean gap in days between consecutive
release dates (date_published), rounded to the nearest integer.
"""
import pandas as pd

from conversion_kit import import_table, manifest_record, matched_reason, pattern_record, score, sqlite_df

IID = "local152"

REFERENCE_SQL = """
WITH d AS (
  SELECT dm.name_id, m.id AS movie_id, m.duration, m.date_published, r.avg_rating, r.total_votes
  FROM director_mapping dm
  JOIN movies m ON m.id = dm.movie_id
  JOIN ratings r ON r.movie_id = m.id),
g AS (
  SELECT name_id, COUNT(*) AS mc, SUM(duration) AS td, AVG(avg_rating) AS ar,
         SUM(total_votes) AS tv, MIN(avg_rating) AS mn, MAX(avg_rating) AS mx
  FROM d GROUP BY name_id),
dates AS (
  SELECT name_id, julianday(date_published)
         - julianday(LAG(date_published) OVER (PARTITION BY name_id ORDER BY date_published)) AS gap
  FROM d),
inter AS (SELECT name_id, ROUND(AVG(gap)) AS aid FROM dates GROUP BY name_id)
SELECT g.name_id AS DIRECTOR_ID, n.name AS DIRECTOR_NAME, g.mc AS MOVIE_COUNT,
       i.aid AS AVG_INTER_MOVIE_DURATION, ROUND(g.ar,2) AS AVG_RATING, g.tv AS TOTAL_VOTES,
       g.mn AS MIN_RATING, g.mx AS MAX_RATING, g.td AS TOTAL_MOVIE_DURATION
FROM g JOIN names n ON n.id = g.name_id JOIN inter i ON i.name_id = g.name_id
ORDER BY g.mc DESC, g.td DESC LIMIT 9
"""

PIPELINE_TEMPLATE = [
    {"$lookup": {"from": "__MOVIES__", "localField": "movie_id", "foreignField": "id", "as": "m"}},
    {"$unwind": "$m"},
    {"$lookup": {"from": "__RATINGS__", "localField": "movie_id", "foreignField": "movie_id", "as": "r"}},
    {"$unwind": "$r"},
    {"$lookup": {"from": "__NAMES__", "localField": "name_id", "foreignField": "id", "as": "nm"}},
    {"$unwind": "$nm"},
    {"$group": {"_id": "$name_id", "name": {"$first": "$nm.name"}, "mc": {"$sum": 1},
                "td": {"$sum": "$m.duration"}, "ar": {"$avg": "$r.avg_rating"},
                "tv": {"$sum": "$r.total_votes"}, "mn": {"$min": "$r.avg_rating"},
                "mx": {"$max": "$r.avg_rating"}, "dates": {"$push": "$m.date_published"}}},
    {"$set": {"sd": {"$filter": {"input": {"$sortArray": {"input": "$dates", "sortBy": 1}}, "as": "dt",
                                 "cond": {"$ne": ["$$dt", None]}}}}},
    {"$set": {"aid": {"$round": [{"$add": [{"$avg": {"$map": {
        "input": {"$range": [1, {"$size": "$sd"}]}, "as": "i", "in": {"$divide": [
            {"$subtract": [
                {"$dateFromString": {"dateString": {"$arrayElemAt": ["$sd", "$$i"]}, "format": "%Y-%m-%d %H:%M:%S"}},
                {"$dateFromString": {"dateString": {"$arrayElemAt": ["$sd", {"$subtract": ["$$i", 1]}]}, "format": "%Y-%m-%d %H:%M:%S"}}]},
            86400000]}}}}, 1e-9]}, 0]}}},
    {"$set": {"ar": {"$round": [{"$add": ["$ar", 1e-9]}, 2]}}},
    {"$sort": {"mc": -1, "td": -1}},
    {"$limit": 9},
    {"$project": {"_id": 0, "DIRECTOR_ID": "$_id", "DIRECTOR_NAME": "$name", "MOVIE_COUNT": "$mc",
                  "AVG_INTER_MOVIE_DURATION": "$aid", "AVG_RATING": "$ar", "TOTAL_VOTES": "$tv",
                  "MIN_RATING": "$mn", "MAX_RATING": "$mx", "TOTAL_MOVIE_DURATION": "$td"}},
]

COLS = ["DIRECTOR_ID", "DIRECTOR_NAME", "MOVIE_COUNT", "AVG_INTER_MOVIE_DURATION", "AVG_RATING",
        "TOTAL_VOTES", "MIN_RATING", "MAX_RATING", "TOTAL_MOVIE_DURATION"]


def run(reg):
    rec = manifest_record(IID)
    ref = sqlite_df(rec, REFERENCE_SQL)
    ref_v, ref_ok = score(ref, rec)

    movies = import_table(reg, rec, IID, "movies", ["id", "duration", "date_published"], index=["id"])
    dm = import_table(reg, rec, IID, "director_mapping", ["movie_id", "name_id"], index=["movie_id"])
    ratings = import_table(reg, rec, IID, "ratings", ["movie_id", "avg_rating", "total_votes"], index=["movie_id"])
    names = import_table(reg, rec, IID, "names", ["id", "name"], index=["id"])

    coll = reg.db[dm["collection"]]
    repl = {"__MOVIES__": movies["collection"], "__RATINGS__": ratings["collection"], "__NAMES__": names["collection"]}
    PIPELINE = []
    for s in PIPELINE_TEMPLATE:
        if "$lookup" in s:
            s = {"$lookup": {**s["$lookup"], "from": repl[s["$lookup"]["from"]]}}
        PIPELINE.append(s)
    res = list(coll.aggregate(PIPELINE, allowDiskUse=True))
    pred = pd.DataFrame(res, columns=COLS)
    v, ok = score(pred, rec)

    artifact = {
        "pattern": "multi-table joins + per-group stats + sorted consecutive date gaps",
        "gold_sql": None, "gold_sql_reproduces_gold_csv": "n/a (no shipped gold SQL)",
        "reference_sql": REFERENCE_SQL, "reference_sql_matches_gold": ref_ok,
        "relational_semantics": ("Per director aggregate movie count, ratings, votes and duration; average inter-movie "
                                 "duration = mean days between consecutive release dates; top 9 by count then duration."),
        "source_schema": {"movies": ["id", "duration", "date_published"],
                          "director_mapping": ["movie_id", "name_id"],
                          "ratings": ["movie_id", "avg_rating", "total_votes"], "names": ["id", "name"]},
        "mongodb": {"database": "codegen_capstone_pilot", "imports": [movies, dm, ratings, names], "_id": "default ObjectId"},
        "mql": {"collection": coll.name, "method": "aggregate", "options": {"allowDiskUse": True}, "pipeline": PIPELINE},
        "mongo_result": res,
        "validation": {"per_gold": v, "matched_any_gold": ok},
        "assumptions": ["AVG_INTER_MOVIE_DURATION = round(mean gap in days between consecutive date_published values).",
                        "AVG_RATING rounded to 2 decimals; rating stats use ratings.avg_rating.",
                        "Order by movie count desc then total duration desc; top 9."],
    }
    return ("passed" if ok else "failed"), matched_reason(v, ok), artifact
