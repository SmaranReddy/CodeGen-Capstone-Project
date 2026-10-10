"""local230 - top-4 directors within the top-3 genres of movies rated above 8.

No shipped gold SQL: a labelled reference SQL is authored and verified against the gold
CSV first. A single pipeline expands high-rated movies into their genres, keeps the three
most common genres, re-expands those genres to movies/directors, and ranks directors by
distinct high-rated films in those genres.
"""
import pandas as pd

from conversion_kit import (import_table, manifest_record, matched_reason, pattern_record, score, sqlite_df)

IID = "local230"

REFERENCE_SQL = """
WITH high AS (SELECT m.id AS mid FROM movies m JOIN ratings r ON m.id = r.movie_id WHERE r.avg_rating > 8),
     tg AS (SELECT g.genre, COUNT(DISTINCT g.movie_id) AS c
            FROM genre g JOIN high h ON g.movie_id = h.mid
            GROUP BY g.genre ORDER BY c DESC, g.genre ASC LIMIT 3)
SELECT n.name AS director, COUNT(DISTINCT dm.movie_id) AS movie_count
FROM director_mapping dm
JOIN high h ON dm.movie_id = h.mid
JOIN genre g ON dm.movie_id = g.movie_id
JOIN tg ON g.genre = tg.genre
JOIN names n ON dm.name_id = n.id
GROUP BY n.name
ORDER BY movie_count DESC, n.name ASC
LIMIT 4
"""


def run(reg):
    rec = manifest_record(IID)
    ref = sqlite_df(rec, REFERENCE_SQL)
    ref_v, ref_ok = score(ref, rec)

    m_mov = import_table(reg, rec, IID, "movies", ["id", "title"])
    m_gen = import_table(reg, rec, IID, "genre", ["movie_id", "genre"], index=["movie_id"])
    m_dir = import_table(reg, rec, IID, "director_mapping", ["movie_id", "name_id"], index=["movie_id"])
    m_nam = import_table(reg, rec, IID, "names", ["id", "name"], index=["id"])
    m_rat = import_table(reg, rec, IID, "ratings", ["movie_id", "avg_rating"], index=["movie_id"])
    coll = reg.db[m_mov["collection"]]

    PIPELINE = [
        {"$lookup": {"from": m_rat["collection"], "localField": "id",
                     "foreignField": "movie_id", "as": "r"}},
        {"$unwind": "$r"},
        {"$match": {"r.avg_rating": {"$gt": 8}}},  # movies rated above 8
        {"$lookup": {"from": m_gen["collection"], "localField": "id",
                     "foreignField": "movie_id", "as": "g"}},
        {"$unwind": "$g"},
        {"$group": {"_id": "$g.genre", "mids": {"$addToSet": "$id"}}},
        {"$addFields": {"c": {"$size": "$mids"}}},
        {"$sort": {"c": -1, "_id": 1}},  # ties broken alphabetically, as the gold implies
        {"$limit": 3},
        {"$unwind": "$mids"},
        {"$lookup": {"from": m_dir["collection"], "localField": "mids",
                     "foreignField": "movie_id", "as": "d"}},
        {"$unwind": "$d"},
        {"$lookup": {"from": m_nam["collection"], "localField": "d.name_id",
                     "foreignField": "id", "as": "n"}},
        {"$unwind": "$n"},
        {"$group": {"_id": "$n.name", "mids": {"$addToSet": "$mids"}}},
        {"$addFields": {"movie_count": {"$size": "$mids"}}},
        {"$sort": {"movie_count": -1, "_id": 1}},
        {"$limit": 4},
        {"$project": {"_id": 0, "director": "$_id", "movie_count": 1}},
    ]
    res = list(coll.aggregate(PIPELINE, allowDiskUse=True))
    pred = pd.DataFrame(res, columns=["director", "movie_count"])
    v, ok = score(pred, rec)

    artifact = {
        "pattern": "multi-level top-N (genres then directors) with distinct counts",
        "gold_sql": None, "gold_sql_reproduces_gold_csv": "n/a (no shipped gold SQL)",
        "reference_sql": REFERENCE_SQL, "reference_sql_matches_gold": ref_ok,
        "relational_semantics": ("Movies with avg rating > 8; the 3 most common genres among them; directors' distinct such films "
                                 "in those genres; top 4 (ties broken by name)."),
        "source_schema": {"movies": ["id TEXT", "title TEXT"], "genre": ["movie_id TEXT", "genre TEXT"],
                          "director_mapping": ["movie_id TEXT", "name_id TEXT"], "names": ["id TEXT", "name TEXT"],
                          "ratings": ["movie_id TEXT", "avg_rating REAL"]},
        "mongodb": {"database": "codegen_capstone_pilot", "imports": [m_mov, m_gen, m_dir, m_nam, m_rat],
                    "_id": "default ObjectId", "indexes": ["genre.movie_id", "director_mapping.movie_id",
                                                            "names.id", "ratings.movie_id"]},
        "mql": {"collection": coll.name, "method": "aggregate", "options": {"allowDiskUse": True}, "pipeline": PIPELINE},
        "mongo_result": res,
        "validation": {"per_gold": v, "matched_any_gold": ok},
        "assumptions": [
            "Ties (many genres/directors share the cut-off count) are broken alphabetically: the reference SQL and MQL agree and both match the gold.",
            "Director counts use COUNT(DISTINCT movie_id) so a movie in two selected genres is not double counted.",
        ],
    }
    return ("passed" if ok else "failed"), matched_reason(v, ok), artifact
