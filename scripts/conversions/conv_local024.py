"""local024 - top 5 countries by average of players' average runs per match.

No shipped gold SQL: a labelled reference SQL is authored and verified against the gold
CSV first. A player's average runs per match = total runs / matches played (matches with
no scoring contribute zero), then countries average those player averages.
"""
import pandas as pd

from conversion_kit import (import_table, manifest_record, matched_reason, pattern_record, score, sqlite_df)

IID = "local024"

REFERENCE_SQL = """
WITH pr AS (
    SELECT pm.player_id AS pid, pm.match_id, COALESCE(SUM(bs.runs_scored), 0) AS runs
    FROM player_match pm
    LEFT JOIN ball_by_ball b ON b.striker = pm.player_id AND b.match_id = pm.match_id
    LEFT JOIN batsman_scored bs ON b.match_id = bs.match_id AND b.over_id = bs.over_id
         AND b.ball_id = bs.ball_id AND b.innings_no = bs.innings_no
    GROUP BY pm.player_id, pm.match_id
),
pa AS (SELECT pid, AVG(runs) AS a FROM pr GROUP BY pid)
SELECT p.country_name, AVG(pa.a) AS country_batting_average
FROM pa JOIN player p ON p.player_id = pa.pid
GROUP BY p.country_name
ORDER BY country_batting_average DESC
LIMIT 5
"""


def run(reg):
    rec = manifest_record(IID)
    ref = sqlite_df(rec, REFERENCE_SQL)
    ref_v, ref_ok = score(ref, rec)

    m_p = import_table(reg, rec, IID, "player", ["player_id", "country_name"], index=["player_id"])
    m_pm = import_table(reg, rec, IID, "player_match", ["match_id", "player_id"], index=["player_id"])
    m_bbb = import_table(reg, rec, IID, "ball_by_ball",
                         ["match_id", "over_id", "ball_id", "innings_no", "striker"], index=["striker"])
    m_bs = import_table(reg, rec, IID, "batsman_scored",
                        ["match_id", "over_id", "ball_id", "innings_no", "runs_scored"],
                        index=["match_id"])
    coll = reg.db[m_p["collection"]]

    runs_pipe = [
        {"$match": {"$expr": {"$eq": ["$striker", "$$pid"]}}},
        {"$lookup": {"from": m_bs["collection"],
                     "let": {"m": "$match_id", "o": "$over_id", "b": "$ball_id", "i": "$innings_no"},
                     "pipeline": [{"$match": {"$expr": {"$and": [
                         {"$eq": ["$match_id", "$$m"]}, {"$eq": ["$over_id", "$$o"]},
                         {"$eq": ["$ball_id", "$$b"]}, {"$eq": ["$innings_no", "$$i"]}]}}},
                         {"$project": {"runs_scored": 1}}],
                     "as": "bs"}},
        {"$unwind": "$bs"},
        {"$group": {"_id": None, "runs": {"$sum": "$bs.runs_scored"}}},
    ]
    PIPELINE = [
        {"$lookup": {"from": m_pm["collection"], "localField": "player_id",
                     "foreignField": "player_id", "as": "pm"}},
        {"$addFields": {"matches": {"$size": "$pm"}}},
        {"$match": {"matches": {"$gt": 0}}},
        {"$lookup": {"from": m_bbb["collection"], "let": {"pid": "$player_id"},
                     "pipeline": runs_pipe, "as": "rr"}},
        {"$addFields": {"runs": {"$ifNull": [{"$arrayElemAt": ["$rr.runs", 0]}, 0]}}},
        {"$addFields": {"avg_runs": {"$divide": ["$runs", "$matches"]}}},
        {"$group": {"_id": "$country_name", "country_batting_average": {"$avg": "$avg_runs"}}},
        {"$sort": {"country_batting_average": -1}},
        {"$limit": 5},
        {"$project": {"_id": 0, "country_name": "$_id", "country_batting_average": 1}},
    ]
    res = list(coll.aggregate(PIPELINE, allowDiskUse=True))
    pred = pd.DataFrame(res, columns=["country_name", "country_batting_average"])
    v, ok = score(pred, rec)

    artifact = {
        "pattern": "per-player total/matches ratio via nested lookups, then country average",
        "gold_sql": None, "gold_sql_reproduces_gold_csv": "n/a (no shipped gold SQL)",
        "reference_sql": REFERENCE_SQL, "reference_sql_matches_gold": ref_ok,
        "relational_semantics": ("Player average = total runs scored / matches played (all player_match rows); countries ranked by the mean "
                                 "of their players' averages; top 5."),
        "source_schema": {"player": ["player_id", "country_name"], "player_match": ["match_id", "player_id"],
                          "ball_by_ball": ["match_id", "over_id", "ball_id", "innings_no", "striker"],
                          "batsman_scored": ["match_id", "over_id", "ball_id", "innings_no", "runs_scored"]},
        "mongodb": {"database": "codegen_capstone_pilot", "imports": [m_p, m_pm, m_bbb, m_bs], "_id": "default ObjectId",
                    "indexes": ["player.player_id", "player_match.player_id", "ball_by_ball.striker", "batsman_scored.match_id"]},
        "mql": {"collection": coll.name, "method": "aggregate", "options": {"allowDiskUse": True}, "pipeline": PIPELINE},
        "mongo_result": res,
        "validation": {"per_gold": v, "matched_any_gold": ok},
        "assumptions": [
            "Average per match divides by matches played (player_match), so scoreless matches count as 0; this matches gold _a.",
            "Players with no player_match rows are excluded.",
        ],
    }
    return ("passed" if ok else "failed"), matched_reason(v, ok), artifact
