"""local221 - top 10 teams by number of match wins across all leagues.

No shipped gold SQL: a labelled reference SQL is authored and verified against the gold
CSV first. A win is home_team_goal > away_team_goal (home) or the reverse (away).
"""
import pandas as pd

from conversion_kit import (import_table, manifest_record, matched_reason, pattern_record, score, sqlite_df)

IID = "local221"

REFERENCE_SQL = """
WITH w AS (
    SELECT home_team_api_id AS tid FROM Match WHERE home_team_goal > away_team_goal
    UNION ALL
    SELECT away_team_api_id FROM Match WHERE away_team_goal > home_team_goal
)
SELECT tm.team_long_name, COUNT(*) AS TOTAL_WINS
FROM w JOIN Team tm ON tm.team_api_id = w.tid
GROUP BY tm.team_long_name
ORDER BY TOTAL_WINS DESC
LIMIT 10
"""


def run(reg):
    rec = manifest_record(IID)
    ref = sqlite_df(rec, REFERENCE_SQL)
    ref_v, ref_ok = score(ref, rec)

    m_m = import_table(reg, rec, IID, "Match",
                       ["home_team_api_id", "away_team_api_id", "home_team_goal", "away_team_goal"])
    m_t = import_table(reg, rec, IID, "Team", ["team_api_id", "team_long_name"], index=["team_api_id"])
    coll = reg.db[m_m["collection"]]

    PIPELINE = [
        {"$facet": {
            "hw": [{"$match": {"$expr": {"$gt": ["$home_team_goal", "$away_team_goal"]}}},
                   {"$project": {"tid": "$home_team_api_id"}}],
            "aw": [{"$match": {"$expr": {"$gt": ["$away_team_goal", "$home_team_goal"]}}},
                   {"$project": {"tid": "$away_team_api_id"}}]}},
        {"$project": {"all": {"$concatArrays": ["$hw", "$aw"]}}},
        {"$unwind": "$all"},
        {"$group": {"_id": "$all.tid", "TOTAL_WINS": {"$sum": 1}}},
        {"$sort": {"TOTAL_WINS": -1}},
        {"$limit": 10},
        {"$lookup": {"from": m_t["collection"], "localField": "_id",
                     "foreignField": "team_api_id", "as": "t"}},
        {"$unwind": "$t"},
        {"$project": {"_id": 0, "team_long_name": "$t.team_long_name", "TOTAL_WINS": 1}},
    ]
    res = list(coll.aggregate(PIPELINE, allowDiskUse=True))
    pred = pd.DataFrame(res, columns=["team_long_name", "TOTAL_WINS"])
    v, ok = score(pred, rec)

    artifact = {
        "pattern": "self-union of two conditional projections + group top-N",
        "gold_sql": None, "gold_sql_reproduces_gold_csv": "n/a (no shipped gold SQL)",
        "reference_sql": REFERENCE_SQL, "reference_sql_matches_gold": ref_ok,
        "relational_semantics": "Count wins per team (home or away) across all matches; the 10 teams with the most wins.",
        "source_schema": {"Match": ["home_team_api_id", "away_team_api_id", "home_team_goal", "away_team_goal"],
                          "Team": ["team_api_id", "team_long_name"]},
        "mongodb": {"database": "codegen_capstone_pilot", "imports": [m_m, m_t], "_id": "default ObjectId",
                    "indexes": ["Team.team_api_id"]},
        "mql": {"collection": coll.name, "method": "aggregate", "options": {"allowDiskUse": True}, "pipeline": PIPELINE},
        "mongo_result": res,
        "validation": {"per_gold": v, "matched_any_gold": ok},
        "assumptions": [
            "Wins are counted from goal comparisons only (no other win definition).",
            "The gold lists a strict descending order with no cut-off tie, so LIMIT 10 is unambiguous.",
        ],
    }
    return ("passed" if ok else "failed"), matched_reason(v, ok), artifact
