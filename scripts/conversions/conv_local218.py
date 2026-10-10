"""local218 - median of each team's highest single-season goal total.

No shipped gold SQL: a labelled reference SQL is authored and verified against the gold
CSV first. Goals are attributed to both the home and away team per match.
"""
import pandas as pd

from conversion_kit import (import_table, manifest_record, matched_reason, pattern_record, score, sqlite_df)

IID = "local218"

REFERENCE_SQL = """
WITH g AS (
    SELECT home_team_api_id AS tid, season, home_team_goal AS gg FROM Match
    UNION ALL
    SELECT away_team_api_id, season, away_team_goal FROM Match
),
ts AS (SELECT tid, season, SUM(gg) AS sg FROM g GROUP BY tid, season),
mx AS (SELECT tid, MAX(sg) AS m FROM ts GROUP BY tid),
v AS (SELECT m, ROW_NUMBER() OVER (ORDER BY m) AS rn, COUNT(*) OVER () AS n FROM mx)
SELECT AVG(m) AS MEDIAN_HIGHEST_SEASON_GOALS
FROM v WHERE rn IN ((n + 1) / 2, (n + 2) / 2)
"""


def run(reg):
    rec = manifest_record(IID)
    ref = sqlite_df(rec, REFERENCE_SQL)
    ref_v, ref_ok = score(ref, rec)

    m_m = import_table(reg, rec, IID, "Match",
                       ["home_team_api_id", "away_team_api_id", "home_team_goal", "away_team_goal", "season"])
    coll = reg.db[m_m["collection"]]

    lo = {"$floor": {"$divide": [{"$subtract": ["$$n", 1]}, 2]}}
    hi = {"$ceil": {"$divide": [{"$subtract": ["$$n", 1]}, 2]}}
    PIPELINE = [
        {"$project": {"pairs": [
            {"tid": "$home_team_api_id", "s": "$season", "g": "$home_team_goal"},
            {"tid": "$away_team_api_id", "s": "$season", "g": "$away_team_goal"}]}},
        {"$unwind": "$pairs"},
        {"$group": {"_id": {"tid": "$pairs.tid", "s": "$pairs.s"}, "sg": {"$sum": "$pairs.g"}}},
        {"$group": {"_id": "$_id.tid", "mx": {"$max": "$sg"}}},
        {"$sort": {"mx": 1}},
        {"$group": {"_id": None, "vals": {"$push": "$mx"}}},
        {"$project": {"_id": 0, "MEDIAN_HIGHEST_SEASON_GOALS": {"$let": {
            "vars": {"a": "$vals", "n": {"$size": "$vals"}},
            "in": {"$avg": [{"$arrayElemAt": ["$$a", lo]}, {"$arrayElemAt": ["$$a", hi]}]}}}}},
    ]
    res = list(coll.aggregate(PIPELINE, allowDiskUse=True))
    pred = pd.DataFrame(res, columns=["MEDIAN_HIGHEST_SEASON_GOALS"])
    v, ok = score(pred, rec)

    artifact = {
        "pattern": "attribution to two roles + grouped max + even/odd median",
        "gold_sql": None, "gold_sql_reproduces_gold_csv": "n/a (no shipped gold SQL)",
        "reference_sql": REFERENCE_SQL, "reference_sql_matches_gold": ref_ok,
        "relational_semantics": ("Per team and season sum goals scored; per team keep its maximum; median across teams "
                                 "(average of the two middle values if the team count is even)."),
        "source_schema": {"Match": ["home_team_api_id", "away_team_api_id", "home_team_goal", "away_team_goal", "season"]},
        "mongodb": {"database": "codegen_capstone_pilot", "imports": [m_m], "_id": "default ObjectId"},
        "mql": {"collection": coll.name, "method": "aggregate", "options": {"allowDiskUse": True}, "pipeline": PIPELINE},
        "mongo_result": res,
        "validation": {"per_gold": v, "matched_any_gold": ok},
        "assumptions": [
            "season as stored is the season label; goals are summed for home and away appearances of each team.",
            "299 teams (odd) yields the single middle value 48.",
        ],
    }
    return ("passed" if ok else "failed"), matched_reason(v, ok), artifact
