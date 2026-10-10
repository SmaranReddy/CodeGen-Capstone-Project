"""local283 - champion team (3-1-0 points) per season and league across countries.

No shipped gold SQL: a labelled reference SQL is authored and verified against the gold
CSV first. Each match is exploded into two team-rows; ties are broken by team api id.
"""
import pandas as pd

from conversion_kit import import_table, manifest_record, matched_reason, pattern_record, score, sqlite_df

IID = "local283"

REFERENCE_SQL = """
WITH res AS (
  SELECT m.season, m.league_id, m.home_team_api_id AS tid,
         CASE WHEN m.home_team_goal > m.away_team_goal THEN 3 WHEN m.home_team_goal = m.away_team_goal THEN 1 ELSE 0 END AS pts
  FROM "Match" m
  UNION ALL
  SELECT m.season, m.league_id, m.away_team_api_id,
         CASE WHEN m.away_team_goal > m.home_team_goal THEN 3 WHEN m.home_team_goal = m.away_team_goal THEN 1 ELSE 0 END
  FROM "Match" m),
s AS (SELECT season, league_id, tid, SUM(pts) AS pts FROM res GROUP BY season, league_id, tid),
rk AS (SELECT *, ROW_NUMBER() OVER (PARTITION BY season, league_id ORDER BY pts DESC, tid) AS rn FROM s)
SELECT rk.season, t.team_long_name AS champion_team, l.name AS league, c.name AS country, rk.pts AS total_points
FROM rk JOIN Team t ON t.team_api_id = rk.tid JOIN League l ON l.id = rk.league_id JOIN Country c ON c.id = l.country_id
WHERE rk.rn = 1 ORDER BY rk.season, rk.league_id
"""

PTS_EXPR = {"$cond": [
    {"$or": [{"$eq": ["$rows.gf", None]}, {"$eq": ["$rows.ga", None]}]}, 0,
    {"$cond": [{"$gt": ["$rows.gf", "$rows.ga"]}, 3,
               {"$cond": [{"$eq": ["$rows.gf", "$rows.ga"]}, 1, 0]}]}]}

PIPELINE_TEMPLATE = [
    {"$project": {"season": 1, "league_id": 1,
                  "rows": [{"tid": "$home_team_api_id", "gf": "$home_team_goal", "ga": "$away_team_goal"},
                           {"tid": "$away_team_api_id", "gf": "$away_team_goal", "ga": "$home_team_goal"}]}},
    {"$unwind": "$rows"},
    {"$project": {"season": 1, "league_id": 1, "tid": "$rows.tid", "pts": PTS_EXPR}},
    {"$group": {"_id": {"s": "$season", "l": "$league_id", "t": "$tid"}, "pts": {"$sum": "$pts"}}},
    {"$sort": {"_id.s": 1, "_id.l": 1, "pts": -1, "_id.t": 1}},
    {"$group": {"_id": {"s": "$_id.s", "l": "$_id.l"}, "champ": {"$first": "$$ROOT"}}},
    {"$project": {"season": "$_id.s", "league_id": "$_id.l", "tid": "$champ._id.t", "total_points": "$champ.pts"}},
    {"$lookup": {"from": "__TEAM__", "localField": "tid", "foreignField": "team_api_id", "as": "t"}},
    {"$unwind": "$t"},
    {"$lookup": {"from": "__LEAGUE__", "localField": "league_id", "foreignField": "id", "as": "l"}},
    {"$unwind": "$l"},
    {"$lookup": {"from": "__COUNTRY__", "localField": "l.country_id", "foreignField": "id", "as": "c"}},
    {"$unwind": "$c"},
    {"$project": {"_id": 0, "season": 1, "champion_team": "$t.team_long_name", "league": "$l.name",
                  "country": "$c.name", "total_points": 1}},
    {"$sort": {"season": 1, "league": 1}},
]


def run(reg):
    rec = manifest_record(IID)
    ref = sqlite_df(rec, REFERENCE_SQL)
    ref_v, ref_ok = score(ref, rec)

    match = import_table(reg, rec, IID, "Match",
                         ["season", "league_id", "home_team_api_id", "away_team_api_id", "home_team_goal", "away_team_goal"])
    league = import_table(reg, rec, IID, "League", ["id", "country_id", "name"], index=["id"])
    country = import_table(reg, rec, IID, "Country", ["id", "name"], index=["id"])
    team = import_table(reg, rec, IID, "Team", ["team_api_id", "team_long_name"], index=["team_api_id"])

    coll = reg.db[match["collection"]]
    repl = {"__TEAM__": team["collection"], "__LEAGUE__": league["collection"], "__COUNTRY__": country["collection"]}
    PIPELINE = []
    for s in PIPELINE_TEMPLATE:
        if "$lookup" in s:
            s = {"$lookup": {**s["$lookup"], "from": repl[s["$lookup"]["from"]]}}
        PIPELINE.append(s)
    res = list(coll.aggregate(PIPELINE, allowDiskUse=True))
    pred = pd.DataFrame(res, columns=["season", "champion_team", "league", "country", "total_points"])
    v, ok = score(pred, rec)

    artifact = {
        "pattern": "explode matches into team rows + 3-1-0 tally + per-group champion via sort/$first",
        "gold_sql": None, "gold_sql_reproduces_gold_csv": "n/a (no shipped gold SQL)",
        "reference_sql": REFERENCE_SQL, "reference_sql_matches_gold": ref_ok,
        "relational_semantics": ("For each season and league award 3/1/0 per match to home and away teams, take the "
                                 "maximum-points team as champion, and report team/league/country/points."),
        "source_schema": {"Match": ["season", "league_id", "home_team_api_id", "away_team_api_id", "home_team_goal", "away_team_goal"],
                          "League": ["id", "country_id", "name"], "Country": ["id", "name"], "Team": ["team_api_id", "team_long_name"]},
        "mongodb": {"database": "codegen_capstone_pilot", "imports": [match, league, country, team], "_id": "default ObjectId"},
        "mql": {"collection": coll.name, "method": "aggregate", "options": {"allowDiskUse": True}, "pipeline": PIPELINE},
        "mongo_result": res,
        "validation": {"per_gold": v, "matched_any_gold": ok},
        "assumptions": [
            "Matches with a NULL goal on either side award 0 points (matching SQL CASE NULL handling).",
            "One champion per season/league; ties broken by team_api_id.",
        ],
    }
    return ("passed" if ok else "failed"), matched_reason(v, ok), artifact
