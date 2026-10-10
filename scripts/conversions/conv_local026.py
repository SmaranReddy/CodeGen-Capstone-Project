"""local026 - top 3 highest-run overs among each match's most expensive over.

No shipped gold SQL: a labelled reference SQL is authored and verified against the gold
CSV first. Runs conceded in an over = batsman runs + extras. A composite ball key makes
the two $lookups indexable.
"""
import pandas as pd

from conversion_kit import import_table, manifest_record, matched_reason, pattern_record, score, sqlite_df

IID = "local026"

REFERENCE_SQL = """
WITH over_runs AS (
  SELECT b.match_id, b.innings_no, b.over_id, b.bowler,
         SUM(COALESCE(bs.runs_scored, 0) + COALESCE(ex.extra_runs, 0)) AS runs
  FROM ball_by_ball b
  LEFT JOIN batsman_scored bs ON b.match_id = bs.match_id AND b.over_id = bs.over_id
       AND b.ball_id = bs.ball_id AND b.innings_no = bs.innings_no
  LEFT JOIN extra_runs ex ON b.match_id = ex.match_id AND b.over_id = ex.over_id
       AND b.ball_id = ex.ball_id AND b.innings_no = ex.innings_no
  GROUP BY b.match_id, b.innings_no, b.over_id, b.bowler),
match_max AS (SELECT match_id, MAX(runs) AS mr FROM over_runs GROUP BY match_id),
tom AS (SELECT o.* FROM over_runs o JOIN match_max m ON m.match_id = o.match_id AND m.mr = o.runs)
SELECT p.player_name AS bowler_name, t.match_id, t.runs AS runs_in_over
FROM tom t JOIN player p ON p.player_id = t.bowler
ORDER BY t.runs DESC LIMIT 3
"""

KEY_EXPR = {"$concat": [{"$toString": "$match_id"}, "|", {"$toString": "$over_id"},
                        "|", {"$toString": "$ball_id"}, "|", {"$toString": "$innings_no"}]}

PIPELINE_TEMPLATE = [
    {"$addFields": {"join_key": KEY_EXPR}},
    {"$lookup": {"from": "__BS__", "localField": "join_key", "foreignField": "join_key", "as": "bs"}},
    {"$lookup": {"from": "__EX__", "localField": "join_key", "foreignField": "join_key", "as": "ex"}},
    {"$project": {"match_id": 1, "innings_no": 1, "over_id": 1, "bowler": 1,
                  "runs": {"$add": [{"$sum": "$bs.runs_scored"}, {"$sum": "$ex.extra_runs"}]}}},
    {"$group": {"_id": {"m": "$match_id", "i": "$innings_no", "o": "$over_id", "bw": "$bowler"}, "runs": {"$sum": "$runs"}}},
    {"$group": {"_id": "$_id.m", "mr": {"$max": "$runs"}, "rows": {"$push": "$$ROOT"}}},
    {"$unwind": "$rows"},
    {"$match": {"$expr": {"$eq": ["$rows.runs", "$mr"]}}},
    {"$sort": {"rows.runs": -1}},
    {"$limit": 3},
    {"$lookup": {"from": "__PLAYER__", "localField": "rows._id.bw", "foreignField": "player_id", "as": "p"}},
    {"$unwind": "$p"},
    {"$project": {"_id": 0, "bowler_name": "$p.player_name", "match_id": "$rows._id.m", "runs_in_over": "$rows.runs"}},
]


def run(reg):
    rec = manifest_record(IID)
    ref = sqlite_df(rec, REFERENCE_SQL)
    ref_v, ref_ok = score(ref, rec)

    bbb = import_table(reg, rec, IID, "ball_by_ball", ["match_id", "innings_no", "over_id", "ball_id", "bowler"], index=["match_id"])
    bs = import_table(reg, rec, IID, "batsman_scored", ["match_id", "innings_no", "over_id", "ball_id", "runs_scored"])
    ex = import_table(reg, rec, IID, "extra_runs", ["match_id", "innings_no", "over_id", "ball_id", "extra_runs"])
    player = import_table(reg, rec, IID, "player", ["player_id", "player_name"], index=["player_id"])
    for info in (bs, ex):
        c = reg.db[info["collection"]]
        c.update_many({}, [{"$set": {"join_key": KEY_EXPR}}])
        c.create_index("join_key")

    coll = reg.db[bbb["collection"]]
    repl = {"__BS__": bs["collection"], "__EX__": ex["collection"], "__PLAYER__": player["collection"]}
    PIPELINE = []
    for s in PIPELINE_TEMPLATE:
        if "$lookup" in s:
            s = {"$lookup": {**s["$lookup"], "from": repl[s["$lookup"]["from"]]}}
        PIPELINE.append(s)
    res = list(coll.aggregate(PIPELINE, allowDiskUse=True))
    pred = pd.DataFrame(res, columns=["bowler_name", "match_id", "runs_in_over"])
    v, ok = score(pred, rec)

    artifact = {
        "pattern": "ball key join (runs + extras) + match-max over + global top-3",
        "gold_sql": None, "gold_sql_reproduces_gold_csv": "n/a (no shipped gold SQL)",
        "reference_sql": REFERENCE_SQL, "reference_sql_matches_gold": ref_ok,
        "relational_semantics": ("Per (match, innings, over, bowler) sum batsman runs plus extras; keep each match's most "
                                 "expensive over; return the 3 highest such overs with bowler name and match id."),
        "source_schema": {"ball_by_ball": ["match_id", "innings_no", "over_id", "ball_id", "bowler"],
                          "batsman_scored": ["match_id", "innings_no", "over_id", "ball_id", "runs_scored"],
                          "extra_runs": ["match_id", "innings_no", "over_id", "ball_id", "extra_runs"],
                          "player": ["player_id", "player_name"]},
        "mongodb": {"database": "codegen_capstone_pilot", "imports": [bbb, bs, ex, player], "_id": "default ObjectId"},
        "mql": {"collection": coll.name, "method": "aggregate", "options": {"allowDiskUse": True}, "pipeline": PIPELINE},
        "mongo_result": res,
        "validation": {"per_gold": v, "matched_any_gold": ok},
        "assumptions": [
            "Runs conceded in an over = batsman_scored.runs_scored + extra_runs.extra_runs.",
            "An over is keyed by match_id, innings_no, over_id; only each match's maximum over is considered.",
            "The final result is the 3 highest-scoring such overs across all matches.",
        ],
    }
    return ("passed" if ok else "failed"), matched_reason(v, ok), artifact
