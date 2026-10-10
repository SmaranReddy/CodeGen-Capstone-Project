"""local025 - each match's single most expensive over, with bowler and the across-match average.

No shipped gold SQL: a labelled reference SQL is authored and verified against the gold
CSV first. Runs in an over = batsman runs + extras; one row per match (tie-break by
innings_no then over_id); bowler name is a left join (5 matches have no player row).
"""
import pandas as pd

from conversion_kit import import_table, manifest_record, matched_reason, pattern_record, score, sqlite_df

IID = "local025"

REFERENCE_SQL = """
WITH overs AS (SELECT DISTINCT match_id, innings_no, over_id FROM ball_by_ball),
bs AS (SELECT match_id, innings_no, over_id, SUM(runs_scored) r FROM batsman_scored GROUP BY 1,2,3),
ex AS (SELECT match_id, innings_no, over_id, SUM(extra_runs) e FROM extra_runs GROUP BY 1,2,3),
ov AS (SELECT o.match_id, o.innings_no, o.over_id, COALESCE(bs.r,0)+COALESCE(ex.e,0) AS runs
       FROM overs o LEFT JOIN bs ON bs.match_id=o.match_id AND bs.innings_no=o.innings_no AND bs.over_id=o.over_id
                     LEFT JOIN ex ON ex.match_id=o.match_id AND ex.innings_no=o.innings_no AND ex.over_id=o.over_id),
mx AS (SELECT match_id, MAX(runs) mr FROM ov GROUP BY match_id),
top AS (SELECT o.match_id, o.innings_no, o.over_id, o.runs,
               ROW_NUMBER() OVER (PARTITION BY o.match_id ORDER BY o.runs DESC, o.innings_no, o.over_id) rn
        FROM ov o WHERE o.runs = (SELECT mr FROM mx WHERE mx.match_id=o.match_id)),
bowl AS (SELECT match_id, innings_no, over_id, bowler,
                ROW_NUMBER() OVER (PARTITION BY match_id, innings_no, over_id ORDER BY ball_id) rn
         FROM ball_by_ball),
avg AS (SELECT AVG(runs) a FROM top WHERE rn=1)
SELECT t.match_id, t.innings_no, t.over_id, b.bowler AS bowler_id, p.player_name AS bowler_name,
       t.runs AS highest_over_runs, (SELECT a FROM avg) AS avg_highest_over_runs_across_matches
FROM top t
LEFT JOIN bowl b ON b.match_id=t.match_id AND b.innings_no=t.innings_no AND b.over_id=t.over_id AND b.rn=1
LEFT JOIN player p ON p.player_id = b.bowler
WHERE t.rn=1 ORDER BY t.match_id
"""

KEY_EXPR = {"$concat": [{"$toString": "$match_id"}, "|", {"$toString": "$over_id"},
                        "|", {"$toString": "$ball_id"}, "|", {"$toString": "$innings_no"}]}

PIPELINE_TEMPLATE = [
    {"$addFields": {"join_key": KEY_EXPR}},
    {"$lookup": {"from": "__BS__", "localField": "join_key", "foreignField": "join_key", "as": "bs"}},
    {"$lookup": {"from": "__EX__", "localField": "join_key", "foreignField": "join_key", "as": "ex"}},
    {"$project": {"match_id": 1, "innings_no": 1, "over_id": 1, "ball_id": 1, "bowler": 1,
                  "runs": {"$add": [{"$sum": "$bs.runs_scored"}, {"$sum": "$ex.extra_runs"}]}}},
    {"$group": {"_id": {"m": "$match_id", "i": "$innings_no", "o": "$over_id"},
                "runs": {"$sum": "$runs"},
                "bw": {"$top": {"sortBy": {"ball_id": 1}, "output": "$bowler"}}}},
    {"$project": {"runs": 1, "bw": 1, "_id": 1}},
    {"$group": {"_id": "$_id.m", "mr": {"$max": "$runs"},
                "cands": {"$push": {"i": "$_id.i", "o": "$_id.o", "bw": "$bw", "runs": "$runs"}}}},
    {"$set": {"top": {"$first": {"$sortArray": {
        "input": {"$filter": {"input": "$cands", "as": "c", "cond": {"$eq": ["$$c.runs", "$mr"]}}},
        "sortBy": {"runs": -1, "i": 1, "o": 1}}}}}},
    {"$project": {"match_id": "$_id", "innings_no": "$top.i", "over_id": "$top.o",
                  "bowler_id": "$top.bw", "runs": "$top.runs"}},
    {"$group": {"_id": None, "avg": {"$avg": "$runs"}, "rows": {"$push": "$$ROOT"}}},
    {"$unwind": "$rows"},
    {"$lookup": {"from": "__PLAYER__", "localField": "rows.bowler_id", "foreignField": "player_id", "as": "p"}},
    {"$project": {"_id": 0, "match_id": "$rows.match_id", "innings_no": "$rows.innings_no",
                  "over_id": "$rows.over_id", "bowler_id": "$rows.bowler_id",
                  "bowler_name": {"$arrayElemAt": ["$p.player_name", 0]},
                  "highest_over_runs": "$rows.runs", "avg_highest_over_runs_across_matches": "$avg"}},
    {"$sort": {"match_id": 1}},
]

COLS = ["match_id", "innings_no", "over_id", "bowler_id", "bowler_name",
        "highest_over_runs", "avg_highest_over_runs_across_matches"]


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
    pred = pd.DataFrame(res, columns=COLS)
    v, ok = score(pred, rec)

    artifact = {
        "pattern": "ball key join (runs + extras) + per-match argmax over + across-match average",
        "gold_sql": None, "gold_sql_reproduces_gold_csv": "n/a (no shipped gold SQL)",
        "reference_sql": REFERENCE_SQL, "reference_sql_matches_gold": ref_ok,
        "relational_semantics": ("Per (match, innings, over) sum batsman runs plus extras; keep each match's single most "
                                 "expensive over; return bowler and the average of those maxima across matches."),
        "source_schema": {"ball_by_ball": ["match_id", "innings_no", "over_id", "ball_id", "bowler"],
                          "batsman_scored": ["match_id", "innings_no", "over_id", "ball_id", "runs_scored"],
                          "extra_runs": ["match_id", "innings_no", "over_id", "ball_id", "extra_runs"],
                          "player": ["player_id", "player_name"]},
        "mongodb": {"database": "codegen_capstone_pilot", "imports": [bbb, bs, ex, player], "_id": "default ObjectId"},
        "mql": {"collection": coll.name, "method": "aggregate", "options": {"allowDiskUse": True}, "pipeline": PIPELINE},
        "mongo_result": res,
        "validation": {"per_gold": v, "matched_any_gold": ok},
        "assumptions": ["Runs in an over = batsman_scored.runs_scored + extra_runs.extra_runs.",
                        "One row per match; ties broken by innings_no then over_id.",
                        "Bowler name is a left join (5 matches have no player row).",
                        "Output columns follow gold variant _a."],
    }
    return ("passed" if ok else "failed"), matched_reason(v, ok), artifact
