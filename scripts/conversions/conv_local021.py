"""local021 - average career runs of strikers who scored >50 in a single match.

No shipped gold SQL: a labelled reference SQL is authored and verified against the gold
CSV first. `ball_by_ball` (striker) is joined to `batsman_scored` (runs_scored) on the
(match_id, over_id, ball_id, innings_no) ball key; a composite key field makes the
$lookup indexable.
"""
import pandas as pd

from conversion_kit import import_table, manifest_record, matched_reason, pattern_record, score, sqlite_df

IID = "local021"

REFERENCE_SQL = """
WITH ms AS (SELECT b.striker AS pid, b.match_id, SUM(bs.runs_scored) AS runs
            FROM ball_by_ball b JOIN batsman_scored bs
              ON b.match_id = bs.match_id AND b.over_id = bs.over_id
             AND b.ball_id = bs.ball_id AND b.innings_no = bs.innings_no
            GROUP BY b.striker, b.match_id),
qual AS (SELECT DISTINCT pid FROM ms WHERE runs > 50),
tot AS (SELECT b.striker AS pid, SUM(bs.runs_scored) AS t
        FROM ball_by_ball b JOIN batsman_scored bs
          ON b.match_id = bs.match_id AND b.over_id = bs.over_id
         AND b.ball_id = bs.ball_id AND b.innings_no = bs.innings_no
        WHERE b.striker IN (SELECT pid FROM qual)
        GROUP BY b.striker)
SELECT AVG(t) AS avg_total_runs FROM tot
"""

KEY_EXPR = {"$concat": [{"$toString": "$match_id"}, "|", {"$toString": "$over_id"},
                        "|", {"$toString": "$ball_id"}, "|", {"$toString": "$innings_no"}]}

PIPELINE_TEMPLATE = [
    {"$addFields": {"join_key": KEY_EXPR}},
    {"$lookup": {"from": "__BS__", "localField": "join_key", "foreignField": "join_key", "as": "bs"}},
    {"$unwind": "$bs"},
    {"$group": {"_id": {"striker": "$striker", "match_id": "$match_id"}, "runs": {"$sum": "$bs.runs_scored"}}},
    {"$group": {"_id": "$_id.striker", "tot": {"$sum": "$runs"}, "max_match": {"$max": "$runs"}}},
    {"$match": {"max_match": {"$gt": 50}}},
    {"$group": {"_id": None, "avg_total_runs": {"$avg": "$tot"}}},
    {"$project": {"_id": 0, "avg_total_runs": 1}},
]


def run(reg):
    rec = manifest_record(IID)
    ref = sqlite_df(rec, REFERENCE_SQL)
    ref_v, ref_ok = score(ref, rec)

    bbb = import_table(reg, rec, IID, "ball_by_ball",
                       ["match_id", "over_id", "ball_id", "innings_no", "striker"], index=["match_id"])
    bs = import_table(reg, rec, IID, "batsman_scored",
                      ["match_id", "over_id", "ball_id", "innings_no", "runs_scored"])
    bs_coll = reg.db[bs["collection"]]
    bs_coll.update_many({}, [{"$set": {"join_key": KEY_EXPR}}])
    bs_coll.create_index("join_key")

    coll = reg.db[bbb["collection"]]
    PIPELINE = []
    for s in PIPELINE_TEMPLATE:
        if "$lookup" in s:
            s = {"$lookup": {**s["$lookup"], "from": bs["collection"]}}
        PIPELINE.append(s)
    res = list(coll.aggregate(PIPELINE, allowDiskUse=True))
    pred = pd.DataFrame(res, columns=["avg_total_runs"])
    v, ok = score(pred, rec)

    artifact = {
        "pattern": "ball-key join + per-match runs + qualify strikers (>50) + average career runs",
        "gold_sql": None, "gold_sql_reproduces_gold_csv": "n/a (no shipped gold SQL)",
        "reference_sql": REFERENCE_SQL, "reference_sql_matches_gold": ref_ok,
        "relational_semantics": ("Sum runs per (striker, match); keep strikers with any single-match score >50; "
                                 "average their total runs across all matches."),
        "source_schema": {"ball_by_ball": ["match_id", "over_id", "ball_id", "innings_no", "striker"],
                          "batsman_scored": ["match_id", "over_id", "ball_id", "innings_no", "runs_scored"]},
        "mongodb": {"database": "codegen_capstone_pilot", "imports": [bbb, bs], "_id": "default ObjectId"},
        "mql": {"collection": coll.name, "method": "aggregate", "options": {"allowDiskUse": True}, "pipeline": PIPELINE},
        "mongo_result": res,
        "validation": {"per_gold": v, "matched_any_gold": ok},
        "assumptions": [
            "A ball key is match_id|over_id|ball_id|innings_no (stored on batsman_scored, added inline on ball_by_ball).",
            "Only balls present in batsman_scored are counted (inner join).",
        ],
    }
    return ("passed" if ok else "failed"), matched_reason(v, ok), artifact
