"""local023 -> MongoDB/MQL.

Question: IPL season-5 per-match batting runs, top-5 by average runs per match
plus each player's batting average (runs / dismissals).

The shipped gold SQL does NOT reproduce the accepted CSV at value level; a
reconstructed reference SQL (runs from ball_by_ball JOIN batsman_scored, dismissals
from wicket_taken, both filtered to season 5) reproduces local023_a.csv exactly.
"""
import pandas as pd

from conversion_kit import import_table, manifest_record, score, sqlite_df

IID = "local023"
SEASON5 = "match_id IN (SELECT match_id FROM match WHERE season_id = 5)"

REFERENCE_SQL = """
SELECT p.player_name,
       SUM(pm.runs) * 1.0 / COUNT(pm.match_id) AS avg_runs_per_match,
       SUM(pm.runs) * 1.0 / MAX(COALESCE(d.n_out, 0), 1) AS batting_average
FROM (
    SELECT bb.striker, bb.match_id, SUM(bs.runs_scored) AS runs
    FROM ball_by_ball bb
    JOIN batsman_scored bs
      ON bb.match_id = bs.match_id AND bb.over_id = bs.over_id
     AND bb.ball_id = bs.ball_id AND bb.innings_no = bs.innings_no
    WHERE bb.match_id IN (SELECT match_id FROM match WHERE season_id = 5)
    GROUP BY bb.striker, bb.match_id
) pm
LEFT JOIN (
    SELECT player_out, COUNT(*) AS n_out FROM wicket_taken
    WHERE match_id IN (SELECT match_id FROM match WHERE season_id = 5)
    GROUP BY player_out
) d ON d.player_out = pm.striker
JOIN player p ON p.player_id = pm.striker
GROUP BY pm.striker
ORDER BY avg_runs_per_match DESC
LIMIT 5;
"""


def run(reg):
    rec = manifest_record(IID)
    bb = import_table(reg, rec, IID, "ball_by_ball",
                      ["match_id", "over_id", "ball_id", "innings_no", "striker"], where=SEASON5)
    bs = import_table(reg, rec, IID, "batsman_scored",
                      ["match_id", "over_id", "ball_id", "innings_no", "runs_scored"], where=SEASON5)
    wt = import_table(reg, rec, IID, "wicket_taken", ["player_out"], where=SEASON5)
    pl = import_table(reg, rec, IID, "player", ["player_id", "player_name"])

    pipeline = [
        {"$lookup": {
            "from": bs["collection"],
            "let": {"m": "$match_id", "o": "$over_id", "b": "$ball_id", "i": "$innings_no"},
            "pipeline": [{"$match": {"$expr": {"$and": [
                {"$eq": ["$match_id", "$$m"]}, {"$eq": ["$over_id", "$$o"]},
                {"$eq": ["$ball_id", "$$b"]}, {"$eq": ["$innings_no", "$$i"]}]}}}],
            "as": "bs"}},
        {"$unwind": "$bs"},
        {"$group": {"_id": {"player_id": "$striker", "match_id": "$match_id"},
                    "total_runs": {"$sum": "$bs.runs_scored"}}},
        {"$group": {"_id": "$_id.player_id", "runs": {"$sum": "$total_runs"},
                    "num_matches": {"$sum": 1}}},
        {"$lookup": {"from": wt["collection"], "let": {"pid": "$_id"},
                     "pipeline": [{"$match": {"$expr": {"$eq": ["$player_out", "$$pid"]}}},
                                  {"$count": "n"}], "as": "w"}},
        {"$set": {"n_out": {"$ifNull": [{"$arrayElemAt": ["$w.n", 0]}, 0]}}},
        {"$set": {"avg_runs_per_match": {"$divide": ["$runs", "$num_matches"]},
                  "batting_average": {"$divide": ["$runs", {"$max": ["$n_out", 1]}]}}},
        {"$lookup": {"from": pl["collection"], "localField": "_id",
                     "foreignField": "player_id", "as": "p"}},
        {"$unwind": "$p"},
        {"$sort": {"avg_runs_per_match": -1}},
        {"$limit": 5},
        {"$project": {"_id": 0, "player_name": "$p.player_name",
                      "avg_runs_per_match": 1, "batting_average": 1}},
    ]

    res = list(reg.db[bb["collection"]].aggregate(pipeline))
    pred = pd.DataFrame(res, columns=["player_name", "avg_runs_per_match", "batting_average"])
    verdicts, ok = score(pred, rec)

    ref = sqlite_df(rec, REFERENCE_SQL)
    artifact = {
        "instance_id": IID,
        "reference_sql": REFERENCE_SQL,
        "reference_sql_result": ref.to_dict(orient="records"),
        "reference_sql_matches_gold": score(ref, rec)[1],
        "shipped_gold_sql_reproduces_gold": False,
        "mql": {"method": "aggregate", "collection": bb["collection"], "pipeline": pipeline},
        "result": res,
        "score": verdicts,
        "notes": ("Shipped gold SQL is not value-faithful; reconstructed reference SQL "
                  "(per-match runs + dismissals) reproduces local023_a.csv."),
    }
    return ("passed" if ok else "failed"), ("matched gold" if ok else "no match"), artifact
