"""local022 - CTEs + composite-key join + GROUP BY/HAVING + conditional join to the losing team + DISTINCT."""
import sys, time; sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import pandas as pd
from harness import *

IID = "local022"
rec = manifest_record(IID)
gold_sql = (ROOT / rec["gold_sql_path"]).read_text(encoding="utf-8")
ref = sqlite_df(rec, gold_sql); ref_v, ref_ok = score(ref, rec)
sq = sqlite_ro(rec)
K = "match_id||'-'||over_id||'-'||ball_id||'-'||innings_no"
pre = {
  "ball_by_ball_key_unique": sq.execute(f"SELECT COUNT(*)=COUNT(DISTINCT {K}) FROM ball_by_ball").fetchone()[0],
  "batsman_scored_key_unique": sq.execute(f"SELECT COUNT(*)=COUNT(DISTINCT {K}) FROM batsman_scored").fetchone()[0],
  "match_winner_nulls": sq.execute("SELECT COUNT(*) FROM match WHERE match_winner IS NULL").fetchone()[0],
  "player_match_unique_match_player": sq.execute("SELECT COUNT(*)=COUNT(DISTINCT match_id||'-'||player_id) FROM player_match").fetchone()[0],
}

db = mongo_db()
imports = [
  import_table(db, rec, IID, "ball_by_ball", ["match_id", "over_id", "ball_id", "innings_no", "striker"]),
  import_table(db, rec, IID, "batsman_scored", ["match_id", "over_id", "ball_id", "innings_no", "runs_scored"]),
  import_table(db, rec, IID, "match", ["match_id", "team_1", "team_2", "match_winner"]),
  import_table(db, rec, IID, "player_match", ["match_id", "player_id", "team_id"]),
  import_table(db, rec, IID, "player", ["player_id", "player_name"]),
]
C = lambda t: f"{IID}__{t}"
KEY = {k: f"${k}" for k in ["match_id", "over_id", "ball_id", "innings_no"]}
keyproj = {k: 1 for k in KEY}

# JOIN ON 4 columns. $lookup only takes one key, so use the 'group both sides on the composite key' pattern:
# push each side's values per key, then $unwind both arrays -> exact inner-join semantics even for non-unique keys.
PIPELINE = [
  {"$project": {"_id": 0, **keyproj, "striker": 1, "side": {"$literal": "L"}}},
  {"$unionWith": {"coll": C("batsman_scored"), "pipeline": [{"$project": {"_id": 0, **keyproj, "runs_scored": 1, "side": {"$literal": "R"}}}]}},
  {"$group": {"_id": KEY,
              "L": {"$push": {"$cond": [{"$eq": ["$side", "L"]}, "$striker", "$$REMOVE"]}},
              "R": {"$push": {"$cond": [{"$eq": ["$side", "R"]}, "$runs_scored", "$$REMOVE"]}}}},
  {"$unwind": "$L"}, {"$unwind": "$R"},                                           # inner join: both sides present
  {"$group": {"_id": {"player_id": "$L", "match_id": "$_id.match_id"}, "total_runs": {"$sum": "$R"}}},
  {"$match": {"total_runs": {"$gte": 100}}},                                      # HAVING SUM(...) >= 100
  {"$lookup": {"from": C("match"), "localField": "_id.match_id", "foreignField": "match_id", "as": "m"}},
  {"$unwind": "$m"},
  {"$addFields": {"loser": {"$cond": [{"$eq": ["$m.match_winner", "$m.team_1"]}, "$m.team_2", "$m.team_1"]}}},
  {"$lookup": {"from": C("player_match"),
               "let": {"p": "$_id.player_id", "mid": "$_id.match_id", "t": "$loser"},
               "pipeline": [{"$match": {"$expr": {"$and": [{"$eq": ["$player_id", "$$p"]}, {"$eq": ["$match_id", "$$mid"]}, {"$eq": ["$team_id", "$$t"]}]}}}],
               "as": "pm"}},
  {"$unwind": "$pm"},                                                              # inner join to the losing-team row
  {"$lookup": {"from": C("player"), "localField": "_id.player_id", "foreignField": "player_id", "as": "p"}},
  {"$unwind": "$p"},
  {"$group": {"_id": "$p.player_name"}},                                           # SELECT DISTINCT player_name
  {"$sort": {"_id": 1}},
  {"$project": {"_id": 0, "player_name": "$_id"}},
]
t0 = time.time()
res = list(db[C("ball_by_ball")].aggregate(PIPELINE, allowDiskUse=True))
secs = round(time.time() - t0, 1)
pred = pd.DataFrame(res)
v, ok = score(pred, rec)
print(IID, "ref_sql_ok", ref_ok, "| mql_ok", ok, f"{secs}s", len(res), "rows", v)

write_artifact(IID, {
  "instance_id": IID, "db": rec["db"], "question": rec["question"], "pattern": "CTE + composite-key join + GROUP BY/HAVING + derived loser + DISTINCT",
  "gold_sql": gold_sql, "gold_sql_reproduces_gold_csv": ref_ok, "prechecks": pre,
  "relational_semantics": "Per (striker, match) sum runs over ball_by_ball JOIN batsman_scored on (match,over,ball,innings); keep >=100; loser = team_1 if winner != team_1 else team_2; keep players whose player_match team_id is the loser; distinct names sorted.",
  "mongodb": {"database": PILOT_DB, "imports": imports, "indexes": [], "_id": "default ObjectId",
              "join_strategy": "composite-key join via $unionWith + $group(push L,R) + double $unwind; later single-key joins via $lookup; 3-condition join via $lookup let/pipeline on the few surviving rows",
              "seconds_to_execute_on_atlas": secs},
  "mql": {"collection": C("ball_by_ball"), "method": "aggregate", "options": {"allowDiskUse": True}, "pipeline": PIPELINE},
  "mongo_result_rows": len(res), "mongo_result": res, "gold": v, "validation_passed": ok,
  "rules_discovered": [
    "Multi-column equi-join: $lookup takes one field. Use $unionWith + $group on the composite key with $push per side + $unwind both sides; this is exact for non-unique keys (cross product per key) and needs no index.",
    "CTE chain -> consecutive pipeline stages; each CTE boundary is usually a $group or $lookup.",
    "HAVING -> $match after $group.",
    "SELECT DISTINCT col -> $group by that column (dedups by the *value*, as SQL does, so equal names of different ids collapse).",
    "Join that needs a conditional/derived key (loser team) -> $addFields first, then $lookup with let/pipeline when more than one key is required; do this only after rows were already reduced.",
  ]})
