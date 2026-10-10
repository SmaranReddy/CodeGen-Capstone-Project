"""local008 - baseball players topping games played / runs / hits / home runs.

SUM the batting counters per player, then take the single max of each metric (4 rows).
condition_cols point at the numeric column for every gold (a=[2], b=[2], c=[1]).
"""
import pandas as pd

from conversion_kit import (import_table, manifest_record, matched_reason, pattern_record, score, sqlite_df)

IID = "local008"


def _max_branch(field, metric):
    return [{"$sort": {field: -1}},
            {"$limit": 1},
            {"$project": {"_id": 0, "metric": {"$literal": metric},
                          "name": "$name_given", "value": f"${field}"}}]


def run(reg):
    rec = manifest_record(IID)
    gold_sql = open(rec["gold_sql_path"], encoding="utf-8").read()
    ref = sqlite_df(rec, gold_sql)
    ref_v, ref_ok = score(ref, rec)

    m_bat = import_table(reg, rec, IID, "batting", ["player_id", "g", "r", "h", "hr"])
    m_pl = import_table(reg, rec, IID, "player", ["player_id", "name_given"], index=["player_id"])
    coll = reg.db[m_bat["collection"]]

    PIPELINE = [
        {"$group": {"_id": "$player_id", "g": {"$sum": "$g"}, "r": {"$sum": "$r"},
                    "h": {"$sum": "$h"}, "hr": {"$sum": "$hr"}}},
        {"$lookup": {"from": m_pl["collection"], "localField": "_id",
                     "foreignField": "player_id", "as": "p"}},
        {"$unwind": "$p"},
        {"$addFields": {"name_given": "$p.name_given"}},
        {"$facet": {
            "gp": _max_branch("g", "games_played"),
            "runs": _max_branch("r", "runs"),
            "hits": _max_branch("h", "hits"),
            "hr": _max_branch("hr", "home_runs")}},
        {"$project": {"rows": {"$concatArrays": ["$gp", "$runs", "$hits", "$hr"]}}},
        {"$unwind": "$rows"},
        {"$replaceRoot": {"newRoot": "$rows"}},
        {"$project": {"_id": 0, "METRIC_NAME": "$metric", "PLAYER_GIVEN_NAME": "$name",
                      "SCORE_VALUE": "$value"}},
    ]
    res = list(coll.aggregate(PIPELINE))
    pred = pd.DataFrame(res, columns=["METRIC_NAME", "PLAYER_GIVEN_NAME", "SCORE_VALUE"])
    v, ok = score(pred, rec)

    artifact = {
        "pattern": "group sums + per-metric global max ($facet of sort+limit)",
        "gold_sql": gold_sql, "gold_sql_reproduces_gold_csv": ref_ok,
        "reference_sql": gold_sql, "reference_sql_matches_gold": ref_ok,
        "relational_semantics": "SUM(g/r/h/hr) per player; return the top player for each of the four metrics.",
        "source_schema": {"batting": ["player_id TEXT", "g INT", "r REAL", "h REAL", "hr REAL"],
                          "player": ["player_id TEXT", "name_given TEXT"]},
        "mongodb": {"database": "codegen_capstone_pilot", "imports": [m_bat, m_pl], "_id": "default ObjectId"},
        "mql": {"collection": coll.name, "method": "aggregate", "pipeline": PIPELINE},
        "mongo_result": res,
        "validation": {"per_gold": v, "matched_any_gold": ok},
        "assumptions": [
            "The four metrics are assumed to have a unique top player; a tie would need every tied player (SQL subquery MAX keeps ties).",
            "Gold _a/_b/_c compare only the numeric column (condition_cols [2],[2],[1]).",
        ],
    }
    return ("passed" if ok else "failed"), matched_reason(v, ok), artifact
