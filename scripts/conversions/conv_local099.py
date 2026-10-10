"""local099 - actors who made more films with Yash Chopra than any other director.

Targets gold _b (107). M_Cast.PID carries stray whitespace (82,835/82,837 rows), so it
is trimmed before grouping; Person.PID and M_Cast.MID / M_Director.MID are clean. The
'M_Cast JOIN M_Director ON MID' is 1:1 (3,475 rows each), so a plain $lookup suffices.
"""
import pandas as pd

from conversion_kit import (import_table, manifest_record, matched_reason, pattern_record, score, sqlite_df)

IID = "local099"
YASH_PID = "nm0007181"


def run(reg):
    rec = manifest_record(IID)
    gold_sql = open(rec["gold_sql_path"], encoding="utf-8").read()
    ref = sqlite_df(rec, gold_sql)
    ref_v, ref_ok = score(ref, rec)

    m_cast = import_table(reg, rec, IID, "M_Cast", ["MID", "PID"])
    m_dir = import_table(reg, rec, IID, "M_Director", ["MID", "PID"], index=["MID"])
    m_person = import_table(reg, rec, IID, "Person", ["PID", "Name"], index=["PID"])
    coll = reg.db[m_cast["collection"]]

    PIPELINE = [
        {"$addFields": {"actor_pid": {"$trim": {"input": "$PID"}}}},
        {"$lookup": {"from": m_dir["collection"], "localField": "MID",
                     "foreignField": "MID", "as": "d"}},
        {"$unwind": "$d"},
        {"$group": {"_id": {"actor": "$actor_pid", "director": "$d.PID"},
                    "mids": {"$addToSet": "$MID"}}},
        {"$project": {"actor": "$_id.actor", "director": "$_id.director",
                      "num": {"$size": "$mids"}}},
        {"$group": {"_id": "$actor",
                    "yc": {"$max": {"$cond": [{"$eq": ["$director", YASH_PID]}, "$num", 0]}},
                    "other": {"$max": {"$cond": [{"$ne": ["$director", YASH_PID]}, "$num", None]}}}},
        {"$match": {"$expr": {"$gt": ["$yc", {"$ifNull": ["$other", 0]}]}}},
        {"$lookup": {"from": m_person["collection"], "localField": "_id",
                     "foreignField": "PID", "as": "p"}},
        {"$unwind": "$p"},
        {"$group": {"_id": "$_id"}},  # SELECT COUNT(DISTINCT Person.PID) restricted to qualifying actors
        {"$count": "num_actor"},
        {"$project": {"_id": 0, "num_actor": 1}},
    ]
    res = list(coll.aggregate(PIPELINE))
    pred = pd.DataFrame(res, columns=["num_actor"])
    v, ok = score(pred, rec)

    artifact = {
        "pattern": "self-contained join graph + per-entity conditional max + count distinct",
        "gold_sql": gold_sql, "gold_sql_reproduces_gold_csv": ref_ok,
        "reference_sql": gold_sql, "reference_sql_matches_gold": ref_ok,
        "prechecks": {"Yash_PIDs": [YASH_PID], "M_Cast.PID_stray_whitespace": "82,835/82,837 rows trimmed",
                      "MID_clean_both_sides": True, "cast_director_join_cardinality": "1:1 on MID"},
        "relational_semantics": ("Count films per (actor, director) with COUNT(DISTINCT MID); for each actor keep yc = films with "
                                 "Yash Chopra and other = max films with any other director; require yc > IFNULL(other,0); "
                                 "count distinct Person.PID."),
        "source_schema": {"M_Cast": ["MID TEXT", "PID TEXT (has whitespace)"],
                          "M_Director": ["MID TEXT", "PID TEXT"], "Person": ["PID TEXT", "Name TEXT"]},
        "mongodb": {"database": "codegen_capstone_pilot", "imports": [m_cast, m_dir, m_person], "_id": "default ObjectId"},
        "mql": {"collection": coll.name, "method": "aggregate", "pipeline": PIPELINE},
        "mongo_result": res,
        "validation": {"per_gold": v, "matched_any_gold": ok, "targeted_gold": "local099_b.csv (107)"},
        "assumptions": [
            "Target gold _b (107); gold _a (266) reflects a different interpretation and is not produced by the shipped SQL.",
            "Only M_Cast.PID needs trimming; the MID join key is clean on both sides.",
        ],
    }
    return ("passed" if ok else "failed"), matched_reason(v, ok), artifact
