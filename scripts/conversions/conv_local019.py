"""local019 - wrestlers in the shortest NXT-title match (excluding title changes).

No shipped gold SQL: a labelled reference SQL is authored and verified against the gold
CSV first. Duration 'MM:SS' is converted to seconds; ties are irrelevant (one output row).
"""
import pandas as pd

from conversion_kit import import_table, manifest_record, matched_reason, pattern_record, score, sqlite_df

IID = "local019"

REFERENCE_SQL = """
SELECT w.name AS WRESTLER1, l.name AS WRESTLER2
FROM Matches m
JOIN Belts b ON b.id = m.title_id AND b.name LIKE '%NXT%'
JOIN Wrestlers w ON w.id = m.winner_id
JOIN Wrestlers l ON l.id = m.loser_id
WHERE m.title_change = 0 AND m.duration IS NOT NULL AND m.duration <> ''
ORDER BY (CAST(substr(m.duration,1,instr(m.duration,':')-1) AS INTEGER) * 60
          + CAST(substr(m.duration,instr(m.duration,':')+1) AS INTEGER)) ASC
LIMIT 1
"""

def build_pipeline(nxt_ids, wres_coll):
    return [
        {"$match": {"title_change": 0, "duration": {"$nin": [None, ""]}, "title_id": {"$in": nxt_ids}}},
        {"$addFields": {"parts": {"$split": ["$duration", ":"]}}},
        {"$addFields": {"dsec": {"$add": [{"$multiply": [{"$toInt": {"$arrayElemAt": ["$parts", 0]}}, 60]},
                                         {"$toInt": {"$arrayElemAt": ["$parts", 1]}}]}}},
        {"$sort": {"dsec": 1, "_id": 1}},
        {"$limit": 1},
        {"$lookup": {"from": wres_coll, "let": {"wid": "$winner_id"},
                     "pipeline": [{"$match": {"$expr": {"$eq": [{"$toString": "$id"}, "$$wid"]}}}], "as": "w"}},
        {"$lookup": {"from": wres_coll, "let": {"lid": "$loser_id"},
                     "pipeline": [{"$match": {"$expr": {"$eq": [{"$toString": "$id"}, "$$lid"]}}}], "as": "l"}},
        {"$project": {"_id": 0, "WRESTLER1": {"$arrayElemAt": ["$w.name", 0]},
                      "WRESTLER2": {"$arrayElemAt": ["$l.name", 0]}}},
    ]


def run(reg):
    rec = manifest_record(IID)
    ref = sqlite_df(rec, REFERENCE_SQL)
    ref_v, ref_ok = score(ref, rec)

    matches = import_table(reg, rec, IID, "Matches", ["id", "winner_id", "loser_id", "duration", "title_id", "title_change"])
    belts = import_table(reg, rec, IID, "Belts", ["id", "name"], index=["id"])
    wres = import_table(reg, rec, IID, "Wrestlers", ["id", "name"], index=["id"])

    nxt_ids = [str(b["id"]) for b in reg.db[belts["collection"]].find({"name": {"$regex": "NXT"}}, {"id": 1})]
    coll = reg.db[matches["collection"]]
    PIPELINE = build_pipeline(nxt_ids, wres["collection"])
    res = list(coll.aggregate(PIPELINE, allowDiskUse=True))
    pred = pd.DataFrame(res, columns=["WRESTLER1", "WRESTLER2"])
    v, ok = score(pred, rec)

    artifact = {
        "pattern": "join title/winner/loser, duration string parse to seconds, argmin with title filter",
        "gold_sql": None, "gold_sql_reproduces_gold_csv": "n/a (no shipped gold SQL)",
        "reference_sql": REFERENCE_SQL, "reference_sql_matches_gold": ref_ok,
        "relational_semantics": ("Filter matches on NXT belts with no title change, convert 'MM:SS' to seconds, "
                                 "take the minimum and return winner/loser names."),
        "source_schema": {"Matches": ["id", "winner_id", "loser_id", "duration", "title_id", "title_change"],
                          "Belts": ["id", "name"], "Wrestlers": ["id", "name"]},
        "mongodb": {"database": "codegen_capstone_pilot", "imports": [matches, belts, wres], "_id": "default ObjectId"},
        "mql": {"collection": coll.name, "method": "aggregate", "options": {"allowDiskUse": True}, "pipeline": PIPELINE},
        "mongo_result": res,
        "validation": {"per_gold": v, "matched_any_gold": ok},
        "assumptions": ["title_change = 0 means no title change; duration is 'MM:SS'."],
    }
    return ("passed" if ok else "failed"), matched_reason(v, ok), artifact
