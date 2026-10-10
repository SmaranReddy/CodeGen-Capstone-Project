"""local329 - sessions that reached /regist/input and later /regist/confirm.

No shipped gold SQL: a labelled reference SQL is authored and verified against the gold
CSV first. The relevant table is form_log (path + stamp).
"""
import pandas as pd

from conversion_kit import (import_table, manifest_record, matched_reason, pattern_record, score, sqlite_df)

IID = "local329"

REFERENCE_SQL = """
SELECT COUNT(DISTINCT a.session) AS n
FROM form_log a
JOIN form_log b
  ON a.session = b.session
 AND a.path = '/regist/input' AND b.path = '/regist/confirm'
 AND a.stamp < b.stamp
"""


def run(reg):
    rec = manifest_record(IID)
    ref = sqlite_df(rec, REFERENCE_SQL)
    ref_v, ref_ok = score(ref, rec)

    m = import_table(reg, rec, IID, "form_log", ["stamp", "session", "action", "path", "status"])
    coll = reg.db[m["collection"]]

    PIPELINE = [
        {"$match": {"path": "/regist/input"}},
        {"$lookup": {"from": m["collection"],
                     "let": {"s": "$session", "st": "$stamp"},
                     "pipeline": [{"$match": {"$expr": {"$and": [
                         {"$eq": ["$session", "$$s"]},
                         {"$eq": ["$path", "/regist/confirm"]},
                         {"$gt": ["$stamp", "$$st"]}]}}}],
                     "as": "c"}},
        {"$unwind": "$c"},
        {"$group": {"_id": "$session"}},
        {"$count": "n"},
    ]
    res = list(coll.aggregate(PIPELINE))
    pred = pd.DataFrame(res, columns=["n"])
    v, ok = score(pred, rec)

    artifact = {
        "pattern": "ordered self-join (two page types per session)",
        "gold_sql": None, "gold_sql_reproduces_gold_csv": "n/a (no shipped gold SQL)",
        "reference_sql": REFERENCE_SQL, "reference_sql_matches_gold": ref_ok,
        "relational_semantics": ("Count distinct sessions having a /regist/input row strictly before a /regist/confirm row."),
        "source_schema": {"form_log": ["stamp varchar", "session varchar", "action varchar", "path varchar", "status varchar"]},
        "mongodb": {"database": "codegen_capstone_pilot", "imports": [m], "_id": "default ObjectId"},
        "mql": {"collection": coll.name, "method": "aggregate", "pipeline": PIPELINE},
        "mongo_result": res,
        "validation": {"per_gold": v, "matched_any_gold": ok},
        "assumptions": [
            "The correct table is form_log (only it contains /regist/input and /regist/confirm).",
            "stamp values sort correctly as strings (fixed 'YYYY-MM-DD HH:MM:SS' format).",
        ],
    }
    return ("passed" if ok else "failed"), matched_reason(v, ok), artifact
