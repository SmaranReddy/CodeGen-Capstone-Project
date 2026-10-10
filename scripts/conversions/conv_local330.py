"""local330 - unique sessions per page where the page is a landing and/or exit page.

No shipped gold SQL: a labelled reference SQL is authored and verified against the gold
CSV first. Paths are canonicalised (trailing slash removed, except the root) and empty
paths dropped; each session counts once per page even if the page is both landing and exit.
"""
import pandas as pd

from conversion_kit import (import_table, manifest_record, matched_reason, pattern_record, score, sqlite_df)

IID = "local330"

NORM = ("CASE WHEN path = '/' THEN '/' "
        "WHEN path LIKE '%/' THEN substr(path, 1, length(path) - 1) "
        "ELSE path END")

REFERENCE_SQL = f"""
WITH a AS (SELECT session, stamp, {NORM} AS p FROM activity_log),
     lo AS (SELECT session, p, stamp,
                   MIN(stamp) OVER (PARTITION BY session) AS mn,
                   MAX(stamp) OVER (PARTITION BY session) AS mx
            FROM a)
SELECT p AS path, COUNT(DISTINCT session) AS session_count
FROM lo
WHERE (stamp = mn OR stamp = mx) AND p <> ''
GROUP BY p
"""


def run(reg):
    rec = manifest_record(IID)
    ref = sqlite_df(rec, REFERENCE_SQL)
    ref_v, ref_ok = score(ref, rec)

    m = import_table(reg, rec, IID, "activity_log", ["stamp", "session", "path"])
    coll = reg.db[m["collection"]]

    norm_expr = {"$cond": [
        {"$eq": ["$path", "/"]}, "/",
        {"$cond": [{"$regexMatch": {"input": "$path", "regex": "/$"}},
                   {"$substrCP": ["$path", 0, {"$subtract": [{"$strLenCP": "$path"}, 1]}]},
                   "$path"]}]}
    PIPELINE = [
        {"$addFields": {"p": norm_expr}},
        {"$match": {"p": {"$ne": ""}}},
        {"$sort": {"session": 1, "stamp": 1}},
        {"$group": {"_id": "$session", "first": {"$first": "$p"}, "last": {"$last": "$p"}}},
        {"$project": {"pages": {"$setUnion": [["$first"], ["$last"]]}}},
        {"$unwind": "$pages"},
        {"$group": {"_id": "$pages", "sessions": {"$addToSet": "$_id"}}},
        {"$project": {"_id": 0, "path": "$_id", "session_count": {"$size": "$sessions"}}},
    ]
    res = list(coll.aggregate(PIPELINE))
    pred = pd.DataFrame(res, columns=["path", "session_count"])
    v, ok = score(pred, rec)

    artifact = {
        "pattern": "per-session first/last via group + set-union union-all",
        "gold_sql": None, "gold_sql_reproduces_gold_csv": "n/a (no shipped gold SQL)",
        "reference_sql": REFERENCE_SQL, "reference_sql_matches_gold": ref_ok,
        "relational_semantics": ("Per session, landing = earliest path and exit = latest path by stamp; count distinct sessions where each "
                                 "page is landing and/or exit; empty trailing-slash duplicates merged; blank pages dropped."),
        "source_schema": {"activity_log": ["stamp varchar", "session varchar", "path varchar"]},
        "mongodb": {"database": "codegen_capstone_pilot", "imports": [m], "_id": "default ObjectId"},
        "mql": {"collection": coll.name, "method": "aggregate", "pipeline": PIPELINE},
        "mongo_result": res,
        "validation": {"per_gold": v, "matched_any_gold": ok},
        "assumptions": [
            "Trailing-slash variants represent the same page and are canonicalised; the gold set contains both spellings across its three files.",
            "Blank path rows are excluded; the matching gold (local330_c) has no blank-path row.",
            "$setUnion deduplicates a page that is both landing and exit for a session so the session is counted once.",
        ],
    }
    return ("passed" if ok else "failed"), matched_reason(v, ok), artifact
