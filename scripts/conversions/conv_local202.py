"""local202 - count of top-10 alien states with more friendly than hostile aliens and avg age > 200.

No shipped gold SQL: a labelled reference SQL is authored and verified against the gold
CSV first. "friendly" = aggressive flag 0, "hostile" = aggressive flag 1.
"""
import pandas as pd

from conversion_kit import import_table, manifest_record, matched_reason, pattern_record, score, sqlite_df

IID = "local202"

REFERENCE_SQL = """
WITH top10 AS (SELECT state, COUNT(*) AS c FROM alien_data GROUP BY state ORDER BY c DESC LIMIT 10),
agg AS (SELECT state,
               SUM(CASE WHEN aggressive = 0 THEN 1 ELSE 0 END) AS friendly,
               SUM(CASE WHEN aggressive = 1 THEN 1 ELSE 0 END) AS hostile,
               AVG(age) AS avgage
        FROM alien_data GROUP BY state)
SELECT COUNT(*) AS qualified_states_count
FROM top10 t JOIN agg a ON a.state = t.state
WHERE a.friendly > a.hostile AND a.avgage > 200
"""

PIPELINE = [
    {"$group": {"_id": "$state", "c": {"$sum": 1},
                "friendly": {"$sum": {"$cond": [{"$eq": ["$aggressive", 0]}, 1, 0]}},
                "hostile": {"$sum": {"$cond": [{"$eq": ["$aggressive", 1]}, 1, 0]}},
                "avgage": {"$avg": "$age"}}},
    {"$sort": {"c": -1, "_id": 1}},
    {"$limit": 10},
    {"$match": {"$expr": {"$and": [{"$gt": ["$friendly", "$hostile"]}, {"$gt": ["$avgage", 200]}]}}},
    {"$count": "qualified_states_count"},
]


def run(reg):
    rec = manifest_record(IID)
    ref = sqlite_df(rec, REFERENCE_SQL)
    ref_v, ref_ok = score(ref, rec)

    m = import_table(reg, rec, IID, "alien_data", ["state", "age", "aggressive"])
    coll = reg.db[m["collection"]]

    res = list(coll.aggregate(PIPELINE, allowDiskUse=True))
    pred = pd.DataFrame(res, columns=["qualified_states_count"])
    v, ok = score(pred, rec)

    artifact = {
        "pattern": "top-N states by population + friendly/hostile ratio + avg-age filter + count",
        "gold_sql": None, "gold_sql_reproduces_gold_csv": "n/a (no shipped gold SQL)",
        "reference_sql": REFERENCE_SQL, "reference_sql_matches_gold": ref_ok,
        "relational_semantics": ("Take the 10 most populated states; keep those where friendly (aggressive=0) aliens "
                                 "outnumber hostile (aggressive=1) aliens and the average alien age exceeds 200; count them."),
        "source_schema": {"alien_data": ["state", "age", "aggressive"]},
        "mongodb": {"database": "codegen_capstone_pilot", "imports": [m], "_id": "default ObjectId"},
        "mql": {"collection": coll.name, "method": "aggregate", "options": {"allowDiskUse": True}, "pipeline": PIPELINE},
        "mongo_result": res,
        "validation": {"per_gold": v, "matched_any_gold": ok},
        "assumptions": [
            "friendly aliens are aggressive=0; hostile aliens are aggressive=1.",
            "The top-10 states are chosen by total alien count (ties broken by state name).",
        ],
    }
    return ("passed" if ok else "failed"), matched_reason(v, ok), artifact
