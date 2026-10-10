"""local133 - weighted musical-style scores and their distance from the average.

No gold SQL exists, so a labelled reference SQL is authored and checked against the
gold CSVs first. Weighted score = 3/2/1 per 1st/2nd/3rd preference; each ranked style's
|score - mean(score)| is returned.
"""
import pandas as pd

from conversion_kit import (import_table, manifest_record, matched_reason, pattern_record, score, sqlite_df)

IID = "local133"

REFERENCE_SQL = """
WITH scores AS (
    SELECT StyleID,
           SUM(CASE PreferenceSeq WHEN 1 THEN 3 WHEN 2 THEN 2 WHEN 3 THEN 1 ELSE 0 END) AS w
    FROM Musical_Preferences
    GROUP BY StyleID
)
SELECT s.StyleName,
       ROUND(ABS(sc.w - (SELECT AVG(w) FROM scores)), 2) AS AbsoluteDifference
FROM scores sc JOIN Musical_Styles s ON sc.StyleID = s.StyleID
ORDER BY s.StyleName
"""


def run(reg):
    rec = manifest_record(IID)
    ref = sqlite_df(rec, REFERENCE_SQL)
    ref_v, ref_ok = score(ref, rec)

    m_pref = import_table(reg, rec, IID, "Musical_Preferences", ["CustomerID", "StyleID", "PreferenceSeq"])
    m_sty = import_table(reg, rec, IID, "Musical_Styles", ["StyleID", "StyleName"], index=["StyleID"])
    coll = reg.db[m_pref["collection"]]

    PIPELINE = [
        {"$group": {"_id": "$StyleID", "w": {"$sum": {"$switch": {"branches": [
            {"case": {"$eq": ["$PreferenceSeq", 1]}, "then": 3},
            {"case": {"$eq": ["$PreferenceSeq", 2]}, "then": 2},
            {"case": {"$eq": ["$PreferenceSeq", 3]}, "then": 1}], "default": 0}}}}},
        {"$setWindowFields": {"output": {"avgW": {"$avg": "$w"}}}},  # global average across ranked styles
        {"$addFields": {"AbsoluteDifference": {"$round": [{"$abs": {"$subtract": ["$w", "$avgW"]}}, 2]}}},
        {"$lookup": {"from": m_sty["collection"], "localField": "_id",
                     "foreignField": "StyleID", "as": "s"}},
        {"$unwind": "$s"},
        {"$project": {"_id": 0, "StyleName": "$s.StyleName",
                      "TOTAL_WEIGHTED_SCORE": "$w", "AVERAGE_SCORE": "$avgW",
                      "AbsoluteDifference": 1}},
    ]
    res = list(coll.aggregate(PIPELINE))
    pred = pd.DataFrame(res)
    v, ok = score(pred, rec)

    artifact = {
        "pattern": "conditional weighted sum + global average (window) + absolute deviation",
        "gold_sql": None, "gold_sql_reproduces_gold_csv": "n/a (no shipped gold SQL)",
        "reference_sql": REFERENCE_SQL, "reference_sql_matches_gold": ref_ok,
        "relational_semantics": ("Per StyleID, score = 3*#1st + 2*#2nd + 1*#3rd; mean over ranked styles; "
                                 "return StyleName and round(|score-mean|, 2)."),
        "source_schema": {"Musical_Preferences": ["CustomerID INT", "StyleID smallint", "PreferenceSeq smallint"],
                          "Musical_Styles": ["StyleID smallint", "StyleName nvarchar"]},
        "mongodb": {"database": "codegen_capstone_pilot", "imports": [m_pref, m_sty], "_id": "default ObjectId"},
        "mql": {"collection": coll.name, "method": "aggregate", "pipeline": PIPELINE},
        "mongo_result": res,
        "validation": {"per_gold": v, "matched_any_gold": ok},
        "assumptions": [
            "Only styles that appear in Musical_Preferences are included (ranked by >= 1 user); the mean is over those styles.",
            "Extra output columns (TOTAL_WEIGHTED_SCORE, AVERAGE_SCORE) are harmless; gold _b checks them too.",
        ],
    }
    return ("passed" if ok else "failed"), matched_reason(v, ok), artifact
