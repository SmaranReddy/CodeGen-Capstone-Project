"""local128 - low-handicap wins at all three venues (Thunderbird, Totem, Bolero).

No shipped gold SQL: a labelled reference SQL is authored and verified against the gold
CSV first. Qualifying bowlers won (HandiCapScore <= 190) at each of the three venues; all
their such win records at those venues are returned.
"""
import pandas as pd

from conversion_kit import (import_table, manifest_record, matched_reason, pattern_record, score, sqlite_df)

IID = "local128"

VENUES = ["Thunderbird Lanes", "Totem Lanes", "Bolero Lanes"]

REFERENCE_SQL = """
WITH w AS (
    SELECT bs.BowlerID, bs.MatchID, bs.GameNumber, bs.HandiCapScore,
           t.TourneyDate, t.TourneyLocation
    FROM Bowler_Scores bs
    JOIN Tourney_Matches m ON bs.MatchID = m.MatchID
    JOIN Tournaments t ON m.TourneyID = t.TourneyID
    WHERE bs.WonGame = 1 AND bs.HandiCapScore <= 190
      AND t.TourneyLocation IN ('Thunderbird Lanes', 'Totem Lanes', 'Bolero Lanes')
),
ok AS (SELECT BowlerID FROM w GROUP BY BowlerID HAVING COUNT(DISTINCT TourneyLocation) = 3)
SELECT b.BowlerID, b.BowlerFirstName, b.BowlerLastName,
       w.MatchID, w.GameNumber, w.HandiCapScore, w.TourneyDate, w.TourneyLocation
FROM w
JOIN ok ON w.BowlerID = ok.BowlerID
JOIN Bowlers b ON b.BowlerID = w.BowlerID
"""


def run(reg):
    rec = manifest_record(IID)
    ref = sqlite_df(rec, REFERENCE_SQL)
    ref_v, ref_ok = score(ref, rec)

    m_bs = import_table(reg, rec, IID, "Bowler_Scores",
                        ["MatchID", "GameNumber", "BowlerID", "RawScore", "HandiCapScore", "WonGame"])
    m_tm = import_table(reg, rec, IID, "Tourney_Matches", ["MatchID", "TourneyID", "Lanes",
                                                           "OddLaneTeamID", "EvenLaneTeamID"], index=["MatchID"])
    m_t = import_table(reg, rec, IID, "Tournaments", ["TourneyID", "TourneyDate", "TourneyLocation"], index=["TourneyID"])
    m_b = import_table(reg, rec, IID, "Bowlers", ["BowlerID", "BowlerFirstName", "BowlerLastName"], index=["BowlerID"])
    coll = reg.db[m_bs["collection"]]

    PIPELINE = [
        {"$match": {"WonGame": 1, "HandiCapScore": {"$lte": 190}}},
        {"$lookup": {"from": m_tm["collection"], "localField": "MatchID",
                     "foreignField": "MatchID", "as": "m"}},
        {"$unwind": "$m"},
        {"$lookup": {"from": m_t["collection"], "localField": "m.TourneyID",
                     "foreignField": "TourneyID", "as": "t"}},
        {"$unwind": "$t"},
        {"$match": {"t.TourneyLocation": {"$in": VENUES}}},
        {"$group": {"_id": "$BowlerID",
                    "locs": {"$addToSet": "$t.TourneyLocation"},
                    "recs": {"$push": {"MatchID": "$MatchID", "GameNumber": "$GameNumber",
                                       "HandiCapScore": "$HandiCapScore", "TourneyDate": "$t.TourneyDate",
                                       "TourneyLocation": "$t.TourneyLocation"}}}},
        {"$match": {"$expr": {"$eq": [{"$size": "$locs"}, 3]}}},  # wins at all three venues
        {"$unwind": "$recs"},
        {"$lookup": {"from": m_b["collection"], "localField": "_id",
                     "foreignField": "BowlerID", "as": "b"}},
        {"$unwind": "$b"},
        {"$project": {"_id": 0, "BowlerID": "$_id", "BowlerFirstName": "$b.BowlerFirstName",
                      "BowlerLastName": "$b.BowlerLastName", "MatchID": "$recs.MatchID",
                      "GameNumber": "$recs.GameNumber", "HandiCapScore": "$recs.HandiCapScore",
                      "TourneyDate": "$recs.TourneyDate", "TourneyLocation": "$recs.TourneyLocation"}},
    ]
    res = list(coll.aggregate(PIPELINE, allowDiskUse=True))
    pred = pd.DataFrame(res, columns=["BowlerID", "BowlerFirstName", "BowlerLastName", "MatchID",
                                      "GameNumber", "HandiCapScore", "TourneyDate", "TourneyLocation"])
    v, ok = score(pred, rec)

    artifact = {
        "pattern": "filtered join + grouped set membership (all three venues) + re-expansion",
        "gold_sql": None, "gold_sql_reproduces_gold_csv": "n/a (no shipped gold SQL)",
        "reference_sql": REFERENCE_SQL, "reference_sql_matches_gold": ref_ok,
        "relational_semantics": ("Wins (WonGame=1, HandiCapScore<=190) at the three venues; bowlers with a win at each venue; all their such records."),
        "source_schema": {"Bowler_Scores": ["MatchID", "GameNumber", "BowlerID", "HandiCapScore", "WonGame"],
                          "Tourney_Matches": ["MatchID", "TourneyID"], "Tournaments": ["TourneyID", "TourneyDate", "TourneyLocation"],
                          "Bowlers": ["BowlerID", "BowlerFirstName", "BowlerLastName"]},
        "mongodb": {"database": "codegen_capstone_pilot", "imports": [m_bs, m_tm, m_t, m_b], "_id": "default ObjectId",
                    "indexes": ["Tourney_Matches.MatchID", "Tournaments.TourneyID", "Bowlers.BowlerID"]},
        "mql": {"collection": coll.name, "method": "aggregate", "options": {"allowDiskUse": True}, "pipeline": PIPELINE},
        "mongo_result": res,
        "validation": {"per_gold": v, "matched_any_gold": ok},
        "assumptions": [
            "Tourney_Matches is joined on MatchID only (it has no GameNumber); the win flag comes from Bowler_Scores.",
            "The three venues are exactly Thunderbird Lanes, Totem Lanes, Bolero Lanes; qualifying bowlers have a win at all three.",
        ],
    }
    return ("passed" if ok else "failed"), matched_reason(v, ok), artifact
