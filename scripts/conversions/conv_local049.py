"""local049 - average new unicorns per year (2019-2021) in the top industry.

No shipped gold SQL: a labelled reference SQL is authored and verified against the gold
CSV first. A unicorn has valuation >= 1e9; the top industry is the one with the most
unicorns overall; the average divides that industry's 2019-2021 unicorn count by 3.
"""
import pandas as pd

from conversion_kit import (import_table, manifest_record, matched_reason, pattern_record, score, sqlite_df)

IID = "local049"

REFERENCE_SQL = """
WITH u AS (
    SELECT cf.company_id, substr(cd.date_joined, 1, 4) AS y, ci.industry
    FROM companies_funding cf
    JOIN companies_dates cd ON cf.company_id = cd.company_id
    JOIN companies_industries ci ON cf.company_id = ci.company_id
    WHERE cf.valuation >= 1e9
),
ti AS (SELECT industry FROM u GROUP BY industry ORDER BY COUNT(*) DESC LIMIT 1)
SELECT COUNT(*) / 3.0 AS AVERAGE_NEW_UNICORNS_PER_YEAR
FROM u JOIN ti ON u.industry = ti.industry
WHERE u.y BETWEEN '2019' AND '2021'
"""


def run(reg):
    rec = manifest_record(IID)
    ref = sqlite_df(rec, REFERENCE_SQL)
    ref_v, ref_ok = score(ref, rec)

    m_f = import_table(reg, rec, IID, "companies_funding", ["company_id", "valuation"])
    m_d = import_table(reg, rec, IID, "companies_dates", ["company_id", "date_joined"], index=["company_id"])
    m_i = import_table(reg, rec, IID, "companies_industries", ["company_id", "industry"], index=["company_id"])
    coll = reg.db[m_f["collection"]]

    PIPELINE = [
        {"$match": {"valuation": {"$gte": 1000000000}}},
        {"$lookup": {"from": m_d["collection"], "localField": "company_id",
                     "foreignField": "company_id", "as": "cd"}},
        {"$unwind": "$cd"},
        {"$lookup": {"from": m_i["collection"], "localField": "company_id",
                     "foreignField": "company_id", "as": "ci"}},
        {"$unwind": "$ci"},
        {"$group": {"_id": "$ci.industry", "mids": {"$addToSet": "$company_id"},
                    "years": {"$push": {"$substrCP": ["$cd.date_joined", 0, 4]}}}},
        {"$addFields": {"total": {"$size": "$mids"}}},
        {"$sort": {"total": -1, "_id": 1}},
        {"$limit": 1},
        {"$project": {"_id": 0, "AVERAGE_NEW_UNICORNS_PER_YEAR": {"$divide": [
            {"$size": {"$filter": {"input": "$years", "as": "y",
                                   "cond": {"$and": [{"$gte": ["$$y", "2019"]}, {"$lte": ["$$y", "2021"]}]}}}}, 3.0]}}},
    ]
    res = list(coll.aggregate(PIPELINE, allowDiskUse=True))
    pred = pd.DataFrame(res, columns=["AVERAGE_NEW_UNICORNS_PER_YEAR"])
    v, ok = score(pred, rec)

    artifact = {
        "pattern": "unicorn filter + top industry + windowed count / years",
        "gold_sql": None, "gold_sql_reproduces_gold_csv": "n/a (no shipped gold SQL)",
        "reference_sql": REFERENCE_SQL, "reference_sql_matches_gold": ref_ok,
        "relational_semantics": ("Unicorns = valuation >= 1e9; top industry by total unicorn count (Fintech, 233); average per year = its "
                                 "2019-2021 unicorns / 3 = 181/3."),
        "source_schema": {"companies_funding": ["company_id", "valuation"], "companies_dates": ["company_id", "date_joined"],
                          "companies_industries": ["company_id", "industry"]},
        "mongodb": {"database": "codegen_capstone_pilot", "imports": [m_f, m_d, m_i], "_id": "default ObjectId",
                    "indexes": ["companies_dates.company_id", "companies_industries.company_id"]},
        "mql": {"collection": coll.name, "method": "aggregate", "options": {"allowDiskUse": True}, "pipeline": PIPELINE},
        "mongo_result": res,
        "validation": {"per_gold": v, "matched_any_gold": ok},
        "assumptions": [
            "Top industry is determined over all years (Fintech); the year window is 2019-2021 inclusive.",
            "The average divides the three-year total by 3, matching gold _a (60.333).",
        ],
    }
    return ("passed" if ok else "failed"), matched_reason(v, ok), artifact
