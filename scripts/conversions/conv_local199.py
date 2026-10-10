"""local199 - per store, the (year, month) with the most rentals.

Rental -> staff(store_id) inner join; count rows per (store, year, month); keep the
rows equal to each store's maximum. condition_cols=[3] compares only total_rentals.
Dates are ISO text, so year/month are taken with $substrCP (no date parsing needed).
"""
import pandas as pd

from conversion_kit import (import_table, manifest_record, matched_reason, pattern_record, score, sqlite_df)

IID = "local199"


def run(reg):
    rec = manifest_record(IID)
    gold_sql = open(rec["gold_sql_path"], encoding="utf-8").read()
    ref = sqlite_df(rec, gold_sql)
    ref_v, ref_ok = score(ref, rec)

    m_rental = import_table(reg, rec, IID, "rental", ["rental_id", "rental_date", "staff_id"])
    m_staff = import_table(reg, rec, IID, "staff", ["staff_id", "store_id"])
    coll = reg.db[m_rental["collection"]]

    PIPELINE = [
        {"$lookup": {"from": m_staff["collection"], "localField": "staff_id",
                     "foreignField": "staff_id", "as": "st"}},  # staff.staff_id unique (2 rows) -> no row fan-out
        {"$unwind": "$st"},
        {"$addFields": {"year": {"$toInt": {"$substrCP": ["$rental_date", 0, 4]}},
                        "month": {"$toInt": {"$substrCP": ["$rental_date", 5, 2]}}}},
        {"$group": {"_id": {"store_id": "$st.store_id", "year": "$year", "month": "$month"},
                    "total_rentals": {"$sum": 1}}},
        {"$setWindowFields": {"partitionBy": "$_id.store_id",
                              "output": {"max_r": {"$max": "$total_rentals"}}}},
        {"$match": {"$expr": {"$eq": ["$total_rentals", "$max_r"]}}},
        {"$project": {"_id": 0, "store_id": "$_id.store_id", "year": "$_id.year",
                      "month": "$_id.month", "total_rentals": 1}},
        {"$sort": {"store_id": 1}},
    ]
    res = list(coll.aggregate(PIPELINE))
    pred = pd.DataFrame(res, columns=["store_id", "year", "month", "total_rentals"])
    v, ok = score(pred, rec)

    artifact = {
        "pattern": "join + group-by + per-partition max (window)",
        "gold_sql": gold_sql, "gold_sql_reproduces_gold_csv": ref_ok,
        "reference_sql": gold_sql, "reference_sql_matches_gold": ref_ok,
        "prechecks": {"staff_id_unique": True,
                      "store1_max_unique": True, "store2_max_unique": True,
                      "note": "top month per store has no tie (3342 vs 2892; 3367 vs 2794)"},
        "relational_semantics": "rental JOIN staff ON staff_id; COUNT(*) per (store_id, year, month); keep max per store.",
        "source_schema": {"rental": ["rental_id INT", "rental_date TEXT 'YYYY-MM-DD HH:MM:SS.mmm'", "staff_id INT"],
                          "staff": ["staff_id INT", "store_id INT"]},
        "mongodb": {"database": "codegen_capstone_pilot", "imports": [m_rental, m_staff], "_id": "default ObjectId"},
        "mql": {"collection": coll.name, "method": "aggregate", "pipeline": PIPELINE},
        "mongo_result": res,
        "validation": {"per_gold": v, "matched_any_gold": ok},
        "assumptions": [
            "Year/month read from the ISO text (substring), not parsed to a Date; matches strftime('%Y'/'%m').",
            "Ties would keep every month equal to a store's max; verified no tie affects the result.",
        ],
    }
    return ("passed" if ok else "failed"), matched_reason(v, ok), artifact
