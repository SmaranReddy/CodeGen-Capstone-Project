"""local065 - Pizza Runner total income (recursive extras split -> $split/$size).

Targets gold _b (TOTAL_INCOME = 142), the same value the shipped gold SQL produces.
The shipped SQL's recursive CTE only counts the number of comma-separated extras per
order; $split + $size reproduces it. Canceled orders (cancellation IS NOT NULL) are
dropped by an inner join + $match on the runner-orders cancellation.
"""
import pandas as pd

from conversion_kit import (import_table, manifest_record, matched_reason, pattern_record, score, sqlite_df)

IID = "local065"


def run(reg):
    rec = manifest_record(IID)
    pat = pattern_record(IID)
    gold_sql = (rec["gold_sql_path"] and open(rec["gold_sql_path"], encoding="utf-8").read())

    ref = sqlite_df(rec, gold_sql)
    ref_v, ref_ok = score(ref, rec)

    m1 = import_table(reg, rec, IID, "pizza_clean_customer_orders", ["order_id", "pizza_id", "extras"])
    m2 = import_table(reg, rec, IID, "pizza_clean_runner_orders", ["order_id", "cancellation"])
    coll = reg.db[m1["collection"]]

    PIPELINE = [
        {"$lookup": {"from": m2["collection"], "localField": "order_id",
                     "foreignField": "order_id", "as": "ro"}},
        {"$unwind": "$ro"},  # inner join, as the SQL JOIN does
        {"$match": {"ro.cancellation": None}},  # WHERE cancellation IS NULL
        {"$addFields": {"total_extras": {"$cond": [
            {"$eq": ["$extras", None]}, 0, {"$size": {"$split": ["$extras", ","]}}]}}},
        {"$group": {
            "_id": {"order_id": "$order_id", "pizza_id": "$pizza_id", "total_extras": "$total_extras"},
            "price": {"$sum": {"$cond": [{"$eq": ["$pizza_id", 1]}, 12,
                                         {"$cond": [{"$eq": ["$pizza_id", 2]}, 10, None]}]}},
            "total_extras": {"$first": "$total_extras"}}},
        {"$group": {"_id": None, "total_income": {"$sum": {"$add": ["$price", "$total_extras"]}}}},
        {"$project": {"_id": 0, "total_income": 1}},
    ]
    res = list(coll.aggregate(PIPELINE))
    pred = pd.DataFrame(res, columns=["total_income"])
    v, ok = score(pred, rec)

    artifact = {
        "pattern": "recursive string split + CASE pricing + conditional join",
        "gold_sql": gold_sql,
        "gold_sql_reproduces_gold_csv": ref_ok,
        "reference_sql": gold_sql,
        "reference_sql_matches_gold": ref_ok,
        "relational_semantics": ("Price each pizza (id1=12, id2=10); add $1 per extra topping; drop canceled orders; "
                                 "sum. The recursive CTE counts comma-separated extras only, so the MQL counts tokens."),
        "source_schema": {
            "pizza_clean_customer_orders": ["order_id INTEGER", "pizza_id INTEGER", "extras TEXT"],
            "pizza_clean_runner_orders": ["order_id INTEGER", "cancellation TEXT (NULL when delivered)"],
        },
        "mongodb": {"database": "codegen_capstone_pilot", "imports": [m1, m2], "_id": "default ObjectId"},
        "mql": {"collection": coll.name, "method": "aggregate", "pipeline": PIPELINE},
        "mongo_result": res, "validation": {"per_gold": v, "matched_any_gold": ok,
                                            "targeted_gold": "local065_b.csv (142)"},
        "assumptions": [
            "Target gold _b (142); gold _a (144) comes from a different extras interpretation and is not reproducible from the shipped SQL.",
            "NULL extras contribute 0 (SQL COUNT(each_extra) ignores NULL).",
        ],
    }
    return ("passed" if ok else "failed"), matched_reason(v, ok), artifact
