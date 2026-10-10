"""local195 - percentage of customers who rented films featuring the top five actors.

No shipped gold SQL: a labelled reference SQL is authored and verified against the gold
CSV first. "Top five actors" = the five credited in the most films (ties by actor id).
"""
import pandas as pd

from conversion_kit import import_table, manifest_record, matched_reason, pattern_record, score, sqlite_df

IID = "local195"

REFERENCE_SQL = """
WITH top5 AS (SELECT actor_id, COUNT(*) AS n FROM film_actor GROUP BY actor_id ORDER BY n DESC, actor_id LIMIT 5),
filmed AS (SELECT DISTINCT film_id FROM film_actor WHERE actor_id IN (SELECT actor_id FROM top5)),
c AS (SELECT DISTINCT r.customer_id FROM rental r JOIN inventory i ON i.inventory_id = r.inventory_id
      WHERE i.film_id IN (SELECT film_id FROM filmed))
SELECT 100.0 * (SELECT COUNT(*) FROM c) / (SELECT COUNT(*) FROM customer) AS PERCENTAGE
"""

PIPELINE_TEMPLATE = [
    {"$group": {"_id": "$actor_id", "n": {"$sum": 1}}},
    {"$sort": {"n": -1, "_id": 1}},
    {"$limit": 5},
    {"$lookup": {"from": "__FA__", "localField": "_id", "foreignField": "actor_id", "as": "films"}},
    {"$unwind": "$films"},
    {"$group": {"_id": None, "film_ids": {"$addToSet": "$films.film_id"}}},
    {"$lookup": {"from": "__INV__", "localField": "film_ids", "foreignField": "film_id", "as": "inv"}},
    {"$unwind": "$inv"},
    {"$group": {"_id": None, "inv_ids": {"$addToSet": "$inv.inventory_id"}}},
    {"$lookup": {"from": "__RENT__", "localField": "inv_ids", "foreignField": "inventory_id", "as": "rents"}},
    {"$unwind": "$rents"},
    {"$group": {"_id": None, "customers": {"$addToSet": "$rents.customer_id"}}},
    {"$lookup": {"from": "__CUST__", "pipeline": [{"$count": "n"}], "as": "tc"}},
    {"$unwind": "$tc"},
    {"$project": {"_id": 0, "PERCENTAGE": {"$multiply": [100, {"$divide": [{"$size": "$customers"}, "$tc.n"]}]}}},
]


def run(reg):
    rec = manifest_record(IID)
    ref = sqlite_df(rec, REFERENCE_SQL)
    ref_v, ref_ok = score(ref, rec)

    fa = import_table(reg, rec, IID, "film_actor", ["actor_id", "film_id"], index=["actor_id", "film_id"])
    inv = import_table(reg, rec, IID, "inventory", ["inventory_id", "film_id"], index=["film_id"])
    rent = import_table(reg, rec, IID, "rental", ["inventory_id", "customer_id"], index=["inventory_id"])
    cust = import_table(reg, rec, IID, "customer", ["customer_id"])

    coll = reg.db[fa["collection"]]
    repl = {"__FA__": fa["collection"], "__INV__": inv["collection"], "__RENT__": rent["collection"], "__CUST__": cust["collection"]}
    PIPELINE = []
    for s in PIPELINE_TEMPLATE:
        if "$lookup" in s:
            s = {"$lookup": {**s["$lookup"], "from": repl[s["$lookup"]["from"]]}}
        PIPELINE.append(s)
    res = list(coll.aggregate(PIPELINE, allowDiskUse=True))
    pred = pd.DataFrame(res, columns=["PERCENTAGE"])
    v, ok = score(pred, rec)

    artifact = {
        "pattern": "top-N group + cascading array $lookups (films->inventory->rental) + coverage percentage",
        "gold_sql": None, "gold_sql_reproduces_gold_csv": "n/a (no shipped gold SQL)",
        "reference_sql": REFERENCE_SQL, "reference_sql_matches_gold": ref_ok,
        "relational_semantics": ("The top five actors are those credited in the most films; of all customers, the percentage "
                                 "who rented at least one film featuring any of them."),
        "source_schema": {"film_actor": ["actor_id", "film_id"], "inventory": ["inventory_id", "film_id"],
                          "rental": ["inventory_id", "customer_id"], "customer": ["customer_id"]},
        "mongodb": {"database": "codegen_capstone_pilot", "imports": [fa, inv, rent, cust], "_id": "default ObjectId"},
        "mql": {"collection": coll.name, "method": "aggregate", "options": {"allowDiskUse": True}, "pipeline": PIPELINE},
        "mongo_result": res,
        "validation": {"per_gold": v, "matched_any_gold": ok},
        "assumptions": [
            "Top five actors are ranked by number of credited films (ties broken by actor_id).",
            "A customer 'reached' rented any film (via inventory) featuring one of those actors.",
        ],
    }
    return ("passed" if ok else "failed"), matched_reason(v, ok), artifact
