"""local196 - average spend and subsequent rentals by the rating of the first rental.

No shipped gold SQL: a labelled reference SQL is authored and verified against the gold
CSV first. The first movie is the film of the rental tied to a customer's earliest payment
date; spend is the customer's total payments and subsequent rentals = rentals - 1.
"""
import pandas as pd

from conversion_kit import (import_table, manifest_record, matched_reason, pattern_record, score, sqlite_df)

IID = "local196"

REFERENCE_SQL = """
WITH firstpay AS (
    SELECT customer_id, MIN(payment_date) AS d FROM payment GROUP BY customer_id
),
fc AS (
    SELECT p.customer_id, f.rating
    FROM firstpay fp
    JOIN payment p ON p.customer_id = fp.customer_id AND p.payment_date = fp.d
    JOIN rental r ON p.rental_id = r.rental_id
    JOIN inventory i ON r.inventory_id = i.inventory_id
    JOIN film f ON i.film_id = f.film_id
),
spend AS (SELECT customer_id, SUM(amount) AS tot FROM payment GROUP BY customer_id),
rents AS (SELECT customer_id, COUNT(*) AS n FROM rental GROUP BY customer_id)
SELECT fc.rating,
       AVG(s.tot) AS avg_total_spend,
       AVG(r.n - 1) AS avg_subsequent_rental_count
FROM fc
JOIN spend s ON s.customer_id = fc.customer_id
JOIN rents r ON r.customer_id = fc.customer_id
GROUP BY fc.rating
"""


def run(reg):
    rec = manifest_record(IID)
    ref = sqlite_df(rec, REFERENCE_SQL)
    ref_v, ref_ok = score(ref, rec)

    m_pay = import_table(reg, rec, IID, "payment", ["customer_id", "rental_id", "amount", "payment_date"])
    m_rent = import_table(reg, rec, IID, "rental", ["rental_id", "customer_id", "inventory_id"], index=["rental_id"])
    m_inv = import_table(reg, rec, IID, "inventory", ["inventory_id", "film_id"], index=["inventory_id"])
    m_film = import_table(reg, rec, IID, "film", ["film_id", "rating"], index=["film_id"])
    coll = reg.db[m_pay["collection"]]

    PIPELINE = [
        {"$sort": {"payment_date": 1}},
        {"$group": {"_id": "$customer_id", "total": {"$sum": "$amount"},
                    "firstRental": {"$first": "$rental_id"}}},
        {"$lookup": {"from": m_rent["collection"], "localField": "firstRental",
                     "foreignField": "rental_id", "as": "r"}},
        {"$unwind": "$r"},
        {"$lookup": {"from": m_inv["collection"], "localField": "r.inventory_id",
                     "foreignField": "inventory_id", "as": "i"}},
        {"$unwind": "$i"},
        {"$lookup": {"from": m_film["collection"], "localField": "i.film_id",
                     "foreignField": "film_id", "as": "f"}},
        {"$unwind": "$f"},
        {"$lookup": {"from": m_rent["collection"], "localField": "_id",
                     "foreignField": "customer_id", "as": "allrentals"}},
        {"$addFields": {"nrentals": {"$size": "$allrentals"}}},
        {"$group": {"_id": "$f.rating",
                    "avg_total_spend": {"$avg": "$total"},
                    "avg_subsequent_rental_count": {"$avg": {"$subtract": ["$nrentals", 1]}}}},
        {"$project": {"_id": 0, "rating": "$_id", "avg_total_spend": 1, "avg_subsequent_rental_count": 1}},
    ]
    res = list(coll.aggregate(PIPELINE, allowDiskUse=True))
    pred = pd.DataFrame(res, columns=["rating", "avg_total_spend", "avg_subsequent_rental_count"])
    v, ok = score(pred, rec)

    artifact = {
        "pattern": "per-customer min-date first rental (sort+first) + chained lookups + group averages",
        "gold_sql": None, "gold_sql_reproduces_gold_csv": "n/a (no shipped gold SQL)",
        "reference_sql": REFERENCE_SQL, "reference_sql_matches_gold": ref_ok,
        "relational_semantics": ("First movie = film of the rental from the customer's earliest payment; group by its rating; average total "
                                 "customer spend and average (rental count - 1)."),
        "source_schema": {"payment": ["customer_id", "rental_id", "amount", "payment_date"], "rental": ["rental_id", "customer_id", "inventory_id"],
                          "inventory": ["inventory_id", "film_id"], "film": ["film_id", "rating"]},
        "mongodb": {"database": "codegen_capstone_pilot", "imports": [m_pay, m_rent, m_inv, m_film], "_id": "default ObjectId",
                    "indexes": ["rental.rental_id", "inventory.inventory_id", "film.film_id"]},
        "mql": {"collection": coll.name, "method": "aggregate", "options": {"allowDiskUse": True}, "pipeline": PIPELINE},
        "mongo_result": res,
        "validation": {"per_gold": v, "matched_any_gold": ok},
        "assumptions": [
            "Earliest payment date determines the first rental; ties would be resolved deterministically by the sort.",
            "Total spend is the customer's lifetime payment sum and rentals come from the rental table.",
        ],
    }
    return ("passed" if ok else "failed"), matched_reason(v, ok), artifact
