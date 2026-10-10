"""local039 - 8-table join chain, text timestamps -> hours, case-insensitive LIKE, group, sum, order, limit 1."""
import sys; sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import pandas as pd
from harness import *

IID = "local039"
rec = manifest_record(IID)
gold_sql = (ROOT / rec["gold_sql_path"]).read_text(encoding="utf-8")
ref = sqlite_df(rec, gold_sql); ref_v, ref_ok = score(ref, rec)
sq = sqlite_ro(rec)


def sqlite_variant(trunc):
    expr = ("CAST((julianday(r.return_date)-julianday(r.rental_date))*24 AS INTEGER)" if trunc
            else "(julianday(r.return_date)-julianday(r.rental_date))*24")
    return sq.execute(f"""SELECT c.name, SUM({expr}) s FROM category c JOIN film_category USING(category_id) JOIN film USING(film_id)
        JOIN inventory USING(film_id) JOIN rental r USING(inventory_id) JOIN customer USING(customer_id) JOIN address USING(address_id) JOIN city USING(city_id)
        WHERE LOWER(city.city) LIKE 'a%' OR city.city LIKE '%-%' GROUP BY c.name ORDER BY s DESC LIMIT 3""").fetchall()


pre = {
  "sqlite_top3_gold_sql_integer_truncation": sqlite_variant(True),
  "sqlite_top3_fractional_hours": sqlite_variant(False),
  "rental_date_samples": sq.execute("SELECT rental_date, return_date FROM rental LIMIT 2").fetchall(),
  "rentals_with_null_return_date": sq.execute("SELECT COUNT(*) FROM rental WHERE return_date IS NULL").fetchone()[0],
  "matching_cities_sqlite": sorted(r[0] for r in sq.execute("SELECT city FROM city WHERE LOWER(city) LIKE 'a%' OR city LIKE '%-%'")),
}

db = mongo_db()
imports = [
  import_table(db, rec, IID, "rental", ["rental_date", "return_date", "inventory_id", "customer_id"]),
  import_table(db, rec, IID, "inventory", ["inventory_id", "film_id"]),
  import_table(db, rec, IID, "film", ["film_id"]),
  import_table(db, rec, IID, "film_category", ["film_id", "category_id"]),
  import_table(db, rec, IID, "category", ["category_id", "name"]),
  import_table(db, rec, IID, "customer", ["customer_id", "address_id"]),
  import_table(db, rec, IID, "address", ["address_id", "city_id"]),
  import_table(db, rec, IID, "city", ["city_id", "city"]),
]
C = lambda t: f"{IID}__{t}"
indexes = []
for t, f in [("inventory", "inventory_id"), ("film", "film_id"), ("film_category", "film_id"), ("category", "category_id"),
             ("customer", "customer_id"), ("address", "address_id"), ("city", "city_id")]:
    db[C(t)].create_index(f)
    indexes.append({"collection": C(t), "key": f})


def lk(frm, lf, ff, a):
    return [{"$lookup": {"from": C(frm), "localField": lf, "foreignField": ff, "as": a}}, {"$unwind": "$" + a}]


FMT = "%Y-%m-%d %H:%M:%S.%L"


def pipeline(truncate):
    ms = {"$dateDiff": {"startDate": {"$dateFromString": {"dateString": "$rental_date", "format": FMT}},
                        "endDate": {"$dateFromString": {"dateString": "$return_date", "format": FMT}}, "unit": "millisecond"}}
    hours = {"$divide": [ms, 3600000]}
    return (lk("inventory", "inventory_id", "inventory_id", "inv") + lk("film", "inv.film_id", "film_id", "f")
            + lk("film_category", "inv.film_id", "film_id", "fc") + lk("category", "fc.category_id", "category_id", "cat")
            + lk("customer", "customer_id", "customer_id", "cu") + lk("address", "cu.address_id", "address_id", "ad")
            + lk("city", "ad.city_id", "city_id", "ci")
            + [{"$match": {"$or": [{"ci.city": {"$regex": "^a", "$options": "i"}}, {"ci.city": {"$regex": "-"}}]}},
               {"$group": {"_id": "$cat.name", "total_rental_hours": {"$sum": {"$trunc": [hours, 0]} if truncate else hours}}},
               {"$sort": {"total_rental_hours": -1, "_id": 1}}, {"$limit": 1},
               {"$project": {"_id": 0, "category_name": "$_id", "total_rental_hours": 1}}])


P_FRAC, P_TRUNC = pipeline(False), pipeline(True)
r_frac = list(db[C("rental")].aggregate(P_FRAC))
r_trunc = list(db[C("rental")].aggregate(P_TRUNC))
cities_mongo = sorted(d["city"] for d in db[C("city")].find({"$or": [{"city": {"$regex": "^a", "$options": "i"}}, {"city": {"$regex": "-"}}]}))
assert cities_mongo == pre["matching_cities_sqlite"], "LIKE translation selects different cities"
vf, okf = score(pd.DataFrame(r_frac), rec)
vt, okt = score(pd.DataFrame(r_trunc), rec)
print(IID, "ref_sql_ok", ref_ok, "| fractional mql_ok", okf, r_frac, vf, "\n  | truncated mql_ok", okt, r_trunc, vt)

write_artifact(IID, {
  "instance_id": IID, "db": rec["db"], "question": rec["question"],
  "pattern": "long inner-join chain + date/time arithmetic on text + case-insensitive LIKE + group/order/limit",
  "gold_sql": gold_sql, "gold_sql_reproduces_gold_csv": ref_ok,
  "ambiguity_documented_not_silently_resolved": {
     "issue": "Gold SQL truncates each rental's hours to INTEGER before summing; gold CSV A (16858.166667) is the FRACTIONAL sum; gold CSV B is only the category name.",
     "primary_mql": "fractional hours (natural reading of 'total rental hours'); passes gold A and gold B",
     "variant_mql": "per-row $trunc like the shipped SQL; passes gold B only (its number is lower than gold A)",
     "same_winner_both": r_frac[0]["category_name"] == r_trunc[0]["category_name"]},
  "prechecks": pre,
  "relational_semantics": "rental->inventory->film->film_category->category and rental->customer->address->city (all inner); keep cities starting with a/A or containing '-'; hours = (return-rental)*24 per rental; sum per category; top 1.",
  "mongodb": {"database": PILOT_DB, "imports": imports, "indexes": indexes, "_id": "default ObjectId",
              "join_strategy": "driving collection = rental (fact table); $lookup+$unwind per dimension on indexed foreign fields; film kept as a pass-through join to preserve INNER JOIN semantics"},
  "mql": {"collection": C("rental"), "method": "aggregate", "pipeline": P_FRAC},
  "mql_variant_truncated_hours": {"collection": C("rental"), "method": "aggregate", "pipeline": P_TRUNC},
  "mongo_result": r_frac, "mongo_result_variant": r_trunc,
  "gold": vf, "gold_variant": vt, "validation_passed": okf,
  "rules_discovered": [
    "Text timestamps ('YYYY-MM-DD HH:MM:SS.000') -> $dateFromString with explicit format incl. .%L, then $dateDiff(unit millisecond)/3.6e6; NULL return_date gives null and $sum ignores it, as SQL SUM does.",
    "SQLite LIKE is case-insensitive: LOWER(x) LIKE 'a%' -> $regex '^a' with $options 'i'; '%-%' -> regex '-'. Verified: the selected city set is identical in SQLite and MongoDB.",
    "SQL CAST(... AS INTEGER) truncation inside an aggregate must be mirrored with $trunc per row, but the gold CSV may encode a different (fractional) variant: keep both variants and report which gold each passes.",
    "Driving collection of a $lookup chain should be the fact table; create indexes on every foreignField of the joined collections.",
  ]})
