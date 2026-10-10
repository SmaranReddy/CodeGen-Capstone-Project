"""local038 - 6-table join with filters, group, count, order, limit 1."""
import sys; sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import pandas as pd
from harness import *

IID = "local038"
rec = manifest_record(IID)
gold_sql = (ROOT / rec["gold_sql_path"]).read_text(encoding="utf-8")
ref = sqlite_df(rec, gold_sql); ref_v, ref_ok = score(ref, rec)

# pre-checks that justify the plan (recorded in the artifact)
sq = sqlite_ro(rec)
pre = {
  "top2_counts": sq.execute("""SELECT COUNT(f.film_id) c FROM actor a JOIN film_actor fa USING(actor_id) JOIN film f USING(film_id)
     JOIN film_category fc USING(film_id) JOIN category c ON c.category_id=fc.category_id JOIN language l ON l.language_id=f.language_id
     WHERE c.name='Children' AND f.release_year BETWEEN 2000 AND 2010 AND f.rating IN ('G','PG') AND l.name='English' AND f.length<=120
     GROUP BY a.actor_id ORDER BY c DESC LIMIT 2""").fetchall(),
  "film.release_year_storage": sq.execute("SELECT DISTINCT typeof(release_year), release_year FROM film").fetchall(),
  "film_category_one_row_per_film": sq.execute("SELECT COUNT(*)=COUNT(DISTINCT film_id) FROM film_category").fetchone()[0],
}
assert pre["top2_counts"][0][0] > pre["top2_counts"][1][0], "LIMIT 1 would be tie-dependent"

db = mongo_db()
imports = [
  import_table(db, rec, IID, "actor", ["actor_id", "first_name", "last_name"]),
  import_table(db, rec, IID, "film_actor", ["actor_id", "film_id"]),
  import_table(db, rec, IID, "film", ["film_id", "release_year", "rating", "length", "language_id"]),
  import_table(db, rec, IID, "film_category", ["film_id", "category_id"]),
  import_table(db, rec, IID, "category", ["category_id", "name"]),
  import_table(db, rec, IID, "language", ["language_id", "name"]),
]
C = lambda t: f"{IID}__{t}"
db[C("film_actor")].create_index("film_id")
idx = [{"collection": C("film_actor"), "key": "film_id"}]

# release_year is TEXT in SQLite, so BETWEEN 2000 AND 2010 is a TEXT comparison -> use string bounds
PIPELINE = [
  {"$match": {"rating": {"$in": ["G", "PG"]}, "length": {"$lte": 120}, "release_year": {"$gte": "2000", "$lte": "2010"}}},
  {"$lookup": {"from": C("language"), "localField": "language_id", "foreignField": "language_id", "as": "lang"}},
  {"$unwind": "$lang"}, {"$match": {"lang.name": "English"}},
  {"$lookup": {"from": C("film_category"), "localField": "film_id", "foreignField": "film_id", "as": "fc"}},
  {"$unwind": "$fc"},
  {"$lookup": {"from": C("category"), "localField": "fc.category_id", "foreignField": "category_id", "as": "cat"}},
  {"$unwind": "$cat"}, {"$match": {"cat.name": "Children"}},
  {"$lookup": {"from": C("film_actor"), "localField": "film_id", "foreignField": "film_id", "as": "fa"}},
  {"$unwind": "$fa"},
  {"$group": {"_id": "$fa.actor_id", "film_count": {"$sum": 1}}},
  {"$sort": {"film_count": -1, "_id": 1}}, {"$limit": 1},
  {"$lookup": {"from": C("actor"), "localField": "_id", "foreignField": "actor_id", "as": "a"}},
  {"$unwind": "$a"},
  {"$project": {"_id": 0, "actor_full_name": {"$concat": ["$a.first_name", " ", "$a.last_name"]}}},
]
res = list(db[C("film")].aggregate(PIPELINE))
pred = pd.DataFrame(res)
v, ok = score(pred, rec)
print(IID, "ref_sql_ok", ref_ok, "| mql_ok", ok, v, res)

write_artifact(IID, {
  "instance_id": IID, "db": rec["db"], "question": rec["question"], "pattern": "multi-table inner join + filter + group + top-1",
  "gold_sql": gold_sql, "gold_sql_reproduces_gold_csv": ref_ok, "prechecks": pre,
  "relational_semantics": "Inner-join actor-film_actor-film-film_category-category-language; keep Children / 2000-2010 / G,PG / English / length<=120; count films per actor; take max.",
  "mongodb": {"database": PILOT_DB, "imports": imports, "indexes": idx, "_id": "default ObjectId",
              "join_strategy": "start at film (most selective filters), $lookup+$unwind for each join; film_actor fan-out reproduced by $unwind"},
  "mql": {"collection": C("film"), "method": "aggregate", "pipeline": PIPELINE},
  "mongo_result": res, "gold": v, "validation_passed": ok,
  "rules_discovered": [
    "Type affinity: film.release_year is TEXT, so SQLite compares BETWEEN 2000 AND 2010 as text; Mongo needs string bounds (numeric bounds match nothing).",
    "Start the pipeline from the most selective collection, not from the SQL FROM table; join order does not change inner-join results.",
    "LIMIT 1 is only safe after verifying the top group is unique (pre-check: 3 vs 2); otherwise the item is tie-dependent and must be flagged. Add a deterministic secondary sort anyway.",
    "Pure pass-through key tables (film_category, film_actor) still need $lookup; fan-out is preserved by $unwind.",
  ]})
