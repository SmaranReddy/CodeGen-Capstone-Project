"""local201 - first 10 words (a-z) starting 'r', length 4-5, that have an anagram.

No shipped gold SQL: a labelled reference SQL is authored and verified against the gold
CSV first. A word's anagram signature is its characters sorted (case-sensitive);
anagram_count = how many *other* word-list entries share that signature. Words are
restricted to length 4-5 and prefix 'r', ordered alphabetically, first 10. Reproduces
gold variant `_a` (word / anagram_count).
"""
import pandas as pd

from conversion_kit import import_table, manifest_record, matched_reason, score, sqlite_df

IID = "local201"

REFERENCE_SQL = """
WITH s AS (
  SELECT w.words AS word,
    (SELECT group_concat(c, '') FROM (
        SELECT substr(w.words, v, 1) AS c
        FROM (SELECT 1 v UNION ALL SELECT 2 UNION ALL SELECT 3 UNION ALL SELECT 4 UNION ALL SELECT 5)
        WHERE v <= length(w.words)
        ORDER BY c)) AS sig
  FROM word_list w
  WHERE length(w.words) BETWEEN 4 AND 5)
SELECT word,
  (SELECT COUNT(*) FROM s s2 WHERE s2.sig = s.sig AND s2.word <> s.word) AS anagram_count
FROM s
WHERE substr(word, 1, 1) = 'r'
  AND (SELECT COUNT(*) FROM s s2 WHERE s2.sig = s.sig AND s2.word <> s.word) >= 1
ORDER BY word LIMIT 10
"""

PIPELINE = [
    {"$match": {"words": {"$type": "string"}}},
    {"$addFields": {"n": {"$strLenCP": "$words"}}},
    {"$match": {"n": {"$in": [4, 5]}}},
    {"$addFields": {"chars": {"$map": {"input": {"$range": [0, "$n"]}, "as": "i",
                                       "in": {"$substrCP": ["$words", "$$i", 1]}}}}},
    {"$addFields": {"sig": {"$reduce": {
        "input": {"$sortArray": {"input": "$chars", "sortBy": 1}},
        "initialValue": "", "in": {"$concat": ["$$value", "$$this"]}}}}},
    {"$group": {"_id": "$sig", "grp": {"$push": "$words"}, "n": {"$sum": 1}}},
    {"$match": {"n": {"$gte": 2}}},
    {"$addFields": {"arr": "$grp"}},
    {"$unwind": "$grp"},
    {"$addFields": {"anagram_count": {"$subtract": ["$n", {"$size": {"$filter": {
        "input": "$arr", "as": "x", "cond": {"$eq": ["$$x", "$grp"]}}}}]}}},
    {"$match": {"anagram_count": {"$gte": 1}, "$expr": {"$eq": [{"$substrCP": ["$grp", 0, 1]}, "r"]}}},
    {"$sort": {"grp": 1}},
    {"$limit": 10},
    {"$project": {"_id": 0, "word": "$grp", "anagram_count": 1}},
]

COLS = ["word", "anagram_count"]


def run(reg):
    rec = manifest_record(IID)
    ref = sqlite_df(rec, REFERENCE_SQL)
    ref_v, ref_ok = score(ref, rec)

    words = import_table(reg, rec, IID, "word_list", ["words"])
    coll = reg.db[words["collection"]]
    res = list(coll.aggregate(PIPELINE, allowDiskUse=True))
    pred = pd.DataFrame(res, columns=COLS)
    v, ok = score(pred, rec)

    artifact = {
        "pattern": "in-document char array + $sortArray signature + anagram group count + top-10",
        "gold_sql": None, "gold_sql_reproduces_gold_csv": "n/a (no shipped gold SQL)",
        "reference_sql": REFERENCE_SQL, "reference_sql_matches_gold": ref_ok,
        "relational_semantics": ("Words of length 4-5 starting with 'r' that have >=1 anagram (another list entry with the "
                                 "same case-sensitive letters); anagram_count counts the other anagrams; first 10 by word."),
        "source_schema": {"word_list": ["words"]},
        "mongodb": {"database": "codegen_capstone_pilot", "imports": [words], "_id": "default ObjectId"},
        "mql": {"collection": coll.name, "method": "aggregate", "options": {"allowDiskUse": True}, "pipeline": PIPELINE},
        "mongo_result": res,
        "validation": {"per_gold": v, "matched_any_gold": ok},
        "assumptions": ["Anagram = same multiset of case-sensitive characters (same length implied).",
                        "anagram_count counts other entries, so duplicate list entries are counted individually.",
                        "Validated against gold variant _a (word / anagram_count)."],
    }
    return ("passed" if ok else "failed"), matched_reason(v, ok), artifact
