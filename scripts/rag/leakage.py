"""Automated leakage checks for the RAG corpus (run by build_corpus.py and by the tests).

A retrievable document (kind example/guide) must not carry anything identifying a validation
or test example. Checked over EVERY text field a retriever or prompt can use (content,
retrieval_text, question, masked_question, the serialized mql):
  * built from a held-out example id, or tagged with a non-train database;
  * a held-out question or gold SQL (case-, whitespace- and punctuation-insensitive match);
  * any held-out database name or held-out collection name (this also covers held-out
    predictions and gold MQL, which can only refer to held-out collections);
  * a guide card whose example starting collection / check_db is held out (`guide_cards`).
Gold result VALUES are not string-matched (short values such as 3 or 'France' are not
identifying); they cannot enter the corpus because examples are built from train items only.
Schema documents are inference-time information and may describe any database, but must not
contain any question, SQL or answer text.

check(corpus, dataset_rows, db_part, collections, guide_cards=()) -> list of violations.
"""
import json
import re

RETRIEVABLE = ("example", "guide")


def _norm(s):
    return re.sub(r"\s+", " ", s.strip().lower())


def _loose(s):
    """lowercase alphanumeric tokens only, so punctuation/spacing variants collapse."""
    return " ".join(re.findall(r"[a-z0-9]+", s.lower()))


def _blob(d):
    mql = json.dumps(d["mql"], ensure_ascii=False) if d.get("mql") else ""
    return " ".join([d.get("content", ""), d.get("retrieval_text", ""), d.get("question", ""),
                     d.get("masked_question", ""), mql])


def check(corpus, dataset_rows, db_part, collections, guide_cards=()):
    errs = []
    heldout = [r for r in dataset_rows if db_part[r["db_id"]] != "train"]
    heldout_ids = {r["id"] for r in heldout}
    heldout_dbs = {db for db, p in db_part.items() if p != "train"}
    heldout_colls = {c for c, i in collections.items() if i["db_id"] in heldout_dbs}
    questions = [_loose(r["question"]) for r in heldout]
    sqls = [_loose(r["gold_sql"]) for r in heldout]
    all_questions = [_loose(r["question"]) for r in dataset_rows]
    seen = set()
    for d in corpus:
        did = d["doc_id"]
        if did in seen:
            errs.append(f"duplicate doc_id {did}")
        seen.add(did)
        blob = _norm(_blob(d))
        loose = " " + _loose(_blob(d)) + " "
        if d["kind"] in RETRIEVABLE:
            if d.get("db_id") and db_part.get(d["db_id"]) != "train":
                errs.append(f"{did}: from non-train database {d['db_id']}")
            src = (d.get("provenance") or {}).get("dataset_id")
            if src and src in heldout_ids:
                errs.append(f"{did}: built from held-out example {src}")
            for c in heldout_colls:
                if c.lower() in blob:
                    errs.append(f"{did}: mentions held-out collection {c}")
            for db in heldout_dbs:
                if re.search(rf"\b{re.escape(db.lower())}\b", blob):
                    errs.append(f"{did}: mentions held-out database {db}")
            for q in questions:
                if f" {q} " in loose:
                    errs.append(f"{did}: contains a held-out question")
                    break
            for s in sqls:
                if f" {s} " in loose:
                    errs.append(f"{did}: contains a held-out gold SQL")
                    break
            if d["kind"] == "example" and d.get("split") != "train":
                errs.append(f"{did}: example not tagged train")
        elif d["kind"] == "schema":
            for q in all_questions:
                if f" {q} " in loose:
                    errs.append(f"{did}: schema doc contains a benchmark question")
                    break
            if "gold" in blob or "select " in blob:
                errs.append(f"{did}: schema doc contains SQL/gold text")
        else:
            errs.append(f"{did}: unknown kind {d['kind']}")
    for card in guide_cards:
        ex = card.get("example", {})
        start_db = (ex.get("collection") or "").split("__")[0]
        for db in filter(None, (start_db, ex.get("check_db"))):
            if db_part.get(db) != "train":
                errs.append(f"{card['id']}: guide example uses non-train database {db}")
    return errs
