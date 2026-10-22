"""Build the versioned Week 3 RAG corpus (data/rag/corpus_v1/) deterministically.

Document kinds
  example  NL question -> MQL pairs from TRAIN-partition databases only. The MQL is a SAVED
           zero-shot prediction (no MQL is written by hand). One per train item, taken from the
           first model in EXAMPLE_SOURCES whose saved record is correct AND whose raw output,
           re-executed now through the benchmark scorer (run_eval.score_output: parse ->
           read-only validation -> Atlas aggregate -> exec_match vs the SQLite gold result),
           is still correct. Items flagged `ambiguous` are skipped.
  guide    MongoDB aggregation guide cards (data/rag/sources/mongodb_guides.yaml), each a
           paraphrase of cited official manual pages with an example executed on Atlas now.
  schema   One document per Atlas collection of every database (fields, observed BSON types,
           declared SQL types, foreign keys). Inference-time information; NOT retrieved (the
           target schema is already in the prompt verbatim), kept for completeness/inspection.

Outputs: corpus.jsonl (byte-deterministic for fixed inputs), corpus_manifest.json (input
hashes, counts, per-kind content hash), build_log.json (timestamps, rejections).

Usage: python scripts/rag/build_corpus.py [--check]   (--check: rebuild in memory, compare)
"""
import argparse
import json
import sys
import time

import yaml

import common as C
from common import K
import leakage
import patterns as PT

import run_eval as R  # noqa: E402
from eval_kit import validate_mql  # noqa: E402

VERSION = "corpus-v1"
EXEC_CFG = yaml.safe_load((C.ROOT / "configs/evaluation_spider1_expanded.yaml")
                          .read_text(encoding="utf-8"))["execution"]


def _keys(obj):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield k
            yield from _keys(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _keys(v)


def check_guide_pipeline(ex, colls):
    """Guide examples get the same read-only contract as model outputs: no forbidden operator
    at any depth; a collection example must pass validate_mql restricted to its own (train)
    database; an inline example must start with $documents and reference no collection."""
    hit = R.forbidden_anywhere(ex["pipeline"])
    if hit:
        raise ValueError(f"forbidden operator {hit}")
    coll = ex.get("collection")
    if coll is None:
        if not ex["pipeline"] or "$documents" not in ex["pipeline"][0]:
            raise ValueError("inline example must start with $documents")
        if any(k in ("from", "coll", "$unionWith", "$lookup") for k in _keys(ex["pipeline"])):
            raise ValueError("inline example references a collection")
        return
    db = coll.split("__")[0]
    ok, why, _ = validate_mql({"collection": coll, "pipeline": ex["pipeline"]},
                              {c for c in colls if c.startswith(db + "__")})
    if not ok:
        raise ValueError(f"validate_mql: {why}")


def schema_docs(collections, db_part):
    tables = {t["db_id"]: t for t in K.load_json(K.SPIDER / "tables.json")}
    docs = []
    for name, info in sorted(collections.items()):
        db = info["db_id"]
        if db not in db_part:
            continue
        tj = tables[db]
        cols, tabs = tj["column_names_original"], tj["table_names_original"]
        fks = [f"{db}__{tabs[cols[x][0]]}.{cols[x][1]} -> {db}__{tabs[cols[y][0]]}.{cols[y][1]}"
               for x, y in tj["foreign_keys"]]
        fks = [f for f in fks if f.startswith(name + ".")]
        lines = [f"Collection {name} (database {db})"]
        lines += [f"  {f}: {'/'.join(t)} [{info['declared'].get(f) or '?'}]"
                  for f, t in info["fields"].items()]
        lines += [f"  FK {f}" for f in fks]
        content = "\n".join(lines)
        docs.append({"doc_id": f"schema_{name}", "kind": "schema", "db_id": db,
                     "split": db_part[db], "collection": name, "content": content,
                     "retrieval_text": "", "patterns": [], "operators": [],
                     "provenance": {"source": "data/interim/spider1/atlas_collections.json + "
                                              "official Spider tables.json foreign keys"}})
    return docs


def guide_docs(mdb, src, db_part, colls):
    cfg = yaml.safe_load(src.read_text(encoding="utf-8"))
    docs, problems = [], []
    for card in cfg["cards"]:
        ex = card["example"]
        coll = ex.get("collection")
        try:
            check_guide_pipeline(ex, colls)
            if coll is None:
                got = list(mdb.aggregate(ex["pipeline"], maxTimeMS=EXEC_CFG["timeout_ms"]))
            else:
                db = coll.split("__")[0]
                if db_part.get(db) != "train":
                    raise ValueError(f"guide example uses non-train database {db}")
                got = list(mdb[coll].aggregate(ex["pipeline"], maxTimeMS=EXEC_CFG["timeout_ms"]))
        except Exception as e:  # noqa: BLE001
            problems.append(f"{card['id']}: execution failed: {type(e).__name__}: {e}")
            continue
        _, rows = K.docs_to_rows(got)
        if coll is None:
            _, exp_rows = K.docs_to_rows([dict(d) for d in ex["expected"]])
            how = "inline $documents result equals `expected`"
        else:
            _, exp_rows = K.run_sql(ex["check_db"], ex["check_sql"])
            how = f"result exec_matches check_sql on SQLite {ex['check_db']}"
        ok, why = K.exec_match(rows, [list(r) for r in exp_rows])
        if not ok:
            problems.append(f"{card['id']}: example result mismatch ({why})")
            continue
        shown = C.compact(ex["pipeline"])
        content = (f"[{card['title']}] {card['text'].strip()}\nExample pipeline: {shown}")
        docs.append({"doc_id": card["id"], "kind": "guide", "db_id": None, "split": "n/a",
                     "title": card["title"], "content": content,
                     "retrieval_text": f"{card['title']} {card['text']}",
                     "patterns": sorted(card["patterns"]),
                     "operators": sorted(set(card["operators"]) | PT.mql_operators(ex["pipeline"])),
                     "provenance": {"source": "data/rag/sources/mongodb_guides.yaml",
                                    "guides_version": cfg["version"], "urls": card["sources"],
                                    "retrieved": cfg["retrieved"],
                                    "example_verified": how}})
    return docs, problems


def example_docs(mdb, colls, rows, db_part, collections):
    infos = {}
    by_model = {}
    for label in C.EXAMPLE_SOURCES:
        path = C.BASELINE_DIR / label / "executions.jsonl"
        lines = [ln for ln in path.read_bytes().splitlines() if ln.strip()]
        by_model[label] = {json.loads(ln)["id"]: (ln, json.loads(ln)) for ln in lines}
    docs, rejected = [], []
    for ex in rows:
        if db_part[ex["db_id"]] != "train":
            continue
        if "ambiguous" in ex.get("flags", []):
            rejected.append({"id": ex["id"], "why": "flagged ambiguous"})
            continue
        allowed = {c for c in colls if c.startswith(ex["db_id"] + "__")}
        chosen = None
        for label in C.EXAMPLE_SOURCES:
            hit = by_model[label].get(ex["id"])
            if not hit or not hit[1]["result"]["correct"]:
                continue
            line, rec = hit
            if rec["question"] != ex["question"] or rec["expected_rows"] != ex["gold_rows"]:
                rejected.append({"id": ex["id"], "model": label, "why": "record/dataset mismatch"})
                continue
            res, extra = R.score_output(mdb, rec["raw_text"], ex, EXEC_CFG, allowed)
            if not res["correct"]:
                rejected.append({"id": ex["id"], "model": label,
                                 "why": f"re-execution not correct: {res['category']}"})
                continue
            chosen = (label, line, rec, extra["mql"], res)
            break
        if chosen is None:
            rejected.append({"id": ex["id"], "why": "no verified correct prediction"})
            continue
        label, line, rec, mql, res = chosen
        info = infos.setdefault(ex["db_id"], PT.schema_info(ex["db_id"], collections))
        shown = C.compact(mql)
        docs.append({
            "doc_id": f"ex_{ex['id']}", "kind": "example", "db_id": ex["db_id"], "split": "train",
            "question": ex["question"], "masked_question": PT.mask_question(ex["question"], info),
            "mql": mql, "hardness": ex["hardness"],
            "content": f"Question: {ex['question']}\nMQL: {shown}",
            "retrieval_text": ex["question"],
            "patterns": sorted(PT.sql_patterns(ex["gold_sql"]) |
                               ({"text_number"} if info["text_number_fields"] else set())),
            "operators": sorted(PT.mql_operators(mql["pipeline"])),
            "provenance": {"dataset_id": ex["id"], "source_model": label,
                           "source_file": f"outputs/evaluation_spider1_expanded/{label}/executions.jsonl",
                           "source_record_sha256": C.sha256_bytes(line),
                           "prompt_version": rec["prompt_version"],
                           "expected_rows_sha256": ex["expected_rows_sha256"],
                           "verification": f"re-executed on Atlas: {res['reason']}"}})
    return docs, rejected


def build(write=True):
    rows = K.read_jsonl(C.DATASET)
    db_part, _ = C.load_split()
    collections = C.load_collections()
    client = K.mongo_client()
    try:
        mdb = client[K.ATLAS_DB]
        colls = mdb.list_collection_names()
        gdocs, gproblems = guide_docs(mdb, C.GUIDES_SRC, db_part, colls)
        edocs, rejected = example_docs(mdb, colls, rows, db_part, collections)
    finally:
        client.close()
    sdocs = schema_docs(collections, db_part)
    if gproblems:
        sys.exit("guide verification failed:\n  " + "\n  ".join(gproblems))
    corpus = gdocs + edocs + sdocs
    cards = yaml.safe_load(C.GUIDES_SRC.read_text(encoding="utf-8"))["cards"]
    errs = leakage.check(corpus, rows, db_part, collections, cards)
    if errs:
        sys.exit("LEAKAGE CHECK FAILED:\n  " + "\n  ".join(errs[:50]))
    body = "".join(C.canon(d) + "\n" for d in corpus).encode("utf-8")
    kinds = {k: [d for d in corpus if d["kind"] == k] for k in ("example", "guide", "schema")}
    manifest = {
        "version": VERSION, "builder": "scripts/rag/build_corpus.py",
        "command": "python scripts/rag/build_corpus.py",
        "corpus_sha256": C.sha256_bytes(body),
        "inputs": {
            "dataset": {"path": "data/processed/spider1_nl_mql_expanded.jsonl", "sha256": C.sha256_file(C.DATASET)},
            "split_manifest": {"path": "data/splits_expanded/split_manifest.json",
                               "sha256": C.sha256_file(C.SPLIT_DIR / "split_manifest.json")},
            "guides": {"path": "data/rag/sources/mongodb_guides.yaml", "sha256": C.sha256_file(C.GUIDES_SRC)},
            "collections": {"path": "data/interim/spider1/atlas_collections.json",
                            "sha256": C.sha256_file(C.COLLECTIONS)},
            **{f"predictions_{lab}": {"path": f"outputs/evaluation_spider1_expanded/{lab}/executions.jsonl",
                                      "sha256": C.sha256_file(C.BASELINE_DIR / lab / "executions.jsonl")}
               for lab in C.EXAMPLE_SOURCES}},
        "counts": {k: len(v) for k, v in kinds.items()},
        "examples_by_source_model": {lab: sum(d["provenance"]["source_model"] == lab for d in kinds["example"])
                                     for lab in C.EXAMPLE_SOURCES},
        "example_databases": sorted({d["db_id"] for d in kinds["example"]}),
        "train_items": sum(1 for r in rows if db_part[r["db_id"]] == "train"),
        "train_items_without_example": len(rejected and {x["id"] for x in rejected
                                                         if x["why"] in ("no verified correct prediction",
                                                                         "flagged ambiguous")}),
        "leakage_check": "passed (scripts/rag/leakage.py)",
        "policy": ("examples: train databases only, saved predictions re-executed and exec_match-correct "
                   "now; no hand-written or generated gold MQL. guides: official MongoDB manual "
                   "paraphrases with executed examples. schema: inference-time information, not retrieved."),
    }
    if write:
        C.CORPUS_DIR.mkdir(parents=True, exist_ok=True)
        (C.CORPUS_DIR / "corpus.jsonl").write_bytes(body)
        (C.CORPUS_DIR / "corpus_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        (C.CORPUS_DIR / "build_log.json").write_text(json.dumps(
            {"built_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "rejected": rejected}, indent=1) + "\n",
            encoding="utf-8")
    return body, manifest, rejected


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="rebuild in memory and compare with disk")
    a = ap.parse_args()
    body, manifest, rejected = build(write=not a.check)
    if a.check:
        same = (C.CORPUS_DIR / "corpus.jsonl").read_bytes() == body
        print("corpus reproduces byte-for-byte" if same else "MISMATCH with data/rag/corpus_v1/corpus.jsonl")
        sys.exit(0 if same else 1)
    print(json.dumps({k: manifest[k] for k in ("corpus_sha256", "counts", "examples_by_source_model",
                                               "train_items", "train_items_without_example")}, indent=1))


if __name__ == "__main__":
    main()
