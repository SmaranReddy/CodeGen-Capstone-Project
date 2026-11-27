"""Eligibility audit of every candidate RAG / fine-tuning source against the 892-item NL->MQL
evaluation set of MQL-0006 (kaggle/nl2mql_spider_dev_1034_bundle_v2).

Purpose: decide, per candidate, whether it may enter a retrieval corpus (and a future
fine-tuning set) for an experiment scored on those 892 items. Nothing is rebuilt, converted,
generated or executed; every input is read only. Outputs go to a NEW directory:

  outputs/nl2mql_spider_dev_1034_rag/corpus_audit_v1/candidates.jsonl  one row per candidate
  outputs/nl2mql_spider_dev_1034_rag/corpus_audit_v1/summary.json      counts + input hashes

Candidates
  * data/rag/corpus_v1 example docs (280)           question->MQL, saved model predictions
  * data/finetune/ft_v1 train + dev_internal (1,864) question->MQL (Spider rows + Kaggle rows)
  * data/rag/sources/mongodb_guides.yaml cards (18) documentation (RAG-only, not supervised)
  * data/rag/corpus_v1 schema docs (80)              schema descriptions (prompt context only)

Supervised pairs are accepted only if ALL of these hold:
  (1) not an evaluation item: record id, normalised question and database all disjoint from
      the 892 (Spider-dev rows are mapped to evaluation ids through dev_index);
  (2) label verified against a real source database, or by a converter that passed its
      documented remediation (sql2mql-v2, docs/nl2mql_next_steps.md P4 V1-V3);
  (3) not on a known-wrong list (docs/sql2mql_independent_audit.md).
Fails closed (exit 1) if the evaluation bundle or any expected count does not match.

Usage: python scripts/nl2mql/audit_rag_corpus_dev1034.py [--out DIR]
"""
import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
BUNDLE_ZIP = ROOT / "kaggle/nl2mql_spider_dev_1034_bundle_v2.zip"
BUNDLE_DIR = ROOT / "kaggle/nl2mql_spider_dev_1034_bundle_v2"
BUNDLE_SHA256 = "e9986dbef13cab8e1e047465b04ff8593d7b54f700cca4bcd1ed28f80d3b0760"
CORPUS = ROOT / "data/rag/corpus_v1/corpus.jsonl"
GUIDES = ROOT / "data/rag/sources/mongodb_guides.yaml"
EXPANDED = ROOT / "data/processed/spider1_nl_mql_expanded.jsonl"
FT_FILES = (ROOT / "data/finetune/ft_v1/train.jsonl", ROOT / "data/finetune/ft_v1/dev_internal.jsonl")
FAILURE_CLASSES = ROOT / "outputs/audit_sql2mql_v1/kaggle_failure_classes.jsonl"
DATE_EQUALITY = ROOT / "outputs/audit_sql2mql_v1/datetime_date_equality.jsonl"
SQL2MQL_V2_DIR = ROOT / "scripts/finetune_v2"   # sql2mql-v2 (2026-10-30)
OUT = ROOT / "outputs/nl2mql_spider_dev_1034_rag/corpus_audit_v1"
EXPECTED = {"eval": 892, "eval_dbs": 17, "corpus_examples": 280, "guides": 18, "schema": 80,
            "ft_rows": 1864}


def sha256_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def read_jsonl(p):
    return [json.loads(ln) for ln in Path(p).read_text(encoding="utf-8").splitlines() if ln.strip()]


def loose(s):
    return " ".join(re.findall(r"[a-z0-9]+", (s or "").lower()))


def rel(p):
    return Path(p).relative_to(ROOT).as_posix()


def load_eval():
    if sha256_file(BUNDLE_ZIP) != BUNDLE_SHA256:
        sys.exit(f"FAIL: {rel(BUNDLE_ZIP)} sha256 is not {BUNDLE_SHA256}")
    man = json.loads((BUNDLE_DIR / "bundle_manifest.json").read_text(encoding="utf-8"))
    path = BUNDLE_DIR / "data/eval_records.jsonl"
    want = man["files"]["data/eval_records.jsonl"]["sha256"]
    if sha256_file(path) != want:
        sys.exit("FAIL: extracted eval_records.jsonl does not match the bundle manifest")
    rows = read_jsonl(path)
    return rows, man


def spider_dev_eval_id(dev_index):
    return f"spider_dev_{int(dev_index):04d}"


def supervised_row(source, rid, question, mql, db_id, licence, status, provenance, ev, extra_reasons):
    reasons = list(extra_reasons)
    dev_eval_id = provenance.get("eval_id")
    in_eval_id = bool(dev_eval_id and dev_eval_id in ev["ids"])
    in_eval_q = loose(question) in ev["questions"]
    in_eval_db = db_id in ev["dbs"]
    if in_eval_id:
        reasons.append(f"evaluation item {dev_eval_id}")
    if in_eval_q and not in_eval_id:
        reasons.append("question matches an evaluation question (normalised)")
    if in_eval_db:
        reasons.append(f"database {db_id} is an evaluation database")
    return {"candidate_kind": "supervised_pair", "source": source, "record_id": rid,
            "db_id": db_id, "question": question, "mql": mql, "licence": licence,
            "verification_status": status, "provenance": provenance,
            "leakage": {"eval_id_match": in_eval_id, "eval_question_match": in_eval_q,
                        "eval_database_match": in_eval_db},
            "decision": "exclude" if reasons else "accept", "reasons": reasons}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args()

    eval_rows, man = load_eval()
    ev = {"ids": {r["id"] for r in eval_rows}, "questions": {loose(r["question"]) for r in eval_rows},
          "dbs": {r["db_id"] for r in eval_rows}}
    errors = []
    if len(eval_rows) != EXPECTED["eval"] or len(ev["dbs"]) != EXPECTED["eval_dbs"]:
        errors.append(f"evaluation set is {len(eval_rows)} rows / {len(ev['dbs'])} dbs")
    if any(r["source"].get("file") != "dev.json" or r["id"] != spider_dev_eval_id(r["source"]["index"])
           for r in eval_rows):
        errors.append("evaluation ids are not spider_dev_<dev.json index>")

    dev_index = {r["id"]: r["dev_index"] for r in read_jsonl(EXPANDED)}
    corpus = read_jsonl(CORPUS)
    by_kind = Counter(d["kind"] for d in corpus)
    rows = []

    # --- corpus_v1 examples: saved model predictions on Spider-dev items
    for d in (d for d in corpus if d["kind"] == "example"):
        pid = d["provenance"]["dataset_id"]
        prov = {"file": rel(CORPUS), "doc_id": d["doc_id"], "dataset_id": pid,
                "dev_index": dev_index.get(pid), "source_model": d["provenance"]["source_model"],
                "eval_id": spider_dev_eval_id(dev_index[pid]) if pid in dev_index else None}
        rows.append(supervised_row(
            "rag_corpus_v1_example", d["doc_id"], d["question"], d["mql"], d["db_id"],
            "CC BY-SA 4.0 (Spider questions); MQL = model output", "execution_verified_source_db "
            "(saved prediction re-executed on Atlas vs Spider gold SQL)", prov, ev, []))

    # --- ft-v1 rows
    bad = {}
    for x in read_jsonl(FAILURE_CLASSES):
        bad.setdefault(x["id"], set()).add(x["class"])
    date_eq = {x["id"] for x in read_jsonl(DATE_EQUALITY)}
    v2_exists = SQL2MQL_V2_DIR.exists()
    ft_count = 0
    for f in FT_FILES:
        for r in read_jsonl(f):
            ft_count += 1
            sid = r["source_record_ids"][0] if r["source_record_ids"] else r["source_pair_id"]
            prov = {"file": rel(f), "id": r["id"], "source_dataset": r["source_dataset"],
                    "source_record_ids": r["source_record_ids"], "split": r["split"],
                    "candidate_source": r["validation"].get("candidate_source")}
            extra = []
            if r["source_dataset"].startswith("spider1_dev"):
                prov["dev_index"] = dev_index.get(sid)
                prov["eval_id"] = spider_dev_eval_id(dev_index[sid]) if sid in dev_index else None
            else:
                if prov["candidate_source"] == "sql2mql-v1":
                    extra.append("label from sql2mql-v1 (never re-derived and re-verified by a "
                                 "remediated converter)")
                if r["validation"]["status"] == "fixture_equivalence_verified":
                    extra.append("fixture_equivalence_verified only: no source database "
                                 "(audit: verifier accepted 10/20 wrong translations)")
                cls = sorted(c for c in bad.get(r["id"], ()) if c.startswith(("B1_", "B2_")))
                if cls:
                    extra.append("known-wrong label: " + ",".join(cls))
                if r["id"] in date_eq:
                    extra.append("SQL-level datetime = date equality error")
            rows.append(supervised_row(
                f"ft_v1:{r['source_dataset']}", r["id"], r["question"], r["target_mql"], r["db_id"],
                (r.get("provenance") or {}).get("license", "CC BY-SA 4.0 (Spider questions)"),
                r["validation"]["status"], prov, ev, extra))

    # --- guide cards (documentation, RAG-only)
    cards = yaml.safe_load(GUIDES.read_text(encoding="utf-8"))["cards"]
    for c in cards:
        ex = c.get("example", {})
        dbs = sorted({x for x in ((ex.get("collection") or "").split("__")[0], ex.get("check_db")) if x})
        reasons = [f"example uses evaluation database {db}" for db in dbs if db in ev["dbs"]]
        reasons.append("licence of the paraphrased MongoDB manual pages not recorded in the source file")
        rows.append({"candidate_kind": "documentation", "source": "mongodb_guides_v1",
                     "record_id": c["id"], "title": c["title"], "urls": c["sources"],
                     "example_databases": dbs, "licence": None,
                     "verification_status": "example executed on Atlas at corpus-v1 build (2026-10-20)",
                     "decision": "exclude" if len(reasons) > 1 else "pending_licence_check",
                     "reasons": reasons})

    # --- schema docs (prompt context; never retrieved in MQL-0003..0005)
    for d in (d for d in corpus if d["kind"] == "schema"):
        rows.append({"candidate_kind": "schema", "source": "rag_corpus_v1_schema",
                     "record_id": d["doc_id"], "db_id": d["db_id"],
                     "decision": "not_a_retrieval_document",
                     "reasons": ["target schema is already in the prompt; other databases' schemas "
                                 "carry no query knowledge"]})

    n_ex = sum(r["source"] == "rag_corpus_v1_example" for r in rows)
    if n_ex != EXPECTED["corpus_examples"]:
        errors.append(f"corpus examples {n_ex}")
    if len(cards) != EXPECTED["guides"] or by_kind["schema"] != EXPECTED["schema"]:
        errors.append(f"guides {len(cards)} / schema {by_kind['schema']}")
    if ft_count != EXPECTED["ft_rows"]:
        errors.append(f"ft-v1 rows {ft_count}")
    if errors:
        sys.exit("FAIL:\n  " + "\n  ".join(errors))

    sup = [r for r in rows if r["candidate_kind"] == "supervised_pair"]
    def tally(group):
        return {"total": len(group), "accepted": sum(r["decision"] == "accept" for r in group),
                "excluded": sum(r["decision"] == "exclude" for r in group),
                "reasons": dict(Counter(re.sub(r"spider_dev_\d+|database \S+ is", lambda m: (
                    "<eval id>" if m.group(0).startswith("spider") else "database <db> is"), x)
                    for r in group for x in r["reasons"]).most_common())}
    groups = {}
    for r in sup:
        groups.setdefault(r["source"], []).append(r)
    unique_pairs = {(loose(r["question"]), json.dumps(r["mql"], sort_keys=True)) for r in sup}
    docs = [r for r in rows if r["candidate_kind"] == "documentation"]
    summary = {
        "audit": "nl2mql_spider_dev_1034_rag corpus eligibility audit v1",
        "script": rel(Path(__file__)),
        "evaluation_bundle": {"path": rel(BUNDLE_ZIP), "sha256": BUNDLE_SHA256,
                              "records": len(eval_rows), "databases": sorted(ev["dbs"])},
        "inputs_sha256": {rel(p): sha256_file(p) for p in
                          (CORPUS, GUIDES, EXPANDED, *FT_FILES, FAILURE_CLASSES, DATE_EQUALITY)},
        "sql2mql_v2_present": v2_exists,
        "supervised_pairs": {"rows": len(sup), "unique_question_mql_pairs": len(unique_pairs),
                             "accepted": sum(r["decision"] == "accept" for r in sup),
                             "by_source": {k: tally(v) for k, v in sorted(groups.items())},
                             "spider_rows_that_are_eval_items_by_id":
                                 sum(r["leakage"]["eval_id_match"] for r in sup)},
        "documentation": {"total": len(docs),
                          "excluded_eval_database": sum(r["decision"] == "exclude" for r in docs),
                          "pending_licence_check": sum(r["decision"] == "pending_licence_check" for r in docs)},
        "schema_docs": by_kind["schema"],
        "native_nl2mql_sources_on_disk": sorted(x.name for x in (ROOT / "data/raw").iterdir()
                                                 if re.search(r"aptmql|sm3|tend", x.name, re.I)),
        "verdict": None,
    }
    acc = summary["supervised_pairs"]["accepted"]
    summary["verdict"] = ("BLOCKED: 0 eligible supervised question->MQL pairs" if acc == 0
                          else f"{acc} eligible supervised pairs")
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "candidates.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    (a.out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("supervised_pairs", "documentation", "verdict")}, indent=1))


if __name__ == "__main__":
    main()
