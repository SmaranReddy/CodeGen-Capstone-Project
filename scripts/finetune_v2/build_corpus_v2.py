"""provisional_corpus_v2 (plan §E.2b, step S11b): sql2mql-v2.1 + verifier 3.1 labels for spider-pool-v3.

    python -B scripts/finetune_v2/build_corpus_v2.py finalize

Inputs (all read-only, hash-recorded):
  * outputs/sql2mql_v2_1/provisional_corpus_v2/records.jsonl  (run_pilot_v4.py corpus-run: one record per candidate)
  * outputs/sql2mql_v2_1/provisional_corpus_v2/candidate_ids.json (the 5,045 corpus-v1 eligibility ids)
  * data/processed/spider_pool_v3/{train.jsonl, manifest.json}
  * corpus v1 (outputs/sql2mql_v2/provisional_corpus_v1/) for the id diff only; never modified.
Corpus rule: INCLUDED iff the record is retained in spider-pool-v3 train, verifier 3.1 ACCEPTED, not
multi_table_3plus, not a pilot cross-review pre-exclusion, and no evaluation-set leakage. No corrected labels.

Leakage screen: question / SQL hashes against corpus v1's evaluation sources EXCEPT data/splits_expanded/test.jsonl,
which is never opened (protected 65-item holdout). That file's items come from Spider dev, which is covered by
data/processed/spider_sql_v1/dev_full.jsonl (all 1,034 Spider dev items); in addition the db-level check proves no
corpus record is on any Spider dev database.
"""
import hashlib
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.dont_write_bytecode = True

CORPUS_VERSION = "provisional_corpus_v2"
OUT = ROOT / "outputs/sql2mql_v2_1/provisional_corpus_v2"
V1 = ROOT / "outputs/sql2mql_v2/provisional_corpus_v1"
POOL3 = ROOT / "data/processed/spider_pool_v3"
EVAL_SOURCES = (
    "data/processed/spider_sql_v1/dev_full.jsonl", "data/processed/spider_sql_v1/dev_clean.jsonl",
    "data/processed/spider_sql_v1/test_full.jsonl", "data/processed/spider_sql_v1/test_clean.jsonl",
    "data/processed/spider1_nl_mql.jsonl", "data/processed/spider1_nl_mql_expanded.jsonl",
    "data/splits_expanded/train.jsonl", "data/splits_expanded/validation.jsonl",
    "data/rag/corpus_v1/corpus.jsonl", "data/processed/nl_mql_dataset.jsonl",
)
NOT_OPENED = ("data/splits_expanded/test.jsonl",)


def rd(p):
    return [json.loads(x) for x in Path(p).read_text(encoding="utf-8").splitlines() if x.strip()]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def norm_text(s):
    return re.sub(r"[^a-z0-9]+", " ", str(s).lower()).strip()


def h(s):
    return hashlib.sha256(norm_text(s).encode("utf-8")).hexdigest()


def eval_hashes():
    dbs, qh, sh, files = set(), set(), set(), {}
    for rel in EVAL_SOURCES:
        data = (ROOT / rel).read_bytes()
        files[rel] = {"sha256": hashlib.sha256(data).hexdigest()}
        n = 0
        for line in data.decode("utf-8").splitlines():
            if not line.strip():
                continue
            x = json.loads(line)
            n += 1
            for k in ("db_id", "db"):
                if x.get(k):
                    dbs.add(str(x[k]))
            if isinstance(x.get("question"), str):
                qh.add(h(x["question"]))
            for k in ("sql", "gold_sql", "reference_sql"):
                if isinstance(x.get(k), str):
                    sh.add(h(x[k]))
        files[rel]["records"] = n
    return {"dbs": dbs, "q": qh, "sql": sh, "files": files}


def leakage_reasons(row, ev):
    out = []
    if row["db_id"] in ev["dbs"]:
        out.append("leakage_db_overlap")
    if h(row["question"]) in ev["q"]:
        out.append("leakage_question_hash")
    if h(row["sql"]) in ev["sql"]:
        out.append("leakage_sql_hash")
    return out


def canon(o):
    return json.dumps(o, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def finalize():
    man_p = OUT / "manifest.json"
    if man_p.exists():
        raise SystemExit(f"refusing to overwrite {man_p}")
    cand = json.loads((OUT / "candidate_ids.json").read_text(encoding="utf-8"))
    recs = {r["id"]: r for r in rd(OUT / "records.jsonl")}
    if set(recs) != set(cand["ids"]) or len(recs) != cand["n"]:
        raise SystemExit(f"records ({len(recs)}) != candidates ({cand['n']}): the corpus run is incomplete")
    p3 = json.loads((POOL3 / "manifest.json").read_text(encoding="utf-8"))
    pool3_train = {r["id"]: r for r in rd(POOL3 / "train.jsonl")}
    pool3_tier = {e["id"]: e for e in rd(POOL3 / "exclusions_v3.jsonl")}
    ev = eval_hashes()
    pre = cand["pre_excluded"]
    status_rows, corpus, excl_acc = [], [], []
    for i in cand["ids"]:
        r = recs[i]
        reasons = []
        if i not in pool3_train:
            t = pool3_tier.get(i, {}).get("tier", "not_in_pool_v3_train")
            reasons.append(f"pool_v3:{t}")
        if r["outcome"] != "accepted":
            reasons.append(f"outcome:{r['outcome']}")
        if r["multi_table"]["multi_table_3plus"]:
            reasons.append("multi_table_3plus")
        if i in pre:
            reasons.append(pre[i])
        row = pool3_train.get(i) or {"db_id": r["db_id"], "question": r["question"], "sql": r["sql"]}
        reasons += leakage_reasons(row, ev)
        disp = "INCLUDED" if not reasons else "EXCLUDED"
        status_rows.append({"id": i, "db_id": r["db_id"], "outcome": r["outcome"], "outcome_detail": r["outcome_detail"],
                            "verifier_status": (r.get("verification") or {}).get("status"), "corpus_disposition": disp,
                            "corpus_exclusion_reasons": reasons})
        if disp == "INCLUDED":
            src = pool3_train[i]
            corpus.append({
                "corpus_entry_id": f"{CORPUS_VERSION}:{i}", "corpus_version": CORPUS_VERSION, "corpus_status": "PROVISIONAL",
                "review_status": "NOT_INDIVIDUALLY_REVIEWED (AI-panel G-AIR sample review only; never human-reviewed)",
                "id": i, "db_id": r["db_id"], "question": src["question"], "question_raw": src.get("question_raw"),
                "sql": src["sql"], "mql": r["mql"], "hardness": src["hardness"], "query_class": src.get("query_class"),
                "sql_pattern": r["sql_profile"]["pattern"], "converter_features": r.get("features"), "stratum": r["stratum"],
                "type_contract": r["type_contract"],
                "source": {"file": "data/processed/spider_pool_v3/train.jsonl", "origin": src.get("source")},
                "verification": {"outcome": "ACCEPTED", "verifier": r["verification"]["verifier"],
                                 "establishes": "SQL-to-MQL result equivalence under TC-1 on the real DB and fixtures; "
                                                "NOT that the SQL answers the question"},
                "converter_version": r["converter"], "v2_1_code_freeze_sha256": r["v2_1_code_freeze_sha256"]})
        elif r["outcome"] == "accepted":
            excl_acc.append({"id": i, "reasons": reasons})
    # id diff against corpus v1
    v1 = {e["id"]: e for e in rd(V1 / "provisional_rag_corpus.jsonl")}
    v2 = {e["id"]: e for e in corpus}
    diff = []
    for i in sorted(set(v1) | set(v2)):
        if i in v1 and i not in v2:
            st = next(s for s in status_rows if s["id"] == i)
            diff.append({"id": i, "change": "removed", "reasons": st["corpus_exclusion_reasons"]})
        elif i in v2 and i not in v1:
            diff.append({"id": i, "change": "added", "reasons": ["not in corpus v1 (v1 exclusion no longer applies)"]})
        elif canon(v1[i]["mql"]) != canon(v2[i]["mql"]):
            diff.append({"id": i, "change": "mql_changed", "reasons": ["sql2mql-v2.1 LIKE translation (Unicode-safe)"
                                                                        if "like" in (v2[i]["converter_features"] or [])
                                                                        else "UNEXPECTED: non-LIKE MQL changed"]})
    OUT.mkdir(parents=True, exist_ok=True)

    def wj(name, rows):
        with open(OUT / name, "w", encoding="utf-8") as f:
            for x in rows:
                f.write(json.dumps(x, ensure_ascii=False, default=str) + "\n")
    wj("provisional_rag_corpus.jsonl", corpus)
    wj("conversion_status.jsonl", status_rows)
    wj("excluded_accepted.jsonl", excl_acc)
    wj("id_diff_vs_corpus_v1.jsonl", diff)
    # checks
    tc = json.loads((ROOT / "data/interim/finetune_v2/type_contract_v1.json").read_text(encoding="utf-8"))
    not_in_manifest = [e["id"] for e in corpus if any(v == "NOT_IN_MANIFEST" for v in e["type_contract"]["fields"].values())]
    tc_sha_bad = [e["id"] for e in corpus if e["type_contract"]["manifest_sha256"] != sha(ROOT / "data/interim/finetune_v2/type_contract_v1.json")]
    nonconf = Counter(k for e in corpus for k, v in e["type_contract"]["fields"].items() if v == "nonconformant")
    dev_dbs = {json.loads(x)["db_id"] for x in (ROOT / "data/processed/spider_sql_v1/dev_full.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()}
    db_overlap = sorted({e["db_id"] for e in corpus} & dev_dbs)
    leaks = Counter(r for s in status_rows for r in s["corpus_exclusion_reasons"] if r.startswith("leakage"))
    unexpected = [d for d in diff if "UNEXPECTED" in d["reasons"][0]]
    checks = {"type_contract_fields_not_in_manifest": not_in_manifest, "type_contract_sha_mismatch": tc_sha_bad,
              "nonconformant_fields_referenced": dict(nonconf),
              "db_overlap_with_spider_dev": db_overlap, "leakage_hits_among_candidates": dict(leaks),
              "unexpected_non_like_mql_changes": unexpected, "duplicate_ids": len(corpus) - len(v2),
              "all_corpus_ids_in_pool_v3_train": all(e["id"] in pool3_train for e in corpus)}
    checks["pass"] = (not not_in_manifest and not tc_sha_bad and not db_overlap and not unexpected
                      and checks["duplicate_ids"] == 0 and checks["all_corpus_ids_in_pool_v3_train"])
    summ = {"candidates": len(status_rows), "outcomes": dict(Counter(s["outcome"] for s in status_rows)),
            "included": len(corpus), "accepted_but_excluded": len(excl_acc),
            "exclusion_reasons": dict(Counter(r.split(":")[0] if r.startswith(("pool_v3", "outcome")) else r
                                              for s in status_rows for r in s["corpus_exclusion_reasons"])),
            "exclusion_reasons_detail": dict(Counter(r for s in status_rows for r in s["corpus_exclusion_reasons"])),
            "diff_vs_v1": dict(Counter(d["change"] for d in diff)), "v1_size": len(v1)}
    man = {"corpus_version": CORPUS_VERSION, "status": "PROVISIONAL", "written_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "builder": "scripts/finetune_v2/build_corpus_v2.py", "builder_sha256": sha(__file__),
           "converter": "sql2mql-v2.1", "verifier": "sql2mql-v2-verifier-3.1",
           "inputs": {"v2_1_code_freeze_sha256": sha(ROOT / "data/interim/finetune_v2/v2_1_code_freeze.json"),
                      "pilot_v4_runner_freeze_sha256": sha(ROOT / "data/interim/finetune_v2/pilot_v4_runner_freeze.json"),
                      "type_contract_v1_sha256": sha(ROOT / "data/interim/finetune_v2/type_contract_v1.json"),
                      "spider_pool_v3_manifest_sha256": sha(POOL3 / "manifest.json"),
                      "spider_pool_v3_train_sha256": sha(POOL3 / "train.jsonl"),
                      "detector_prereg_sha256": sha(ROOT / "data/interim/source_label_audit_v1/prereg.json"),
                      "records_sha256": sha(OUT / "records.jsonl"), "candidate_ids_sha256": sha(OUT / "candidate_ids.json"),
                      "corpus_v1_rag_sha256": sha(V1 / "provisional_rag_corpus.jsonl"),
                      "leakage_sources": ev["files"], "leakage_sources_not_opened": list(NOT_OPENED)},
           "outputs": {n: sha(OUT / n) for n in ("provisional_rag_corpus.jsonl", "conversion_status.jsonl",
                                                 "excluded_accepted.jsonl", "id_diff_vs_corpus_v1.jsonl")},
           "corpus_rule": "pool-v3 train AND verifier 3.1 ACCEPTED AND NOT multi_table_3plus AND NOT pilot cross-review "
                          "pre-exclusion AND no leakage; no corrected labels",
           "review": "AI-reviewed (multi-agent panel) on the G-AIR sample only; never human-reviewed",
           "summary": summ, "checks": checks}
    man_p.write_text(json.dumps(man, indent=1), encoding="utf-8")
    (OUT / "summary.json").write_text(json.dumps(summ, indent=1), encoding="utf-8")
    print(json.dumps({"summary": summ, "checks": checks}, indent=1))


if __name__ == "__main__":
    {"finalize": finalize}[sys.argv[1]]()
