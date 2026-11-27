"""Build kaggle/nl2mql_mql0007_rag_v1/ (data, frozen retrieval plans, manifests) and its zip. PACKAGE PREPARATION ONLY:
no model is loaded, no inference is run, Atlas is never contacted, .env is never read.

Run with a Python that has `transformers` (tokenizer only, no torch needed):
  python scripts/nl2mql/build_mql0007_package.py --tokenizer-dir <dir with the pinned revision's tokenizer files>

Order (fail-fast): leakage gates FIRST (thresholds fixed in configs/experiment.json before running) -> schema-word
validation -> data files -> retrieval plans (token budgeting with the model's own tokenizer) -> manifest -> zip.
On any leakage hit the builder stops and writes nothing; the corpus is never filtered or modified.
"""
import argparse
import json
import re
import shutil
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PKG = ROOT / "kaggle/nl2mql_mql0007_rag_v1"
sys.path.insert(0, str(PKG / "src"))
import mql0007_core as C  # noqa: E402

CORPUS_DIR = ROOT / "outputs/sql2mql_v2/provisional_corpus_v1"
BUNDLE = ROOT / "kaggle/nl2mql_spider_dev_1034_bundle_v2"
MQL6 = ROOT / "outputs/nl2mql_spider_dev_1034_model_comparison/v1/predictions"
SPIDER_TABLES = ROOT / "data/raw/spider1/spider_data/tables.json"
SQLV1 = ROOT / "data/processed/spider_sql_v1"
ZIP_PATH = ROOT / "kaggle/nl2mql_mql0007_rag_v1_bundle.zip"

# files copied verbatim into the package (src -> package-relative dest)
VERBATIM = {
    CORPUS_DIR / "provisional_rag_corpus.jsonl": "data/corpus/provisional_rag_corpus.jsonl",
    BUNDLE / "data/prompts.jsonl": "data/eval/prompts.jsonl",
    BUNDLE / "context/atlas_collections.json": "data/context/atlas_collections.json",
}
GOLD_FIELDS = ("sql", "gold_rows", "ordered_sha256", "multiset_sha256")
ALLOW_EXT = {".py", ".json", ".jsonl", ".md", ".ipynb", ".txt"}


def jl(p):
    return C.read_jsonl(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer-dir", required=True)
    ap.add_argument("--no-zip", action="store_true")
    a = ap.parse_args()
    cfg = json.loads((PKG / "configs/experiment.json").read_text(encoding="utf-8"))
    log = {}

    # ------------------------------------------------------------ inputs
    corpus = jl(CORPUS_DIR / "provisional_rag_corpus.jsonl")
    cman = json.loads((CORPUS_DIR / "manifest.json").read_text(encoding="utf-8"))
    ev_full = jl(BUNDLE / "data/eval_records.jsonl")           # contains gold SQL: used for CHECKS only, never shipped
    bman = json.loads((BUNDLE / "bundle_manifest.json").read_text(encoding="utf-8"))
    prompts = jl(BUNDLE / "data/prompts.jsonl")
    atlas = json.loads((BUNDLE / "context/atlas_collections.json").read_text(encoding="utf-8"))
    tables_all = {t["db_id"]: t for t in json.loads(SPIDER_TABLES.read_text(encoding="utf-8"))}

    C.need(len(corpus) == C.EXPECTED_CORPUS and len({r["id"] for r in corpus}) == C.EXPECTED_CORPUS, "corpus != 3,589")
    C.need(C.sha256_file(CORPUS_DIR / "provisional_rag_corpus.jsonl") == cfg["corpus"]["source_sha256"] ==
           cman["outputs"]["provisional_rag_corpus.jsonl"]["sha256"], "corpus hash != manifest/config")
    ids = [r["id"] for r in ev_full]
    C.need(len(ids) == C.EXPECTED_EVAL and C.sha256_text("\n".join(ids)) == cfg["evaluation"]["ids_sha256"] ==
           bman["evaluation_set"]["ids_sha256"], "evaluation ids != 892 / ids hash")
    C.need([p["id"] for p in prompts] == ids, "prompt ids/order != evaluation ids")
    C.need(C.sha256_file(BUNDLE / "data/prompts.jsonl") == bman["files"]["data/prompts.jsonl"]["sha256"],
           "prompts.jsonl hash != bundle manifest")
    C.need(bman["prompt"]["prompts_sha256"] == cfg["prompt"]["zero_shot_prompts_sha256"], "prompts_sha256 != config")
    C.need({p["template_sha256"] for p in prompts} == {cfg["prompt"]["zero_shot_template_sha256"]}, "template hash")
    eval_dbs = sorted({r["db_id"] for r in ev_full})
    held_out_dbs = set(bman["evaluation_set"]["excluded_db_ids"])

    # ------------------------------------------------------------ leakage / eligibility gates (BEFORE anything is written)
    thr = cfg["leakage_gates"]["near_duplicate_token_jaccard"]
    lk = C.leakage_report(corpus, ev_full, thr)
    ack = cfg["leakage_gates"]["acknowledged_near_duplicates"]
    ack_pairs = {(e, ack["corpus_id"]) for e in ack["eval_ids"]}
    log["leakage_eval"] = {"exact_hits": len(lk["exact"]), "near_hits": len(lk["near"]), "threshold": thr,
                           "max_token_jaccard": lk["max_jaccard"], "db_overlap": lk["db_overlap"],
                           "near_hits_acknowledged": [list(x) for x in lk["near"]], "acknowledgement": ack["decision"]}
    C.need(not lk["exact"] and not lk["db_overlap"] and {(e, c) for e, c, _ in lk["near"]} == ack_pairs,
           f"LEAKAGE vs evaluation set beyond the acknowledged pairs: {log['leakage_eval']} "
           f"{lk['exact'][:3]} {lk['near'][:3]} -- STOP, ask the user (do not filter)")
    # protected sets: all Spider dev (1,034, includes the 65 held-out questions and the 3 held-out DBs) and Spider test
    dev_full, test_full = jl(SQLV1 / "dev_full.jsonl"), jl(SQLV1 / "test_full.jsonl")
    prot = {"spider_dev_1034": dev_full, "spider_test_2147": test_full}
    prot_log = {}
    corp_q = {C.norm_question(r["question"]) for r in corpus} | {C.norm_question(r.get("question_raw")) for r in corpus}
    corp_sql = {C.norm_sql(r["sql"]) for r in corpus}
    corp_ts = [C.token_set(r["question"]) | C.token_set(r.get("question_raw") or "") for r in corpus]
    for name, rows in prot.items():
        q_hits = [r["id"] for r in rows if C.norm_question(r["question"]) in corp_q]
        s_hits = [r["id"] for r in rows if C.norm_sql(r["sql"]) in corp_sql]
        near = [(r["id"], round(max(C.jaccard(C.token_set(r["question"]), cs) for cs in corp_ts), 3)) for r in rows]
        n_near = sum(1 for i, j in near if j >= thr and i not in ack["eval_ids"])   # acknowledged eval ids are Spider-dev rows too
        prot_log[name] = {"n": len(rows), "question_exact_hits": len(q_hits), "sql_exact_hits": len(s_hits),
                          "near_dup_hits": n_near, "db_overlap": sorted({r["db_id"] for r in rows} & {c["db_id"] for c in corpus})}
        # Protected-set gate = MEMBERSHIP (exact question, exact SQL, database overlap). Template-level near-duplicates
        # (Jaccard >= 0.8, one noun differs, other database) are REPORTED here, not gated: Spider test has 16 such
        # pairs with short templated questions; none is the same item. The 0.8 near-dup GATE applies to the evaluation set.
        C.need(not q_hits and not s_hits and not prot_log[name]["db_overlap"],
               f"LEAKAGE vs {name}: {prot_log[name]} {q_hits[:3]} {s_hits[:3]} -- STOP")
    C.need(not (held_out_dbs & {c["db_id"] for c in corpus}), "corpus uses a held-out MQL test database")
    prot_log["held_out_mql_test_dbs"] = sorted(held_out_dbs)
    gold_sql = {C.norm_sql(r["sql"]) for r in ev_full}
    prot_log["corpus_sql_equal_to_eval_gold_sql"] = sum(C.norm_sql(r["sql"]) in gold_sql for r in corpus)
    C.need(prot_log["corpus_sql_equal_to_eval_gold_sql"] == 0, "corpus SQL equals an evaluation gold SQL")
    log["leakage_protected"] = prot_log
    # eligibility: all train, verifier accepted, none of the pre-excluded/excluded-accepted ids, no flagged leakage_sql_hash
    excl = {r["id"]: r["reasons"] for r in jl(CORPUS_DIR / "excluded_accepted.jsonl")}
    recs = {r["id"]: r for r in jl(CORPUS_DIR / "records.jsonl")}
    flagged = {i for i, r in recs.items() if "leakage_sql_hash" in json.dumps(r, default=str)}
    elig = {"split_train": all(r["source"]["split"] == "train" for r in corpus),
            "verifier_accepted": all(r["verification"]["status"] == "accepted" for r in corpus),
            "corpus_converter": sorted({r["converter_version"] for r in corpus}),
            "in_excluded_accepted": sum(r["id"] in excl for r in corpus),
            "in_pre_excluded": sum(r["id"] in cman["eligibility_inputs"]["pre_excluded"] for r in corpus),
            "leakage_sql_hash_flagged_in_corpus": sum(r["id"] in flagged for r in corpus),
            "leakage_sql_hash_flagged_total": len(flagged),
            "source_label_audit": {s: sum(r["source_label_audit"]["status"] == s for r in corpus)
                                   for s in {r["source_label_audit"]["status"] for r in corpus}},
            "human_review_status": sorted({r["human_review_status"] for r in corpus}),
            "n_databases": len({r["db_id"] for r in corpus})}
    C.need(elig["split_train"] and elig["verifier_accepted"] and not elig["in_excluded_accepted"] and
           not elig["in_pre_excluded"] and not elig["leakage_sql_hash_flagged_in_corpus"], f"ineligible corpus rows: {elig}")
    log["eligibility"] = elig
    # schema words from tables.json == schema words from the Atlas manifest, on the 17 evaluation databases
    for db in eval_dbs:
        a1 = C.PT.schema_info(db, C.schema_manifest_from_tables(tables_all[db]))
        a2 = C.PT.schema_info(db, atlas)
        C.need(a1["schema_words"] == a2["schema_words"] and a1["tables"] == a2["tables"],
               f"schema words from tables.json != Atlas manifest for {db} -- STOP")
    log["schema_words_tables_json_equals_atlas_manifest"] = (f"{len(eval_dbs)}/{len(eval_dbs)} evaluation databases "
                                                             "(tables.json's SQLite-internal sqlite_sequence table ignored)")
    C.retriever_constants_ok(cfg)

    # ------------------------------------------------------------ write data files
    for sub in ("data/corpus", "data/eval", "data/context", "data/plans", "data/zero_shot_mql0006"):
        (PKG / sub).mkdir(parents=True, exist_ok=True)
    for src, dst in VERBATIM.items():
        shutil.copyfile(src, PKG / dst)
    ev_in = [{"id": r["id"], "db_id": r["db_id"], "question": r["question"]} for r in ev_full]   # inference inputs only
    C.write_jsonl(PKG / "data/eval/eval_inputs.jsonl", ev_in)
    needed_dbs = sorted({r["db_id"] for r in corpus} | set(eval_dbs))
    C.write_json(PKG / "data/context/tables_subset.json", [tables_all[d] for d in needed_dbs])
    C.write_json(PKG / "data/corpus/excluded_accepted_ids.json", {"source": "excluded_accepted.jsonl of provisional_corpus_v1",
                 "ids": sorted(excl), "pre_excluded": sorted(cman["eligibility_inputs"]["pre_excluded"])})
    # zero-shot reuse (MQL-0006): copy the verified predictions byte for byte
    for name in ("predictions.jsonl", "run_config.json"):
        shutil.copyfile(MQL6 / "models/qwen2.5-coder-3b-instruct" / name, PKG / "data/zero_shot_mql0006" / name)
    shutil.copyfile(MQL6 / "SHA256SUMS.txt", PKG / "data/zero_shot_mql0006/MQL-0006_SHA256SUMS.txt")

    # ------------------------------------------------------------ corpus features (frozen; Kaggle re-derives and compares)
    tables_by_db = {t["db_id"]: t for t in json.loads((PKG / "data/context/tables_subset.json").read_text(encoding="utf-8"))}
    docs = C.build_docs(corpus, tables_by_db)
    C.write_jsonl(PKG / "data/corpus/corpus_features.jsonl",
                  [{k: d[k] for k in ("doc_id", "db_id", "question", "masked_question", "patterns", "content")} for d in docs])

    # ------------------------------------------------------------ retrieval plans with the model's own tokenizer
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tokenizer_dir)
    count = lambda s: len(tok.encode(s, add_special_tokens=False))   # noqa: E731
    einfo = C.eval_schema_infos(atlas, eval_dbs)
    plan_hashes = {}
    by_id = {d["doc_id"]: d for d in docs}
    for arm in C.ARMS[1:]:
        plan = C.build_plan(cfg, arm, ev_in, prompts, docs, einfo, count)          # primary: with the evaluation-specific exclusions
        p = PKG / f"data/plans/retrieval_plan_{arm}.jsonl"
        C.write_jsonl(p, plan)
        plan_hashes[arm] = C.sha256_file(p)
        assert [x["id"] for x in plan] == ids
        for x in plan:   # selected examples must be corpus members, never from the target database
            C.need(all(i in by_id and by_id[i]["db_id"] != x["db_id"] for i in x["selected_ids"]), f"bad selection {x['id']}")
        viol = C.selection_neardup_violations(plan, ev_in, by_id, thr)
        C.need(not viol, f"{arm}: a SELECTED example is a near-duplicate of its query: {viol[:3]}")
    plans = {arm: C.read_jsonl(PKG / f"data/plans/retrieval_plan_{arm}.jsonl") for arm in C.ARMS[1:]}
    # exclusion proof: the pre-exclusion plans differ from the primary plans on EXACTLY the expected items (and B is unchanged)
    unex = {arm: C.build_plan(cfg, arm, ev_in, prompts, docs, einfo, count, exclusions=[]) for arm in C.ARMS[1:]}
    diff = {arm: C.plan_diff_ids(plans[arm], unex[arm]) for arm in C.ARMS[1:]}
    C.need(diff["B_random_k3"] == [] and diff["C_hybrid_k3"] == ["spider_dev_0383"],
           f"exclusion changed unexpected plans: {diff} (expected B none, C only spider_dev_0383)")
    sup_rows = [e for e in ev_in if e["id"] in set(diff["C_hybrid_k3"])]
    supp = C.build_plan(cfg, "C_hybrid_k3", sup_rows, prompts, docs, einfo, count, exclusions=[], arm_label=C.SUPP_ARM)
    C.write_jsonl(PKG / f"data/plans/retrieval_plan_{C.SUPP_ARM}.jsonl", supp)
    C.need(all("spider_train_00650" in x["selected_ids"] for x in supp), "supplementary plan does not use the near-duplicate")
    log["exclusion_proof"] = {"plan_diff_ids_primary_vs_unexcluded": diff, "supplementary_arm_ids": [x["id"] for x in supp],
                              "excluded_by_rule_logged": {arm: [(x["id"], x["excluded_by_rule"]) for x in pl if x["excluded_by_rule"]]
                                                          for arm, pl in plans.items()},
                              "selected_examples_with_jaccard_ge_threshold": 0}
    log["plans"] = {arm: {"empty_context": sum(x["empty_context"] for x in pl),
                          "items_with_budget_skips": sum(bool(x["dropped_budget"]) for x in pl),
                          "mean_selected": round(sum(len(x["selected_ids"]) for x in pl) / len(pl), 3),
                          "max_context_tokens": max(x["context_tokens"] for x in pl),
                          "max_prompt_tokens_est": max(x["prompt_tokens_est"] or 0 for x in pl)} for arm, pl in plans.items()}
    log["tokenizer"] = {"dir_files": sorted(p.name for p in Path(a.tokenizer_dir).iterdir() if p.is_file()),
                        "tokenizer_json_sha256": C.sha256_file(Path(a.tokenizer_dir) / "tokenizer.json"),
                        "transformers": __import__("transformers").__version__,
                        "tokenizers": __import__("tokenizers").__version__}
    # gold-leak scan of everything shipped under data/ (exact SQL strings of the 892 gold queries, outside the corpus)
    squash = lambda s: re.sub(r"\s+", " ", s).strip().lower()   # noqa: E731
    golds = [squash(r["sql"]) for r in ev_full if len(r["sql"]) > 12]
    scan = {}
    for f in sorted((PKG / "data").rglob("*")):
        if f.is_file() and f.name not in ("provisional_rag_corpus.jsonl", "corpus_features.jsonl", "retrieval_plan_B_random_k3.jsonl",
                                          "retrieval_plan_C_hybrid_k3.jsonl", f"retrieval_plan_{C.SUPP_ARM}.jsonl"):
            txt = squash(f.read_text(encoding="utf-8"))
            scan[f.relative_to(PKG).as_posix()] = sum(g in txt for g in golds)
    C.need(not any(scan.values()), f"gold SQL text found in shipped data: {scan}")
    # prompts embed no gold: reuse the MQL-0006 rule (gold SQL / metadata markers absent from prompts)
    for p in prompts:
        t = squash(p["system"] + " " + " ".join(m["content"] for m in p["messages"]))
        g = squash(next(r["sql"] for r in ev_full if r["id"] == p["id"]))
        C.need(g not in t, f"gold SQL in prompt {p['id']}")
        C.need(all(m not in t for m in ("ordered_sha256", "multiset_sha256", '"sql":')), "gold marker in prompt")
    log["gold_leak_scan"] = {"files_scanned": len(scan), "gold_sql_strings_found": 0, "prompts_checked": len(prompts)}
    log["shipped_eval_fields"] = ["id", "db_id", "question"]
    log["not_shipped"] = ["eval_records.jsonl (gold SQL)", "reference.jsonl.gz (gold rows)", "SQLite databases",
                          "reference_expected.json", ".env / credentials", "held-out 65-example partition file"]
    C.write_json(PKG / "data/leakage_report.json", {"generated_by": "scripts/nl2mql/build_mql0007_package.py",
                                                    "gates_fixed_in": "configs/experiment.json (before execution)", **log})
    print(json.dumps(log, indent=1)[:4000])

    # ------------------------------------------------------------ manifest + zip (allowlist: the package dir only)
    manifest_and_zip(a.no_zip)


def package_files():
    out = []
    for p in sorted(PKG.rglob("*")):
        rel = p.relative_to(PKG)
        if (not p.is_file() or "__pycache__" in rel.parts or ".pytest_cache" in rel.parts or rel.parts[0] in ("outputs", "work")
                or rel.as_posix() in ("package_manifest.json", "SHA256SUMS.txt") or p.suffix not in ALLOW_EXT):
            continue
        out.append(rel.as_posix())
    return out


def manifest_and_zip(no_zip=False):
    files = package_files()
    # builder copy for provenance
    man = {"package": "nl2mql_mql0007_rag_v1", "experiment": "MQL-0007", "status": "PREPARED - NOT RUN",
           "built_at": C.now(), "files": {n: {"bytes": (PKG / n).stat().st_size, "sha256": C.sha256_file(PKG / n)} for n in files}}
    C.write_json(PKG / "package_manifest.json", man)
    (PKG / "SHA256SUMS.txt").write_text("".join(f"{m['sha256']}  {n}\n" for n, m in man["files"].items()), encoding="utf-8")
    if no_zip:
        return
    if ZIP_PATH.exists():
        ZIP_PATH.unlink()
    with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as z:
        for n in files + ["package_manifest.json", "SHA256SUMS.txt"]:
            z.write(PKG / n, f"nl2mql_mql0007_rag_v1/{n}")
    print("zip:", ZIP_PATH, ZIP_PATH.stat().st_size)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--finalize":
        manifest_and_zip()
    else:
        main()
