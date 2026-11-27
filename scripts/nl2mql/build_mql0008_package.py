"""Build kaggle/nl2mql_mql0008_approved_corpus_v2_1/ (data, frozen retrieval plan, audits, manifest) and its zip.

MQL-0008 = zero-shot vs hybrid RAG over the APPROVED corpus v2.1. PACKAGE PREPARATION ONLY: no model is loaded, no
inference is run, Atlas is never contacted, no credential is read. Derived from build_mql0007_package.py.

  python scripts/nl2mql/make_mql0008_notebook.py
  python scripts/nl2mql/build_mql0008_package.py --tokenizer-dir <dir with the pinned revision's tokenizer files>

Order (fail-fast, nothing is written before the gates pass): corpus identity -> frozen-configuration equality with
MQL-0007 -> leakage audit against the approved corpus -> eligibility -> data files -> retrieval plan (token budgeting
with the model's own tokenizer) -> audits -> manifest -> zip. The corpus is never filtered or modified.

Protected data: this builder does NOT open Spider dev/test or the 65-item held-out partition. See
`protected_sets` in the leakage report for how membership is established without reading them.
"""
import argparse
import json
import re
import shutil
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PKG = ROOT / "kaggle/nl2mql_mql0008_approved_corpus_v2_1"
OUT = ROOT / "outputs/nl2mql_mql0008_approved_corpus_v2_1"
sys.dont_write_bytecode = True
sys.path.insert(0, str(PKG / "src"))
import mql0008_core as C  # noqa: E402

CORPUS_REL = "outputs/sql2mql_v2_1/approved_corpus_v2_1/corpus.jsonl"       # the ONLY retrieval source
CORPUS = ROOT / CORPUS_REL
CORPUS_MANIFEST = CORPUS.parent / "manifest.json"
V1_CORPUS = ROOT / "outputs/sql2mql_v2/provisional_corpus_v1/provisional_rag_corpus.jsonl"   # checks only, never shipped
V1_SHA = "61553d03fe4244d7d14a82c434482ef73f5e74b59f95e0cf983dd07806c7fc3d"
M7_PKG = ROOT / "kaggle/nl2mql_mql0007_rag_v1"                                  # frozen reference, read only
BUNDLE = ROOT / "kaggle/nl2mql_spider_dev_1034_bundle_v2"
SPIDER_TABLES = ROOT / "data/raw/spider1/spider_data/tables.json"
ZIP_PATH = ROOT / "kaggle/nl2mql_mql0008_approved_corpus_v2_1_bundle.zip"
PLAN_REL = f"data/plans/retrieval_plan_{C.RAG_ARM}.jsonl"
ALLOW_EXT = {".py", ".json", ".jsonl", ".md", ".ipynb", ".txt"}
PROVENANCE = ("build_mql0008_package.py", "make_mql0008_notebook.py")


def jl(p):
    return C.read_jsonl(p)


def dig(d, dotted):
    for k in dotted.split("."):
        d = d[k]
    return d


def frozen_config_diff(cfg, m7):
    """Every setting that must equal the frozen MQL-0007 configuration -> {key: (MQL-0007, now)} for differences."""
    pairs = {k: (dig(m7, k), dig(cfg, k)) for k in ("model", "prompt", "retrieval", "context", "retrieval_exclusions",
                                                     "leakage_gates.near_duplicate_token_jaccard")}
    pairs["rag arm retrieval"] = (m7["arms"]["C_hybrid_k3"]["retrieval"], cfg["arms"][C.RAG_ARM]["retrieval"])
    pairs["rag arm k"] = (m7["arms"]["C_hybrid_k3"]["k"], cfg["arms"][C.RAG_ARM]["k"])
    ack7, ack8 = m7["leakage_gates"]["acknowledged_near_duplicates"], cfg["leakage_gates"]["acknowledged_near_duplicates"]
    pairs["acknowledged pairs"] = ((ack7["corpus_id"], ack7["eval_ids"]), (ack8["corpus_id"], ack8["eval_ids"]))
    diff = {}
    for k, (a, b) in pairs.items():
        if k == "model":                      # the dtype note mentions MQL-0007 as well; every functional key must match
            a, b = ({x: v for x, v in d.items() if x != "dtype_note"} for d in (a, b))
        if a != b:
            diff[k] = (a, b)
    return diff


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer-dir", required=True)
    ap.add_argument("--no-zip", action="store_true")
    a = ap.parse_args()
    cfg = json.loads((PKG / "configs/experiment.json").read_text(encoding="utf-8"))
    m7 = json.loads((M7_PKG / "configs/experiment.json").read_text(encoding="utf-8"))
    log = {}

    # ------------------------------------------------------------ 1. corpus identity (abort on any difference)
    sha = C.sha256_file(CORPUS)
    corpus = jl(CORPUS)
    C.need(sha == C.EXPECTED_CORPUS_SHA256 == cfg["corpus"]["source_sha256"],
           f"ABORT: approved corpus sha256 {sha} != expected {C.EXPECTED_CORPUS_SHA256}")
    C.need(len(corpus) == C.EXPECTED_CORPUS == cfg["corpus"]["n_entries"] and len({r["id"] for r in corpus}) == len(corpus),
           f"ABORT: approved corpus has {len(corpus)} records, expected {C.EXPECTED_CORPUS} unique")
    C.need(cfg["corpus"]["source_path"] == CORPUS_REL, "config corpus path")
    cman = json.loads(CORPUS_MANIFEST.read_text(encoding="utf-8"))
    C.need(cman["files"]["corpus.jsonl"] == {"sha256": sha, "records": len(corpus)} and
           cman["status"].startswith("CONDITIONAL GO - APPROVED FOR TRAINING"), "approval manifest does not cover this corpus file")
    log["corpus"] = {"path": CORPUS_REL, "records": len(corpus), "sha256": sha, "approval_status": cman["status"],
                     "approval_manifest_sha256": C.sha256_file(CORPUS_MANIFEST), "review_label": cman["review_label"],
                     "databases": len({r["db_id"] for r in corpus})}

    # ------------------------------------------------------------ 2. frozen configuration == MQL-0007
    diff = frozen_config_diff(cfg, m7)
    C.need(not diff, f"configuration differs from the frozen MQL-0007 configuration (MQL-0007, now): {diff}")
    C.retriever_constants_ok(cfg)
    for f in ("patterns.py", "retrieval.py", "rag_block.py"):
        C.need(C.sha256_file(PKG / "src" / f) == C.sha256_file(M7_PKG / "src" / f), f"src/{f} differs from the MQL-0007 copy")
    log["frozen_configuration"] = {
        "reference": "kaggle/nl2mql_mql0007_rag_v1/configs/experiment.json",
        "reference_sha256": C.sha256_file(M7_PKG / "configs/experiment.json"),
        "equal": ["model (id, revision, dtype, decoding, max_new_tokens, batch size, seed, OOM policy)", "prompt",
                  "retrieval (BM25 k1 1.5, b 0.75, hybrid weight 1.0, hybrid_masked, ties, dedup, no guides)",
                  "context (k 3, 1,500-token cap, margin 64, n_ctx 8192)", "near-duplicate threshold 0.8",
                  "retrieval exclusions", "RAG arm retrieval method and k"],
        "retrieval_code_identical_to_mql0007": ["src/patterns.py", "src/retrieval.py", "src/rag_block.py"]}

    # ------------------------------------------------------------ evaluation inputs (the 892 ids of MQL-0006 / MQL-0007)
    ev_full = jl(BUNDLE / "data/eval_records.jsonl")      # holds gold SQL: used for CHECKS only, never shipped
    bman = json.loads((BUNDLE / "bundle_manifest.json").read_text(encoding="utf-8"))
    prompts = jl(BUNDLE / "data/prompts.jsonl")
    atlas = json.loads((BUNDLE / "context/atlas_collections.json").read_text(encoding="utf-8"))
    tables_all = {t["db_id"]: t for t in json.loads(SPIDER_TABLES.read_text(encoding="utf-8"))}
    ids = [r["id"] for r in ev_full]
    C.need(len(ids) == C.EXPECTED_EVAL == len(set(ids)) and C.sha256_text("\n".join(ids)) == cfg["evaluation"]["ids_sha256"] ==
           bman["evaluation_set"]["ids_sha256"], "evaluation ids != 892 / ids hash")
    C.need(ids == [r["id"] for r in jl(M7_PKG / "data/eval/eval_inputs.jsonl")], "evaluation ids/order != MQL-0007's")
    C.need([p["id"] for p in prompts] == ids, "prompt ids/order != evaluation ids")
    C.need(C.sha256_file(BUNDLE / "data/prompts.jsonl") == bman["files"]["data/prompts.jsonl"]["sha256"] ==
           C.sha256_file(M7_PKG / "data/eval/prompts.jsonl"), "prompts.jsonl differs from the MQL-0006 bundle / MQL-0007 package")
    C.need(bman["prompt"]["prompts_sha256"] == cfg["prompt"]["zero_shot_prompts_sha256"], "prompts_sha256 != config")
    C.need({p["template_sha256"] for p in prompts} == {cfg["prompt"]["zero_shot_template_sha256"]}, "template hash")
    eval_dbs = sorted({r["db_id"] for r in ev_full})
    held_out_dbs = set(bman["evaluation_set"]["excluded_db_ids"])

    # ------------------------------------------------------------ 3. leakage audit against the approved corpus
    thr = cfg["leakage_gates"]["near_duplicate_token_jaccard"]
    lk = C.leakage_report(corpus, ev_full, thr)
    raw_q = {r["question"] for r in corpus} | {r["question_raw"] for r in corpus}
    raw_exact = [r["id"] for r in ev_full if r["question"] in raw_q]
    ack = cfg["leakage_gates"]["acknowledged_near_duplicates"]
    ack_pairs = {(e, ack["corpus_id"]) for e in ack["eval_ids"]}
    by_cid = {r["id"]: r for r in corpus}
    by_eid = {r["id"]: r for r in ev_full}
    log["leakage_eval"] = {
        "checks": {"1_exact_question_string": len(raw_exact), "2_normalised_question": len(lk["exact"]),
                   "3_near_duplicate_token_jaccard_ge_threshold": len(lk["near"]),
                   "4_database_overlap": lk["db_overlap"]},
        "threshold": thr, "max_token_jaccard": lk["max_jaccard"],
        "near_hits": [{"eval_id": e, "corpus_id": c, "token_jaccard": j, "eval_question": by_eid[e]["question"],
                       "corpus_question": by_cid[c]["question"], "eval_db": by_eid[e]["db_id"], "corpus_db": by_cid[c]["db_id"]}
                      for e, c, j in lk["near"]],
        "rule_applied": ack["decision"],
        "exclusions": cfg["retrieval_exclusions"],
        "evaluation_items_removed": 0}
    C.need(not raw_exact and not lk["exact"] and not lk["db_overlap"] and {(e, c) for e, c, _ in lk["near"]} == ack_pairs,
           f"LEAKAGE vs evaluation set beyond the documented pairs: {log['leakage_eval']['checks']} "
           f"{lk['exact'][:3]} {lk['near'][:5]} -- STOP (the corpus is not filtered)")
    gold_sql = {C.norm_sql(r["sql"]) for r in ev_full}
    n_sql = sum(C.norm_sql(r["sql"]) in gold_sql for r in corpus)
    C.need(n_sql == 0, "a corpus SQL equals an evaluation gold SQL")
    log["leakage_eval"]["checks"]["5_corpus_sql_equal_to_an_evaluation_gold_sql"] = n_sql
    # protected sets, WITHOUT reading them: the approved corpus is an id-subset of corpus v1 with identical question,
    # question_raw, SQL and database per id; MQL-0007's membership audit of corpus v1 found zero hits, so zero carries over.
    C.need(C.sha256_file(V1_CORPUS) == V1_SHA, "corpus v1 reference hash")
    v1 = {r["id"]: r for r in jl(V1_CORPUS)}
    same = sum(r["id"] in v1 and all(r[k] == v1[r["id"]][k] for k in ("question", "question_raw", "sql", "db_id")) for r in corpus)
    m7_leak = json.loads((M7_PKG / "data/leakage_report.json").read_text(encoding="utf-8"))["leakage_protected"]
    v1_zero = all(m7_leak[n][k] in (0, []) for n in ("spider_dev_1034", "spider_test_2147")
                  for k in ("question_exact_hits", "sql_exact_hits", "db_overlap"))
    C.need(same == len(corpus) and v1_zero, f"protected-set carry-over does not hold: {same}/{len(corpus)} identical, v1 zero {v1_zero}")
    C.need(not (held_out_dbs & {c["db_id"] for c in corpus}), "corpus uses a held-out MQL test database")
    log["protected_sets"] = {
        "files_opened": "none (Spider dev/test and the 65-item held-out partition were not read)",
        "method": "subset carry-over from the MQL-0007 audit of corpus v1",
        "approved_records_identical_to_corpus_v1_record": f"{same}/{len(corpus)} (id, question, question_raw, SQL, database)",
        "corpus_v1_membership_audit_mql0007": {n: {k: m7_leak[n][k] for k in ("n", "question_exact_hits", "sql_exact_hits",
                                                                                 "near_dup_hits", "db_overlap")}
                                                for n in ("spider_dev_1034", "spider_test_2147")},
        "conclusion": "exact-question, exact-SQL and database membership are zero for Spider dev (incl. the 65 held-out items) "
                      "and Spider test; template near-duplicates with Spider test are at most the 16 reported for corpus v1 "
                      "(reported, not gated, as in MQL-0007)",
        "held_out_mql_test_dbs_absent_from_corpus": sorted(held_out_dbs)}

    # ------------------------------------------------------------ eligibility
    excl_ids = {r["id"] for r in jl(CORPUS.parent / "exclusions.jsonl")}
    elig = {"ids_all_spider_train": all(r["id"].startswith("spider_train_") for r in corpus),
            "source_files": sorted({r["source"]["file"] for r in corpus}),
            "verification_outcomes": sorted({r["verification"]["outcome"] for r in corpus}),
            "verifiers": sorted({r["verification"]["verifier"] for r in corpus}),
            "converters": sorted({r["converter_version"] for r in corpus}),
            "corpus_versions": sorted({r["corpus_version"] for r in corpus}),
            "in_approval_exclusions": sum(r["id"] in excl_ids for r in corpus),
            "approval_exclusions": sorted(excl_ids), "n_databases": len({r["db_id"] for r in corpus})}
    C.need(elig["ids_all_spider_train"] and elig["verification_outcomes"] == ["ACCEPTED"] and
           elig["verifiers"] == ["sql2mql-v2-verifier-3.1"] and elig["converters"] == ["sql2mql-v2.1"] and
           not elig["in_approval_exclusions"] and all("train" in s for s in elig["source_files"]), f"ineligible corpus rows: {elig}")
    log["eligibility"] = elig
    # the 'text_number' flag: empty for corpus v1 (so computing it as empty for v2.1 changes nothing)
    tn_v1 = C.text_number_dbs(list(v1.values()))
    C.need(not tn_v1 and not C.text_number_dbs(corpus), f"text_number flag is not empty: {sorted(tn_v1)[:5]}")
    log["text_number_flag"] = {"corpus_v1_databases_flagged": 0, "approved_corpus_databases_flagged": 0}
    for db in eval_dbs:
        a1 = C.PT.schema_info(db, C.schema_manifest_from_tables(tables_all[db]))
        a2 = C.PT.schema_info(db, atlas)
        C.need(a1["schema_words"] == a2["schema_words"] and a1["tables"] == a2["tables"],
               f"schema words from tables.json != Atlas manifest for {db} -- STOP")
    log["schema_words_tables_json_equals_atlas_manifest"] = f"{len(eval_dbs)}/{len(eval_dbs)} evaluation databases"

    # ------------------------------------------------------------ write data files (only after every gate above)
    for sub in ("data/corpus", "data/eval", "data/context", "data/plans", "provenance"):
        (PKG / sub).mkdir(parents=True, exist_ok=True)
    shutil.copyfile(CORPUS, PKG / C.CORPUS_FILE)
    C.need(C.sha256_file(PKG / C.CORPUS_FILE) == sha, "packaged corpus != approved corpus")
    shutil.copyfile(BUNDLE / "data/prompts.jsonl", PKG / "data/eval/prompts.jsonl")
    shutil.copyfile(BUNDLE / "context/atlas_collections.json", PKG / "data/context/atlas_collections.json")
    ev_in = [{"id": r["id"], "db_id": r["db_id"], "question": r["question"]} for r in ev_full]   # inference inputs only
    C.write_jsonl(PKG / "data/eval/eval_inputs.jsonl", ev_in)
    C.need(C.sha256_file(PKG / "data/eval/eval_inputs.jsonl") == C.sha256_file(M7_PKG / "data/eval/eval_inputs.jsonl"),
           "eval_inputs.jsonl is not byte-identical to MQL-0007's")
    needed_dbs = sorted({r["db_id"] for r in corpus} | set(eval_dbs))
    C.write_json(PKG / "data/context/tables_subset.json", [tables_all[d] for d in needed_dbs])
    for f in PROVENANCE:
        shutil.copyfile(ROOT / "scripts/nl2mql" / f, PKG / "provenance" / f)

    # ------------------------------------------------------------ corpus features + retrieval plan (pinned tokenizer)
    tables_by_db = {t["db_id"]: t for t in json.loads((PKG / "data/context/tables_subset.json").read_text(encoding="utf-8"))}
    docs = C.build_docs(corpus, tables_by_db)
    by_id = {d["doc_id"]: d for d in docs}
    C.write_jsonl(PKG / "data/corpus/corpus_features.jsonl",
                  [{k: d[k] for k in ("doc_id", "db_id", "question", "masked_question", "patterns", "content")} for d in docs])
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tokenizer_dir)
    count = lambda s: len(tok.encode(s, add_special_tokens=False))   # noqa: E731
    einfo = C.eval_schema_infos(atlas, eval_dbs)
    plan = C.build_plan(cfg, C.RAG_ARM, ev_in, prompts, docs, einfo, count)
    C.write_jsonl(PKG / PLAN_REL, plan)
    plan = jl(PKG / PLAN_REL)
    C.need([x["id"] for x in plan] == ids, "plan ids/order")
    ctx = cfg["context"]
    for x in plan:
        C.need(all(i in by_id and by_id[i]["db_id"] != x["db_id"] for i in x["selected_ids"]), f"bad selection {x['id']}")
        C.need(x["context_tokens"] <= ctx["max_context_tokens"] and len(x["selected_ids"]) <= ctx["k"] and
               x["prompt_tokens_est"] + ctx["max_new_tokens"] + ctx["margin_tokens"] <= ctx["n_ctx"], f"budget violated {x['id']}")
    viol = C.selection_neardup_violations(plan, ev_in, by_id, thr)
    C.need(not viol, f"a SELECTED example is a near-duplicate of its query: {viol[:3]}")
    for x in cfg["retrieval_exclusions"]:
        C.need(x["corpus_id"] not in next(p for p in plan if p["id"] == x["eval_id"])["selected_ids"], f"exclusion {x} violated")
    unex = C.build_plan(cfg, C.RAG_ARM, ev_in, prompts, docs, einfo, count, exclusions=[])
    diff_ids = C.plan_diff_ids(plan, unex)
    C.need(set(diff_ids) <= set(ack["eval_ids"]), f"the exclusion changed unexpected items: {diff_ids}")
    again = C.build_plan(cfg, C.RAG_ARM, ev_in, prompts, docs, einfo, count)
    C.need(again == plan, "retrieval plan is not deterministic")
    m7_plan = {p["id"]: p for p in jl(M7_PKG / "data/plans/retrieval_plan_C_hybrid_k3.jsonl")}
    approved_ids = set(by_id)
    tok_sha = C.sha256_file(Path(a.tokenizer_dir) / "tokenizer.json")
    retrieval_audit = {
        "experiment": C.EXPERIMENT, "arm": C.RAG_ARM, "generated_by": "scripts/nl2mql/build_mql0008_package.py",
        "corpus_sha256": sha, "plan_file": PLAN_REL, "plan_sha256": C.sha256_file(PKG / PLAN_REL),
        "retrieval_config_fingerprint": C.retrieval_fingerprint(cfg),
        "items": len(plan), "selected_per_item": {str(k): sum(len(x["selected_ids"]) == k for x in plan) for k in range(ctx["k"] + 1)},
        "empty_context_items": sum(x["empty_context"] for x in plan),
        "items_with_over_budget_skips": sum(bool(x["dropped_budget"]) for x in plan),
        "max_context_tokens": max(x["context_tokens"] for x in plan), "context_token_cap": ctx["max_context_tokens"],
        "max_prompt_tokens_est": max(x["prompt_tokens_est"] for x in plan),
        "selected_from_target_database": 0, "selected_ids_outside_the_approved_corpus": 0,
        "selected_examples_with_jaccard_ge_threshold": 0, "deterministic_rebuild_identical": True,
        "distinct_corpus_examples_used": len({i for x in plan for i in x["selected_ids"]}),
        "exclusions_applied": [{"eval_id": x["id"], "excluded": x["excluded_by_rule"], "selected_ids": x["selected_ids"]}
                               for x in plan if x["excluded_by_rule"]],
        "items_whose_selection_the_exclusion_changes": diff_ids,
        "comparison_with_mql0007_arm_C_plan_corpus_v1": {
            "note": "descriptive only; shows how much the corpus change moves retrieval. Not an accuracy result.",
            "items_with_identical_selected_ids": sum(m7_plan[x["id"]]["selected_ids"] == x["selected_ids"] for x in plan),
            "items_with_different_selected_ids": sum(m7_plan[x["id"]]["selected_ids"] != x["selected_ids"] for x in plan),
            "mql0007_items_that_used_an_example_not_in_the_approved_corpus":
                sum(any(i not in approved_ids for i in m7_plan[x["id"]]["selected_ids"]) for x in plan)},
        "tokenizer": {"tokenizer_json_sha256": tok_sha, "transformers": __import__("transformers").__version__,
                      "tokenizers": __import__("tokenizers").__version__,
                      "same_tokenizer_json_as_mql0007_build": tok_sha == json.loads(
                          (M7_PKG / "data/leakage_report.json").read_text(encoding="utf-8"))["tokenizer"]["tokenizer_json_sha256"]}}
    C.need(retrieval_audit["tokenizer"]["same_tokenizer_json_as_mql0007_build"], "tokenizer.json differs from the MQL-0007 build's")
    C.write_json(PKG / "data/retrieval_audit.json", retrieval_audit)

    # ------------------------------------------------------------ gold-leak scan of everything shipped under data/
    squash = lambda s: re.sub(r"\s+", " ", s).strip().lower()   # noqa: E731
    golds = [squash(r["sql"]) for r in ev_full if len(r["sql"]) > 12]
    scan = {}
    for f in sorted((PKG / "data").rglob("*")):
        if f.is_file() and f.name not in (Path(C.CORPUS_FILE).name, "corpus_features.jsonl", Path(PLAN_REL).name):
            txt = squash(f.read_text(encoding="utf-8"))
            scan[f.relative_to(PKG).as_posix()] = sum(g in txt for g in golds)
    C.need(not any(scan.values()), f"gold SQL text found in shipped data: {scan}")
    arm_prompts = C.prompts_for_arm(plan, prompts, by_id)
    for p in list(prompts) + arm_prompts:
        t = squash(p["system"] + " " + " ".join(m["content"] for m in p["messages"]))
        C.need(squash(by_eid[p["id"]]["sql"]) not in t, f"gold SQL in prompt {p['id']}")
        C.need(all(m not in t for m in ("ordered_sha256", "multiset_sha256", '"sql":')), "gold marker in prompt")
    log["gold_leak_scan"] = {"files_scanned": len(scan), "gold_sql_strings_found": 0,
                             "prompts_checked": {"zero_shot": len(prompts), "rag": len(arm_prompts)}}
    log["shipped_eval_fields"] = ["id", "db_id", "question"]
    log["not_shipped"] = ["eval_records.jsonl (gold SQL)", "reference.jsonl.gz (gold rows)", "SQLite databases",
                          "credentials of any kind", "held-out 65-example partition", "corpus v1", "provisional corpus v2",
                          "spider-pool-v3", "MQL-0006 / MQL-0007 predictions and scores"]
    C.write_json(PKG / "data/leakage_report.json", {
        "experiment": C.EXPERIMENT, "generated_by": "scripts/nl2mql/build_mql0008_package.py", "generated_at": C.now(),
        "audited_corpus": log["corpus"], "verdict": "PASS: no leakage beyond the two documented template near-duplicates, "
        "which are excluded from retrieval for their two evaluation items", **log})
    (OUT / "preflight").mkdir(parents=True, exist_ok=True)
    for f in ("data/leakage_report.json", "data/retrieval_audit.json"):
        shutil.copyfile(PKG / f, OUT / "preflight" / Path(f).name)
    print(json.dumps({"leakage": log["leakage_eval"]["checks"], "retrieval": {k: v for k, v in retrieval_audit.items()
                                                                              if not isinstance(v, (list, dict))}}, indent=1))
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
    cfg = json.loads((PKG / "configs/experiment.json").read_text(encoding="utf-8"))
    for f in PROVENANCE:
        shutil.copyfile(ROOT / "scripts/nl2mql" / f, PKG / "provenance" / f)
    files = package_files()
    C.need((PKG / f"{C.PACKAGE}.ipynb").is_file(), "notebook missing: run make_mql0008_notebook.py first")
    C.need(C.sha256_file(PKG / C.CORPUS_FILE) == C.EXPECTED_CORPUS_SHA256, "packaged corpus hash")
    n_corpus = sum(1 for _ in open(PKG / C.CORPUS_FILE, encoding="utf-8"))
    man = {"package": C.PACKAGE, "experiment": C.EXPERIMENT, "status": "PACKAGE BUILT - NOT YET EXECUTED", "built_at": C.now(),
           "arms": list(C.ARMS),
           "retrieval_corpus": {"source_path": cfg["corpus"]["source_path"], "package_path": C.CORPUS_FILE,
                                "records": n_corpus, "sha256": C.sha256_file(PKG / C.CORPUS_FILE)},
           "model": {"id": cfg["model"]["id"], "revision": cfg["model"]["revision"]},
           "evaluation": {"n_items": cfg["evaluation"]["n_items"], "ids_sha256": cfg["evaluation"]["ids_sha256"]},
           "retrieval_config_fingerprint": C.retrieval_fingerprint(cfg),
           "files": {n: {"bytes": (PKG / n).stat().st_size, "sha256": C.sha256_file(PKG / n)} for n in files}}
    C.write_json(PKG / "package_manifest.json", man)
    (PKG / "SHA256SUMS.txt").write_text("".join(f"{m['sha256']}  {n}\n" for n, m in man["files"].items()),
                                         encoding="utf-8", newline="\n")
    C.verify_package(PKG)
    if no_zip:
        return
    if ZIP_PATH.exists():
        ZIP_PATH.unlink()
    with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as z:
        for n in files + ["package_manifest.json", "SHA256SUMS.txt"]:
            z.write(PKG / n, f"{C.PACKAGE}/{n}")
    print("zip:", ZIP_PATH, ZIP_PATH.stat().st_size, C.sha256_file(ZIP_PATH))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--finalize":
        manifest_and_zip()
    else:
        main()
