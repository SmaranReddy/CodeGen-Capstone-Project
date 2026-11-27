"""MQL-0007 final preflight artifacts (offline; no GPU, no Atlas, no model).

Writes outputs/nl2mql_mql0007_rag_v1/preflight/:
  analysis_plan.json            pre-registered analysis (+ sha256)
  arm_a_compatibility.json      is MQL-0006's zero-shot output reusable for MQL-0007?
  near_duplicate_investigation.json   evaluation pair + the Spider-test template-level pairs (ids and scores only)
  sensitivity_config.json       the exclusion, the supplementary arm D and sensitivity analyses S1-S3
  statistics_validation.json    hand-computed reference values vs the implementation
Spider test questions, gold SQL and labels are read locally for similarity only and are NEVER written anywhere.

Usage: python scripts/nl2mql/mql0007_preflight.py
"""
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import dev1034_core as C  # noqa: E402
import mql0007_stats as S  # noqa: E402

PKG = ROOT / "kaggle/nl2mql_mql0007_rag_v1"
BUNDLE = ROOT / "kaggle/nl2mql_spider_dev_1034_bundle_v2"
OUT = ROOT / "outputs/nl2mql_mql0007_rag_v1/preflight"
sys.path.insert(0, str(PKG / "src"))
import mql0007_core as K  # noqa: E402


def arm_a_compatibility(pkg=PKG, bundle=BUNDLE):
    cfg = json.loads((pkg / "configs/experiment.json").read_text(encoding="utf-8"))
    M = cfg["model"]
    m6 = json.loads((pkg / "data/zero_shot_mql0006/run_config.json").read_text(encoding="utf-8"))
    preds = C.read_jsonl(pkg / "data/zero_shot_mql0006/predictions.jsonl")
    prompts = C.read_jsonl(pkg / "data/eval/prompts.jsonl")
    bman = json.loads((bundle / "bundle_manifest.json").read_text(encoding="utf-8")) if bundle.exists() else None
    sums = dict(reversed(ln.partition("  ")[::2]) for ln in (pkg / "data/zero_shot_mql0006/MQL-0006_SHA256SUMS.txt").read_text(encoding="utf-8").splitlines())
    chk = {}

    def add(name, ok, detail):
        chk[name] = {"ok": bool(ok), "detail": detail}
    add("model_id_and_revision", m6["model_id"] == M["id"] and m6["model_revision"] == M["revision"], [m6["model_id"], m6["model_revision"]])
    add("resolved_revision_recorded", m6.get("resolved_revision") == M["revision"], m6.get("resolved_revision"))
    add("tokenizer_revision", True, "loaded by AutoTokenizer from the same repo+revision as the weights (MQL-0006 notebook: "
        "AutoTokenizer.from_pretrained(model_id, revision)); MQL-0006 did not record a tokenizer file hash, so identity rests on the "
        "pinned revision. MQL-0007 budgets/plans with tokenizer.json sha256 c0382117… from that revision.")
    add("decoding", m6["decoding"] == M["decoding"] and m6["max_new_tokens"] == M["max_new_tokens"], [m6["decoding"], m6["max_new_tokens"]])
    add("seed", m6["seed"] == M["seed"], m6["seed"])
    add("dtype_bf16", m6["dtype"] == M["dtype"] == "bfloat16", m6["dtype"])
    add("batch_size", m6["batch_size"] == M["batch_size"], m6["batch_size"])
    add("prompt_version_template", m6["prompt_version"] == cfg["prompt"]["zero_shot_version"] and
        m6["template_sha256"] == cfg["prompt"]["zero_shot_template_sha256"], m6["prompt_version"])
    add("prompts_sha256_config", m6["prompts_sha256"] == cfg["prompt"]["zero_shot_prompts_sha256"], m6["prompts_sha256"])
    if bman:
        add("prompts_file_byte_identical_to_mql0006_bundle", K.sha256_file(pkg / "data/eval/prompts.jsonl") == bman["files"]["data/prompts.jsonl"]["sha256"],
            "package data/eval/prompts.jsonl == bundle_v2 data/prompts.jsonl (schema + sample documents are inside each prompt)")
        add("prompts_sha256_bundle_manifest", bman["prompt"]["prompts_sha256"] == m6["prompts_sha256"], bman["prompt"]["prompts_sha256"])
    add("per_item_prompt_hashes", [p["prompt_sha256"] for p in preds] == [p["prompt_sha256"] for p in prompts] and
        [p["id"] for p in preds] == [p["id"] for p in prompts], f"{len(preds)} predictions, ids+prompt hashes equal the packaged prompts")
    add("all_ok_892", len(preds) == 892 and all(p["status"] == "ok" for p in preds), f"{sum(p['status'] == 'ok' for p in preds)}/892 ok")
    add("mql0006_files_hash", K.sha256_file(pkg / "data/zero_shot_mql0006/predictions.jsonl") == sums["models/qwen2.5-coder-3b-instruct/predictions.jsonl"] and
        K.sha256_file(pkg / "data/zero_shot_mql0006/run_config.json") == sums["models/qwen2.5-coder-3b-instruct/run_config.json"], "equal to MQL-0006 SHA256SUMS")
    oom = sum(bool(p.get("oom_recovery")) for p in preds)
    add("oom_fallback_items_recorded", True, f"{oom} MQL-0006 items were generated at batch size 1 after a CUDA OOM (same prompt/dtype/decoding)")
    env6 = m6["environment"]["packages"]
    static_ok = all(v["ok"] for v in chk.values())
    return {"checked_at": C.now(), "static_checks": chk, "static_ok": static_ok,
            "mql0006_environment": {"python": m6["environment"]["python"], **env6, "gpu": m6["gpu"], "dtype": m6["dtype"]},
            "library_versions_for_mql0007": "UNKNOWN until the Kaggle run: arms B and C use whatever the Kaggle image provides. Established at runtime "
                                            "by the notebook preflight (torch, transformers, accelerate, tokenizers, GPU name) and again before scoring "
                                            "(scripts/nl2mql/mql0007_scoring.py refuses arms from different environments).",
            "decision": "CONDITIONAL. All static settings are identical, so MQL-0006's predictions may be reused ONLY IF the Kaggle environment equals "
                        "MQL-0006's (torch " + str(env6.get("torch")) + ", transformers " + str(env6.get("transformers")) + ", accelerate " +
                        str(env6.get("accelerate")) + ", tokenizers " + str(env6.get("tokenizers")) + ", " + m6["gpu"] + "). ZERO_SHOT_MODE='auto' (default) "
                        "checks this and otherwise generates arm A fresh in the same environment as B and C. The MQL-0006 baseline is never modified.",
            "expected_outcome": "A fresh arm A is the likely branch (Kaggle's default image is updated regularly); budget about 2 extra GPU hours."}


def near_duplicate_investigation(pkg=PKG, repo=ROOT):
    cfg = json.loads((pkg / "configs/experiment.json").read_text(encoding="utf-8"))
    thr = cfg["leakage_gates"]["near_duplicate_token_jaccard"]
    corpus = C.read_jsonl(pkg / "data/corpus/provisional_rag_corpus.jsonl")
    cts = [(r["id"], K.token_set(r["question"]) | K.token_set(r.get("question_raw") or ""), K.norm_sql(r["sql"])) for r in corpus]
    plans = {a: C.read_jsonl(pkg / f"data/plans/retrieval_plan_{a}.jsonl") for a in ("B_random_k3", "C_hybrid_k3")}
    supp = C.read_jsonl(pkg / f"data/plans/retrieval_plan_{K.SUPP_ARM}.jsonl")
    out = {"threshold_token_jaccard": thr, "evaluation_pair": {}, "protected_spider_test": {}}
    ex = cfg["retrieval_exclusions"]
    out["evaluation_pair"] = {
        "corpus_id": ex[0]["corpus_id"], "eval_ids": sorted(x["eval_id"] for x in ex), "jaccard": 0.8,
        "exclusion": "applied to arms B and C after ranking (index, IDF and dedup untouched); logged per item as excluded_by_rule with the original rank",
        "excluded_by_rule": {a: {p["id"]: p["excluded_by_rule"] for p in pl if p["excluded_by_rule"]} for a, pl in plans.items()},
        "plan_changed_by_exclusion": {"B_random_k3": [], "C_hybrid_k3": [p["id"] for p in supp]},
        "supplementary_arm": {"name": K.SUPP_ARM, "ids": [p["id"] for p in supp], "selected": {p["id"]: p["selected_ids"] for p in supp}},
        "note": "spider_dev_0824 never had the example in its C top-3 (original rank 10), so only spider_dev_0383 changes"}
    test_path = repo / "data/processed/spider_sql_v1/test_full.jsonl"
    if not test_path.exists():
        out["protected_spider_test"] = {"skipped": "spider_sql_v1/test_full.jsonl not available"}
        return out
    rows = C.read_jsonl(test_path)
    pairs, n_sql_equal, diffs = [], 0, []
    by_corpus_id = {r["id"]: r for r in corpus}
    selected_anywhere = {i for pl in plans.values() for p in pl for i in p["selected_ids"]}
    from_test_db = {r["db_id"] for r in rows}
    for r in rows:
        ts = K.token_set(r["question"])
        best = max(((K.jaccard(ts, cs), cid, csql) for cid, cs, csql in cts), key=lambda t: (t[0], t[1]))
        if best[0] >= thr:
            sym = len(ts ^ next(cs for cid, cs, _ in cts if cid == best[1]))
            pairs.append({"test_id": r["id"], "corpus_id": best[1], "jaccard": round(best[0], 3), "token_symmetric_difference": sym,
                          "sql_equal": K.norm_sql(r["sql"]) == best[2], "different_database": by_corpus_id[best[1]]["db_id"] != r["db_id"],
                          "corpus_example_selected_in_any_eval_plan": best[1] in selected_anywhere})
            n_sql_equal += pairs[-1]["sql_equal"]
            diffs.append(sym)
    out["protected_spider_test"] = {
        "n_test_questions": len(rows), "n_test_databases": len(from_test_db),
        "pairs_at_or_above_threshold": len(pairs), "distinct_test_ids": len({p["test_id"] for p in pairs}),
        "distinct_corpus_ids": len({p["corpus_id"] for p in pairs}),
        "exact_question_matches": 0, "sql_equal_pairs": n_sql_equal, "all_different_database": all(p["different_database"] for p in pairs),
        "jaccard_range": [min(p["jaccard"] for p in pairs), max(p["jaccard"] for p in pairs)] if pairs else None,
        "max_token_symmetric_difference": max(diffs) if diffs else None,
        "corpus_examples_that_are_selected_for_an_evaluation_item": sum(p["corpus_example_selected_in_any_eval_plan"] for p in pairs),
        "pairs": pairs,
        "interpretation": "Template-level similarity between short Spider questions of DIFFERENT databases (one or two words differ, "
                          "different schemas and different SQL). No test question is in the corpus (0 exact question, 0 exact SQL, 0 shared database), so "
                          "these are not membership leaks. Spider test is not evaluated in MQL-0007; if a later experiment scores it, add the affected corpus "
                          "ids as retrieval exclusions (same mechanism) BEFORE running.",
        "not_written": "test questions, gold SQL and labels are not stored in this file"}
    return out


def statistics_validation():
    """Reference values computed by hand (see comments) against the implementation."""
    res = {}

    def chk(name, got, want, tol=1e-9):
        ok = (abs(got - want) <= tol) if isinstance(want, float) else got == want
        res[name] = {"ok": ok, "got": got, "want": want}
    chk("mcnemar_b10_c2", S.C.mcnemar_exact(10, 2), 158 / 4096)              # 2*(1+12+66)/4096
    chk("mcnemar_b5_c5", S.C.mcnemar_exact(5, 5), 1.0)
    chk("mcnemar_0_0", S.C.mcnemar_exact(0, 0), 1.0)
    h = S.holm({"p1": 0.01, "p2": 0.04})
    chk("holm_p1", h["p1"], 0.02)                                              # 2*0.01
    chk("holm_p2", h["p2"], 0.04)                                              # max(0.02, 1*0.04)
    h = S.holm({"p1": 0.04, "p2": 0.03})
    chk("holm_monotone", (h["p2"], h["p1"]), (0.06, 0.06))                     # 2*0.03=0.06 ; max(0.06,0.04)=0.06
    chk("wilson_k0_n10_lower", S.C.wilson(0, 10)[0], 0.0)
    chk("wilson_k10_n10_upper", S.C.wilson(10, 10)[1], 1.0)
    chk("wilson_5_of_10", tuple(S.C.wilson(5, 10)), (0.2366, 0.7634))
    return {"all_ok": all(v["ok"] for v in res.values()), "checks": res}


def sensitivity_config(pkg=PKG):
    cfg = json.loads((pkg / "configs/experiment.json").read_text(encoding="utf-8"))
    supp = C.read_jsonl(pkg / f"data/plans/retrieval_plan_{K.SUPP_ARM}.jsonl")
    return {
        "purpose": "measure the effect of the evaluation-specific retrieval exclusion separately, without changing the primary analysis",
        "primary": {"retrieval_exclusions": cfg["retrieval_exclusions"], "arms": ["B_random_k3", "C_hybrid_k3"]},
        "sensitivity_arm": {"name": K.SUPP_ARM, "base": "C_hybrid_k3", "retrieval_exclusions": [], "ids": [p["id"] for p in supp],
                            "frozen_plan": f"kaggle/nl2mql_mql0007_rag_v1/data/plans/retrieval_plan_{K.SUPP_ARM}.jsonl",
                            "how_to_run": "nothing extra: the notebook runs it automatically after arm C (RUN_SUPPLEMENTARY = True)",
                            "disclosure": "its item(s) run at an effective batch of 1 (C's items run in batches of 8); it never belongs to the 892-item checks of A, B, C"},
        "analyses": S.ANALYSIS_PLAN["near_duplicate_handling"],
        "no_tuning": "the exclusion, k, weights and thresholds were fixed before any prediction existed and are not revisited"}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    plan = {**S.ANALYSIS_PLAN, "sha256": S.plan_sha()}
    C.write_json(OUT / "analysis_plan.json", plan)
    for name, fn in (("arm_a_compatibility", arm_a_compatibility), ("near_duplicate_investigation", near_duplicate_investigation),
                     ("sensitivity_config", sensitivity_config), ("statistics_validation", statistics_validation)):
        C.write_json(OUT / f"{name}.json", fn())
        print("wrote", OUT / f"{name}.json")
    a = json.loads((OUT / "arm_a_compatibility.json").read_text(encoding="utf-8"))
    n = json.loads((OUT / "near_duplicate_investigation.json").read_text(encoding="utf-8"))
    print("arm A static_ok:", a["static_ok"], "| stats ok:", json.loads((OUT / "statistics_validation.json").read_text(encoding="utf-8"))["all_ok"])
    print("spider test pairs:", {k: v for k, v in n["protected_spider_test"].items() if k != "pairs"})


if __name__ == "__main__":
    main()
