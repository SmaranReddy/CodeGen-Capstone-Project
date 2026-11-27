"""MQL-0008 statistical analysis (offline; no Atlas, no model). Pure functions over scored outcomes.

Pre-registered in outputs/nl2mql_mql0008_approved_corpus_v2_1/preflight/analysis_plan.json BEFORE any MQL-0008
prediction exists (ANALYSIS_PLAN below is the registered text; its sha256 is stored in every scoring run).

  * metric      correct = (outcome == "correct") over ALL 892 items; every other outcome counts as not correct
  * primary     arm B (hybrid RAG, approved corpus v2.1) vs arm A (zero-shot): exact two-sided McNemar, one test
  * intervals   Wilson 95% per arm; paired bootstrap 95% percentile CI of accuracy(B) - accuracy(A)
  * secondary   exploratory only (outcome categories, transitions, strata, retrieval diagnostics, sensitivity)

The statistics primitives (wilson, mcnemar_exact, paired_bootstrap_ci) and the outcome categories are the unchanged
MQL-0006 / MQL-0007 ones (scripts/nl2mql/dev1034_core.py, scripts/nl2mql/mql0007_stats.py). Nothing is re-implemented.

  python scripts/nl2mql_mql0008/mql0008_stats.py --write-plan     write the plan file (refuses to overwrite a different plan)
"""
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "scripts/nl2mql"))
import dev1034_core as C  # noqa: E402  (unchanged MQL-0006 core)
import mql0007_stats as S7  # noqa: E402  (unchanged: category, account, acc, paired, CATEGORIES)

ARM_A, ARM_B = "arm_a_zero_shot", "arm_b_hybrid_rag"
ARMS = (ARM_A, ARM_B)
CATEGORIES = S7.CATEGORIES
PLAN_PATH = ROOT / "outputs/nl2mql_mql0008_approved_corpus_v2_1/preflight/analysis_plan.json"
PAIR_CLASSES = ("both_correct", "rag_fixes_baseline_error", "rag_introduces_error", "both_wrong")

ANALYSIS_PLAN = {
    "plan_version": "mql0008-analysis-v1",
    "experiment": "MQL-0008",
    "registered": "2026-11-16, before any MQL-0008 prediction or score exists (the Kaggle notebook had not been run)",
    "arms": {ARM_A: "zero-shot Qwen2.5-Coder-3B-Instruct", ARM_B: "hybrid RAG, approved corpus v2.1 (3,445 records)"},
    "sampling_unit": "the evaluation item (892 paired items; both arms answer the same question with the same schema)",
    "metric": "cross-engine execution accuracy: correct / 892, correct == (outcome == 'correct'). invalid, execution_error, "
              "timeout, generation_failure, infrastructure_failure, not_comparable and incorrect all count as not correct. "
              "The denominator is always the full 892-item set; no item is excluded.",
    "outcome_categories": list(CATEGORIES),
    "outcome_category_note": "the established categories of MQL-0006 / MQL-0007; 'generation_failure' is the model-failure "
                             "category, 'infrastructure_failure' is an Atlas outage (retried on re-run, never a model error)",
    "primary_comparison": [ARM_B, ARM_A],
    "primary_test": "exact two-sided McNemar (binomial) test on the discordant pairs; alpha 0.05",
    "multiplicity": "one primary comparison, so no multiple-comparison correction",
    "reported_for_primary": [
        "correct and accuracy per arm", "absolute difference B - A in percentage points", "Wilson 95% interval per arm",
        "2x2 paired correctness table", "items where RAG fixes a baseline error (B correct, A not)",
        "items where RAG introduces an error (A correct, B not)", "exact McNemar p-value",
        "paired bootstrap 95% percentile interval of the accuracy difference"],
    "bootstrap": {"resamples": C.BOOTSTRAP_REPS, "seed": C.BOOTSTRAP_SEED, "interval": "95% percentile",
                  "unit": "evaluation item, resampled with replacement, pairs kept together"},
    "claim_rule": "RAG is described as better (or worse) than zero-shot only if the McNemar p-value is below 0.05. A higher "
                  "raw accuracy alone is reported as a difference with its uncertainty, not as an improvement.",
    "secondary_exploratory": [
        "full outcome-category counts per arm and the A-category x B-category transition table",
        "per-record differences (paired_comparison.jsonl)",
        "accuracy by hardness, database and gold-SQL query pattern (labels from the local evaluation bundle)",
        "retrieval coverage: examples per item, empty-context items, context tokens",
        "retrieved example ids, ranks and scores for every item whose correctness differs between the arms",
        "outcome categories that RAG resolves or introduces",
        "sensitivity S2: both arms on the 890 items without the two template near-duplicate items",
        "sensitivity S3: both arms on the items with reference_status == 'ok'"],
    "secondary_note": "all secondary analyses are EXPLORATORY and descriptive; no p-value from them supports a claim",
    "historical_context_only": {
        "MQL-0006 zero-shot (no retrieval)": "421/892 (47.20%)",
        "MQL-0007 hybrid RAG (corpus v1, 3,589 records)": "437/892 (48.99%)",
        "note": "context only; not MQL-0008 results; MQL-0008 regenerates and scores both of its arms itself"},
    "forbidden": "retrieval settings, prompts, k, weights, thresholds, exclusions, the scorer or this plan may not be changed "
                 "after seeing any MQL-0008 score; nothing is tuned against the 892 evaluation results",
}


def plan_sha():
    return hashlib.sha256(json.dumps(ANALYSIS_PLAN, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def write_plan(path=PLAN_PATH):
    """Write the registered plan once. An existing file with another plan body is never overwritten."""
    path = Path(path)
    if path.exists():
        old = json.loads(path.read_text(encoding="utf-8"))
        if old.get("sha256") != plan_sha() or old.get("plan") != ANALYSIS_PLAN:
            raise RuntimeError(f"{path} holds a different analysis plan; a registered plan is never overwritten")
        return path
    C.write_json(path, {"sha256": plan_sha(), "sha256_of": "json.dumps(plan, sort_keys=True, ensure_ascii=False)",
                        "written_at": C.now(), "plan": ANALYSIS_PLAN})
    return path


def load_registered_plan(path=PLAN_PATH):
    """The plan file must exist and equal the plan in this module (same body, same hash)."""
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    if doc.get("plan") != ANALYSIS_PLAN or doc.get("sha256") != plan_sha():
        raise RuntimeError("the registered analysis plan differs from the plan in mql0008_stats.py")
    return doc


# ------------------------------------------------------------------ primary analysis
def vectors(outcomes, ids):
    return {arm: [outcomes[arm][i] == "correct" for i in ids] for arm in ARMS}


def primary(outcomes, ids):
    """Arm B vs arm A on the same items. outcomes: {arm: {id: outcome}}."""
    v = vectors(outcomes, ids)
    va, vb = v[ARM_A], v[ARM_B]
    n = len(ids)
    p = S7.paired(vb, va, boot=False)                 # a_only = B correct & A not ; b_only = A correct & B not
    fixes, breaks = p["a_only"], p["b_only"]
    out = {"n": n, "arms": {ARM_A: S7.acc(va), ARM_B: S7.acc(vb)},
           "difference_b_minus_a": {"correct": sum(vb) - sum(va),
                                    "percentage_points": round(100 * (sum(vb) - sum(va)) / n, 2) if n else None},
           "paired_table": {"both_correct": p["both_correct"], "rag_fixes_baseline_error": fixes,
                            "rag_introduces_error": breaks, "both_wrong": p["both_wrong"]},
           "mcnemar_exact_p": C.mcnemar_exact(fixes, breaks),
           "bootstrap95_difference_b_minus_a": C.paired_bootstrap_ci([int(x) for x in vb], [int(x) for x in va]) if n else None,
           "bootstrap": {"resamples": C.BOOTSTRAP_REPS, "seed": C.BOOTSTRAP_SEED}}
    out["significant_at_0.05"] = out["mcnemar_exact_p"] < 0.05
    d = out["difference_b_minus_a"]["correct"]
    if not out["significant_at_0.05"]:
        out["conclusion"] = ("no statistically significant difference between hybrid RAG and zero-shot "
                             f"(difference {d:+d} items, exact McNemar p = {out['mcnemar_exact_p']:.4g}); "
                             "the raw difference alone does not show that either arm is better")
    else:
        out["conclusion"] = (f"hybrid RAG is {'better' if d > 0 else 'worse'} than zero-shot "
                             f"(difference {d:+d} items, exact McNemar p = {out['mcnemar_exact_p']:.4g})")
    return out


def pair_class(a_correct, b_correct):
    if a_correct:
        return "both_correct" if b_correct else "rag_introduces_error"
    return "rag_fixes_baseline_error" if b_correct else "both_wrong"


# ------------------------------------------------------------------ secondary (exploratory)
def breakdown(outcomes, ids, key_of):
    groups = {}
    for i in ids:
        for g in key_of(i):
            groups.setdefault(g, []).append(i)
    out = {}
    for g, gi in sorted(groups.items()):
        v = vectors(outcomes, gi)
        out[g] = {"n": len(gi), "arms": {a: S7.acc(v[a]) for a in ARMS},
                  "rag_fixes_baseline_error": sum(b and not a for a, b in zip(v[ARM_A], v[ARM_B])),
                  "rag_introduces_error": sum(a and not b for a, b in zip(v[ARM_A], v[ARM_B]))}
    return out


def transitions(categories, ids):
    """A-category x B-category counts. categories: {arm: {id: category}}."""
    cnt = Counter((categories[ARM_A][i], categories[ARM_B][i]) for i in ids)
    table = {a: {b: cnt.get((a, b), 0) for b in CATEGORIES} for a in CATEGORIES}
    resolved = {a: table[a]["correct"] for a in CATEGORIES if a != "correct" and table[a]["correct"]}
    introduced = {b: table["correct"][b] for b in CATEGORIES if b != "correct" and table["correct"][b]}
    return {"rows_arm_a_columns_arm_b": table, "arm_a_error_categories_resolved_by_rag": resolved,
            "error_categories_introduced_by_rag": introduced}


def retrieval_diagnostics(plan_rows, pairs):
    """plan_rows: arm B retrieval log rows (in evaluation order). pairs: {id: pair class}."""
    n_sel = Counter(len(r["selected_ids"]) for r in plan_rows)
    by_class = {}
    for r in plan_rows:
        by_class.setdefault(pairs[r["id"]], []).append(r)
    mean = lambda xs: round(sum(xs) / len(xs), 4) if xs else None   # noqa: E731
    return {"items": len(plan_rows), "examples_per_item": {str(k): n_sel[k] for k in sorted(n_sel)},
            "empty_context_items": sum(bool(r["empty_context"]) for r in plan_rows),
            "items_with_over_budget_skips": sum(bool(r["dropped_budget"]) for r in plan_rows),
            "items_with_rule_exclusions": [r["id"] for r in plan_rows if r["excluded_by_rule"]],
            "context_tokens": {"min": min(r["context_tokens"] for r in plan_rows), "max": max(r["context_tokens"] for r in plan_rows),
                               "mean": mean([r["context_tokens"] for r in plan_rows])},
            "by_pair_class": {c: {"items": len(rs), "mean_top1_score": mean([r["selected_scores"][0] for r in rs if r["selected_scores"]]),
                                  "mean_context_tokens": mean([r["context_tokens"] for r in rs])}
                              for c, rs in sorted(by_class.items())}}


def paired_rows(evals, ids, meta, plan_rows):
    """One row per evaluation item: both arms' outcome and category, the pair class, and what arm B retrieved."""
    by = {a: {e["id"]: e for e in evals[a]} for a in ARMS}
    plan = {r["id"]: r for r in plan_rows}
    rows = []
    for i in ids:
        ea, eb = by[ARM_A][i], by[ARM_B][i]
        ac, bc = ea["outcome"] == "correct", eb["outcome"] == "correct"
        p = plan[i]
        rows.append({"id": i, "db_id": meta[i]["db_id"], "hardness": meta[i]["hardness"],
                     "reference_status": meta[i]["reference_status"],
                     "arm_a": {"outcome": ea["outcome"], "category": S7.category(ea), "correct": ac, "detail": ea.get("detail")},
                     "arm_b": {"outcome": eb["outcome"], "category": S7.category(eb), "correct": bc, "detail": eb.get("detail")},
                     "pair_class": pair_class(ac, bc), "category_changed": S7.category(ea) != S7.category(eb),
                     "retrieved_ids": p["selected_ids"], "retrieved_ranks": p.get("selected_ranks"),
                     "retrieved_scores": p["selected_scores"], "retrieved_db_ids": p["selected_db_ids"],
                     "context_tokens": p["context_tokens"], "excluded_by_rule": p["excluded_by_rule"]})
    return rows


def analyze(evals, ids, meta, patterns, near_dup_ids, plan_rows):
    """Full pre-registered analysis from saved per-record results. evals: {arm: [evaluation records in id order]};
    meta: {id: {hardness, db_id, reference_status}}; patterns: {id: set}; plan_rows: arm B retrieval log."""
    acct = {a: S7.account(evals[a], ids) for a in ARMS}                 # raises unless exactly the expected ids, once each
    outcomes = {a: {e["id"]: e["outcome"] for e in evals[a]} for a in ARMS}
    cats = {a: {e["id"]: S7.category(e) for e in evals[a]} for a in ARMS}
    rows = paired_rows(evals, ids, meta, plan_rows)
    pairs = {r["id"]: r["pair_class"] for r in rows}
    keep = [i for i in ids if i not in set(near_dup_ids)]
    ok = [i for i in ids if meta[i]["reference_status"] == "ok"]
    out = {"experiment": "MQL-0008", "analysis_plan_sha256": plan_sha(), "denominator": len(ids),
           "primary_analysis": primary(outcomes, ids),
           "secondary_exploratory": {
               "label": "EXPLORATORY - descriptive only, no claim is based on these",
               "outcome_categories": acct,
               "category_transitions": transitions(cats, ids),
               "breakdowns": {"hardness": breakdown(outcomes, ids, lambda i: [meta[i]["hardness"]]),
                              "database": breakdown(outcomes, ids, lambda i: [meta[i]["db_id"]]),
                              "query_pattern": breakdown(outcomes, ids, lambda i: sorted(patterns[i]) or ["none"])},
               "retrieval": retrieval_diagnostics(plan_rows, pairs),
               "changed_items": [{k: r[k] for k in ("id", "db_id", "pair_class", "retrieved_ids", "retrieved_ranks", "retrieved_scores")}
                                 | {"arm_a_category": r["arm_a"]["category"], "arm_b_category": r["arm_b"]["category"]}
                                 for r in rows if r["arm_a"]["correct"] != r["arm_b"]["correct"]],
               "sensitivity": {"S2_without_near_duplicate_items": primary(outcomes, keep),
                               "S3_reference_status_ok": primary(outcomes, ok)},
               "near_duplicate_items": {i: {a: outcomes[a][i] for a in ARMS} for i in near_dup_ids}},
           "historical_context_only": ANALYSIS_PLAN["historical_context_only"]}
    return out, rows


if __name__ == "__main__":
    if sys.argv[1:] == ["--write-plan"]:
        print("analysis plan:", write_plan(), "sha256", plan_sha())
    else:
        print(__doc__)
