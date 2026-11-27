"""MQL-0007 statistical analysis (offline; no Atlas, no model). Pure functions over scored outcomes.

Pre-registered in outputs/nl2mql_mql0007_rag_v1/preflight/analysis_plan.json BEFORE any prediction exists:
  * metric        correct = (outcome == "correct") over ALL 892 items; not_comparable, invalid, failed count as not correct
  * primary       C vs A and C vs B (paired, exact two-sided McNemar), Holm-corrected over these two
  * secondary     B vs A (reported unadjusted)
  * intervals     Wilson 95% per arm; paired bootstrap 95% CI of the accuracy difference
  * breakdowns    hardness, database, gold-SQL query pattern (patterns.sql_patterns on the local gold SQL; never shipped)
  * sensitivity   S1 C with arm D's outcome swapped in for the items the exclusion changed
                  S2 all arms on the 890 items without the two near-duplicate items
                  S3 reference_status == ok subset
No retrieval or analysis choice may be made from evaluation outcomes.
"""
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import dev1034_core as C  # noqa: E402  (unchanged MQL-0006 core: wilson, mcnemar_exact, paired_bootstrap_ci)

ARMS = ("A_zero_shot", "B_random_k3", "C_hybrid_k3")
PRIMARY = (("C_hybrid_k3", "A_zero_shot"), ("C_hybrid_k3", "B_random_k3"))
SECONDARY = (("B_random_k3", "A_zero_shot"),)
CATEGORIES = ("correct", "incorrect", "invalid", "execution_error", "timeout", "generation_failure",
              "infrastructure_failure", "not_comparable")

ANALYSIS_PLAN = {
    "plan_version": "mql0007-analysis-v1",
    "registered": "2026-11-06, before any MQL-0007 prediction or score exists",
    "metric": "cross-engine execution accuracy: correct / 892, correct == (outcome == 'correct'); not_comparable, invalid, "
              "timeout, failed all count as not correct (as MQL-0006); denominator is always the full evaluation set",
    "primary_comparisons": [list(p) for p in PRIMARY],
    "primary_test": "exact two-sided McNemar on the discordant pairs; Holm correction over the two primary p-values; alpha 0.05",
    "secondary_comparisons": [list(p) for p in SECONDARY],
    "secondary_note": "reported unadjusted; not part of the Holm family",
    "intervals": "Wilson 95% per arm; paired bootstrap 95% percentile CI (10,000 resamples, seed 20261029) of accuracy differences",
    "breakdowns": ["hardness", "database", "query_pattern (sql_patterns of the local gold SQL)"],
    "breakdown_note": "descriptive only; no multiplicity-adjusted claims",
    "near_duplicate_handling": {
        "primary": "all 892 items in every arm; the evaluation-specific retrieval exclusion (spider_train_00650 for spider_dev_0383 and "
                   "spider_dev_0824) applies identically to arms B and C; arm A has no retrieval",
        "S1_swap_supplementary_arm": "arm C's outcome for the items whose selection the exclusion changed is replaced by arm D's outcome",
        "S2_drop_pair": "all arms on the 890 items without spider_dev_0383 and spider_dev_0824",
        "S3_reference_ok": "items with reference_status == 'ok' only"},
    "forbidden": "retrieval settings, prompts, k, weights, thresholds or exclusions may not be changed after seeing any score",
}


def plan_sha():
    return hashlib.sha256(json.dumps(ANALYSIS_PLAN, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def category(e):
    """The eight reported outcome categories (generation errors and Atlas outages are kept apart)."""
    o = e["outcome"]
    if o in ("correct", "incorrect", "invalid", "timeout", "not_comparable"):
        return o
    if o == "failed":
        if e.get("failure_source") == "infrastructure":
            return "generation_failure" if e.get("detail") == "generation_error" else "infrastructure_failure"
        return "execution_error"
    raise ValueError(f"unknown outcome {o!r}")


def account(evals, expected_ids):
    """Exact accounting: every expected id exactly once, categories sum to N."""
    ids = [e["id"] for e in evals]
    if ids != list(expected_ids):
        raise ValueError(f"evaluation ids != expected ids ({len(ids)} vs {len(expected_ids)}, "
                         f"{len(set(ids))} unique)")
    cnt = Counter(category(e) for e in evals)
    out = {c: cnt.get(c, 0) for c in CATEGORIES}
    if sum(out.values()) != len(expected_ids):
        raise ValueError("category counts do not sum to N")
    return out


def holm(pvals):
    """Holm-Bonferroni adjusted p-values (monotone, capped at 1). pvals: {name: p}."""
    order = sorted(pvals, key=lambda k: (pvals[k], k))
    m, run, adj = len(order), 0.0, {}
    for i, k in enumerate(order):
        run = max(run, min(1.0, (m - i) * pvals[k]))
        adj[k] = run
    return adj


def acc(vec):
    k, n = sum(vec), len(vec)
    return {"correct": k, "total": n, "percent": round(100 * k / n, 2) if n else None, "wilson95": C.wilson(k, n)}


def paired(va, vb, boot=True):
    """A vs B on identical items. a_only = A correct & B not; b_only = B correct & A not."""
    assert len(va) == len(vb)
    a_only = sum(x and not y for x, y in zip(va, vb))
    b_only = sum(y and not x for x, y in zip(va, vb))
    n = len(va)
    out = {"n": n, "both_correct": sum(x and y for x, y in zip(va, vb)), "a_only": a_only, "b_only": b_only,
           "both_wrong": sum((not x) and (not y) for x, y in zip(va, vb)),
           "diff_a_minus_b": round((sum(va) - sum(vb)) / n, 4) if n else None,
           "mcnemar_exact_p": C.mcnemar_exact(a_only, b_only)}
    if boot and n:
        out["diff_bootstrap95"] = C.paired_bootstrap_ci([int(x) for x in va], [int(y) for y in vb])
    return out


def vectors(outcomes, ids):
    return {arm: [outcomes[arm][i] == "correct" for i in ids] for arm in outcomes}


def compare_arms(outcomes, ids, holm_family=True):
    """outcomes: {arm: {id: outcome}}. Primary (Holm), secondary, per-arm accuracy."""
    v = vectors(outcomes, ids)
    res = {"n": len(ids), "arms": {a: acc(v[a]) for a in v}, "primary": {}, "secondary": {}}
    raw = {}
    for a, b in PRIMARY:
        if a in v and b in v:
            key = f"{a}_vs_{b}"
            res["primary"][key] = paired(v[a], v[b])
            raw[key] = res["primary"][key]["mcnemar_exact_p"]
    if holm_family and raw:
        adj = holm(raw)
        for k, p in adj.items():
            res["primary"][k]["holm_adjusted_p"] = p
            res["primary"][k]["significant_at_0.05_after_holm"] = p < 0.05
    for a, b in SECONDARY:
        if a in v and b in v:
            res["secondary"][f"{a}_vs_{b}"] = paired(v[a], v[b])
    return res


def breakdown(outcomes, ids, key_of):
    """Per-group accuracy per arm (+ the C-vs-A discordant counts). key_of(id) -> iterable of group names."""
    groups = {}
    for i in ids:
        for g in key_of(i):
            groups.setdefault(g, []).append(i)
    out = {}
    for g, gi in sorted(groups.items()):
        v = vectors(outcomes, gi)
        row = {"n": len(gi), "arms": {a: acc(v[a]) for a in v}}
        if "C_hybrid_k3" in v and "A_zero_shot" in v:
            p = paired(v["C_hybrid_k3"], v["A_zero_shot"], boot=False)
            row["C_vs_A"] = {k: p[k] for k in ("a_only", "b_only", "diff_a_minus_b")}
        out[g] = row
    return out


def analyze(outcomes, ids, meta, patterns, near_dup_ids, supp_outcomes=None):
    """Full pre-registered analysis. outcomes {arm:{id:outcome}} over `ids` (all 892); meta {id: {hardness, db_id,
    reference_status}}; patterns {id: set of pattern labels}; supp_outcomes {id: outcome} for arm D (may be empty)."""
    out = {"analysis_plan_sha256": plan_sha(), "primary_analysis": compare_arms(outcomes, ids)}
    out["breakdowns"] = {
        "hardness": breakdown(outcomes, ids, lambda i: [meta[i]["hardness"]]),
        "database": breakdown(outcomes, ids, lambda i: [meta[i]["db_id"]]),
        "query_pattern": breakdown(outcomes, ids, lambda i: sorted(patterns[i]) or ["none"])}
    sens = {}
    if supp_outcomes and "C_hybrid_k3" in outcomes:
        swapped = {**outcomes, "C_hybrid_k3": {**outcomes["C_hybrid_k3"], **supp_outcomes}}
        sens["S1_swap_supplementary_arm_into_C"] = {"swapped_ids": sorted(supp_outcomes),
                                                    "result": compare_arms(swapped, ids),
                                                    "changed_outcome": {i: [outcomes["C_hybrid_k3"][i], o]
                                                                        for i, o in supp_outcomes.items()
                                                                        if outcomes["C_hybrid_k3"][i] != o}}
    keep = [i for i in ids if i not in set(near_dup_ids)]
    sens["S2_without_near_duplicate_items"] = compare_arms(outcomes, keep)
    ok = [i for i in ids if meta[i]["reference_status"] == "ok"]
    sens["S3_reference_status_ok"] = compare_arms(outcomes, ok)
    out["sensitivity"] = sens
    out["near_duplicate_items"] = {i: {a: outcomes[a][i] for a in outcomes} for i in near_dup_ids}
    return out
