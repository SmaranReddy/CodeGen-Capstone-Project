"""MQL-0009 statistical analysis (offline; no Atlas, no model). Pure functions over scored outcomes.

The analysis plan below is REGISTERED before any MQL-0009 prediction exists. Its sha256 is frozen in the Kaggle
package config (configs/experiment.json -> analysis_plan_sha256) and in
outputs/nl2mql_mql0009_retrieval_control/preflight/analysis_plan.json. Changing it after predictions exist is forbidden;
a changed plan is a new plan version with its own registration.

Unit of analysis = evaluation question. The five random-seed runs are NEVER pooled as 5 x N independent items: the
random arm enters the primary contrast as the per-question mean correctness over the five seeds (R-bar_q in {0, .2, ..,
1}). Query-family labels come from the LOCAL gold SQL, only here, after inference (never shipped to Kaggle).
"""
import hashlib
import json
import random
import statistics
import sys
from collections import Counter
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "scripts/nl2mql"))
import dev1034_core as C   # noqa: E402  (unchanged MQL-0006 core: wilson, mcnemar_exact, paired_bootstrap_ci)
import mql0007_stats as S7  # noqa: E402  (unchanged: category, account, holm)

ZERO, FIXED, HYBRID = "A_zero_shot", "C_fixed", "D_hybrid"
RANDOM = ("B_random_s1", "B_random_s2", "B_random_s3", "B_random_s4", "B_random_s5")
ARM_RUNS = (ZERO, *RANDOM, FIXED, HYBRID)
RBAR = "B_random_mean"
CATEGORIES = S7.CATEGORIES
SIGNFLIP_REPS = 20000
SIGNFLIP_SEED = 20261118
PLAN_PATH = ROOT / "outputs/nl2mql_mql0009_retrieval_control/preflight/analysis_plan.json"

ANALYSIS_PLAN = {
    "plan_version": "mql0009-analysis-v1",
    "experiment": "MQL-0009",
    "registered": "2026-11-18, before any MQL-0009 prediction or score exists",
    "question": "Does the MQL-0008 hybrid-RAG gain come from RELEVANT retrieval, or from supplying any three demonstrations?",
    "unit": "evaluation question (N = all items of the frozen held-out Spider-train evaluation set); random seeds are "
            "never treated as independent items",
    "metric": "cross-engine execution accuracy: correct == (outcome == 'correct'); invalid, execution_error, timeout, "
              "not_comparable, generation_failure, infrastructure_failure all count as not correct; denominator = full set",
    "primary": {
        "H1_relevance": "D_hybrid vs B_random_mean: difference of means of d_q = D_q - Rbar_q over questions; paired bootstrap "
                        "95% percentile CI (10,000 resamples, seed 20261029, dev1034_core.paired_bootstrap_ci); two-sided "
                        "sign-flip randomisation test on d_q (20,000 flips, seed 20261118)",
        "H2_relevance_vs_fixed": "D_hybrid vs C_fixed: exact two-sided McNemar; paired bootstrap 95% CI",
        "multiplicity": "Holm over {H1, H2}; alpha 0.05",
        "interpretation_rule": "relevance effect supported only if H1 is significant after Holm with D > Rbar; "
                               "'examples-only' explanation supported if Rbar - A and C - A are positive while H1 is not",
    },
    "secondary": {
        "pairs_mcnemar": [["D_hybrid", "A_zero_shot"], ["C_fixed", "A_zero_shot"]] +
                         [[s, "A_zero_shot"] for s in RANDOM] + [["D_hybrid", s] for s in RANDOM],
        "B_random_mean_vs_A_zero_shot": "sign-flip test + paired bootstrap on Rbar_q - A_q",
        "note": "unadjusted; descriptive support for the primary family only",
    },
    "random_seed_variability": "per-seed accuracy, mean, sample SD, min, max; pairwise seed agreement on correctness; "
                               "per-question Rbar distribution",
    "rates_per_arm": ["accuracy", "invalid", "execution_error", "timeout", "executable_but_incorrect (= incorrect)",
                      "not_comparable", "generation_failure", "infrastructure_failure"],
    "breakdowns_exploratory": ["query_family (local gold SQL: set_op_or_subquery > join > single_table)",
                               "gold-SQL pattern", "hardness", "database"],
    "transitions": "per-question category transition matrices and fixes/regressions for A->D, A->C, A->each seed, C->D",
    "tokens": "per-arm prompt input tokens (generator count) and planned context tokens: mean, median, max",
    "sensitivity": {"S1_reference_ok": "all comparisons on items with reference_status == 'ok'",
                    "S2_no_exclusion_items": "all comparisons without items that have any evaluation-specific retrieval exclusion",
                    "S3_no_sql_equal_items": "all comparisons without the items whose normalised gold SQL equals a corpus SQL "
                                             "(leakage gate L6, report-only; list in preflight/leakage_audit.json)"},
    "forbidden": "retrieval settings, prompts, K, seeds, fixed ids, weights, thresholds, exclusions or this plan may not be "
                 "changed after seeing any prediction or score",
}


def plan_sha():
    return hashlib.sha256(json.dumps(ANALYSIS_PLAN, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def write_plan(path=PLAN_PATH):
    """Register the plan (refuses to overwrite a different registered plan)."""
    path = Path(path)
    doc = {"analysis_plan": ANALYSIS_PLAN, "analysis_plan_sha256": plan_sha(), "written_at": C.now()}
    if path.exists():
        old = json.loads(path.read_text(encoding="utf-8"))
        if old.get("analysis_plan_sha256") != plan_sha():
            raise RuntimeError(f"{path} holds a DIFFERENT registered plan; never overwrite a registered plan")
        return old
    C.write_json(path, doc)
    return doc


def load_registered_plan(path=PLAN_PATH):
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    if doc.get("analysis_plan_sha256") != plan_sha():
        raise RuntimeError("registered analysis plan != the plan in mql0009_stats.py")
    return doc


# ------------------------------------------------------------------ primitives
def correct_vec(evals_by_id, ids):
    return [1 if evals_by_id[i]["outcome"] == "correct" else 0 for i in ids]


def signflip_p(d, reps=SIGNFLIP_REPS, seed=SIGNFLIP_SEED):
    """Two-sided paired randomisation test of mean(d) == 0 (Monte Carlo sign flips; +1 smoothing)."""
    n = len(d)
    obs = abs(sum(d))
    nz = [x for x in d if x]
    if not nz:
        return 1.0
    rng = random.Random(seed)
    hits = 0
    for _ in range(reps):
        s = sum(x if rng.random() < 0.5 else -x for x in nz)
        if abs(s) >= obs - 1e-12:
            hits += 1
    return round((hits + 1) / (reps + 1), 6) if n else None


def paired_binary(va, vb):
    out = S7.paired([bool(x) for x in va], [bool(y) for y in vb])
    return out


def paired_fractional(va, vb):
    d = [a - b for a, b in zip(va, vb)]
    n = len(d)
    return {"n": n, "mean_a": round(sum(va) / n, 4), "mean_b": round(sum(vb) / n, 4), "diff_a_minus_b": round(sum(d) / n, 4),
            "diff_bootstrap95": C.paired_bootstrap_ci(list(va), list(vb)), "signflip_p": signflip_p(d),
            "questions_a_better": sum(x > 0 for x in d), "questions_b_better": sum(x < 0 for x in d)}


def rates(evals, ids):
    acct = S7.account(evals, ids)
    n = len(ids)
    r = {k: {"count": v, "percent": round(100 * v / n, 2)} for k, v in acct.items()}
    r["executable_but_incorrect"] = r["incorrect"]
    return {"n": n, "accounting": acct, "accuracy": S7.acc([e["outcome"] == "correct" for e in evals]), "rates": r}


def family_of(patterns):
    if patterns & {"nested", "anti_join", "intersect", "union"}:
        return "set_op_or_subquery"
    if "join" in patterns:
        return "join"
    return "single_table"


def transitions(ev_a, ev_b, ids):
    m = Counter((S7.category(ev_a[i]), S7.category(ev_b[i])) for i in ids)
    fixes = sum(ev_a[i]["outcome"] != "correct" and ev_b[i]["outcome"] == "correct" for i in ids)
    regs = sum(ev_a[i]["outcome"] == "correct" and ev_b[i]["outcome"] != "correct" for i in ids)
    return {"matrix": {f"{a}->{b}": c for (a, b), c in sorted(m.items())}, "fixes": fixes, "regressions": regs,
            "net": fixes - regs}


def token_stats(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None
    return {"mean": round(sum(xs) / len(xs), 1), "median": statistics.median(xs), "max": max(xs), "total": sum(xs)}


# ------------------------------------------------------------------ full analysis
def analyze(evals_by_run, ids, meta, patterns, exclusion_ids=(), preds_by_run=None, plans_by_run=None, sql_equal_ids=()):
    """evals_by_run {run: [eval records in id order]}; meta {id: {hardness, db_id, reference_status}};
    patterns {id: set(sql pattern labels)} from LOCAL gold SQL; preds_by_run / plans_by_run for token usage."""
    for r in ARM_RUNS:
        if r not in evals_by_run:
            raise ValueError(f"missing arm run {r}; the analysis needs all 8 runs over the same items")
    ev = {r: {e["id"]: e for e in evals_by_run[r]} for r in ARM_RUNS}
    out = {"analysis_plan_sha256": plan_sha(), "n": len(ids)}
    out["main"] = _compare(ev, ids)
    out["per_arm_rates"] = {r: rates(evals_by_run[r], ids) for r in ARM_RUNS}
    out["random_seed_variability"] = _seed_var(ev, ids)
    fam = {i: family_of(patterns[i]) for i in ids}
    out["breakdowns_exploratory"] = {
        "query_family": _breakdown(ev, ids, lambda i: [fam[i]]),
        "pattern": _breakdown(ev, ids, lambda i: sorted(patterns[i]) or ["select_only"]),
        "hardness": _breakdown(ev, ids, lambda i: [meta[i]["hardness"]]),
        "database": _breakdown(ev, ids, lambda i: [meta[i]["db_id"]]),
    }
    pairs = [(ZERO, HYBRID), (ZERO, FIXED), *[(ZERO, s) for s in RANDOM], (FIXED, HYBRID)]
    out["transitions"] = {f"{a}->{b}": transitions(ev[a], ev[b], ids) for a, b in pairs}
    s1 = [i for i in ids if meta[i]["reference_status"] == "ok"]
    s2 = [i for i in ids if i not in set(exclusion_ids)]
    s3 = [i for i in ids if i not in set(sql_equal_ids)]
    out["sensitivity"] = {"S1_reference_ok": {"n": len(s1), **_compare(ev, s1, secondary=False)},
                          "S2_no_exclusion_items": {"n": len(s2), **_compare(ev, s2, secondary=False)},
                          "S3_no_sql_equal_items": {"n": len(s3), **_compare(ev, s3, secondary=False)}}
    if preds_by_run:
        out["tokens"] = {r: {"input_tokens": token_stats([p.get("input_tokens") for p in preds_by_run[r]]),
                             "output_tokens": token_stats([p.get("output_tokens") for p in preds_by_run[r]]),
                             "context_tokens_planned": token_stats([p["context_tokens"] for p in (plans_by_run or {}).get(r, [])])}
                         for r in ARM_RUNS if r in preds_by_run}
    return out


def _vectors(ev, ids):
    v = {r: correct_vec(ev[r], ids) for r in ARM_RUNS}
    v[RBAR] = [sum(v[s][k] for s in RANDOM) / len(RANDOM) for k in range(len(ids))]
    return v


def _compare(ev, ids, secondary=True):
    if not ids:
        return {"primary": None}
    v = _vectors(ev, ids)
    acc = {r: S7.acc([bool(x) for x in v[r]]) for r in ARM_RUNS}
    acc[RBAR] = {"mean_correct": round(sum(v[RBAR]), 2), "total": len(ids), "percent": round(100 * sum(v[RBAR]) / len(ids), 2)}
    h1 = paired_fractional(v[HYBRID], v[RBAR])
    h2 = paired_binary(v[HYBRID], v[FIXED])
    adj = S7.holm({"H1": h1["signflip_p"], "H2": h2["mcnemar_exact_p"]})
    h1["holm_adjusted_p"], h2["holm_adjusted_p"] = adj["H1"], adj["H2"]
    out = {"accuracy": acc, "primary": {"H1_D_hybrid_vs_B_random_mean": h1, "H2_D_hybrid_vs_C_fixed": h2}}
    if secondary:
        sec = {f"{a}_vs_{b}": paired_binary(v[a], v[b]) for a, b in ANALYSIS_PLAN["secondary"]["pairs_mcnemar"]}
        sec["B_random_mean_vs_A_zero_shot"] = paired_fractional(v[RBAR], v[ZERO])
        out["secondary_unadjusted"] = sec
    return out


def _seed_var(ev, ids):
    v = {s: correct_vec(ev[s], ids) for s in RANDOM}
    accs = [100 * sum(v[s]) / len(ids) for s in RANDOM]
    agree = {}
    for i, a in enumerate(RANDOM):
        for b in RANDOM[i + 1:]:
            agree[f"{a}~{b}"] = round(sum(x == y for x, y in zip(v[a], v[b])) / len(ids), 4)
    rbar = [sum(v[s][k] for s in RANDOM) for k in range(len(ids))]
    return {"per_seed_percent": dict(zip(RANDOM, [round(a, 2) for a in accs])), "mean_percent": round(statistics.mean(accs), 2),
            "sd_percent": round(statistics.stdev(accs), 2), "min_percent": round(min(accs), 2), "max_percent": round(max(accs), 2),
            "pairwise_agreement": agree, "questions_by_seeds_correct": {str(k): rbar.count(k) for k in range(6)}}


def _breakdown(ev, ids, key_of):
    groups = {}
    for i in ids:
        for g in key_of(i):
            groups.setdefault(g, []).append(i)
    out = {}
    for g, gi in sorted(groups.items()):
        v = _vectors(ev, gi)
        out[g] = {"n": len(gi), **{r: round(100 * sum(v[r]) / len(gi), 2) for r in (ZERO, RBAR, FIXED, HYBRID)},
                  "D_minus_Rbar": round((sum(v[HYBRID]) - sum(v[RBAR])) / len(gi), 4)}
    return out
