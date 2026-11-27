"""MQL-0014 registered analysis (pure Python; pinned by hash in the registration before any generation).

Primary (single hypothesis, no multiplicity correction):
  R3V vs R0V on all 716 items, strict execution accuracy on Atlas (anything but "correct" counts as wrong,
  including invalid, timeout, failed and not_comparable). Exact two-sided McNemar, alpha 0.05, plus a paired
  bootstrap 95% CI of the difference (10,000 resamples, seed 20261123).
Descriptive only (no claim): R3 vs R0 (first answers), R0V vs R0 and R3V vs R3 (recovery), the 60% target for R3V
(descriptive yes/no), outcome taxonomy, recovery statistics, accuracy by Spider hardness
by join requirement, by nesting / set operation and by gold-SQL structure label (labels computed from gold SQL AFTER
scoring, for reporting only); reproducibility: the original MQL-0010 D0V rescored in the same Atlas session vs R0V.
This is a SECONDARY re-analysis of the MQL-0010 holdout, not a fresh independent test.
"""
import math
import random
from collections import Counter

ARMS = ("R0", "R0V", "R3", "R3V")
PRIMARY = ("R3V", "R0V")
DESCRIPTIVE = (("R3", "R0"), ("R0V", "R0"), ("R3V", "R3"))
ALPHA = 0.05
TARGET = 0.60
BOOT_REPS = 10000
BOOT_SEED = 20261123
NESTED = {"nested", "anti_join", "intersect", "union", "except"}
TAXONOMY = ("correct", "incorrect", "invalid", "execution_failure", "timeout", "generation_failure", "infrastructure",
            "not_comparable")


def mcnemar_exact(b, c):
    n, k = b + c, min(b, c)
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def compare(correct, ids, a, b, reps=BOOT_REPS, seed=BOOT_SEED):
    va = [correct[a][i] for i in ids]
    vb = [correct[b][i] for i in ids]
    a_only = sum(x and not y for x, y in zip(va, vb))
    b_only = sum(y and not x for x, y in zip(va, vb))
    d = [int(x) - int(y) for x, y in zip(va, vb)]
    n = len(d)
    rng = random.Random(seed)
    ms = sorted(sum(d[rng.randrange(n)] for _ in range(n)) / n for _ in range(reps))
    lo, hi = ms[int(0.025 * reps)], ms[int(0.975 * reps) - 1]
    both = sum(x and y for x, y in zip(va, vb))
    return {"a": a, "b": b, "n": n, "correct_a": sum(va), "correct_b": sum(vb), "a_only": a_only, "b_only": b_only,
            "both_correct": both, "both_wrong": n - both - a_only - b_only,
            "diff_pp": round(100 * sum(d) / n, 3), "ci95_pp": [round(100 * lo, 3), round(100 * hi, 3)],
            "p_mcnemar": mcnemar_exact(a_only, b_only)}


def taxonomy(ev):
    """Evaluator record -> one TAXONOMY label."""
    o = ev["outcome"]
    if o == "failed":
        if ev.get("detail") == "generation_error":
            return "generation_failure"
        return "execution_failure" if ev.get("failure_source") == "model" else "infrastructure"
    return o


def wilson(k, n, z=1.959964):
    if n == 0:
        return [None, None]
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [round(100 * (c - h), 2), round(100 * (c + h), 2)]


def by_group(correct, ids, groups, arms=ARMS):
    """groups: {id: [labels]} -> {label: {n, arm: correct}} (an item counts in every label it carries)."""
    out = {}
    for i in ids:
        for g in groups[i]:
            row = out.setdefault(g, {"n": 0, **{a: 0 for a in arms}})
            row["n"] += 1
            for a in arms:
                row[a] += int(correct[a][i])
    for row in out.values():
        row["R3V_minus_R0V_pp"] = round(100 * (row["R3V"] - row["R0V"]) / row["n"], 2)
    return dict(sorted(out.items(), key=lambda kv: -kv[1]["n"]))


REPRO_ARM = "D0V_mql0010"   # the original MQL-0010 D0V predictions, rescored in the same Atlas session (descriptive)


def analyze(evals, ids, hardness, structure, recs=None, reps=BOOT_REPS, seed=BOOT_SEED):
    """evals: {arm: {id: evaluator record}} for ARMS (+ optional REPRO_ARM); hardness: {id: str};
    structure: {id: [labels]}; recs: {arm: [predictions]}."""
    assert set(ARMS) <= set(evals) <= set(ARMS) | {REPRO_ARM} and all(set(evals[a]) == set(ids) for a in evals)
    n = len(ids)
    correct = {a: {i: evals[a][i]["outcome"] == "correct" for i in ids} for a in ARMS}
    acc = {a: sum(correct[a].values()) for a in ARMS}
    prim = compare(correct, ids, *PRIMARY, reps=reps, seed=seed)
    prim["reject_h0"] = prim["p_mcnemar"] < ALPHA
    prim["direction"] = "R3V better" if prim["diff_pp"] > 0 else "R0V better" if prim["diff_pp"] < 0 else "equal"
    need_target = math.ceil(TARGET * n)
    out = {
        "n": n,
        "accuracy": {a: {"correct": acc[a], "total": n, "pct": round(100 * acc[a] / n, 2), "wilson95": wilson(acc[a], n)}
                     for a in ARMS},
        "primary": prim,
        "descriptive": [compare(correct, ids, a, b, reps=reps, seed=seed) for a, b in DESCRIPTIVE],
        "target_60pct": {"threshold_correct": need_target, "R3V_correct": acc["R3V"], "met": acc["R3V"] >= need_target,
                         "gap_items": need_target - acc["R3V"]},
        "taxonomy": {a: {t: sum(taxonomy(evals[a][i]) == t for i in ids) for t in TAXONOMY} for a in ARMS},
        "by_hardness": by_group(correct, ids, {i: [hardness[i]] for i in ids}),
        "by_structure": by_group(correct, ids, structure),
        "by_join": by_group(correct, ids, {i: ["join" if "join" in structure[i] else "no_join"] for i in ids}),
        "by_nesting": by_group(correct, ids, {i: ["nested_or_setop" if set(structure[i]) & NESTED else "flat"] for i in ids}),
        "by_reference_status": by_group(correct, ids, {i: [evals["R0V"][i].get("reference_status") or "?"] for i in ids}),
    }
    if REPRO_ARM in evals:
        rc = {i: evals[REPRO_ARM][i]["outcome"] == "correct" for i in ids}
        allc = {**correct, REPRO_ARM: rc}
        out["reproducibility"] = {"note": "original MQL-0010 D0V (same plan / prompts as R0V, generated 2026-11-20) vs the "
                                          "regenerated R0V; descriptive only",
                                  "D0V_mql0010_correct": sum(rc.values()),
                                  "compare_R0V_vs_D0V_mql0010": compare(allc, ids, "R0V", REPRO_ARM, reps=reps, seed=seed)}
    if recs is not None:
        out["recovery"] = {}
        for va, src in (("R0V", "R0"), ("R3V", "R3")):
            by = {r["id"]: r for r in recs[va]}
            ret = [i for i in ids if by[i]["recovery"]["retried"]]
            kept = [i for i in ret if by[i]["recovery"]["kept"] == "retry"]
            out["recovery"][va] = {
                "retried": len(ret), "kept": len(kept),
                "fixed_by_retry": sum(correct[va][i] and not correct[src][i] for i in kept),
                "broken_by_retry": sum(not correct[va][i] and correct[src][i] for i in kept)}
        r0 = {r["id"]: r for r in recs["R0"]}
        out["shared_prompt_items"] = sum(r0[i].get("source") == "generated:shared" for i in ids)
        out["budget_drops"] = {a: sum(bool((r.get("retrieval") or {}).get("dropped_budget")) for r in recs[a]) for a in ("R0", "R3")}
    return out


def render(a, title="MQL-0014 results (secondary re-analysis of the MQL-0010 holdout)"):
    L = [f"# {title}", "", "Strict execution accuracy on Atlas (cross-engine vs SQLite gold); everything not `correct` "
         "counts as wrong.", "", "| arm | correct / total | % | Wilson 95% |", "|---|---:|---:|---|"]
    for k, v in a["accuracy"].items():
        L.append(f"| {k} | {v['correct']} / {v['total']} | {v['pct']:.2f} | {v['wilson95']} |")
    p = a["primary"]
    L += ["", "## Primary (registered): R3V vs R0V", "",
          f"{p['diff_pp']:+.2f} pp, paired bootstrap 95% CI {p['ci95_pp']}, exact McNemar p = {p['p_mcnemar']:.4g} "
          f"(R3V-only {p['a_only']}, R0V-only {p['b_only']}, both correct {p['both_correct']}, both wrong "
          f"{p['both_wrong']}); H0 rejected at 0.05: {p['reject_h0']}.", "",
          f"60% target (descriptive): R3V {a['target_60pct']['R3V_correct']} vs threshold "
          f"{a['target_60pct']['threshold_correct']} -> met: {a['target_60pct']['met']}.", "",
          "## Descriptive comparisons", ""]
    for d in a["descriptive"]:
        L.append(f"- {d['a']} vs {d['b']}: {d['diff_pp']:+.2f} pp, CI {d['ci95_pp']}, McNemar p {d['p_mcnemar']:.4g} "
                 f"({d['a_only']}/{d['b_only']})")
    L += ["", "## Outcome taxonomy", "", "| arm | " + " | ".join(TAXONOMY) + " |", "|---|" + "---:|" * len(TAXONOMY)]
    for k, v in a["taxonomy"].items():
        L.append(f"| {k} | " + " | ".join(str(v[t]) for t in TAXONOMY) + " |")
    for key, name in (("by_hardness", "Spider hardness"), ("by_join", "join requirement (gold SQL)"),
                      ("by_nesting", "nesting / set operations (gold SQL)"), ("by_structure", "gold-SQL structure label")):
        L += ["", f"## By {name}", "", "| group | n | R0 | R0V | R3 | R3V | R3V-R0V pp |", "|---|---:|---:|---:|---:|---:|---:|"]
        for g, r in a[key].items():
            L.append(f"| {g} | {r['n']} | {r['R0']} | {r['R0V']} | {r['R3']} | {r['R3V']} | {r['R3V_minus_R0V_pp']:+.2f} |")
    if "reproducibility" in a:
        r = a["reproducibility"]
        c = r["compare_R0V_vs_D0V_mql0010"]
        L += ["", "## Reproducibility (descriptive)", "",
              f"Original MQL-0010 D0V rescored now: {r['D0V_mql0010_correct']} / {c['n']}; regenerated R0V: {c['correct_a']}; "
              f"difference {c['diff_pp']:+.2f} pp, CI {c['ci95_pp']}, McNemar p {c['p_mcnemar']:.4g} ({c['a_only']}/{c['b_only']})."]
    if "recovery" in a:
        L += ["", "## Recovery", ""]
        for k, v in a["recovery"].items():
            L.append(f"- {k}: retried {v['retried']}, kept {v['kept']}, fixed {v['fixed_by_retry']}, broken {v['broken_by_retry']}")
    return "\n".join(L) + "\n"


def structure_labels(sql, PT, S):
    """Gold-SQL structure labels for reporting (patterns.sql_patterns -> structure.signature); 'simple' if none."""
    sig = sorted(S.signature(PT.sql_patterns(sql)))
    return sig or ["simple"]


def counts(xs):
    return dict(Counter(xs).most_common())
