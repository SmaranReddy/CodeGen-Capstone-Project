"""MQL-0015 registered analysis (pure Python; hash-pinned in the package config before any generation).

Registration fb5d07d0... + amendment A1:
  Primary     FT-R3V vs R3V, strict execution accuracy on Atlas over the 1,000 items (anything not "correct" is
              wrong), exact two-sided McNemar at alpha 0.05, paired bootstrap 95% CI (10,000 resamples, seed 20261123).
  Secondary   R3V vs A0V and FT-R3V vs A0V, exact McNemar, Holm over the two at family alpha 0.05; Holm over all three
              comparisons reported as a robustness line.
  Descriptive first answers (A0, R3, FT-R3) and recovery per arm; outcome taxonomy; accuracy by Spider hardness, join,
              nested, set operation, anti-join and primary structure class (gold-derived metadata, reporting only).
  Sensitivity cluster bootstrap by (db_id, gold SQL); excluding degenerate-gold items, the 5 FT-L5 items, the 11 D3
              near-duplicates, the 7 question_in_train items; R3 prompts > 2,048 vs <= 2,048 tokens; population
              post-stratified accuracy over hardness x HAVING.
"""
import math
import random
from collections import Counter

ARMS = ("A0", "A0V", "R3", "R3V", "FT-R3", "FT-R3V")
REGISTERED = ("A0V", "R3V", "FT-R3V")
PRIMARY = ("FT-R3V", "R3V")
SECONDARY = (("R3V", "A0V"), ("FT-R3V", "A0V"))
FIRST_VS_V = (("A0V", "A0"), ("R3V", "R3"), ("FT-R3V", "FT-R3"))
DESCRIPTIVE = (("FT-R3", "R3"), ("R3", "A0"), ("FT-R3", "A0"))
ALPHA = 0.05
BOOT_REPS = 10000
BOOT_SEED = 20261123
TAXONOMY = ("correct", "incorrect", "invalid", "execution_failure", "timeout", "generation_failure", "infrastructure",
            "not_comparable")
LONG_PROMPT_TOKENS = 2048


def mcnemar_exact(b, c):
    n, k = b + c, min(b, c)
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def holm(pvals, alpha=ALPHA):
    """pvals: {name: p} -> {name: {"p": p, "p_holm": adjusted, "reject": bool}} (step-down, monotone)."""
    order = sorted(pvals, key=lambda k: (pvals[k], k))
    m, running, out, stop = len(order), 0.0, {}, False
    for r, k in enumerate(order):
        adj = min(1.0, (m - r) * pvals[k])
        running = max(running, adj)
        rej = (not stop) and pvals[k] <= alpha / (m - r)
        stop = stop or not rej
        out[k] = {"p": pvals[k], "p_holm": running, "reject": rej}
    return out


def _pct(lo, hi):
    return [round(100 * lo, 3), round(100 * hi, 3)]


def compare(correct, ids, a, b, reps=BOOT_REPS, seed=BOOT_SEED, clusters=None):
    va = [bool(correct[a][i]) for i in ids]
    vb = [bool(correct[b][i]) for i in ids]
    a_only = sum(x and not y for x, y in zip(va, vb))
    b_only = sum(y and not x for x, y in zip(va, vb))
    d = [int(x) - int(y) for x, y in zip(va, vb)]
    n = len(d)
    out = {"a": a, "b": b, "n": n, "correct_a": sum(va), "correct_b": sum(vb), "a_only": a_only, "b_only": b_only,
           "diff_pp": round(100 * sum(d) / n, 3) if n else None, "p_mcnemar": mcnemar_exact(a_only, b_only)}
    if n and reps:
        rng = random.Random(seed)
        ms = sorted(sum(rng.choices(d, k=n)) / n for _ in range(reps))
        out["ci95_pp"] = _pct(ms[int(0.025 * reps)], ms[int(0.975 * reps) - 1])
        if clusters is not None:
            groups = {}
            for i, x in zip(ids, d):
                groups.setdefault(clusters[i], []).append(x)
            gl = [(sum(v), len(v)) for _, v in sorted(groups.items())]
            rng = random.Random(seed + 1)
            cs = []
            for _ in range(reps):
                pick = rng.choices(gl, k=len(gl))
                cs.append(sum(s for s, _ in pick) / sum(m for _, m in pick))
            cs.sort()
            out["cluster_ci95_pp"] = _pct(cs[int(0.025 * reps)], cs[int(0.975 * reps) - 1])
            out["n_clusters"] = len(gl)
    return out


def taxonomy(ev):
    o = ev["outcome"]
    if o == "failed":
        if ev.get("detail") == "generation_error":
            return "generation_failure"
        return "execution_failure" if ev.get("failure_source") == "model" else "infrastructure"
    return o if o in TAXONOMY else "not_comparable"


def wilson(k, n, z=1.959964):
    if n == 0:
        return [None, None]
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [round(100 * (c - h), 2), round(100 * (c + h), 2)]


def by_group(correct, ids, groups, arms=ARMS):
    out = {}
    for i in ids:
        for g in groups[i]:
            row = out.setdefault(g, {"n": 0, **{a: 0 for a in arms}})
            row["n"] += 1
            for a in arms:
                row[a] += int(correct[a][i])
    for row in out.values():
        row["FT-R3V_minus_R3V_pp"] = round(100 * (row["FT-R3V"] - row["R3V"]) / row["n"], 2)
        row["R3V_minus_A0V_pp"] = round(100 * (row["R3V"] - row["A0V"]) / row["n"], 2)
    return dict(sorted(out.items(), key=lambda kv: (-kv[1]["n"], kv[0])))


def structure_groups(meta, ids):
    g = {}
    for i in ids:
        s = set(meta[i]["structure_labels"])
        g[i] = ["join" if "join" in s else "no_join",
                "nested" if "nested" in s else "not_nested",
                "set_operation" if "set_operation" in s else "no_set_operation",
                "anti_join" if "anti_join_not_in_or_exists" in s else "no_anti_join",
                "class:" + meta[i]["primary_class"]]
    return g


def subsets(meta, ids, near_dup, prompt_tokens):
    deg = {i for i in ids if {"empty_result", "all_null_result"} & set(meta[i]["flags"])}
    ftl5 = {i for i in ids if any("FT_trained" in p["visible_to"] for p in meta[i].get("l5_corpus_pairs", []))}
    qit = {i for i in ids if "question_in_train" in meta[i]["flags"]}
    long_ = {i for i in ids if prompt_tokens[i] > LONG_PROMPT_TOKENS}
    nd = set(near_dup)
    assert nd <= set(ids), "near-duplicate ids outside the sample"
    return {"degenerate_gold": deg, "ft_l5_partner": ftl5, "near_duplicate_11": nd, "question_in_train": qit,
            "r3_prompt_over_2048": long_}


def post_stratified(correct, ids, meta, pop_cells, arms=REGISTERED):
    """Weights w_c = population share / sample share over cells (hardness, has HAVING)."""
    cell = {i: f"{meta[i]['hardness']}|{'having' if 'having' in meta[i]['structure_labels'] else 'no_having'}" for i in ids}
    samp = Counter(cell.values())
    n = len(ids)
    w = {c: pop_cells.get(c, 0.0) / (samp[c] / n) for c in samp}
    tot = sum(w[cell[i]] for i in ids)
    return {"cells": {c: {"sample_n": samp[c], "sample_share": round(samp[c] / n, 5), "population_share": round(pop_cells.get(c, 0.0), 5),
                          "weight": round(w[c], 5)} for c in sorted(samp)},
            "accuracy_pct": {a: round(100 * sum(w[cell[i]] * int(correct[a][i]) for i in ids) / tot, 3) for a in arms},
            "raw_accuracy_pct": {a: round(100 * sum(int(correct[a][i]) for i in ids) / n, 3) for a in arms}}


def analyze(evals, ids, meta, near_dup, prompt_tokens, clusters, pop_cells, recs=None, reps=BOOT_REPS, seed=BOOT_SEED):
    """evals: {arm: {id: evaluator record}}; meta: sample-manifest item metadata by id; prompt_tokens: {id: R3 tokens};
    clusters: {id: (db, gold SQL) key}; pop_cells: {cell: population share}; recs: {arm: [predictions]}."""
    assert set(evals) == set(ARMS) and all(set(evals[a]) == set(ids) for a in ARMS)
    n = len(ids)
    correct = {a: {i: evals[a][i]["outcome"] == "correct" for i in ids} for a in ARMS}
    acc = {a: sum(correct[a].values()) for a in ARMS}
    prim = compare(correct, ids, *PRIMARY, reps=reps, seed=seed, clusters=clusters)
    prim["alpha"] = ALPHA
    prim["reject_h0"] = prim["p_mcnemar"] < ALPHA
    prim["direction"] = "FT-R3V better" if prim["diff_pp"] > 0 else "R3V better" if prim["diff_pp"] < 0 else "equal"
    sec = [compare(correct, ids, a, b, reps=reps, seed=seed) for a, b in SECONDARY]
    h2 = holm({f"{s['a']} vs {s['b']}": s["p_mcnemar"] for s in sec})
    for s in sec:
        s.update(h2[f"{s['a']} vs {s['b']}"])
    h3 = holm({f"{PRIMARY[0]} vs {PRIMARY[1]}": prim["p_mcnemar"], **{f"{s['a']} vs {s['b']}": s["p_mcnemar"] for s in sec}})
    sub = subsets(meta, ids, near_dup, prompt_tokens)
    sens = {}
    for name in ("degenerate_gold", "ft_l5_partner", "near_duplicate_11", "question_in_train"):
        keep = [i for i in ids if i not in sub[name]]
        sens[f"exclude_{name}"] = {"n_excluded": len(sub[name]), "n": len(keep),
                                   "accuracy": {a: sum(correct[a][i] for i in keep) for a in REGISTERED},
                                   "primary": compare(correct, keep, *PRIMARY, reps=reps, seed=seed)}
    long_ids = [i for i in ids if i in sub["r3_prompt_over_2048"]]
    short_ids = [i for i in ids if i not in sub["r3_prompt_over_2048"]]
    sens["r3_prompt_length"] = {"over_2048": compare(correct, long_ids, *PRIMARY, reps=reps, seed=seed) if long_ids else None,
                                "at_most_2048": compare(correct, short_ids, *PRIMARY, reps=reps, seed=seed)}
    sens["cluster_bootstrap_primary"] = {"ci95_pp": prim.get("cluster_ci95_pp"), "n_clusters": prim.get("n_clusters")}
    sens["post_stratified_hardness_x_having"] = post_stratified(correct, ids, meta, pop_cells)
    infra = {a: sum(taxonomy(evals[a][i]) == "infrastructure" for i in ids) for a in ARMS}
    out = {
        "n": n,
        "accuracy": {a: {"correct": acc[a], "total": n, "pct": round(100 * acc[a] / n, 2), "wilson95": wilson(acc[a], n)}
                     for a in ARMS},
        "primary": prim,
        "secondary": sec,
        "holm_all_three_robustness": h3,
        "descriptive": [compare(correct, ids, a, b, reps=reps, seed=seed) for a, b in DESCRIPTIVE + FIRST_VS_V],
        "taxonomy": {a: {t: sum(taxonomy(evals[a][i]) == t for i in ids) for t in TAXONOMY} for a in ARMS},
        "residual_infrastructure": infra,
        "by_hardness": by_group(correct, ids, {i: [meta[i]["hardness"]] for i in ids}),
        "by_structure": by_group(correct, ids, structure_groups(meta, ids)),
        "sensitivity": sens,
        "subset_sizes": {k: len(v) for k, v in sub.items()},
    }
    if recs is not None:
        out["recovery"] = {}
        for va, src in (("A0V", "A0"), ("R3V", "R3"), ("FT-R3V", "FT-R3")):
            by = {r["id"]: r for r in recs[va]}
            ret = [i for i in ids if by[i]["recovery"]["retried"]]
            kept = [i for i in ret if by[i]["recovery"]["kept"] == "retry"]
            out["recovery"][va] = {
                "retried": len(ret), "kept": len(kept),
                "retry_generation_failed": sum(bool(by[i]["recovery"].get("retry_generation_failed")) for i in ids),
                "first_generation_failed": sum(bool(by[i]["recovery"].get("first_generation_failed")) for i in ids),
                "fixed_by_retry": sum(correct[va][i] and not correct[src][i] for i in kept),
                "broken_by_retry": sum(not correct[va][i] and correct[src][i] for i in kept)}
    return out


def render(a, title="MQL-0015 results"):
    L = [f"# {title}", "", "Strict execution accuracy on Atlas (cross-engine vs SQLite gold); everything not `correct` counts "
         "as wrong. Registered arms: A0V, R3V, FT-R3V; first-answer arms are descriptive.", "",
         "| arm | correct / total | % | Wilson 95% |", "|---|---:|---:|---|"]
    for k, v in a["accuracy"].items():
        L.append(f"| {k} | {v['correct']} / {v['total']} | {v['pct']:.2f} | {v['wilson95']} |")
    p = a["primary"]
    L += ["", "## Primary (registered): FT-R3V vs R3V", "",
          f"{p['diff_pp']:+.2f} pp, paired bootstrap 95% CI {p.get('ci95_pp')}, cluster-bootstrap CI {p.get('cluster_ci95_pp')}, "
          f"exact McNemar p = {p['p_mcnemar']:.4g} (FT-R3V-only {p['a_only']}, R3V-only {p['b_only']}); "
          f"H0 rejected at 0.05: {p['reject_h0']}.", "", "## Secondary (Holm over the two)", ""]
    for s in a["secondary"]:
        L.append(f"- {s['a']} vs {s['b']}: {s['diff_pp']:+.2f} pp, CI {s.get('ci95_pp')}, p {s['p_mcnemar']:.4g}, "
                 f"Holm p {s['p_holm']:.4g}, reject {s['reject']} ({s['a_only']}/{s['b_only']})")
    L += ["", "Holm over all three (robustness): " + "; ".join(f"{k}: p_holm {v['p_holm']:.4g} reject {v['reject']}"
                                                             for k, v in a["holm_all_three_robustness"].items()),
          "", "## Descriptive comparisons", ""]
    for d in a["descriptive"]:
        L.append(f"- {d['a']} vs {d['b']}: {d['diff_pp']:+.2f} pp, CI {d.get('ci95_pp')}, McNemar p {d['p_mcnemar']:.4g} "
                 f"({d['a_only']}/{d['b_only']})")
    L += ["", "## Outcome taxonomy", "", "| arm | " + " | ".join(TAXONOMY) + " |", "|---|" + "---:|" * len(TAXONOMY)]
    for k, v in a["taxonomy"].items():
        L.append(f"| {k} | " + " | ".join(str(v[t]) for t in TAXONOMY) + " |")
    for key, name in (("by_hardness", "Spider hardness"), ("by_structure", "structure (gold-SQL labels)")):
        L += ["", f"## By {name}", "", "| group | n | A0V | R3V | FT-R3V | FT-R3V−R3V pp | R3V−A0V pp |",
              "|---|---:|---:|---:|---:|---:|---:|"]
        for g, r in a[key].items():
            L.append(f"| {g} | {r['n']} | {r['A0V']} | {r['R3V']} | {r['FT-R3V']} | {r['FT-R3V_minus_R3V_pp']:+.2f} | "
                     f"{r['R3V_minus_A0V_pp']:+.2f} |")
    L += ["", "## Sensitivity (descriptive)", ""]
    s = a["sensitivity"]
    for k in ("exclude_degenerate_gold", "exclude_ft_l5_partner", "exclude_near_duplicate_11", "exclude_question_in_train"):
        v = s[k]
        L.append(f"- {k} (−{v['n_excluded']}, n {v['n']}): A0V {v['accuracy']['A0V']}, R3V {v['accuracy']['R3V']}, "
                 f"FT-R3V {v['accuracy']['FT-R3V']}; primary {v['primary']['diff_pp']:+.2f} pp, p {v['primary']['p_mcnemar']:.4g}")
    for k, v in s["r3_prompt_length"].items():
        if v:
            L.append(f"- R3 prompt {k} tokens (n {v['n']}): FT-R3V−R3V {v['diff_pp']:+.2f} pp, p {v['p_mcnemar']:.4g}")
    L.append(f"- cluster bootstrap (primary): CI {s['cluster_bootstrap_primary']['ci95_pp']} over "
             f"{s['cluster_bootstrap_primary']['n_clusters']} clusters")
    ps = s["post_stratified_hardness_x_having"]
    L.append(f"- post-stratified accuracy %: {ps['accuracy_pct']} (raw {ps['raw_accuracy_pct']})")
    if "recovery" in a:
        L += ["", "## Recovery", ""]
        for k, v in a["recovery"].items():
            L.append(f"- {k}: retried {v['retried']}, kept {v['kept']}, fixed {v['fixed_by_retry']}, broken {v['broken_by_retry']}, "
                     f"retry generation failed {v['retry_generation_failed']}, first generation failed {v['first_generation_failed']}")
    return "\n".join(L) + "\n"
