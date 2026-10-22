"""Week 3 pilot analysis (no model calls) -> outputs/rag_week3/report.{json,md}.

Compares every RAG experiment with the zero-shot baseline on the SAME validation items:
  * baseline A: `zs_rerun` (k = 0 through the RAG runner, same session/hardware) -- primary;
  * baseline B: the stored zero-shot Qwen-3B records of the expanded benchmark.
Reports execution accuracy with Wilson 95% CIs, outcome categories, hardness and gold-SQL
pattern slices (gold SQL used for ANALYSIS only), exact McNemar (two-sided binomial test on the
discordant pairs), a paired bootstrap 95% CI for the accuracy difference, latency and context
size, cross-database name copying, and per-item helped/hurt lists.

Usage: python scripts/rag/rag_report.py [--partition validation|train]
  validation -> report.{json,md}; train -> report_train.{json,md} (runs in <experiment>__train/)
"""
import argparse
import json
import math
from collections import Counter

import numpy as np
from scipy.stats import binomtest

import common as C
from common import K
import patterns as PT

OUT = C.ROOT / "outputs/rag_week3"
STORED = C.BASELINE_DIR / "qwen2.5-coder-3b/executions.jsonl"
ORDER = ["zs_rerun", "random_k3", "hybrid_k3", "hybrid_k3_g3", "hybrid_k3_fk"]
CATS = ["correct", "format_error", "invalid_mql", "execution_error", "timeout",
        "wrong_row_count", "wrong_column_count", "wrong_values", "order_mismatch", "provider_error"]
CAT_LABEL = {"format_error": "malformed", "invalid_mql": "validation rejected"}
BOOT = 10000
# planned contrasts vs zero-shot (the Holm family). hybrid_k3_fk was designed after inspecting
# validation failures, so it is exploratory and excluded from the family.
CONFIRMATORY = ("random_k3", "hybrid_k3", "hybrid_k3_g3")


def wilson(c, n, z=1.959964):
    if not n:
        return (None, None)
    p = c / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(100 * (centre - half), 1), round(100 * (centre + half), 1))


def paired(a, b, seed=20261020):
    """a, b: {id: bool}. -> counts, McNemar exact p, bootstrap CI of acc(b) - acc(a) in points."""
    ids = sorted(set(a) & set(b))
    x = np.array([a[i] for i in ids], dtype=int)
    y = np.array([b[i] for i in ids], dtype=int)
    only_a, only_b = int(((x == 1) & (y == 0)).sum()), int(((x == 0) & (y == 1)).sum())
    p = binomtest(only_b, only_a + only_b, 0.5).pvalue if only_a + only_b else 1.0
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(ids), size=(BOOT, len(ids)))
    diffs = (y[idx].mean(1) - x[idx].mean(1)) * 100
    return {"n": len(ids), "both": int(((x == 1) & (y == 1)).sum()), "only_baseline": only_a,
            "only_rag": only_b, "neither": int(((x == 0) & (y == 0)).sum()),
            "diff_points": round((y.mean() - x.mean()) * 100, 1),
            "bootstrap_95ci_points": [round(float(np.percentile(diffs, 2.5)), 1),
                                      round(float(np.percentile(diffs, 97.5)), 1)],
            "mcnemar_exact_p": round(float(p), 4)}


def stats(recs, data):
    n = len(recs)
    c = sum(r["result"]["correct"] for r in recs)
    cats = Counter(r["result"]["category"] for r in recs)
    by_h = {}
    for h in ("easy", "medium", "hard", "extra"):
        hs = [r for r in recs if r["hardness"] == h]
        if hs:
            k = sum(r["result"]["correct"] for r in hs)
            by_h[h] = {"n": len(hs), "correct": k, "acc": round(100 * k / len(hs), 1)}
    by_p = {}
    for p in PT.PATTERNS:
        if p in ("select", "where", "text_number"):
            continue
        ps = [r for r in recs if p in PT.sql_patterns(data[r["id"]]["gold_sql"])]
        if len(ps) >= 5:
            k = sum(r["result"]["correct"] for r in ps)
            by_p[p] = {"n": len(ps), "correct": k, "acc": round(100 * k / len(ps), 1)}
    return {"n": n, "correct": c, "acc": round(100 * c / n, 1) if n else None, "wilson95": wilson(c, n),
            "categories": {k: cats.get(k, 0) for k in CATS if cats.get(k, 0) or k in CATS[:5]},
            "by_hardness": by_h, "by_pattern": by_p}


def latency(recs):
    def q(vals):
        v = sorted(x for x in vals if x is not None)
        if not v:
            return None
        return {"mean": round(sum(v) / len(v), 3), "median": v[len(v) // 2], "p90": v[min(len(v) - 1, int(len(v) * .9))]}
    if "retrieval" not in recs[0]:
        return {"generation_s": q([r["latency_s"] for r in recs]), "input_tokens": q([r["input_tokens"] for r in recs])}
    return {"retrieval_ms": q([1000 * r["retrieval"]["retrieval_latency_s"] for r in recs]),
            "context_build_ms": q([1000 * r["retrieval"]["context_build_latency_s"] for r in recs]),
            "generation_s": q([r["generation_wall_s"] for r in recs]),
            "scoring_s": q([r["scoring_latency_s"] for r in recs]),
            "total_s": q([r["total_latency_s"] for r in recs]),
            "context_tokens": q([r["retrieval"]["context_tokens"] for r in recs]),
            "input_tokens": q([r["input_tokens"] for r in recs]),
            "output_tokens": q([r["output_tokens"] for r in recs])}


def cross_db_copy(recs):
    """Outputs that name a collection of a different database (always rejected by validation)."""
    hits = []
    for r in recs:
        why = r["result"].get("reason") or ""
        if r["result"]["category"] == "invalid_mql" and "__" in why and r["db_id"] + "__" not in why:
            hits.append(r["id"])
    return hits


def holm(pvals):
    """Holm-Bonferroni adjusted p-values for {name: p}."""
    items = sorted(pvals.items(), key=lambda t: t[1])
    m, out, run = len(items), {}, 0.0
    for i, (n, p) in enumerate(items):
        run = max(run, min(1.0, (m - i) * p))
        out[n] = round(run, 4)
    return out


def top1_similarity(recs, corpus_q):
    """Raw-question token Jaccard between each query and its top-1 selected example."""
    vals = []
    for r in recs:
        sel = r["retrieval"]["selected_examples"]
        if sel:
            a, b = set(PT.tokenize(r["question"])), set(PT.tokenize(corpus_q[sel[0]]))
            vals.append(len(a & b) / len(a | b) if a | b else 0.0)
    v = sorted(vals)
    return {"n": len(v), "mean": round(sum(v) / len(v), 3), "median": round(v[len(v) // 2], 3),
            "share_ge_0.5": round(sum(x >= 0.5 for x in v) / len(v), 3)} if v else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--partition", default="validation", choices=["validation", "train"])
    partition = ap.parse_args().partition
    suffix = "" if partition == "validation" else f"__{partition}"
    data = {r["id"]: r for r in K.read_jsonl(C.DATASET)}
    db_part, _ = C.load_split()
    corpus_q = {d["doc_id"]: d["question"] for d in C.load_corpus() if d["kind"] == "example"}
    val_ids = [i for i, r in data.items() if db_part[r["db_id"]] == partition]
    runs = {}
    for name in ORDER:
        p = OUT / (name + suffix) / "executions.jsonl"
        if p.exists():
            runs[name] = {r["id"]: r for r in K.read_jsonl(p)}
    stored = {r["id"]: r for r in K.read_jsonl(STORED) if r["id"] in set(val_ids)}
    rep = {"partition": partition, "n_items": len(val_ids), "model": "Qwen2.5-Coder-3B-Instruct GGUF q4_k_m",
           "stored_zero_shot": {**stats(list(stored.values()), data), "latency": latency(list(stored.values()))},
           "experiments": {}, "paired_vs_zs_rerun": {}, "paired_vs_stored_zero_shot": {}}
    complete = {n: r for n, r in runs.items() if set(r) >= set(val_ids)}
    rep["incomplete_experiments"] = sorted(set(runs) - set(complete))
    for name, recs in complete.items():
        rl = [recs[i] for i in val_ids]
        rep["experiments"][name] = {**stats(rl, data), "latency": latency(rl), "config": rl[0]["config"],
                                    "cross_db_name_copies": cross_db_copy(rl),
                                    "empty_context": sum(r["retrieval"]["empty_context"] for r in rl),
                                    "budget_drops": sum(bool(r["retrieval"]["dropped_budget"]) for r in rl),
                                    "top1_question_similarity": top1_similarity(rl, corpus_q)}
        corr = {i: recs[i]["result"]["correct"] for i in val_ids}
        rep["paired_vs_stored_zero_shot"][name] = paired({i: stored[i]["result"]["correct"] for i in val_ids}, corr)
        if "zs_rerun" in complete and name != "zs_rerun":
            zs = {i: complete["zs_rerun"][i]["result"]["correct"] for i in val_ids}
            rep["paired_vs_zs_rerun"][name] = paired(zs, corr)
    rep["paired_vs_random_k3"] = {}
    if "random_k3" in complete:
        rnd = {i: complete["random_k3"][i]["result"]["correct"] for i in val_ids}
        for name in complete:
            if name not in ("zs_rerun", "random_k3"):
                rep["paired_vs_random_k3"][name] = paired(rnd, {i: complete[name][i]["result"]["correct"] for i in val_ids})
    if "zs_rerun" in complete:
        zs = complete["zs_rerun"]
        rep["determinism_zs_rerun_vs_stored"] = {
            "raw_text_identical": sum(zs[i]["raw_text"] == stored[i]["raw_text"] for i in val_ids),
            "verdict_identical": sum(zs[i]["result"]["category"] == stored[i]["result"]["category"] for i in val_ids),
            "differing_ids": [i for i in val_ids if zs[i]["raw_text"] != stored[i]["raw_text"]]}
        examples = {}
        for name in complete:
            if name == "zs_rerun":
                continue
            helped = [i for i in val_ids if complete[name][i]["result"]["correct"] and not zs[i]["result"]["correct"]]
            hurt = [i for i in val_ids if zs[i]["result"]["correct"] and not complete[name][i]["result"]["correct"]]

            def item(i, name=name):
                r = complete[name][i]
                return {"id": i, "db_id": r["db_id"], "hardness": r["hardness"], "question": r["question"],
                        "zero_shot": zs[i]["result"]["category"], "rag": r["result"]["category"],
                        "retrieved": r["retrieval"]["selected_examples"], "guides": r["retrieval"]["selected_guides"],
                        "zero_shot_mql": zs[i].get("mql"), "rag_mql": r.get("mql"),
                        "rag_reason": r["result"].get("reason")}
            examples[name] = {"helped": [item(i) for i in helped], "hurt": [item(i) for i in hurt]}
        rep["helped_hurt_vs_zs_rerun"] = examples
    base_key = "paired_vs_zs_rerun" if rep["paired_vs_zs_rerun"] else "paired_vs_stored_zero_shot"
    contrasts = {n: p["mcnemar_exact_p"] for n, p in rep[base_key].items() if n in CONFIRMATORY}
    rep["holm_adjusted_mcnemar_vs_zero_shot"] = {"baseline": base_key, "p_adj": holm(contrasts)}
    if partition == "train":
        sub = [i for i in val_ids if data[i]["db_id"] not in ("concert_singer", "singer")]
        base = complete.get("zs_rerun") or stored
        rep["without_concert_singer_and_singer"] = {
            n: paired({i: base[i]["result"]["correct"] for i in sub},
                      {i: complete[n][i]["result"]["correct"] for i in sub})
            for n in complete if n != "zs_rerun"}
    stem = "report" if partition == "validation" else f"report_{partition}"
    (OUT / f"{stem}.json").write_text(json.dumps(rep, indent=1, default=str) + "\n", encoding="utf-8")
    (OUT / f"{stem}.md").write_text(render_md(rep), encoding="utf-8")
    print(render_md(rep))


def _cell(slices, key):
    return f"{slices[key]['acc']} ({slices[key]['n']})" if key in slices else "-"


def render_md(rep):
    L = [f"# Week 3 RAG pilot - {rep['partition']} ({rep['n_items']} items, {rep['model']})", ""]
    L += ["| run | correct | acc % [Wilson 95%] | " + " | ".join(CAT_LABEL.get(c, c) for c in CATS[1:9]) + " |",
          "|---|---:|---|" + "---:|" * 8]
    rows = [("stored zero-shot", rep["stored_zero_shot"])] + list(rep["experiments"].items())
    for name, s in rows:
        L.append(f"| {name} | {s['correct']}/{s['n']} | {s['acc']} [{s['wilson95'][0]}, {s['wilson95'][1]}] | "
                 + " | ".join(str(s["categories"].get(c, 0)) for c in CATS[1:9]) + " |")
    for title, key in (("Paired vs zs_rerun", "paired_vs_zs_rerun"), ("Paired vs stored zero-shot", "paired_vs_stored_zero_shot"),
                       ("Paired vs random_k3 (baseline = random control)", "paired_vs_random_k3")):
        if rep[key]:
            L += ["", f"## {title}", "", "| run | both | only baseline | only RAG | neither | diff (pts) | bootstrap 95% CI | McNemar p |",
                  "|---|---:|---:|---:|---:|---:|---|---:|"]
            for n, p in rep[key].items():
                L.append(f"| {n} | {p['both']} | {p['only_baseline']} | {p['only_rag']} | {p['neither']} | "
                         f"{p['diff_points']:+} | [{p['bootstrap_95ci_points'][0]}, {p['bootstrap_95ci_points'][1]}] | {p['mcnemar_exact_p']} |")
    if rep["experiments"]:
        L += ["", "## By hardness (acc %, n)", "", "| run | " + " | ".join(("easy", "medium", "hard", "extra")) + " |", "|---|---|---|---|---|"]
        for name, s in rows:
            L.append(f"| {name} | " + " | ".join(_cell(s["by_hardness"], h) for h in ("easy", "medium", "hard", "extra")) + " |")
        pats = list(rep["stored_zero_shot"]["by_pattern"])
        L += ["", "## By gold-SQL pattern (acc %, n; patterns overlap)", "", "| run | " + " | ".join(pats) + " |",
              "|---|" + "---|" * len(pats)]
        for name, s in rows:
            L.append(f"| {name} | " + " | ".join(_cell(s["by_pattern"], p) for p in pats) + " |")
        L += ["", "## Latency and context (mean / median)", "", "| run | retrieval ms | generation s | total s | context tokens | input tokens |",
              "|---|---|---|---|---|---|"]
        for name, s in rep["experiments"].items():
            la = s["latency"]
            L.append(f"| {name} | {la['retrieval_ms']['mean']:.2f} / {la['retrieval_ms']['median']:.2f} | "
                     f"{la['generation_s']['mean']} / {la['generation_s']['median']} | {la['total_s']['mean']} / {la['total_s']['median']} | "
                     f"{la['context_tokens']['mean']} / {la['context_tokens']['median']} | {la['input_tokens']['mean']} / {la['input_tokens']['median']} |")
    h = rep.get("holm_adjusted_mcnemar_vs_zero_shot")
    if h and h["p_adj"]:
        L += ["", f"Holm-adjusted McNemar p, planned contrasts only ({h['baseline']}; hybrid_k3_fk exploratory, excluded): " + ", ".join(f"{k} {v}" for k, v in h["p_adj"].items())]
    if rep["experiments"]:
        L += ["", "Top-1 retrieved example vs query, raw-question token Jaccard: " + "; ".join(
            f"{n} mean {s['top1_question_similarity']['mean']} (>=0.5: {s['top1_question_similarity']['share_ge_0.5']})"
            for n, s in rep["experiments"].items() if s.get("top1_question_similarity"))]
    if rep.get("without_concert_singer_and_singer"):
        L += ["", "Without concert_singer and singer: " + "; ".join(
            f"{n} {p['diff_points']:+} pts [{p['bootstrap_95ci_points'][0]}, {p['bootstrap_95ci_points'][1]}] (n={p['n']}, p={p['mcnemar_exact_p']})"
            for n, p in rep["without_concert_singer_and_singer"].items())]
    if "determinism_zs_rerun_vs_stored" in rep:
        d = rep["determinism_zs_rerun_vs_stored"]
        L += ["", f"zs_rerun vs stored zero-shot: raw output identical {d['raw_text_identical']}/{rep['n_items']}, "
                  f"verdict identical {d['verdict_identical']}/{rep['n_items']}."]
    if rep["incomplete_experiments"]:
        L += ["", f"Incomplete (not reported): {rep['incomplete_experiments']}"]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
