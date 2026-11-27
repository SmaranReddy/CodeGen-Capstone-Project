"""Three-arm analysis of the clean Week 3 control (no model calls, no Atlas).

Reads outputs/rag_week3_clean/<package>/{zs_rerun,random_k3,hybrid_k3}/executions.jsonl (written
by score_kaggle_outputs.py) and writes report.{json,md} next to them:
  * outcome counts (correct / incorrect / malformed / rejected / timeout / execution failure /
    provider error) and accuracy with Wilson 95% CIs;
  * all three paired comparisons: exact McNemar, paired bootstrap 95% CI, Holm over the family;
    hybrid_k3 vs random_k3 is the question this experiment exists to answer;
  * hardness / gold-SQL-pattern slices (gold SQL used for analysis only), latency, and
    representative items where retrieved examples helped or hurt relative to random ones.
Refuses to report unless every arm has a scored verdict for all package items and the
re-execution check exists.

Usage: python scripts/kaggle_week3/clean_report.py [--package week3_control_v1]
"""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "rag"))

import common as C  # noqa: E402
from common import K  # noqa: E402
import patterns as PT  # noqa: E402
import rag_report as RPT  # noqa: E402

ARMS = ("zs_rerun", "random_k3", "hybrid_k3")
CONTRASTS = (("random_k3", "hybrid_k3"), ("zs_rerun", "hybrid_k3"), ("zs_rerun", "random_k3"))
GROUPS = {"correct": ("correct",), "incorrect": ("wrong_row_count", "wrong_column_count", "wrong_values",
                                                 "order_mismatch"),
          "malformed": ("format_error",), "rejected": ("invalid_mql",), "timeout": ("timeout",),
          "execution_failure": ("execution_error",), "provider_error": ("provider_error",)}


def outcome_groups(recs):
    cats = [r["result"]["category"] for r in recs]
    out = {g: sum(c in cs for c in cats) for g, cs in GROUPS.items()}
    assert sum(out.values()) == len(recs), "unmapped category"
    return out


def examples(recs_a, recs_b, corpus, data, want_a, want_b, n=6):
    """Items with verdict (want_a, want_b) in arms a, b, with the questions of the selected examples."""
    out = []
    for i, rb in recs_b.items():
        ra = recs_a[i]
        if (ra["result"]["correct"], rb["result"]["correct"]) != (want_a, want_b):
            continue
        out.append({"id": i, "db_id": rb["db_id"], "hardness": rb["hardness"], "question": rb["question"],
                    "gold_patterns": sorted(PT.sql_patterns(data[i]["gold_sql"]) - {"select", "where"}),
                    "a_category": ra["result"]["category"], "b_category": rb["result"]["category"],
                    "a_examples": [corpus[d]["question"] for d in ra["retrieval"]["selected_examples"]],
                    "b_examples": [corpus[d]["question"] for d in rb["retrieval"]["selected_examples"]]})
    out.sort(key=lambda e: ({"easy": 0, "medium": 1, "hard": 2, "extra": 3}[e["hardness"]], e["id"]))
    return out[:n], len(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--package", default="week3_control_v1")
    ap.add_argument("--out-root", default=str(C.ROOT / "outputs/rag_week3_clean"))
    a = ap.parse_args()
    root = Path(a.out_root) / a.package
    man = json.loads((C.ROOT / "kaggle" / a.package / "package_manifest.json").read_text(encoding="utf-8"))
    ver = json.loads((root / "verification.json").read_text(encoding="utf-8"))
    chk = json.loads((root / "rescore_check.json").read_text(encoding="utf-8"))
    if not ver["passed"]:
        sys.exit("verification did not pass")
    data = {r["id"]: r for r in K.read_jsonl(C.DATASET)}
    corpus = {d["doc_id"]: d for d in C.load_corpus()}
    recs = {}
    for arm in ARMS:
        rs = K.read_jsonl(root / arm / "executions.jsonl")
        ids = [r["id"] for r in rs]
        if len(ids) != len(set(ids)) or set(ids) != set(man["item_ids"]):
            sys.exit(f"{arm}: {len(set(ids))}/{man['n_items']} items scored; arm incomplete, no report")
        recs[arm] = rs
    rep = {"package": a.package, "n_items": man["n_items"], "backend": ver["backend"],
           "verification": {k: v for k, v in ver["checks"].items()}, "rescore_check": chk["runs"], "arms": {},
           "contrasts": {}}
    for arm in ARMS:
        st = RPT.stats(recs[arm], data)
        st["outcomes"] = outcome_groups(recs[arm])
        # RPT.latency's no-retrieval branch: generation_s from latency_s, plus input_tokens
        lat = RPT.latency([{"latency_s": r["generation_wall_s"], "input_tokens": r["input_tokens"]}
                           for r in recs[arm]])
        n = len(recs[arm])
        st["latency"] = {**lat, "output_tokens_mean": round(sum(r["output_tokens"] or 0 for r in recs[arm]) / n, 1),
                         "context_tokens_mean": round(sum(r["retrieval"]["context_tokens"] for r in recs[arm]) / n, 1)}
        rep["arms"][arm] = st
    verdict = {arm: {r["id"]: bool(r["result"]["correct"]) for r in recs[arm]} for arm in ARMS}
    pvals = {}
    for base, other in CONTRASTS:
        p = RPT.paired(verdict[base], verdict[other])
        p = {"baseline": base, "treatment": other, **{k.replace("rag", "treatment"): v for k, v in p.items()}}
        rep["contrasts"][f"{other}_vs_{base}"] = p
        pvals[f"{other}_vs_{base}"] = p["mcnemar_exact_p"]
    plan = man["analysis_plan"]
    assert set(pvals) == set(plan["confirmatory_family"]), "contrasts differ from the frozen plan"
    for k, v in RPT.holm(pvals).items():
        rep["contrasts"][k]["holm_p"] = v
        rep["contrasts"][k]["supported_at_0.05_after_holm"] = v < 0.05
    rep["analysis_plan"] = plan
    by = {arm: {r["id"]: r for r in recs[arm]} for arm in ARMS}
    helped, n_h = examples(by["random_k3"], by["hybrid_k3"], corpus, data, False, True)
    hurt, n_u = examples(by["random_k3"], by["hybrid_k3"], corpus, data, True, False)
    rep["hybrid_vs_random_items"] = {"hybrid_only_correct": n_h, "random_only_correct": n_u,
                                     "helped_examples": helped, "hurt_examples": hurt}
    rep["top1_similarity"] = {arm: RPT.top1_similarity(recs[arm], {d: c["question"] for d, c in corpus.items() if c["kind"] == "example"})
                              for arm in ("random_k3", "hybrid_k3")}
    (root / "report.json").write_text(json.dumps(rep, indent=1, default=str) + "\n", encoding="utf-8")
    (root / "report.md").write_text(render(rep), encoding="utf-8")
    print(render(rep))


def render(rep):
    L = [f"# Clean Week 3 control — {rep['package']} ({rep['n_items']} train items, one GPU backend)", ""]
    be = rep["backend"]
    L += [f"Backend: llama-cpp-python {be['llama_cpp_python_version']}, n_gpu_layers {be['n_gpu_layers']}, "
          f"GPU {be['gpu'].get('gpus')}, KV reset per item {be['kv_reset_per_item']}.", "",
          "| arm | correct | acc [Wilson 95%] | incorrect | malformed | rejected | timeout | exec failure | provider err |",
          "|---|---:|---|---:|---:|---:|---:|---:|---:|"]
    for arm, st in rep["arms"].items():
        o = st["outcomes"]
        L.append(f"| `{arm}` | {o['correct']}/{st['n']} | {st['acc']} [{st['wilson95'][0]}, {st['wilson95'][1]}] | "
                 f"{o['incorrect']} | {o['malformed']} | {o['rejected']} | {o['timeout']} | "
                 f"{o['execution_failure']} | {o['provider_error']} |")
    L += ["", "| contrast | both | only baseline | only treatment | diff (pts) | bootstrap 95% CI | McNemar p | Holm p |",
          "|---|---:|---:|---:|---:|---|---:|---:|"]
    for k, c in rep["contrasts"].items():
        L.append(f"| {k} | {c['both']} | {c['only_baseline']} | {c['only_treatment']} | {c['diff_points']:+} | "
                 f"[{c['bootstrap_95ci_points'][0]:+}, {c['bootstrap_95ci_points'][1]:+}] | "
                 f"{c['mcnemar_exact_p']} | {c['holm_p']} |")
    L += ["", "Re-execution of saved verdicts: " + ", ".join(
        f"{k} {v['verdicts_reproduced']}/{v['records']}" for k, v in rep["rescore_check"].items()), "",
          "| arm | easy | medium | hard | extra | mean gen s | mean out tokens | mean context tokens |",
          "|---|---|---|---|---|---:|---:|---:|"]
    for arm, st in rep["arms"].items():
        h = st["by_hardness"]
        L.append(f"| `{arm}` | " + " | ".join(f"{h[x]['correct']}/{h[x]['n']}" for x in ("easy", "medium", "hard", "extra"))
                 + f" | {st['latency']['generation_s']['mean']} | {st['latency']['output_tokens_mean']} | "
                 f"{st['latency']['context_tokens_mean']} |")
    hv = rep["hybrid_vs_random_items"]
    plan = rep["analysis_plan"]
    L += ["", f"hybrid-only correct: {hv['hybrid_only_correct']}; random-only correct: {hv['random_only_correct']}", "",
          f"Pre-specified (frozen in the package): primary = {plan['primary']}; confirmatory family = "
          f"{', '.join(plan['confirmatory_family'])} ({plan['tests']}). Exploratory: {'; '.join(plan['exploratory'])}.",
          "", "Caveats: " + " ".join(f"({i + 1}) {c}." for i, c in enumerate(plan["caveats"]))]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
