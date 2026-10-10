"""Compare model runs on the validated Spider 1.0 NL -> MQL dataset.

Restricts every run to the ids in data/processed/spider1_nl_mql.jsonl (runs may have been
executed on the full frozen sample). Supports partially evaluated runs:
  * "own coverage": each model scored on every dataset id it actually answered
    (provider errors and never-evaluated ids are excluded, never counted as wrong);
  * "paired": all models scored on the exact intersection of answered ids.
Writes outputs/evaluation_spider1/comparison.json + comparison.md with:
  execution accuracy (overall / hardness / flagged-excluded), 95% Wilson CIs,
  paired McNemar exact test, error categories, latency, tokens and cost.

Usage: python scripts/spider1/compare.py <label_a> <label_b> [<label_c> ...]
"""
import json
import math
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import spider1_kit as K  # noqa: E402

OUT = K.ROOT / "outputs/evaluation_spider1"
HARD = ("easy", "medium", "hard", "extra")


def wilson(k, n, z=1.96):
    if not n:
        return (None, None)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(c - h, 4), round(c + h, 4))


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def load_run(label, ids):
    recs = {r["id"]: r for r in K.read_jsonl(OUT / label / "executions.jsonl")
            if r["result"]["category"] != "provider_error"}
    missing = [i for i in ids if i not in recs]
    summ = json.loads((OUT / label / "summary.json").read_text(encoding="utf-8"))
    return {i: recs[i] for i in ids if i in recs}, missing, summ


def paired_ids(all_ids, answered_by_label):
    """Dataset ids (in dataset order) answered by every run."""
    return [i for i in all_ids if all(i in recs for recs in answered_by_label.values())]


def own_coverage(all_ids, recs):
    """Accuracy on the ids a run answered; unanswered ids are reported, not scored."""
    answered = [i for i in all_ids if i in recs]
    c = sum(recs[i]["result"]["correct"] for i in answered)
    n = len(answered)
    return {"n_answered": n, "n_not_evaluated": len(all_ids) - n, "correct": c,
            "accuracy": round(c / n, 4) if n else None, "ci95": wilson(c, n)}


def stats(recs, data, price):
    n = len(recs)
    c = sum(r["result"]["correct"] for r in recs.values())
    out = {"n": n, "correct": c, "accuracy": round(c / n, 4) if n else None, "ci95": wilson(c, n)}
    out["by_hardness"] = {}
    for h in HARD:
        hs = [r for i, r in recs.items() if data[i]["hardness"] == h]
        hc = sum(r["result"]["correct"] for r in hs)
        out["by_hardness"][h] = {"n": len(hs), "correct": hc,
                                 "accuracy": round(hc / len(hs), 4) if hs else None}
    unflagged = [r for i, r in recs.items() if not data[i]["flags"]]
    uc = sum(r["result"]["correct"] for r in unflagged)
    out["unflagged_only"] = {"n": len(unflagged), "correct": uc,
                             "accuracy": round(uc / len(unflagged), 4) if unflagged else None}
    devonly = [r for i, r in recs.items() if data[i]["evidence"] == "dev+testsuite"]
    dc = sum(r["result"]["correct"] for r in devonly)
    out["testsuite_backed_only"] = {"n": len(devonly), "correct": dc,
                                    "accuracy": round(dc / len(devonly), 4) if devonly else None}
    out["categories"] = dict(Counter(r["result"]["category"] for r in recs.values()).most_common())
    lat = sorted(r["latency_s"] for r in recs.values())
    out["latency_s"] = {"mean": round(sum(lat) / n, 2), "median": lat[n // 2],
                        "p90": lat[min(n - 1, int(0.9 * n))], "total": round(sum(lat), 1)}
    itok = sum(r.get("input_tokens") or 0 for r in recs.values())
    otok = sum(r.get("output_tokens") or 0 for r in recs.values())
    out["tokens"] = {"input": itok, "output": otok}
    if price:
        out["cost_usd"] = round(itok / 1000 * (price.get("input_per_1k") or 0)
                                + otok / 1000 * (price.get("output_per_1k") or 0), 4)
    return out


def main():
    import yaml
    labels = sys.argv[1:]
    cfg = yaml.safe_load((K.ROOT / "configs/evaluation_spider1.yaml").read_text(encoding="utf-8"))
    data = {d["id"]: d for d in K.read_jsonl(K.ROOT / cfg["dataset"])}
    all_ids = list(data)
    loaded = {}
    for lab in labels:
        loaded[lab] = load_run(lab, all_ids)
    ids = paired_ids(all_ids, {lab: loaded[lab][0] for lab in labels})
    complete = len(ids) == len(all_ids)
    runs = {}
    report = {"dataset": cfg["dataset"], "n_dataset": len(all_ids), "n_compared": len(ids),
              "complete": complete,
              "coverage": {lab: len(all_ids) - len(loaded[lab][1]) for lab in labels},
              "own_coverage": {lab: own_coverage(all_ids, loaded[lab][0]) for lab in labels},
              "runs": {}, "paired": {}}
    for lab in labels:
        recs, missing, summ = loaded[lab]
        recs = {i: recs[i] for i in ids}
        mid = summ.get("model_id") or ""
        prov = summ.get("provider") or ("frontier" if "gemini" in mid else
                                        "frontier_claude" if "claude" in mid else
                                        "frontier_groq" if "gpt-oss" in mid else
                                        "qwen7b" if "7B" in mid else "qwen")
        runs[lab] = recs
        report["runs"][lab] = {"model_id": mid, "provider": prov, "config": summ.get("config"),
                               "prompt_sha256": summ.get("prompt_sha256"),
                               **stats(recs, data, cfg["pricing"].get(prov))}
    shas = {r["prompt_sha256"] for r in report["runs"].values()}
    report["same_prompt_template"] = len(shas) == 1
    for i, a in enumerate(labels):
        for b in labels[i + 1:]:
            ab = sum(runs[a][x]["result"]["correct"] and not runs[b][x]["result"]["correct"] for x in ids)
            ba = sum(runs[b][x]["result"]["correct"] and not runs[a][x]["result"]["correct"] for x in ids)
            both = sum(runs[a][x]["result"]["correct"] and runs[b][x]["result"]["correct"] for x in ids)
            report["paired"][f"{a} vs {b}"] = {"both_correct": both, f"only_{a}": ab, f"only_{b}": ba,
                                               "neither": len(ids) - both - ab - ba,
                                               "mcnemar_exact_p": mcnemar_exact(ab, ba)}
    (OUT / "comparison.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    cov = ", ".join(f"{k} {v}/{len(all_ids)}" for k, v in report["coverage"].items())
    md = [f"# Spider 1.0 NL→MQL comparison ({len(all_ids)} validated examples"
          f"{'' if complete else f'; paired subset {len(ids)} - PARTIAL coverage'})\n",
          f"Coverage: {cov}\n",
          f"Dataset: `{cfg['dataset']}`; same prompt template for all runs: {report['same_prompt_template']}\n",
          "## Each model on the examples it answered\n",
          "Not-evaluated examples are excluded (neither correct nor wrong). Rows with different "
          "coverage are not directly comparable; use the paired table below.\n",
          "| run | answered | not evaluated | EX accuracy | 95% CI |",
          "|---|---|---|---|---|"]
    for lab, o in report["own_coverage"].items():
        md.append(f"| {lab} | {o['n_answered']} | {o['n_not_evaluated']} | {o['correct']}/{o['n_answered']} = "
                  f"{o['accuracy']:.1%} | {o['ci95'][0]:.1%}–{o['ci95'][1]:.1%} |")
    md += [f"\n## Paired: every model on the same {len(ids)} example ids\n",
           "| run | model | EX accuracy | 95% CI | easy | medium | hard | extra | median latency (s) | cost (USD) |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for lab, r in report["runs"].items():
        bh = r["by_hardness"]
        md.append(f"| {lab} | {r['model_id']} | {r['correct']}/{r['n']} = {r['accuracy']:.1%} | "
                  f"{r['ci95'][0]:.1%}–{r['ci95'][1]:.1%} | "
                  + " | ".join(f"{bh[h]['correct']}/{bh[h]['n']}" for h in HARD)
                  + f" | {r['latency_s']['median']} | {r.get('cost_usd', 0)} |")
    md.append("\n## Error categories\n")
    for lab, r in report["runs"].items():
        md.append(f"- **{lab}**: " + ", ".join(f"{k} {v}" for k, v in r["categories"].items()))
    md.append("\n## Robustness slices\n")
    for lab, r in report["runs"].items():
        u, t = r["unflagged_only"], r["testsuite_backed_only"]
        md.append(f"- **{lab}**: unflagged only {u['correct']}/{u['n']} = {u['accuracy']:.1%}; "
                  f"test-suite-backed only {t['correct']}/{t['n']} = {t['accuracy']:.1%}")
    md.append("\n## Paired comparison\n")
    for k, v in report["paired"].items():
        md.append(f"- **{k}**: " + ", ".join(f"{a}={b}" for a, b in v.items()))
    (OUT / "comparison.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
