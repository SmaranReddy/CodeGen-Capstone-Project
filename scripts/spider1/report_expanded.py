"""Report for the expanded Spider 1.0 NL -> MQL benchmark (no model calls, no Atlas calls).

Reads outputs/evaluation_spider1_expanded/<label>/executions.jsonl for every run and writes
report.json / report.md next to them:
  * completeness gate: every dataset id has exactly one record that is not a provider error
    (anything else is listed; the report refuses to call a run complete otherwise)
  * execution accuracy (strict: unanswered = wrong) with Wilson 95% CIs, overall, per
    hardness, on the v1 279 subset (must equal the v1 numbers), on the 176 new items, and
    without the items flagged `ambiguous`
  * output validity kept SEPARATE from accuracy: valid JSON / passed read-only validation /
    executed without error
  * outcome categories: malformed output (truncated vs other), validation rejection,
    execution error, timeout, result mismatch (row count / column count / values / order)
  * paired comparison on the identical item set: exact McNemar test per model pair
  * sensitivity (never replaces the primary metric): (a) numeric-text equivalence, where a
    number stored as TEXT in SQLite ('9') counts as equal to the number 9 in the MQL result;
    (b) accuracy without POST_HOC items, whose reference answer was questioned only after
    reviewing model failures (the dataset itself stays frozen)

Usage: python scripts/spider1/report_expanded.py
"""
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import spider1_kit as K  # noqa: E402

CFG = K.ROOT / "configs/evaluation_spider1_expanded.yaml"
LABELS = ["qwen2.5-coder-3b", "qwen2.5-coder-7b", "gpt-oss-120b"]
V1_REPORTED = {"qwen2.5-coder-3b": 132, "qwen2.5-coder-7b": 181, "gpt-oss-120b": 232}
HARD = ("easy", "medium", "hard", "extra")
MISMATCH = ("wrong_row_count", "wrong_column_count", "wrong_values", "order_mismatch")
NUMERIC = re.compile(r"[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?")
# found while reviewing gpt-oss-120b failures on the 176 new items (not a systematic re-audit)
POST_HOC = {
    "spider1_dev_0035": "'number of concerts for each singer': inner join omits singers with 0 concerts",
    "spider1_dev_0367": "'all document ids and the number of paragraphs': inner join omits documents with 0 paragraphs",
    "spider1_dev_0369": "same as 0367 (with document names)",
    "spider1_dev_0884": "'high school students and their number of friends': inner join omits students with 0 friends",
    "spider1_dev_0419": "'visited some museums more than once' can mean the same museum twice; SQL counts any 2 visits",
    "spider1_dev_0537": "'most number of students': SQL counts enrolments (Bachelor 8 vs Master 7); by distinct students it is a 5-5 tie",
    "spider1_dev_0778": "SQL draws countries only from countrylanguage; countries with no language rows are omitted",
}


def wilson(k, n, z=1.959964):
    if not n:
        return None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(c - h, 4), round(c + h, 4)]


def acc(recs):
    n, k = len(recs), sum(r["result"]["correct"] for r in recs)
    return {"n": n, "correct": k, "accuracy": round(k / n, 4) if n else None, "wilson95": wilson(k, n)}


def mcnemar_exact(b, c, digits=6):
    """Two-sided exact McNemar p-value (binomial on discordant pairs)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)
    return round(p, digits) if digits is not None else float(f"{p:.3g}")


def numeric_text(v):
    return float(v) if isinstance(v, str) and NUMERIC.fullmatch(v.strip()) else v


def lenient_correct(rec, ex):
    """Correct, or an executed mismatch that matches once numeric TEXT is read as a number."""
    if rec["result"]["correct"]:
        return True
    if rec["result"]["category"] not in MISMATCH or rec.get("pred_rows") is None or rec.get("pred_rows_truncated"):
        return False
    pred = [[numeric_text(v) for v in r] for r in rec["pred_rows"]]
    gold = [[numeric_text(v) for v in r] for r in ex["gold_rows"]]
    return K.exec_match(pred, gold, ex.get("tie_blocks"), ex["order_by"])[0]


def outcome(rec, max_new_tokens):
    cat = rec["result"]["category"]
    if cat == "format_error":
        trunc = rec.get("stop_reason") in ("length", "max_tokens") or (rec.get("output_tokens") or 0) >= max_new_tokens
        return "malformed_output_truncated" if trunc else "malformed_output"
    return {"invalid_mql": "validation_rejected", "execution_error": "execution_error",
            "timeout": "timeout", "correct": "correct"}.get(cat, "result_mismatch" if cat in MISMATCH else cat)


def main():
    cfg = yaml.safe_load(CFG.read_text(encoding="utf-8"))
    data = K.read_jsonl(K.ROOT / cfg["dataset"])
    ids = [d["id"] for d in data]
    meta = {d["id"]: d for d in data}
    out_dir = K.ROOT / cfg["output_dir"]
    max_new = cfg["generation"]["max_new_tokens"]
    report = {"dataset": cfg["dataset"], "n_dataset": len(ids),
              "dataset_summary": K.load_json(K.INTERIM / "expanded/dataset_summary.json"), "runs": {}}
    verdicts = {}
    for label in LABELS:
        path = out_dir / label / "executions.jsonl"
        recs = K.read_jsonl(path) if path.exists() else []
        cnt = Counter(r["id"] for r in recs)
        by = {r["id"]: r for r in recs if r["id"] in meta}
        missing = [i for i in ids if i not in by]
        provider_err = [i for i in ids if i in by and by[i]["result"]["category"] == "provider_error"]
        complete = not missing and not provider_err and all(cnt[i] == 1 for i in ids)
        scored = [by[i] for i in ids if i in by and i not in provider_err]
        # strict accuracy: missing / provider error count as wrong
        strict = [by[i] if (i in by and i not in provider_err) else
                  {"id": i, "result": {"correct": False, "category": "missing"}, "hardness": meta[i]["hardness"]}
                  for i in ids]
        verdicts[label] = {r["id"]: r["result"]["correct"] for r in strict}
        oc = Counter(outcome(r, max_new) if r["result"]["category"] != "missing" else "missing" for r in strict)
        mism = Counter(r["result"]["category"] for r in strict if r["result"]["category"] in MISMATCH)
        n = len(scored)
        validity = {
            "valid_json": sum(r["result"]["category"] != "format_error" for r in scored),
            "passed_validation": sum(r["result"]["category"] not in ("format_error", "invalid_mql") for r in scored),
            "executed_ok": sum(r["result"]["executed"] and r["result"]["category"] != "execution_error" for r in scored),
            "of": n,
        }
        v1_sub = [r for r in strict if meta[r["id"]]["in_v1_benchmark"]]
        run = {
            "model_id": next((r.get("model_id") for r in scored), None),
            "complete": complete, "records": len(recs), "scored": n,
            "missing_ids": missing, "provider_error_ids": provider_err,
            "duplicate_ids": sorted(i for i, c in cnt.items() if c > 1),
            "reused_v1_records": sum(1 for r in scored if r.get("reused_from")),
            "generated_now": sum(1 for r in scored if not r.get("reused_from")),
            "overall": acc(strict),
            "by_hardness": {h: acc([r for r in strict if meta[r["id"]]["hardness"] == h]) for h in HARD},
            "v1_279_subset": acc(v1_sub),
            "v1_279_matches_reported": sum(r["result"]["correct"] for r in v1_sub) == V1_REPORTED[label],
            "new_176_subset": acc([r for r in strict if not meta[r["id"]]["in_v1_benchmark"]]),
            "without_flagged": acc([r for r in strict if not meta[r["id"]]["flags"]]),
            "sensitivity_numeric_text": acc([{"result": {"correct": lenient_correct(r, meta[r["id"]])}}
                                             if r["result"]["category"] != "missing" else r for r in strict]),
            "sensitivity_without_post_hoc": acc([r for r in strict if r["id"] not in POST_HOC]),
            "validity": validity,
            "outcomes": dict(oc), "mismatch_breakdown": dict(mism),
            "tokens": {"input": sum(r.get("input_tokens") or 0 for r in scored),
                       "output": sum(r.get("output_tokens") or 0 for r in scored)},
            "cost_usd": 0.0,
        }
        report["runs"][label] = run
    pairs = {}
    for i, a in enumerate(LABELS):
        for b in LABELS[i + 1:]:
            only_a = sum(verdicts[a][x] and not verdicts[b][x] for x in ids)
            only_b = sum(verdicts[b][x] and not verdicts[a][x] for x in ids)
            both = sum(verdicts[a][x] and verdicts[b][x] for x in ids)
            pairs[f"{a} vs {b}"] = {"both_correct": both, f"only_{a}": only_a, f"only_{b}": only_b,
                                    "neither": len(ids) - both - only_a - only_b,
                                    "accuracy_diff": round((only_b - only_a) / len(ids), 4),
                                    "mcnemar_exact_p": mcnemar_exact(only_a, only_b, digits=None)}
    report["paired"] = pairs
    report["post_hoc_items"] = POST_HOC
    report["all_complete"] = all(r["complete"] for r in report["runs"].values())
    (out_dir / "report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    (out_dir / "report.md").write_text(render_md(report), encoding="utf-8")
    print(render_md(report))


def fmt_p(p):
    return "< 1e-6" if p < 1e-6 else f"{p:.2g}"


def pct(a):
    return f"{a['correct']}/{a['n']} = {100 * a['accuracy']:.1f}% [{100 * a['wilson95'][0]:.1f}, {100 * a['wilson95'][1]:.1f}]" if a["n"] else "n/a"


def render_md(rep):
    L = list(rep["runs"])
    ds = rep["dataset_summary"]
    out = [f"# Expanded Spider 1.0 NL -> MQL benchmark: results (n = {rep['n_dataset']})", "",
           (f"Dataset `{rep['dataset']}` (sha256 `{ds['dataset_sha256'][:16]}...`): eligible pool {ds['eligible_pool']}, "
            f"included {ds['included']}, excluded {ds['excluded']}. Execution accuracy is strict "
            "(an unanswered item counts as wrong); brackets are Wilson 95% CIs."), "",
           f"All runs complete: **{rep['all_complete']}**", "",
           "| | " + " | ".join(L) + " |", "|---|" + "---:|" * len(L)]

    def row(name, f):
        out.append(f"| {name} | " + " | ".join(f(rep["runs"][x]) for x in L) + " |")
    row("records scored / dataset", lambda r: f"{r['scored']}/{rep['n_dataset']}")
    row("reused v1 / generated now", lambda r: f"{r['reused_v1_records']} / {r['generated_now']}")
    row("**execution accuracy (all)**", lambda r: pct(r["overall"]))
    for h in HARD:
        row(f"  {h}", lambda r, h=h: pct(r["by_hardness"][h]))
    row("v1 279 subset", lambda r: pct(r["v1_279_subset"]) + (" (= v1)" if r["v1_279_matches_reported"] else " (DIFFERS from v1)"))
    row("176 new items", lambda r: pct(r["new_176_subset"]))
    row("without `ambiguous` flags", lambda r: pct(r["without_flagged"]))
    row("sensitivity: numeric TEXT == number", lambda r: pct(r["sensitivity_numeric_text"]))
    row("sensitivity: without 7 post-hoc items", lambda r: pct(r["sensitivity_without_post_hoc"]))
    out += ["", "**Output validity (separate from accuracy)**", "", "| | " + " | ".join(L) + " |", "|---|" + "---:|" * len(L)]
    row("valid JSON", lambda r: f"{r['validity']['valid_json']}/{r['validity']['of']}")
    row("passed read-only validation", lambda r: f"{r['validity']['passed_validation']}/{r['validity']['of']}")
    row("executed without error", lambda r: f"{r['validity']['executed_ok']}/{r['validity']['of']}")
    cats = ["correct", "malformed_output_truncated", "malformed_output", "validation_rejected", "execution_error",
            "timeout", "result_mismatch", "missing"]
    out += ["", "**Outcome categories**", "", "| | " + " | ".join(L) + " |", "|---|" + "---:|" * len(L)]
    for c in cats:
        row(c, lambda r, c=c: str(r["outcomes"].get(c, 0)))
    for c in MISMATCH:
        row(f"  mismatch: {c}", lambda r, c=c: str(r["mismatch_breakdown"].get(c, 0)))
    out += ["", "**Paired comparison (same items)**", "", "| pair | both | only first | only second | neither | McNemar exact p |",
            "|---|---:|---:|---:|---:|---:|"]
    for k, v in rep["paired"].items():
        a, b = k.split(" vs ")
        out.append(f"| {k} | {v['both_correct']} | {v['only_' + a]} | {v['only_' + b]} | {v['neither']} | {fmt_p(v['mcnemar_exact_p'])} |")
    inc = {k: v for k, v in rep["runs"].items() if not v["complete"]}
    if inc:
        out += ["", "**Incomplete runs**", ""]
        for k, v in inc.items():
            out.append(f"- {k}: missing {len(v['missing_ids'])}, provider errors {len(v['provider_error_ids'])}, "
                       f"duplicates {len(v['duplicate_ids'])}")
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    main()
