"""Verify the Spider 1.0 NL -> MQL accuracies from SAVED predictions (no model calls).

Two levels of evidence, written to outputs/evaluation_spider1/week2_verification/:

  offline (default, no network)
    * record integrity: unique ids, ids in the dataset, one model id / prompt hash per run,
      every `correct` verdict has executed=True, category "correct" and a saved MQL;
    * recount of strict execution accuracy on the full 279 (unanswered = failure) and on
      the answered ids, plus the paired subset answered by every run;
    * outcome buckets: correct / malformed output (truncated vs malformed JSON) /
      invalid MQL / execution failure / result mismatch / missing prediction;
    * partial re-check without Atlas: format errors re-parsed from `raw_text`, and
      mismatches whose full result was saved (<= 10 rows) re-scored with exec_match
      against the current gold rows. "correct" verdicts store no rows, so they can only
      be confirmed by --reexecute.

  --reexecute (live Atlas, read-only; still no model calls)
    * gold replay: every gold MQL through the same score path (expects 279/279; if not,
      the Atlas data has drifted and model re-execution is meaningless);
    * every saved model `raw_text` re-scored through run_eval.score_output
      (parse -> validate -> aggregate on `codegen_spider1` -> exec_match) and compared
      with the recorded verdict. Recorded verdicts stay canonical; disagreements are listed.

The original executions.jsonl / summary.json files are never modified.

Usage:
  python scripts/spider1/rescore.py [labels ...]              # offline audit
  python scripts/spider1/rescore.py --reexecute [labels ...]  # + live Atlas re-execution
"""
import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import compare as C  # noqa: E402
import spider1_kit as K  # noqa: E402

CONFIG = K.ROOT / "configs/evaluation_spider1.yaml"
RUNS = K.ROOT / "outputs/evaluation_spider1"
OUT = RUNS / "week2_verification"
LABELS = ["qwen2.5-coder-3b", "qwen2.5-coder-7b", "gpt-oss-120b"]
MISMATCH = ("wrong_row_count", "wrong_column_count", "wrong_values", "order_mismatch")
BUCKETS = ("correct", "malformed_truncated", "malformed_json", "invalid_mql",
           "execution_failure", "result_mismatch", "missing_prediction")


def bucket(rec, max_new_tokens):
    """Map a saved record (or None = never evaluated) to one outcome bucket."""
    if rec is None or rec["result"]["category"] == "provider_error":
        return "missing_prediction"
    cat = rec["result"]["category"]
    if cat == "correct":
        return "correct"
    if cat == "format_error":
        truncated = (rec.get("stop_reason") in ("length", "max_tokens")
                     or (rec.get("output_tokens") or 0) >= max_new_tokens)
        return "malformed_truncated" if truncated else "malformed_json"
    if cat == "invalid_mql":
        return "invalid_mql"
    if cat in ("execution_error", "timeout"):  # includes "result too large" (executed=True)
        return "execution_failure"
    if cat in MISMATCH:
        return "result_mismatch"
    raise ValueError(f"unknown category {cat!r} for {rec['id']}")


def integrity(records, data):
    ids = [r["id"] for r in records]
    in_ds = [r for r in records if r["id"] in data]
    bad_correct = [r["id"] for r in in_ds if r["result"]["correct"] and not (
        r["result"]["category"] == "correct" and r["result"]["executed"] and r.get("mql"))]
    bad_category = [r["id"] for r in in_ds if (r["result"]["category"] == "correct") != r["result"]["correct"]]
    return {"records": len(records), "unique_ids": len(set(ids)), "in_dataset": len(in_ds),
            "outside_dataset": len(records) - len(in_ds),
            "model_ids": sorted({str(r.get("model_id")) for r in in_ds}),
            "prompt_sha256": sorted({str(r.get("prompt_sha256")) for r in in_ds}),
            "provider_errors": sum(r["result"]["category"] == "provider_error" for r in in_ds),
            "correct_without_execution_evidence": bad_correct,
            "category_flag_disagreements": bad_category}


def offline_recheck(recs, data):
    """Re-check what saved records allow without Atlas. Returns counts and disagreements."""
    sys.path.insert(0, str(HERE.parent / "evaluation"))
    from run_nl_mql_eval import parse_output
    out = {"format_errors_reparsed": 0, "mismatches_rescored": 0,
           "mismatches_not_rescorable": 0, "disagreements": []}
    for i, r in recs.items():
        cat = r["result"]["category"]
        if cat == "format_error":
            out["format_errors_reparsed"] += 1
            if parse_output(r.get("raw_text"))[0] is not None:
                out["disagreements"].append({"id": i, "recorded": cat, "now": "parses"})
        elif cat in MISMATCH:
            prev = r.get("pred_preview")
            if prev is None or r["result"].get("n_rows") != len(prev):
                out["mismatches_not_rescorable"] += 1
                continue
            out["mismatches_rescored"] += 1
            ex = data[i]
            if K.exec_match([tuple(x) for x in prev], ex["gold_rows"], ex.get("tie_blocks"),
                            ex["order_by"])[0]:
                out["disagreements"].append({"id": i, "recorded": cat, "now": "correct"})
    return out


def audit(labels, data, cfg):
    all_ids = list(data)
    max_new = cfg["generation"]["max_new_tokens"]
    report = {"dataset": cfg["dataset"], "n_dataset": len(all_ids), "runs": {}}
    answered = {}
    for lab in labels:
        records = K.read_jsonl(RUNS / lab / "executions.jsonl")
        recs = {r["id"]: r for r in records
                if r["id"] in data and r["result"]["category"] != "provider_error"}
        answered[lab] = recs
        b = Counter(bucket(recs.get(i), max_new) for i in all_ids)
        c = b["correct"]
        n = len(recs)
        report["runs"][lab] = {
            "integrity": integrity(records, data),
            "strict_full": {"correct": c, "n": len(all_ids), "accuracy": round(c / len(all_ids), 4),
                            "ci95": C.wilson(c, len(all_ids))},
            "strict_answered": {"correct": c, "n": n, "accuracy": round(c / n, 4) if n else None,
                                "ci95": C.wilson(c, n)},
            "buckets": {k: b.get(k, 0) for k in BUCKETS},
            "categories": dict(Counter(r["result"]["category"] for r in recs.values()).most_common()),
            "offline_recheck": offline_recheck(recs, data)}
    ids = C.paired_ids(all_ids, answered)
    paired = {"n": len(ids), "runs": {}, "pairs": {}}
    for lab in labels:
        sub = {i: answered[lab][i] for i in ids}
        c = sum(r["result"]["correct"] for r in sub.values())
        paired["runs"][lab] = {"correct": c, "n": len(ids),
                               "accuracy": round(c / len(ids), 4) if ids else None,
                               "ci95": C.wilson(c, len(ids)),
                               "buckets": dict(Counter(bucket(r, max_new) for r in sub.values()))}
    for x, a in enumerate(labels):
        for b_ in labels[x + 1:]:
            ra, rb = answered[a], answered[b_]
            only_a = sum(ra[i]["result"]["correct"] and not rb[i]["result"]["correct"] for i in ids)
            only_b = sum(rb[i]["result"]["correct"] and not ra[i]["result"]["correct"] for i in ids)
            both = sum(ra[i]["result"]["correct"] and rb[i]["result"]["correct"] for i in ids)
            paired["pairs"][f"{a} vs {b_}"] = {
                "both_correct": both, f"only_{a}": only_a, f"only_{b_}": only_b,
                "neither": len(ids) - both - only_a - only_b,
                "mcnemar_exact_p": C.mcnemar_exact(only_a, only_b)}
    report["paired"] = paired
    report["paired_ids"] = ids
    return report, answered


def reexecute(labels, data, cfg, answered):
    from run_eval import score_output
    client = K.mongo_client()
    try:
        mdb = client[cfg["atlas_db"]]
        colls = mdb.list_collection_names()

        def allowed(db_id):
            return {c for c in colls if c.startswith(db_id + "__")}

        out = {}
        gold = []
        for ex in data.values():
            res, _ = score_output(mdb, json.dumps(ex["gold_mql"]), ex, cfg["execution"], allowed(ex["db_id"]))
            gold.append({"id": ex["id"], **res})
        K.write_jsonl(OUT / "gold_replay.jsonl", gold)
        out["gold_replay"] = {"n": len(gold), "correct": sum(g["correct"] for g in gold),
                              "failures": [g["id"] for g in gold if not g["correct"]]}
        print(f"gold replay: {out['gold_replay']['correct']}/{len(gold)}", flush=True)
        for lab in labels:
            rows, flips = [], []
            for i, r in answered[lab].items():
                ex = data[i]
                res, _ = score_output(mdb, r.get("raw_text"), ex, cfg["execution"], allowed(ex["db_id"]))
                same = (res["correct"] == r["result"]["correct"]
                        and res["category"] == r["result"]["category"])
                rows.append({"id": i, "recorded": r["result"]["category"], "reexecuted": res["category"],
                             "reexecuted_correct": res["correct"], "reason": res["reason"], "agrees": same})
                if not same:
                    flips.append({"id": i, "recorded": r["result"]["category"],
                                  "reexecuted": res["category"], "reason": res["reason"]})
            K.write_jsonl(OUT / f"reexec_{lab}.jsonl", rows)
            c = sum(x["reexecuted_correct"] for x in rows)
            out[lab] = {"n": len(rows), "reexecuted_correct": c,
                        "recorded_correct": sum(r["result"]["correct"] for r in answered[lab].values()),
                        "agreements": len(rows) - len(flips), "disagreements": flips}
            print(f"{lab}: re-executed {c}/{len(rows)} correct; {len(flips)} verdict disagreements",
                  flush=True)
    finally:
        client.close()
    out["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    return out


def to_markdown(rep):
    pct = lambda d: f"{d['correct']}/{d['n']} = {d['accuracy']:.1%} ({d['ci95'][0]:.1%}–{d['ci95'][1]:.1%})"  # noqa: E731
    md = [f"# Spider 1.0 NL→MQL — verification of saved results ({rep['n_dataset']} examples)\n",
          "Generated by `scripts/spider1/rescore.py`; no model calls.\n",
          "## Strict execution accuracy\n",
          "| run | answered | strict EX on all 279 (unanswered = fail) | EX on answered ids |",
          "|---|---|---|---|"]
    for lab, r in rep["runs"].items():
        md.append(f"| {lab} | {r['strict_answered']['n']} | {pct(r['strict_full'])} | {pct(r['strict_answered'])} |")
    p = rep["paired"]
    md += [f"\n## Paired: same {p['n']} ids\n", "| run | EX |", "|---|---|"]
    md += [f"| {lab} | {pct(r)} |" for lab, r in p["runs"].items()]
    md.append("")
    for k, v in p["pairs"].items():
        md.append(f"- **{k}**: " + ", ".join(f"{a}={b:.3g}" if isinstance(b, float) else f"{a}={b}"
                                             for a, b in v.items()))
    md += ["\n## Outcome buckets (all 279 ids)\n", "| run | " + " | ".join(BUCKETS) + " |",
           "|---|" + "---|" * len(BUCKETS)]
    for lab, r in rep["runs"].items():
        md.append(f"| {lab} | " + " | ".join(str(r["buckets"][k]) for k in BUCKETS) + " |")
    md.append("\n## Record integrity and offline re-check\n")
    for lab, r in rep["runs"].items():
        it, oc = r["integrity"], r["offline_recheck"]
        md.append(f"- **{lab}**: {it['records']} records ({it['outside_dataset']} outside the dataset, "
                  f"ignored), {it['unique_ids']} unique ids, model ids {it['model_ids']}, "
                  f"{len(it['prompt_sha256'])} prompt hash(es), provider errors {it['provider_errors']}, "
                  f"correct without execution evidence {len(it['correct_without_execution_evidence'])}; "
                  f"format errors re-parsed {oc['format_errors_reparsed']}, mismatches re-scored offline "
                  f"{oc['mismatches_rescored']} (not re-scorable offline: {oc['mismatches_not_rescorable']}), "
                  f"disagreements {len(oc['disagreements'])}")
    rx = rep.get("reexecution")
    md.append("\n## Live Atlas re-execution\n")
    if not rx:
        md.append("Not run in this invocation (use `--reexecute`).")
    else:
        g = rx["gold_replay"]
        md.append(f"Run at {rx['timestamp']} against `codegen_spider1`.\n")
        md.append(f"- **gold MQL replay**: {g['correct']}/{g['n']} correct; failures {g['failures']}")
        for lab in rep["runs"]:
            x = rx[lab]
            md.append(f"- **{lab}**: re-executed {x['reexecuted_correct']}/{x['n']} correct "
                      f"(recorded {x['recorded_correct']}); verdict agreement {x['agreements']}/{x['n']}")
            for f in x["disagreements"]:
                md.append(f"  - `{f['id']}`: recorded {f['recorded']} → re-executed {f['reexecuted']} ({f['reason']})")
    return "\n".join(md) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("labels", nargs="*", default=LABELS)
    ap.add_argument("--reexecute", action="store_true", help="re-run saved MQL on Atlas (read-only)")
    a = ap.parse_args()
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    data = {d["id"]: d for d in K.read_jsonl(K.ROOT / cfg["dataset"])}
    rep, answered = audit(a.labels, data, cfg)
    if a.reexecute:
        OUT.mkdir(parents=True, exist_ok=True)
        rep["reexecution"] = reexecute(a.labels, data, cfg, answered)
    OUT.mkdir(parents=True, exist_ok=True)
    name = "verification" if a.reexecute else "audit_offline"
    (OUT / f"{name}.json").write_text(json.dumps(rep, indent=2, default=str), encoding="utf-8")
    md = to_markdown(rep)
    (OUT / f"{name}.md").write_text(md, encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
