"""MQL-0010 post-hoc evaluation-noise sensitivity (descriptive; the registered analysis.json stays official).

Reused records (byte-identical raw_text to their source arm) are given their source's outcome, then the UNCHANGED
registered scoring.analyze is re-run. Read-only; no DB access.
Usage: python -I -B scripts/nl2mql_mql0010/analysis/posthoc_noise_sensitivity.py <run_dir> <out_json>
"""
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "kaggle/nl2mql_mql0010_holdout/src"))
import scoring as SC

run, outp = Path(sys.argv[1]), Path(sys.argv[2])
L = lambda p: {json.loads(l)["id"]: json.loads(l) for l in open(p, encoding="utf-8") if l.strip()}
ev = {a: L(run / f"scoring/arms/{a}/evaluation_results.jsonl") for a in SC.ARMS}
pr = {a: L(run / f"predictions/{a}/predictions.jsonl") for a in SC.ARMS}
ids = list(pr["A0"])
o = {a: {i: ev[a][i]["outcome"] for i in ids} for a in SC.ARMS}
src = {"EP": "E", "EF": "E", "A0V": "A0", "D0V": "D0", "D0G": "D0", "FULL": "EF"}   # EF before FULL
flips = []
for a in ("EP", "EF", "A0V", "D0V", "D0G", "FULL"):
    s = src[a]
    for i in ids:
        if pr[a][i]["raw_text"] == pr[s][i]["raw_text"]:
            if ev[a][i]["outcome"] != ev[s][i]["outcome"]:
                flips.append({"arm": a, "source": s, "id": i, "source_outcome": ev[s][i]["outcome"], "arm_outcome": ev[a][i]["outcome"]})
            o[a][i] = o[s][i]
an = SC.analyze(o, ids, {a: list(pr[a].values()) for a in SC.ARMS})
out = {"note": "post hoc, descriptive; registered result = scoring/analysis.json", "n_flips": len(flips), "flips": flips, "analysis": an}
outp.write_text(json.dumps(out, indent=1), encoding="utf-8")
print(json.dumps({"n_flips": len(flips), "accuracy": an["accuracy"],
                  "step2": [(r["a"], r["b"], r["diff_pp"], r["p_holm"], r["reject"]) for r in an["step2"]],
                  "step3": [(r["a"], r["b"], r["diff_pp"], r["p_holm"], r["reject"]) for r in an["step3"]]}))
