"""Descriptive recount of MQL-0010 run_final from stored evaluation_results (read-only)."""
import json, math, sys, hashlib
from collections import Counter
from pathlib import Path

run = Path(sys.argv[1])
ARMS = ("A0", "A0V", "D0", "D0V", "D0G", "E", "EP", "EF", "FULL")
ev = {}
for a in ARMS:
    rows = [json.loads(l) for l in open(run / "scoring/arms" / a / "evaluation_results.jsonl", encoding="utf-8") if l.strip()]
    ev[a] = {r["id"]: r for r in rows}
ids = list(ev["A0"])
assert all(set(ev[a]) == set(ids) and len(ev[a]) == len(ids) for a in ARMS)
pred = {}
for a in ARMS:
    p = run / "predictions" / a / "predictions.jsonl"
    pred[a] = {json.loads(l)["id"]: json.loads(l) for l in open(p, encoding="utf-8") if l.strip()}

def wilson(k, n, z=1.959964):
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return round(100 * (c - h) / d, 2), round(100 * (c + h) / d, 2)

out = {"n": len(ids), "arms": {}}
for a in ARMS:
    c = Counter(r["outcome"] for r in ev[a].values())
    det = Counter((r["outcome"], r.get("detail")) for r in ev[a].values() if r["outcome"] != "correct")
    k = c["correct"]
    out["arms"][a] = {"outcomes": dict(c), "correct": k, "acc_pct": round(100 * k / len(ids), 2), "wilson95": wilson(k, len(ids)),
                      "detail": {f"{o}:{d}": v for (o, d), v in det.most_common()}}

def pair(a, b):
    fx = sum(ev[a][i]["outcome"] == "correct" and ev[b][i]["outcome"] != "correct" for i in ids)
    rg = sum(ev[a][i]["outcome"] != "correct" and ev[b][i]["outcome"] == "correct" for i in ids)
    return {"fixes": fx, "regressions": rg, "net": fx - rg}

out["pairs"] = {f"{a}-{b}": pair(a, b) for a, b in [("A0V", "A0"), ("D0V", "D0"), ("D0G", "D0"), ("D0", "A0"), ("E", "D0"),
                                                       ("EF", "EP"), ("EF", "E"), ("EP", "E"), ("FULL", "EF"), ("FULL", "D0"),
                                                       ("D0V", "D0G"), ("FULL", "D0V")]}
# delta-subset (52 items whose EF prompt differs)
delta = [i for i in ids if pred["EF"][i].get("source") == "generated:matched_delta"]
out["delta_subset"] = {"n": len(delta), **{a: sum(ev[a][i]["outcome"] == "correct" for i in delta) for a in ("E", "EP", "EF")}}
# retried subset per retry arm
src = {"A0V": "A0", "D0V": "D0", "D0G": "D0", "FULL": "EF"}
out["retried_subset"] = {}
for a, s in src.items():
    ret = [i for i in ids if pred[a][i]["recovery"]["retried"]]
    out["retried_subset"][a] = {"n": len(ret), "src_correct": sum(ev[s][i]["outcome"] == "correct" for i in ret),
                                "arm_correct": sum(ev[a][i]["outcome"] == "correct" for i in ret),
                                "src_outcomes": dict(Counter(ev[s][i]["outcome"] for i in ret)),
                                "arm_outcomes": dict(Counter(ev[a][i]["outcome"] for i in ret))}
out["infra_failures"] = {a: sum(r["outcome"] == "failed" and r.get("failure_source") == "infrastructure" for r in ev[a].values()) for a in ARMS}
# determinism: reused records must have identical outcomes
det = {}
for a in ("EP", "EF"):
    det[a + "==E (non-delta)"] = sum(ev[a][i]["outcome"] != ev["E"][i]["outcome"] for i in ids if i not in set(delta))
for a, s in src.items():
    det[f"{a}=={s} (not retried)"] = sum(ev[a][i]["outcome"] != ev[s][i]["outcome"] for i in ids if not pred[a][i]["recovery"]["retried"])
    det[f"{a}=={s} (retried, retry not kept)"] = sum(ev[a][i]["outcome"] != ev[s][i]["outcome"] for i in ids
                                                    if pred[a][i]["recovery"]["retried"] and pred[a][i]["recovery"]["kept"] != "retry")
out["determinism_mismatches"] = det
out["eval_sha256"] = {a: hashlib.sha256((run / "scoring/arms" / a / "evaluation_results.jsonl").read_bytes()).hexdigest() for a in ARMS}
print(json.dumps(out, indent=1))
