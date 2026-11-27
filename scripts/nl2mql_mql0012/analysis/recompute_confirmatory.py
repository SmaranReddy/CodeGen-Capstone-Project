"""MQL-0012 independent recompute of every confirmatory value from results/per_item.jsonl ONLY.

Does not import run_mql0012 or any retrieval code; re-implements the registered statistics from the config text
(numpy default_rng(seed / seed+1 / seed+2), int32 index matrices in chunks of 1,000, index percentile, exact McNemar,
one-sided NI p = (1 + #{delta <= -1.0}) / (B + 1), Holm over {p_X1, p_X2}) and compares with results/results.json.
Writes results/recompute_confirmatory.json.
"""
import json
from math import comb
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
RES = ROOT / "outputs/nl2mql_mql0012/results"


def boot_dist(diffs, n, seed, B=10000):
    rng = np.random.default_rng(seed)
    d = np.asarray(diffs, dtype=np.float64)
    out = []
    for s in range(0, B, 1000):
        ix = rng.integers(0, n, size=(min(1000, B - s), n), dtype=np.int32)
        out.append(d[ix].mean(axis=1))
    return np.sort(np.concatenate(out)) * 100


def mc(b, c):
    n = b + c
    return 1.0 if not n else min(1.0, 2 * sum(comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def cmp(per, a, b, key, seed, margin=None):
    x = [(bool(i[a][key]), bool(i[b][key])) for i in per]
    ao, bo = sum(p and not q for p, q in x), sum(q and not p for p, q in x)
    dist = boot_dist([int(p) - int(q) for p, q in x], len(x), seed)
    r = {"n": len(x), "a_only": ao, "b_only": bo, "delta_pp": round(100 * (ao - bo) / len(x), 2),
         "ci95": [round(float(dist[250]), 2), round(float(dist[9749]), 2)], "mcnemar_p": mc(ao, bo)}
    if margin is not None:
        r["ni_p_one_sided"] = (1 + int((dist <= margin + 1e-12).sum())) / 10001
    return r


def main():
    per = [json.loads(x) for x in (RES / "per_item.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    res = json.loads((RES / "results.json").read_text(encoding="utf-8"))["confirmatory"]
    seed = 20261123
    att = [i for i in per if i["attainable"]]
    shp = [i for i in per if i["has_skeleton"]]
    join = [i for i in att if "join" in i["subgroups"]]
    mine = {"X1_vs_R3_structure": cmp(att, "X1", "R3", "hit3", seed),
            "X1_vs_R3_join": cmp(join, "X1", "R3", "hit3", seed + 2),
            "X2_vs_R3_structure": cmp(att, "X2", "R3", "hit3", seed, -1.0),
            "X2_vs_R3_shape": cmp(shp, "X2", "R3", "skel3", seed + 1, -1.0)}
    p1 = mine["X1_vs_R3_structure"]["mcnemar_p"]
    p2 = max(mine["X2_vs_R3_structure"]["ni_p_one_sided"], mine["X2_vs_R3_shape"]["ni_p_one_sided"])
    order = sorted([("X1", p1), ("X2", p2)], key=lambda t: t[1])
    holm = {order[0][0]: order[0][1] <= 0.025}
    holm[order[1][0]] = holm[order[0][0]] and order[1][1] <= 0.05
    x1 = (mine["X1_vs_R3_structure"]["delta_pp"] >= 2.0 and mine["X1_vs_R3_structure"]["ci95"][0] > 0
          and mine["X1_vs_R3_join"]["delta_pp"] >= -2.0 and holm["X1"])
    x2 = mine["X2_vs_R3_structure"]["ci95"][0] > -1.0 and mine["X2_vs_R3_shape"]["ci95"][0] > -1.0 and holm["X2"]
    mism = []
    for k, v in mine.items():
        for f, val in v.items():
            if res[k][f] != val:
                mism.append([k, f, val, res[k][f]])
    for k, a, b in (("p_X1", p1, res["p_X1"]), ("p_X2", p2, res["p_X2"]), ("holm", holm, res["holm_reject"]),
                    ("X1_PASS", x1, res["X1"]["PASS"]), ("X2_PASS", x2, res["X2"]["PASS"])):
        if a != b:
            mism.append([k, a, b])
    out = {"recomputed": mine, "p_X1": p1, "p_X2": p2, "holm": holm, "X1_PASS": x1, "X2_PASS": x2,
           "mismatches": mism, "ok": not mism}
    (RES / "recompute_confirmatory.json").write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"ok": out["ok"], "mismatches": mism, "X1_PASS": x1, "X2_PASS": x2}))


if __name__ == "__main__":
    main()
