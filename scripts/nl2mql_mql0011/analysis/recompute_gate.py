"""MQL-0011 independent recompute of the stage-1 screen metrics and gate from per_item.jsonl ONLY.

Does not import retrieval_screen or any retrieval code. Recomputes counts, hit@3 (attainable), skeleton hit@3, the
registered deltas, the paired-bootstrap CIs (same registered seed/resamples, re-implemented here) and the verdict, and
compares every value with screen_results.json. Writes outputs/nl2mql_mql0011/screen/gate_recompute.json.
"""
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SCREEN = ROOT / "outputs/nl2mql_mql0011/screen"


def main():
    items = [json.loads(x) for x in (SCREEN / "per_item.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    res = json.loads((SCREEN / "screen_results.json").read_text(encoding="utf-8"))

    def rate(arm, key, sub):
        return 100.0 * sum(bool(i[arm][key]) for i in sub) / len(sub)

    def ci(diffs, B=10000, seed=20261122):
        rng = random.Random(seed)
        n = len(diffs)
        st = sorted(sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(B))
        return [round(100 * st[int(0.025 * B)], 2), round(100 * st[int(0.975 * B) - 1], 2)]

    def block(sub):
        att = [i for i in sub if i["attainable"]]
        sk = [i for i in sub if i["has_skeleton"]]
        d = lambda a, b: [int(i[a]["hit3"]) - int(i[b]["hit3"]) for i in att]   # noqa: E731
        return {"n_attainable": len(att), "n_skeleton": len(sk),
                "R3_minus_R0_hit3_att_pp": round(rate("R3", "hit3", att) - rate("R0", "hit3", att), 2),
                "R3_minus_R0_ci95": ci(d("R3", "R0")),
                "R3_minus_R1_hit3_att_pp": round(rate("R3", "hit3", att) - rate("R1", "hit3", att), 2),
                "R3_minus_R1_ci95": ci(d("R3", "R1")),
                "R3_minus_R0_skeleton_pp": round(rate("R3", "skel3", sk) - rate("R0", "skel3", sk), 2)}

    full = block(items)
    nt = block([i for i in items if not i["paraphrase_twin"]])
    arms = {a: {"hit@3_attainable": round(rate(a, "hit3", [i for i in items if i["attainable"]]), 2),
                "skeleton_hit@3": round(rate(a, "skel3", [i for i in items if i["has_skeleton"]]), 2)}
            for a in ("R0", "R1", "R2", "R3", "R0E", "R1E", "R3E")}
    g = lambda b: (b["R3_minus_R0_hit3_att_pp"] >= 3.0, b["R3_minus_R1_ci95"][0] > 0, b["R3_minus_R0_skeleton_pp"] >= -1.0)  # noqa: E731
    verdict = all(g(full)) and all(g(nt))
    out = {"n_items": len(items), "n_twins": sum(i["paraphrase_twin"] for i in items), "arms": arms,
           "all_items": full, "non_twin_items": nt, "PASS": verdict}
    mism = []
    for k in ("n_attainable", "n_skeleton", "R3_minus_R0_hit3_att_pp", "R3_minus_R0_ci95", "R3_minus_R1_hit3_att_pp",
              "R3_minus_R1_ci95", "R3_minus_R0_skeleton_pp"):
        for part in ("all_items", "non_twin_items"):
            if out[part][k] != res["gate"][part][k]:
                mism.append([part, k, out[part][k], res["gate"][part][k]])
    for a, m in arms.items():
        for k, v in m.items():
            if v != res["arms"][a][k]:
                mism.append([a, k, v, res["arms"][a][k]])
    if out["n_twins"] != res["n_paraphrase_twins"] or out["n_items"] != res["n_items"] or verdict != res["gate"]["PASS"]:
        mism.append(["counts/verdict", out["n_twins"], out["n_items"], verdict])
    out["mismatches_vs_screen_results"] = mism
    out["ok"] = not mism
    (SCREEN / "gate_recompute.json").write_text(json.dumps(out, indent=1, sort_keys=True) + "\n", encoding="utf-8",
                                               newline="\n")
    print(json.dumps({"ok": out["ok"], "mismatches": mism, "PASS": verdict, "n_twins": out["n_twins"]}))


if __name__ == "__main__":
    main()
