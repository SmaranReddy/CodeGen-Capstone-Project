"""MQL-0015 planning power for the exact two-sided McNemar test, n = 1,000 paired items (CPU, exact enumeration).

Model: each item is discordant with probability pd; given D discordant items, b ~ Binomial(D, q), where
q = (1 + delta/pd) / 2, so that delta = pd * (2q - 1) is the paired accuracy difference. The test rejects when the exact
two-sided binomial p-value of b out of D (null q = 0.5) is <= alpha. Power = sum_D P(D) * P(reject | D).
Planning assumptions (NOT guarantees):
  A. MQL-0014 R3V vs R0V discordance: 62/716 = 8.66% (requested planning reference; same weights, different retrieval)
  B. MQL-0010-FT1 vs base A0 discordance: 193/716 = 26.96% (weights differ, as in FT-R3V vs R3V)
  C. MQL-0010 D0 vs A0 discordance, a RAG-vs-zero-shot reference, read from MQL-0010 run_final if present
     (used for the R3V vs A0V secondary)
Prints MDE (smallest delta, 0.1 pp grid, with power >= 0.80) at alpha 0.05 (primary) and alpha 0.025 (Holm worst case for
the two secondaries), and the power at a few deltas.
"""
import json
import math
import sys
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
N = 1000


def log_binom_pmf(k, n, p):
    if p <= 0:
        return 0.0 if k == 0 else -math.inf
    if p >= 1:
        return 0.0 if k == n else -math.inf
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1) + k * math.log(p) + (n - k) * math.log1p(-p)


@lru_cache(maxsize=None)
def reject_set(D, alpha):
    """Smallest b_hi such that b >= b_hi (and symmetric b <= D - b_hi) rejects at two-sided exact p <= alpha."""
    if D == 0:
        return None
    pmf = [math.exp(log_binom_pmf(k, D, 0.5)) for k in range(D + 1)]
    tail = 0.0
    hi = None
    for b in range(D, (D - 1) // 2, -1):
        tail += pmf[b]
        if min(1.0, 2 * tail) <= alpha + 1e-15:
            hi = b
        else:
            break
    return hi


def power(pd, delta, alpha, n=N):
    q = (1 + delta / pd) / 2
    if not 0 <= q <= 1:
        return float("nan")
    tot = 0.0
    lo_D = max(0, int(n * pd - 8 * math.sqrt(n * pd * (1 - pd))) - 5)
    hi_D = min(n, int(n * pd + 8 * math.sqrt(n * pd * (1 - pd))) + 5)
    for D in range(lo_D, hi_D + 1):
        pD = math.exp(log_binom_pmf(D, n, pd))
        hi = reject_set(D, alpha)
        if hi is None or pD < 1e-14:
            continue
        pr = sum(math.exp(log_binom_pmf(b, D, q)) for b in range(hi, D + 1))
        pr += sum(math.exp(log_binom_pmf(b, D, q)) for b in range(0, D - hi + 1))
        tot += pD * pr
    return tot


def mde(pd, alpha, target=0.80):
    d = 0.001
    while d < pd:
        if power(pd, d, alpha) >= target:
            return d
        d += 0.001
    return None


def main():
    refs = {"A_mql0014_R3V_vs_R0V": 62 / 716, "B_ft1_vs_baseA0": 193 / 716}
    p = ROOT / "outputs/nl2mql_mql0010/scoring_runs/run_final/scoring/arms"
    try:
        def load(a):
            return {json.loads(x)["id"]: json.loads(x)["outcome"] == "correct"
                    for x in (p / a / "evaluation_results.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()}
        a0, d0v = load("A0V"), load("D0V")
        disc = sum(a0[i] != d0v[i] for i in a0)
        refs["C_mql0010_D0V_vs_A0V"] = disc / len(a0)
    except (OSError, KeyError, ValueError):
        pass
    out = {"n": N, "test": "exact two-sided McNemar", "assumptions_not_guarantees": {k: round(v, 4) for k, v in refs.items()},
           "results": {}}
    for k, pd in refs.items():
        r = {}
        for alpha in (0.05, 0.025):
            m = mde(pd, alpha)
            r[f"mde_pp_power80_alpha{alpha}"] = round(100 * m, 1) if m else None
            r[f"power_alpha{alpha}"] = {f"{dd}pp": round(power(pd, dd / 100, alpha), 3) for dd in (2, 3, 4, 5, 7, 10)
                                        if dd / 100 < pd}
        out["results"][k] = r
    print(json.dumps(out, indent=1))
    if len(sys.argv) > 1:
        Path(sys.argv[1]).write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
