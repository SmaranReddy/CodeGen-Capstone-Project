"""Round-3 (G4) sampling strata and statistics. Pure functions; no MongoDB, no converter.

Strata are computed from the ORIGINAL SQL text only (sqlglot parse of the top-level SELECT),
with fixed precedence J2 > OL > J1 > G:
  J2  multi-table join: >= 2 JOIN clauses in the top-level SELECT (>= 3 tables)
  OL  ORDER BY ... LIMIT (top-level; includes LIMIT 1), fewer than 2 joins
  J1  exactly one JOIN, no ORDER BY ... LIMIT
  G   everything else (single table, subqueries, set operations, unparsable)

G4 estimator for a stratified sample (pre-registered, data/interim/finetune_v2/round3_prereg.json):
  unit         an ACCEPTED label (verifier-3 accepted the converter's MQL)
  y_i          1 if the label has >= 1 verifier-accepted mutant the oracle shows non-equivalent
  weight       w_h = N_h / n_h  (N_h population records, n_h sampled records of stratum h);
               every accepted label of stratum h carries w_h, so sum(w*1[accepted]) estimates
               the number of accepted labels the stratum would yield in the population
  p_w          = sum(w_i y_i) / sum(w_i)            (ratio estimator over accepted labels)
  n_eff        = (sum w_i)^2 / sum(w_i^2)           (Kish effective sample size)
  CI           Wilson 95% (z = 1.96, the method of rounds 1-2: v2_common.wilson) applied to
               (k = p_w * n_eff, n = n_eff)
  gate         Wilson upper bound <= 0.05, compared UNROUNDED (stricter than rounding first)
With equal weights this reduces exactly to the round-1/2 statistic (k/n, Wilson).
"""
import math

from sqlglot import exp, parse_one

import v2_common as C

STRATA = ("J2", "OL", "J1", "G")


def stratum(sql):
    try:
        t = parse_one(sql, read="sqlite")
    except Exception:  # noqa: BLE001
        return "G"
    if not isinstance(t, exp.Select):
        return "G"
    nj = len(t.args.get("joins") or [])
    if nj >= 2:
        return "J2"
    if t.args.get("order") is not None and t.args.get("limit") is not None:
        return "OL"
    if nj == 1:
        return "J1"
    return "G"


def wilson_raw(k, n, z=1.96):
    """Unrounded Wilson interval (v2_common.wilson rounds to 4 decimals); the gate uses this."""
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, (c - h) / d), min(1.0, (c + h) / d)


def weighted_g4(labels, N, n):
    """labels: [(stratum, y)] for every ACCEPTED label evaluated; N/n: {stratum: records}.
    -> dict with unweighted and weighted estimates and per-stratum results."""
    w = {h: N[h] / n[h] for h in N if n.get(h)}
    sw = sum(w[h] for h, _ in labels)
    sw2 = sum(w[h] ** 2 for h, _ in labels)
    swy = sum(w[h] * y for h, y in labels)
    k = sum(y for _, y in labels)
    out = {"unweighted": {"k": k, "n": len(labels), "rate": round(k / len(labels), 4) if labels else None,
                          "wilson95": list(C.wilson(k, len(labels)))}}
    if labels:
        p_w = swy / sw
        n_eff = sw * sw / sw2
        lo, hi = C.wilson(p_w * n_eff, n_eff)
        hi_raw = wilson_raw(p_w * n_eff, n_eff)[1]
        out["weighted"] = {"p_w": round(p_w, 4), "n_eff": round(n_eff, 1),
                           "k_eff": round(p_w * n_eff, 2), "wilson95": [lo, hi], "upper_unrounded": hi_raw,
                           "estimated_accepted_population": round(sw)}
        out["gate_G4_upper_bound_le_0.05"] = hi_raw <= 0.05
    by = {}
    for h in STRATA:
        ys = [y for hh, y in labels if hh == h]
        by[h] = {"k": sum(ys), "n": len(ys), "rate": round(sum(ys) / len(ys), 4) if ys else None,
                 "wilson95": list(C.wilson(sum(ys), len(ys))), "weight_per_label": round(w.get(h, 0), 3)}
    out["by_stratum"] = by
    return out


def max_k_passing(n_eff):
    """Largest (real) k with Wilson upper(k, n_eff) <= 0.05, by bisection."""
    lo, hi = 0.0, n_eff
    if wilson_raw(0, n_eff)[1] > 0.05:
        return None
    for _ in range(60):
        mid = (lo + hi) / 2
        if wilson_raw(mid, n_eff)[1] <= 0.05:
            lo = mid
        else:
            hi = mid
    return lo


def pass_probability(n_eff, p):
    """P(gate passes) if the true label-level false-accept rate is p, approximating the weighted
    count by Binomial(round(n_eff), p) (normal approximation is not used)."""
    n = int(round(n_eff))
    kmax = max_k_passing(n)
    if kmax is None:
        return 0.0
    kk = int(math.floor(kmax + 1e-9))
    return round(sum(math.comb(n, j) * p ** j * (1 - p) ** (n - j) for j in range(kk + 1)), 4)
