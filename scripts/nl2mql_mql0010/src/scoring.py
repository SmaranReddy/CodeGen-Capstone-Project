"""MQL-0010 scoring adapter: export verification, scoring with an injected evaluator, and the registered analysis.

The evaluator is injected so the whole path can be dry-run offline. For the real run, `atlas_evaluator` wraps the
UNCHANGED registered MQL-0006..0009 evaluator (dev1034_core.evaluate_prediction) against a read-only Atlas database
that passed one full-parity preflight; that path is not executed by any test.

Registered arms (amendment A4): A0, A0V, D0, D0V, D0G, E, EP, EF, FULL.
Registered test sequence (amendment A4, family-wise alpha 0.05, serial gatekeeping):
  step 1   D0V vs D0                                   exact two-sided McNemar
           + practical criterion: bootstrap 95% CI lower bound of (D0V - D0) >= +1.0 pp
  step 2   only if step 1 rejects: Holm over {FULL vs D0, D0V vs D0G}
  step 3   only if BOTH step-2 hypotheses reject: Holm over {E vs D0, EF vs EP}
Descriptive (no claim): A0 vs D0, A0V vs A0, D0G vs D0, recovery statistics, outcome taxonomy.
"""
import math
import random

ARMS = ("A0", "A0V", "D0", "D0V", "D0G", "E", "EP", "EF", "FULL")
RETRY_SOURCES = {"A0V": ("A0", "schema"), "D0V": ("D0", "schema"), "D0G": ("D0", "generic"), "FULL": ("EF", "schema")}
STEP1 = ("D0V", "D0")
STEP2 = (("FULL", "D0"), ("D0V", "D0G"))
STEP3 = (("E", "D0"), ("EF", "EP"))
DESCRIPTIVE = (("D0", "A0"), ("A0V", "A0"), ("D0G", "D0"))
PRACTICAL_MIN_PP = 1.0
ALPHA = 0.05


class ExportError(ValueError):
    pass


def need(c, msg):
    if not c:
        raise ExportError(msg)


# ------------------------------------------------------------------ export verification
def verify_exports(recs, plans, ids):
    """recs: {arm: [prediction records]}, plans: {"A0"|"D0"|"E"|"EF": [plan rows]}, ids: registered holdout ids.
    Fails closed on any missing arm/item, duplicate, failed generation or broken reuse contract."""
    need(set(recs) == set(ARMS), f"arms {sorted(recs)} != registered {list(ARMS)}")
    by = {}
    for a in ARMS:
        rows = recs[a]
        got = [r["id"] for r in rows]
        need(len(got) == len(set(got)), f"{a}: duplicate ids")
        need(set(got) == set(ids) and len(got) == len(ids), f"{a}: ids differ from the registered holdout")
        need(all(r["status"] == "ok" for r in rows), f"{a}: generation failures present")
        need(all(r["arm"] == a for r in rows), f"{a}: record arm label mismatch")
        by[a] = {r["id"]: r for r in rows}
    # retry arms reuse their source's first answers; only flagged items may change; at most one retry per item
    for a, (src, mode) in RETRY_SOURCES.items():
        for i in ids:
            r, s = by[a][i], by[src][i]
            need(r["first_raw_text"] == s["first_raw_text"], f"{a}/{i}: first answer differs from {src}")
            t = r.get("recovery")
            need(t is not None and t.get("feedback_mode") == mode, f"{a}/{i}: missing or wrong recovery trace")
            if not t["retried"]:
                need(r["raw_text"] == r["first_raw_text"], f"{a}/{i}: changed without a retry")
            need(r["raw_text"] in (r["first_raw_text"],) or t["kept"] == "retry", f"{a}/{i}: unexplained change")
    # R5: the generic control retries exactly the items the schema arm retries (same trigger, same first answers)
    rv = {i for i in ids if by["D0V"][i]["recovery"]["retried"]}
    rg = {i for i in ids if by["D0G"][i]["recovery"]["retried"]}
    need(rv == rg, f"D0V and D0G retried different items ({len(rv ^ rg)} differ)")
    # R2: EF / EP differ from E only on items whose EF prompt differs from E's plan, both generated in the delta pass
    pe = {r["id"]: r for r in plans["E"]}
    pf = {r["id"]: r for r in plans["EF"]}
    changed = {i for i in ids if pf[i]["prompt_sha256"] != pe[i]["prompt_sha256"]}
    for i in ids:
        for a in ("EF", "EP"):
            r = by[a][i]
            if i in changed:
                need(r["source"] == "generated:matched_delta", f"{a}/{i}: changed prompt not generated in the matched delta pass")
            else:
                need(r["first_raw_text"] == by["E"][i]["first_raw_text"] and r["source"] == "reused:E", f"{a}/{i}: should reuse E")
        if i in changed:
            need(by["EP"][i]["prompt_sha256"] == pe[i]["prompt_sha256"], f"EP/{i}: control prompt is not E's prompt")
            need(by["EF"][i]["prompt_sha256"] == pf[i]["prompt_sha256"], f"EF/{i}: prompt is not the EF plan prompt")
    for a, plan in (("A0", plans["A0"]), ("D0", plans["D0"]), ("E", plans["E"])):
        ph = {r["id"]: r["prompt_sha256"] for r in plan}
        need(all(by[a][i]["prompt_sha256"] == ph[i] for i in ids), f"{a}: prompt hash differs from its plan")
    return {"ok": True, "n": len(ids), "delta_items": len(changed), "retried": {a: sum(by[a][i]["recovery"]["retried"]
            for i in ids) for a in RETRY_SOURCES}}


# ------------------------------------------------------------------ scoring
def score_all(recs, evaluate):
    """evaluate(record) -> outcome string ("correct", "incorrect", "invalid", "failed", "timeout", "not_comparable").
    Every arm is scored by the same evaluator instance (one session, one preflight)."""
    return {a: {r["id"]: evaluate(r) for r in recs[a]} for a in ARMS}


def atlas_evaluator(mdb, K, refs_by_id, allowed, evaluate_prediction):
    """Real-run adapter around dev1034_core.evaluate_prediction (passed in, unchanged). Not executed by tests."""
    def ev(rec):
        out, _rows = evaluate_prediction(mdb, K, rec, refs_by_id[rec["id"]], allowed)
        return out["outcome"]
    return ev


# ------------------------------------------------------------------ statistics
def mcnemar_exact(b, c):
    n, k = b + c, min(b, c)
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def compare(outcomes, ids, a, b, reps=10000, seed=20261119):
    va = [outcomes[a][i] == "correct" for i in ids]
    vb = [outcomes[b][i] == "correct" for i in ids]
    a_only = sum(x and not y for x, y in zip(va, vb))
    b_only = sum(y and not x for x, y in zip(va, vb))
    d = [int(x) - int(y) for x, y in zip(va, vb)]
    n = len(d)
    rng = random.Random(seed)
    ms = sorted(sum(d[rng.randrange(n)] for _ in range(n)) / n for _ in range(reps))
    lo, hi = ms[int(0.025 * reps)], ms[int(0.975 * reps) - 1]
    return {"a": a, "b": b, "n": n, "acc_a": sum(va), "acc_b": sum(vb), "a_only": a_only, "b_only": b_only,
            "diff_pp": round(100 * sum(d) / n, 3), "ci95_pp": [round(100 * lo, 3), round(100 * hi, 3)],
            "p": mcnemar_exact(a_only, b_only)}


def holm(results, alpha=ALPHA):
    order = sorted(range(len(results)), key=lambda k: results[k]["p"])
    m, run, rejected_all = len(results), 0.0, True
    for rank, k in enumerate(order):
        run = max(run, min(1.0, (m - rank) * results[k]["p"]))
        results[k]["p_holm"] = round(run, 6)
    for k in order:
        results[k]["reject"] = results[k]["p_holm"] < alpha
    return all(r["reject"] for r in results)


def analyze(outcomes, ids, recs=None, reps=10000, seed=20261119):
    out = {"sequence": "A4 serial gatekeeping, alpha 0.05", "step1": None, "step2": None, "step3": None}
    s1 = compare(outcomes, ids, *STEP1, reps=reps, seed=seed)
    s1["reject"] = s1["p"] < ALPHA
    s1["practically_meaningful"] = s1["reject"] and s1["ci95_pp"][0] >= PRACTICAL_MIN_PP
    out["step1"] = s1
    if s1["reject"]:
        fam2 = [compare(outcomes, ids, a, b, reps=reps, seed=seed) for a, b in STEP2]
        all2 = holm(fam2)
        out["step2"] = fam2
        if all2:
            fam3 = [compare(outcomes, ids, a, b, reps=reps, seed=seed) for a, b in STEP3]
            holm(fam3)
            out["step3"] = fam3
        else:
            out["step3"] = "not tested (gate: both step-2 hypotheses must reject)"
    else:
        out["step2"] = out["step3"] = "not tested (gate: step 1 did not reject)"
    out["descriptive"] = [compare(outcomes, ids, a, b, reps=reps, seed=seed) for a, b in DESCRIPTIVE]
    out["accuracy"] = {a: sum(outcomes[a][i] == "correct" for i in ids) for a in ARMS}
    if recs is not None:
        out["recovery"] = {}
        for a, (src, _) in RETRY_SOURCES.items():
            by = {r["id"]: r for r in recs[a]}
            ret = [i for i in ids if by[i]["recovery"]["retried"]]
            kept = [i for i in ret if by[i]["recovery"]["kept"] == "retry"]
            out["recovery"][a] = {"retried": len(ret), "kept": len(kept),
                                  "retry_zero_issues": sum(by[i]["recovery"]["retry_issues"] == [] for i in ret),
                                  "fixed_by_retry": sum(outcomes[a][i] == "correct" and outcomes[src][i] != "correct" for i in kept),
                                  "broken_by_retry": sum(outcomes[a][i] != "correct" and outcomes[src][i] == "correct" for i in kept)}
    return out
