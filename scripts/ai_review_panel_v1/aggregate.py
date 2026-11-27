"""AI review panel v1: validation, disputes, final labels, calibration, G-AIR, S9 precision, S10 log, stability.

    python -B scripts/ai_review_panel_v1/aggregate.py validate <phase>     phase = main | stability
    python -B scripts/ai_review_panel_v1/aggregate.py disputes <phase>     -> work/<batch>/dispute_list.json
    python -B scripts/ai_review_panel_v1/aggregate.py final <phase>        -> <phase>/final_labels.jsonl
    python -B scripts/ai_review_panel_v1/aggregate.py report               calibration + G-AIR + S9 + S10 + stability
                                                                            -> outputs/ai_review_panel_v1/panel_report.json

Decision rules (pre-registered in outputs/ai_review_panel_v1/prereg.json):
  * per dimension, assessing agents: source = A1, A3, A4(pass 1); translation = A2, A3, A4(pass 1);
  * unanimous YES on both assessed dimensions -> final YES (no adjudication); any NO / UNSURE / disagreement -> Agent 4
    adjudication (fresh evidence) decides; majority vote is never used;
  * NOT_APPLICABLE (no MQL) is carried through for the translation dimension.
All outputs are labelled AI-reviewed (multi-agent panel), never human-reviewed.
"""
import json
import math
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PANEL = ROOT / "outputs/ai_review_panel_v1"
ORCH = PANEL / "_orchestrator_only"
V = {"YES", "NO", "UNSURE"}
SRC_AGENTS, TRN_AGENTS = ("agent_1", "agent_3", "agent_4"), ("agent_2", "agent_3", "agent_4")
LABEL = "AI-reviewed (multi-agent panel); not human-reviewed"


def rd(p):
    return [json.loads(x) for x in Path(p).read_text(encoding="utf-8").splitlines() if x.strip()]


def wilson(k, n, z=1.959963984540054):
    if n == 0:
        return None
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [round(max(0.0, c - h), 4), round(min(1.0, c + h), 4)]


def key(phase):
    return {k["panel_id"]: k for k in rd(ORCH / f"panel_key_{phase}.jsonl")}


def batches(phase):
    return sorted(p.stem for p in (PANEL / phase / "package").glob("batch_*.jsonl"))


def pkg(phase, b):
    return {it["panel_id"]: it for it in rd(PANEL / phase / "package" / f"{b}.jsonl")}


def agent_rows(phase, b, agent):
    p = PANEL / phase / "work" / b / agent / "output.jsonl"
    rows = {}
    for r in rd(p):
        rows[r["panel_id"]] = r
    return rows


def exec_ids(phase, b, agent):
    p = PANEL / phase / "work" / b / agent / "exec_log.jsonl"
    return {json.loads(x)["id"] for x in open(p, encoding="utf-8") if x.strip()} if p.exists() else set()


def norm(v):
    v = str(v or "").strip().upper()
    return v if v in V | {"NOT_ASSESSED", "NOT_APPLICABLE"} else "INVALID:" + v


def validate(phase):
    out, ok = {}, True
    for b in batches(phase):
        items = pkg(phase, b)
        for agent in ("agent_1", "agent_2", "agent_3", "agent_4", "agent_4_adj"):
            p = PANEL / phase / "work" / b / agent / "output.jsonl"
            if not p.exists():
                out[f"{b}/{agent}"] = "missing"
                ok &= agent == "agent_4_adj"
                continue
            rows = agent_rows(phase, b, agent)
            want = set(items) if agent != "agent_4_adj" else {d["panel_id"] for d in json.loads(
                (PANEL / phase / "work" / b / "dispute_list.json").read_text(encoding="utf-8"))}
            probs = []
            if set(rows) != want:
                probs.append(f"ids: missing {len(want - set(rows))} extra {len(set(rows) - want)}")
            known = exec_ids(phase, b, agent)
            for pid, r in rows.items():
                s, t = norm(r.get("source_sql_correct")), norm(r.get("mql_translation_correct"))
                need_s = agent in ("agent_1", "agent_3", "agent_4", "agent_4_adj")
                need_t = agent in ("agent_2", "agent_3", "agent_4", "agent_4_adj")
                if need_s and s not in V:
                    probs.append(f"{pid} source={s}")
                if need_t and t not in V and not (t == "NOT_APPLICABLE" and items.get(pid, {}).get("mql") is None):
                    probs.append(f"{pid} translation={t}")
                cites = set(re.findall(r"[A-Za-z0-9_]+-\d{4}", json.dumps(r.get("execution_evidence", []))))
                tag = next(iter(known)).split("-")[0] if known else None
                cites = {f"{tag}-{c[2:]}" if c.startswith("T-") and tag else c for c in cites}  # deviation DEV-1
                if cites - known:
                    probs.append(f"{pid} cites unknown ids {sorted(cites - known)[:3]}")
                if "NO" in (s, t) and not cites and not r.get("counterexample"):
                    probs.append(f"{pid} NO without evidence")
            out[f"{b}/{agent}"] = {"n": len(rows), "problems": probs[:20], "n_problems": len(probs)}
            ok &= not probs
    print(json.dumps(out, indent=1))
    return ok


def votes(phase, b):
    items = pkg(phase, b)
    data = {a: agent_rows(phase, b, a) for a in ("agent_1", "agent_2", "agent_3", "agent_4")}
    res = {}
    for pid, it in items.items():
        s = {a: norm(data[a][pid].get("source_sql_correct")) for a in SRC_AGENTS}
        if it.get("mql") is None:
            t = {a: "NOT_APPLICABLE" for a in TRN_AGENTS}
        else:
            t = {a: norm(data[a][pid].get("mql_translation_correct")) for a in TRN_AGENTS}
        res[pid] = (s, t, {a: data[a][pid] for a in data})
    return res


def needs_adj(s, t):
    vals = list(s.values()) + [x for x in t.values() if x != "NOT_APPLICABLE"]
    return any(v != "YES" for v in vals)


def disputes(phase):
    for b in batches(phase):
        out = []
        for pid, (s, t, rows) in votes(phase, b).items():
            if needs_adj(s, t):
                out.append({"panel_id": pid, "source_votes": s, "translation_votes": t,
                            "reviewer_outputs": {a: {k: rows[a].get(k) for k in ("source_sql_correct", "mql_translation_correct",
                                                                                 "confidence", "source_error_category",
                                                                                 "translation_error_category", "evidence",
                                                                                 "execution_evidence", "counterexample")}
                                                 for a in ("agent_1", "agent_2", "agent_3")}})
        p = PANEL / phase / "work" / b / "dispute_list.json"
        p.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
        print(b, len(out), "disputed of", len(pkg(phase, b)))


def final(phase):
    rows = []
    for b in batches(phase):
        adj_p = PANEL / phase / "work" / b / "agent_4_adj" / "output.jsonl"
        adj = agent_rows(phase, b, "agent_4_adj") if adj_p.exists() else {}
        for pid, (s, t, raw) in votes(phase, b).items():
            if not needs_adj(s, t):
                fs, ft, basis = "YES", ("YES" if t["agent_2"] != "NOT_APPLICABLE" else "NOT_APPLICABLE"), "unanimous_yes"
            elif pid in adj:
                fs = norm(adj[pid].get("source_sql_correct"))
                ft = "NOT_APPLICABLE" if t["agent_2"] == "NOT_APPLICABLE" else norm(adj[pid].get("mql_translation_correct"))
                basis = "agent4_adjudication"
            else:
                fs, ft, basis = "MISSING", "MISSING", "adjudication_missing"
            a = adj.get(pid, {})
            rows.append({"panel_id": pid, "batch": b, "source_votes": s, "translation_votes": t,
                         "final_source": fs, "final_translation": ft, "basis": basis,
                         "src_category": a.get("source_error_category") if basis != "unanimous_yes" else "none",
                         "trn_category": a.get("translation_error_category") if basis != "unanimous_yes" else "none",
                         "evidence_sufficient": a.get("evidence_sufficient"), "label": LABEL})
    p = PANEL / phase / "final_labels.jsonl"
    with open(p, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(phase, len(rows), Counter((r["final_source"], r["final_translation"]) for r in rows).most_common(8))


def kappa(pairs):
    cats = sorted({x for p in pairs for x in p})
    n = len(pairs)
    if n == 0:
        return None
    po = sum(a == b for a, b in pairs) / n
    pa = Counter(a for a, _ in pairs)
    pb = Counter(b for _, b in pairs)
    pe = sum(pa[c] * pb[c] for c in cats) / (n * n)
    return 1.0 if pe == 1 else round((po - pe) / (1 - pe), 4)


def report():
    pr = json.loads((PANEL / "prereg.json").read_text(encoding="utf-8"))
    th = pr["thresholds"]
    K = key("main")
    F = {r["panel_id"]: r for r in rd(PANEL / "main/final_labels.jsonl")}
    if set(K) != set(F) or any(r["basis"] == "adjudication_missing" for r in F.values()):
        raise SystemExit("final labels incomplete")
    by = defaultdict(list)
    for pid, k in K.items():
        by[k["origin"]["kind"]].append(pid)
    rep = {"written_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "label": LABEL, "prereg_sha256": pr["_sha256"]}
    # ---- calibration
    cal = {}
    for dim, kind, fkey in (("source", "planted_source", "final_source"), ("translation", "planted_translation", "final_translation")):
        ids = by[kind]
        caught = [p for p in ids if F[p][fkey] == "NO"]
        caught_or_unsure = [p for p in ids if F[p][fkey] in ("NO", "UNSURE")]
        cls = defaultdict(lambda: [0, 0])
        for p in ids:
            c = K[p]["origin"]["class"]
            cls[c][1] += 1
            cls[c][0] += F[p][fkey] == "NO"
        other = "final_translation" if dim == "source" else "final_source"
        cal[dim] = {"n": len(ids), "caught_NO": len(caught), "sensitivity": round(len(caught) / len(ids), 4) if ids else None,
                    "wilson95": wilson(len(caught), len(ids)), "caught_NO_or_UNSURE": len(caught_or_unsure),
                    "per_class": {c: {"caught": v[0], "n": v[1], "rate": round(v[0] / v[1], 3)} for c, v in sorted(cls.items())},
                    "other_dimension_flagged_on_these_items": sum(F[p][other] in ("NO", "UNSURE") for p in ids),
                    "missed_ids": sorted(set(ids) - set(caught))}
        cal[dim]["pass"] = (len(caught) >= th["min_caught_per_dimension"] and len(ids) >= th["min_planted_per_dimension"]
                            and all(v[0] / v[1] >= th["min_class_rate"] for v in cls.values()))
    clean = by["clean_control"]
    cal["clean_controls"] = {"n": len(clean),
                             "false_positive_source_NO": sum(F[p]["final_source"] == "NO" for p in clean),
                             "false_positive_translation_NO": sum(F[p]["final_translation"] == "NO" for p in clean),
                             "quarantined_any_not_YES": sum(F[p]["final_source"] != "YES" or F[p]["final_translation"] not in ("YES", "NOT_APPLICABLE") for p in clean),
                             "note": "clean = accepted in both dimensions by the run01 AI audit and by verifier 3.1; not ground truth"}
    for k2 in ("false_positive_source_NO", "false_positive_translation_NO", "quarantined_any_not_YES"):
        cal["clean_controls"][k2 + "_wilson95"] = wilson(cal["clean_controls"][k2], len(clean))
    cal["pass"] = cal["source"]["pass"] and cal["translation"]["pass"]
    cal["scope"] = "sensitivity measures the planted defect classes only; natural defects outside them are unmeasured"
    rep["calibration"] = cal
    # ---- stability
    st = {}
    sk_p = ORCH / "panel_key_stability.jsonl"
    if sk_p.exists() and (PANEL / "stability/final_labels.jsonl").exists():
        SK = {k["panel_id"]: k for k in rd(sk_p)}
        SF = {r["panel_id"]: r for r in rd(PANEL / "stability/final_labels.jsonl")}
        for dim, fk in (("source", "final_source"), ("translation", "final_translation")):
            pairs = [(F[SK[s]["main_panel_id"]][fk], SF[s][fk]) for s in SK
                     if s in SF and F[SK[s]["main_panel_id"]][fk] != "NOT_APPLICABLE"]
            st[dim] = {"n": len(pairs), "agreement": round(sum(a == b for a, b in pairs) / len(pairs), 4) if pairs else None,
                       "cohen_kappa": kappa(pairs), "disagreements": sum(a != b for a, b in pairs)}
        st["pass"] = all(st[d]["cohen_kappa"] is not None and st[d]["cohen_kappa"] >= th["min_kappa"] for d in ("source", "translation"))
    else:
        st = {"pass": False, "status": "stability re-run not available"}
    rep["stability"] = st
    usable = cal["pass"] and st.get("pass", False)
    rep["panel_labels_usable"] = usable
    # ---- G-AIR
    g = by["gair"]
    nos = sum(F[p]["final_translation"] == "NO" for p in g)
    uns = sum(F[p]["final_translation"] == "UNSURE" for p in g)
    snos = sum(F[p]["final_source"] == "NO" for p in g)
    suns = sum(F[p]["final_source"] == "UNSURE" for p in g)
    p3_t = Counter(F[p]["trn_category"] for p in g if F[p]["final_translation"] == "NO")
    p3_s = Counter(F[p]["src_category"] for p in g if F[p]["final_source"] == "NO")
    gate_errors = nos + uns
    gair = {"n": len(g), "label": LABEL,
            "translation": {"wrong_NO": nos, "unsure": uns, "wilson95_NO": wilson(nos, len(g)),
                            "wilson95_NO_plus_UNSURE": wilson(gate_errors, len(g)),
                            "P3_categories": dict(p3_t), "P3_pass": all(v < 2 for v in p3_t.values())},
            "source": {"wrong_NO": snos, "unsure": suns, "wilson95_NO": wilson(snos, len(g)),
                       "wilson95_NO_plus_UNSURE": wilson(snos + suns, len(g)), "P3_categories": dict(p3_s),
                       "P3_pass": all(v < 2 for v in p3_s.values()), "threshold": "none (user decision 2026-11-13: report only)"}}
    up = wilson(gate_errors, len(g))[1] if g else None
    gair["translation_gate"] = {"rule": "Wilson95 upper of (NO + UNSURE)/n <= 0.05 AND calibration pass AND stability kappa >= 0.6",
                                "upper": up, "bound_ok": up is not None and up <= th["gair_translation_upper"],
                                "pass": bool(up is not None and up <= th["gair_translation_upper"] and usable)}
    gair["caveat"] = ("AI agreement is not ground truth; a 200-item AI-only sample does not prove a real-world error rate "
                      "below 5%. This is G-AIR, not the frozen human P2 gate, which remains NOT passed.")
    rep["gair"] = gair
    # ---- S9 precision
    prec = {}
    det_items = defaultdict(list)
    for p in by["s9_precision"]:
        for d in K[p]["origin"]["detectors"]:
            det_items[d].append(p)
    for d, ids in sorted(det_items.items()):
        tp = sum(F[p]["final_source"] == "NO" for p in ids)
        un = sum(F[p]["final_source"] == "UNSURE" for p in ids)
        prec[d] = {"sampled": len(ids), "panel_source_NO": tp, "panel_UNSURE": un, "precision": round(tp / len(ids), 4),
                   "wilson95": wilson(tp, len(ids)), "precision_incl_unsure": round((tp + un) / len(ids), 4)}
    rep["s9_precision"] = {"label": LABEL, "per_detector": prec,
                           "note": "precision = share of sampled detector hits the panel judged source-wrong; labels usable only if panel_labels_usable"}
    # ---- S10
    s10 = []
    for p in by["s10_adjudication"]:
        o = K[p]["origin"]
        keep = F[p]["final_source"] == "YES" and F[p]["final_translation"] in ("YES", "NOT_APPLICABLE")
        s10.append({"record_id": o["record_id"], "role": o["role"], "run01_review_id": o["run01_review_id"],
                    "final_source": F[p]["final_source"], "final_translation": F[p]["final_translation"],
                    "panel_decision": "accept" if keep and usable else "quarantine", "basis": F[p]["basis"]})
    rep["s10"] = sorted(s10, key=lambda x: (x["run01_review_id"], x["role"]))
    rep["s9_record_decisions"] = {K[p]["origin"]["record_id"]: ("accept" if usable and F[p]["final_source"] == "YES"
                                                                 and F[p]["final_translation"] in ("YES", "NOT_APPLICABLE")
                                                                 else "quarantine") for p in by["s9_precision"]}
    rep["gair_record_decisions"] = {K[p]["origin"]["record_id"]: {"final_source": F[p]["final_source"],
                                                                   "final_translation": F[p]["final_translation"]} for p in g}
    rep["counts"] = {k: len(v) for k, v in by.items()}
    (PANEL / "panel_report.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"calibration": {d: {k: cal[d][k] for k in ("n", "caught_NO", "wilson95", "pass")} for d in ("source", "translation")},
                      "clean": cal["clean_controls"], "stability": st, "gair": {k: gair[k] for k in ("translation", "source", "translation_gate")},
                      "s9": prec}, indent=1))


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "report":
        report()
    else:
        {"validate": validate, "disputes": disputes, "final": final}[cmd](sys.argv[2])
