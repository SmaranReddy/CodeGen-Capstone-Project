"""spider-pool-v3 (plan §E.2, step S11): spider_pool_v2 minus the new semantic-quarantine tiers. Never edits v2.

    python -I -B scripts/spider_pool_v3/prepare_spider_pool_v3.py build

Rules
  * Inputs referenced by sha256: data/processed/spider_pool_v2/{train,val_internal,quarantine,exclusions}.jsonl +
    manifest.json, outputs/source_label_audit_v1/{hits.jsonl,finding_siblings.json}, the detector pre-registration,
    and the AI panel report (outputs/ai_review_panel_v1/panel_report.json).
  * Retained records keep their ORIGINAL line bytes and source order. No SQL or question is changed; no corrected
    label is added (candidate corrections live only in data/interim/source_label_adjudication_v1.jsonl).
  * New tiers (exclusions_v3.jsonl, one row per removed record, with every reason):
      A_semantic_defect_adjudicated  the AI panel (usable: calibration + stability passed) judged the source SQL wrong
                                     (final NO) AND the record is an A-candidate (R084/R101/R111, their siblings, or a
                                     D1a hit). The run01 audit alone never places a record in tier A.
      C_semantic_review              every other run01 finding / sibling, every detector quarantine hit the panel did
                                     not clearly accept, every D3 hit ('all X' policy = quarantine, user 2026-11-13),
                                     and every G-AIR record whose final source label is not YES. Anything not clearly
                                     accepted stays here (asymmetric rule).
  * v2's own quarantine.jsonl / exclusions.jsonl are carried over byte-identical.
"""
import hashlib
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
V2 = ROOT / "data/processed/spider_pool_v2"
V3 = ROOT / "data/processed/spider_pool_v3"
AUD = ROOT / "outputs/source_label_audit_v1"
PANEL = ROOT / "outputs/ai_review_panel_v1/panel_report.json"
A_CANDIDATE_FINDINGS = {"R084", "R101", "R111"}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def lines(p):
    return [x for x in Path(p).read_bytes().split(b"\n") if x.strip()]


def build():
    if (V3 / "manifest.json").exists():
        raise SystemExit("refusing to overwrite spider_pool_v3")
    v2man = json.loads((V2 / "manifest.json").read_text(encoding="utf-8"))
    for f, meta in v2man["files"].items():
        if sha(V2 / f) != meta["sha256"]:
            raise SystemExit(f"spider_pool_v2/{f} differs from its manifest: stop")
    panel = json.loads(PANEL.read_text(encoding="utf-8"))
    usable = bool(panel["panel_labels_usable"])
    sib = json.loads((AUD / "finding_siblings.json").read_text(encoding="utf-8"))
    reasons = defaultdict(list)
    a_cand = set()
    for x in sib:
        ids = [x["finding_id"]] + [s["id"] for s in x["siblings"]]
        for i in ids:
            role = "finding" if i == x["finding_id"] else "sibling"
            reasons[i].append(f"run01_{role}:{x['review_id']}({x['confidence']})")
            if x["review_id"] in A_CANDIDATE_FINDINGS:
                a_cand.add(i)
    hits = defaultdict(set)
    for h in (json.loads(l) for l in lines(AUD / "hits.jsonl")):
        if h["action"] == "quarantine":
            hits[h["id"]].add(h["subclass"])
    for i, subs in hits.items():
        reasons[i].append("detector:" + ",".join(sorted(subs)))
        if "D1a" in subs:
            a_cand.add(i)
    # panel outcomes (used only if the panel's calibration and stability passed)
    panel_src = {}
    for x in panel["s10"]:
        panel_src[x["record_id"]] = (x["final_source"], x["final_translation"], "s10")
    s9 = panel["s9_record_decisions"]
    for rid, d in panel["gair_record_decisions"].items():
        if d["final_source"] != "YES":
            reasons[rid].append(f"gair_final_source:{d['final_source']}")
        panel_src.setdefault(rid, (d["final_source"], d["final_translation"], "gair"))
    s9_final = {}
    if (ROOT / "outputs/ai_review_panel_v1/main/final_labels.jsonl").exists():
        key = {json.loads(l)["panel_id"]: json.loads(l) for l in lines(ROOT / "outputs/ai_review_panel_v1/_orchestrator_only/panel_key_main.jsonl")}
        for l in lines(ROOT / "outputs/ai_review_panel_v1/main/final_labels.jsonl"):
            r = json.loads(l)
            o = key[r["panel_id"]]["origin"]
            if o["kind"] == "s9_precision":
                s9_final[o["record_id"]] = (r["final_source"], r["final_translation"], "s9")
    for rid, v in s9_final.items():
        panel_src.setdefault(rid, v)
    tier = {}
    for i, rs in reasons.items():
        ps = panel_src.get(i)
        cleared = (usable and ps is not None and ps[0] == "YES" and ps[1] in ("YES", "NOT_APPLICABLE")
                   and not any(r.startswith("detector:") and ("D3a" in r or "D3b" in r) for r in rs)
                   and not any(r.startswith("gair_final_source") for r in rs))
        if ps is not None:
            rs.append(f"panel:{ps[2]}:source={ps[0]},translation={ps[1]}" + ("" if usable else " (panel not usable)"))
        if cleared:
            rs.append("cleared_by_panel")
            continue
        if usable and ps is not None and ps[0] == "NO" and i in a_cand:
            tier[i] = "A_semantic_defect_adjudicated"
        else:
            tier[i] = "C_semantic_review"
    V3.mkdir(parents=True, exist_ok=True)
    out_counts, excl_rows = {}, []
    for split in ("train.jsonl", "val_internal.jsonl"):
        keep = []
        n = 0
        for raw in lines(V2 / split):
            n += 1
            r = json.loads(raw)
            if r["id"] in tier:
                excl_rows.append({"id": r["id"], "db_id": r["db_id"], "from_split": split.split(".")[0], "tier": tier[r["id"]],
                                  "reasons": reasons[r["id"]]})
            else:
                keep.append(raw)
        (V3 / split).write_bytes(b"\n".join(keep) + b"\n")
        out_counts[split] = {"v2": n, "v3": len(keep), "removed": n - len(keep)}
    for f in ("quarantine.jsonl", "exclusions.jsonl"):
        (V3 / f).write_bytes((V2 / f).read_bytes())
    with open(V3 / "exclusions_v3.jsonl", "w", encoding="utf-8") as fh:
        for e in excl_rows:
            fh.write(json.dumps(e, ensure_ascii=False) + "\n")
    not_in_pool = sorted(i for i in tier if i not in {e["id"] for e in excl_rows})
    cleared = sorted(i for i, rs in reasons.items() if "cleared_by_panel" in rs)
    man = {"version": "spider-pool-v3", "written_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "builder": {"path": "scripts/spider_pool_v3/prepare_spider_pool_v3.py", "sha256": sha(__file__),
                       "command": "python -I -B scripts/spider_pool_v3/prepare_spider_pool_v3.py build"},
           "inputs": {"spider_pool_v2_manifest_sha256": sha(V2 / "manifest.json"),
                      "spider_pool_v2_files": {f: sha(V2 / f) for f in ("train.jsonl", "val_internal.jsonl", "quarantine.jsonl", "exclusions.jsonl")},
                      "detector_prereg_sha256": sha(ROOT / "data/interim/source_label_audit_v1/prereg.json"),
                      "hits_sha256": sha(AUD / "hits.jsonl"), "finding_siblings_sha256": sha(AUD / "finding_siblings.json"),
                      "panel_report_sha256": sha(PANEL), "panel_labels_usable": usable,
                      "type_contract_v1_sha256": sha(ROOT / "data/interim/finetune_v2/type_contract_v1.json")},
           "counts": out_counts, "tiers": dict(Counter(tier.values())),
           "tiers_by_split": dict(Counter((e["from_split"], e["tier"]) for e in excl_rows).items()).__repr__(),
           "cleared_by_panel": cleared, "quarantine_decisions_for_ids_not_in_pool": not_in_pool,
           "files": {f: {"sha256": sha(V3 / f), "records": len(lines(V3 / f))}
                     for f in ("train.jsonl", "val_internal.jsonl", "quarantine.jsonl", "exclusions.jsonl", "exclusions_v3.jsonl")},
           "rules": __doc__.split("Rules", 1)[1].strip(),
           "review": "AI-reviewed (multi-agent panel) where the panel was used; never human-reviewed"}
    (V3 / "manifest.json").write_text(json.dumps(man, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: man[k] for k in ("counts", "tiers", "tiers_by_split")}, indent=1),
          "\ncleared:", len(cleared), "not in pool:", not_in_pool)


if __name__ == "__main__":
    {"build": build}[sys.argv[1]]()
