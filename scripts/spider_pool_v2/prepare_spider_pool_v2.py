"""Prepare spider-pool-v2: the audited NL->SQL training source for the NL->MQL pipeline (dataset preparation only).

Source  : spider_pool.zip (pinned sha256), members spider_pool/train.jsonl and spider_pool/val_internal.jsonl ONLY.
          dev.jsonl and rejected.jsonl are never opened.
Evidence: outputs/spider_pool_audit_v1/records/recommended_exclusions.jsonl (pinned sha256): tiers and reasons used verbatim.
          outputs/spider_pool_repro_audit_v1/stage9_empty_table_recheck.json (pinned sha256): source of the one added quarantine row.

Rules (no other filtering, no repair, no re-serialisation of records):
  Tier A  -> excluded
  Tier B  -> excluded
  Tier C  -> quarantined (pending a human decision)
  spider_train_02208 -> quarantined (pending review; added by the reproducibility audit)
  everything else in train / val_internal -> retained, as the ORIGINAL BYTES of its source line, in source order.

Outputs (refuses to run if the output directory already exists; writes to a temp dir, then renames):
  train.jsonl, val_internal.jsonl  retained records, byte-identical lines
  quarantine.jsonl                 quarantined records, byte-identical lines (reasons are in exclusions.jsonl)
  exclusions.jsonl                 one entry per removed record (excluded AND quarantined): disposition, tier, reason, provenance
  manifest.json                    source hashes, exclusion provenance, rules, counts, output hashes

Usage: python -I -B scripts/spider_pool_v2/prepare_spider_pool_v2.py [--out data/processed/spider_pool_v2]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import zipfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ZIP = ROOT / "spider_pool.zip"
ZIP_SHA = "31848241861988ebb0619e1802c38fb5d90cfb62f818a65a53cf601340888a6b"
EXCL = ROOT / "outputs/spider_pool_audit_v1/records/recommended_exclusions.jsonl"
EXCL_SHA = "008cae9cd9325a456815b632adb3f0f246f2b2f9033302b0092450807590d5b3"
RECHECK = ROOT / "outputs/spider_pool_repro_audit_v1/stage9_empty_table_recheck.json"
RECHECK_SHA = "6b633c3e46d518ceb79c8b5757d5093250efc611131f37fffde9b6086d13a762"
MEMBERS = {"train": "spider_pool/train.jsonl", "val_internal": "spider_pool/val_internal.jsonl"}
TIER_RULE = {"A_confirmed_defect": "excluded", "B_engine_or_row_choice_dependent": "excluded", "C_review_before_use": "quarantined"}
ADDED_QUARANTINE = {
    "spider_train_02208": {
        "tier": "C_review_added_by_reproducibility_audit",
        "reason": "same SQL as Tier C spider_train_02207 (reads empty table pitstops); missed by the earlier empty-table screen; "
                  "pending review (docs/spider_pool_reproducibility_audit.md section 5)",
        "evidence": "outputs/spider_pool_repro_audit_v1/stage9_empty_table_recheck.json",
    }
}
EVIDENCE_DOCS = ["docs/spider_pool_independent_audit.md", "docs/spider_pool_reproducibility_audit.md",
                 "outputs/spider_pool_repro_audit_v1/audit_summary.json"]


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def fsha(p: Path) -> str:
    return sha(p.read_bytes())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "data/processed/spider_pool_v2"))
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"refusing to overwrite existing output version: {out}")

    for p, want in ((ZIP, ZIP_SHA), (EXCL, EXCL_SHA), (RECHECK, RECHECK_SHA)):
        got = fsha(p)
        if got != want:
            raise SystemExit(f"pinned input changed: {p} {got} != {want}")

    z = zipfile.ZipFile(ZIP)
    zman = json.loads(z.read("spider_pool/manifest.json"))
    src = {}
    for split, member in MEMBERS.items():
        raw = z.read(member)
        if sha(raw) != zman["files_sha256"][member.split("/")[-1]]:
            raise SystemExit(f"zip member hash differs from zip manifest: {member}")
        if b"\r" in raw or not raw.endswith(b"\n"):
            raise SystemExit(f"unexpected line endings in {member}")
        src[split] = (member, raw, raw[:-1].split(b"\n"))

    # tiers, verbatim
    tiers = {}
    for line in EXCL.read_text(encoding="utf-8").splitlines():
        x = json.loads(line)
        if x["id"] in tiers:
            raise SystemExit(f"duplicate id in exclusion list: {x['id']}")
        tiers[x["id"]] = {"tier": x["tier"], "reason": x["reason"], "disposition": TIER_RULE[x["tier"]],
                          "listed_split": x["split"], "listed_db_id": x["db_id"], "listed_group": x["group"],
                          "evidence": "outputs/spider_pool_audit_v1/records/recommended_exclusions.jsonl"}
    rc = json.loads(RECHECK.read_text(encoding="utf-8"))
    for i, a in ADDED_QUARANTINE.items():
        if i in tiers:
            raise SystemExit(f"{i} already tiered")
        if i not in rc["missed_by_earlier_screen"]:
            raise SystemExit(f"{i} not in recheck evidence")
        tiers[i] = {**a, "disposition": "quarantined"}

    kept = {"train": [], "val_internal": []}
    quarantine, exclusions, seen = [], [], set()
    for split, (member, _, lines) in src.items():
        for n, line in enumerate(lines, 1):
            r = json.loads(line)
            i = r["id"]
            if i in seen:
                raise SystemExit(f"duplicate id in source: {i}")
            seen.add(i)
            if r["split"] != split or i.startswith(("spider_dev_", "spider_test_")):
                raise SystemExit(f"unexpected record in {member}: {i}")
            t = tiers.get(i)
            if t is None:
                kept[split].append(line)
                continue
            for k in ("listed_split", "listed_db_id", "listed_group"):
                if k in t and t[k] != r[k.replace("listed_", "")]:
                    raise SystemExit(f"exclusion list disagrees with source on {k} for {i}")
            exclusions.append({"id": i, "split": split, "db_id": r["db_id"], "group": r["group"], "disposition": t["disposition"],
                               "tier": t["tier"], "reason": t["reason"], "evidence": t["evidence"],
                               "source": {"zip_member": member, "line": n, "line_sha256": sha(line)}})
            if t["disposition"] == "quarantined":
                quarantine.append(line)
    missing = sorted(set(tiers) - seen)
    if missing:
        raise SystemExit(f"exclusion ids not found in train/val: {missing}")

    tmp = out.with_name(out.name + ".tmp_build")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)

    def write_lines(name, lines):
        b = b"".join(l + b"\n" for l in lines)
        (tmp / name).write_bytes(b)
        return {"sha256": sha(b), "records": len(lines), "bytes": len(b)}

    def write_json_lines(name, rows):
        b = "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows).encode("utf-8")
        (tmp / name).write_bytes(b)
        return {"sha256": sha(b), "records": len(rows), "bytes": len(b)}

    files = {
        "train.jsonl": write_lines("train.jsonl", kept["train"]),
        "val_internal.jsonl": write_lines("val_internal.jsonl", kept["val_internal"]),
        "quarantine.jsonl": write_lines("quarantine.jsonl", quarantine),
        "exclusions.jsonl": write_json_lines("exclusions.jsonl", exclusions),
    }
    cnt = Counter((e["split"], e["disposition"]) for e in exclusions)
    tcnt = Counter((e["split"], e["tier"]) for e in exclusions)
    counts = {s: {"source": len(src[s][2]), "excluded": cnt[(s, "excluded")], "quarantined": cnt[(s, "quarantined")],
                  "retained": len(kept[s])} for s in src}
    for s, c in counts.items():
        if c["source"] != c["excluded"] + c["quarantined"] + c["retained"]:
            raise SystemExit(f"count does not reconcile for {s}")
    manifest = {
        "version": "spider-pool-v2",
        "purpose": "audited NL->SQL training source for SQL->MQL conversion; SQL labels are NOT certified correct",
        "builder": {"path": "scripts/spider_pool_v2/prepare_spider_pool_v2.py", "sha256": fsha(Path(__file__)),
                    "python": sys.version.split()[0], "command": "python -I -B scripts/spider_pool_v2/prepare_spider_pool_v2.py"},
        "source": {
            "zip": {"path": "spider_pool.zip", "sha256": ZIP_SHA, "version": zman["version"], "zip_manifest_code_sha256": zman["code_sha256"]},
            "members_used": {m: zman["files_sha256"][m.split("/")[-1]] for m in MEMBERS.values()},
            "members_not_opened": ["spider_pool/dev.jsonl", "spider_pool/rejected.jsonl", "spider_pool/README.md"],
            "zip_manifest_sha256": sha(z.read("spider_pool/manifest.json")),
        },
        "exclusion_provenance": {
            "tier_list": {"path": str(EXCL.relative_to(ROOT)).replace("\\", "/"), "sha256": EXCL_SHA, "entries": len(tiers) - len(ADDED_QUARANTINE)},
            "added_quarantine": {"path": str(RECHECK.relative_to(ROOT)).replace("\\", "/"), "sha256": RECHECK_SHA, "ids": sorted(ADDED_QUARANTINE)},
            "audit_documents_sha256_at_build": {d: fsha(ROOT / d) for d in EVIDENCE_DOCS},
        },
        "rules": {"A_confirmed_defect": "excluded", "B_engine_or_row_choice_dependent": "excluded",
                  "C_review_before_use": "quarantined", "spider_train_02208": "quarantined",
                  "retained_records": "original source line bytes, source order, no field changed"},
        "counts": counts,
        "counts_by_tier": {f"{s}|{t}": v for (s, t), v in sorted(tcnt.items())},
        "files": files,
        "line_endings": "LF",
        "protected_data": "no Spider dev, Spider test or held-out MQL record; dev.jsonl and rejected.jsonl never opened",
        "license": "CC BY-SA 4.0 (Spider)",
    }
    (tmp / "manifest.json").write_bytes((json.dumps(manifest, indent=1, ensure_ascii=False) + "\n").encode("utf-8"))
    os.replace(tmp, out) if not out.exists() else (_ for _ in ()).throw(SystemExit("output appeared during build"))
    print(json.dumps({"counts": counts, "files": {k: v["records"] for k, v in files.items()}}, indent=1))


if __name__ == "__main__":
    main()
