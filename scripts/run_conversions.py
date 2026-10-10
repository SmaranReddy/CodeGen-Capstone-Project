"""Batch driver for the Spider 2.0-Lite -> MongoDB/MQL conversions.

Runs one `scripts/conversions/conv_<id>.py` module per example (each exposes
`run(reg)` returning `(status, reason, artifact)`), writes the per-example artifact,
updates the consolidated status manifest, and rebuilds the validated dataset.

Usage:
    python scripts/run_conversions.py local199 local078 ...
    python scripts/run_conversions.py --all          # every conv_*.py present
"""
import importlib
import json
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from conversion_kit import (CONV_DIR, DATASET_PATH, EXISTING_VALIDATED, ROOT, Registry, load_status,
                            manifest_record, mongo_client, pattern_record, save_status,
                            write_conversion)

MODULES_DIR = ROOT / "scripts/conversions"


def discover_modules():
    return {p.stem.replace("conv_", ""): p.stem for p in MODULES_DIR.glob("conv_*.py")}


def normalize_artifact(iid, status, artifact):
    """Guarantee the artifact carries every field the task definition requires."""
    rec = manifest_record(iid)
    pat = pattern_record(iid)
    artifact.setdefault("instance_id", iid)
    artifact.setdefault("db", rec["db"])
    artifact.setdefault("question", rec["question"])
    artifact.setdefault("category", pat["category"])
    artifact.setdefault("category_name", pat["category_name"])
    artifact.setdefault("query_basis", pat["basis"])
    artifact.setdefault("source_semantics",
                        "gold_sql" if pat["basis"] == "gold_sql" else "inferred_reference_sql")
    artifact.setdefault("external_knowledge", rec.get("external_knowledge"))
    artifact.setdefault("expected_result_references", [Path(p).name for p in rec["gold_result_paths"]])
    artifact.setdefault("assumptions", [])
    artifact["final_status"] = status
    return artifact


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    mods = discover_modules()
    if "--all" in sys.argv or not args:
        ids = sorted(mods)
    else:
        ids = args
    status = load_status()
    client = mongo_client()
    try:
        db = client["codegen_capstone_pilot"]
        reg = Registry(db)
        results = {}
        for iid in ids:
            if iid not in mods:
                print(f"[{iid}] no conversion module; skipped")
                continue
            mod = importlib.import_module(f"conversions.{mods[iid]}")
            try:
                st, reason, artifact = mod.run(reg)
            except Exception as e:
                st, reason = "blocked", f"{type(e).__name__}: {e}"
                artifact = {"error": traceback.format_exc()}
                print(f"[{iid}] ERROR {reason}")
            artifact = normalize_artifact(iid, st, artifact)
            path = write_conversion(iid, artifact)
            results[iid] = st
            # update status
            entry = status.get(iid, {"instance_id": iid})
            entry.update({
                "instance_id": iid, "db": manifest_record(iid)["db"],
                "category": pattern_record(iid)["category"],
                "category_name": pattern_record(iid)["category_name"],
                "basis": pattern_record(iid)["basis"],
                "flags": pattern_record(iid)["flags"],
                "status": st, "reason": reason,
                "artifact_path": str(path.relative_to(ROOT)).replace("\\", "/"),
                "source_semantics": artifact["source_semantics"],
                "validation": artifact.get("validation"),
                "newly_converted_this_run": True,
            })
            status[iid] = entry
            print(f"[{iid}] {st} :: {reason}")
        save_status(status)
        print("status counts:", _counts(status))
    finally:
        client.close()


def _counts(status):
    from collections import Counter
    return dict(Counter(v["status"] for v in status.values()))


def rebuild_dataset():
    """Consolidated dataset = passed artifacts (this run) + still-valid existing pilots."""
    rows = []
    conv = CONV_DIR
    for path in sorted(conv.glob("*.json")):
        if path.name.startswith("_"):
            continue
        a = json.loads(path.read_text(encoding="utf-8"))
        if a.get("final_status") == "passed":
            rows.append(_dataset_row(a))
    for iid in EXISTING_VALIDATED:
        p = ROOT / "data/interim/pilot" / f"{iid}.json"
        if not p.exists():
            continue
        a = json.loads(p.read_text(encoding="utf-8"))
        passed = bool(a.get("validation_passed")) or bool(a.get("validation", {}).get("matched_any_gold"))
        if passed:
            rows.append(_dataset_row(a, pilot=True))
    rows.sort(key=lambda r: r["instance_id"])
    DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(DATASET_PATH, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, default=str, ensure_ascii=False) + "\n")
    return rows


def _dataset_row(a, pilot=False):
    iid = a["instance_id"]
    rec = manifest_record(iid)
    pat = pattern_record(iid)
    mql = a.get("mql") or {}
    return {
        "instance_id": iid,
        "db": rec["db"],
        "question": rec["question"],
        "schema": a.get("schema", a.get("source_schema")),
        "query": mql.get("pipeline", mql),
        "query_method": mql.get("method", "aggregate") if isinstance(mql, dict) else "aggregate",
        "query_type": pat.get("category_name"),
        "category": pat.get("category"),
        "query_basis": pat.get("basis"),
        "source_semantics": "gold_sql" if pat.get("basis") == "gold_sql" else "inferred_reference_sql",
        "gold_sql": a.get("gold_sql"),
        "reference_sql": a.get("reference_sql") or a.get("reference_sql_written_by_us"),
        "expected_result_references": [Path(p).name for p in rec["gold_result_paths"]],
        "validation": a.get("validation"),
        "validation_passed": True,
        "from_existing_pilot": pilot,
        "assumptions": a.get("assumptions", []),
    }


if __name__ == "__main__":
    main()
    rows = rebuild_dataset()
    print(f"rebuilt dataset: {len(rows)} validated examples -> {DATASET_PATH}")
