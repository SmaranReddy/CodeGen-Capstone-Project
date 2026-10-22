"""Database-disjoint train/validation/test split of the EXPANDED 455-example benchmark.

The database -> partition assignment is INHERITED from the v1 split (data/splits/
split_manifest.json, seed 20261015, draw 1003). The expanded set has the same 20 dev
databases, so every expanded example goes to the partition of its db_id. Consequences:
  * database-disjoint by construction (no schema, table or collection is shared);
  * every v1 item keeps its v1 partition (the v1 split is a subset of this one);
  * no new randomness and no model output or score is used.
The v1 split files are never touched; this script writes data/splits_expanded/.

RETROSPECTIVE: the zero-shot baselines (Qwen-3B/7B, gpt-oss-120b) were generated on all 455
examples BEFORE this split existed. Per-partition scores of those runs are not untouched
held-out results.

Usage:
    python scripts/spider1/make_splits_expanded.py            # verify (exit 1 on mismatch)
    python scripts/spider1/make_splits_expanded.py --write    # create (refuses to overwrite)
"""
import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "data/processed/spider1_nl_mql_expanded.jsonl"
V1_MANIFEST = ROOT / "data/splits/split_manifest.json"
OUT_DIR = ROOT / "data/splits_expanded"
PARTITIONS = ("train", "validation", "test")
HARDNESS = ("easy", "medium", "hard", "extra")
RETROSPECTIVE_NOTE = (
    "Retrospective split: zero-shot Qwen2.5-Coder-3B (217/455), Qwen2.5-Coder-7B (287/455) "
    "and gpt-oss-120b (385/455) were evaluated on all 455 examples BEFORE this split existed. "
    "Per-partition scores of those runs are not an untouched held-out evaluation. Week 3 RAG "
    "uses train for the retrieval corpus, validation for development, and keeps test unused.")


def load_source(path=SOURCE):
    raw = path.read_bytes()
    lines = [ln for ln in raw.splitlines(keepends=True) if ln.strip()]
    return raw, [(ln, json.loads(ln)) for ln in lines]


def v1_assignment(path=V1_MANIFEST):
    m = json.loads(path.read_text(encoding="utf-8"))
    return {db: v["partition"] for db, v in m["database_assignments"].items()}, m


def build(source=SOURCE, v1_manifest=V1_MANIFEST):
    raw, items = load_source(source)
    rows = [r for _, r in items]
    db_part, v1 = v1_assignment(v1_manifest)
    missing = sorted({r["db_id"] for r in rows} - set(db_part))
    if missing:
        raise ValueError(f"databases without a v1 assignment: {missing}")
    files = {p: b"".join(ln for ln, r in items if db_part[r["db_id"]] == p) for p in PARTITIONS}
    parts = {}
    for p in PARTITIONS:
        sub = [r for r in rows if db_part[r["db_id"]] == p]
        parts[p] = {
            "file": f"data/splits_expanded/{p}.jsonl",
            "sha256": hashlib.sha256(files[p]).hexdigest(),
            "n_examples": len(sub),
            "fraction": round(len(sub) / len(rows), 4),
            "databases": sorted({r["db_id"] for r in sub}),
            "hardness": {h: sum(r["hardness"] == h for r in sub) for h in HARDNESS},
            "n_in_v1": sum(bool(r.get("in_v1_benchmark")) for r in sub),
            "ids": [r["id"] for r in sub],
        }
    size = Counter(r["db_id"] for r in rows)
    manifest = {
        "dataset": "Spider 1.0 dev -> MQL, expanded benchmark (455)",
        "source": {"path": "data/processed/spider1_nl_mql_expanded.jsonl",
                   "sha256": hashlib.sha256(raw).hexdigest(), "n_examples": len(rows),
                   "n_databases": len(size)},
        "methodology": {
            "type": "database-disjoint (grouped by db_id), assignment inherited from v1",
            "script": "scripts/spider1/make_splits_expanded.py",
            "command": "python scripts/spider1/make_splits_expanded.py --write",
            "inherited_from": {"path": "data/splits/split_manifest.json",
                               "sha256": hashlib.sha256(v1_manifest.read_bytes()).hexdigest(),
                               "seed": v1["seed"], "chosen_draw": v1["methodology"]["chosen_draw"]},
            "rationale": ("same 20 databases as v1; inheriting keeps every v1 item in its v1 "
                          "partition and needs no new randomness"),
            "selection_inputs": "db_id only; no model outputs or scores",
            "file_format": "source JSONL lines copied byte-for-byte, source order kept",
        },
        "database_disjoint": True,
        "partition_counts": {p: parts[p]["n_examples"] for p in PARTITIONS},
        "database_assignments": {db: {"partition": db_part[db], "n_examples": size[db]}
                                 for db in sorted(size)},
        "partitions": parts,
        "examples": [{"id": r["id"], "db_id": r["db_id"], "partition": db_part[r["db_id"]]}
                     for r in rows],
        "retrospective_evaluation_note": RETROSPECTIVE_NOTE,
    }
    return manifest, files


def manifest_bytes(manifest):
    return (json.dumps(manifest, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


def partition_of(out_dir=OUT_DIR):
    """{example id: partition} from the written manifest."""
    m = json.loads((out_dir / "split_manifest.json").read_text(encoding="utf-8"))
    return {e["id"]: e["partition"] for e in m["examples"]}


def db_partition(out_dir=OUT_DIR):
    m = json.loads((out_dir / "split_manifest.json").read_text(encoding="utf-8"))
    return {db: v["partition"] for db, v in m["database_assignments"].items()}


def verify(out_dir=OUT_DIR, source=SOURCE, v1_manifest=V1_MANIFEST):
    errs = []
    mpath = out_dir / "split_manifest.json"
    if not mpath.exists():
        return [f"missing {mpath}"]
    manifest, files = build(source, v1_manifest)
    if mpath.read_bytes() != manifest_bytes(manifest):
        errs.append("split_manifest.json differs from the re-derived manifest")
    for p in PARTITIONS:
        f = out_dir / f"{p}.jsonl"
        if not f.exists() or f.read_bytes() != files[p]:
            errs.append(f"{f.name} differs from the re-derived split")
    v1 = json.loads(v1_manifest.read_text(encoding="utf-8"))
    exp = {e["id"]: e["partition"] for e in manifest["examples"]}
    for e in v1["examples"]:
        if exp.get(e["id"]) != e["partition"]:
            errs.append(f"v1 item {e['id']} changed partition")
    return errs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    if a.write:
        existing = [p for p in [*(OUT_DIR / f"{q}.jsonl" for q in PARTITIONS),
                                OUT_DIR / "split_manifest.json"] if p.exists()]
        if existing:
            sys.exit(f"refusing to overwrite existing split files: {[p.name for p in existing]}")
        manifest, files = build()
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        for p in PARTITIONS:
            (OUT_DIR / f"{p}.jsonl").write_bytes(files[p])
        (OUT_DIR / "split_manifest.json").write_bytes(manifest_bytes(manifest))
        print("wrote", OUT_DIR, manifest["partition_counts"])
    errs = verify()
    for e in errs:
        print("FAIL:", e)
    if errs:
        sys.exit(1)
    m = json.loads((OUT_DIR / "split_manifest.json").read_text(encoding="utf-8"))
    print("OK", m["partition_counts"], "database-disjoint; inherited from v1 split")


if __name__ == "__main__":
    main()
