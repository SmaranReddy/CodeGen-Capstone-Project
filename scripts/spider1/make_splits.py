"""Database-disjoint train/validation/test split of the 279-example Spider 1.0 NL -> MQL dataset.

Every example of a Spider `db_id` goes to one partition, so no database (schema, table or
collection names) is shared across partitions. Method (fully determined by --seed/--draws):

  1. dbs = sorted(distinct db_id)                       (sorted: independent of PYTHONHASHSEED)
  2. rng = random.Random(seed); for each of `draws` draws: rng.shuffle(dbs), then walk the
     shuffled list and put each db in validation while validation < its target, then in test
     while test < its target, else in train. Targets = round(15% of n) for validation and test.
  3. Keep the draw with the smallest (count deviation, hardness deviation); ties -> earliest.
     count deviation = |n_val - target| + |n_test - target|; hardness deviation = sum over
     partitions and hardness levels of |share in partition - share overall|. Hardness is the
     official Spider label; no model output or score is used.

Split files hold the source lines byte-for-byte (the dataset is never re-serialised or changed).

The split is RETROSPECTIVE: all models were already evaluated on all 279 examples, so per-split
scores of those runs are not an untouched held-out evaluation.

Usage:
    python scripts/spider1/make_splits.py            # verify existing split (exit 1 on mismatch)
    python scripts/spider1/make_splits.py --write    # create the split (refuses to overwrite)
"""
import argparse
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "data/processed/spider1_nl_mql.jsonl"
OUT_DIR = ROOT / "data/splits"
PARTITIONS = ("train", "validation", "test")
SEED = 20261015
DRAWS = 10000
FRACTIONS = {"train": 0.70, "validation": 0.15, "test": 0.15}
HARDNESS = ("easy", "medium", "hard", "extra")
RETROSPECTIVE_NOTE = (
    "Retrospective split: Qwen2.5-Coder-3B (132/279), Qwen2.5-Coder-7B (181/279) and "
    "gpt-oss-120b (232/279) were evaluated on all 279 examples BEFORE this split existed. "
    "No model was trained or tuned on any partition, but previously reported results are "
    "not an untouched held-out evaluation, and per-partition scores of those runs must not "
    "be presented as one. The split is for future training/tuning/selection work."
)


def load_source(path=SOURCE):
    """-> (raw bytes, [(raw line bytes incl. terminator, parsed record)])."""
    raw = path.read_bytes()
    lines = [ln for ln in raw.splitlines(keepends=True) if ln.strip()]
    return raw, [(ln, json.loads(ln)) for ln in lines]


def targets(n):
    val = test = round(n * FRACTIONS["validation"])
    return {"train": n - val - test, "validation": val, "test": test}


def _hardness_dev(rows, db_part):
    overall = Counter(r["hardness"] for r in rows)
    dev = 0.0
    for p in PARTITIONS:
        sub = Counter(r["hardness"] for r in rows if db_part[r["db_id"]] == p)
        n = sum(sub.values())
        if n:
            dev += sum(abs(sub[h] / n - overall[h] / len(rows)) for h in HARDNESS)
    return dev


def assign(rows, seed=SEED, draws=DRAWS):
    """rows: parsed records -> ({db_id: partition}, info). Deterministic for (rows, seed, draws)."""
    size = Counter(r["db_id"] for r in rows)
    dbs = sorted(size)
    tgt = targets(len(rows))
    rng = random.Random(seed)
    best, best_key, best_draw = None, None, None
    for d in range(draws):
        rng.shuffle(dbs)
        cnt = {p: 0 for p in PARTITIONS}
        db_part = {}
        for db in dbs:
            p = next((q for q in ("validation", "test") if cnt[q] < tgt[q]), "train")
            db_part[db] = p
            cnt[p] += size[db]
        count_dev = abs(cnt["validation"] - tgt["validation"]) + abs(cnt["test"] - tgt["test"])
        key = (count_dev, round(_hardness_dev(rows, db_part), 12))
        if best_key is None or key < best_key:
            best, best_key, best_draw = dict(db_part), key, d
    return best, {"targets": tgt, "count_deviation": best_key[0],
                  "hardness_deviation": best_key[1], "chosen_draw": best_draw}


def build(seed=SEED, draws=DRAWS, source=SOURCE):
    """-> (manifest dict, {partition: bytes of the split file})."""
    raw, items = load_source(source)
    rows = [r for _, r in items]
    db_part, info = assign(rows, seed, draws)
    files = {p: b"".join(ln for ln, r in items if db_part[r["db_id"]] == p) for p in PARTITIONS}
    parts = {}
    for p in PARTITIONS:
        sub = [r for r in rows if db_part[r["db_id"]] == p]
        parts[p] = {
            "file": f"data/splits/{p}.jsonl",
            "sha256": hashlib.sha256(files[p]).hexdigest(),
            "n_examples": len(sub),
            "fraction": round(len(sub) / len(rows), 4),
            "databases": sorted({r["db_id"] for r in sub}),
            "hardness": {h: sum(r["hardness"] == h for r in sub) for h in HARDNESS},
            "ids": [r["id"] for r in sub],
        }
    size = Counter(r["db_id"] for r in rows)
    manifest = {
        "dataset": "Spider 1.0 dev -> MQL (validated)",
        "source": {"path": "data/processed/spider1_nl_mql.jsonl",
                   "sha256": hashlib.sha256(raw).hexdigest(), "n_examples": len(rows),
                   "n_databases": len(size)},
        "seed": seed,
        "methodology": {
            "type": "database-disjoint (grouped by db_id)",
            "script": "scripts/spider1/make_splits.py",
            "command": f"python scripts/spider1/make_splits.py --write --seed {seed} --draws {draws}",
            "target_fractions": FRACTIONS,
            "target_counts": info["targets"],
            "draws": draws,
            "procedure": (
                "dbs = sorted(distinct db_id); rng = random.Random(seed); for each draw: "
                "rng.shuffle(dbs), assign dbs in shuffled order to validation while its count "
                "< target, then to test while its count < target, else train; keep the draw "
                "minimising (|n_val-target|+|n_test-target|, hardness L1 deviation), ties -> "
                "earliest draw"),
            "chosen_draw": info["chosen_draw"],
            "count_deviation": info["count_deviation"],
            "hardness_deviation": info["hardness_deviation"],
            "selection_inputs": "db_id and official Spider hardness only; no model outputs or scores",
            "file_format": "source JSONL lines copied byte-for-byte (CRLF kept), source order kept",
        },
        "database_disjoint": True,
        "partition_counts": {p: parts[p]["n_examples"] for p in PARTITIONS},
        "database_assignments": {db: {"partition": db_part[db], "n_examples": size[db]}
                                 for db in sorted(db_part)},
        "partitions": parts,
        "examples": [{"id": r["id"], "db_id": r["db_id"], "partition": db_part[r["db_id"]]}
                     for r in rows],
        "retrospective_evaluation_note": RETROSPECTIVE_NOTE,
    }
    return manifest, files


def manifest_bytes(manifest):
    return (json.dumps(manifest, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


def verify(out_dir=OUT_DIR, source=SOURCE):
    """Re-derive the split from the recorded seed/draws and compare with the files. -> [errors]."""
    errs = []
    mpath = out_dir / "split_manifest.json"
    if not mpath.exists():
        return [f"missing {mpath}"]
    rec = json.loads(mpath.read_text(encoding="utf-8"))
    manifest, files = build(rec["seed"], rec["methodology"]["draws"], source)
    if rec != manifest:
        errs.append("split_manifest.json differs from the re-derived manifest")
    for p in PARTITIONS:
        f = out_dir / f"{p}.jsonl"
        if not f.exists() or f.read_bytes() != files[p]:
            errs.append(f"{f.name} differs from the re-derived split")
    _, items = load_source(source)
    seen = Counter()
    db_parts = {}
    for p in PARTITIONS:
        f = out_dir / f"{p}.jsonl"
        if f.exists():
            for line in f.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    r = json.loads(line)
                    seen[r["id"]] += 1
                    db_parts.setdefault(r["db_id"], set()).add(p)
    src_ids = {r["id"] for _, r in items}
    if set(seen) != src_ids or any(v != 1 for v in seen.values()):
        errs.append("examples are not each in exactly one partition")
    if any(len(v) > 1 for v in db_parts.values()):
        errs.append("a database occurs in more than one partition")
    return errs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true", help="create the split files")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--draws", type=int, default=DRAWS)
    args = ap.parse_args()
    if args.write:
        existing = [p for p in [*(OUT_DIR / f"{q}.jsonl" for q in PARTITIONS),
                                OUT_DIR / "split_manifest.json"] if p.exists()]
        if existing:
            sys.exit(f"refusing to overwrite existing split files: {[p.name for p in existing]}")
        manifest, files = build(args.seed, args.draws)
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
    print("OK", m["partition_counts"], "database-disjoint; reproduced from seed", m["seed"])


if __name__ == "__main__":
    main()
