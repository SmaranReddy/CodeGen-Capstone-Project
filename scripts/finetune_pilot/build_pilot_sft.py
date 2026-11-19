"""Build a PILOT supervised fine-tuning set (question -> MQL) from a RAG corpus JSONL.

    python scripts/finetune_pilot/build_pilot_sft.py \
        --corpus provisional_rag_corpus.jsonl \
        --out data/finetune_pilot/pilot_v0

Swap in the approved corpus later by changing --corpus (same record layout):
    --corpus outputs/sql2mql_v2_1/approved_corpus_v2_1/corpus.jsonl

Output (same file names scripts/finetune_v1/train_qlora.py already expects):
    sft_train.prompt_completion.jsonl   {"id", "prompt": [system, user], "completion": [assistant]}
    sft_dev.prompt_completion.jsonl     database-disjoint dev split (loss monitoring only)
    meta.jsonl                          id -> db_id, split, hardness, query_class (not fed to the model)
    manifest.json                       counts, hashes, deviations from the benchmark prompt

The prompt text is the benchmark's own (scripts/spider1/prompt_spider1.py: SYSTEM + USER_TEMPLATE,
read with ast, nothing is executed). Reference SQL is never put in a prompt or a completion.

KNOWN DEVIATIONS from the benchmark prompt (this is a pilot, not a clean replica):
  1. Collections are the union of every collection any corpus entry of that database touches,
     not the full database (about 20% of tables are never touched, so they are missing).
  2. Foreign keys are written "(none)" and sample documents "(none)": the corpus has neither.
  3. Field types are mapped from the corpus's observed_kind + sqlite_affinity, not read from data.
A model trained on this prompt is evaluated on the fuller benchmark prompt, so there is a
train/test prompt shift. Rebuild with real FKs/samples (needs the Spider train SQLite files)
before quoting a number as a clean fine-tuning result.
"""
import argparse
import ast
import collections
import hashlib
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROMPT_SRC = ROOT / "scripts/spider1/prompt_spider1.py"
EVAL_TABLES = ROOT / "kaggle/nl2mql_spider_dev_1034_bundle_v2/context/tables.json"
SEED = 20261118
DEV_DB_FRACTION = 0.10
WARN_EST_TOKENS = 3500  # train_qlora.py truncates silently at --max-length (default 4096)


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load_prompt_constants():
    """Read SYSTEM / USER_TEMPLATE / PROMPT_VERSION from the benchmark file without importing it."""
    want = {"SYSTEM", "USER_TEMPLATE", "PROMPT_VERSION"}
    found = {}
    for node in ast.parse(PROMPT_SRC.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 \
                and isinstance(node.targets[0], ast.Name) and node.targets[0].id in want:
            found[node.targets[0].id] = ast.literal_eval(node.value)
    missing = want - set(found)
    if missing:
        sys.exit(f"could not read {sorted(missing)} from {PROMPT_SRC}")
    return found


def bson_types(col):
    """Approximate the benchmark's 'observed type' string from the corpus's per-column summary."""
    kind, aff = col["observed_kind"], col["sqlite_affinity"]
    if kind == "num":
        return {"INTEGER": "int", "REAL": "double"}.get(aff, "int/double")
    if kind == "text":
        return "string"
    if kind == "mixed":
        return "int/string"
    return "null"


def build_db_schemas(rows):
    """db_id -> {collection: (columns, primary_key)} merged over all entries of that database."""
    schemas = collections.defaultdict(dict)
    for r in rows:
        for t in r["schema"]["touched"]:
            cols = t["columns"]
            cur = schemas[r["db_id"]].get(t["collection"])
            if cur is None:
                schemas[r["db_id"]][t["collection"]] = cols
            else:
                names_cur = [c["name"] for c in cur]
                names_new = [c["name"] for c in cols]
                if names_cur != names_new:
                    sys.exit(f"inconsistent columns for {t['collection']}: {names_cur} vs {names_new}")
    return schemas


def render_user(consts, db_id, schema, question):
    lines = []
    for coll in sorted(schema):
        lines.append(f"- {coll}")
        for c in schema[coll]:
            lines.append(f"    {c['name']}: {bson_types(c)} [{c['declared_type'] or '?'}]")
    return consts["USER_TEMPLATE"].format(db=db_id, schema="\n".join(lines), fks="(none)",
                                          samples="(none)", question=question)


def completion_text(mql):
    return json.dumps({"collection": mql["collection"], "pipeline": mql["pipeline"]},
                      ensure_ascii=False, separators=(",", ":"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--eval-tables", default=str(EVAL_TABLES),
                    help="tables.json of the evaluation databases; training DBs must not appear in it")
    args = ap.parse_args()

    corpus = Path(args.corpus)
    rows = [json.loads(line) for line in corpus.open(encoding="utf-8")]
    ids = [r["id"] for r in rows]
    if len(ids) != len(set(ids)):
        sys.exit("duplicate ids in corpus")
    for r in rows:
        if "mql" not in r or "pipeline" not in r["mql"] or not r.get("question"):
            sys.exit(f"bad record {r.get('id')}")

    # Leakage guard 1: no training database may be an evaluation database.
    eval_dbs = {t["db_id"] for t in json.loads(Path(args.eval_tables).read_text(encoding="utf-8"))}
    overlap = sorted({r["db_id"] for r in rows} & eval_dbs)
    if overlap:
        sys.exit(f"LEAK: corpus databases also in the evaluation set: {overlap}")

    consts = load_prompt_constants()
    schemas = build_db_schemas(rows)

    # Database-disjoint dev split (seeded, deterministic).
    dbs = sorted({r["db_id"] for r in rows})
    rng = random.Random(SEED)
    shuffled = dbs[:]
    rng.shuffle(shuffled)
    n_dev_dbs = max(1, round(len(dbs) * DEV_DB_FRACTION))
    dev_dbs = set(shuffled[:n_dev_dbs])

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    files = {"train": [], "dev": []}
    meta, lengths = [], []
    for r in sorted(rows, key=lambda x: x["id"]):
        split = "dev" if r["db_id"] in dev_dbs else "train"
        user = render_user(consts, r["db_id"], schemas[r["db_id"]], r["question"])
        comp = completion_text(r["mql"])
        rec = {"id": r["id"],
               "prompt": [{"role": "system", "content": consts["SYSTEM"]},
                          {"role": "user", "content": user}],
               "completion": [{"role": "assistant", "content": comp}]}
        files[split].append(rec)
        est = (len(consts["SYSTEM"]) + len(user) + len(comp)) / 3.2  # rough chars/token; NOT a tokenizer
        lengths.append(est)
        meta.append({"id": r["id"], "db_id": r["db_id"], "split": split,
                     "hardness": r.get("hardness"), "query_class": r.get("query_class")})

    for split, recs in files.items():
        with (out / f"sft_{split}.prompt_completion.jsonl").open("w", encoding="utf-8", newline="\n") as f:
            for rec in recs:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    with (out / "meta.jsonl").open("w", encoding="utf-8", newline="\n") as f:
        for m in meta:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")

    over = sum(1 for x in lengths if x > WARN_EST_TOKENS)
    manifest = {
        "dataset": "finetune_pilot_v0",
        "status": "PILOT_NOT_BENCHMARK_FAITHFUL",
        "builder": "scripts/finetune_pilot/build_pilot_sft.py",
        "corpus": {"path": str(corpus), "sha256": sha256(corpus), "records": len(rows),
                   "corpus_version": rows[0].get("corpus_version"),
                   "corpus_status": rows[0].get("corpus_status"),
                   "human_review_status": rows[0].get("human_review_status")},
        "prompt": {"source": str(PROMPT_SRC.relative_to(ROOT)), "version": consts["PROMPT_VERSION"]},
        "seed": SEED,
        "counts": {"train": len(files["train"]), "dev": len(files["dev"]),
                   "databases": len(dbs), "dev_databases": sorted(dev_dbs)},
        "estimated_tokens_chars_div_3_2": {"mean": round(sum(lengths) / len(lengths)),
                                            "max": round(max(lengths)),
                                            "records_over_warn": over, "warn_at": WARN_EST_TOKENS,
                                            "note": "approximation only; run a real tokenizer audit"},
        "leakage_checks": {"corpus_dbs_vs_eval_dbs_overlap": overlap, "eval_dbs_checked": len(eval_dbs),
                           "question_overlap_with_892": "NOT checked here (the 892 questions are not in git)"},
        "deviations_from_benchmark_prompt": [
            "collections = union of collections touched by any corpus entry of the database (unused tables missing)",
            "foreign keys written '(none)'",
            "sample documents written '(none)'",
            "field types mapped from observed_kind + sqlite_affinity, not read from stored data",
        ],
        "outputs": {p.name: sha256(p) for p in sorted(out.glob("*.jsonl"))},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in ("counts", "estimated_tokens_chars_div_3_2")}, indent=1))


if __name__ == "__main__":
    main()
