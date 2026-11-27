"""Build the private Kaggle bundle for MQL-0006 (Qwen-3B vs Groq-hosted gpt-oss-120b, NL->MQL, Spider dev).

No model call, no Atlas connection, no network.

Evaluation set: the 1,034 spider-sql-v1 dev records MINUS every record on a database assigned to the held-out MQL
test partition. That partition is database-disjoint; its databases come from the ID-only split manifest
(data/splits_expanded/split_manifest.json). The partition file itself (test.jsonl) is never opened. Records on those
databases, their SQLite files, schemas, prompt samples and mapping entries are not packaged at all.

  data/eval_records.jsonl      the evaluation records: byte-identical lines of data/processed/spider_sql_v1/
                               dev_full.jsonl, original order. Gold SQL is in it for the reference step only
  data/prompts.jsonl           frozen nl-mql-spider1-v1 prompts, built by the unmodified
                               scripts/spider1/prompt_spider1.py from (db_id, question) only
  data/reference.jsonl.gz      the gold reference (rows, tie blocks, status) computed here; both models are scored
                               against exactly these rows, on Kaggle and in the local verifier
  data/reference_expected.json per-id digest of that reference; the notebook re-executes every gold SQL and fails
                               closed only on real differences (dev1034_core.recheck_reference)
  data/atlas_mapping.json      SQLite table -> Atlas collection mapping for the evaluation DBs: fields, types,
                               counts and a type-tagged row-multiset digest per table (the Kaggle preflight checks
                               Atlas against it)
  data/prior_audit.json        earlier 455-benchmark audit status of evaluation items (sensitivity subset only)
  context/                     atlas_collections.json, prompt_samples.json (the parity.json-hashed sources,
                               restricted to the evaluation DBs) and tables.json (evaluation DB entries)
  database/<db>/<db>.sqlite    the evaluation DBs' SQLite files (byte copies)
  code/                        dev1034_core.py, and byte copies of spider1_kit.py, prompt_spider1.py, mongo_conn.py
  ATTRIBUTION.md, bundle_manifest.json

Fails closed on: changed source hashes, counts, order or ids; an inconsistent split manifest; a missing database;
prompt/context drift; gold results that do not reproduce the recorded hashes; any denied path (train/val/test,
splits, .env, caches, checkpoints); any mention of a held-out database in a packaged data file; credential-like
strings; tree or zip contents that differ from the allowlist.

Usage (repo root, Windows or Linux):
  python scripts/nl2mql/prepare_spider_dev_1034_bundle.py [--version v2] [--overwrite]
  python scripts/nl2mql/prepare_spider_dev_1034_bundle.py --verify kaggle/nl2mql_spider_dev_1034_bundle_v2
"""
import argparse
import gzip
import hashlib
import json
import re
import shutil
import sqlite3
import sys
import time
import zipfile
from collections import Counter
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import dev1034_core as C  # noqa: E402

PROCESSED = ROOT / "data/processed/spider_sql_v1"
SPIDER = ROOT / "data/raw/spider1/spider_data"
INTERIM = ROOT / "data/interim/spider1"
EXPANDED = ROOT / "data/processed/spider1_nl_mql_expanded.jsonl"
SPLIT_MANIFEST = ROOT / "data/splits_expanded/split_manifest.json"   # ID-only metadata; test.jsonl is never opened
DEFAULT_VERSION = "v2"
CODE_SOURCES = {
    "code/dev1034_core.py": HERE / "dev1034_core.py",
    "code/spider1_kit.py": ROOT / "scripts/spider1/spider1_kit.py",
    "code/prompt_spider1.py": ROOT / "scripts/spider1/prompt_spider1.py",
    "code/mongo_conn.py": ROOT / "scripts/mongo_conn.py",
}
PROMPT_VERSION = "nl-mql-spider1-v1"
PROMPT_TEMPLATE_SHA256 = "7401bf140dd1b8a9dac057edd9f729a8624823604fa2bd6ae9cfaf1c1890c1be"  # MQL-0001..0005
DENY = re.compile(r"(^|/)(test[_.]|test_database|test_tables|testsuite|train|val_internal|dev_clean|rejected|"
                  r"splits|\.env|__pycache__|\.ipynb_checkpoints|checkpoint|\.git/)|\.(pyc|gguf|safetensors|bin|pt)$",
                  re.I)
SECRET_PATTERNS = [r"mongodb(\+srv)?://[^\s\"'<]*@", r"MONGODB_URI\s*=\s*\S", r"gsk_[A-Za-z0-9]{20,}",
                   r"AIza[0-9A-Za-z_\-]{30,}", r"sk-ant-[A-Za-z0-9\-]{20,}", r"sk-[A-Za-z0-9]{32,}",
                   r"hf_[A-Za-z0-9]{30,}", r"(API_KEY|PASSWORD|SECRET)\s*=\s*['\"]?[A-Za-z0-9]"]

ATTRIBUTION = """# Attribution

Private, evaluation-only bundle for one experiment (MQL-0006). Not for redistribution beyond the licences below.

- **Spider 1.0 dev set and databases** (Yu et al., EMNLP 2018), https://yale-lily.github.io/spider - CC BY-SA 4.0.
  Files: data/eval_records.jsonl (Spider dev records with execution metadata added by spider-sql-v1, restricted
  to the evaluation databases), database/*/*.sqlite (unmodified), context/tables.json (entries of tables.json),
  and the files derived from them (prompts, reference, Atlas mapping, prompt sample documents). Redistribution must
  keep this attribution and licence.
- Project code under code/ (capstone repository, written for this project).
"""


class BundleError(RuntimeError):
    pass


def need(cond, msg):
    if not cond:
        raise BundleError(msg)


def _jbytes(obj):
    return (json.dumps(obj, indent=1, ensure_ascii=False, default=str) + "\n").encode("utf-8")


def _jlbytes(records):
    return "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records).encode("utf-8")


def held_re(db_ids):
    """Matches a held-out database name as a whole identifier (also as the `<db>__` collection prefix)."""
    return re.compile(r"(?<![A-Za-z0-9_])(" + "|".join(map(re.escape, db_ids)) + r")(?![A-Za-z0-9])")


def repo_modules():
    sys.path.insert(0, str(ROOT / "scripts/spider1"))
    import prompt_spider1 as P  # noqa: E402
    import spider1_kit as K  # noqa: E402
    return K, P


def build_mapping(db_ids, atlas_colls, parity):
    colls = {}
    for d in db_ids:
        path = SPIDER / "database" / d / f"{d}.sqlite"
        con = C._ro_connect(path, 60)
        try:
            tables = [r[0] for r in con.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name")]
            for t in tables:
                info = con.execute(f'PRAGMA table_info("{t}")').fetchall()
                fields = [r[1] for r in info]
                declared = {r[1]: r[2] for r in info}
                name = f"{d}__{t}"
                need(name in atlas_colls, f"{name} missing from atlas_collections.json")
                a = atlas_colls[name]
                need(list(a["fields"]) == fields, f"{name}: Atlas field list differs from SQLite columns")
                n, dig, replaced = C.sqlite_table_digest(path, t, fields)
                need(a["documents"] == n, f"{name}: atlas_collections.json has {a['documents']} docs, SQLite {n} rows")
                need(replaced == C.EXPECTED_UTF8_REPLACED.get(name, 0),
                     f"{name}: {replaced} invalid-UTF-8 strings, expected {C.EXPECTED_UTF8_REPLACED.get(name, 0)}")
                pt = parity["databases"][d]["tables"][t]
                need(pt["sqlite_rows"] == n and pt["atlas_docs"] == n and not pt["rows_only_in_sqlite"]
                     and not pt["rows_only_in_atlas"] and not pt["missing_fields"] and not pt["extra_fields"],
                     f"{name}: parity.json does not record an exact match")
                colls[name] = {"db_id": d, "table": t, "fields": fields, "declared_sql_types": declared,
                               "observed_bson_types": a["fields"], "documents": n, "sqlite_rows": n,
                               "row_multiset_sha256": dig, "utf8_replaced_strings": replaced,
                               "parity_2026_10_20": {"match": pt["match"], "utf8_replaced_strings": pt["utf8_replaced_strings"],
                                                     "natural_order_matches_rowid": pt["natural_order_matches_rowid"]}}
        finally:
            con.close()
    need(set(colls) == set(atlas_colls), "prompt context collections != SQLite tables of the evaluation DBs")
    return {
        "atlas_db": C.ATLAS_DB, "db_ids": db_ids, "collections": dict(sorted(colls.items())),
        "conventions": {
            "naming": "one collection per SQLite table, named <db_id>__<table> (case preserved)",
            "fields": "every column kept under its SQLite name (case preserved); no renaming, no embedding",
            "types": "SQLite storage class kept per value: INTEGER->int32/int64, REAL->double, TEXT->string, "
                     "NULL->explicit null (no missing fields on Atlas); no BLOB values in the dev DBs",
            "dates": "dates/times are TEXT in SQLite and strings on Atlas (no BSON Date conversion)",
            "numbers_as_text": "TEXT columns holding numbers stay strings (e.g. car_1 horsepower)",
            "duplicates": "duplicate rows preserved (multiset parity)",
            "relationships": "flat: foreign keys are listed in the prompt; no embedding",
            "_id": "generated ObjectId per document; the comparator drops an ObjectId _id",
            "utf8": "wta_1__players holds 2 invalid-UTF-8 strings in SQLite; BSON stores them U+FFFD-replaced. "
                    "The digest here decodes SQLite text with errors='replace', so it equals the Atlas form.",
        },
        "evidence": {
            "atlas_import": "scripts/spider1/import_atlas.py (2026-10; codegen_spider1: 80 collections for all 20 dev "
                            "DBs, data 54.4 MB, index 9.6 MB per outputs/spider1_import_wta.log)",
            "parity_check": f"data/interim/spider1/expanded/parity.json checked_at {parity['checked_at']}: every table "
                            "of the evaluation DBs identical (wta_1 differs only in the 2 UTF-8-replaced strings)",
            "read_only_user": "codegen_eval_readonly (custom role codegen_spider1_read), verified 2026-10-20",
            "not_reverified_offline": "Atlas was not contacted while building this bundle; the Kaggle preflight "
                                      "re-checks privileges, coverage, counts and full parity at run time",
        },
        "storage": {"codegen_spider1_data_mb": 54.4, "codegen_spider1_index_mb": 9.6, "atlas_tier_limit_mb": 512,
                    "new_data_required": False},
    }


def prior_audit_map(all_records, eval_ids):
    """Earlier 455-benchmark audit status by dev id, for evaluation-set items only (sensitivity subset)."""
    by_idx = {r["source"]["index"]: r for r in all_records}
    pool = C.read_jsonl(INTERIM / "pool.jsonl")
    status = {s["id"]: s for s in C.read_jsonl(INTERIM / "expanded/status.jsonl")}
    need(len(pool) == 493 and set(status) == {p["id"] for p in pool}, "pool/status mismatch")
    out = {}
    for p in pool:
        r = by_idx[p["dev_index"]]
        need((r["db_id"], r["question"]) == (p["db_id"], p["question"]), f"pool item {p['id']} != dev record")
        if r["id"] not in eval_ids:
            continue
        s = status[p["id"]]
        out[r["id"]] = {"pool_id": p["id"], "status": s["status"], "stage": s.get("stage"), "kind": s.get("kind"),
                        "reason": s.get("reason"), "flags": s.get("flags") or []}
    return dict(sorted(out.items()))


def heldout_exclusion(all_records):
    """Databases of the held-out MQL test partition, from ID-only metadata. Never opens the partition file.

    Source: data/splits_expanded/split_manifest.json (`database_assignments`, `partitions.test.{databases,ids}`,
    `examples` = [{id, db_id, partition}]), cross-checked against the 455-item expanded dataset (whose hash the
    manifest records) and mapped onto dev ids through the 493-item pool's `dev_index`. The partition is
    database-disjoint, so every dev record on those databases is excluded (user decision, 2026-10-30): excluding
    only the overlapping questions would keep paraphrases whose gold SQL is identical to a held-out item's.
    """
    man = json.loads(SPLIT_MANIFEST.read_text(encoding="utf-8"))
    need(man.get("database_disjoint") is True, "split manifest is not database-disjoint")
    need(C.sha256_file(EXPANDED) == man["source"]["sha256"], "expanded dataset hash differs from the split manifest")
    test_dbs = sorted(d for d, v in man["database_assignments"].items() if v["partition"] == "test")
    need(test_dbs and test_dbs == sorted(man["partitions"]["test"]["databases"]), "test databases inconsistent")
    test_ids = [e["id"] for e in man["examples"] if e["partition"] == "test"]
    need(len(test_ids) == man["partition_counts"]["test"] == len(set(test_ids)) and
         set(test_ids) == set(man["partitions"]["test"]["ids"]), "test ids inconsistent in the manifest")
    expd = {r["id"]: r for r in C.read_jsonl(EXPANDED)}
    need({i for i, r in expd.items() if r["db_id"] in test_dbs} == set(test_ids),
         "held-out ids are not exactly the expanded items on the held-out databases")
    pool = {p["id"]: p for p in C.read_jsonl(INTERIM / "pool.jsonl")}
    by_idx = {r["source"]["index"]: r for r in all_records}
    overlap = set()
    for i in test_ids:
        p = pool[i]
        r = by_idx[p["dev_index"]]
        need((r["db_id"], r["question"]) == (p["db_id"], p["question"]) == (expd[i]["db_id"], expd[i]["question"]),
             f"cannot map held-out id {i} onto dev_full")
        overlap.add(r["id"])
    excluded = [r["id"] for r in all_records if r["db_id"] in test_dbs]
    need(overlap <= set(excluded), "a held-out question lies outside the excluded databases")
    return {"rule": "exclude every Spider dev record on a database assigned to the held-out MQL test partition",
            "excluded_db_ids": test_dbs, "excluded_records": len(excluded),
            "overlap_with_heldout_questions": len(overlap),
            "excluded_beyond_overlap": len(excluded) - len(overlap),
            "source": "ID-only split manifest (database_assignments, partition ids); the partition file was not opened",
            "split_manifest_sha256": C.sha256_file(SPLIT_MANIFEST)}


def plan():
    """Validate the sources and return ({bundle path: source path or bytes}, facts). Writes nothing."""
    src_man = json.loads((PROCESSED / "manifest.json").read_text(encoding="utf-8"))
    need(src_man.get("version") == "spider-sql-v1", "processed manifest is not spider-sql-v1")
    dev_path = PROCESSED / "dev_full.jsonl"
    need(C.sha256_file(dev_path) == src_man["files_sha256"]["dev_full.jsonl"], "dev_full.jsonl hash changed")
    need(C.sha256_file(SPIDER / "tables.json") == src_man["inputs_sha256"]["tables.json"], "tables.json hash changed")
    lines = [ln for ln in dev_path.read_bytes().splitlines(keepends=True) if ln.strip()]
    all_recs = [json.loads(ln) for ln in lines]
    need(len(all_recs) == C.N_SOURCE, f"dev_full has {len(all_recs)} records")
    need([r["id"] for r in all_recs] == [f"spider_dev_{i:04d}" for i in range(C.N_SOURCE)], "ids not in order")
    need([r["source"]["index"] for r in all_recs] == list(range(C.N_SOURCE)), "not in dev.json order")
    need(all(r["partition"] == "dev" and r["source"]["file"] == "dev.json" for r in all_recs), "non-dev record")
    need(all(r["question"].strip() and r["sql"].strip() for r in all_recs), "empty question or SQL")
    need({r["hardness"] for r in all_recs} <= set(C.HARDNESS), "unknown hardness label")
    need(len({r["db_id"] for r in all_recs}) == 20, "dev_full does not cover the 20 dev DBs")

    excl = heldout_exclusion(all_recs)
    keep = [(ln, r) for ln, r in zip(lines, all_recs) if r["db_id"] not in excl["excluded_db_ids"]]
    recs = [r for _, r in keep]
    need(len(recs) == C.N_SOURCE - excl["excluded_records"], "evaluation-set size arithmetic failed")
    db_ids = sorted({r["db_id"] for r in recs})
    need(len(db_ids) == 20 - len(excl["excluded_db_ids"]), f"{len(db_ids)} databases")

    parity = json.loads((INTERIM / "expanded/parity.json").read_text(encoding="utf-8"))
    ctx_src = {n: INTERIM / n for n in ("atlas_collections.json", "prompt_samples.json")}
    for n, p in ctx_src.items():
        need(C.sha256_file(p) == parity["prompt_context_sha256"][n], f"{n} differs from the hash in parity.json")
    atlas_all = json.loads(ctx_src["atlas_collections.json"].read_text(encoding="utf-8"))
    samples_all = json.loads(ctx_src["prompt_samples.json"].read_text(encoding="utf-8"))
    need(len({v["db_id"] for v in atlas_all.values()}) == 20, "prompt context does not cover the 20 dev DBs")
    atlas_colls = {k: v for k, v in atlas_all.items() if v["db_id"] in db_ids}
    samples = {k: v for k, v in samples_all.items() if k in atlas_colls}
    tables = [t for t in json.loads((SPIDER / "tables.json").read_text(encoding="utf-8")) if t["db_id"] in db_ids]
    need(sorted(t["db_id"] for t in tables) == db_ids, "tables.json entries != evaluation DBs")

    K, P = repo_modules()
    need(P.PROMPT_VERSION == PROMPT_VERSION and P.template_hash() == PROMPT_TEMPLATE_SHA256,
         "prompt_spider1 template changed (version/hash differ from MQL-0001..0005)")
    prompts = C.build_prompts(recs, P)   # the repo prompt module on the full, unfiltered context
    C.check_no_gold_leak(prompts, recs)

    refs = C.compute_reference(recs, K, SPIDER / "database")
    bad = [r["id"] for r in refs if r["reference_status"] == "reference_error"]
    need(not bad, f"gold SQL does not reproduce the recorded result for {bad[:5]}")
    ref_expected = {"items": [C.reference_digest(r) for r in refs],
                    "status_counts": dict(sorted(Counter(r["reference_status"] for r in refs).items()))}

    mapping = build_mapping(db_ids, atlas_colls, parity)
    mapping["sqlite_sha256"] = {d: C.sha256_file(SPIDER / "database" / d / f"{d}.sqlite") for d in db_ids}
    audit = prior_audit_map(all_recs, {r["id"] for r in recs})
    ids = [r["id"] for r in recs]

    files = {"data/eval_records.jsonl": b"".join(ln for ln, _ in keep),
             "data/prompts.jsonl": _jlbytes(prompts),
             "data/reference_expected.json": _jbytes(ref_expected),
             "data/reference.jsonl.gz": gzip.compress(_jlbytes(refs), mtime=0),
             "data/atlas_mapping.json": _jbytes(mapping),
             "data/prior_audit.json": _jbytes(audit),
             "context/atlas_collections.json": _jbytes(atlas_colls),
             "context/prompt_samples.json": _jbytes(samples),
             "context/tables.json": _jbytes(tables),
             "ATTRIBUTION.md": ATTRIBUTION.encode("utf-8"),
             **CODE_SOURCES}
    for d in db_ids:
        src = SPIDER / "database" / d / f"{d}.sqlite"
        need(src.is_file(), f"missing {src}")
        files[f"database/{d}/{d}.sqlite"] = src
    held = held_re(excl["excluded_db_ids"])
    for name, src in files.items():
        need(not DENY.search(name), f"refusing to package a denied path: {name}")
        need(not held.search(name), f"refusing to package a held-out database path: {name}")
        if not isinstance(src, bytes):
            need(Path(src).is_file(), f"missing source {src}")
            need(not DENY.search(Path(src).relative_to(ROOT).as_posix()), f"denied source path {src}")
        if name.endswith((".py", ".json", ".jsonl", ".md", ".txt", ".gz")):
            raw = src if isinstance(src, bytes) else Path(src).read_bytes()
            text = (gzip.decompress(raw) if name.endswith(".gz") else raw).decode("utf-8")
            for pat in SECRET_PATTERNS:
                need(not re.search(pat, text), f"credential-like string ({pat}) in {name}")
            if name.startswith(("data/", "context/")):
                need(not held.search(text), f"{name} mentions a held-out database")
    facts = {"records": len(recs), "db_ids": db_ids, "src_manifest": src_man, "parity_checked_at": parity["checked_at"],
             "collections": len(atlas_colls), "exclusion": excl, "ids_sha256": C.sha256_text("\n".join(ids)),
             "records_per_db": {d: sum(r["db_id"] == d for r in recs) for d in db_ids},
             "hardness": {h: sum(r["hardness"] == h for r in recs) for h in C.HARDNESS},
             "reference_status": ref_expected["status_counts"],
             "prior_audit": {"included": sum(v["status"] == "included" for v in audit.values()),
                             "excluded": sum(v["status"] == "excluded" for v in audit.values()),
                             "not_audited": len(recs) - len(audit)},
             "prompt_template_sha256": P.template_hash(), "sqlite_version": sqlite3.sqlite_version}
    return files, facts


def write_bundle(out_dir, files, facts, version):
    out_dir.mkdir(parents=True)
    meta = {}
    for name, src in sorted(files.items()):
        dest = out_dir / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(src, bytes):
            dest.write_bytes(src)
            origin = "derived"
        else:
            shutil.copyfile(src, dest)
            origin = Path(src).relative_to(ROOT).as_posix()
            need(C.sha256_file(dest) == C.sha256_file(src), f"copy of {name} is not byte-identical")
        meta[name] = {"sha256": C.sha256_file(dest), "bytes": dest.stat().st_size, "source": origin}
    meta["data/eval_records.jsonl"]["source"] = ("data/processed/spider_sql_v1/dev_full.jsonl (byte-identical lines, "
                                                 "original order, held-out databases removed)")
    sm = facts["src_manifest"]
    manifest = {
        "bundle": out_dir.name, "version": version, "experiment": C.EXPERIMENT_ID,
        "purpose": "zero-shot NL->MQL, Qwen2.5-Coder-3B-Instruct vs Groq-hosted open-weight openai/gpt-oss-120b, "
                   f"cross-engine execution accuracy on {facts['records']} Spider dev items (held-out test databases "
                   "excluded); evaluation only, private",
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "builder": "scripts/nl2mql/prepare_spider_dev_1034_bundle.py",
        "builder_sha256": C.sha256_file(Path(__file__)),
        "code_sha256": {n: meta[n]["sha256"] for n in CODE_SOURCES},
        "prompt": {"version": PROMPT_VERSION, "template_sha256": facts["prompt_template_sha256"],
                   "prompts_sha256": meta["data/prompts.jsonl"]["sha256"]},
        "evaluation_set": {"records": facts["records"], "databases": len(facts["db_ids"]),
                           "collections": facts["collections"], "ids_sha256": facts["ids_sha256"],
                           "source_records": C.N_SOURCE, **facts["exclusion"]},
        "counts": {"records": facts["records"], "databases": len(facts["db_ids"]), "collections": facts["collections"],
                   "files": len(meta), "records_per_db": facts["records_per_db"],
                   "hardness": facts["hardness"], "reference_status": facts["reference_status"],
                   "prior_audit": facts["prior_audit"]},
        "db_ids": facts["db_ids"],
        "provenance": {
            "dataset": "spider-sql-v1 dev_full (subset)", "dataset_manifest_built_at": sm.get("built_at"),
            "dev_full_sha256": sm["files_sha256"]["dev_full.jsonl"],
            "spider_inputs_sha256": {k: sm["inputs_sha256"][k] for k in ("dev.json", "tables.json")},
            "databases_from": "data/raw/spider1/spider_data/database/<db>/<db>.sqlite (byte copies)",
            "prompt_context_from": "data/interim/spider1/{atlas_collections,prompt_samples}.json (hashes in "
                                   f"expanded/parity.json, checked_at {facts['parity_checked_at']}), restricted to "
                                   "the evaluation databases",
            "data_license": sm.get("license"),
            "reference_built_with_sqlite": facts["sqlite_version"],
        },
        "excluded_by_design": ["every record, SQLite file, schema and sample of the held-out test databases",
                               "spider-sql-v1 train/val_internal/dev_clean/test_full/test_clean/rejected",
                               "data/splits*/ files", "test.json", "test_tables.json", "test_database/",
                               "test-suite variant DBs", "non-dev databases", "gold MQL", "RAG corpus",
                               "fine-tuning data", "model files", ".env and any credential", "caches"],
        "files": meta,
    }
    (out_dir / "bundle_manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n",
                                                  encoding="utf-8")
    return manifest


def verify_dir(out_dir):
    """Re-check a written bundle: manifest hashes, allowlist and held-out rules, prompts/reference re-derived."""
    out_dir = Path(out_dir)
    man = C.verify_bundle(out_dir)
    held = held_re(man["evaluation_set"]["excluded_db_ids"])
    for name in man["files"]:
        need(not DENY.search(name), f"denied path in bundle: {name}")
        need(not held.search(name), f"held-out database path in bundle: {name}")
        if name.startswith(("data/", "context/")):
            raw = (out_dir / name).read_bytes()
            need(not held.search((gzip.decompress(raw) if name.endswith(".gz") else raw).decode("utf-8")),
                 f"{name} mentions a held-out database")
    need(all(n.count("/") == 2 and n.endswith(".sqlite") for n in man["files"] if n.startswith("database/")),
         "database/ may hold only <db>/<db>.sqlite")
    need(sorted(n.split("/")[1] for n in man["files"] if n.startswith("database/")) == man["db_ids"],
         "database files != evaluation databases")
    K, P = C.bundle_modules(out_dir)
    recs = C.load_dev(out_dir)
    frozen = C.read_jsonl(out_dir / "data/prompts.jsonl")
    C.compare_prompts(frozen, C.build_prompts(recs, P))
    C.check_no_gold_leak(frozen, recs)
    exp = json.loads((out_dir / "data/reference_expected.json").read_text(encoding="utf-8"))
    need([C.reference_digest(r) for r in C.load_reference(out_dir)] == exp["items"],
         "data/reference.jsonl.gz does not match data/reference_expected.json")
    mapping = C.load_mapping(out_dir)
    need(sorted({m["db_id"] for m in mapping["collections"].values()}) == man["db_ids"] and
         len(mapping["collections"]) == man["evaluation_set"]["collections"], "mapping != evaluation databases")
    return man


def write_zip(out_dir, zip_path, manifest):
    names = sorted(list(manifest["files"]) + ["bundle_manifest.json"])
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for name in names:
            z.write(out_dir / name, f"{out_dir.name}/{name}")
    with zipfile.ZipFile(zip_path) as z:
        need(z.testzip() is None, "zip CRC check failed")
        got = set(z.namelist())
        for name in names:
            need(hashlib.sha256(z.read(f"{out_dir.name}/{name}")).hexdigest() ==
                 (manifest["files"][name]["sha256"] if name in manifest["files"] else C.sha256_file(out_dir / name)),
                 f"zip member differs: {name}")
    need(got == {f"{out_dir.name}/{n}" for n in names}, "zip contents differ from the allowlist")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", default=DEFAULT_VERSION)
    ap.add_argument("--out-root", default=str(ROOT / "kaggle"))
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--verify", metavar="BUNDLE_DIR")
    a = ap.parse_args(argv)
    try:
        if a.verify:
            man = verify_dir(a.verify)
            print(f"OK {a.verify}: {man['counts']['records']} records, {len(man['db_ids'])} databases, "
                  f"{man['counts']['files']} files; prompts re-derived identically; no held-out database content")
            return 0
        name = f"{C.BUNDLE_NAME}_{a.version}"
        out_dir, zip_path = Path(a.out_root) / name, Path(a.out_root) / f"{name}.zip"
        files, facts = plan()
        if out_dir.exists() or zip_path.exists():
            need(a.overwrite, f"{out_dir} or {zip_path} exists; pass --overwrite or a new --version")
            need(not out_dir.exists() or (out_dir / "bundle_manifest.json").is_file(), f"{out_dir} is not a bundle")
            shutil.rmtree(out_dir, ignore_errors=True)
            zip_path.unlink(missing_ok=True)
        manifest = write_bundle(out_dir, files, facts, a.version)
        verify_dir(out_dir)
        write_zip(out_dir, zip_path, manifest)
        size = sum(m["bytes"] for m in manifest["files"].values())
        print(f"bundle: {out_dir}  ({manifest['counts']['files']} files + manifest, {size / 1e6:.1f} MB)")
        print(f"zip:    {zip_path}  ({zip_path.stat().st_size / 1e6:.1f} MB)")
        print(f"records {facts['records']}, databases {len(facts['db_ids'])}, collections {facts['collections']}, "
              f"exclusion {json.dumps({k: facts['exclusion'][k] for k in ('excluded_db_ids', 'excluded_records', 'overlap_with_heldout_questions')})}")
        print(f"reference status {facts['reference_status']}, prior audit {facts['prior_audit']}")
        return 0
    except (BundleError, C.PreflightError) as e:
        print(f"BUNDLE ERROR: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
