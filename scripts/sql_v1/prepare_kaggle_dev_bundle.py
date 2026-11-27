"""Build the private Kaggle input bundle for the zero-shot SQL baseline on Spider dev (no model calls).

Copies an explicit allowlist, nothing else:
  data/dev_full.jsonl            byte copy of data/processed/spider_sql_v1/dev_full.jsonl (1,034 records, gold SQL
                                 included for scoring after generation; the notebook never puts it in a prompt)
  data/schemas_ddl_dev.json      schemas_ddl.json restricted to the 20 dev databases (prompt schema)
  data/tables_dev.json           official tables.json entries for the 20 dev databases (evaluator foreign-key maps)
  database/<db>/<db>.sqlite      the 20 dev SQLite files from spider_data/database/ (never test_database/)
  code/spider_zs_sql.py          prompt / resumable generation log / scoring module used by the notebook
  code/official_eval/            evaluation.py + process_sql.py (taoyds/spider @ b7b5b8c, Apache-2.0), LICENSE,
                                 SOURCE_COMMIT.txt
  ATTRIBUTION.md                 Spider (CC BY-SA 4.0) and evaluator (Apache-2.0) attribution
  bundle_manifest.json           sha256 + size of every file, counts, db_ids, provenance

Fails closed: pinned source hashes, exact counts and order, every referenced DB present, a deny-list on output
names (test / train / val / dev_clean never packaged -- the test files are not even opened), a secret scan, and
the written tree and the zip are compared with the allowlist as sets.

Usage (from the repo root):
  python scripts/sql_v1/prepare_kaggle_dev_bundle.py [--version v1] [--overwrite]
  python scripts/sql_v1/prepare_kaggle_dev_bundle.py --verify kaggle/spider_sql_v1_dev_bundle_v1
  -> kaggle/spider_sql_v1_dev_bundle_<version>/ and kaggle/spider_sql_v1_dev_bundle_<version>.zip
"""
import argparse
import hashlib
import json
import re
import shutil
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data/processed/spider_sql_v1"
SPIDER = ROOT / "data/raw/spider1/spider_data"
EVAL_SRC = ROOT / "data/raw/spider1/official_eval"
MODULE_SRC = Path(__file__).resolve().parent / "spider_zs_sql.py"

N_DEV = 1034
N_DBS = 20
EVAL_COMMIT = "b7b5b8c890cd30e35427348bb9eb8c6d1350ca7c"
# checked 2026-10-27 against raw.githubusercontent.com/taoyds/spider/<EVAL_COMMIT>/<file>
PINNED = {
    "evaluation.py": "c61f18f40bec5af1984af0e8731e678f6cee7976611d910e6e5c144d757fd827",
    "process_sql.py": "909833d02b64124f407b218bdfa7bf6d3b3ab17593ae9d2bd4ebc8f4ab498354",
    "LICENSE": "a8d1d88a0f9cce3cdbff73b03a80dc5e5d69e3ba2443946c268815a550c9e0ec",
}
DENY = re.compile(r"(^|/)(test[_.]|test_database|test_tables|train|val_internal|dev_clean|rejected|\.env)", re.I)
SECRET_PATTERNS = [r"mongodb(\+srv)?://", r"MONGODB_URI\s*=", r"gsk_[A-Za-z0-9]{20,}", r"AIza[0-9A-Za-z_\-]{30,}",
                   r"sk-ant-[A-Za-z0-9\-]{20,}", r"sk-[A-Za-z0-9]{32,}", r"hf_[A-Za-z0-9]{30,}",
                   r"(API_KEY|PASSWORD|SECRET)\s*=\s*\S+"]

ATTRIBUTION = """# Attribution

This bundle is a private, evaluation-only subset prepared for one zero-shot baseline run.

- **Spider 1.0 dev set and databases** (Yu et al., EMNLP 2018), https://yale-lily.github.io/spider -
  CC BY-SA 4.0. Files: data/dev_full.jsonl (the 1,034 official dev records with execution metadata added by
  spider-sql-v1), data/schemas_ddl_dev.json and data/tables_dev.json (derived from the dev databases and
  tables.json), database/*/*.sqlite (unmodified). Redistribution must keep this attribution and licence.
- **Official Spider evaluation code** (evaluation.py, process_sql.py), https://github.com/taoyds/spider at
  commit {commit} - Apache License 2.0, full text in code/official_eval/LICENSE. Unmodified.
"""


class BundleError(RuntimeError):
    pass


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def need(cond, msg):
    if not cond:
        raise BundleError(msg)


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def plan():
    """Validate the sources and return ({bundle path: source path or bytes}, facts). Writes nothing."""
    man_path = PROCESSED / "manifest.json"
    for p in (man_path, PROCESSED / "dev_full.jsonl", PROCESSED / "schemas_ddl.json", SPIDER / "tables.json",
              MODULE_SRC, EVAL_SRC / "SOURCE_COMMIT.txt", *(EVAL_SRC / n for n in PINNED)):
        need(p.is_file(), f"missing required source file: {p}")
    src_man = json.loads(man_path.read_text(encoding="utf-8"))
    need(src_man.get("version") == "spider-sql-v1", "processed manifest is not spider-sql-v1")

    dev_path = PROCESSED / "dev_full.jsonl"
    need(sha256_file(dev_path) == src_man["files_sha256"]["dev_full.jsonl"],
         "dev_full.jsonl hash differs from data/processed/spider_sql_v1/manifest.json")
    need(sha256_file(PROCESSED / "schemas_ddl.json") == src_man["files_sha256"]["schemas_ddl.json"],
         "schemas_ddl.json hash differs from the processed manifest")
    need(sha256_file(SPIDER / "tables.json") == src_man["inputs_sha256"]["tables.json"],
         "tables.json hash differs from the processed manifest")
    for name, h in PINNED.items():
        need(sha256_file(EVAL_SRC / name) == h, f"official_eval/{name} differs from the pinned upstream hash")
    need((EVAL_SRC / "SOURCE_COMMIT.txt").read_text(encoding="utf-8").strip() == EVAL_COMMIT,
         "official_eval/SOURCE_COMMIT.txt is not the pinned commit")

    recs = read_jsonl(dev_path)
    need(len(recs) == N_DEV, f"dev_full has {len(recs)} records, expected {N_DEV}")
    ids = [r["id"] for r in recs]
    need(len(set(ids)) == N_DEV, "duplicate ids in dev_full")
    need(ids == [f"spider_dev_{i:04d}" for i in range(N_DEV)], "dev_full ids are not spider_dev_0000..1033 in order")
    need([r["source"]["index"] for r in recs] == list(range(N_DEV)), "dev_full is not in dev.json order")
    need(all(r["partition"] == "dev" and r["source"]["file"] == "dev.json" for r in recs), "non-dev record in dev_full")
    need(all(r["sql"].strip() and "\t" not in r["sql"] and "\n" not in r["sql"] for r in recs),
         "gold SQL must be non-empty single lines without tabs (official evaluator format)")
    need(all(r["question"].strip() for r in recs), "empty question in dev_full")
    db_ids = sorted({r["db_id"] for r in recs})
    need(len(db_ids) == N_DBS, f"dev_full references {len(db_ids)} databases, expected {N_DBS}")

    ddl_all = json.loads((PROCESSED / "schemas_ddl.json").read_text(encoding="utf-8"))
    need(all(d in ddl_all and ddl_all[d] for d in db_ids), "a dev database has no DDL in schemas_ddl.json")
    ddl = {d: ddl_all[d] for d in db_ids}
    tables = json.loads((SPIDER / "tables.json").read_text(encoding="utf-8"))
    entries = [e for e in tables if e["db_id"] in set(db_ids)]
    need(sorted(e["db_id"] for e in entries) == db_ids, "tables.json must hold exactly one entry per dev database")

    files = {"data/dev_full.jsonl": dev_path,
             "data/schemas_ddl_dev.json": (json.dumps(ddl, indent=1, ensure_ascii=False) + "\n").encode("utf-8"),
             "data/tables_dev.json": (json.dumps(entries, indent=1, ensure_ascii=False) + "\n").encode("utf-8"),
             "code/spider_zs_sql.py": MODULE_SRC,
             "code/official_eval/evaluation.py": EVAL_SRC / "evaluation.py",
             "code/official_eval/process_sql.py": EVAL_SRC / "process_sql.py",
             "code/official_eval/LICENSE": EVAL_SRC / "LICENSE",
             "code/official_eval/SOURCE_COMMIT.txt": EVAL_SRC / "SOURCE_COMMIT.txt",
             "ATTRIBUTION.md": ATTRIBUTION.format(commit=EVAL_COMMIT).encode("utf-8")}
    for d in db_ids:
        src = SPIDER / "database" / d / f"{d}.sqlite"
        need(src.is_file(), f"missing dev database file {src}")
        files[f"database/{d}/{d}.sqlite"] = src
    for name in files:
        need(not DENY.search(name), f"refusing to package a denied path: {name}")
    for name, src in files.items():
        if name.endswith((".py", ".json", ".jsonl", ".md", ".txt")):
            text = src.decode("utf-8") if isinstance(src, bytes) else Path(src).read_text(encoding="utf-8")
            for pat in SECRET_PATTERNS:
                need(not re.search(pat, text), f"credential-like string ({pat}) in {name}")
    facts = {"records": N_DEV, "db_ids": db_ids,
             "records_per_db": {d: sum(r["db_id"] == d for r in recs) for d in db_ids},
             "hardness": {h: sum(r["hardness"] == h for r in recs) for h in ("easy", "medium", "hard", "extra")},
             "src_manifest": src_man}
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
        meta[name] = {"sha256": sha256_file(dest), "bytes": dest.stat().st_size, "source": origin}
    for name in ("data/dev_full.jsonl", "database/wta_1/wta_1.sqlite", "code/official_eval/evaluation.py"):
        if not isinstance(files[name], bytes):
            need(meta[name]["sha256"] == sha256_file(files[name]), f"copy of {name} is not byte-identical")
    src_man = facts["src_manifest"]
    manifest = {
        "bundle": out_dir.name,
        "version": version,
        "purpose": "zero-shot NL->SQL baseline on Spider dev (spider-sql-v1 dev_full); evaluation only, private",
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "builder": "scripts/sql_v1/prepare_kaggle_dev_bundle.py",
        "builder_sha256": sha256_file(Path(__file__)),
        "counts": {"records": facts["records"], "databases": len(facts["db_ids"]), "files": len(meta),
                   "records_per_db": facts["records_per_db"], "hardness_in_records": facts["hardness"]},
        "db_ids": facts["db_ids"],
        "provenance": {
            "dataset": "spider-sql-v1", "dataset_manifest_built_at": src_man.get("built_at"),
            "dataset_builder_code_sha256": src_man.get("code_sha256"),
            "dev_full_sha256_in_dataset_manifest": src_man["files_sha256"]["dev_full.jsonl"],
            "schemas_ddl_sha256_in_dataset_manifest": src_man["files_sha256"]["schemas_ddl.json"],
            "spider_inputs_sha256": {k: src_man["inputs_sha256"][k] for k in ("dev.json", "tables.json")},
            "databases_from": "data/raw/spider1/spider_data/database/<db>/<db>.sqlite (byte copies)",
            "official_eval": {"repo": "https://github.com/taoyds/spider", "commit": EVAL_COMMIT, "license": "Apache-2.0"},
            "data_license": src_man.get("license"),
        },
        "excluded_by_design": ["train.jsonl", "val_internal.jsonl", "dev_clean.jsonl", "rejected.jsonl",
                               "test_full.jsonl", "test_clean.jsonl", "test.json", "test_tables.json",
                               "test_database/", "non-dev databases", "MQL datasets", "Kaggle source datasets"],
        "files": meta,
    }
    (out_dir / "bundle_manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n",
                                                  encoding="utf-8")
    return manifest


def verify_dir(out_dir):
    """Re-check a written bundle against its own manifest and the allowlist rules. Returns the manifest."""
    out_dir = Path(out_dir)
    man = json.loads((out_dir / "bundle_manifest.json").read_text(encoding="utf-8"))
    on_disk = {p.relative_to(out_dir).as_posix() for p in out_dir.rglob("*") if p.is_file()} - {"bundle_manifest.json"}
    need(on_disk == set(man["files"]), f"tree != manifest: extra={sorted(on_disk - set(man['files']))} "
                                       f"missing={sorted(set(man['files']) - on_disk)}")
    for name, m in man["files"].items():
        need(not DENY.search(name), f"denied path in bundle: {name}")
        need(sha256_file(out_dir / name) == m["sha256"], f"hash mismatch: {name}")
    dbs = {n.split("/")[1] for n in on_disk if n.startswith("database/")}
    need(sorted(dbs) == man["db_ids"] and len(dbs) == N_DBS, "database set differs from db_ids")
    need(all(n.endswith(".sqlite") and n.count("/") == 2 for n in on_disk if n.startswith("database/")),
         "database/ may hold only <db>/<db>.sqlite")
    recs = read_jsonl(out_dir / "data/dev_full.jsonl")
    need(len(recs) == N_DEV and sorted({r["db_id"] for r in recs}) == man["db_ids"], "record count or db_ids changed")
    for d in man["db_ids"]:
        need((out_dir / "database" / d / f"{d}.sqlite").is_file(), f"database {d} referenced but missing")
    return man


def purge_python_caches(out_dir):
    """Delete __pycache__ directories and stray *.pyc / *.pyo under a bundle directory; return what was removed.

    A bundle is written from an explicit allowlist, so it never contains caches when built; they appear only when
    code is later imported out of the bundle directory. verify_dir stays strict and still rejects them.
    """
    out_dir = Path(out_dir)
    need((out_dir / "bundle_manifest.json").is_file(), f"{out_dir} is not a bundle")
    removed = []
    for d in sorted(p for p in out_dir.rglob("__pycache__") if p.is_dir()):
        removed += [f.relative_to(out_dir).as_posix() for f in sorted(d.rglob("*")) if f.is_file()]
        shutil.rmtree(d)
    for f in sorted(p for p in out_dir.rglob("*") if p.is_file() and p.suffix in (".pyc", ".pyo")):
        removed.append(f.relative_to(out_dir).as_posix())
        f.unlink()
    return removed


def write_zip(out_dir, zip_path, manifest):
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for name in sorted(list(manifest["files"]) + ["bundle_manifest.json"]):
            z.write(out_dir / name, f"{out_dir.name}/{name}")
    with zipfile.ZipFile(zip_path) as z:
        names = set(z.namelist())
        need(z.testzip() is None, "zip CRC check failed")
    expect = {f"{out_dir.name}/{n}" for n in list(manifest["files"]) + ["bundle_manifest.json"]}
    need(names == expect, f"zip contents differ from allowlist: {sorted(names ^ expect)[:10]}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", default="v1")
    ap.add_argument("--out-root", default=str(ROOT / "kaggle"))
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--no-zip", action="store_true")
    ap.add_argument("--verify", metavar="BUNDLE_DIR", help="only re-verify an existing bundle directory")
    ap.add_argument("--purge-caches", metavar="BUNDLE_DIR",
                    help="remove __pycache__/ and *.pyc left in an existing bundle directory, then re-verify it")
    a = ap.parse_args(argv)
    try:
        if a.purge_caches:
            removed = purge_python_caches(a.purge_caches)
            verify_dir(a.purge_caches)
            print(f"OK {a.purge_caches}: removed {len(removed)} cache file(s) {removed}")
            return 0
        if a.verify:
            man = verify_dir(a.verify)
            print(f"OK {a.verify}: {man['counts']['records']} records, {len(man['db_ids'])} databases, "
                  f"{man['counts']['files']} files")
            return 0
        name = f"spider_sql_v1_dev_bundle_{a.version}"
        out_dir, zip_path = Path(a.out_root) / name, Path(a.out_root) / f"{name}.zip"
        files, facts = plan()
        if out_dir.exists() or zip_path.exists():
            need(a.overwrite, f"{out_dir} or {zip_path} exists; pass --overwrite or a new --version")
            # only ever delete our own bundle
            need(not out_dir.exists() or (out_dir / "bundle_manifest.json").is_file(), f"{out_dir} is not a bundle")
            shutil.rmtree(out_dir, ignore_errors=True)
            zip_path.unlink(missing_ok=True)
        manifest = write_bundle(out_dir, files, facts, a.version)
        purge_python_caches(out_dir)
        verify_dir(out_dir)
        if not a.no_zip:
            write_zip(out_dir, zip_path, manifest)
        size = sum(m["bytes"] for m in manifest["files"].values())
        print(f"bundle: {out_dir}  ({manifest['counts']['files']} files + manifest, {size / 1e6:.1f} MB)")
        if not a.no_zip:
            print(f"zip:    {zip_path}  ({zip_path.stat().st_size / 1e6:.1f} MB)")
        print(f"records {manifest['counts']['records']}, databases {manifest['counts']['databases']}: "
              f"{', '.join(manifest['db_ids'])}")
        return 0
    except BundleError as e:
        print(f"BUNDLE ERROR: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
