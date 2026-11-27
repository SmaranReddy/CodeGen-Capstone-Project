"""Build the Spider 1.0 NL->SQL dataset (spider-sql-v1): filtered train, internal validation,
and the official dev/test sets in full and "clean" form.

Reads data/raw/spider1 read-only. Writes only to --out-dir (default data/processed/spider_sql_v1)
and to data/interim/spider_sql_v1/testsuite_variants/ (one populated test-suite variant for each
official DB that ships empty). Every rule is fixed here, before any model sees the data, and uses
only the gold data -- never model outputs.

Train rules (train_spider.json + train_others.json, 8,659 records):
  T1 gold SQL executes read-only on its DB within --timeout seconds        -> else rejected
     official DB empty -> executed on a populated test-suite variant instead (tier exec_ok_testsuite_variant);
     no variant (music_2) -> executed on the empty official DB, which checks names only (tier exec_ok_empty_db)
  T2 exact duplicate (db_id, question, query)                              -> later copies rejected
  T3 normalised question equals a dev or test question                     -> rejected (leakage)
  T4 same (db_id, normalised question) with gold results that differ       -> all members rejected
  T5 gold-label defect (label_defects(): placeholder slot, tokenised literal, = "null") -> rejected
  flags only (kept): empty_result, all_null_result, literal_ungrounded, short_question, hardness_unparsed
Internal validation: whole train_spider DBs (populated, not music_2), seeded shuffle of db_ids, added
until >= --val-fraction of train_spider records. Selection uses db_id only.

Dev / test rules (official evaluation sets, never trained on):
  *_full.jsonl  every official record, with execution metadata and flags (headline benchmark)
  *_clean.jsonl drops records whose gold SQL errors/times out, or carries a T5 label defect (confirmed against the DB)
  flags only: empty_result, all_null_result, limit_tie, question_in_train, literal_ungrounded, T5 defect codes

Usage: python scripts/sql_v1/build_spider_sql.py [--out-dir DIR] [--overwrite]
"""
import argparse
import hashlib
import json
import random
import re
import sqlite3
import sys
import time
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "spider1"))
import spider1_kit as K  # noqa: E402
from build_pool import norm_sql  # noqa: E402

VERSION = "spider-sql-v1"
SEED = 20261027
TS_ZIP = K.ROOT / "data/raw/spider1_testsuite_download/testsuite_database.zip"
TS_OUT = K.ROOT / "data/interim/spider_sql_v1/testsuite_variants"
TEST_DB_DIR = K.SPIDER / "test_database"
# official train DB -> test-suite DB name (verified: same 7 tables for geo/geography)
TS_NAME = {"academic": "academic", "geo": "geography", "imdb": "imdb",
           "restaurants": "restaurants", "scholar": "scholar", "yelp": "yelp"}
LICENSE = {"name": "CC BY-SA 4.0", "source": "https://yale-lily.github.io/spider (per docs/dataset_research_report.md §6; "
           "no licence file in spider_data.zip)"}


def norm_q(q):
    return re.sub(r"\s+", " ", q.strip().lower()).rstrip("?.! ")


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


# ---------------------------------------------------------------- databases
def official_db(db_id, split):
    base = TEST_DB_DIR if split == "test" else K.DB_DIR
    return base / db_id / f"{db_id}.sqlite"


def connect_ro(path):
    con = sqlite3.connect("file:" + Path(path).as_posix() + "?mode=ro", uri=True)
    con.text_factory = lambda b: b.decode("utf-8", errors="replace")
    return con


def total_rows(path):
    con = connect_ro(path)
    try:
        tabs = [r[0] for r in con.execute("select name from sqlite_master where type='table'")]
        return sum(con.execute(f'select count(*) from "{t}"').fetchone()[0] for t in tabs)
    finally:
        con.close()


def extract_variants(empty_dbs):
    """One populated test-suite variant per empty official DB: the first .sqlite in sorted order
    that has rows (the zip also ships the original, empty <db>.sqlite, which is skipped)."""
    out = {}
    if not TS_ZIP.exists():
        return out
    with zipfile.ZipFile(TS_ZIP) as z:
        names = [n for n in z.namelist() if n.endswith(".sqlite") and not n.startswith("__MACOSX")]
        for db in sorted(empty_dbs):
            ts = TS_NAME.get(db)
            cand = sorted(n for n in names if n.startswith(f"database/{ts}/")
                          and Path(n).stem != ts) if ts else []
            for name in cand:
                dest = TS_OUT / db / Path(name).name
                if not dest.exists():
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    with z.open(name) as src, open(dest, "wb") as dst:
                        dst.write(src.read())
                if total_rows(dest) > 0:
                    out[db] = {"path": dest, "variant": f"{ts}/{Path(name).stem}", "sha256": sha256_file(dest)}
                    break
                dest.unlink()
    return out


def run_timed(path, sql, timeout):
    con = connect_ro(path)
    deadline = time.monotonic() + timeout
    con.set_progress_handler(lambda: 1 if time.monotonic() > deadline else 0, 10000)
    t0 = time.monotonic()
    try:
        cur = con.execute(sql)
        cols = [d[0] for d in cur.description] if cur.description else []
        rows = [tuple(r) for r in cur.fetchall()]
        return {"status": "ok", "cols": cols, "rows": rows, "ms": round((time.monotonic() - t0) * 1000)}
    except sqlite3.OperationalError as e:
        st = "timeout" if "interrupted" in str(e) else "error"
        return {"status": st, "error": str(e)[:200], "ms": round((time.monotonic() - t0) * 1000)}
    except Exception as e:  # noqa: BLE001
        return {"status": "error", "error": f"{type(e).__name__}: {e}"[:200], "ms": round((time.monotonic() - t0) * 1000)}
    finally:
        con.close()


def _canon(v):
    return round(v, 6) if isinstance(v, float) else v


def result_hashes(rows):
    canon = [json.dumps([_canon(v) for v in r], default=str) for r in rows]
    ordered = hashlib.sha256("\n".join(canon).encode()).hexdigest()
    multiset = hashlib.sha256("\n".join(sorted(canon)).encode()).hexdigest()
    return ordered, multiset


def hardness(path, sql):
    ps, ev = K.official()
    try:
        schema = ps.Schema(ps.get_schema(str(path)))
        return ev.Evaluator().eval_hardness(ps.get_sql(schema, sql))
    except Exception:  # noqa: BLE001
        return None


def has_order_by(sql):
    return bool(K._top_level_positions(sql, "ORDER BY"))


def limit_tie(path, sql, timeout):
    """True when an ORDER BY ... LIMIT k gold cuts inside a tie (several answers are equally correct).
    Same keyed-query idea as spider1_kit.tie_analysis, but path-aware (test DBs live in test_database/).
    None when the query shape is not analyzable."""
    try:
        info = K.order_info(sql)
    except Exception:  # noqa: BLE001
        return None
    if not info["order_by"] or info["limit"] is None:
        return False
    if not info["analyzable"]:
        return None
    keyed = (f"SELECT {info['select_list']}, {info['order_expr']} {info['from_rest']} "
             f"ORDER BY {info['order_expr']} {info['direction']}")
    res = run_timed(path, keyed, timeout)
    if res["status"] != "ok":
        return None
    k, rows = info["limit"], res["rows"]
    return len(rows) > k > 0 and rows[k][-1] == rows[k - 1][-1]


# ---------------------------------------------------------------- literal grounding
_LIT = re.compile(r"'((?:[^']|'')*)'|\"([^\"]*)\"")
_VALUE_CACHE = {}


def db_text_values(path):
    key = str(path)
    if key not in _VALUE_CACHE:
        vals = set()
        con = connect_ro(path)
        try:
            for (t,) in con.execute("select name from sqlite_master where type='table'").fetchall():
                for c in [r[1] for r in con.execute(f'pragma table_info("{t}")')]:
                    try:
                        q = f'select distinct lower(trim("{c}")) from "{t}" where typeof("{c}")=\'text\' limit 200000'
                        vals.update(v for (v,) in con.execute(q) if v)
                    except sqlite3.Error:
                        pass
        finally:
            con.close()
        _VALUE_CACHE.clear()  # one DB at a time: records arrive grouped by DB, some DBs are large
        _VALUE_CACHE[key] = vals
    return _VALUE_CACHE[key]


def _sql_literals(sql):
    return [(a or b).replace("''", "'") for a, b in _LIT.findall(sql)]


def _in_db(lit, path):
    core = lit.strip("%").strip().lower()
    vals = db_text_values(path)
    return any(core in v for v in vals) if "%" in lit else core in vals


def missing_in_db(sql, path):
    """SQL string literals that match no text value in the DB (numbers and 1-char values skipped)."""
    out = set()
    for lit in _sql_literals(sql):
        core = lit.strip("%").strip().lower()
        if len(core) >= 2 and not re.fullmatch(r"[-+0-9.: /]+", core) and not _in_db(lit, path):
            out.add(lit)
    return out


def ungrounded_literals(question, sql, missing):
    """Literals found neither in the question nor in the DB."""
    ql = question.lower()
    return [lit for lit in _sql_literals(sql) if lit in missing
            and lit.strip("%").strip().lower() not in ql
            and lit.strip("%").strip().lower().replace("_", " ") not in ql]


_PH = r"[a-z]+(?:_[a-z]+)*[0-9]+"


def label_defects(question, sql, missing):
    """Gold-label defects found in the 2026-10-27 spot-check of literal_ungrounded rows (gold only).
    `missing` = literals absent from the DB (missing_in_db), or None when the DB is empty/unchecked.
    placeholder_literal   SQL value is an anonymised slot ('actor_name0') the question does not contain
    placeholder_question  question contains an anonymised slot ('keyphrase0') the SQL does not contain
    tokenized_literal     value re-joined from tokens ('Louisville ,  Kentucky') that matches no DB value
    null_string_compare   compares to the string "null", and the DB holds no such string (means IS NULL)"""
    ql, sl = question.lower(), sql.lower()
    lits = _sql_literals(sql)
    out = []
    if any(re.fullmatch(_PH, lit) and lit.lower() not in ql for lit in lits):
        out.append("placeholder_literal")
    if any(tok not in sl for tok in re.findall(rf"\b{_PH}\b", ql)):
        out.append("placeholder_question")
    if missing is not None:
        if any(re.search(r"\s,\s", lit) and lit in missing for lit in lits):
            out.append("tokenized_literal")
        if (re.search(r"""(=|!=|<>)\s*["']null["']""", sql, re.I)
                and any(lit.lower() == "null" and lit in missing for lit in lits)):
            out.append("null_string_compare")
    return out


# ---------------------------------------------------------------- build
def load_split(name):
    return K.load_json(K.SPIDER / f"{name}.json")


def make_record(rid, src_file, idx, ex, split, path, backend, timeout, tier_ok):
    res = run_timed(path, ex["query"], timeout)
    rec = {"id": rid, "source": {"file": src_file, "index": idx}, "db_id": ex["db_id"],
           "question": ex["question"], "sql": ex["query"], "flags": []}
    ex_meta = {"status": res["status"], "backend": backend}  # no timings: keeps output hashes reproducible
    if res["status"] == "ok":
        o, m = result_hashes(res["rows"])
        ex_meta.update(n_rows=len(res["rows"]), n_cols=len(res["cols"]), ordered_sha256=o, multiset_sha256=m,
                       order_sensitive=has_order_by(ex["query"]))
        if not res["rows"]:
            rec["flags"].append("empty_result")
        elif all(v is None for r in res["rows"] for v in r):
            rec["flags"].append("all_null_result")
        rec["tier"] = tier_ok
    else:
        ex_meta["error"] = res["error"]
        rec["tier"] = None
    rec["execution"] = ex_meta
    rec["hardness"] = hardness(official_db(ex["db_id"], split), ex["query"])
    if rec["hardness"] is None:
        rec["flags"].append("hardness_unparsed")
    if len(ex["question"].split()) < 3:
        rec["flags"].append("short_question")
    missing = missing_in_db(ex["query"], path) if res["status"] == "ok" and backend != "official_db_empty" else None
    defects = label_defects(ex["question"], ex["query"], missing)
    if defects:
        rec["label_defects"] = defects
        rec["flags"].extend(defects)
    if missing:
        lits = ungrounded_literals(ex["question"], ex["query"], missing)
        if lits:
            rec["flags"].append("literal_ungrounded")
            rec["ungrounded_literals"] = lits
    return rec, res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(K.ROOT / "data/processed/spider_sql_v1"))
    ap.add_argument("--timeout", type=float, default=30.0)
    ap.add_argument("--val-fraction", type=float, default=0.10)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="smoke test: first N records per split file")
    args = ap.parse_args()
    out = Path(args.out_dir)
    if out.exists() and any(out.iterdir()) and not args.overwrite:
        sys.exit(f"{out} exists; pass --overwrite to rebuild")

    raw = {n: load_split(n)[:args.limit] for n in ["train_spider", "train_others", "dev", "test"]}
    inputs = {f: sha256_file(K.SPIDER / f) for f in
              ["train_spider.json", "train_others.json", "dev.json", "test.json", "tables.json", "test_tables.json"]}
    train_dbs = sorted({e["db_id"] for n in ("train_spider", "train_others") for e in raw[n]})
    empty = {d for d in train_dbs if total_rows(official_db(d, "train")) == 0}
    variants = extract_variants(empty)
    print(f"empty train DBs: {sorted(empty)}; populated variants: {sorted(variants)}", flush=True)

    dev_q = {norm_q(e["question"]) for e in raw["dev"]}
    test_q = {norm_q(e["question"]) for e in raw["test"]}
    train_q_all = {norm_q(e["question"]) for n in ("train_spider", "train_others") for e in raw[n]}

    # ---- train
    funnel = Counter()
    kept, rejected, seen = [], [], set()
    results = {}
    i = 0
    for src in ("train_spider", "train_others"):
        for idx, ex in enumerate(raw[src]):
            rid = f"spider_train_{i:05d}"
            i += 1
            funnel["input"] += 1
            db = ex["db_id"]
            if db in variants:
                path, backend, tier = variants[db]["path"], f"testsuite_variant:{variants[db]['variant']}", "exec_ok_testsuite_variant"
            elif db in empty:
                path, backend, tier = official_db(db, "train"), "official_db_empty", "exec_ok_empty_db"
            else:
                path, backend, tier = official_db(db, "train"), "official_db", "exec_ok"
            rec, res = make_record(rid, f"{src}.json", idx, ex, "train", path, backend, args.timeout, tier)
            reason = None
            key = (db, ex["question"], ex["query"])
            if rec["execution"]["status"] != "ok":
                reason = f"T1_exec_{rec['execution']['status']}"
            elif key in seen:
                reason = "T2_exact_duplicate"
            elif norm_q(ex["question"]) in dev_q or norm_q(ex["question"]) in test_q:
                reason = "T3_question_in_" + ("dev" if norm_q(ex["question"]) in dev_q else "test")
            elif rec.get("label_defects"):
                reason = "T5_" + rec["label_defects"][0]
            seen.add(key)
            if reason:
                rec["reject_reason"] = reason
                rejected.append(rec)
                funnel[reason] += 1
            else:
                kept.append(rec)
            if i % 1000 == 0:
                print(f"train {i}/8659", flush=True)

    # T4 conflicting labels: same (db, normalised question), different normalised SQL, different results
    groups = defaultdict(list)
    for r in kept:
        groups[(r["db_id"], norm_q(r["question"]))].append(r)
    conflict_ids = set()
    for g in groups.values():
        if len({norm_sql(r["sql"]) for r in g}) < 2:
            continue
        sig = {(r["execution"]["ordered_sha256"] if r["execution"]["order_sensitive"] else r["execution"]["multiset_sha256"])
               for r in g}
        if len(sig) > 1 or any(r["tier"] == "exec_ok_empty_db" for r in g):
            conflict_ids.update(r["id"] for r in g)
    for r in [r for r in kept if r["id"] in conflict_ids]:
        r["reject_reason"] = "T4_conflicting_label"
        rejected.append(r)
        funnel["T4_conflicting_label"] += 1
    kept = [r for r in kept if r["id"] not in conflict_ids]

    # internal validation: whole populated train_spider DBs, db_id-only seeded selection
    ts_dbs = sorted({e["db_id"] for e in raw["train_spider"]} - empty)
    rng = random.Random(SEED)
    order = ts_dbs[:]
    rng.shuffle(order)
    n_ts = sum(1 for r in kept if r["source"]["file"] == "train_spider.json")
    target = args.val_fraction * n_ts
    per_db = Counter(r["db_id"] for r in kept)
    val_dbs, n_val = [], 0
    for d in order:
        if n_val >= target:
            break
        val_dbs.append(d)
        n_val += per_db[d]
    val_set = set(val_dbs)
    for r in kept:
        r["partition"] = "val_internal" if r["db_id"] in val_set else "train"
    train = [r for r in kept if r["partition"] == "train"]
    val = [r for r in kept if r["partition"] == "val_internal"]

    # ---- dev / test
    evalsets = {}
    for split in ("dev", "test"):
        full = []
        for idx, ex in enumerate(raw[split]):
            path = official_db(ex["db_id"], split)
            rec, res = make_record(f"spider_{split}_{idx:04d}", f"{split}.json", idx, ex, split, path,
                                   "official_db", args.timeout, "exec_ok")
            rec["partition"] = split
            if norm_q(ex["question"]) in train_q_all:
                rec["flags"].append("question_in_train")
            if res["status"] == "ok" and res["rows"] and limit_tie(path, ex["query"], args.timeout):
                rec["flags"].append("limit_tie")
            full.append(rec)
        clean = [r for r in full if r["execution"]["status"] == "ok" and not r.get("label_defects")]
        evalsets[split] = (full, clean)
        print(f"{split}: {len(full)} full, {len(clean)} clean", flush=True)

    # ---- schemas (DDL from the SQLite files actually used)
    ddl = {}
    for d in sorted({r["db_id"] for r in kept} | {e["db_id"] for s in ("dev", "test") for e in raw[s]}):
        con = connect_ro(official_db(d, "test"))
        try:
            ddl[d] = [s for (s,) in con.execute("select sql from sqlite_master where type='table' and sql is not null")]
        finally:
            con.close()

    # ---- write
    out.mkdir(parents=True, exist_ok=True)
    files = {"train.jsonl": train, "val_internal.jsonl": val, "rejected.jsonl": rejected,
             "dev_full.jsonl": evalsets["dev"][0], "dev_clean.jsonl": evalsets["dev"][1],
             "test_full.jsonl": evalsets["test"][0], "test_clean.jsonl": evalsets["test"][1]}
    for name, recs in files.items():
        K.write_jsonl(out / name, recs)
    (out / "schemas_ddl.json").write_text(json.dumps(ddl, indent=1, ensure_ascii=False), encoding="utf-8")

    def summary(recs):
        return {"records": len(recs), "databases": len({r["db_id"] for r in recs}),
                "tiers": dict(Counter(r["tier"] for r in recs)),
                "flags": dict(Counter(f for r in recs for f in r["flags"])),
                "hardness": dict(Counter(str(r["hardness"]) for r in recs)),
                "distinct_questions_norm": len({norm_q(r["question"]) for r in recs}),
                "distinct_db_sql_norm": len({(r["db_id"], norm_sql(r["sql"])) for r in recs})}

    manifest = {
        "version": VERSION,
        "builder": "scripts/sql_v1/build_spider_sql.py",
        "command": "python scripts/sql_v1/build_spider_sql.py",
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "code_sha256": sha256_file(Path(__file__)),
        "license": LICENSE,
        "inputs_sha256": inputs,
        "parameters": {"timeout_s": args.timeout, "val_fraction": args.val_fraction, "seed": SEED},
        "testsuite_variants": {d: {"variant": v["variant"], "sha256": v["sha256"]} for d, v in variants.items()},
        "empty_official_train_dbs": sorted(empty),
        "train_funnel": dict(funnel) | {"kept": len(kept), "train": len(train), "val_internal": len(val)},
        "val_internal_dbs": sorted(val_dbs),
        "splits": {"train": summary(train), "val_internal": summary(val), "rejected": summary(rejected),
                   "dev_full": summary(evalsets["dev"][0]), "dev_clean": summary(evalsets["dev"][1]),
                   "test_full": summary(evalsets["test"][0]), "test_clean": summary(evalsets["test"][1])},
        "rejected_by_reason": dict(Counter(r["reject_reason"] for r in rejected)),
        "files_sha256": {n: sha256_file(out / n) for n in list(files) + ["schemas_ddl.json"]},
        "usage": {
            "train.jsonl": "supervised fine-tuning",
            "val_internal.jsonl": "checkpoint selection / early stopping (DB-disjoint from train, dev and test)",
            "dev_full.jsonl": "headline benchmark, official evaluator; do not tune on it repeatedly",
            "dev_clean.jsonl": "dev minus golds that fail to execute or carry a T5 label defect; secondary report only",
            "test_full.jsonl": "final one-shot evaluation only",
            "test_clean.jsonl": "test minus golds that fail to execute or carry a T5 label defect; secondary report only",
        },
        "notes": [
            "Eval sets are not filtered beyond gold execution failure, so dev_full/test_full stay comparable to published Spider results.",
            "Flags are informational and were not used to drop eval records.",
            "The MQL-era artifacts (data/processed/spider1_nl_mql*.jsonl, data/splits*, data/rag, data/finetune/ft_v1) are untouched and unused; they are Spider-dev derived.",
        ],
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest["train_funnel"], indent=1))
    print(json.dumps(manifest["rejected_by_reason"], indent=1))


if __name__ == "__main__":
    main()
