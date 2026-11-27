"""Test-suite execution accuracy (Zhong et al., 2020) for a Spider-dev SQL run. Development split only.

The decision logic is the official one, imported unmodified from the vendored
taoyds/test-suite-sql-eval @ e97acc5 (scripts/sql_v1/testsuite_official_e97acc5/): `postprocess`,
`remove_distinct`, `result_eq`, `replace_cur_year`, and the `p_str.replace("value", "1")` step of
`evaluate()`. A prediction is correct only if its denotation equals gold's on EVERY .sqlite file in
<db_root>/<db_id>/ (the original DB plus all test-suite variants), with gold and prediction executed on
the same file. Settings match the official defaults: values predicted (no --plug_value), DISTINCT
removed (no --keep_distinct; --keep-distinct is a sensitivity option).

Disclosed deviations from the official exec path (they change robustness, not the verdict rule):
  * read-only connections (mode=ro) instead of read-write;
  * a real 60 s per-query timeout (official TIMEOUT=60 wraps synchronous code in asyncio.wait_for, so it
    never fires); a prediction that times out counts as incorrect;
  * a gold failure on any file is recorded as `gold_error`/`gold_timeout` (not evaluable) instead of
    aborting the whole run with an assert;
  * `random` is seeded before each comparison (`get_constraint_permutation` samples rows; this only prunes
    impossible column permutations, so it affects speed, never the verdict);
  * variants are visited in a fixed order (original DB first, then sorted) and the check stops at the first
    failing file, as the official loop does.
`evaluate_test_suite_dev.py parity` runs the unmodified official evaluation.py on the same inputs.

Usage (python -I recommended):
  evaluate_test_suite_dev.py extract  [--db-root DIR]
  evaluate_test_suite_dev.py evaluate --run RUN_DIR --out OUT_DIR [--db-root DIR] [--keep-distinct]
                                      [--gold-as-pred] [--workers N] [--timeout S]
  evaluate_test_suite_dev.py parity   --run RUN_DIR --out OUT_DIR [--db-root DIR]
  evaluate_test_suite_dev.py compare  --run RUN_DIR --out EVAL_DIR   (EVAL_DIR holds predictions/, writes comparison.json)
"""
import argparse
import collections
import hashlib
import importlib.util
import json
import os
import platform
import random
import sqlite3
import subprocess
import sys
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from urllib.parse import quote

ROOT = Path(__file__).resolve().parents[2]
OFFICIAL_DIR = Path(__file__).resolve().parent / "testsuite_official_e97acc5"
OFFICIAL_COMMIT = "e97acc546ecbee8fa27fa8dbf025ef61493a876c"
OFFICIAL_SHA256 = {
    "evaluation.py": "7401e4014a8955376a7919c06903a7f0ab403c99e89f94204cd8f4c8e32ae779",
    "exec_eval.py": "29d034db28904490c28037537a14fbb0150b6e86cef0049076c0511d6b6b77f7",
    "parse.py": "ef04211a6e1c1e142571157f5c1999613e3451084c044083b2de1977f1f622c5",
    "process_sql.py": "927fc564f7a8e34f09f009a2f5564a83fdf95226440dde84c87871fd65fe55a1",
}
DEV_FILE = ROOT / "data/processed/spider_sql_v1/dev_full.jsonl"
ZIP = ROOT / "data/raw/spider1_testsuite_download/testsuite_database.zip"
# ~3.5 GB (wta_1 alone is 3.46 GB): kept outside the OneDrive-synced repo by default.
DEFAULT_DB_ROOT = Path(os.environ.get("SPIDER_TS_DB_ROOT", Path.home() / "codegen_data/spider1_testsuite_e97acc5/database"))
TIMEOUT_S = 60  # official exec_eval.TIMEOUT
PASS = "pass"
FAIL_STATUSES = ("mismatch", "pred_error", "pred_timeout", "pred_preprocess_error")
NOT_EVALUABLE = ("gold_error", "gold_timeout", "gold_preprocess_error", "missing_variants")


# ------------------------------------------------------------------ dev-only guard
class HeldOutDataError(RuntimeError):
    pass


def guard_dev(records, source=""):
    """Refuse anything that is not a Spider-dev record. Called on every input before any SQL runs."""
    if "test" in Path(str(source)).name.lower():
        raise HeldOutDataError(f"refusing held-out input file: {source}")
    for r in records:
        rid = str(r.get("id", ""))
        if not rid.startswith("spider_dev_") or r.get("partition", "dev") != "dev":
            raise HeldOutDataError(f"non-dev record {rid!r} (partition={r.get('partition')!r}) in {source}")
    return records


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


# ------------------------------------------------------------------ official code (loaded by file path)
_OFFICIAL = None


def load_official():
    """Import the vendored exec_eval/parse after checking their hashes. parse.py shares its name with the
    PyPI `parse` package, so it is registered under that name only while exec_eval imports it."""
    global _OFFICIAL
    if _OFFICIAL is not None:
        return _OFFICIAL
    for name in ("exec_eval.py", "parse.py"):
        got = sha256_file(OFFICIAL_DIR / name)
        if got != OFFICIAL_SHA256[name]:
            raise RuntimeError(f"vendored {name} sha256 {got} != pinned {OFFICIAL_SHA256[name]}")
    saved = sys.modules.get("parse")
    try:
        spec = importlib.util.spec_from_file_location("parse", OFFICIAL_DIR / "parse.py")
        parse = importlib.util.module_from_spec(spec)
        sys.modules["parse"] = parse
        spec.loader.exec_module(parse)
        spec = importlib.util.spec_from_file_location("ts_exec_eval", OFFICIAL_DIR / "exec_eval.py")
        exec_eval = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(exec_eval)
    finally:
        if saved is None:
            sys.modules.pop("parse", None)
        else:
            sys.modules["parse"] = saved
    _OFFICIAL = exec_eval
    return exec_eval


# ------------------------------------------------------------------ variants
def dev_db_ids(dev_records):
    return sorted({r["db_id"] for r in dev_records})


def zip_members(zf, db_ids):
    """database/<db>/<file>.sqlite members for the given dev DBs; __MACOSX resource forks are skipped."""
    want = set(db_ids)
    out = collections.defaultdict(list)
    for info in zf.infolist():
        parts = info.filename.split("/")
        if (len(parts) == 3 and parts[0] == "database" and parts[1] in want and parts[2].endswith(".sqlite")
                and not parts[2].startswith("._")):
            out[parts[1]].append(info)
    return out


def variant_paths(db_root, db_id):
    """Every file the official loop would use (`'.sqlite' in basename`), original first, then sorted.
    Returns [] if the directory or the original <db_id>.sqlite is missing."""
    d = Path(db_root) / db_id
    if not d.is_dir():
        return []
    files = sorted(p for p in d.iterdir() if p.is_file() and ".sqlite" in p.name)
    orig = [p for p in files if p.name == f"{db_id}.sqlite"]
    if not orig:
        return []
    return orig + [p for p in files if p.name != orig[0].name]


def extract(db_root, db_ids, zip_path=ZIP):
    """Idempotent: writes missing files (via .part + rename), verifies existing ones by sha256 against the
    zip's CRC-checked bytes, never overwrites. Returns the manifest."""
    db_root = Path(db_root)
    manifest = {"zip": str(zip_path), "zip_size": Path(zip_path).stat().st_size, "dbs": {}}
    with zipfile.ZipFile(zip_path) as zf:
        members = zip_members(zf, db_ids)
        for db in db_ids:
            (db_root / db).mkdir(parents=True, exist_ok=True)
            files = []
            for info in sorted(members.get(db, []), key=lambda i: i.filename):
                dest = db_root / db / Path(info.filename).name
                h = hashlib.sha256()
                if dest.exists():
                    with zf.open(info) as src:
                        for b in iter(lambda: src.read(1 << 20), b""):
                            h.update(b)
                    if dest.stat().st_size != info.file_size or sha256_file(dest) != h.hexdigest():
                        raise RuntimeError(f"{dest} differs from the zip member; not overwriting")
                else:
                    tmp = dest.with_name(dest.name + ".part")
                    with zf.open(info) as src, open(tmp, "wb") as out:
                        for b in iter(lambda: src.read(1 << 20), b""):
                            h.update(b)
                            out.write(b)
                    tmp.rename(dest)
                files.append({"name": dest.name, "size": info.file_size, "sha256": h.hexdigest()})
            manifest["dbs"][db] = files
    return manifest


def check_variants(db_root, db_ids, manifest=None):
    """Per-DB coverage: files present, original present, each file opens read-only and lists tables, and
    (when a manifest is given) sizes match it."""
    cov = {}
    for db in db_ids:
        paths = variant_paths(db_root, db)
        bad = []
        for p in paths:
            try:
                con = ro_connect(p, 30)
                con.execute("SELECT count(*) FROM sqlite_master WHERE type='table'").fetchone()
                con.close()
            except sqlite3.Error as e:
                bad.append(f"{p.name}: {e}")
        if manifest is not None:
            want = {f["name"]: f["size"] for f in manifest["dbs"].get(db, [])}
            have = {p.name: p.stat().st_size for p in paths}
            if want != have:
                bad.append(f"files differ from manifest: missing={sorted(set(want) - set(have))} "
                           f"extra={sorted(set(have) - set(want))}")
        cov[db] = {"n_files": len(paths), "n_variants_excl_original": max(0, len(paths) - 1),
                   "original_present": bool(paths), "unreadable": bad}
    return cov


# ------------------------------------------------------------------ execution
def ro_connect(path, timeout_s):
    uri = "file:" + quote(Path(path).resolve().as_posix(), safe="/:") + "?mode=ro"
    con = sqlite3.connect(uri, uri=True)
    con.text_factory = lambda b: b.decode(errors="ignore")  # as official get_cursor_from_path
    deadline = time.monotonic() + timeout_s
    con.set_progress_handler(lambda: 1 if time.monotonic() > deadline else 0, 10_000)
    return con


def exec_on_db(path, query, timeout_s):
    """('result', rows) | ('exception', msg) | ('timeout', msg). Same query rewriting as official
    exec_on_db_ (replace_cur_year)."""
    ee = load_official()
    query = ee.replace_cur_year(query)
    t0 = time.monotonic()
    con = None
    try:
        con = ro_connect(path, timeout_s)
        rows = con.execute(query).fetchall()
        return "result", rows
    except sqlite3.OperationalError as e:
        if "interrupted" in str(e) and time.monotonic() - t0 >= timeout_s:
            return "timeout", f"exceeded {timeout_s}s"
        return "exception", f"{type(e).__name__}: {e}"
    except Exception as e:  # noqa: BLE001 - any failure of the predicted SQL is an incorrect prediction
        return "exception", f"{type(e).__name__}: {e}"
    finally:
        if con is not None:
            con.close()


def prepare(p_str, g_str, keep_distinct):
    """Official string handling, in evaluate()/eval_exec_match order. Returns (p, g, order_matters, err)."""
    ee = load_official()
    p_str = p_str.replace("value", "1")  # evaluation.py evaluate(): applied to every prediction
    p_str, g_str = ee.postprocess(p_str), ee.postprocess(g_str)
    if not keep_distinct:
        try:
            g_str = ee.remove_distinct(g_str)
        except Exception as e:  # noqa: BLE001
            return None, None, None, ("gold_preprocess_error", f"{type(e).__name__}: {e}")
        try:
            p_str = ee.remove_distinct(p_str)
        except Exception as e:  # noqa: BLE001
            return None, g_str, None, ("pred_preprocess_error", f"{type(e).__name__}: {e}")
    return p_str, g_str, "order by" in g_str.lower(), None


def judge(paths, p_str, g_str, keep_distinct=False, timeout_s=TIMEOUT_S):
    """Test-suite verdict for one example over `paths` (gold and prediction always on the same file)."""
    ee = load_official()
    res = {"n_files": len(paths), "n_checked": 0, "order_matters": None, "fail_file": None, "detail": None}
    if not paths:
        return {**res, "status": "missing_variants", "ts_exec": None}
    p, g, order_matters, err = prepare(p_str, g_str, keep_distinct)
    res["order_matters"] = order_matters
    if err is not None and err[0] == "gold_preprocess_error":
        return {**res, "status": err[0], "ts_exec": None, "detail": err[1]}
    for path in paths:
        g_flag, g_rows = exec_on_db(path, g, timeout_s)
        if g_flag != "result":
            st = "gold_timeout" if g_flag == "timeout" else "gold_error"
            return {**res, "status": st, "ts_exec": None, "fail_file": path.name, "detail": g_rows}
        if err is not None:  # pred could not be pre-processed; gold ran fine on the first file
            return {**res, "status": err[0], "ts_exec": 0, "n_checked": 1, "fail_file": path.name,
                    "detail": err[1]}
        p_flag, p_rows = exec_on_db(path, p, timeout_s)
        res["n_checked"] += 1
        if p_flag != "result":
            st = "pred_timeout" if p_flag == "timeout" else "pred_error"
            return {**res, "status": st, "ts_exec": 0, "fail_file": path.name, "detail": p_rows}
        random.seed(0)
        if not ee.result_eq(g_rows, p_rows, order_matters=order_matters):
            return {**res, "status": "mismatch", "ts_exec": 0, "fail_file": path.name,
                    "detail": f"gold_rows={len(g_rows)} pred_rows={len(p_rows)}"}
    return {**res, "status": PASS, "ts_exec": 1}


def _judge_task(args):
    ex, db_root, keep_distinct, timeout_s = args
    paths = variant_paths(db_root, ex["db_id"])
    out = judge(paths, ex["pred"], ex["gold"], keep_distinct, timeout_s)
    out["orig_db_pass"] = (None if out["status"] in NOT_EVALUABLE
                           else out["status"] == PASS or out["fail_file"] != f"{ex['db_id']}.sqlite")
    return {"id": ex["id"], "db_id": ex["db_id"], "hardness": ex["hardness"],
            "value_replaced": "value" in ex["pred"], **out}


# ------------------------------------------------------------------ run inputs
def load_inputs(run_dir, dev_path=DEV_FILE, gold_as_pred=False):
    """Pairs each SQL-0001 eval_sql (the string the official evaluator saw) with its dev gold."""
    dev = guard_dev(read_jsonl(dev_path), dev_path)
    res = guard_dev(read_jsonl(Path(run_dir) / "evaluation_results.jsonl"), run_dir)
    if [d["id"] for d in dev] != [r["id"] for r in res]:
        raise RuntimeError("run ids do not match dev_full order")
    exs = []
    for d, r in zip(dev, res):
        if r["gold_sql"] != d["sql"] or r["db_id"] != d["db_id"]:
            raise RuntimeError(f"{d['id']}: run gold/db differ from dev_full")
        exs.append({"id": d["id"], "db_id": d["db_id"], "hardness": d["hardness"], "gold": d["sql"],
                    "pred": d["sql"] if gold_as_pred else r["eval_sql"]})
    return exs, res


def summarise(rows):
    n = len(rows)
    st = collections.Counter(r["status"] for r in rows)
    evaluable = [r for r in rows if r["status"] not in NOT_EVALUABLE]
    hits = sum(r["ts_exec"] == 1 for r in rows)
    by_h = {}
    for h in ("easy", "medium", "hard", "extra"):
        hr = [r for r in rows if r["hardness"] == h]
        by_h[h] = {"n": len(hr), "ts_exec": sum(r["ts_exec"] == 1 for r in hr)}
    by_db = {}
    for r in rows:
        b = by_db.setdefault(r["db_id"], {"n": 0, "ts_exec": 0, "n_files": r["n_files"]})
        b["n"] += 1
        b["ts_exec"] += r["ts_exec"] == 1
    return {"n_examples": n, "ts_exec": hits, "ts_exec_rate": round(hits / n, 4) if n else None,
            "n_evaluable": len(evaluable), "n_not_evaluable": n - len(evaluable),
            "status_counts": dict(sorted(st.items())), "by_hardness": by_h, "by_db": dict(sorted(by_db.items())),
            "value_replaced": sum(r["value_replaced"] for r in rows)}


def versions():
    import sqlparse
    return {"python": platform.python_version(), "sqlite": sqlite3.sqlite_version, "sqlparse": sqlparse.__version__,
            "official_commit": OFFICIAL_COMMIT, "official_files_sha256": OFFICIAL_SHA256}


def new_out_dir(out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise SystemExit(f"{out} must be empty (artifacts are never overwritten)")
    return out


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ------------------------------------------------------------------ CLI
def cmd_extract(a):
    dev = guard_dev(read_jsonl(DEV_FILE), DEV_FILE)
    ids = dev_db_ids(dev)
    man = extract(a.db_root, ids)
    man["coverage"] = check_variants(a.db_root, ids, man)
    (Path(a.db_root).parent / "dev_manifest.json").write_text(json.dumps(man, indent=1), encoding="utf-8")
    for db, c in man["coverage"].items():
        print(db, c["n_files"], "files", "UNREADABLE: " + "; ".join(c["unreadable"]) if c["unreadable"] else "")


def cmd_evaluate(a):
    out = new_out_dir(a.out)
    load_official()
    exs, _ = load_inputs(a.run, gold_as_pred=a.gold_as_pred)
    ids = sorted({e["db_id"] for e in exs})
    man_path = Path(a.db_root).parent / "dev_manifest.json"
    man = json.loads(man_path.read_text(encoding="utf-8")) if man_path.exists() else None
    cov = check_variants(a.db_root, ids, man)
    t0 = time.time()
    tasks = [(e, str(a.db_root), a.keep_distinct, a.timeout) for e in exs]
    if a.workers > 1:
        with ProcessPoolExecutor(a.workers) as pool:
            rows = list(pool.map(_judge_task, tasks, chunksize=4))
    else:
        rows = [_judge_task(t) for t in tasks]
    config = {"run": str(Path(a.run).resolve().relative_to(ROOT)) if Path(a.run).resolve().is_relative_to(ROOT)
              else str(a.run), "dev_file": str(DEV_FILE.relative_to(ROOT)), "db_root": str(a.db_root),
              "manifest": str(man_path) if man else None, "plug_value": False, "keep_distinct": a.keep_distinct,
              "gold_as_pred": a.gold_as_pred, "timeout_s": a.timeout, "workers": a.workers,
              "variant_order": "original first, then sorted by file name; stop at first failing file",
              "connection": "sqlite3 mode=ro, text_factory decode(errors='ignore')"}
    summary = {"metric": "test-suite execution accuracy (Zhong et al. 2020), e97acc5 exec path",
               "config": config, "versions": versions(), "coverage": cov, **summarise(rows),
               "elapsed_s": round(time.time() - t0, 1)}
    write_jsonl(out / "testsuite_results.jsonl", rows)
    (out / "testsuite_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    if man is not None:
        (out / "dev_manifest.json").write_text(json.dumps(man, indent=1), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("ts_exec", "n_examples", "n_not_evaluable", "status_counts",
                                              "by_hardness", "elapsed_s")}, indent=1))


def cmd_parity(a):
    """Run the unmodified official evaluation.py (exec only) on the same gold/pred strings."""
    out = new_out_dir(a.out)
    exs, _ = load_inputs(a.run)
    for name in ("evaluation.py", "process_sql.py"):
        if sha256_file(OFFICIAL_DIR / name) != OFFICIAL_SHA256[name]:
            raise RuntimeError(f"vendored {name} changed")
    gold, pred = (out / "gold.txt").resolve(), (out / "pred.txt").resolve()
    gold.write_text("".join(f"{e['gold']}\t{e['db_id']}\n" for e in exs), encoding="utf-8", newline="\n")
    pred.write_text("".join(f"{e['pred']}\n" for e in exs), encoding="utf-8", newline="\n")
    argv = [str(OFFICIAL_DIR / "evaluation.py"), "--gold", str(gold), "--pred", str(pred), "--db",
            str(Path(a.db_root).resolve()), "--etype", "exec"]
    # The unmodified script runs as __main__; faulthandler dumps stacks to stderr if it stalls. Its
    # exec_eval calls asyncio.run() twice per DB file; on Windows each call builds a new Proactor loop with a
    # loopback socketpair, and on 2026-10-28 a run blocked in that socketpair's accept() within its first
    # 10 examples (root cause not established). asyncio.run is therefore pointed at one persistent loop;
    # the coroutines run unchanged.
    boot = ("import asyncio, faulthandler, runpy, sys; faulthandler.dump_traceback_later(600, repeat=True); "
            "_loop = asyncio.new_event_loop(); asyncio.run = lambda coro: _loop.run_until_complete(coro); "
            f"sys.path.insert(0, {str(OFFICIAL_DIR)!r}); sys.argv = {argv!r}; "
            f"runpy.run_path({argv[0]!r}, run_name='__main__')")
    cmd = [sys.executable, "-u", "-X", "utf8", "-c", boot]
    (out / "command.json").write_text(json.dumps({"cmd": cmd, "versions": versions()}, indent=1), encoding="utf-8")
    with open(out / "official_stdout.txt", "w", encoding="utf-8") as so, open(out / "official_stderr.txt", "w",
                                                                               encoding="utf-8") as se:
        rc = subprocess.run(cmd, stdout=so, stderr=se, cwd=str(OFFICIAL_DIR)).returncode
    print("official evaluation.py exit code", rc)


def compare(ts_rows, run_rows, audit_rows=(), variant_rows=(), kd_rows=None):
    """Cross-tabs of test-suite EX against the SQL-0001 per-example metrics and the parser audit. Read-only;
    no historical metric is recomputed or redefined."""
    ts = {r["id"]: r for r in ts_rows}
    run = {r["id"]: r for r in run_rows}
    assert set(ts) == set(run)
    audit = {r["id"]: r for r in audit_rows}
    ids = sorted(ts)

    def tab(key):
        c = collections.Counter((key(run[i]), ts[i]["ts_exec"]) for i in ids)
        return {f"{k}|ts={t}": n for (k, t), n in sorted(c.items(), key=str)}

    def idlist(pred):
        return [i for i in ids if pred(i)]

    off_pass_ts_fail = idlist(lambda i: run[i]["official_exec"] == 1 and ts[i]["ts_exec"] != 1)
    rej_ts_pass = idlist(lambda i: not run[i]["official_parse_ok"] and ts[i]["ts_exec"] == 1)
    rej_match = idlist(lambda i: not run[i]["official_parse_ok"] and run[i]["unofficial_result_match"] == 1)
    parsed_off0_ts1 = idlist(lambda i: run[i]["official_parse_ok"] and run[i]["official_exec"] == 0
                             and ts[i]["ts_exec"] == 1)
    out = {
        "official_exec_x_ts": tab(lambda r: f"official_exec={r['official_exec']}"),
        "official_exact_x_ts": tab(lambda r: f"official_exact={r['official_exact']}"),
        "parse_ok_x_ts": tab(lambda r: f"parse_ok={r['official_parse_ok']}"),
        "unofficial_match_x_ts": tab(lambda r: f"unofficial_match={r['unofficial_result_match']}"),
        "sqlite_exec_x_ts": tab(lambda r: f"exec_status={r['exec_status']}"),
        "official_pass_ts_fail": {"n": len(off_pass_ts_fail),
                                  "fail_on_original_db": sum(not ts[i]["orig_db_pass"] for i in off_pass_ts_fail),
                                  "ids": off_pass_ts_fail},
        "parser_rejected_ts_pass": {"n": len(rej_ts_pass), "ids": rej_ts_pass,
                                    "by_audit_category": dict(collections.Counter(
                                        audit.get(i, {}).get("category") for i in rej_ts_pass))},
        "parser_rejected_by_audit_category_x_ts": dict(sorted(collections.Counter(
            f"{audit.get(i, {}).get('category')}|ts={ts[i]['ts_exec']}" for i in ids
            if not run[i]["official_parse_ok"]).items())),
        "parser_rejected_result_match_185": {"n": len(rej_match),
                                             "ts_pass": sum(ts[i]["ts_exec"] == 1 for i in rej_match)},
        "parsed_official0_ts1": {"n": len(parsed_off0_ts1), "ids": parsed_off0_ts1},
        "fail_both": sum(run[i]["official_exec"] == 0 and ts[i]["ts_exec"] == 0 for i in ids),
        "orig_db_pass_x_unofficial_match": dict(collections.Counter(
            f"orig_pass={ts[i]['orig_db_pass']}|unofficial={run[i]['unofficial_result_match']}" for i in ids)),
        "ts_fail_status": dict(collections.Counter(ts[i]["status"] for i in ids if ts[i]["ts_exec"] != 1)),
        "mismatch_on_original_vs_variant": dict(collections.Counter(
            "original" if not ts[i]["orig_db_pass"] else "variant_only" for i in ids if ts[i]["status"] == "mismatch")),
    }
    if variant_rows:  # parser audit's 3-variant check (column-order sensitive, DISTINCT kept)
        out["audit_3variant_x_ts"] = {}
        for g in sorted({v["group"] for v in variant_rows}):
            c = collections.Counter(f"{v['variant']['status']}|ts={ts[v['id']]['ts_exec']}"
                                    for v in variant_rows if v["group"] == g)
            out["audit_3variant_x_ts"][g] = dict(sorted(c.items()))
    if kd_rows is not None:
        kd = {r["id"]: r for r in kd_rows}
        out["keep_distinct"] = {"ts_exec": sum(kd[i]["ts_exec"] == 1 for i in ids),
                                "lost_vs_default": idlist(lambda i: ts[i]["ts_exec"] == 1 and kd[i]["ts_exec"] != 1),
                                "gained_vs_default": idlist(lambda i: ts[i]["ts_exec"] != 1 and kd[i]["ts_exec"] == 1)}
    return out


def cmd_compare(a):
    out = Path(a.out)
    target = out / "comparison.json"
    if target.exists():
        raise SystemExit(f"{target} exists (artifacts are never overwritten)")
    ts_rows = guard_dev(read_jsonl(out / "predictions/testsuite_results.jsonl"))
    run_rows = guard_dev(read_jsonl(Path(a.run) / "evaluation_results.jsonl"), a.run)
    audit_dir = Path(a.run).parent / "parser_audit"
    audit = read_jsonl(audit_dir / "parser_rejections.jsonl") if (audit_dir / "parser_rejections.jsonl").exists() else []
    var = read_jsonl(audit_dir / "variant_checks.jsonl") if (audit_dir / "variant_checks.jsonl").exists() else []
    kd_path = out / "predictions_keep_distinct/testsuite_results.jsonl"
    kd = read_jsonl(kd_path) if kd_path.exists() else None
    res = compare(ts_rows, run_rows, audit, var, kd)
    target.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if not isinstance(v, dict) or "ids" not in v}, indent=1))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("extract", "evaluate", "parity", "compare"):
        s = sub.add_parser(name)
        s.add_argument("--db-root", type=Path, default=DEFAULT_DB_ROOT)
        if name != "extract":
            s.add_argument("--run", type=Path, required=True)
            s.add_argument("--out", type=Path, required=True)
        if name == "evaluate":
            s.add_argument("--keep-distinct", action="store_true")
            s.add_argument("--gold-as-pred", action="store_true")
            s.add_argument("--workers", type=int, default=4)
            s.add_argument("--timeout", type=int, default=TIMEOUT_S)
    a = ap.parse_args(argv)
    {"extract": cmd_extract, "evaluate": cmd_evaluate, "parity": cmd_parity, "compare": cmd_compare}[a.cmd](a)


if __name__ == "__main__":
    main()
