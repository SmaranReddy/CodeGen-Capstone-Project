"""Zero-shot NL->SQL baseline on Spider dev (spider-sql-v1 dev_full, 1,034 records).

Shipped inside the Kaggle bundle as code/spider_zs_sql.py and imported by
notebooks/kaggle/spider_sql_v1_zero_shot_3b.ipynb. The same file is imported by the local tests, which
drive it with a fake generator (no model, no GPU). torch / transformers are never imported here: the
notebook passes in a generate function.

Pieces:
  * prompt: fixed template (PROMPT_VERSION) built from the question + the DB's CREATE TABLE statements only.
    The prompt view is built from {id, db_id, question}; gold SQL and execution metadata never reach it.
  * generation log: append-only JSONL keyed by id, flushed + fsynced per batch, resumable, refuses records
    written under a different run fingerprint.
  * scoring: the official Spider evaluator (taoyds/spider @ b7b5b8c, evaluation.py + process_sql.py, unmodified)
    called one example at a time, as its evaluate() loop does. Documented wrappers, comparison logic unchanged:
      - sqlite3.connect inside the evaluator is replaced by a read-only connection with a time limit
        (the original has no timeout and opens the DB read-write) and a lenient UTF-8 text decoder
        (wta_1 has non-UTF-8 bytes: without it Python 3 fails even gold-vs-gold on spider_dev_0455/0456);
      - an empty/failed prediction is replaced by EMPTY_SENTINEL, because evaluate() drops blank lines and
        would misalign every later row. The sentinel does not parse or execute, so it scores 0.
    An unofficial result-set match is reported next to the official numbers and labelled as such.

CLI (local, CPU only):
  python spider_zs_sql.py score  --bundle DIR --predictions predictions.jsonl --out DIR
  python spider_zs_sql.py verify --bundle DIR --outputs DIR
"""
import argparse
import contextlib
import hashlib
import importlib
import io
import json
import os
import re
import sqlite3
import sys
import time
import types
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

BUNDLE_NAME = "spider_sql_v1_dev_bundle"
N_EXPECTED = 1034
PROMPT_VERSION = "spider-sql-zs-v1"
EMPTY_SENTINEL = "INVALID_EMPTY_PREDICTION"
EXEC_TIMEOUT_S = 30.0
ROW_CAP = 1_000_000          # gold max is 20,662 rows; a longer result cannot match (see score_one)
FINAL_STATUSES = ("ok", "empty", "truncated", "timeout", "error")

SYSTEM = """You are an expert in SQLite. Given a database schema and a question, write one SQLite query that answers the question.

Rules:
- Use only the tables and columns that appear in the schema.
- Write a single read-only SELECT statement.
- Output only the SQL query: no explanation, no comments, no markdown."""

USER_TEMPLATE = """Database: {db_id}

Schema:
{schema}

Question: {question}

SQL:"""

# the "no SQL" dict that evaluation.evaluate() substitutes when a prediction does not parse
EMPTY_SQL = {"except": None, "from": {"conds": [], "table_units": []}, "groupBy": [], "having": [],
             "intersect": None, "limit": None, "orderBy": [], "select": [False, []], "union": None, "where": []}


# ---------------------------------------------------------------- utils
def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def sha256_text(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def canon(obj):
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, path)
    return path


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, ensure_ascii=False, default=str), encoding="utf-8", newline="\n")
    os.replace(tmp, path)
    return path


# ---------------------------------------------------------------- bundle
def find_bundle(search_root="/kaggle/input", override=None):
    """Return the bundle dir. Kaggle may add a nesting level when it extracts the zip, so search for the manifest."""
    if override:
        p = Path(override)
        if not (p / "bundle_manifest.json").is_file():
            raise FileNotFoundError(f"BUNDLE_DIR={p} has no bundle_manifest.json")
        return p
    hits = []
    for m in Path(search_root).rglob("bundle_manifest.json"):
        try:
            if json.loads(m.read_text(encoding="utf-8")).get("bundle", "").startswith(BUNDLE_NAME):
                hits.append(m.parent)
        except (OSError, ValueError):
            continue
    if len(hits) != 1:
        raise FileNotFoundError(f"expected exactly one {BUNDLE_NAME}* bundle under {search_root}, found {hits}")
    return hits[0]


def verify_bundle(bundle):
    """Every manifest file present with its hash, no extra files, the expected counts. Returns the manifest."""
    bundle = Path(bundle)
    man = json.loads((bundle / "bundle_manifest.json").read_text(encoding="utf-8"))
    files = man["files"]
    on_disk = {p.relative_to(bundle).as_posix() for p in bundle.rglob("*") if p.is_file()}
    on_disk -= {"bundle_manifest.json"}
    on_disk = {p for p in on_disk if "__pycache__" not in p}
    if on_disk != set(files):
        raise RuntimeError(f"bundle file set differs from manifest: extra={sorted(on_disk - set(files))} "
                           f"missing={sorted(set(files) - on_disk)}")
    for name, meta in files.items():
        if sha256_file(bundle / name) != meta["sha256"]:
            raise RuntimeError(f"hash mismatch: {name}")
    recs = read_jsonl(bundle / "data/dev_full.jsonl")
    if len(recs) != N_EXPECTED or man["counts"]["records"] != N_EXPECTED:
        raise RuntimeError(f"expected {N_EXPECTED} records, found {len(recs)}")
    dbs = sorted({r["db_id"] for r in recs})
    if dbs != man["db_ids"]:
        raise RuntimeError("db_ids differ from manifest")
    for d in dbs:
        if not (bundle / "database" / d / f"{d}.sqlite").is_file():
            raise RuntimeError(f"missing database {d}")
    return man


def load_dev(bundle):
    recs = read_jsonl(Path(bundle) / "data/dev_full.jsonl")
    ids = [r["id"] for r in recs]
    if len(recs) != N_EXPECTED or len(set(ids)) != N_EXPECTED:
        raise RuntimeError("dev_full must hold 1,034 unique ids")
    return recs


def prompt_view(records):
    """The only fields generation may see."""
    return [{"id": r["id"], "db_id": r["db_id"], "question": r["question"]} for r in records]


def load_ddl(bundle):
    return json.loads((Path(bundle) / "data/schemas_ddl_dev.json").read_text(encoding="utf-8"))


# ---------------------------------------------------------------- prompt
def schema_text(ddl_list):
    return "\n\n".join(s.strip() for s in ddl_list)


def build_messages(item, ddl):
    user = USER_TEMPLATE.format(db_id=item["db_id"], schema=schema_text(ddl[item["db_id"]]),
                                question=item["question"].strip())
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def template_sha():
    return sha256_text(PROMPT_VERSION + "\n" + SYSTEM + "\n" + USER_TEMPLATE)


def build_prompts(items, ddl):
    if any(set(it) != {"id", "db_id", "question"} for it in items):
        raise RuntimeError("prompt items must carry only id, db_id, question")
    return {it["id"]: build_messages(it, ddl) for it in items}


def _squash(s):
    return re.sub(r"\s+", " ", s).strip().lower()


def check_no_gold_leak(prompts, records):
    """Gold SQL (and its whitespace/case-normalised form) must not occur in any prompt."""
    leaks = []
    for r in records:
        text = canon(prompts[r["id"]])
        if r["sql"] in text or _squash(r["sql"]) in _squash(text):
            leaks.append(r["id"])
    if leaks:
        raise RuntimeError(f"gold SQL found in prompts: {leaks[:10]}")
    return True


# ---------------------------------------------------------------- output -> SQL
def first_statement(s):
    """Cut at the first ';' outside quotes."""
    quote_ch = None
    for i, ch in enumerate(s):
        if quote_ch:
            if ch == quote_ch:
                quote_ch = None
        elif ch in "'\"`":
            quote_ch = ch
        elif ch == ";":
            return s[:i]
    return s


def extract_sql(text):
    """Return (one-line SQL or "", extraction method). Newlines/tabs become spaces (the evaluator is line/tab based)."""
    text = text or ""
    m = re.search(r"```[ \t]*(?:sqlite|sql)?[ \t]*\n?(.*?)```", text, re.S | re.I)
    if m:
        body, how = m.group(1), "fenced"
    elif "```" in text:
        body, how = re.sub(r"^[ \t]*(?:sqlite|sql)?", "", text.split("```", 1)[1], flags=re.I), "unclosed_fence"
    else:
        body, how = text, "raw"
    body = re.sub(r"^\s*(?:sqlite|sql)\s*:\s*", "", body.strip(), flags=re.I)
    sql = re.sub(r"[\t\r\n]+", " ", first_statement(body)).strip()
    return sql, (how if sql else "empty")


# ---------------------------------------------------------------- generation log
def run_fingerprint(config):
    return sha256_text(canon(config))[:16]


class PredictionLog:
    """Append-only JSONL, one record per finished example, keyed by id. A later record for the same id is
    accepted only when the earlier one has status "error" (a retry)."""

    def __init__(self, path, expected_ids, fingerprint):
        self.path = Path(path)
        self.expected = list(expected_ids)
        self.expected_set = set(self.expected)
        self.fingerprint = fingerprint
        self.records = {}
        self.warnings = []
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            self._load()

    def _load(self):
        raw = self.path.read_bytes()
        lines = raw.split(b"\n")
        tail = lines.pop()  # b"" when the file ends with a newline
        if tail.strip():
            # crash mid-write: drop the partial last line, keep everything before it
            self.warnings.append(f"dropped truncated last line ({len(tail)} bytes)")
            with open(self.path, "r+b") as f:
                f.truncate(len(raw) - len(tail))
        for n, line in enumerate(lines, 1):
            if not line.strip():
                continue
            try:
                rec = json.loads(line.decode("utf-8"))
            except ValueError as e:
                raise RuntimeError(f"{self.path}:{n} is not valid JSON ({e}); refusing to resume") from e
            self._accept(rec, where=f"{self.path}:{n}")

    def _accept(self, rec, where=""):
        if rec.get("id") not in self.expected_set:
            raise RuntimeError(f"{where} unknown id {rec.get('id')}")
        if rec.get("fingerprint") != self.fingerprint:
            raise RuntimeError(f"{where} written under run fingerprint {rec.get('fingerprint')}, "
                               f"current is {self.fingerprint}: use a new OUT_DIR or restore the old config")
        if rec.get("status") not in FINAL_STATUSES:
            raise RuntimeError(f"{where} bad status {rec.get('status')}")
        old = self.records.get(rec["id"])
        if old is not None and old["status"] != "error":
            raise RuntimeError(f"{where} duplicate id {rec['id']}")
        self.records[rec["id"]] = rec

    def append(self, recs):
        for r in recs:
            self._accept(r, where="append")
        with open(self.path, "a", encoding="utf-8", newline="\n") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())

    def todo(self, retry_errors=True):
        return [i for i in self.expected
                if i not in self.records or (retry_errors and self.records[i]["status"] == "error")]

    def missing(self):
        return [i for i in self.expected if i not in self.records]


def _pred_record(item, out, fingerprint, batch_meta, latency_s):
    text = out.get("text", "")
    sql, how = extract_sql(text)
    finish = out.get("finish")
    if finish == "time":
        status = "timeout"
    elif not sql:
        status = "empty"
    elif finish == "length":
        status = "truncated"
    else:
        status = "ok"
    return {"id": item["id"], "db_id": item["db_id"], "pred_sql": sql, "raw_output": text, "extraction": how,
            "status": status, "finish": finish, "error": None,
            "n_prompt_tokens": out.get("n_prompt_tokens"), "n_new_tokens": out.get("n_new_tokens"),
            "latency_s": round(latency_s, 4), **batch_meta, "fingerprint": fingerprint, "generated_at": now()}


def _error_record(item, err, fingerprint, batch_meta, latency_s):
    return {"id": item["id"], "db_id": item["db_id"], "pred_sql": "", "raw_output": "", "extraction": "empty",
            "status": "error", "finish": None, "error": f"{type(err).__name__}: {str(err)[:500]}",
            "n_prompt_tokens": None, "n_new_tokens": None, "latency_s": round(latency_s, 4), **batch_meta,
            "fingerprint": fingerprint, "generated_at": now()}


def run_generation(items, prompts, generate_fn, log, batch_size, session_id, retry_errors=True,
                   on_batch=None, on_error=None):
    """generate_fn(list of message lists) -> list of {"text", "finish": eos|length|time, "n_new_tokens",
    "n_prompt_tokens"}. A failing batch is retried one example at a time; an example that still fails gets a
    status "error" record (never dropped). Latency is the batch wall time divided by the batch size."""
    by_id = {it["id"]: it for it in items}
    todo = [by_id[i] for i in log.todo(retry_errors)]
    for b0 in range(0, len(todo), batch_size):
        batch = todo[b0:b0 + batch_size]
        meta = {"session_id": session_id, "batch_key": f"{session_id}:{b0 // batch_size}", "batch_size": len(batch)}
        t0 = time.perf_counter()
        try:
            outs = generate_fn([prompts[it["id"]] for it in batch])
            if len(outs) != len(batch):
                raise RuntimeError(f"generate_fn returned {len(outs)} outputs for {len(batch)} inputs")
            wall = time.perf_counter() - t0
            recs = [_pred_record(it, o, log.fingerprint, {**meta, "batch_wall_s": round(wall, 4)}, wall / len(batch))
                    for it, o in zip(batch, outs)]
        except Exception as e:  # noqa: BLE001 -- retried singly below, then recorded
            if on_error:
                on_error(e)
            recs = []
            for it in batch:
                single = {"session_id": session_id, "batch_key": f"{session_id}:{b0 // batch_size}:{it['id']}",
                          "batch_size": 1, "retried_after": f"{type(e).__name__}"}
                t1 = time.perf_counter()
                try:
                    o = generate_fn([prompts[it["id"]]])[0]
                    w = time.perf_counter() - t1
                    recs.append(_pred_record(it, o, log.fingerprint, {**single, "batch_wall_s": round(w, 4)}, w))
                except Exception as e2:  # noqa: BLE001
                    if on_error:
                        on_error(e2)
                    w = time.perf_counter() - t1
                    recs.append(_error_record(it, e2, log.fingerprint, {**single, "batch_wall_s": round(w, 4)}, w))
        log.append(recs)
        if on_batch:
            on_batch(len(log.records), len(log.expected), recs)
    return log


def finalize_predictions(log, out_path):
    """Write predictions.jsonl in dev order. Raises if any id has no record."""
    missing = log.missing()
    if missing:
        raise RuntimeError(f"{len(missing)} examples have no prediction record, e.g. {missing[:5]}")
    recs = [log.records[i] for i in log.expected]
    write_jsonl(out_path, recs)
    return recs


# ---------------------------------------------------------------- official evaluator
def ensure_nltk():
    """process_sql.tokenize uses nltk.word_tokenize, which needs the punkt (or punkt_tab) data."""
    import nltk
    try:
        nltk.word_tokenize("select a from b")
    except LookupError:
        for res in ("punkt_tab", "punkt"):
            try:
                nltk.download(res, quiet=True)
            except Exception:  # noqa: BLE001
                pass
        nltk.word_tokenize("select a from b")  # raises if still unavailable
    return nltk.__version__


def ro_connect(db, timeout_s):
    """Read-only SQLite connection that aborts any statement running past the deadline."""
    uri = "file:" + quote(Path(db).resolve().as_posix(), safe="/:") + "?mode=ro"
    con = sqlite3.connect(uri, uri=True)
    # wta_1 holds non-UTF-8 bytes; Python 3's default text factory raises on them, even for gold SQL. Same
    # lenient decode as the spider-sql-v1 builder (the test-suite evaluator uses errors="ignore").
    con.text_factory = lambda b: b.decode("utf-8", errors="replace")
    deadline = time.monotonic() + timeout_s
    con.set_progress_handler(lambda: 1 if time.monotonic() > deadline else 0, 10_000)
    return con


class OfficialEval:
    def __init__(self, bundle, exec_timeout_s=EXEC_TIMEOUT_S):
        self.bundle = Path(bundle)
        self.timeout = exec_timeout_s
        self.nltk_version = ensure_nltk()
        eval_dir = str(self.bundle / "code/official_eval")
        if eval_dir not in sys.path:
            sys.path.insert(0, eval_dir)
        self.process_sql = importlib.import_module("process_sql")
        self.ev = importlib.import_module("evaluation")
        for mod in (self.process_sql, self.ev):
            if Path(mod.__file__).resolve().parent != Path(eval_dir).resolve():
                raise RuntimeError(f"{mod.__name__} imported from {mod.__file__}, not the bundle")
        # documented wrapper: the evaluator's sqlite3.connect -> read-only + time limit (pred and gold share a
        # connection inside eval_exec_match, so it gets 3x the pre-check budget)
        shim = types.SimpleNamespace(connect=lambda db, *a, **k: ro_connect(db, 3 * self.timeout))
        self.process_sql.sqlite3 = shim
        self.ev.sqlite3 = shim
        self.kmaps = self.ev.build_foreign_key_map_from_json(str(self.bundle / "data/tables_dev.json"))
        self.evaluator = self.ev.Evaluator()
        self._schemas = {}

    def db_path(self, db_id):
        return str(self.bundle / "database" / db_id / f"{db_id}.sqlite")

    def schema(self, db_id):
        if db_id not in self._schemas:
            self._schemas[db_id] = self.process_sql.Schema(self.process_sql.get_schema(self.db_path(db_id)))
        return self._schemas[db_id]

    def _run(self, db, sql):
        """(status, rows or None, error). status in ok | error | timeout | row_cap."""
        con = ro_connect(db, self.timeout)
        try:
            cur = con.execute(sql)
            rows = []
            while True:
                chunk = cur.fetchmany(10_000)
                if not chunk:
                    break
                rows.extend(chunk)
                if len(rows) > ROW_CAP:
                    return "row_cap", None, f"more than {ROW_CAP} rows"
            return "ok", rows, None
        except sqlite3.OperationalError as e:
            return ("timeout" if "interrupted" in str(e) else "error"), None, str(e)[:300]
        except Exception as e:  # noqa: BLE001
            return "error", None, f"{type(e).__name__}: {str(e)[:300]}"
        finally:
            con.close()

    def _prepares(self, db, sql):
        con = ro_connect(db, self.timeout)
        try:
            con.execute("EXPLAIN " + sql).fetchall()
            return True, None
        except Exception as e:  # noqa: BLE001
            return False, str(e)[:300]
        finally:
            con.close()

    def score_one(self, rec, pred_sql):
        """Mirror of one iteration of evaluation.evaluate(): same parse fallback, rebuild, exec and exact match."""
        ev, db_id, gold = self.ev, rec["db_id"], rec["sql"]
        db = self.db_path(db_id)
        eval_sql = pred_sql if pred_sql.strip() else EMPTY_SENTINEL
        out = {"id": rec["id"], "db_id": db_id, "eval_sql": eval_sql, "gold_sql": gold,
               "hardness": None, "official_parse_ok": None, "official_exact": 0, "official_exec": 0,
               "sqlite_prepare_ok": None, "sqlite_prepare_error": None, "exec_status": None, "exec_error": None,
               "pred_n_rows": None, "unofficial_result_match": 0, "evaluator_error": None}
        try:
            schema = self.schema(db_id)
            g_sql = ev.get_sql(schema, gold)
            out["hardness"] = self.evaluator.eval_hardness(g_sql)
            gold_ordered = bool(g_sql["orderBy"])
            try:
                p_sql = ev.get_sql(schema, eval_sql)
                out["official_parse_ok"] = True
            except Exception:  # noqa: BLE001 -- the official evaluator uses a bare except here
                p_sql = json.loads(json.dumps(EMPTY_SQL))
                out["official_parse_ok"] = False
            kmap = self.kmaps[db_id]
            g_valid = ev.build_valid_col_units(g_sql["from"]["table_units"], schema)
            g_sql = ev.rebuild_sql_col(g_valid, ev.rebuild_sql_val(g_sql), kmap)
            p_valid = ev.build_valid_col_units(p_sql["from"]["table_units"], schema)
            p_sql = ev.rebuild_sql_col(p_valid, ev.rebuild_sql_val(p_sql), kmap)

            out["sqlite_prepare_ok"], out["sqlite_prepare_error"] = self._prepares(db, eval_sql)
            status, p_rows, err = self._run(db, eval_sql)
            out["exec_status"], out["exec_error"] = status, err
            out["pred_n_rows"] = len(p_rows) if p_rows is not None else None
            # official execution match. It catches pred errors itself (-> False). Skipped only when our own
            # timed run already failed: timeout/error would fail there too, and a row_cap result has more rows
            # than any gold result, so its per-column lists cannot be equal.
            if status == "ok":
                out["official_exec"] = int(bool(ev.eval_exec_match(db, eval_sql, gold, p_sql, g_sql)))
            out["official_exact"] = int(bool(self.evaluator.eval_exact_match(p_sql, g_sql)))

            g_status, g_rows, g_err = self._run(db, gold)
            if g_status != "ok":
                raise RuntimeError(f"gold SQL failed: {g_status} {g_err}")
            if p_rows is not None:
                same = (p_rows == g_rows) if gold_ordered else (Counter(p_rows) == Counter(g_rows))
                out["unofficial_result_match"] = int(same)
        except Exception as e:  # noqa: BLE001 -- recorded; the example still counts in every denominator
            out["evaluator_error"] = f"{type(e).__name__}: {str(e)[:300]}"
        return out

    def official_cli_match(self, records, eval_sqls, work_dir):
        """Run evaluation.evaluate(etype="match") on the same gold/pred files the CLI would read.
        Execution is not run this way: the CLI path has no per-query guard against huge results."""
        work_dir = Path(work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
        gold_f, pred_f = work_dir / "official_gold.txt", work_dir / "official_pred.txt"
        for s in eval_sqls:
            if not s.strip() or "\t" in s or "\n" in s:
                raise RuntimeError("predictions must be non-empty single lines without tabs")
        gold_f.write_text("".join(f"{r['sql']}\t{r['db_id']}\n" for r in records), encoding="utf-8")
        pred_f.write_text("".join(s + "\n" for s in eval_sqls), encoding="utf-8")
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                self.ev.evaluate(str(gold_f), str(pred_f), str(self.bundle / "database"), "match", self.kmaps)
            error = None
        except Exception as e:  # noqa: BLE001
            error = f"{type(e).__name__}: {str(e)[:300]}"
        text = buf.getvalue()
        (work_dir / "official_eval_match_stdout.txt").write_text(text, encoding="utf-8")
        result = {"error": error, "exact_match_by_level": None}
        m = re.search(r"^exact match\s+(.*)$", text, re.M)
        cnt = re.search(r"^count\s+(.*)$", text, re.M)
        if m and cnt:
            levels = ["easy", "medium", "hard", "extra", "all"]
            result["exact_match_by_level"] = dict(zip(levels, map(float, m.group(1).split())))
            result["count_by_level"] = dict(zip(levels, map(int, cnt.group(1).split())))
        return result


def score_all(bundle, records, predictions, out_dir, exec_timeout_s=EXEC_TIMEOUT_S, progress=None):
    """Score every dev record. predictions: list in dev order (from predictions.jsonl)."""
    if [p["id"] for p in predictions] != [r["id"] for r in records]:
        raise RuntimeError("predictions are not exactly the dev ids in dev order")
    oe = OfficialEval(bundle, exec_timeout_s)
    results = []
    for k, (r, p) in enumerate(zip(records, predictions)):
        res = oe.score_one(r, p["pred_sql"] if p["status"] != "error" else "")
        res["gen_status"] = p["status"]
        res["hardness_record"] = r.get("hardness")
        results.append(res)
        if progress and (k + 1) % 100 == 0:
            progress(k + 1, len(records))
    write_jsonl(Path(out_dir) / "evaluation_results.jsonl", results)
    cli = oe.official_cli_match(records, [x["eval_sql"] for x in results], Path(out_dir) / "official_eval")
    return results, cli, {"nltk": oe.nltk_version}


# ---------------------------------------------------------------- summary
def _metric(num, den):
    return {"numerator": int(num), "denominator": int(den), "rate": round(num / den, 4) if den else None}


def summarize(predictions, results, cli, config, extra=None):
    n = len(results)
    complete = (n == N_EXPECTED and len(predictions) == N_EXPECTED
                and [p["id"] for p in predictions] == [r["id"] for r in results])
    gen_status = Counter(p["status"] for p in predictions)
    batches = {}
    for p in predictions:
        batches[p["batch_key"]] = p["batch_wall_s"]
    lat = [p["latency_s"] for p in predictions if p["status"] != "error"]
    s = lambda key: sum(int(bool(r[key])) for r in results)  # noqa: E731
    metrics = {
        "official_execution_accuracy": _metric(s("official_exec"), n),
        "official_exact_match": _metric(s("official_exact"), n),
        "official_parser_success": _metric(s("official_parse_ok"), n),
        "sqlite_prepare_success": _metric(s("sqlite_prepare_ok"), n),
        "sqlite_execution_success": _metric(sum(r["exec_status"] == "ok" for r in results), n),
        "unofficial_result_set_match": _metric(s("unofficial_result_match"), n),
    }
    by_level = {}
    for lvl in ("easy", "medium", "hard", "extra"):
        sub = [r for r in results if r["hardness"] == lvl]
        by_level[lvl] = {"count": len(sub),
                         "official_execution_accuracy": _metric(sum(r["official_exec"] for r in sub), len(sub)),
                         "official_exact_match": _metric(sum(r["official_exact"] for r in sub), len(sub))}
    em_cli = (cli or {}).get("exact_match_by_level") or {}
    cross = None
    if "all" in em_cli:
        mine = metrics["official_exact_match"]["rate"]
        cross = {"cli_exact_match_all": em_cli["all"], "per_example_exact_match_all": mine,
                 "agree_to_3dp": abs(round(mine, 3) - em_cli["all"]) < 1e-9}
    summary = {
        "status": "complete" if complete else "incomplete",
        "complete": complete,
        "n_expected": N_EXPECTED, "n_predictions": len(predictions), "n_evaluated": n,
        "metrics": metrics,
        "metrics_by_official_hardness": by_level,
        "generation": {
            "status_counts": dict(gen_status),
            "generation_failures": gen_status.get("error", 0),
            "timeouts": gen_status.get("timeout", 0),
            "truncated_at_max_new_tokens": gen_status.get("truncated", 0),
            "empty_outputs": gen_status.get("empty", 0),
            "mean_latency_s_per_example": round(sum(lat) / len(lat), 4) if lat else None,
            "latency_denominator": len(lat),
            "total_generation_wall_s": round(sum(batches.values()), 2),
            "n_batches": len(batches),
            "sessions": sorted({p["session_id"] for p in predictions}),
        },
        "evaluation": {
            "exec_status_counts": dict(Counter(r["exec_status"] for r in results)),
            "evaluator_errors": sum(r["evaluator_error"] is not None for r in results),
            "empty_sentinel_used": sum(r["eval_sql"] == EMPTY_SENTINEL for r in results),
            "official_cli_match": cli,
            "official_cli_crosscheck": cross,
        },
        "config": config,
        "created_at": now(),
    }
    if extra:
        summary.update(extra)
    return summary


def render_report(summary):
    m, g, c = summary["metrics"], summary["generation"], summary["config"]
    pct = lambda x: f"{100 * x['rate']:.1f}% ({x['numerator']}/{x['denominator']})" if x["rate"] is not None else "n/a"  # noqa: E731
    lines = [f"# Zero-shot baseline: {c.get('model_id')} on Spider dev (spider-sql-v1 dev_full)", "",
             f"Status: **{summary['status']}** - {summary['n_evaluated']}/{summary['n_expected']} examples evaluated.", "",
             "| Metric | Result |", "|---|---|",
             f"| Execution accuracy (official evaluation.py) | {pct(m['official_execution_accuracy'])} |",
             f"| Exact-set match (official evaluation.py) | {pct(m['official_exact_match'])} |",
             f"| Parsed by official process_sql | {pct(m['official_parser_success'])} |",
             f"| Compiles in SQLite (EXPLAIN) | {pct(m['sqlite_prepare_success'])} |",
             f"| Executes in SQLite | {pct(m['sqlite_execution_success'])} |",
             f"| Result-set match (unofficial, secondary) | {pct(m['unofficial_result_set_match'])} |",
             "", "| Hardness | n | EX (official) | EM (official) |", "|---|---:|---:|---:|"]
    for lvl, v in summary["metrics_by_official_hardness"].items():
        lines.append(f"| {lvl} | {v['count']} | {pct(v['official_execution_accuracy'])} | {pct(v['official_exact_match'])} |")
    lines += ["", "## Generation", "",
              f"- Status counts: {g['status_counts']}",
              f"- Generation failures: {g['generation_failures']}; timeouts: {g['timeouts']}; "
              f"truncated at max_new_tokens: {g['truncated_at_max_new_tokens']}; empty: {g['empty_outputs']}",
              f"- Mean latency per example (batch wall / batch size): {g['mean_latency_s_per_example']} s "
              f"over {g['latency_denominator']} examples; total generation wall time {g['total_generation_wall_s']} s "
              f"in {g['n_batches']} batches",
              "", "## Configuration", "", "```json", json.dumps(c, indent=1), "```", "",
              "## Limitations", "",
              "- Official EX is evaluation.py at commit b7b5b8c: it compares the returned rows column by column in "
              "returned order, keyed by the parsed SELECT items. A correct query the official parser cannot parse "
              "scores 0 on EX and EM. It is not the test-suite execution accuracy most papers report, so the numbers "
              "are not directly comparable to published Qwen2.5-Coder Spider scores.",
              "- Exact-set match ignores literal values (DISABLE_VALUE) and DISTINCT, as in the official script.",
              "- Wrappers around the official code, comparison logic unchanged: read-only SQLite connections with a "
              "time limit and a lenient UTF-8 decoder (wta_1 has non-UTF-8 bytes; without it Python 3 fails even "
              f"gold-vs-gold on 2 examples), and the sentinel `{EMPTY_SENTINEL}` for empty or failed generations "
              "(scores 0).",
              "- The result-set match is ours (ordered if the gold query has ORDER BY, else multiset); it is secondary.",
              "- Batched greedy decoding with left padding can differ slightly from batch size 1.",
              "- Zero-shot: schema DDL only, no sample rows, no few-shot examples, no self-correction.", ""]
    return "\n".join(lines)


def output_sums(out_dir, names):
    return {n: sha256_file(Path(out_dir) / n) for n in names if (Path(out_dir) / n).exists()}


# ---------------------------------------------------------------- local CLI
def _cmd_score(a):
    bundle = Path(a.bundle)
    verify_bundle(bundle)
    records = load_dev(bundle)
    preds = read_jsonl(a.predictions)
    out = Path(a.out)
    results, cli, versions = score_all(bundle, records, preds, out, a.exec_timeout,
                                       progress=lambda k, n: print(f"scored {k}/{n}", flush=True))
    summary = summarize(preds, results, cli, {"source": str(a.predictions), "scored_locally": True}, {"versions": versions})
    write_json(out / "baseline_summary.json", summary)
    print(json.dumps(summary["metrics"], indent=1))
    print("cross-check:", summary["evaluation"]["official_cli_crosscheck"])


def _cmd_verify(a):
    """Check a downloaded Kaggle output dir: completeness, ids/order, hashes, and that the summary
    is reproduced from evaluation_results.jsonl."""
    bundle, out = Path(a.bundle), Path(a.outputs)
    records = load_dev(bundle)
    status = json.loads((out / "RUN_STATUS.json").read_text(encoding="utf-8"))
    preds = read_jsonl(out / "predictions.jsonl")
    results = read_jsonl(out / "evaluation_results.jsonl")
    summary = json.loads((out / "baseline_summary.json").read_text(encoding="utf-8"))
    ids = [r["id"] for r in records]
    problems = []
    if status.get("status") != "complete":
        problems.append(f"RUN_STATUS is {status.get('status')}")
    if [p["id"] for p in preds] != ids:
        problems.append("predictions.jsonl ids != dev ids in order")
    if [r["id"] for r in results] != ids:
        problems.append("evaluation_results.jsonl ids != dev ids in order")
    if len({p["fingerprint"] for p in preds}) != 1 or preds[0]["fingerprint"] != summary["config"].get("fingerprint"):
        problems.append("predictions carry more than one run fingerprint, or not the summary's")
    for name, h in status.get("sha256", {}).items():
        if sha256_file(out / name) != h:
            problems.append(f"hash mismatch {name}")
    again = summarize(preds, results, summary["evaluation"]["official_cli_match"], summary["config"])
    if again["metrics"] != summary["metrics"]:
        problems.append("metrics do not recompute from evaluation_results.jsonl")
    print(json.dumps({"problems": problems, "metrics": again["metrics"]}, indent=1))
    if problems:
        sys.exit(1)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("score")
    s.add_argument("--bundle", required=True)
    s.add_argument("--predictions", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--exec-timeout", type=float, default=EXEC_TIMEOUT_S)
    s.set_defaults(fn=_cmd_score)
    v = sub.add_parser("verify")
    v.add_argument("--bundle", required=True)
    v.add_argument("--outputs", required=True)
    v.set_defaults(fn=_cmd_verify)
    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
