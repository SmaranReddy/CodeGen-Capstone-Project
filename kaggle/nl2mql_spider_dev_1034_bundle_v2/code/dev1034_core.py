"""NL->MQL cross-engine evaluation on Spider dev (experiment MQL-0006).

Evaluation set: the 1,034 Spider dev records MINUS every record on the databases assigned to the held-out MQL
test partition (database-disjoint, data/splits_expanded/split_manifest.json). The set (ids, order, databases) is
fixed by the bundle manifest; nothing in this module hardcodes its size.

Shipped inside the private Kaggle bundle as code/dev1034_core.py and imported by
notebooks/kaggle/nl2mql_spider_dev_1034_model_comparison.ipynb. The same file is imported by the bundle
builder, the local verifier and the tests. torch / transformers are never imported here; pymongo / bson are
imported lazily, only by the Atlas functions.

Protocol (identical for every model; see docs/nl2mql_spider_dev_1034_model_comparison.md):
  prompt     nl-mql-spider1-v1, produced by the unmodified prompt_spider1.py from (db_id, question) only.
             Prompts are frozen in the bundle at build time and re-derived + compared before any model runs.
  reference  gold SQL executed read-only on the bundled SQLite file; the rows must reproduce the
             ordered/multiset sha256 that spider-sql-v1 recorded. Tie blocks from spider1_kit.tie_analysis.
  safety     parse_output -> validate_mql (eval_kit, byte-for-byte copies below; a test checks the AST
             against the repo) -> forbidden_anywhere (run_eval) -> extra_safety (this file: no cross-db
             `db` keys, non-string `from`, oversized or too deep pipelines).
  execution  Atlas aggregate with maxTimeMS and a bounded cursor, on the item's own collections only,
             as the read-only user.
  comparison PRIMARY: spider1_kit.docs_to_rows + spider1_kit.exec_match, unchanged (the MQL-0001..0005
             comparator). SECONDARY (reported only): the same after bson_prenormalize().
  outcome    correct | incorrect | invalid | timeout | failed | not_comparable (see classify docs below).

Every metric uses the full evaluation-set denominator (N in the bundle manifest); nothing is silently excluded.
"""
import gzip
import hashlib
import importlib.util
import json
import math
import os
import random
import re
import sqlite3
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

EXPERIMENT_ID = "MQL-0006"
BUNDLE_NAME = "nl2mql_spider_dev_1034_bundle"
N_SOURCE = 1034                  # spider-sql-v1 dev_full; the evaluation set is a fixed, ordered subset of it
ATLAS_DB = "codegen_spider1"
HARDNESS = ("easy", "medium", "hard", "extra")
OUTCOMES = ("correct", "incorrect", "invalid", "timeout", "failed", "not_comparable")
EXEC_TIMEOUT_MS = 30000          # as configs/evaluation_spider1_expanded.yaml
MAX_RESULT_ROWS = 25000          # gold max is 20,662 rows; any larger result cannot match its gold
MAX_PIPELINE_CHARS = 50000
MAX_PIPELINE_DEPTH = 40
SQLITE_TIMEOUT_S = 60.0
BOOTSTRAP_REPS = 10000
BOOTSTRAP_SEED = 20261029
# expected invalid-UTF-8 strings that Atlas stores U+FFFD-replaced (parity.json 2026-10-20)
EXPECTED_UTF8_REPLACED = {"wta_1__players": 2}

# ---------------------------------------------------------------- small utils
def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def sha256_text(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def canon(obj):
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str)


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def read_jsonl(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    if str(path).endswith(".gz"):
        # mtime=0 -> byte-reproducible gzip
        with open(tmp, "wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as gz:
            for r in records:
                gz.write((json.dumps(r, ensure_ascii=False, default=str) + "\n").encode("utf-8"))
    else:
        with open(tmp, "w", encoding="utf-8", newline="\n") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
    os.replace(tmp, path)
    return path


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, ensure_ascii=False, default=str) + "\n", encoding="utf-8", newline="\n")
    os.replace(tmp, path)
    return path


class PreflightError(RuntimeError):
    pass


def need(cond, msg, exc=PreflightError):
    if not cond:
        raise exc(msg)


# ---------------------------------------------------------------- bundle
def find_bundle(search_root="/kaggle/input", override=None):
    """The bundle dir (Kaggle may add a nesting level when it extracts the zip)."""
    if override:
        p = Path(override)
        need((p / "bundle_manifest.json").is_file(), f"BUNDLE_DIR={p} has no bundle_manifest.json")
        return p
    hits = []
    for m in Path(search_root).rglob("bundle_manifest.json"):
        try:
            if json.loads(m.read_text(encoding="utf-8")).get("bundle", "").startswith(BUNDLE_NAME):
                hits.append(m.parent)
        except (OSError, ValueError):
            continue
    need(len(hits) == 1, f"expected exactly one {BUNDLE_NAME}* bundle under {search_root}, found {hits}")
    return hits[0]


def verify_bundle(bundle):
    """Every manifest file present with its hash, no extra files, exact counts and order. Returns the manifest."""
    bundle = Path(bundle)
    man = json.loads((bundle / "bundle_manifest.json").read_text(encoding="utf-8"))
    need(man.get("bundle", "").startswith(BUNDLE_NAME), "not an nl2mql Spider-dev bundle")
    on_disk = {p.relative_to(bundle).as_posix() for p in bundle.rglob("*") if p.is_file()} - {"bundle_manifest.json"}
    need(on_disk == set(man["files"]), f"bundle tree != manifest: extra={sorted(on_disk - set(man['files']))[:5]} "
                                       f"missing={sorted(set(man['files']) - on_disk)[:5]}")
    for name, m in man["files"].items():
        need(sha256_file(bundle / name) == m["sha256"], f"hash mismatch: {name}")
    recs = load_dev(bundle)
    ids = [r["id"] for r in recs]
    ev = man["evaluation_set"]
    need(len(recs) == ev["records"] == man["counts"]["records"], f"{len(recs)} records, manifest says {ev['records']}")
    need(sha256_text("\n".join(ids)) == ev["ids_sha256"], "evaluation ids or their order changed")
    idx = [r["source"]["index"] for r in recs]
    need(idx == sorted(set(idx)) and all(r["id"] == f"spider_dev_{r['source']['index']:04d}" for r in recs),
         "records are not a strictly ordered subset of Spider dev")
    excluded = set(ev["excluded_db_ids"])
    need(not [r["id"] for r in recs if r["db_id"] in excluded], "a record on an excluded (held-out) database")
    dbs = sorted({r["db_id"] for r in recs})
    need(dbs == man["db_ids"] and len(dbs) == ev["databases"], "db_ids differ from the manifest")
    need(not set(dbs) & excluded, "an excluded database is in the evaluation set")
    for d in dbs:
        need((bundle / "database" / d / f"{d}.sqlite").is_file(), f"missing SQLite file for {d}")
    return man


def load_dev(bundle):
    """The evaluation records (an ordered subset of spider-sql-v1 dev_full)."""
    return read_jsonl(Path(bundle) / "data/eval_records.jsonl")


def load_mapping(bundle):
    return json.loads((Path(bundle) / "data/atlas_mapping.json").read_text(encoding="utf-8"))


# ---------------------------------------------------------------- shipped repo modules
def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_repo_modules(code_dir, context_dir, database_dir):
    """Load the bundled byte copies of spider1_kit.py / prompt_spider1.py and point them at the bundle.

    prompt_spider1 does `import spider1_kit as K`; the bundled copy is placed in sys.modules for that import
    only, so a repo copy already imported in the same process is never shadowed or modified.
    """
    code_dir, context_dir, database_dir = Path(code_dir), Path(context_dir), Path(database_dir)
    K = _load_module("dev1034_spider1_kit", code_dir / "spider1_kit.py")
    K.INTERIM = context_dir          # atlas_collections.json, prompt_samples.json
    K.SPIDER = context_dir           # tables.json (dev entries only)
    K.DB_DIR = database_dir          # <db>/<db>.sqlite
    saved = sys.modules.get("spider1_kit")
    sys.modules["spider1_kit"] = K
    try:
        P = _load_module("dev1034_prompt_spider1", code_dir / "prompt_spider1.py")
    finally:
        if saved is None:
            sys.modules.pop("spider1_kit", None)
        else:
            sys.modules["spider1_kit"] = saved
    need(P.K is K, "prompt module is not bound to the bundled spider1_kit")
    return K, P


def bundle_modules(bundle):
    b = Path(bundle)
    return load_repo_modules(b / "code", b / "context", b / "database")


# ---------------------------------------------------------------- prompts (input isolation)
class PromptInput(tuple):
    """The only object the prompt builder accepts: (id, db_id, question). No gold field can reach it."""
    __slots__ = ()
    FIELDS = ("id", "db_id", "question")

    def __new__(cls, id, db_id, question):
        for v in (id, db_id, question):
            need(isinstance(v, str) and v.strip(), "PromptInput fields must be non-empty strings", TypeError)
        return super().__new__(cls, (id, db_id, question))

    id = property(lambda s: s[0])
    db_id = property(lambda s: s[1])
    question = property(lambda s: s[2])


def prompt_input(record):
    """Explicitly pick the three allowed fields from a dataset record."""
    return PromptInput(record["id"], record["db_id"], record["question"])


def build_prompt(pi, P):
    """(system, messages) for one PromptInput, via the unmodified nl-mql-spider1-v1 template."""
    if not isinstance(pi, PromptInput):
        raise TypeError("build_prompt accepts only a PromptInput (id, db_id, question)")
    return P.SYSTEM, P.messages(pi.db_id, pi.question)


def prompt_record(pi, P):
    system, msgs = build_prompt(pi, P)
    return {"id": pi.id, "db_id": pi.db_id, "prompt_version": P.PROMPT_VERSION,
            "template_sha256": P.template_hash(), "system": system, "messages": msgs,
            "prompt_sha256": sha256_text(canon({"system": system, "messages": msgs}))}


def build_prompts(records, P):
    return [prompt_record(prompt_input(r), P) for r in records]


def _squash(s):
    return re.sub(r"\s+", " ", s or "").strip().lower()


def check_no_gold_leak(prompts, records):
    """No prompt contains its item's gold SQL (whitespace/case-insensitive) or any gold-only field name."""
    by_id = {r["id"]: r for r in records}
    for p in prompts:
        text = _squash(p["system"] + " " + " ".join(m["content"] for m in p["messages"]))
        gold = _squash(by_id[p["id"]]["sql"])
        if gold and gold in text:
            raise PreflightError(f"gold SQL of {p['id']} appears in its prompt")
        for marker in ("ordered_sha256", "multiset_sha256", '"sql":'):
            if marker in text:
                raise PreflightError(f"gold metadata marker {marker!r} in prompt {p['id']}")
    return True


def compare_prompts(frozen, rebuilt):
    need(len(frozen) == len(rebuilt) and frozen, "prompt count differs")
    bad = [a["id"] for a, b in zip(frozen, rebuilt) if canon(a) != canon(b)]
    need(not bad, f"{len(bad)} prompts differ from the frozen bundle prompts, e.g. {bad[:3]}")
    return True


# ---------------------------------------------------------------- reference (gold SQL on SQLite)
def result_hashes(rows):
    """Same canonical form as scripts/sql_v1/build_spider_sql.py:result_hashes (spider-sql-v1 metadata)."""
    canon_rows = [json.dumps([round(v, 6) if isinstance(v, float) else v for v in r], default=str) for r in rows]
    return (hashlib.sha256("\n".join(canon_rows).encode()).hexdigest(),
            hashlib.sha256("\n".join(sorted(canon_rows)).encode()).hexdigest())


def storage_cell(v):
    """JSON-safe cell whose spider1_kit._norm value equals _norm of the original (exact offline recompute)."""
    if v is None or isinstance(v, (bool, int, float, str)):
        return v
    if isinstance(v, (bytes, bytearray)):
        return bytes(v).decode("utf-8", errors="replace")
    return str(v)


def _ro_connect(path, timeout_s=SQLITE_TIMEOUT_S):
    con = sqlite3.connect("file:" + Path(path).as_posix() + "?mode=ro", uri=True)
    con.text_factory = lambda b: b.decode("utf-8", errors="replace")
    deadline = time.monotonic() + timeout_s
    con.set_progress_handler(lambda: 1 if time.monotonic() > deadline else 0, 10000)
    return con


def run_sql_ro(path, sql, timeout_s=SQLITE_TIMEOUT_S):
    con = _ro_connect(path, timeout_s)
    try:
        cur = con.execute(sql)
        return [d[0] for d in cur.description], [tuple(r) for r in cur.fetchall()]
    finally:
        con.close()


def atlas_form_copy(path):
    """In-memory copy of a SQLite DB with invalid-UTF-8 TEXT replaced as Atlas stores it
    (same procedure as scripts/spider1/build_expanded.py:atlas_form_copy). Returns (connection, n_changed)."""
    src = sqlite3.connect("file:" + Path(path).as_posix() + "?mode=ro", uri=True)
    mem = sqlite3.connect(":memory:")
    src.backup(mem)
    src.close()
    mem.text_factory = bytes
    changed = 0
    tabs = [r[0].decode() if isinstance(r[0], bytes) else r[0] for r in mem.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")]
    for t in tabs:
        for col in [r[1].decode() if isinstance(r[1], bytes) else r[1]
                    for r in mem.execute(f'PRAGMA table_info("{t}")')]:
            for rowid, v in mem.execute(
                    f'SELECT rowid, "{col}" FROM "{t}" WHERE typeof("{col}")=\'text\'').fetchall():
                try:
                    v.decode("utf-8")
                except UnicodeDecodeError:
                    mem.execute(f'UPDATE "{t}" SET "{col}"=? WHERE rowid=?',
                                (v.decode("utf-8", errors="replace"), rowid))
                    changed += 1
    mem.text_factory = lambda b: b.decode("utf-8", errors="replace")
    mem.commit()
    return mem, changed


def utf8_replacements(path):
    """Number of invalid-UTF-8 TEXT values that the Atlas form replaces in a SQLite file."""
    con, n = atlas_form_copy(path)
    con.close()
    return n


REFERENCE_STATUSES = ("ok", "tie_at_limit", "order_ties_unverified", "utf8_dependent", "reference_error")


def compute_reference(records, K, database_dir, progress=None):
    """Gold result per item, checked against the spider-sql-v1 hashes. Returns a list in dataset order.

    reference_status:
      ok                     deterministic gold; tie blocks (if any) allow any order inside a tied block
      tie_at_limit           ORDER BY .. LIMIT cuts through a tie: another tied row is an equally valid answer
      order_ties_unverified  ORDER BY whose tie blocks could not be derived (strict order is used)
      utf8_dependent         the result changes when wta_1's 2 invalid-UTF-8 strings are replaced as on Atlas
      reference_error        the gold SQL failed or did not reproduce the recorded hashes
    """
    database_dir = Path(database_dir)
    out, mem_cache = [], {}
    for i, r in enumerate(records):
        path = database_dir / r["db_id"] / f"{r['db_id']}.sqlite"
        ref = {"id": r["id"], "db_id": r["db_id"], "hardness": r["hardness"]}
        try:
            t0 = time.monotonic()
            cols, rows = run_sql_ro(path, r["sql"])
            ref["gold_ms"] = round((time.monotonic() - t0) * 1000)
        except Exception as e:  # noqa: BLE001
            ref.update(reference_status="reference_error", reason=f"{type(e).__name__}: {e}"[:300],
                       gold_columns=None, gold_rows=None, order_by=False, tie_blocks=None)
            out.append(ref)
            continue
        o, m = result_hashes(rows)
        ex = r["execution"]
        ref.update(gold_columns=cols, gold_rows=[[storage_cell(v) for v in row] for row in rows],
                   n_rows=len(rows), ordered_sha256=o, multiset_sha256=m)
        ta = K.tie_analysis(r["db_id"], r["sql"], rows)
        ref["order_by"] = bool(ta["order_by"])
        ref["tie_blocks"] = ta["blocks"]
        ref["tie_status"] = ta["status"] if not str(ta["status"]).startswith("keyed_query_failed") \
            else "keyed_query_failed"
        ref["limit"] = ta["limit"]
        status, reason = "ok", None
        if m != ex["multiset_sha256"] or (ex.get("order_sensitive") and o != ex["ordered_sha256"]):
            status, reason = "reference_error", "gold rows do not reproduce the spider-sql-v1 hashes"
        elif ta["order_by"] and ta["status"] == "ok" and ta["tie_at_limit"]:
            status, reason = "tie_at_limit", "ORDER BY .. LIMIT cuts through a tie"
        elif ta["order_by"] and ta["status"] != "ok":
            status, reason = "order_ties_unverified", f"tie analysis: {ref['tie_status']}"
        if status != "reference_error" and r["db_id"] in EXPECTED_UTF8_REPLACED_DBS:
            if r["db_id"] not in mem_cache:
                mem, changed = atlas_form_copy(path)
                need(changed == EXPECTED_UTF8_REPLACED_DBS[r["db_id"]],
                     f"{r['db_id']}: {changed} invalid-UTF-8 values replaced, expected "
                     f"{EXPECTED_UTF8_REPLACED_DBS[r['db_id']]}")
                mem_cache[r["db_id"]] = mem
            try:
                rows2 = [tuple(x) for x in mem_cache[r["db_id"]].execute(r["sql"]).fetchall()]
                if result_hashes(rows2) != (o, m):
                    status, reason = "utf8_dependent", "result changes with Atlas-form UTF-8 replacement"
            except Exception as e:  # noqa: BLE001
                status, reason = "reference_error", f"Atlas-form re-run failed: {e}"[:300]
        ref["reference_status"], ref["reason"] = status, reason
        out.append(ref)
        if progress and (i + 1) % 100 == 0:
            progress(i + 1)
    for con in mem_cache.values():
        con.close()
    return out


def load_reference(bundle):
    """The reference computed at bundle-build time (gold rows, tie blocks, status). Both models are scored against
    exactly these rows, on Kaggle and in the local verifier."""
    return read_jsonl(Path(bundle) / "data/reference.jsonl.gz")


def recheck_reference(records, refs, K, database_dir):
    """Re-execute every gold SQL in this environment and compare with the bundled reference.

    SQLite versions may legitimately differ in row order without ORDER BY, in the order inside ties and in the last
    bits of floats. Those are WARNINGS (scoring uses the bundled rows and is order/tie/rounding-tolerant). PROBLEMS
    (fail closed) are: an execution error; a changed multiset (floats rounded to 6 dp) on an item whose answer is
    deterministic; an order-sensitive item whose rows differ from the bundled gold outside its tie blocks.
    """
    database_dir = Path(database_dir)
    need([r["id"] for r in records] == [x["id"] for x in refs], "bundled reference ids differ from the dataset")
    problems, warnings = [], []
    for r, ref in zip(records, refs):
        path = database_dir / r["db_id"] / f"{r['db_id']}.sqlite"
        try:
            rows = run_sql_ro(path, r["sql"])[1]
        except Exception as e:  # noqa: BLE001
            problems.append(f"{r['id']}: gold SQL failed: {type(e).__name__}: {e}"[:300])
            continue
        o, m = result_hashes(rows)
        ex = r["execution"]
        ambiguous = ref["reference_status"] == "tie_at_limit" or (not ref.get("order_by") and ref.get("limit"))
        if m != ex["multiset_sha256"]:
            (warnings if ambiguous else problems).append(f"{r['id']}: gold multiset differs"
                                                         + (" (ambiguous gold: tie/LIMIT)" if ambiguous else ""))
            continue
        if o != ex["ordered_sha256"]:
            if ex.get("order_sensitive"):
                ok, _ = K.exec_match(rows, ref["gold_rows"], ref.get("tie_blocks"), True)
                (warnings if ok else problems).append(
                    f"{r['id']}: ordered rows differ" + (" only inside tie blocks" if ok else " outside tie blocks"))
            else:
                warnings.append(f"{r['id']}: row order differs (no ORDER BY; not scored)")
    return {"checked": len(records), "sqlite_version": sqlite3.sqlite_version, "problems": problems,
            "warnings": warnings}


EXPECTED_UTF8_REPLACED_DBS = {}
for _c, _n in EXPECTED_UTF8_REPLACED.items():
    EXPECTED_UTF8_REPLACED_DBS[_c.split("__")[0]] = EXPECTED_UTF8_REPLACED_DBS.get(_c.split("__")[0], 0) + _n


def reference_digest(ref):
    """Compact, row-free fingerprint of a reference item (bundled at build time; re-checked on Kaggle)."""
    return {k: ref.get(k) for k in ("id", "db_id", "reference_status", "n_rows", "ordered_sha256",
                                    "multiset_sha256", "order_by", "tie_status", "limit")} | \
        {"tie_blocks_sha256": sha256_text(canon(ref.get("tie_blocks")))}


# ---------------------------------------------------------------- MQL safety
# --- byte-for-byte copies (tests/test_nl2mql_dev1034.py compares their AST with the repo sources) ---
# from scripts/evaluation/eval_kit.py
WRITE_STAGES = {"$out", "$merge"}
JS_STAGES = {"$where", "$function", "$accumulator"}
SERVER_ADMIN_STAGES = {
    "$listSessions", "$listLocalSessions", "$currentOp", "$collStats", "$indexStats",
    "$planCacheStats", "$shardedDataDistribution", "$listClusterCatalog",
    "$queryStats", "$rankFusion", "$scoreFusion",
}
FORBIDDEN_STAGES = WRITE_STAGES | JS_STAGES | SERVER_ADMIN_STAGES


def iter_stages(pipeline):
    for stage in pipeline or []:
        if isinstance(stage, dict):
            for name, body in stage.items():
                yield name, body
                if name == "$facet" and isinstance(body, dict):
                    for sub in body.values():
                        yield from iter_stages(sub)


def referenced_collections(obj):
    refs = set()
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in ("from", "coll") and isinstance(v, str):
                refs.add(v)
            else:
                refs |= referenced_collections(v)
    elif isinstance(obj, list):
        for item in obj:
            refs |= referenced_collections(item)
    return refs


UNION_WITH_KEYS = {"coll", "pipeline"}


def union_with_targets(obj):
    """Targets of every `$unionWith` at any depth -> (targets: set, error: str|None).

    Short form `{"$unionWith": "<coll>"}` must name a non-blank collection; long form must be
    `{"coll": "<coll>", "pipeline"?: [...]}` (no `db` or other keys, no collection-less form).
    """
    targets = set()
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == "$unionWith":
                if isinstance(v, str):
                    if not v.strip():
                        return targets, "empty $unionWith target"
                    targets.add(v)
                elif isinstance(v, dict):
                    coll = v.get("coll")
                    if not isinstance(coll, str) or not coll.strip():
                        return targets, "$unionWith without a collection name"
                    if set(v) - UNION_WITH_KEYS:
                        return targets, f"unsupported $unionWith keys {sorted(set(v) - UNION_WITH_KEYS)}"
                    if "pipeline" in v and not isinstance(v["pipeline"], list):
                        return targets, "$unionWith 'pipeline' must be a list"
                    targets.add(coll)
                else:
                    return targets, f"malformed $unionWith ({type(v).__name__})"
            sub, err = union_with_targets(v)
            targets |= sub
            if err:
                return targets, err
    elif isinstance(obj, list):
        for item in obj:
            sub, err = union_with_targets(item)
            targets |= sub
            if err:
                return targets, err
    return targets, None


def validate_mql(pred, allowed):
    """Structural validation of a model output against the read-only contract.

    Malformed `$unionWith` is always rejected. When `allowed` is non-empty, the target
    collection and every `from`/`coll`/`$unionWith` reference must be in it (an empty
    `allowed` disables the reference checks, as before).

    Returns (ok: bool, reason: str, parsed: dict|None).
    """
    if not isinstance(pred, dict):
        return False, "output is not a JSON object", None
    method = pred.get("method", "aggregate")
    if method not in ("aggregate", "find"):
        return False, f"unsupported method '{method}'", None
    coll = pred.get("collection")
    if not isinstance(coll, str) or not coll:
        return False, "missing 'collection'", None
    if allowed and coll not in allowed:
        return False, f"collection '{coll}' not among the example's allowed collections", None
    if method == "find":
        filt = pred.get("filter", {})
        if not isinstance(filt, dict):
            return False, "'filter' must be an object", None
        pipeline = [{"$match": filt}]
        if "projection" in pred and pred["projection"]:
            pipeline.append({"$project": pred["projection"]})
    else:
        pipeline = pred.get("pipeline")
        if not isinstance(pipeline, list) or not pipeline:
            return False, "missing non-empty 'pipeline' list", None
    for name, _ in iter_stages(pipeline):
        if name in FORBIDDEN_STAGES:
            return False, f"forbidden stage {name}", None
    union_targets, union_err = union_with_targets(pipeline)
    if union_err:
        return False, union_err, None
    if allowed:
        bad = (referenced_collections(pipeline) | union_targets) - allowed
        if bad:
            return False, f"references collections outside the example: {sorted(bad)}", None
    return True, "ok", {"method": method, "collection": coll, "pipeline": pipeline}


# from scripts/spider1/run_eval.py
def forbidden_anywhere(obj):
    """Forbidden operators at ANY depth ($function/$where inside expressions, $out inside
    $lookup/$unionWith sub-pipelines); validate_mql only walks top-level and $facet stages."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in FORBIDDEN_STAGES:
                return k
            hit = forbidden_anywhere(v)
            if hit:
                return hit
    elif isinstance(obj, list):
        for v in obj:
            hit = forbidden_anywhere(v)
            if hit:
                return hit
    return None


# from scripts/evaluation/run_nl_mql_eval.py
def parse_output(text):
    if text is None:
        return None, "empty output"
    t = text.strip()
    if t.startswith("```"):
        t = t.strip("`")
        if t.startswith("json"):
            t = t[4:]
        t = t.strip()
    try:
        return json.loads(t), None
    except Exception:
        pass
    start = t.find("{")
    if start == -1:
        return None, "no JSON object found"
    depth = 0
    for i in range(start, len(t)):
        if t[i] == "{":
            depth += 1
        elif t[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(t[start:i + 1]), None
                except Exception as e:
                    return None, f"JSON parse error: {e}"
    return None, "unbalanced JSON braces"
# --- end of copies ---


# additional rules for this experiment (stricter than MQL-0001..0005; documented)
EXTRA_FORBIDDEN = {"$changeStream", "$changeStreamSplitLargeEvent", "$listSearchIndexes", "$search",
                   "$searchMeta", "$vectorSearch", "$listSampledQueries", "$currentOp", "$documents"}
LOOKUP_LIKE = ("$lookup", "$graphLookup", "$unionWith")


def _depth(obj, d=0):
    if isinstance(obj, dict):
        return max([_depth(v, d + 1) for v in obj.values()] or [d + 1])
    if isinstance(obj, list):
        return max([_depth(v, d + 1) for v in obj] or [d + 1])
    return d


def extra_safety(pipeline):
    """None if safe, else the reason. Never selects another database; bounded size and depth."""
    try:
        size = len(json.dumps(pipeline))
    except (TypeError, ValueError) as e:
        return f"pipeline not JSON-serialisable: {e}"
    if size > MAX_PIPELINE_CHARS:
        return f"pipeline too large ({size} chars > {MAX_PIPELINE_CHARS})"
    if _depth(pipeline) > MAX_PIPELINE_DEPTH:
        return f"pipeline nested deeper than {MAX_PIPELINE_DEPTH}"

    def walk(o):
        if isinstance(o, dict):
            for k, v in o.items():
                if k in EXTRA_FORBIDDEN:
                    return f"forbidden stage {k}"
                if k in LOOKUP_LIKE and isinstance(v, dict):
                    if "db" in v:
                        return f"{k} with a 'db' key (cross-database access)"
                    if "from" in v and not isinstance(v["from"], str):
                        return f"{k} 'from' must be a collection name string"
                hit = walk(v)
                if hit:
                    return hit
        elif isinstance(o, list):
            for v in o:
                hit = walk(v)
                if hit:
                    return hit
        return None
    return walk(pipeline)


def check_mql(text, allowed):
    """Raw model text -> (stage reached, normalised MQL or None, detail category, reason).

    stage: 'format' (no JSON object), 'structure' (validate_mql structural failure), 'safety' (forbidden or
    out-of-scope reference), or 'ok'. The normalisation of a bare pipeline object / empty pipeline is
    exactly run_eval.score_output's.
    """
    parsed, perr = parse_output(text)
    if parsed is None:
        return "format", None, "format_error", perr
    if isinstance(parsed, dict) and "method" not in parsed:
        parsed = {"method": "aggregate", **parsed}
    if isinstance(parsed, dict) and parsed.get("method") == "aggregate" and parsed.get("pipeline") == []:
        parsed = {**parsed, "pipeline": [{"$match": {}}]}
    ok, why, norm = validate_mql(parsed, allowed)
    if not ok:
        unsafe = why.startswith(("forbidden stage", "references collections outside", "collection '")) or \
            "unionWith" in why
        return ("safety" if unsafe else "structure"), None, ("unsafe_mql" if unsafe else "invalid_mql"), why
    hit = forbidden_anywhere(norm["pipeline"])
    if hit:
        return "safety", None, "unsafe_mql", f"forbidden operator {hit} (nested)"
    extra = extra_safety(norm["pipeline"])
    if extra:
        return "safety", None, "unsafe_mql", extra
    return "ok", {"collection": norm["collection"], "pipeline": norm["pipeline"]}, None, None


# ---------------------------------------------------------------- execution + comparison
# OperationFailure codes that are not the model's fault
INFRA_CODES = {6, 7, 13, 18, 89, 91, 189, 9001, 10107, 11600, 11602, 13435, 13436}
TIMEOUT_CODES = {50, 262}


def classify_exception(exc):
    """(outcome, failure_source, detail) for an exception raised while executing a validated pipeline."""
    name = type(exc).__name__
    code = getattr(exc, "code", None)
    if code in TIMEOUT_CODES or name in ("ExecutionTimeout",):
        return "timeout", "model", "timeout"
    if name in ("AutoReconnect", "NetworkTimeout", "ServerSelectionTimeoutError", "ConnectionFailure",
                "NotPrimaryError", "OperationFailure") and (code in INFRA_CODES or name != "OperationFailure"):
        return "failed", "infrastructure", "atlas_unavailable"
    if code == 8000:  # AtlasError: e.g. operator not allowed on this tier
        return "failed", "model", "atlas_tier_restriction"
    return "failed", "model", "execution_error"


def run_pipeline(mdb, collection, pipeline, timeout_ms=EXEC_TIMEOUT_MS, max_rows=MAX_RESULT_ROWS):
    """Execute with a server time limit and a bounded cursor. Returns (docs, truncated)."""
    cur = mdb[collection].aggregate(pipeline, maxTimeMS=timeout_ms, allowDiskUse=True, batchSize=1000)
    docs = []
    try:
        for d in cur:
            docs.append(d)
            if len(docs) > max_rows:
                return docs, True
    finally:
        cur.close()
    return docs, False


def bson_prenormalize(docs):
    """SECONDARY comparison only: Decimal128 -> float, datetime -> 'YYYY-MM-DD HH:MM:SS[.ffffff]',
    nested ObjectId -> str. The top-level ObjectId `_id` is kept so docs_to_rows still drops it.
    Returns (docs, Counter of conversions)."""
    notes = Counter()

    def conv(v, top_id=False):
        tn = type(v).__name__
        if tn == "Decimal128":
            notes["decimal128"] += 1
            return float(v.to_decimal())
        if isinstance(v, datetime):
            notes["datetime"] += 1
            return v.strftime("%Y-%m-%d %H:%M:%S") + (f".{v.microsecond:06d}" if v.microsecond else "")
        if tn == "ObjectId" and not top_id:
            notes["objectid"] += 1
            return str(v)
        if isinstance(v, dict):
            return {k: conv(x) for k, x in v.items()}
        if isinstance(v, list):
            return [conv(x) for x in v]
        return v
    out = [{k: conv(x, top_id=(k == "_id")) for k, x in d.items()} if isinstance(d, dict) else conv(d)
           for d in docs]
    return out, notes


def compare(K, rows, ref):
    """PRIMARY verdict: spider1_kit.exec_match with the reference's order/tie data. -> (match, reason)."""
    return K.exec_match(rows, ref["gold_rows"], ref.get("tie_blocks"), ref.get("order_by", False))


def mismatch_detail(K, rows, ref):
    ok, why = K.exec_match(rows, ref["gold_rows"])  # order-insensitive re-check
    if ref.get("order_by") and ok:
        return "order_mismatch"
    if why.startswith("row count"):
        return "wrong_row_count"
    if why.startswith("column count"):
        return "wrong_column_count"
    return "wrong_values"


def decide(K, rows, ref):
    """Final outcome for an executed prediction, given the reference status (identical for every model)."""
    status = ref["reference_status"]
    if status in ("reference_error", "utf8_dependent"):
        return "not_comparable", "reference_" + status, f"reference status {status}"
    match, why = compare(K, rows, ref)
    if match:
        return "correct", "correct", why
    detail = mismatch_detail(K, rows, ref)
    if status == "tie_at_limit":
        return "not_comparable", "tie_at_limit_mismatch", \
            "mismatch on an item whose gold LIMIT cuts a tie: another tied row may be valid"
    if status == "order_ties_unverified" and detail == "order_mismatch":
        return "not_comparable", "order_ties_unverified", "same rows, different order; gold ties unverifiable"
    return "incorrect", detail, why


def evaluate_prediction(mdb, K, pred, ref, allowed, timeout_ms=EXEC_TIMEOUT_MS, max_rows=MAX_RESULT_ROWS):
    """One prediction record -> (evaluation record, stored prediction rows or None).

    Outcome rules (each record has exactly one):
      failed / infrastructure   generation error (provider/runtime) or Atlas unreachable; retried on resume
      invalid                   no JSON object / structurally invalid / unsafe (rejected before execution)
      timeout                   Atlas maxTimeMS exceeded (model-attributable)
      failed / model            Atlas rejected the validated pipeline (unknown operator, type error, tier limit)
      incorrect                 executed and compared, mismatch (including results over MAX_RESULT_ROWS)
      not_comparable            reference cannot judge this output (see decide())
      correct                   executed and matched the gold result
    """
    rec = {"id": pred["id"], "db_id": ref["db_id"], "hardness": ref["hardness"],
           "reference_status": ref["reference_status"], "format_valid": False, "structure_valid": False,
           "readonly_valid": False, "executed": False, "outcome": None, "failure_source": None,
           "detail": None, "reason": None, "mql": None, "n_rows": None, "exec_ms": None,
           "secondary_outcome": None, "bson_normalizations": None}
    if pred.get("status") != "ok":
        rec.update(outcome="failed", failure_source="infrastructure", detail="generation_error",
                   reason=(pred.get("error") or "no generation")[:300])
        return rec, None
    stage, mql, detail, why = check_mql(pred.get("raw_text"), allowed)
    rec["format_valid"] = stage != "format"
    rec["structure_valid"] = stage in ("safety", "ok")
    rec["readonly_valid"] = stage == "ok"
    if stage != "ok":
        rec.update(outcome="invalid", failure_source="model", detail=detail, reason=(why or "")[:300])
        return rec, None
    rec["mql"] = mql
    t0 = time.monotonic()
    try:
        docs, truncated = run_pipeline(mdb, mql["collection"], mql["pipeline"], timeout_ms, max_rows)
    except Exception as e:  # noqa: BLE001
        rec["exec_ms"] = round((time.monotonic() - t0) * 1000)
        outcome, src, det = classify_exception(e)
        rec.update(outcome=outcome, failure_source=src, detail=det, reason=f"{type(e).__name__}: {str(e)[:300]}")
        return rec, None
    rec["exec_ms"] = round((time.monotonic() - t0) * 1000)
    rec["executed"] = True
    if truncated:
        rec.update(n_rows=f">{max_rows}", outcome="incorrect", failure_source="model", detail="result_too_large",
                   reason=f"more than {max_rows} result rows (gold has {ref.get('n_rows')})")
        return rec, None
    rec["n_rows"] = len(docs)
    cols, rows = K.docs_to_rows(docs)
    stored = [[storage_cell(v) for v in r] for r in rows]
    outcome, det, why = decide(K, rows, ref)
    rec.update(outcome=outcome, detail=det, reason=why, failure_source=None if outcome == "correct" else "model")
    if outcome == "not_comparable":
        rec["failure_source"] = "reference"
    ndocs, notes = bson_prenormalize(docs)
    rec["bson_normalizations"] = dict(notes) or None
    rec["secondary_outcome"] = decide(K, K.docs_to_rows(ndocs)[1], ref)[0] if notes else outcome
    return rec, {"id": pred["id"], "columns": cols, "rows": stored}


def recompute_outcome(K, ev, rows_rec, ref):
    """Offline re-derivation of an executed record's outcome from stored rows (verifier)."""
    if not ev["executed"] or ev.get("detail") == "result_too_large":
        return ev["outcome"]
    return decide(K, [tuple(r) for r in rows_rec["rows"]], ref)[0]


# ---------------------------------------------------------------- resumable checkpoints
class Checkpoint:
    """Append-only JSONL keyed by id, fsynced per append. A record is final unless `retry(rec)` says so.
    Refuses records written under another fingerprint; tolerates a torn last line after a crash."""

    def __init__(self, path, expected_ids, fingerprint, retry=lambda r: False):
        self.path, self.expected, self.fp, self.retry = Path(path), list(expected_ids), fingerprint, retry
        self.records, self.warnings = {}, []
        exp = set(self.expected)
        if self.path.exists():
            lines = self.path.read_text(encoding="utf-8").splitlines()
            for n, line in enumerate(lines):
                if not line.strip():
                    continue
                try:
                    r = json.loads(line)
                except ValueError:
                    if n == len(lines) - 1:
                        self.warnings.append(f"ignored torn last line {n + 1}")
                        continue
                    raise RuntimeError(f"{self.path}: corrupt line {n + 1}")
                if r.get("fingerprint") != fingerprint:
                    raise RuntimeError(f"{self.path}: record {r.get('id')} has a different run fingerprint; "
                                       "refusing to mix runs (start a fresh OUT_DIR)")
                if r.get("id") not in exp:
                    raise RuntimeError(f"{self.path}: unexpected id {r.get('id')}")
                self.records[r["id"]] = r
            if self.warnings:  # rewrite without the torn line
                write_jsonl(self.path, list(self.records.values()))

    def done(self, iid):
        r = self.records.get(iid)
        return r is not None and not self.retry(r)

    def todo(self):
        return [i for i in self.expected if not self.done(i)]

    def append(self, recs):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8", newline="\n") as f:
            for r in recs:
                if r.get("fingerprint") != self.fp:
                    raise RuntimeError("record fingerprint differs from the checkpoint's")
                if self.done(r["id"]):
                    raise RuntimeError(f"duplicate final record for {r['id']}")
                f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
                self.records[r["id"]] = r
            f.flush()
            os.fsync(f.fileno())

    def ordered(self):
        return [self.records[i] for i in self.expected if i in self.records]


def fingerprint(obj):
    return sha256_text(canon(obj))[:16]


def seed_from_previous(prev_dir, out_dir, rel_paths):
    """Copy checkpoint files from an attached earlier version's output (never overwrites). Returns copied."""
    copied = []
    for rel in rel_paths:
        src, dst = Path(prev_dir) / rel, Path(out_dir) / rel
        if src.is_file() and not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(src.read_bytes())
            copied.append(rel)
    return copied


# ---------------------------------------------------------------- generation loops
def is_generation_retry(rec):
    return rec.get("status") != "ok"


def pred_record(item_id, db_id, fp, model_label, text, meta):
    return {"id": item_id, "db_id": db_id, "fingerprint": fp, "model": model_label, "status": "ok",
            "raw_text": text, "error": None, "created_at": now(), **meta}


def error_record(item_id, db_id, fp, model_label, err, meta):
    return {"id": item_id, "db_id": db_id, "fingerprint": fp, "model": model_label, "status": "error",
            "raw_text": None, "error": str(err)[:500], "created_at": now(), **meta}


def run_generation(prompts, generate_batch, ckpt, model_label, batch_size=1, stop=None, progress=None):
    """Generate for every prompt not yet final in `ckpt`.

    generate_batch(list_of_prompt_records) -> list of (text | None, meta dict, error | None), one per prompt.
    stop(meta) -> str|None: a non-empty reason stops the loop cleanly (e.g. provider daily quota, token cap).
    Returns {"stopped": reason|None, "generated": n}.
    """
    todo = set(ckpt.todo())
    queue = [p for p in prompts if p["id"] in todo]
    n = 0
    for i in range(0, len(queue), batch_size):
        batch = queue[i:i + batch_size]
        try:
            outs = generate_batch(batch)
        except QuotaStop as e:  # nothing from this batch is recorded; it is regenerated on resume
            return {"stopped": str(e)[:300], "generated": n}
        if len(outs) != len(batch):
            raise RuntimeError("generate_batch returned a different number of outputs")
        recs, stop_reason = [], None
        for p, (text, meta, err) in zip(batch, outs):
            meta = dict(meta or {})
            meta["prompt_sha256"] = p["prompt_sha256"]
            if err is None:
                recs.append(pred_record(p["id"], p["db_id"], ckpt.fp, model_label, text, meta))
            else:
                recs.append(error_record(p["id"], p["db_id"], ckpt.fp, model_label, err, meta))
            if stop and not stop_reason:
                stop_reason = stop(meta)
        ckpt.append(recs)
        n += len(recs)
        if progress:
            progress(n, len(queue))
        if stop_reason:
            return {"stopped": stop_reason, "generated": n}
    return {"stopped": None, "generated": n}


def run_evaluation(preds, refs, mapping, mdb, K, ckpt, rows_ckpt, timeout_ms=EXEC_TIMEOUT_MS,
                   max_rows=MAX_RESULT_ROWS, progress=None):
    """Execute + compare every prediction not yet finally evaluated (infrastructure failures are retried)."""
    ref_by = {r["id"]: r for r in refs}
    allowed_by_db = allowed_collections(mapping)
    todo = set(ckpt.todo())
    n = 0
    for p in preds:
        if p["id"] not in todo:
            continue
        ref = ref_by[p["id"]]
        rec, rows = evaluate_prediction(mdb, K, p, ref, allowed_by_db[ref["db_id"]], timeout_ms, max_rows)
        rec["fingerprint"] = ckpt.fp
        rec["prediction_sha256"] = sha256_text(canon({k: p.get(k) for k in ("id", "raw_text", "status")}))
        rec["evaluated_at"] = now()
        if rows is not None:
            rows["fingerprint"] = rows_ckpt.fp
            if rows_ckpt.done(p["id"]):
                rows_ckpt.records.pop(p["id"])  # re-evaluation after an infrastructure retry
            rows_ckpt.append([rows])
        ckpt.append([rec])
        n += 1
        if progress and n % 50 == 0:
            progress(n, len(todo))
    return n


def is_eval_retry(rec):
    return rec.get("failure_source") == "infrastructure" and rec.get("detail") == "atlas_unavailable"


def allowed_collections(mapping):
    out = {}
    for coll, m in mapping["collections"].items():
        out.setdefault(m["db_id"], set()).add(coll)
    return out


# ---------------------------------------------------------------- Groq (OpenAI-compatible) client
class QuotaStop(Exception):
    pass


def parse_retry_in(msg):
    """Seconds from 'try again in 1h2m3.5s' / '4m35.6s' / '432ms' / '7.2s'; None if absent
    (same as scripts/evaluation/providers.py:parse_retry_in)."""
    m = re.search(r"try again in ((?:\d+h)?(?:\d+m(?!s))?(?:[\d.]+s|[\d.]+ms)?)", msg)
    if not m or not m.group(1):
        return None
    total = 0.0
    for num, unit in re.findall(r"([\d.]+)(h|ms|m|s)", m.group(1)):
        total += float(num) * {"h": 3600, "m": 60, "s": 1, "ms": 0.001}[unit]
    return total


class GroqClient:
    """Chat completions on an OpenAI-compatible endpoint (default Groq). The key is passed in (from Kaggle
    Secrets) and is never logged. Retries 429/5xx for transport only (never for a wrong answer); a per-day
    quota error stops the run cleanly instead of sleeping for hours. `opener` is injectable for tests."""

    def __init__(self, api_key, model, base_url="https://api.groq.com/openai/v1", temperature=0.0, top_p=1.0,
                 seed=42, max_tokens=8192, reasoning_effort=None, retries=6, max_wait_s=120, timeout_s=300,
                 opener=None, sleep=time.sleep):
        need(api_key, "frontier API key is empty", ValueError)
        self._key, self.model, self.base_url = api_key, model, base_url.rstrip("/")
        self.params = {"temperature": temperature, "top_p": top_p, "seed": seed, "max_tokens": max_tokens}
        if reasoning_effort:
            self.params["reasoning_effort"] = reasoning_effort
        self.retries, self.max_wait_s, self.timeout_s, self.sleep = retries, max_wait_s, timeout_s, sleep
        import urllib.request
        self._ur = urllib.request
        self._open = opener or (lambda req, timeout: urllib.request.urlopen(req, timeout=timeout))

    def _req(self, path, body=None):
        headers = {"Authorization": f"Bearer {self._key}", "Content-Type": "application/json",
                   "User-Agent": "codegen-capstone-mql0006/1.0"}
        data = json.dumps(body).encode("utf-8") if body is not None else None
        return self._ur.Request(self.base_url + path, data=data, headers=headers)

    def redact(self, s):
        return (s or "").replace(self._key, "<API_KEY>")

    def list_models(self):
        with self._open(self._req("/models"), self.timeout_s) as r:
            return [m.get("id") for m in json.loads(r.read().decode("utf-8")).get("data", [])]

    def chat(self, system, messages):
        """-> (text|None, meta, error|None). Raises QuotaStop on a daily-quota error."""
        import urllib.error
        body = {"model": self.model, "messages": [{"role": "system", "content": system}] + messages, **self.params}
        t0, attempt, last = time.monotonic(), 0, None
        while attempt <= self.retries:
            attempt += 1
            t_req = time.monotonic()
            try:
                with self._open(self._req("/chat/completions", body), self.timeout_s) as r:
                    data = json.loads(r.read().decode("utf-8"))
                    hdr = {k.lower(): v for k, v in (r.headers.items() if r.headers else [])}
                usage = data.get("usage") or {}
                ch = data["choices"][0]
                details = usage.get("completion_tokens_details") or {}
                meta = {"latency_s": round(time.monotonic() - t_req, 3), "wall_s": round(time.monotonic() - t0, 3),
                        "input_tokens": usage.get("prompt_tokens"), "output_tokens": usage.get("completion_tokens"),
                        "reasoning_tokens": details.get("reasoning_tokens"),
                        "cached_input_tokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
                        "server_total_time_s": usage.get("total_time"), "finish_reason": ch.get("finish_reason"),
                        "served_model": data.get("model"), "system_fingerprint": data.get("system_fingerprint"),
                        "request_id": data.get("id") or hdr.get("x-request-id"), "retries": attempt - 1,
                        "rate_limit": {k: v for k, v in hdr.items() if k.startswith("x-ratelimit")} or None}
                return ch["message"].get("content") or "", meta, None
            except urllib.error.HTTPError as e:
                raw = self.redact(e.read().decode("utf-8", "replace"))
                last = f"HTTP {e.code}: {raw[:300]}"
                if e.code == 429 and re.search(r"per day|\(RPD\)|\(TPD\)|daily", raw, re.I):
                    raise QuotaStop(f"DAILY_QUOTA_EXHAUSTED: {raw[:300]}")
                if e.code not in (408, 429, 500, 502, 503, 504):
                    break
                wait = None
                try:
                    wait = float(e.headers.get("retry-after"))
                except (TypeError, ValueError, AttributeError):
                    wait = parse_retry_in(raw)
                self.sleep(min(self.max_wait_s, (wait if wait is not None else 5 * 2 ** attempt) + 1))
            except Exception as e:  # noqa: BLE001
                last = self.redact(f"{type(e).__name__}: {str(e)[:300]}")
                self.sleep(min(self.max_wait_s, 5 * 2 ** attempt))
        return None, {"latency_s": round(time.monotonic() - t0, 3), "retries": attempt - 1}, last


# ---------------------------------------------------------------- Atlas mapping parity + preflight
def type_tag(v):
    """Storage-class tag used by the parity digest (TEXT '3' != INTEGER 3 != REAL 3.0; NULL != missing)."""
    if v is None:
        return ["null"]
    if isinstance(v, bool):
        return ["bool", v]
    if isinstance(v, int):
        return ["int", v]
    if isinstance(v, float):
        return ["real", v]
    if isinstance(v, (bytes, bytearray)):
        return ["blob", bytes(v).hex()]
    if isinstance(v, str):
        return ["text", v]
    return ["other:" + type(v).__name__, str(v)]


_MISSING = ["missing"]


def rows_digest(field_rows):
    """sha256 of the sorted multiset of type-tagged rows."""
    keys = sorted(json.dumps(r, ensure_ascii=False) for r in field_rows)
    return hashlib.sha256("\n".join(keys).encode("utf-8")).hexdigest()


def sqlite_table_digest(path, table, columns):
    con = _ro_connect(path, 600)
    try:
        q = ", ".join(f'"{c}"' for c in columns)
        rows = con.execute(f'SELECT {q} FROM "{table}"').fetchall()
    finally:
        con.close()
    replaced = sum(isinstance(v, str) and "�" in v for r in rows for v in r)
    return len(rows), rows_digest([[type_tag(v) for v in r] for r in rows]), replaced


def atlas_collection_digest(mdb, coll, columns, batch_size=5000):
    """Stream every document (projection without _id). -> (n_docs, digest, extra_field_counts)."""
    keys, extra, n = [], Counter(), 0
    cur = mdb[coll].find({}, {"_id": 0}, batch_size=batch_size)
    try:
        for d in cur:
            n += 1
            for k in d:
                if k not in columns:
                    extra[k] += 1
            keys.append(json.dumps([type_tag(d[c]) if c in d else _MISSING for c in columns], ensure_ascii=False))
    finally:
        cur.close()
    keys.sort()
    return n, hashlib.sha256("\n".join(keys).encode("utf-8")).hexdigest(), dict(extra)


WRITE_OR_ADMIN_ACTIONS = {
    "insert", "update", "remove", "createCollection", "dropCollection", "dropDatabase", "createIndex",
    "dropIndex", "renameCollectionSameDB", "collMod", "convertToCapped", "compact", "reIndex",
    "bypassDocumentValidation", "createSearchIndexes", "dropSearchIndex", "updateSearchIndex", "applyOps",
    "createUser", "dropUser", "updateUser", "grantRole", "revokeRole", "createRole", "dropRole", "updateRole",
    "grantRolesToUser", "revokeRolesFromUser", "shutdown", "enableSharding", "moveChunk", "splitChunk",
    "cleanupOrphaned", "killop", "killAnySession", "dropConnections", "setParameter", "fsync",
}
_WRITE_PREFIXES = ("create", "drop", "insert", "update", "remove", "grant", "revoke", "rename", "apply",
                   "compact", "shutdown", "kill", "set", "enable", "move", "split", "convert", "bypass")
READ_ACTIONS_OK = {"killCursors"}  # part of MongoDB's built-in `read` role: closes the user's own cursors


def privileges_report(conn_status):
    """From connectionStatus(showPrivileges=True): (users, actions per resource, offending actions)."""
    info = conn_status.get("authInfo", {})
    users = [u.get("user") for u in info.get("authenticatedUsers", [])]
    per, bad = {}, set()
    for p in info.get("authenticatedUserPrivileges", []):
        res = p.get("resource", {})
        key = "cluster" if res.get("cluster") else f"{res.get('db', '')}.{res.get('collection', '')}"
        acts = set(p.get("actions", []))
        per.setdefault(key, set()).update(acts)
        for a in acts:
            if a in READ_ACTIONS_OK:
                continue
            if a in WRITE_OR_ADMIN_ACTIONS or a.startswith(_WRITE_PREFIXES):
                bad.add(f"{key}:{a}")
    return users, {k: sorted(v) for k, v in per.items()}, sorted(bad)


def atlas_preflight(client, mapping, full_parity=True, log=print):
    """Fail-closed readiness check of the Atlas data the evaluation will read. Reads only.

    1. ping + server version; 2. read-only privileges (no write/admin action on any resource, `find` on the
    target database); 3. all 80 expected collections present and no unexpected collection with a dev prefix;
    4. exact document counts; 5. (full_parity) type-tagged multiset digest of every collection equals the
    digest of the bundled SQLite table, and only the known invalid-UTF-8 strings differ in form.
    Raises PreflightError; returns the report dict on success.
    """
    rep = {"atlas_db": mapping["atlas_db"], "checked_at": now(), "full_parity": full_parity}
    client.admin.command("ping")
    bi = client.admin.command("buildInfo")
    rep["server_version"] = bi.get("version")
    mdb = client[mapping["atlas_db"]]
    cs = mdb.command({"connectionStatus": 1, "showPrivileges": True})
    users, per, bad = privileges_report(cs)
    rep.update(authenticated_users=users, privileges=per, offending_actions=bad)
    need(users, "Atlas connection is not authenticated")
    need(not bad, f"Atlas user is NOT read-only (write/admin actions granted: {bad[:10]}). Use the read-only "
                  "evaluation account (codegen_eval_readonly).")
    db_actions = set().union(*[set(v) for k, v in per.items()
                               if k == "." or k == f"{mapping['atlas_db']}."]) if per else set()
    need("find" in db_actions, f"read-only user has no 'find' on database {mapping['atlas_db']}")
    names = set(mdb.list_collection_names())
    expected = set(mapping["collections"])
    prefixes = tuple(f"{d}__" for d in mapping["db_ids"])
    missing = sorted(expected - names)
    unexpected = sorted(n for n in names - expected if n.startswith(prefixes))
    rep.update(collections_found=len(names & expected), missing_collections=missing,
               unexpected_dev_collections=unexpected)
    need(not missing, f"{len(missing)} expected collections are missing on Atlas, e.g. {missing[:5]} "
                      "(see the seed workflow in docs/kaggle_nl2mql_spider_dev_1034.md)")
    need(not unexpected, f"unexpected collections with a dev-db prefix on Atlas: {unexpected[:5]}")
    counts, parity = {}, {}
    for coll in sorted(expected):
        m = mapping["collections"][coll]
        counts[coll] = mdb[coll].count_documents({})
        need(counts[coll] == m["documents"], f"{coll}: {counts[coll]} documents on Atlas, expected {m['documents']}")
        if full_parity:
            t0 = time.monotonic()
            n, dig, extra = atlas_collection_digest(mdb, coll, m["fields"])
            parity[coll] = {"documents": n, "digest_match": dig == m["row_multiset_sha256"], "extra_fields": extra,
                            "seconds": round(time.monotonic() - t0, 1)}
            need(not extra, f"{coll}: fields not in the mapping: {sorted(extra)[:5]}")
            need(dig == m["row_multiset_sha256"], f"{coll}: Atlas documents differ from the bundled SQLite table "
                                                  "(type-tagged multiset digest mismatch)")
            if log:
                log(f"  parity OK {coll} ({n} docs, {parity[coll]['seconds']} s)")
    rep["counts"], rep["parity"] = counts, parity
    try:
        st = mdb.command("dbStats")
        rep["db_stats"] = {k: st.get(k) for k in ("collections", "objects", "dataSize", "storageSize", "indexSize")}
    except Exception as e:  # noqa: BLE001  (dbStats may be denied to a narrow role)
        rep["db_stats"] = f"not available: {type(e).__name__}"
    rep["ok"] = True
    return rep


def preflight_sqlite(bundle, records, man):
    """Every referenced db_id has its bundled SQLite file with the manifest hash, and opens read-only."""
    for d in sorted({r["db_id"] for r in records}):
        rel = f"database/{d}/{d}.sqlite"
        need(rel in man["files"], f"{rel} not in the bundle manifest")
        con = _ro_connect(Path(bundle) / rel, 10)
        try:
            con.execute("SELECT count(*) FROM sqlite_master").fetchone()
            try:
                con.execute("CREATE TABLE __w (x)")
                raise PreflightError(f"{rel} is writable; expected read-only")
            except sqlite3.OperationalError:
                pass
        finally:
            con.close()
    return True


# ---------------------------------------------------------------- statistics
def wilson(k, n, z=1.959963984540054):
    if n == 0:
        return None
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [round(max(0.0, c - h), 4), round(min(1.0, c + h), 4)]


def mcnemar_exact(b, c):
    """Two-sided exact McNemar p-value (binomial test on the discordant pairs)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def paired_bootstrap_ci(a, b, reps=BOOTSTRAP_REPS, seed=BOOTSTRAP_SEED):
    """95% percentile CI of mean(a) - mean(b) over items resampled with replacement (paired)."""
    d = [x - y for x, y in zip(a, b)]
    n = len(d)
    rng = random.Random(seed)
    stats = sorted(sum(rng.choices(d, k=n)) / n for _ in range(reps))
    return [round(stats[int(0.025 * reps)], 4), round(stats[int(0.975 * reps) - 1], 4)]


def pct(k, n):
    return round(100 * k / n, 2) if n else None


def frac(k, n):
    return {"count": k, "total": n, "percent": pct(k, n)}


def latency_stats(xs):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None

    def q(p):
        return round(xs[min(len(xs) - 1, max(0, math.ceil(p * len(xs)) - 1))], 3)
    return {"n": len(xs), "mean": round(sum(xs) / len(xs), 3), "median": q(0.5), "p90": q(0.9), "p95": q(0.95),
            "p99": q(0.99), "max": round(xs[-1], 3), "total": round(sum(xs), 1)}


# ---------------------------------------------------------------- summaries
def model_summary(label, run_config, preds, evals, refs, expected_ids, pricing=None):
    """baseline_summary.json content for one model. Status is 'complete' only with every expected id evaluated,
    no duplicates and no unresolved infrastructure failure."""
    exp = list(expected_ids)
    pid, eid = [p["id"] for p in preds], [e["id"] for e in evals]
    problems = []
    for name, ids in (("predictions", pid), ("evaluation", eid)):
        if len(ids) != len(set(ids)):
            problems.append(f"duplicate ids in {name}")
        if set(ids) - set(exp):
            problems.append(f"unexpected ids in {name}")
        if len(set(ids)) != len(exp):
            problems.append(f"{name}: {len(set(ids))}/{len(exp)} ids")
    N = len(exp)
    out = Counter(e["outcome"] for e in evals)
    infra = sum(e.get("failure_source") == "infrastructure" for e in evals)
    if infra:
        problems.append(f"{infra} unresolved infrastructure failures")
    correct = out.get("correct", 0)
    compared = out.get("correct", 0) + out.get("incorrect", 0)
    by_h = {}
    for h in HARDNESS:
        hs = [e for e in evals if e["hardness"] == h]
        nh = sum(r["hardness"] == h for r in refs)
        c = sum(e["outcome"] == "correct" for e in hs)
        by_h[h] = {"total": nh, "correct": c, "percent": pct(c, nh), "wilson95": wilson(c, nh)}
    lat = [p.get("latency_s") for p in preds if p.get("status") == "ok"]
    itok = sum(p.get("input_tokens") or 0 for p in preds)
    otok = sum(p.get("output_tokens") or 0 for p in preds)
    cost = None
    if pricing and pricing.get("input_per_1m") is not None and pricing.get("output_per_1m") is not None:
        cost = round(itok / 1e6 * pricing["input_per_1m"] + otok / 1e6 * pricing["output_per_1m"], 4)
    secondary = sum((e.get("secondary_outcome") or e["outcome"]) == "correct" for e in evals)
    return {
        "experiment": EXPERIMENT_ID, "model_label": label, "model_id": run_config.get("model_id"),
        "model_revision": run_config.get("model_revision"), "status": "complete" if not problems else "partial",
        "problems": problems, "n_expected": N,
        "primary_metric": f"cross-engine execution accuracy: correct / {N} (Atlas MQL result vs SQLite gold-SQL result)",
        "correct": correct, "accuracy": frac(correct, N), "accuracy_wilson95": wilson(correct, N),
        "outcomes": {o: out.get(o, 0) for o in OUTCOMES},
        "detail_categories": dict(Counter(e["detail"] for e in evals)),
        "failure_source": dict(Counter(e.get("failure_source") or "none" for e in evals)),
        "format_validity": frac(sum(e["format_valid"] for e in evals), N),
        "structural_validity": frac(sum(e["structure_valid"] for e in evals), N),
        "readonly_validation": frac(sum(e["readonly_valid"] for e in evals), N),
        "atlas_execution_success": frac(sum(e["executed"] for e in evals), N),
        "comparison_coverage": frac(compared, N),
        "accuracy_on_compared": frac(correct, compared),
        "not_comparable": out.get("not_comparable", 0),
        "secondary_bson_normalized_correct": frac(secondary, N),
        "by_hardness": by_h,
        "latency_s": latency_stats(lat),
        "atlas_exec_ms": latency_stats([e.get("exec_ms") for e in evals]),
        "generation_wall_s": run_config.get("generation_wall_s"),
        "evaluation_wall_s": run_config.get("evaluation_wall_s"),
        "tokens": {"input": itok, "output": otok,
                   "reasoning": sum(p.get("reasoning_tokens") or 0 for p in preds) or None},
        "cost_usd": cost, "pricing_used": pricing,
        "generation_errors": sum(p.get("status") != "ok" for p in preds),
    }


def sensitivity_subsets(refs, prior_audit):
    """Item-id subsets reported next to the primary full evaluation set (never replacing it)."""
    all_ids = [r["id"] for r in refs]
    flagged = {r["id"] for r in refs if r["reference_status"] != "ok"}
    excluded_audit = {i for i, a in prior_audit.items() if a.get("status") == "excluded"}
    return {
        "all_items": all_ids,
        "reference_status_ok": [i for i in all_ids if i not in flagged],
        "without_prior_audit_exclusions": [i for i in all_ids if i not in excluded_audit],
    }


def comparison_summary(summaries, evals_by_label, refs, prior_audit):
    """Paired comparison of two models on identical ids. Defined only when both are complete."""
    labels = list(summaries)
    need(len(labels) == 2, "comparison needs exactly two models", ValueError)
    a, b = labels
    status = "complete" if all(summaries[x]["status"] == "complete" for x in labels) else "partial"
    res = {"experiment": EXPERIMENT_ID, "status": status, "models": labels,
           "per_model": {x: {k: summaries[x][k] for k in ("model_id", "correct", "accuracy", "accuracy_wilson95",
                                                       "outcomes", "status")} for x in labels}}
    if status != "complete":
        res["note"] = "at least one model run is partial: no paired statistics are reported"
        return res
    ea = {e["id"]: e["outcome"] == "correct" for e in evals_by_label[a]}
    eb = {e["id"]: e["outcome"] == "correct" for e in evals_by_label[b]}
    subsets = {}
    for name, ids in sensitivity_subsets(refs, prior_audit).items():
        va, vb = [int(ea[i]) for i in ids], [int(eb[i]) for i in ids]
        both = sum(x and y for x, y in zip(va, vb))
        only_a = sum(x and not y for x, y in zip(va, vb))
        only_b = sum(y and not x for x, y in zip(va, vb))
        n = len(ids)
        subsets[name] = {
            "n": n, f"{a}_correct": sum(va), f"{b}_correct": sum(vb),
            "difference_pp": round(100 * (sum(vb) - sum(va)) / n, 2) if n else None,
            "difference_definition": f"{b} minus {a}, percentage points",
            "both_correct": both, f"only_{a}": only_a, f"only_{b}": only_b, "neither": n - both - only_a - only_b,
            "mcnemar_exact_p": round(mcnemar_exact(only_a, only_b), 6),
            "paired_bootstrap95_pp": [round(100 * x, 2) for x in paired_bootstrap_ci(vb, va)],
        }
    res["paired"] = subsets
    res["primary"] = "all_items"
    res["statistics"] = {"test": "exact McNemar (two-sided binomial on discordant pairs), alpha 0.05, single "
                                 "pre-specified primary contrast on all_items; subsets are sensitivity analyses",
                         "ci": f"paired bootstrap percentile, {BOOTSTRAP_REPS} resamples, seed {BOOTSTRAP_SEED}",
                         "per_model_ci": "Wilson score 95%"}
    p = subsets["all_items"]["mcnemar_exact_p"]
    res["significant_at_0.05"] = p < 0.05
    return res


# ---------------------------------------------------------------- report
LIMITATION = ("The metric measures cross-engine execution equivalence between generated MQL on MongoDB Atlas and "
              "Spider gold SQL on SQLite. It is not native MQL exact-match accuracy. Results depend on the fidelity "
              "of the database mapping and the chosen normalization policy. Matching on one database instance does "
              "not prove general semantic correctness, and the gold SQL labels have not been independently "
              "human-verified for every question.")


def _f(x):
    return "pending" if x is None else (f"{x['count']}/{x['total']} ({x['percent']}%)" if isinstance(x, dict) else str(x))


def render_report(run_status, summaries, comparison):
    n = run_status.get("n_expected")
    L = [f"# {EXPERIMENT_ID}: Qwen2.5-Coder-3B vs Groq-hosted open-weight gpt-oss-120b, NL→MQL on Spider dev "
         f"({n} items; held-out test databases excluded)", "",
         f"Generated {now()} by the Kaggle notebook. Run status: **{run_status['status']}**.", "",
         "> " + LIMITATION, ""]
    if run_status["status"] != "complete":
        L += ["**PARTIAL RUN: these numbers are not final results.** Problems:", ""]
        L += [f"- {p}" for p in run_status.get("problems", [])] + [""]
    L += [f"## Results (denominator {n} for every row)", "", "| Metric | " + " | ".join(summaries) + " |",
          "|---|" + "---|" * len(summaries)]
    rows = [("Status", lambda s: s["status"]),
            ("Cross-engine correct", lambda s: _f(s["accuracy"])),
            ("Wilson 95% CI", lambda s: f"{100 * s['accuracy_wilson95'][0]:.1f}–{100 * s['accuracy_wilson95'][1]:.1f}%"),
            ("Format validity (JSON object)", lambda s: _f(s["format_validity"])),
            ("Structural validity", lambda s: _f(s["structural_validity"])),
            ("Read-only validation passed", lambda s: _f(s["readonly_validation"])),
            ("Atlas execution success", lambda s: _f(s["atlas_execution_success"])),
            ("Comparison coverage (correct+incorrect)", lambda s: _f(s["comparison_coverage"]))]
    rows += [(f"Outcome: {o}", (lambda o: lambda s: str(s["outcomes"][o]))(o)) for o in OUTCOMES]
    rows += [(f"{h} ({s0['by_hardness'][h]['total']})", (lambda h: lambda s: f"{s['by_hardness'][h]['correct']} "
              f"({s['by_hardness'][h]['percent']}%)")(h)) for s0 in list(summaries.values())[:1] for h in HARDNESS]
    rows += [("Latency mean / median / p95 (s)", lambda s: "/".join(str(s["latency_s"][k]) for k in ("mean", "median", "p95"))
              if s["latency_s"] else "n/a"),
             ("Generation wall time (s)", lambda s: str(s["generation_wall_s"])),
             ("Tokens in / out", lambda s: f"{s['tokens']['input']} / {s['tokens']['output']}"),
             ("Cost (USD)", lambda s: "n/a" if s["cost_usd"] is None else str(s["cost_usd"])),
             ("Secondary (BSON-normalised) correct", lambda s: _f(s["secondary_bson_normalized_correct"]))]
    for name, fn in rows:
        L.append(f"| {name} | " + " | ".join(fn(s) for s in summaries.values()) + " |")
    L += [""]
    if comparison and comparison.get("paired"):
        L += ["## Paired comparison", "", "| Subset | n | " + " | ".join(comparison["models"]) +
              " | Difference (pp) | Bootstrap 95% CI | Exact McNemar p |", "|---|---|---|---|---|---|---|"]
        a, b = comparison["models"]
        for name, s in comparison["paired"].items():
            L.append(f"| {name} | {s['n']} | {s[f'{a}_correct']} | {s[f'{b}_correct']} | {s['difference_pp']} | "
                     f"{s['paired_bootstrap95_pp']} | {s['mcnemar_exact_p']} |")
        L += ["", f"Difference = {b} minus {a}. Primary contrast: all_items. "
                  f"Significant at 0.05: {comparison['significant_at_0.05']}.", ""]
    else:
        L += ["## Paired comparison", "", "Not computed: " + (comparison or {}).get("note", "comparison missing"), ""]
    return "\n".join(L) + "\n"


def output_sums(out_dir, names):
    out_dir = Path(out_dir)
    return {n: sha256_file(out_dir / n) for n in sorted(names) if (out_dir / n).is_file()}
