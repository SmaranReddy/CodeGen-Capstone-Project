"""sql2mql-v2 Phase-2 pilot runner, VERIFIER-3 version (new file; run_v2.py pilot is verifier-2 and is NOT used).

    python -B scripts/finetune_v2/run_pilot_v3.py check                  integrity gate only (no execution)
    python -B scripts/finetune_v2/run_pilot_v3.py freeze  --run run01    write the runner freeze (before execution)
    python -B scripts/finetune_v2/run_pilot_v3.py run     --run run01 --i-understand-this-is-the-pilot [--budget S]
    python -B scripts/finetune_v2/run_pilot_v3.py finalize --run run01   per-example results, classifications, summary
    python -B scripts/finetune_v2/run_pilot_v3.py review  --run run01    P2 human-review package (frozen template)
    python -B scripts/finetune_v2/run_pilot_v3.py repro   --run run01 --i-understand-this-is-the-pilot [--budget S]
                                                                          independent second execution + comparison

Fail-closed integrity gate (runs before ANY frozen module is imported):
  * data/interim/finetune_v2/round3_code_freeze.json: its own sha256 is pinned here; it must list exactly the 19
    frozen files and each file's bytes must match; its verifier must be sql2mql-v2-verifier-3;
  * the stage-1 round-3 pre-registration body hash, the pilot freeze (freeze_sha256 recomputed), the
    spider-sql-v1 train file and review_package.py are pinned here;
  * after import, every frozen module must resolve inside scripts/finetune_v2 and verifier3.VERIFIER_VERSION must
    be sql2mql-v2-verifier-3. There is no verifier-2 path: the runner never imports verifier.py or run_v2.py
    itself (verifier3 imports verifier-2's checks by design and runs them first), and every stored record must
    carry verifier-3's version or the run stops.

Sample: exactly the 300 frozen pilot ids (pilot_freeze.json), rows taken byte-for-byte (parsed JSON, unchanged)
from data/processed/spider_sql_v1/train.jsonl, the frozen protocol's source. Nothing is resampled, filtered or
replaced. Membership in data/processed/spider_pool_v2 is recorded per id (train / val_internal / quarantined /
excluded / not_in_pool) and only affects CORPUS candidacy, never pilot processing or pilot reporting.

Outcome per example (exactly one): accepted | conversion_rejected (converter Unsupported) | conversion_error
(converter crash) | safety_rejected (runner read-only scan; the MQL is never executed) | timeout (MongoDB
maxTimeMS or SQLite progress-handler interrupt inside verifier-3) | verification_failed (verifier-3 rejection;
sub-class semantic_mismatch / structure_or_safety / untestable_precondition / other) | verifier_error (an
unexpected exception in verifier-3; never accepted). Loss of the MongoDB connection aborts the run WITHOUT
writing a record, so a resume re-runs that example (infrastructure faults never become pilot data).

Three-or-more-table rule (frozen before execution): an example is multi_table_3plus if its SQL is in the
J2 stratum (>= 2 top-level JOINs), OR its SQL names >= 3 distinct tables anywhere, OR its MQL has >= 2 $lookup
stages anywhere. Such examples stay in every pilot count and are reported separately; they are only
quarantined from CORPUS candidacy.

Corpus candidacy is NOT training eligibility: an example is at most `candidate_pending_P2_review` (accepted,
in spider_pool_v2 train.jsonl, not multi_table_3plus). No corpus file is written here.

Resume: one record is appended (and flushed) per finished example. A trailing line cut by a kill is dropped
and logged; a duplicate id, an id outside the 300, a record from another runner version or a non-verifier-3
record stops the run (fail closed). Local MongoDB 127.0.0.1:27018 only (v2_common.local_client).
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import sys
import time
from collections import Counter
from pathlib import Path

RUNNER_VERSION = "sql2mql-v2-pilot-runner-v3.1"
EXPECTED_VERIFIER = "sql2mql-v2-verifier-3"
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

# ---- pinned inputs (all computed before this runner existed; none may change)
CODE_FREEZE_REL = "data/interim/finetune_v2/round3_code_freeze.json"
CODE_FREEZE_SHA = "db3204433ced307a20d986a63ff5cfe279968d4fcbe8bf092afcb753fca92aca"
PREREG_REL = "data/interim/finetune_v2/round3_prereg.json"
PREREG_BODY_SHA = "944e9063c64e9a6d315c01793f05b86ce61daee6b3a180ef3edd23249738d3e6"
PILOT_FREEZE_REL = "data/interim/finetune_v2/pilot_freeze.json"
PILOT_FREEZE_SHA = "fb1affb9d146488453462996430ba1fb6d629a28a66602525838ead5b5aa5b6a"
TRAIN_REL = "data/processed/spider_sql_v1/train.jsonl"
TRAIN_SHA = "40c1a7c1591e741d484072fc11a32c46edfbf0f19e326241b88c6ed75c2f3e42"
REVIEW_PACKAGE_SHA = "616903b80144f3cef97737bf1b6fe3406eae12dd8bd192aec0c24cd3365b23c2"
POOL_REL = "data/processed/spider_pool_v2"
POOL_MANIFEST_SHA = "cb80a118315daa0ac958837a4af8306f033de918ee80eecd43a7342992b38053"
PILOT_BY_HARDNESS = {"easy": 72, "medium": 118, "hard": 64, "extra": 46}
FROZEN_FILES = (
    "sql2mql_v2.py", "schema_profile.py", "verifier3.py", "fixtures3.py", "verifier.py", "v2_fixtures.py",
    "comparator.py", "mongo_import.py", "index_real.py", "v2_common.py", "oracle_v2.py", "mutants_v2.py",
    "run_v3.py", "mutant_study_v3.py", "round3_stats.py", "round3_report.py", "freeze_round3.py",
    "freeze_round3_code.py", "scripts/evaluation/eval_kit.py",
)
FROZEN_MODULES = ("v2_common", "schema_profile", "sql2mql_v2", "verifier3", "verifier", "fixtures3", "v2_fixtures",
                  "comparator", "mongo_import", "index_real", "round3_stats", "run_v3", "eval_kit")
MONGO_DIGEST = "sha256:b9924c330a8bbc5800fe9c69f04a84e221eca7b28fd049a0354147d9908fa142"
OUT_BASE_REL = "outputs/sql2mql_v2/pilot_v3"
DOCKER_FALLBACK = ("C:/Program Files/Docker/Docker/resources/bin/docker.exe",)
RUN_FREEZE_DIR_REL = "data/interim/finetune_v2"

# ---- outcome classification (frozen with the runner)
SEMANTIC = {"l1_mismatch_explicit_null", "l1_mismatch_missing", "fixture_mismatch", "v3_reorder_mismatch",
            "nondeterministic"}
STRUCTURE = {"unsafe_mql", "structure", "v3_lookup_keys"}
PRECONDITION = {"empty_referenced_table", "tie_cut", "bad_text", "too_few_fixtures", "non_discriminating",
                "limit_without_order", "order_plan", "sql_parse", "sqlite_error", "mongo_error"}
WRITE_OR_JS_OPERATORS = {"$out", "$merge", "$function", "$accumulator", "$where"}
OUTCOMES = ("accepted", "conversion_rejected", "conversion_error", "safety_rejected", "timeout",
            "verification_failed", "verifier_error")
MEMBERSHIPS = ("train", "val_internal", "quarantined", "excluded", "not_in_pool")


class IntegrityError(SystemExit):
    def __init__(self, msg):
        super().__init__(f"INTEGRITY FAILURE (fail closed, nothing executed): {msg}")
        self.msg = msg


class InfrastructureAbort(SystemExit):
    pass


def sha_b(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha_f(p) -> str:
    return sha_b(Path(p).read_bytes())


def canon(obj):
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str)


# ===================================================================== integrity gate (no frozen import)
def check_frozen_code(finetune_dir: Path = HERE, root: Path = ROOT, freeze_path: Path | None = None,
                      freeze_sha: str | None = CODE_FREEZE_SHA):
    """-> {file: {"expected", "actual", "match"}} for all 19 files; raises IntegrityError on any problem."""
    fp = Path(freeze_path) if freeze_path else root / CODE_FREEZE_REL
    if not fp.exists():
        raise IntegrityError(f"frozen-code manifest missing: {fp}")
    raw = fp.read_bytes()
    if freeze_sha is not None and sha_b(raw) != freeze_sha:
        raise IntegrityError(f"frozen-code manifest itself changed: {fp}")
    try:
        fz = json.loads(raw)
        code = fz["code_sha256"]
    except Exception as e:  # noqa: BLE001
        raise IntegrityError(f"frozen-code manifest unreadable: {e}") from e
    if fz.get("verifier") != EXPECTED_VERIFIER:
        raise IntegrityError(f"frozen manifest is for {fz.get('verifier')!r}, not {EXPECTED_VERIFIER}")
    if not isinstance(code, dict) or len(code) != 19 or set(code) != set(FROZEN_FILES):
        raise IntegrityError(f"frozen manifest incomplete or altered: {len(code) if isinstance(code, dict) else code!r} "
                             f"entries; missing {sorted(set(FROZEN_FILES) - set(code or {}))}, "
                             f"unexpected {sorted(set(code or {}) - set(FROZEN_FILES))}")
    res, bad = {}, []
    for name in FROZEN_FILES:
        p = (root / name) if "/" in name else (finetune_dir / name)
        if not p.exists():
            res[name] = {"expected": code[name], "actual": None, "match": False}
            bad.append(f"{name} (missing)")
            continue
        got = sha_f(p)
        res[name] = {"expected": code[name], "actual": got, "match": got == code[name]}
        if got != code[name]:
            bad.append(name)
    if bad:
        raise IntegrityError(f"frozen code differs from the 19-hash manifest: {bad}")
    return res


def check_inputs(root: Path = ROOT):
    """Pinned non-code inputs. -> dict of recorded hashes."""
    pre = json.loads((root / PREREG_REL).read_text(encoding="utf-8"))
    body = {k: v for k, v in pre.items() if k != "sha256_of_body_without_this_field"}
    if sha_b(json.dumps(body, indent=1, ensure_ascii=False).encode("utf-8")) != PREREG_BODY_SHA or \
            pre.get("sha256_of_body_without_this_field") != PREREG_BODY_SHA:
        raise IntegrityError("round-3 stage-1 pre-registration changed")
    fr = json.loads((root / PILOT_FREEZE_REL).read_text(encoding="utf-8"))
    if sha_b(canon({k: v for k, v in fr.items() if k != "freeze_sha256"}).encode("utf-8")) != PILOT_FREEZE_SHA \
            or fr.get("freeze_sha256") != PILOT_FREEZE_SHA:
        raise IntegrityError("pilot freeze changed (freeze_sha256 recomputed)")
    if sha_f(root / TRAIN_REL) != TRAIN_SHA:
        raise IntegrityError("spider-sql-v1 train.jsonl changed")
    if sha_f(HERE / "review_package.py") != REVIEW_PACKAGE_SHA:
        raise IntegrityError("review_package.py changed (P2 selection rule / template)")
    pman = root / POOL_REL / "manifest.json"
    if sha_f(pman) != POOL_MANIFEST_SHA:
        raise IntegrityError("spider_pool_v2 manifest changed")
    man = json.loads(pman.read_text(encoding="utf-8"))
    for n, meta in man["files"].items():
        if sha_f(root / POOL_REL / n) != meta["sha256"]:
            raise IntegrityError(f"spider_pool_v2/{n} differs from its manifest")
    return {"round3_prereg_body_sha256": PREREG_BODY_SHA, "pilot_freeze_sha256": PILOT_FREEZE_SHA,
            "pilot_freeze_file_sha256": sha_f(root / PILOT_FREEZE_REL),
            "spider_sql_v1_train_sha256": TRAIN_SHA, "review_package_sha256": REVIEW_PACKAGE_SHA,
            "spider_pool_v2_manifest_sha256": sha_f(pman),
            "spider_pool_v2_files": {n: m["sha256"] for n, m in man["files"].items()}}


def load_frozen_modules():
    """Import the frozen modules ONLY after check_frozen_code passed. -> namespace dict."""
    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))
    ek_dir = str(ROOT / "scripts/evaluation")
    if ek_dir not in sys.path:
        sys.path.insert(1, ek_dir)
    try:
        mods = {m: importlib.import_module(m) for m in FROZEN_MODULES}
    except Exception as e:  # noqa: BLE001  (includes a missing verifier3: never a fallback)
        raise IntegrityError(f"frozen module import failed: {type(e).__name__}: {e}") from e
    for m, mod in mods.items():
        f = Path(mod.__file__).resolve()
        want = ROOT / "scripts/evaluation" if m == "eval_kit" else HERE
        if f.parent != want:
            raise IntegrityError(f"module {m} resolved to {f}, not inside {want}")
    if getattr(mods["verifier3"], "VERIFIER_VERSION", None) != EXPECTED_VERIFIER:
        raise IntegrityError(f"verifier3.VERIFIER_VERSION is not {EXPECTED_VERIFIER}")
    if mods["run_v3"].V3 is not mods["verifier3"]:
        raise IntegrityError("run_v3 is not bound to verifier3")
    return mods


def gate():
    """Full integrity gate: 19 hashes -> pinned inputs -> import -> post-import checks."""
    hashes = check_frozen_code()
    inputs = check_inputs()
    mods = load_frozen_modules()
    return hashes, inputs, mods


# ===================================================================== sample + membership
def load_pilot(root: Path = ROOT):
    fr = json.loads((root / PILOT_FREEZE_REL).read_text(encoding="utf-8"))
    ids = fr["pilot"]["ids"]
    if len(ids) != 300 or len(set(ids)) != 300 or fr["pilot"]["n"] != 300:
        raise IntegrityError("pilot is not 300 unique ids")
    rows = {}
    for line in (root / TRAIN_REL).read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            if r["id"] in set(ids):
                rows[r["id"]] = r
    if set(rows) != set(ids):
        raise IntegrityError(f"pilot ids missing from source: {sorted(set(ids) - set(rows))[:5]}")
    hard = Counter(rows[i]["hardness"] for i in ids)
    if dict(hard) != PILOT_BY_HARDNESS or dict(hard) != fr["pilot"]["by_hardness"]:
        raise IntegrityError(f"pilot hardness mix differs: {dict(hard)}")
    pre = json.loads((root / PREREG_REL).read_text(encoding="utf-8"))
    r3 = {i for v in pre["sampling"]["ids"].values() for i in v}
    if len(r3) != 740 or set(ids) & r3:
        raise IntegrityError("pilot overlaps the 740 round-3 records (or round-3 list is not 740)")
    if set(ids) & set(fr["dev"]["ids"]):
        raise IntegrityError("pilot overlaps the 500 DEV ids")
    return ids, rows


def load_membership(root: Path = ROOT):
    d = root / POOL_REL
    m = {}

    def ids_of(name):
        return [json.loads(x)["id"] for x in (d / name).read_text(encoding="utf-8").splitlines() if x.strip()]
    pool_rows = {}
    for name, cat in (("train.jsonl", "train"), ("val_internal.jsonl", "val_internal"), ("quarantine.jsonl", "quarantined")):
        for x in (d / name).read_text(encoding="utf-8").splitlines():
            if x.strip():
                r = json.loads(x)
                if r["id"] in m:
                    raise IntegrityError(f"{r['id']} in two spider_pool_v2 files")
                m[r["id"]] = cat
                pool_rows[r["id"]] = r
    excl = {}
    for x in (d / "exclusions.jsonl").read_text(encoding="utf-8").splitlines():
        if x.strip():
            e = json.loads(x)
            excl[e["id"]] = e
            if e["disposition"] == "excluded":
                if e["id"] in m:
                    raise IntegrityError(f"{e['id']} excluded but also present in a record file")
                m[e["id"]] = "excluded"
    return m, pool_rows, excl


def membership_of(i, m):
    return m.get(i, "not_in_pool")


# ===================================================================== classification helpers
def sql_profile(sql):
    from sqlglot import exp, parse_one
    try:
        t = parse_one(sql, read="sqlite")
    except Exception:  # noqa: BLE001
        return {"parse_ok": False, "distinct_tables": None, "joins_anywhere": None, "pattern": ["unparsed"]}
    tables = {x.name.lower() for x in t.find_all(exp.Table)}
    joins = len(list(t.find_all(exp.Join)))
    pat = []
    if isinstance(t, (exp.Union, exp.Intersect, exp.Except)):
        pat.append("set_operation")
    if joins:
        pat.append("join")
    if any(s.find_ancestor(exp.Select) is not None for s in t.find_all(exp.Select)):
        pat.append("subquery")
    if list(t.find_all(exp.Group)):
        pat.append("group_by")
    if list(t.find_all(exp.Having)):
        pat.append("having")
    if list(t.find_all(exp.Order)):
        pat.append("order_by")
    if list(t.find_all(exp.Limit)):
        pat.append("limit")
    if list(t.find_all(exp.AggFunc)):
        pat.append("aggregate")
    if list(t.find_all(exp.Distinct)):
        pat.append("distinct")
    if list(t.find_all(exp.Like)):
        pat.append("like")
    return {"parse_ok": True, "distinct_tables": len(tables), "joins_anywhere": joins, "pattern": pat or ["simple_select"]}


def count_lookups(obj):
    if isinstance(obj, dict):
        return sum((1 if k == "$lookup" else 0) + count_lookups(v) for k, v in obj.items())
    if isinstance(obj, list):
        return sum(count_lookups(v) for v in obj)
    return 0


def multi_table_flags(stratum, prof, mql):
    flags = {
        "j2_top_level_stratum": stratum == "J2",
        "sql_distinct_tables_ge3": bool(prof["distinct_tables"] and prof["distinct_tables"] >= 3),
        "mql_lookups_ge2": count_lookups(mql) >= 2 if mql is not None else False,
    }
    flags["multi_table_3plus"] = any(flags.values())
    return flags


def runner_safety(mql, allowed, ek):
    """Read-only safety: eval_kit.validate_mql (unchanged) + a recursive scan for write / server-JS operators."""
    ok, why, _ = ek.validate_mql(mql, allowed)
    found = sorted(_ops(mql) & WRITE_OR_JS_OPERATORS)
    return {"eval_kit_validate_mql": why, "eval_kit_ok": bool(ok), "write_or_js_operators": found,
            "ok": bool(ok) and not found}


def _ops(obj):
    out = set()
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(k, str) and k.startswith("$"):
                out.add(k)
            out |= _ops(v)
    elif isinstance(obj, list):
        for v in obj:
            out |= _ops(v)
    return out


def classify_verification(v):
    st = v.get("status")
    if st == "accepted":
        return "accepted", None
    if st == "mongo_timeout" or (st == "sqlite_error" and "interrupted" in str(v.get("detail", "")).lower()):
        return "timeout", "mongo_timeout" if st == "mongo_timeout" else "sqlite_interrupted"
    if st in SEMANTIC:
        return "verification_failed", "semantic_mismatch"
    if st in STRUCTURE:
        return "verification_failed", "structure_or_safety"
    if st in PRECONDITION:
        return "verification_failed", "untestable_precondition"
    return "verification_failed", "other"


def corpus_status(outcome, membership, flags):
    if outcome != "accepted":
        return "not_candidate", [f"outcome:{outcome}"]
    reasons = []
    if membership != "train":
        reasons.append(f"membership:{membership}")
    if flags.get("multi_table_3plus"):
        reasons.append("multi_table_3plus_quarantine")
    if not reasons:
        return "candidate_pending_P2_review", []
    if reasons == ["multi_table_3plus_quarantine"]:
        return "quarantined_multi_table_3plus", reasons
    return "not_candidate", reasons


# ===================================================================== per-example processing
def allowed_collections(db_id, C):
    con = C.sqlite_ro(C.sqlite_path(db_id))
    try:
        return {C.coll(db_id, n) for (n,) in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    finally:
        con.close()


def process_one(r, mods, real, profs, membership):
    """-> record dict. Raises InfrastructureAbort on lost MongoDB connectivity (no record written)."""
    from pymongo.errors import ConnectionFailure
    C, R3, V3, EK, S = mods["v2_common"], mods["run_v3"], mods["verifier3"], mods["eval_kit"], mods["round3_stats"]
    t0 = time.time()
    rec = R3.convert(r, profs)        # frozen: records id, db_id, hardness, question, sql, stratum, converter
    rec["runner"] = RUNNER_VERSION
    rec["membership"] = membership
    prof = sql_profile(r["sql"])
    rec["sql_profile"] = prof
    if rec["conversion"] == "unsupported":
        outcome, sub = "conversion_rejected", rec["code"]
    elif rec["conversion"] == "crash":
        outcome, sub = "conversion_error", rec["code"]
    else:
        rec["safety"] = runner_safety(rec["mql"], allowed_collections(r["db_id"], C), EK)
        if not rec["safety"]["ok"]:
            outcome, sub = "safety_rejected", rec["safety"]["eval_kit_validate_mql"]
        else:
            try:
                v = V3.verify({"id": r["id"], "db_id": r["db_id"], "sql": r["sql"]}, rec["mql"], real)
            except ConnectionFailure as e:
                raise InfrastructureAbort(f"MongoDB connection lost on {r['id']}: {type(e).__name__}: "
                                          f"{str(e)[:200]} (no record written; resume re-runs it)") from e
            except Exception as e:  # noqa: BLE001
                v = {"id": r["id"], "db_id": r["db_id"], "verifier": V3.VERIFIER_VERSION, "status": "verifier_error",
                     "detail": f"{type(e).__name__}: {str(e)[:300]}", "checks": {}}
                rec["verification"] = v
                outcome, sub = "verifier_error", type(e).__name__
            else:
                if v.get("verifier") != EXPECTED_VERIFIER:
                    raise IntegrityError(f"verification record for {r['id']} is not {EXPECTED_VERIFIER}")
                rec["verification"] = v
                outcome, sub = classify_verification(v)
    rec["outcome"] = outcome
    rec["outcome_detail"] = sub
    rec["multi_table"] = multi_table_flags(rec["stratum"], prof, rec.get("mql"))
    cs, why = corpus_status(outcome, membership, rec["multi_table"])
    rec["corpus_status"], rec["corpus_reasons"] = cs, why
    rec["runner_seconds"] = round(time.time() - t0, 2)
    return rec


# ===================================================================== resume-safe record store
def load_records(path: Path, pilot_ids, log_path: Path | None = None):
    """Fail-closed load. Drops (and logs) only a cut trailing line."""
    if not path.exists():
        return []
    lines = [x for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    rows, cut = [], 0
    for k, x in enumerate(lines):
        try:
            rows.append(json.loads(x))
        except json.JSONDecodeError:
            if k != len(lines) - 1:
                raise IntegrityError(f"{path}: unparseable line {k + 1} that is not the last line")
            cut = 1
    seen = set()
    for x in rows:
        if x.get("id") not in pilot_ids:
            raise IntegrityError(f"{path}: record {x.get('id')} is not a pilot id")
        if x["id"] in seen:
            raise IntegrityError(f"{path}: duplicate record {x['id']}")
        seen.add(x["id"])
        if x.get("runner") != RUNNER_VERSION:
            raise IntegrityError(f"{path}: record {x['id']} from runner {x.get('runner')}")
        if "verification" in x and x["verification"].get("verifier") != EXPECTED_VERIFIER:
            raise IntegrityError(f"{path}: record {x['id']} verified by {x['verification'].get('verifier')}")
        if x.get("outcome") not in OUTCOMES:
            raise IntegrityError(f"{path}: record {x['id']} has outcome {x.get('outcome')}")
    if cut:
        path.write_text("".join(json.dumps(x, ensure_ascii=False, default=str) + "\n" for x in rows), encoding="utf-8")
    if log_path is not None and (rows or cut):
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"at": time.strftime("%Y-%m-%dT%H:%M:%S"), "records_kept": len(rows),
                                "cut_trailing_line_dropped": cut}) + "\n")
    return rows


def append_record(path: Path, rec):
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
        f.flush()      # survives a killed process (os is not imported in this package: Atlas guard)


def execute(out: Path, ids, rows, membership, mods, process=process_one, budget=None, real=None):
    """Process every id without a record; resumable. -> (records, finished: bool)."""
    out.mkdir(parents=True, exist_ok=True)
    path = out / "records.jsonl"
    prev = load_records(path, set(ids), out / "resume_log.jsonl")
    done = {x["id"] for x in prev}
    todo = sorted((rows[i] for i in ids if i not in done), key=lambda r: (r["db_id"], r["id"]))
    profs = {}
    t0 = time.time()
    for k, r in enumerate(todo, 1):
        if budget is not None and time.time() - t0 > budget:
            print(f"budget reached after {k - 1}/{len(todo)}; re-run to resume", flush=True)
            break
        rec = process(r, mods, real, profs, membership_of(r["id"], membership))
        append_record(path, rec)
        if k % 20 == 0 or k == len(todo):
            print(f"{k}/{len(todo)} ({len(done)} earlier)  {time.time() - t0:.0f}s", flush=True)
    recs = load_records(path, set(ids))
    return recs, {x["id"] for x in recs} == set(ids)


# ===================================================================== reporting
def pct(k, n):
    return {"k": k, "n": n, "pct": round(100 * k / n, 2) if n else None}


def block(rs):
    n = len(rs)
    oc = Counter(r["outcome"] for r in rs)
    return {"examples": n, "converted": pct(sum(r["conversion"] == "converted" for r in rs), n),
            "accepted": pct(oc["accepted"], n),
            "outcomes": {o: oc.get(o, 0) for o in OUTCOMES}}


def summarize(recs, ids, wilson):
    assert {r["id"] for r in recs} == set(ids) and len(recs) == len(ids) == 300
    recs = sorted(recs, key=lambda r: r["id"])      # order-independent (resumes change record order)
    acc =[r for r in recs if r["outcome"] == "accepted"]
    conv = [r for r in recs if r["conversion"] == "converted"]
    lo, hi = wilson(len(acc), len(recs))
    s = {
        "examples": len(recs),
        "all": block(recs),
        "acceptance_rate_wilson95": [lo, hi],
        "accepted_of_converted": pct(len(acc), len(conv)),
        "conversion_rejection_codes": dict(Counter(r["code"] for r in recs if r["outcome"] == "conversion_rejected").most_common()),
        "conversion_error_codes": dict(Counter(r["code"] for r in recs if r["outcome"] == "conversion_error").most_common()),
        "verification_failed": {
            "by_subclass": dict(Counter(r["outcome_detail"] for r in recs if r["outcome"] == "verification_failed").most_common()),
            "by_status": dict(Counter(r["verification"]["status"] for r in recs if r["outcome"] == "verification_failed").most_common())},
        "timeouts": dict(Counter(r["outcome_detail"] for r in recs if r["outcome"] == "timeout")),
        "safety_rejected": sum(r["outcome"] == "safety_rejected" for r in recs),
        "verifier_error": sum(r["outcome"] == "verifier_error" for r in recs),
        "by_membership": {m: block([r for r in recs if r["membership"] == m]) for m in MEMBERSHIPS},
        "by_hardness": {h: block([r for r in recs if r["hardness"] == h]) for h in ("easy", "medium", "hard", "extra")},
        "by_stratum": {h: block([r for r in recs if r["stratum"] == h]) for h in ("J2", "OL", "J1", "G")},
        "by_sql_pattern": {p: block([r for r in recs if p in r["sql_profile"]["pattern"]])
                           for p in sorted({p for r in recs for p in r["sql_profile"]["pattern"]})},
        "by_converter_feature_accepted": dict(sorted(Counter(f for r in acc for f in r.get("features", [])).items())),
        "multi_table_3plus": {
            "rule": "J2 top-level stratum OR >=3 distinct SQL tables anywhere OR >=2 $lookup stages in the MQL",
            "flagged": block([r for r in recs if r["multi_table"]["multi_table_3plus"]]),
            "not_flagged": block([r for r in recs if not r["multi_table"]["multi_table_3plus"]]),
            "by_indicator": {k: sum(r["multi_table"][k] for r in recs) for k in
                             ("j2_top_level_stratum", "sql_distinct_tables_ge3", "mql_lookups_ge2")},
            "indicator_disagreements": sorted(
                [{"id": r["id"], **{k: r["multi_table"][k] for k in ("j2_top_level_stratum", "sql_distinct_tables_ge3",
                                                                    "mql_lookups_ge2")}, "outcome": r["outcome"]}
                 for r in recs if r["multi_table"]["multi_table_3plus"] and
                 len({r["multi_table"][k] for k in ("j2_top_level_stratum", "sql_distinct_tables_ge3", "mql_lookups_ge2")
                      if not (k == "mql_lookups_ge2" and r.get("mql") is None)}) > 1], key=lambda x: x["id"]),
            "accepted_and_flagged_ids": sorted(r["id"] for r in acc if r["multi_table"]["multi_table_3plus"])},
        "corpus_status": dict(Counter(r["corpus_status"] for r in recs).most_common()),
        "corpus_status_by_membership": {m: dict(Counter(r["corpus_status"] for r in recs if r["membership"] == m))
                                        for m in MEMBERSHIPS},
        "gates": p_gates(recs),
    }
    return s


def p_gates(recs):
    acc = [r for r in recs if r["outcome"] == "accepted"]
    p1_fail = []
    for r in acc:
        ch = r["verification"]["checks"]
        fam = ch.get("fixtures", {})
        for k in ("l1_explicit_null", "l1_missing", "v6_determinism", "v3_reorder_reversed",
                  "v3_reorder_shuffled_missing", "v3_lookup_keys"):
            if not ch.get(k) or str(ch.get(k)).startswith("FAIL"):
                p1_fail.append((r["id"], k))
        # V5 (no ORDER BY ... LIMIT tie cut) is enforced inside verifier-3: a tie cut is a rejection (tie_cut)
        if ch.get("v8_validate_mql") != "ok" or not r["safety"]["ok"]:
            p1_fail.append((r["id"], "v8"))
        if not fam or any(str(v).startswith("FAIL") for v in fam.values()):
            p1_fail.append((r["id"], "fixtures"))
    unclassified = [r["id"] for r in recs if r["outcome"] != "accepted" and not r.get("outcome_detail")]
    return {
        "P1_equivalence_checks_on_accepted": {"accepted": len(acc), "missing_or_failed_checks": p1_fail,
                                              "status": "pass" if not p1_fail else "fail",
                                              "note": "verifier-3 acceptance (L1 explicit+absent NULL, every fixture "
                                                      "family, V6 double execution, V5 no tie cut, V8); V9 leakage "
                                                      "screen is part of the frozen pilot population"},
        "P2_manual_review": {"status": "PENDING", "note": "needs the named independent human reviewer; the review "
                                                         "package is built, judgements_filled = 0"},
        "P3_no_systematic_error": {"status": "PENDING", "note": "computed from the P2 review"},
        "P4_no_open_critical_bug": {"every_failure_classified": not unclassified, "unclassified": unclassified,
                                    "accepted_known_wrong": "none known; acceptance is not proof of correctness, "
                                                            "P2 decides",
                                    "status": "classification complete; final only after P2"},
        "pilot_gate": "NOT PASSED YET (P2 and P3 pending human review)",
    }


def outcome_key(r):
    """Fields that must reproduce exactly (timings excluded)."""
    v = dict(r.get("verification") or {})
    v.pop("seconds", None)
    keep = {k: r.get(k) for k in ("id", "db_id", "question", "sql", "hardness", "conversion", "code", "detail",
                                  "features", "mql", "stratum", "membership", "outcome", "outcome_detail",
                                  "multi_table", "corpus_status", "corpus_reasons", "safety", "sql_profile")}
    keep["verification"] = v
    return canon(keep)


def write_outputs(out: Path, recs, ids, inputs, hashes, extra_manifest, wilson):
    by = {r["id"]: r for r in recs}
    ordered = [by[i] for i in sorted(ids)]
    files = {}

    def wjsonl(name, rows):
        b = "".join(json.dumps(x, ensure_ascii=False, default=str, sort_keys=True) + "\n" for x in rows).encode("utf-8")
        (out / name).write_bytes(b)
        files[name] = {"sha256": sha_b(b), "records": len(rows)}

    wjsonl("results_per_example.jsonl", ordered)
    wjsonl("accepted.jsonl", [r for r in ordered if r["outcome"] == "accepted"])
    wjsonl("rejected.jsonl", [r for r in ordered if r["outcome"] != "accepted"])
    wjsonl("classifications.jsonl", [{
        "id": r["id"], "db_id": r["db_id"], "hardness": r["hardness"], "stratum": r["stratum"],
        "membership": r["membership"], "outcome": r["outcome"], "outcome_detail": r["outcome_detail"],
        "verifier_status": (r.get("verification") or {}).get("status"),
        "multi_table": r["multi_table"], "corpus_status": r["corpus_status"], "corpus_reasons": r["corpus_reasons"],
        "pilot_evaluation_included": True} for r in ordered])
    wjsonl("corpus_ineligible.jsonl", [{"id": r["id"], "corpus_status": r["corpus_status"], "reasons": r["corpus_reasons"],
                                        "membership": r["membership"], "outcome": r["outcome"]}
                                       for r in ordered if r["corpus_status"] == "not_candidate"])
    wjsonl("multi_table_3plus_quarantine.jsonl", [{"id": r["id"], "membership": r["membership"], "outcome": r["outcome"],
                                                   "multi_table": r["multi_table"], "corpus_status": r["corpus_status"]}
                                                  for r in ordered if r["multi_table"]["multi_table_3plus"]])
    summ = summarize(recs, ids, wilson)
    b = (json.dumps(summ, indent=1, ensure_ascii=False) + "\n").encode("utf-8")
    (out / "summary.json").write_bytes(b)
    files["summary.json"] = {"sha256": sha_b(b)}
    man = {"runner": RUNNER_VERSION, "runner_sha256": sha_f(Path(__file__)), "pilot_ids": sorted(ids),
           "pilot_ids_sha256": sha_b(canon(sorted(ids)).encode("utf-8")), "inputs": inputs,
           "frozen_code_19": {k: v["actual"] for k, v in hashes.items()}, "records_jsonl_sha256": sha_f(out / "records.jsonl"),
           "outputs": files, **extra_manifest}
    (out / "run_manifest.json").write_text(json.dumps(man, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return summ, man


# ===================================================================== commands
def run_dir(run):
    if not run or not run.replace("_", "").isalnum():
        sys.exit("--run must be a simple name like run01")
    return ROOT / OUT_BASE_REL / run


def freeze_path(run):
    return ROOT / RUN_FREEZE_DIR_REL / f"pilot_v3_{run}_runner_freeze.json"


def mongo_digest():
    import shutil
    import subprocess
    exe = shutil.which("docker") or next((str(p) for p in DOCKER_FALLBACK if Path(p).exists()), None)
    if exe is None:
        raise IntegrityError("docker CLI not found: cannot confirm the MongoDB image digest")
    try:
        out = subprocess.run([exe, "inspect", "codegen-mongo-8034", "--format", "{{.Image}}"],
                             capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception as e:  # noqa: BLE001
        raise IntegrityError(f"cannot read the MongoDB container image digest: {e}") from e
    if out != MONGO_DIGEST:
        raise IntegrityError(f"MongoDB container image {out!r} is not the pinned {MONGO_DIGEST}")
    return out


def check_run_freeze(run, hashes, inputs):
    fp = freeze_path(run)
    if not fp.exists():
        raise IntegrityError(f"runner freeze missing ({fp}); run `freeze --run {run}` before execution")
    fz = json.loads(fp.read_text(encoding="utf-8"))
    if fz["runner_sha256"] != sha_f(Path(__file__)):
        raise IntegrityError("runner changed after its freeze: a changed runner needs a NEW run name")
    if fz["frozen_code_19"] != {k: v["actual"] for k, v in hashes.items()} or fz["inputs"] != inputs:
        raise IntegrityError("frozen code or inputs differ from the runner freeze")
    return fz


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("check", "freeze", "run", "finalize", "review", "repro"))
    ap.add_argument("--run", default=None)
    ap.add_argument("--budget", type=float, default=None, help="stop cleanly after S seconds (resume later)")
    ap.add_argument("--i-understand-this-is-the-pilot", action="store_true")
    a = ap.parse_args(argv)
    hashes, inputs, mods = gate()
    C = mods["v2_common"]
    ids, rows = load_pilot()
    membership, pool_rows, _ = load_membership()
    mem = Counter(membership_of(i, membership) for i in ids)
    if a.cmd == "check":
        print(json.dumps({"frozen_code_19": {k: v["match"] for k, v in hashes.items()},
                          "all_19_match": all(v["match"] for v in hashes.values()), "inputs": inputs,
                          "pilot": len(ids), "membership": dict(mem), "verifier": mods["verifier3"].VERIFIER_VERSION,
                          "runner": RUNNER_VERSION, "runner_sha256": sha_f(Path(__file__))}, indent=1))
        return 0
    out = run_dir(a.run)
    if a.cmd == "freeze":
        fp = freeze_path(a.run)
        if fp.exists() or out.exists():
            sys.exit(f"{fp} or {out} exists: refusing to overwrite; use a new run name")
        same = {i: all(pool_rows[i].get(k) == rows[i].get(k) for k in ("question", "sql", "db_id"))
                for i in ids if i in pool_rows}
        doc = {"written_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "run": a.run, "runner": RUNNER_VERSION,
               "runner_sha256": sha_f(Path(__file__)), "verifier": EXPECTED_VERIFIER,
               "converter": mods["sql2mql_v2"].CONVERTER_VERSION, "frozen_code_19": {k: v["actual"] for k, v in hashes.items()},
               "round3_code_freeze_sha256": CODE_FREEZE_SHA, "inputs": inputs,
               "conversion_kit_sha256 (imported by eval_kit, not in the 19)": sha_f(ROOT / "scripts/conversion_kit.py"),
               "mongo": {"image": C.MONGO_IMAGE, "digest": mongo_digest(), "host": f"{C.LOCAL_HOST}:{C.LOCAL_PORT}"},
               "config": {"verify_seed": "verifier3.verify default (20261030)", "fixtures": True,
                          "mongo_maxTimeMS": C.TIMEOUT_MS, "order": "(db_id, id)", "extra_runner_time_limit": None},
               "pilot_ids_sha256": sha_b(canon(sorted(ids)).encode("utf-8")), "pilot_membership": dict(mem),
               "pool_text_equal_for_in_pool_ids": {"checked": len(same), "equal": sum(same.values()),
                                                   "differs": sorted(i for i, v in same.items() if not v)},
               "multi_table_rule": "J2 top-level stratum OR >=3 distinct SQL tables anywhere OR >=2 $lookup stages",
               "corpus_rule": "accepted AND spider_pool_v2 train membership AND NOT multi_table_3plus -> "
                              "candidate_pending_P2_review (never training-eligible here)",
               "note": "written BEFORE any pilot example was converted or verified"}
        fp.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        print(json.dumps({k: doc[k] for k in ("run", "runner_sha256", "pilot_membership", "pool_text_equal_for_in_pool_ids")}, indent=1))
        return 0
    fz = check_run_freeze(a.run, hashes, inputs)
    if a.cmd in ("run", "repro"):
        if not a.i_understand_this_is_the_pilot:
            sys.exit("the pilot runs only with --i-understand-this-is-the-pilot")
        if mongo_digest() != fz["mongo"]["digest"]:
            raise IntegrityError("MongoDB image differs from the runner freeze")
        target = out if a.cmd == "run" else out.with_name(out.name + "_repro")
        if a.cmd == "repro" and not (out / "run_manifest.json").exists():
            sys.exit("finalize the primary run before the reproducibility run")
        client = C.local_client()
        C.check_server(client)
        real = mods["verifier3"].RealDbs(client)
        recs, finished = execute(target, ids, rows, membership, mods, budget=a.budget, real=real)
        print(json.dumps({"records": len(recs), "finished": finished, "out": str(target)}))
        if a.cmd == "repro" and finished:
            prim = {r["id"]: outcome_key(r) for r in load_records(out / "records.jsonl", set(ids))}
            sec = {r["id"]: outcome_key(r) for r in recs}
            diff = sorted(i for i in ids if prim[i] != sec[i])
            rep = {"compared": len(ids), "identical_excluding_timings": len(ids) - len(diff), "differing_ids": diff,
                   "primary_records_sha256": sha_f(out / "records.jsonl"),
                   "repro_records_sha256": sha_f(target / "records.jsonl"),
                   "excluded_fields": ["verification.seconds", "runner_seconds", "record order (resume-dependent)"]}
            (out / "reproducibility.json").write_text(json.dumps(rep, indent=1) + "\n", encoding="utf-8")
            print(json.dumps({k: v for k, v in rep.items() if k != "differing_ids"} | {"differing": len(diff)}))
            return 0 if not diff else 1
        return 0 if finished else 3
    recs = load_records(out / "records.jsonl", set(ids))
    if {r["id"] for r in recs} != set(ids) or len(recs) != 300:
        sys.exit(f"run incomplete: {len(recs)}/300 records")
    if a.cmd == "finalize":
        summ, man = write_outputs(out, recs, ids, inputs, hashes,
                                  {"runner_freeze": {"path": str(freeze_path(a.run).relative_to(ROOT)).replace("\\", "/"),
                                                     "sha256": sha_f(freeze_path(a.run))},
                                   "converter": mods["sql2mql_v2"].CONVERTER_VERSION, "verifier": EXPECTED_VERIFIER,
                                   "mongo": fz["mongo"], "config": fz["config"],
                                   "finalized_at": time.strftime("%Y-%m-%dT%H:%M:%S")}, C.wilson)
        print(json.dumps({k: summ[k] for k in ("all", "by_membership", "corpus_status")}, indent=1))
        return 0
    if a.cmd == "review":
        rp = importlib.import_module("review_package")
        if sha_f(rp.__file__) != REVIEW_PACKAGE_SHA:
            raise IntegrityError("review_package.py changed")
        rout = out / "review_package"
        if rout.exists():
            sys.exit(f"{rout} exists: refusing to overwrite a review package")
        client = C.local_client()
        C.check_server(client)
        rp.build(recs, rout, f"PILOT P2 review (sql2mql-v2, verifier-3, {RUNNER_VERSION}, {a.run})", client)
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
