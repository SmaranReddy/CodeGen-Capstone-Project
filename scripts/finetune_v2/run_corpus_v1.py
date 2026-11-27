"""sql2mql-v2 provisional RAG corpus v1 runner (new file; frozen converter + verifier-3, no verifier-2 path).

    python -B scripts/finetune_v2/run_corpus_v1.py check              eligibility + integrity gate only (no execution)
    python -B scripts/finetune_v2/run_corpus_v1.py freeze             write the runner freeze (before execution)
    python -B scripts/finetune_v2/run_corpus_v1.py run [--budget S]   convert + verify + safety-check (resumable)
    python -B scripts/finetune_v2/run_corpus_v1.py finalize           status, corpus, report, manifest (never overwrites)

Nothing here modifies the converter, verifier-3 or any frozen file. Per-record processing is
run_pilot_v3.process_one imported unchanged (same convert -> safety -> verifier-3 path the pilot used; that module
is hash-pinned in the freeze). The integrity gate (19 frozen code hashes, pinned inputs, MongoDB image digest) runs
before any frozen module is imported and fails closed.

Eligibility (derived, asserted, frozen BEFORE execution):
  * source = data/processed/spider_pool_v2/train.jsonl (5,387 rows, never edited);
  * audit = outputs/spider_pool_semantic_audit_v1/record_audit.jsonl; its id set must equal the source id set;
  * eligible = audit status NO_ISSUE_DETECTED (must be exactly 5,045); must not touch val_internal / quarantine /
    exclusions. NO_ISSUE_DETECTED is NOT proof that the SQL answers the question.
All 5,045 are converted, so each has exactly one processing status. Corpus exclusion is a separate field
(corpus_disposition) so the status counts reconcile.

Processing status (frozen mapping from run_pilot_v3 outcomes):
  accepted -> ACCEPTED; conversion_rejected -> UNSUPPORTED; safety_rejected -> INVALID; timeout -> TIMEOUT;
  verifier_error -> VERIFICATION_ERROR; conversion_error (converter crash) -> EXECUTION_ERROR;
  verification_failed with verifier status sqlite_error / mongo_error -> EXECUTION_ERROR; other verification_failed
  -> REJECTED.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(HERE))
import run_pilot_v3 as R  # noqa: E402  (hash-pinned in the freeze; this module only reads it)

RUNNER_VERSION = "sql2mql-v2-corpus-runner-v1.0"
CORPUS_VERSION = "provisional_corpus_v1"
OUT_REL = "outputs/sql2mql_v2/provisional_corpus_v1"
FREEZE_REL = "data/interim/finetune_v2/corpus_v1_runner_freeze.json"
POOL = "data/processed/spider_pool_v2"
AUDIT_DIR = "outputs/spider_pool_semantic_audit_v1"
CROSS_REVIEW_CSV = "outputs/sql2mql_v2/pilot_v3/run01/cross_review/cross_verified_review.csv"
PILOT_RUN01_RECORDS = "outputs/sql2mql_v2/pilot_v3/run01/records.jsonl"
PILOT_RUN01_FREEZE = "data/interim/finetune_v2/pilot_v3_run01_runner_freeze.json"
EXPECTED_SOURCE_ROWS, EXPECTED_ELIGIBLE = 5387, 5045

STATUSES = ("ACCEPTED", "REJECTED", "UNSUPPORTED", "INVALID", "TIMEOUT", "EXECUTION_ERROR", "VERIFICATION_ERROR")
STATUS_MAPPING = {
    "accepted": "ACCEPTED", "conversion_rejected": "UNSUPPORTED", "safety_rejected": "INVALID",
    "timeout": "TIMEOUT", "verifier_error": "VERIFICATION_ERROR", "conversion_error": "EXECUTION_ERROR",
    "verification_failed[sqlite_error|mongo_error]": "EXECUTION_ERROR", "verification_failed[other]": "REJECTED",
}
EXEC_ERROR_VERIFIER_STATUSES = {"sqlite_error", "mongo_error"}

# evaluation sources used by the leakage screen (relative paths, all read-only)
EVAL_SOURCES = (
    "data/processed/spider_sql_v1/dev_full.jsonl", "data/processed/spider_sql_v1/dev_clean.jsonl",
    "data/processed/spider_sql_v1/test_full.jsonl", "data/processed/spider_sql_v1/test_clean.jsonl",
    "data/processed/spider1_nl_mql.jsonl", "data/processed/spider1_nl_mql_expanded.jsonl",
    "data/splits_expanded/train.jsonl", "data/splits_expanded/validation.jsonl", "data/splits_expanded/test.jsonl",
    "data/rag/corpus_v1/corpus.jsonl", "data/processed/nl_mql_dataset.jsonl",
)
DB_KEYS = ("db_id", "db")
Q_KEYS = ("question",)
SQL_KEYS = ("sql", "gold_sql", "reference_sql")


class CorpusIntegrityError(SystemExit):
    pass


def rd_jsonl(path):
    return [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]


def jbytes(rows):
    return "".join(json.dumps(x, ensure_ascii=False, default=str, sort_keys=True) + "\n" for x in rows).encode("utf-8")


def norm_text(s):
    return re.sub(r"[^a-z0-9]+", " ", str(s).lower()).strip()


def h(s):
    return hashlib.sha256(norm_text(s).encode("utf-8")).hexdigest()


# ===================================================================== eligibility
def derive_eligibility(root: Path = ROOT):
    """-> dict(eligible_ids, rows(by id), audit(by id), pre_excluded, overlaps...). Asserts every step."""
    d = root / POOL
    train = rd_jsonl(d / "train.jsonl")
    rows = {r["id"]: r for r in train}
    if len(rows) != len(train) or len(train) != EXPECTED_SOURCE_ROWS:
        raise CorpusIntegrityError(f"pool train is not {EXPECTED_SOURCE_ROWS} unique rows: {len(train)}")
    audit = {}
    for x in rd_jsonl(root / AUDIT_DIR / "record_audit.jsonl"):
        if x["id"] in audit:
            raise CorpusIntegrityError(f"duplicate audit record {x['id']}")
        audit[x["id"]] = x
    if set(audit) != set(rows):
        raise CorpusIntegrityError("audit id set != pool train id set")
    for i, a in audit.items():     # the audit must describe the same text
        if (a["question"], a["sql"], a["db_id"]) != (rows[i]["question"], rows[i]["sql"], rows[i]["db_id"]):
            raise CorpusIntegrityError(f"audit record {i} text differs from the pool train row")
    counts = Counter(a["status"] for a in audit.values())
    summ = json.loads((root / AUDIT_DIR / "audit_summary.json").read_text(encoding="utf-8"))
    if {k: v for k, v in counts.items() if v} != {k: v for k, v in summ["counts"].items() if v}:
        raise CorpusIntegrityError(f"audit counts differ from audit_summary.json: {dict(counts)}")
    eligible = sorted(i for i, a in audit.items() if a["status"] == "NO_ISSUE_DETECTED")
    if len(eligible) != EXPECTED_ELIGIBLE:
        raise CorpusIntegrityError(f"eligible = {len(eligible)}, expected {EXPECTED_ELIGIBLE}")
    other = {}
    for name in ("val_internal.jsonl", "quarantine.jsonl", "exclusions.jsonl"):
        other[name] = {x["id"] for x in rd_jsonl(d / name)}
        if set(eligible) & other[name]:
            raise CorpusIntegrityError(f"eligible ids intersect {name}")
    # pilot findings (AI cross-review of the 120 P2 items; not human). Only INCORRECT / UNCERTAIN are excluded.
    pre = {}
    cr_rows = list(csv.DictReader((root / CROSS_REVIEW_CSV).open(encoding="utf-8-sig", newline="")))
    if len(cr_rows) != 120:
        raise CorpusIntegrityError("cross-review is not 120 rows")
    cross = {}
    for r in cr_rows:
        dec = r["cross_decision"].strip().upper()
        if dec not in ("CORRECT", "INCORRECT", "UNCERTAIN", "BLOCKED"):
            raise CorpusIntegrityError(f"unknown cross-review decision {dec!r}")
        cross[r["item_id"]] = dec
        if dec != "CORRECT":      # INCORRECT, UNCERTAIN and (fail closed) BLOCKED
            pre[r["item_id"]] = f"pilot_cross_review_{dec}"
    return {"eligible": eligible, "rows": rows, "audit": audit, "audit_counts": dict(counts), "pre_excluded": pre,
            "pre_excluded_in_eligible": sorted(i for i in pre if i in set(eligible)),
            "cross_review": cross, "other_ids": {k: sorted(v) for k, v in other.items()}}


def overlap_slices(eligible, root: Path = ROOT):
    el = set(eligible)
    pilot = set(json.loads((root / R.PILOT_FREEZE_REL).read_text(encoding="utf-8"))["pilot"]["ids"])
    fz = json.loads((root / R.PILOT_FREEZE_REL).read_text(encoding="utf-8"))
    dev500 = set(fz["dev"]["ids"])
    pre = json.loads((root / R.PREREG_REL).read_text(encoding="utf-8"))
    r3 = {i for v in pre["sampling"]["ids"].values() for i in v}
    return {"pilot_300": sorted(pilot & el), "converter_dev_500": sorted(dev500 & el), "round3_740": sorted(r3 & el)}


# ===================================================================== leakage screen (read-only, hashes only)
def load_eval_hashes(root: Path = ROOT):
    dbs, qh, sh, files = set(), set(), set(), {}
    for rel in EVAL_SOURCES:
        p = root / rel
        data = p.read_bytes()
        files[rel] = {"sha256": R.sha_b(data), "bytes": len(data)}
        n = 0
        for line in data.decode("utf-8").splitlines():
            if not line.strip():
                continue
            x = json.loads(line)
            n += 1
            for k in DB_KEYS:
                if x.get(k):
                    dbs.add(str(x[k]))
            for k in Q_KEYS:
                if isinstance(x.get(k), str):
                    qh.add(h(x[k]))
            for k in SQL_KEYS:
                if isinstance(x.get(k), str):
                    sh.add(h(x[k]))
        files[rel]["records"] = n
    return {"dbs": dbs, "q": qh, "sql": sh, "files": files}


def leakage_reasons(row, ev):
    out = []
    if row["db_id"] in ev["dbs"]:
        out.append("leakage_db_overlap")
    if h(row["question"]) in ev["q"]:
        out.append("leakage_question_hash")
    if h(row["sql"]) in ev["sql"]:
        out.append("leakage_sql_hash")
    return out


# ===================================================================== status mapping
def map_status(rec):
    oc = rec["outcome"]
    if oc == "verification_failed":
        return "EXECUTION_ERROR" if (rec.get("verification") or {}).get("status") in EXEC_ERROR_VERIFIER_STATUSES else "REJECTED"
    return STATUS_MAPPING[oc]


# ===================================================================== per-record processing
def schema_meta(rec, prof):
    """Schema facts for the tables the MQL touches (collection + $lookup.from), from the converter's own profile."""
    mql = rec.get("mql")
    if not mql:
        return None
    cols = {mql["collection"]}

    def walk(o):
        if isinstance(o, dict):
            if "$lookup" in o and isinstance(o["$lookup"], dict) and o["$lookup"].get("from"):
                cols.add(o["$lookup"]["from"])
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)
    walk(mql.get("pipeline"))
    pre = rec["db_id"] + "__"
    out = []
    for c in sorted(cols):
        t = prof.table(c[len(pre):]) if c.startswith(pre) else None
        if t is None:
            out.append({"collection": c, "table": None, "columns": []})
            continue
        out.append({"collection": c, "table": t, "primary_key": list(prof.pk.get(t, [])),
                    "columns": [{"name": n, "declared_type": col.decl, "sqlite_affinity": col.affinity,
                                 "observed_kind": col.kind} for n, col in prof.tables[t].items()]})
    return {"database_tables": sorted(prof.tables), "touched": out}


def process_one(r, mods, real, profs, state):
    if state.get("db") != r["db_id"]:
        profs.clear()
        state["db"] = r["db_id"]
    rec = R.process_one(r, mods, real, profs, "train")      # unchanged pilot path (convert -> safety -> verifier-3)
    rec["runner"] = RUNNER_VERSION
    for k in ("corpus_status", "corpus_reasons"):            # pilot-only fields; the corpus decision is made in finalize
        rec.pop(k, None)
    rec["status"] = map_status(rec)
    rec["schema_meta"] = schema_meta(rec, profs[r["db_id"]]) if rec.get("conversion") == "converted" else None
    return rec


# ===================================================================== resume-safe store
def load_records(path: Path, eligible_ids, log_path: Path | None = None):
    if not path.exists():
        return []
    lines = [x for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    rows, cut = [], 0
    for k, x in enumerate(lines):
        try:
            rows.append(json.loads(x))
        except json.JSONDecodeError:
            if k != len(lines) - 1:
                raise CorpusIntegrityError(f"{path}: unparseable line {k + 1} that is not the last line")
            cut = 1
    seen = set()
    for x in rows:
        if x.get("id") not in eligible_ids:
            raise CorpusIntegrityError(f"{path}: record {x.get('id')} is not an eligible id")
        if x["id"] in seen:
            raise CorpusIntegrityError(f"{path}: duplicate record {x['id']}")
        seen.add(x["id"])
        if x.get("runner") != RUNNER_VERSION:
            raise CorpusIntegrityError(f"{path}: record {x['id']} from runner {x.get('runner')}")
        if "verification" in x and x["verification"].get("verifier") != R.EXPECTED_VERIFIER:
            raise CorpusIntegrityError(f"{path}: record {x['id']} verified by {x['verification'].get('verifier')}")
        if x.get("outcome") not in R.OUTCOMES or x.get("status") not in STATUSES:
            raise CorpusIntegrityError(f"{path}: record {x['id']} has outcome/status {x.get('outcome')}/{x.get('status')}")
    if cut:
        path.write_text("".join(json.dumps(x, ensure_ascii=False, default=str) + "\n" for x in rows), encoding="utf-8")
    if log_path is not None and (rows or cut):
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"at": time.strftime("%Y-%m-%dT%H:%M:%S"), "records_kept": len(rows),
                                "cut_trailing_line_dropped": cut}) + "\n")
    return rows


def execute(out: Path, eligible, rows, mods, budget=None, real=None, process=process_one):
    out.mkdir(parents=True, exist_ok=True)
    path = out / "records.jsonl"
    ids = set(eligible)
    prev = load_records(path, ids, out / "resume_log.jsonl")
    done = {x["id"] for x in prev}
    todo = sorted((rows[i] for i in eligible if i not in done), key=lambda r: (r["db_id"], r["id"]))
    profs, state, t0 = {}, {}, time.time()
    for k, r in enumerate(todo, 1):
        if budget is not None and time.time() - t0 > budget:
            print(f"budget reached after {k - 1}/{len(todo)}; re-run to resume", flush=True)
            break
        rec = process(r, mods, real, profs, state)
        R.append_record(path, rec)
        if k % 25 == 0 or k == len(todo):
            print(f"{k}/{len(todo)} ({len(done)} earlier)  {time.time() - t0:.0f}s", flush=True)
    recs = load_records(path, ids)
    return recs, {x["id"] for x in recs} == ids


# ===================================================================== finalize (pure functions of records + inputs)
def corpus_disposition(rec, pre_excluded, ev, row):
    """-> (disposition, reasons). INCLUDED only if ACCEPTED, not multi-table-3plus, not pilot-flagged, no leakage."""
    reasons = []
    if rec["status"] != "ACCEPTED":
        reasons.append(f"status:{rec['status']}")
    if rec["multi_table"]["multi_table_3plus"]:
        reasons.append("multi_table_3plus")
    if rec["id"] in pre_excluded:
        reasons.append(pre_excluded[rec["id"]])
    reasons += leakage_reasons(row, ev)
    return ("INCLUDED" if not reasons else "EXCLUDED"), reasons


def sql_pattern_key(rec):
    return rec["sql_profile"]["pattern"]


def corpus_entry(rec, row, audit, cross, freeze_ctx):
    cr = cross.get(rec["id"])
    return {
        "corpus_entry_id": f"{CORPUS_VERSION}:{rec['id']}",
        "corpus_version": CORPUS_VERSION,
        "corpus_status": "PROVISIONAL",
        "human_review_status": "NOT_REVIEWED",
        "id": rec["id"],
        "db_id": rec["db_id"],                      # retrieval-time target-DB exclusion key (drop entry.db_id == target)
        "question": row["question"],
        "question_raw": row.get("question_raw"),
        "sql": row["sql"],
        "mql": rec["mql"],
        "hardness": row["hardness"],
        "query_class": row.get("query_class"),
        "sql_group": row.get("group"),
        "sql_features": row.get("features"),
        "sql_pattern": rec["sql_profile"]["pattern"],
        "converter_features": rec.get("features"),
        "stratum": rec["stratum"],
        "schema": rec["schema_meta"],
        "source": {"file": "data/processed/spider_pool_v2/train.jsonl", "split": row["split"], "origin": row.get("source")},
        "source_label_audit": {"audit": "spider_pool_semantic_audit_v1", "status": audit["status"],
                               "note": "NO_ISSUE_DETECTED is absence of detector flags, not proof that the SQL answers the question",
                               "pilot_ai_cross_review": ({"decision": cr, "reviewer": "AI (not human)"} if cr else None)},
        "verification": {"outcome": "ACCEPTED", "verifier": R.EXPECTED_VERIFIER, "status": rec["verification"]["status"],
                         "establishes": "SQL-to-MQL result equivalence only; NOT that the SQL answers the question"},
        "converter_version": rec["converter"],
        "verifier_version": R.EXPECTED_VERIFIER,
        "code_freeze_sha256": freeze_ctx["code_freeze_sha256"],
        "pipeline_runner": RUNNER_VERSION,
    }


def exclude_database(entries, target_db_id):
    """Retrieval helper: never return examples from the database being evaluated."""
    return [e for e in entries if e["db_id"] != target_db_id]


def status_row(rec, disp, reasons):
    return {"id": rec["id"], "db_id": rec["db_id"], "hardness": rec["hardness"], "status": rec["status"],
            "raw_outcome": rec["outcome"], "outcome_detail": rec["outcome_detail"],
            "verifier_status": (rec.get("verification") or {}).get("status"),
            "conversion": rec["conversion"], "converter_code": rec.get("code"),
            "multi_table_3plus": rec["multi_table"]["multi_table_3plus"],
            "corpus_disposition": disp, "corpus_exclusion_reasons": reasons}


def pct(k, n):
    return {"k": k, "n": n, "pct": round(100 * k / n, 2) if n else None}


def group_block(rs):
    n = len(rs)
    c = Counter(r["status"] for r in rs)
    return {"records": n, "converted": pct(sum(r["conversion"] == "converted" for r in rs), n),
            "accepted": pct(c["ACCEPTED"], n), "statuses": {s: c.get(s, 0) for s in STATUSES},
            "in_corpus": sum(r["_disp"] == "INCLUDED" for r in rs)}


def build_outputs(recs, ctx):
    """Pure + deterministic: records -> {filename: bytes}. ctx carries eligibility/eval/freeze context."""
    el, rows, audit = ctx["eligible"], ctx["rows"], ctx["audit"]
    by = {r["id"]: r for r in recs}
    if set(by) != set(el) or len(recs) != len(el):
        raise CorpusIntegrityError("records do not cover the eligible set exactly once")
    ordered = [by[i] for i in sorted(el)]
    files = {}
    for r in ordered:
        r["_disp"], r["_reasons"] = corpus_disposition(r, ctx["pre_excluded"], ctx["ev"], rows[r["id"]])
    clean = [{k: v for k, v in r.items() if not k.startswith("_")} for r in ordered]
    files["per_record_results.jsonl"] = jbytes(clean)
    files["conversion_status.jsonl"] = jbytes([status_row(r, r["_disp"], r["_reasons"]) for r in ordered])
    files["accepted_conversions.jsonl"] = jbytes([x for x, r in zip(clean, ordered) if r["status"] == "ACCEPTED"])
    files["rejected_conversions.jsonl"] = jbytes([x for x, r in zip(clean, ordered) if r["status"] != "ACCEPTED"])
    corpus = [corpus_entry(r, rows[r["id"]], audit[r["id"]], ctx["cross_review"], ctx["freeze"])
              for r in ordered if r["_disp"] == "INCLUDED"]
    files["provisional_rag_corpus.jsonl"] = jbytes(corpus)
    files["excluded_accepted.jsonl"] = jbytes([
        {"id": r["id"], "db_id": r["db_id"], "reasons": r["_reasons"]} for r in ordered
        if r["status"] == "ACCEPTED" and r["_disp"] == "EXCLUDED"])
    n = len(ordered)
    c = Counter(r["status"] for r in ordered)
    acc = [r for r in ordered if r["status"] == "ACCEPTED"]
    reason_ct = Counter(x for r in ordered for x in r["_reasons"] if not x.startswith("status:"))
    acc_reason_ct = Counter(x for r in acc for x in r["_reasons"])
    sub = lambda rs: group_block(rs)  # noqa: E731
    summary = {
        "eligible_source_records": n,
        "converted": pct(sum(r["conversion"] == "converted" for r in ordered), n),
        "statuses": {s: c.get(s, 0) for s in STATUSES},
        "status_sum_equals_eligible": sum(c.values()) == n,
        "accepted": pct(c["ACCEPTED"], n),
        "accepted_of_converted": pct(c["ACCEPTED"], sum(r["conversion"] == "converted" for r in ordered)),
        "unsupported_by_converter_code": dict(Counter(r["code"] for r in ordered if r["status"] == "UNSUPPORTED").most_common()),
        "rejected_by_verifier_status": dict(Counter(r["verification"]["status"] for r in ordered if r["status"] == "REJECTED").most_common()),
        "execution_error_detail": dict(Counter(f"{r['outcome']}:{(r.get('verification') or {}).get('status') or r['code'] if r['outcome']=='conversion_error' else (r.get('verification') or {}).get('status')}" for r in ordered if r["status"] == "EXECUTION_ERROR")),
        "excluded_from_corpus_accepted": {
            "accepted_total": len(acc), "accepted_excluded": sum(r["_disp"] == "EXCLUDED" for r in acc),
            "by_reason_(non-exclusive)": dict(acc_reason_ct.most_common())},
        "additional_restriction_flags_all_records": dict(reason_ct.most_common()),
        "multi_table_3plus": {"all": sum(r["multi_table"]["multi_table_3plus"] for r in ordered),
                              "accepted": sum(r["multi_table"]["multi_table_3plus"] for r in acc)},
        "corpus_size": len(corpus),
        "by_database": {d: sub([r for r in ordered if r["db_id"] == d]) for d in sorted({r["db_id"] for r in ordered})},
        "by_hardness": {k: sub([r for r in ordered if r["hardness"] == k]) for k in ("easy", "medium", "hard", "extra")},
        "by_stratum": {k: sub([r for r in ordered if r["stratum"] == k]) for k in ("J2", "OL", "J1", "G")},
        "by_sql_pattern": {p: sub([r for r in ordered if p in sql_pattern_key(r)])
                           for p in sorted({p for r in ordered for p in sql_pattern_key(r)})},
        "corpus_by_database": dict(sorted(Counter(e["db_id"] for e in corpus).items())),
        "corpus_by_hardness": dict(Counter(e["hardness"] for e in corpus)),
        "reporting_slices": {k: sub([by[i] for i in v]) for k, v in ctx["slices"].items()},
        "pilot_cross_review_ids_in_eligible": ctx["pre_excluded_in_eligible"],
        "leakage": ctx["leakage_summary"],
    }
    for r in ordered:
        r.pop("_disp", None)
        r.pop("_reasons", None)
    files["summary.json"] = (json.dumps(summary, indent=1, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")
    return files, summary


def pilot_overlap_repro(recs, root: Path = ROOT):
    """Compare re-run records with run01 for ids in both (no second 2.3h run needed)."""
    p = root / PILOT_RUN01_RECORDS
    old = {x["id"]: x for x in rd_jsonl(p)}
    new = {r["id"]: r for r in recs}
    common = sorted(set(old) & set(new))
    strip = ("runner", "membership", "corpus_status", "corpus_reasons", "status", "schema_meta", "runner_seconds")

    def full(x):
        y = {k: v for k, v in x.items() if k not in strip}
        v = dict(y.get("verification") or {})
        v.pop("seconds", None)
        y["verification"] = v
        return R.canon(y)

    def subst(x):
        return R.canon({"outcome": x["outcome"], "detail": x["outcome_detail"], "mql": x.get("mql"),
                        "vstatus": (x.get("verification") or {}).get("status"), "sql": x["sql"], "question": x["question"]})
    text_differs = [i for i in common if (old[i]["question"], old[i]["sql"]) != (new[i]["question"], new[i]["sql"])]
    comp = [i for i in common if i not in text_differs]
    return {"in_both": len(common), "source_text_differs_pool_vs_sql_v1": text_differs, "compared": len(comp),
            "strict_identical_excluding_timings": sum(full(old[i]) == full(new[i]) for i in comp),
            "substantive_identical(outcome,detail,mql,verifier_status)": sum(subst(old[i]) == subst(new[i]) for i in comp),
            "strict_differing_ids": [i for i in comp if full(old[i]) != full(new[i])],
            "substantive_differing_ids": [i for i in comp if subst(old[i]) != subst(new[i])]}


def make_report(summary, man, repro):
    s = summary
    L = []
    w = L.append
    w("# Provisional RAG corpus v1: conversion report\n")
    w(f"Generated by `scripts/finetune_v2/run_corpus_v1.py` ({RUNNER_VERSION}); converter `sql2mql-v2`, verifier "
      f"`{R.EXPECTED_VERIFIER}` (frozen, 19 code hashes verified before execution). Local MongoDB 8.0.34 label validation, "
      "not an Atlas metric and not a model result.\n")
    w("> **Provisional, not validated.** Verifier-3 establishes SQL-to-MQL result equivalence. It does not establish that the "
      "original Spider SQL correctly answers the question. `NO_ISSUE_DETECTED` means the semantic-audit detectors raised no flag "
      "(they found about 24% of already-known-bad rows at likely-or-above level). Every entry is `human_review_status: NOT_REVIEWED`.\n")
    w("## 1. Counts (every eligible record has exactly one status)\n")
    n = s["eligible_source_records"]
    w("| Quantity | Count | % of eligible |\n|---|---:|---:|")
    w(f"| Eligible source records (audit `NO_ISSUE_DETECTED`) | {n} | 100.00 |")
    w(f"| Converted by the V2 converter | {s['converted']['k']} | {s['converted']['pct']} |")
    for st in STATUSES:
        w(f"| Status {st} | {s['statuses'][st]} | {round(100 * s['statuses'][st] / n, 2)} |")
    w(f"| Sum of statuses | {sum(s['statuses'].values())} | reconciles: {s['status_sum_equals_eligible']} |")
    w(f"| ACCEPTED of converted | {s['accepted_of_converted']['k']} / {s['accepted_of_converted']['n']} | {s['accepted_of_converted']['pct']} |")
    ex = s["excluded_from_corpus_accepted"]
    w(f"| Accepted but excluded from corpus | {ex['accepted_excluded']} | |")
    w(f"| **Final provisional corpus size** | **{s['corpus_size']}** | {round(100 * s['corpus_size'] / n, 2)} |\n")
    w("Status mapping (frozen before execution): " + "; ".join(f"`{k}` -> {v}" for k, v in STATUS_MAPPING.items()) + ".\n")
    w("UNSUPPORTED by converter code: " + ", ".join(f"`{k}` {v}" for k, v in s["unsupported_by_converter_code"].items()) + ".\n")
    w("REJECTED by verifier status: " + (", ".join(f"`{k}` {v}" for k, v in s["rejected_by_verifier_status"].items()) or "none") + ".\n")
    w("## 2. Additional eligibility restrictions\n")
    w("Reasons are not exclusive; an accepted record can carry several. Counts over ACCEPTED records:\n")
    w("| Reason | Accepted records |\n|---|---:|")
    for k, v in ex["by_reason_(non-exclusive)"].items():
        w(f"| {k} | {v} |")
    w(f"\n3+-table rule (identical to the pilot): J2 stratum, or >= 3 distinct SQL tables, or >= 2 `$lookup` stages. "
      f"Flagged: {s['multi_table_3plus']['all']} of all records, {s['multi_table_3plus']['accepted']} of accepted.\n")
    w(f"Pilot cross-review INCORRECT/UNCERTAIN ids that are in the eligible set: {len(s['pilot_cross_review_ids_in_eligible'])} "
      f"({', '.join(s['pilot_cross_review_ids_in_eligible']) or 'none'}).\n")

    def tab(title, d, key_label):
        w(f"## {title}\n")
        w(f"| {key_label} | records | converted | accepted | acc % | corpus |\n|---|---:|---:|---:|---:|---:|")
        for k, b in d.items():
            w(f"| {k} | {b['records']} | {b['converted']['k']} | {b['accepted']['k']} | {b['accepted']['pct']} | {b['in_corpus']} |")
        w("")
    tab("3. By query difficulty (Spider hardness)", s["by_hardness"], "hardness")
    tab("4. By converter stratum", s["by_stratum"], "stratum")
    tab("5. By SQL pattern (patterns overlap)", s["by_sql_pattern"], "pattern")
    tab("6. By database", s["by_database"], "database")
    tab("7. Reporting slices (not exclusions)", s["reporting_slices"], "slice")
    w("The `converter_dev_500` slice was used while the converter was developed and `round3_740` to gate verifier-3, so their "
      "acceptance rates are in-sample for the converter; they are not independent estimates.\n")
    lk = s["leakage"]
    w("## 8. Integrity and leakage checks\n")
    w(f"- 19/19 frozen code hashes matched before any frozen module was imported; MongoDB image digest pinned.\n"
      f"- Audit id set == pool train id set ({EXPECTED_SOURCE_ROWS}); eligible = {n}; disjoint from val_internal, quarantine and exclusions.\n"
      f"- Leakage screen over {len(lk['eval_files'])} evaluation files (hashes only): db_id overlap {lk['db_overlap_with_corpus']}, "
      f"question-hash hits among eligible {lk['question_hash_hits_eligible']}, sql-hash hits {lk['sql_hash_hits_eligible']}; "
      f"corpus databases: {lk['corpus_databases']}, evaluation databases screened: {lk['eval_databases']}.\n"
      f"- Pool val_internal / quarantine / exclusions / dev / test ids in the corpus: 0.\n"
      f"- Re-run vs pilot run01 on shared ids: {repro['compared']} compared, {repro['substantive_identical(outcome,detail,mql,verifier_status)']} "
      f"substantive-identical, {repro['strict_identical_excluding_timings']} strict-identical "
      f"(source text differs pool vs sql-v1 for {repro['source_text_differs_pool_vs_sql_v1']}).\n")
    w("## 9. Remaining quality limitations\n")
    w("- The corpus holds only what the converter supports: no subqueries or set operations, and 3+-table joins are held out. "
      "It is biased toward easy and medium single/two-table queries.\n"
      "- Verifier-3 cannot see a wrong gold SQL. The pilot's AI cross-review found 6 of 120 verifier-accepted labels wrong "
      "(5% of a sample) and 3 uncertain, all errors in Spider's SQL rather than the conversion. Expect a comparable rate here.\n"
      "- The semantic audit's recall is low, so unflagged does not mean correct. No human review has happened (P2 pending).\n"
      "- Acceptance is a local MongoDB 8.0.34 check on bundled SQLite data plus synthetic fixtures; it is not an Atlas result.\n"
      "- `human_review_status` is `NOT_REVIEWED` for every entry, including pilot items that had AI cross-review.\n"
      "- Do not use this corpus for fine-tuning without a reviewed decision. Retrieval must exclude the target database "
      "(`exclude_database`).\n")
    return "\n".join(L) + "\n"


# ===================================================================== commands
def out_dir():
    return ROOT / OUT_REL


def runner_hashes():
    return {"run_corpus_v1.py": R.sha_f(Path(__file__)), "run_pilot_v3.py": R.sha_f(HERE / "run_pilot_v3.py"),
            "scripts/conversion_kit.py": R.sha_f(ROOT / "scripts/conversion_kit.py")}


def input_hashes(elig):
    r = ROOT
    return {"pool_train.jsonl": R.sha_f(r / POOL / "train.jsonl"),
            "record_audit.jsonl": R.sha_f(r / AUDIT_DIR / "record_audit.jsonl"),
            "audit_summary.json": R.sha_f(r / AUDIT_DIR / "audit_summary.json"),
            "pilot_cross_review_csv": R.sha_f(r / CROSS_REVIEW_CSV),
            "pilot_run01_records.jsonl": R.sha_f(r / PILOT_RUN01_RECORDS),
            "pilot_run01_runner_freeze.json": R.sha_f(r / PILOT_RUN01_FREEZE),
            "eligible_ids_sha256": R.sha_b(R.canon(elig["eligible"]).encode("utf-8")),
            "pre_excluded": elig["pre_excluded"]}


def freeze_context():
    fz = json.loads((ROOT / FREEZE_REL).read_text(encoding="utf-8")) if (ROOT / FREEZE_REL).exists() else None
    return fz


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("check", "freeze", "run", "finalize"))
    ap.add_argument("--budget", type=float, default=None)
    a = ap.parse_args(argv)
    hashes, inputs, mods = R.gate()
    elig = derive_eligibility()
    slices = overlap_slices(elig["eligible"])
    ih = input_hashes(elig)
    summary_view = {"all_19_match": all(v["match"] for v in hashes.values()), "eligible": len(elig["eligible"]),
                    "audit_counts": elig["audit_counts"], "pre_excluded_total": len(elig["pre_excluded"]),
                    "pre_excluded_in_eligible": elig["pre_excluded_in_eligible"],
                    "slices": {k: len(v) for k, v in slices.items()}, "runner": runner_hashes()}
    if a.cmd == "check":
        print(json.dumps(summary_view, indent=1))
        return 0
    C = mods["v2_common"]
    if a.cmd == "freeze":
        fp = ROOT / FREEZE_REL
        if fp.exists() or out_dir().exists():
            sys.exit(f"{fp} or {out_dir()} exists: refusing to overwrite")
        ev = load_eval_hashes()
        doc = {"written_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "runner": RUNNER_VERSION, "runner_hashes": runner_hashes(),
               "verifier": R.EXPECTED_VERIFIER, "converter": mods["sql2mql_v2"].CONVERTER_VERSION,
               "code_freeze_sha256": R.CODE_FREEZE_SHA, "frozen_code_19": {k: v["actual"] for k, v in hashes.items()},
               "pinned_inputs": inputs, "eligibility_inputs": ih, "audit_counts": elig["audit_counts"],
               "eligible_n": len(elig["eligible"]), "eligible_ids": elig["eligible"],
               "reporting_slices": {k: len(v) for k, v in slices.items()},
               "status_mapping": STATUS_MAPPING, "eval_sources": ev["files"],
               "mongo": {"image": C.MONGO_IMAGE, "digest": R.mongo_digest(), "host": f"{C.LOCAL_HOST}:{C.LOCAL_PORT}"},
               "config": {"verify_seed": "verifier3.verify default (20261030)", "fixtures": True,
                          "mongo_maxTimeMS": C.TIMEOUT_MS, "order": "(db_id, id)", "batch_specific_rules": None},
               "multi_table_rule": "J2 top-level stratum OR >=3 distinct SQL tables anywhere OR >=2 $lookup stages (identical to the pilot)",
               "corpus_rule": "ACCEPTED AND NOT multi_table_3plus AND NOT pilot cross-review INCORRECT/UNCERTAIN AND no leakage",
               "dependency_versions": dep_versions(),
               "note": "written BEFORE any eligible record was converted or verified"}
        fp.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        print(json.dumps({"freeze": FREEZE_REL, "eligible": doc["eligible_n"], "eval_files": len(ev["files"])}, indent=1))
        return 0
    fp = ROOT / FREEZE_REL
    if not fp.exists():
        raise R.IntegrityError("corpus runner freeze missing; run `freeze` before execution")
    fz = json.loads(fp.read_text(encoding="utf-8"))
    if fz["runner_hashes"] != runner_hashes():
        raise R.IntegrityError("runner, run_pilot_v3.py or conversion_kit.py changed after the freeze")
    if fz["frozen_code_19"] != {k: v["actual"] for k, v in hashes.items()} or fz["pinned_inputs"] != inputs \
            or fz["eligibility_inputs"] != ih or fz["eligible_ids"] != elig["eligible"]:
        raise R.IntegrityError("frozen code, inputs or the eligible set differ from the freeze")
    out = out_dir()
    if a.cmd == "run":
        if R.mongo_digest() != fz["mongo"]["digest"]:
            raise R.IntegrityError("MongoDB image differs from the freeze")
        client = C.local_client()
        C.check_server(client)
        real = mods["verifier3"].RealDbs(client)
        recs, finished = execute(out, elig["eligible"], elig["rows"], mods, budget=a.budget, real=real)
        print(json.dumps({"records": len(recs), "finished": finished}))
        return 0 if finished else 3
    # finalize
    if (out / "manifest.json").exists():
        sys.exit("manifest.json exists: refusing to overwrite a finalized corpus (use a new version)")
    recs = load_records(out / "records.jsonl", set(elig["eligible"]))
    if {r["id"] for r in recs} != set(elig["eligible"]):
        sys.exit(f"run incomplete: {len(recs)}/{len(elig['eligible'])}")
    ev = load_eval_hashes()
    if {k: {x: v[x] for x in ("sha256", "records")} for k, v in ev["files"].items()} != \
            {k: {x: v[x] for x in ("sha256", "records")} for k, v in fz["eval_sources"].items()}:
        raise R.IntegrityError("evaluation sources changed since the freeze")
    leaks = {i: leakage_reasons(elig["rows"][i], ev) for i in elig["eligible"]}
    corpus_dbs = {elig["rows"][i]["db_id"] for i in elig["eligible"]}
    lk = {"eval_files": sorted(ev["files"]), "eval_databases": len(ev["dbs"]), "corpus_databases": len(corpus_dbs),
          "db_overlap_with_corpus": sorted(corpus_dbs & ev["dbs"]),
          "question_hash_hits_eligible": sum("leakage_question_hash" in v for v in leaks.values()),
          "sql_hash_hits_eligible": sum("leakage_sql_hash" in v for v in leaks.values()),
          "question_hash_hit_ids": sorted(i for i, v in leaks.items() if "leakage_question_hash" in v),
          "sql_hash_hit_ids": sorted(i for i, v in leaks.items() if "leakage_sql_hash" in v)}
    ctx = {"eligible": elig["eligible"], "rows": elig["rows"], "audit": elig["audit"], "pre_excluded": elig["pre_excluded"],
           "pre_excluded_in_eligible": elig["pre_excluded_in_eligible"], "cross_review": elig["cross_review"], "ev": ev,
           "freeze": fz, "slices": slices, "leakage_summary": lk}
    files, summary = build_outputs(recs, ctx)
    repro = pilot_overlap_repro(recs)
    files["pilot_overlap_reproducibility.json"] = (json.dumps(repro, indent=1) + "\n").encode("utf-8")
    man = {"corpus_version": CORPUS_VERSION, "corpus_status": "PROVISIONAL", "human_review_status": "NOT_REVIEWED",
           "runner": RUNNER_VERSION, "runner_hashes": runner_hashes(), "converter": fz["converter"], "verifier": R.EXPECTED_VERIFIER,
           "code_freeze_sha256": R.CODE_FREEZE_SHA, "frozen_code_19": fz["frozen_code_19"], "pinned_inputs": inputs,
           "eligibility_inputs": ih, "runner_freeze": {"path": FREEZE_REL, "sha256": R.sha_f(fp)},
           "config": fz["config"], "mongo": fz["mongo"], "dependency_versions": fz["dependency_versions"],
           "status_mapping": STATUS_MAPPING, "eval_sources": ev["files"], "records_jsonl_sha256": R.sha_f(out / "records.jsonl"),
           "counts": {"eligible": summary["eligible_source_records"], "statuses": summary["statuses"],
                      "corpus_size": summary["corpus_size"]},
           "outputs": {k: {"sha256": R.sha_b(v), "bytes": len(v)} for k, v in sorted(files.items())},
           "finalized_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    rep = make_report(summary, man, repro).encode("utf-8")
    files["conversion_report.md"] = rep
    man["outputs"]["conversion_report.md"] = {"sha256": R.sha_b(rep), "bytes": len(rep)}
    for k, v in files.items():
        if (out / k).exists():
            sys.exit(f"{k} exists: refusing to overwrite")
    for k, v in files.items():
        (out / k).write_bytes(v)
    (out / "manifest.json").write_text(json.dumps(man, indent=1, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("eligible_source_records", "converted", "statuses", "accepted", "corpus_size")}, indent=1))
    return 0


def dep_versions():
    import importlib.metadata as md
    import platform
    out = {"python": platform.python_version(), "platform": platform.platform()}
    for p in ("pymongo", "sqlglot"):
        try:
            out[p] = md.version(p)
        except Exception:  # noqa: BLE001
            out[p] = None
    import sqlite3
    out["sqlite3"] = sqlite3.sqlite_version
    return out


if __name__ == "__main__":
    sys.exit(main())
