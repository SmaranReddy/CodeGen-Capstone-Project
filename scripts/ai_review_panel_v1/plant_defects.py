"""Planted-defect set for AI-panel calibration (plan §E.7.3). ORCHESTRATOR-ONLY tool.

    python -B scripts/ai_review_panel_v1/plant_defects.py build

Builds 60 source-side and 60 translation-side planted items (one known defect each) plus 60 clean controls, from
correct accepted labels:
  * bases: the run01 review items the multi-agent audit accepted in BOTH dimensions (108), then, if a class needs
    more, seeded supplementary spider_pool_v2 train records with NO detector hit, not a finding/sibling and not in
    the 300 pilot ids. Every base is re-converted with sql2mql-v2.1 and must be ACCEPTED by verifier 3.1.
  * source plants: the SQL is changed (or the question truncated); the MQL is re-converted with v2.1 from the
    planted SQL and must be ACCEPTED by verifier 3.1 (so the translation dimension stays correct); the planted SQL
    must return a different result from the base SQL on the real database (so, the base being correct for its
    question, the planted SQL is wrong for it).
  * translation plants: the SQL is the base SQL; the MQL is a single-edit mutant (mutants_v2, imported read-only)
    or one of two known run01 classes (v2 Unicode-folding regex; untyped numeric $gt on a TEXT column); the mutant
    must be REJECTED by verifier 3.1 with a semantic status (real data or fixture counterexample exists).
Planted items are evaluation controls only: they are never written to any dataset or corpus. Their key lives in
outputs/ai_review_panel_v1/_orchestrator_only/ (outside every agent's allowed paths).

Verifier fixture collections use databases of their own (panel_plant_*), so this never touches collections used
by any concurrent runner.
"""
import copy
import csv
import hashlib
import json
import random
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / "scripts/finetune_v2"))
import sqlglot  # noqa: E402
from sqlglot import exp  # noqa: E402

import v2_common as C  # noqa: E402
import schema_profile as SP  # noqa: E402
import sql2mql_v2 as V2  # noqa: E402
import sql2mql_v2_1 as N  # noqa: E402
import verifier3 as V3  # noqa: E402
import verifier3_1 as V31  # noqa: E402
import mutants_v2 as MU  # noqa: E402

V3.FIXTURE_DB, V3.REORDER_DB, V31.FIXTURE_DB = "panel_plant_v3_fixtures", "panel_plant_v3_reorder", "panel_plant_v31_fixtures"
SEED = 20261113
OUT = ROOT / "outputs/ai_review_panel_v1/_orchestrator_only"
RUN01 = ROOT / "outputs/sql2mql_v2/pilot_v3/run01"
CASE_CSV = RUN01 / "manual_human_review/ai_multiagent_review/case_level_results.csv"
SEMANTIC = {"l1_mismatch_explicit_null", "l1_mismatch_missing", "fixture_mismatch", "v3_reorder_mismatch"}
SRC_QUOTA = {"sort_direction_flip": 10, "literal_to_sibling_column": 10, "distinct_dropped": 8,
             "count_distinct_to_star": 8, "truncated_question": 10, "comparison_flip": 8, "left_to_inner_all_x": 6}
TRN_CLASSES = {
    "filter": ("match_atom_drop", "literal_plus1", "in_drop_value"),
    "null_handling": ("null_guard_drop", "ifnull_drop"),
    "sort_limit": ("sort_flip", "sort_drop", "limit_plus1", "limit_drop", "skip_drop"),
    "aggregation": ("count_nonnull_to_star", "sum_guard_drop", "facet_unwrap_B1", "trunc_drop", "zero_guard_drop"),
    "join": ("left_to_inner", "inner_to_left", "lookup_field_swap"),
    "projection_grouping": ("group_key_swap", "project_swap"),
    "regex_anchor": ("regex_anchor_end", "regex_anchor_start"),
}
TRN_QUOTA = {"filter": 8, "null_handling": 7, "sort_limit": 8, "aggregation": 8, "join": 8, "projection_grouping": 6,
             "regex_anchor": 5, "unicode_regex_fold": 6, "untyped_gt_text": 4}
N_SRC, N_TRN, N_CLEAN = 60, 60, 60


def rd(p):
    return [json.loads(x) for x in Path(p).read_text(encoding="utf-8").splitlines() if x.strip()]


def sha_s(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


class Env:
    def __init__(self):
        cl = C.local_client()
        C.check_server(cl)
        self.real = V31.RealDbs(cl)
        self.client = cl
        self.profs = {}

    def prof(self, db):
        if db not in self.profs:
            if len(self.profs) > 8:
                self.profs.pop(next(iter(self.profs)))
            self.profs[db] = SP.Profile(db)
        return self.profs[db]

    def convert(self, db, sql):
        try:
            return N.convert(sql, self.prof(db))[0]
        except Exception:  # noqa: BLE001
            return None

    def verify(self, rid, db, sql, mql):
        try:
            return V31.verify({"id": rid, "db_id": db, "sql": sql}, mql, self.real)
        except Exception as e:  # noqa: BLE001
            return {"status": "verifier_error", "detail": str(e)[:200]}

    def sql_result(self, db, sql):
        con = C.sqlite_ro(C.sqlite_path(db))
        try:
            return [tuple(r) for r in con.execute(sql).fetchall()]
        except Exception:  # noqa: BLE001
            return None
        finally:
            con.close()


# ---------------------------------------------------------------- source-side plants (SQL / question edits)
DIR_CUES = r"\b(descending|ascending|highest|lowest|most|least|largest|smallest|top|bottom|maximum|minimum|biggest|greatest|fewest|oldest|youngest|longest|shortest|best|worst|first|last|alphabetical|decreasing|increasing|order)\b"


def plant_source(rec, env):
    """-> list of (class, planted_sql or None, planted_question or None, description)."""
    sql, q, db = rec["sql"], rec["question"], rec["db_id"]
    try:
        tree = sqlglot.parse_one(sql, read="sqlite")
    except Exception:  # noqa: BLE001
        return []
    out = []
    ql = q.lower()
    # sort direction flip (question states an order)
    for o in tree.find_all(exp.Ordered):
        if re.search(DIR_CUES, ql):
            t2 = tree.copy()
            for o2 in t2.find_all(exp.Ordered):
                o2.set("desc", not bool(o2.args.get("desc")))
            out.append(("sort_direction_flip", t2.sql(dialect="sqlite"), None, "every ORDER BY direction reversed"))
        break
    # comparison flip
    for node in tree.find_all(exp.GT, exp.LT, exp.GTE, exp.LTE):
        t2 = tree.copy()
        flip = {exp.GT: exp.LT, exp.LT: exp.GT, exp.GTE: exp.LTE, exp.LTE: exp.GTE}
        for n2 in list(t2.find_all(exp.GT, exp.LT, exp.GTE, exp.LTE)):
            n2.replace(flip[type(n2)](this=n2.this.copy(), expression=n2.expression.copy()))
            break
        out.append(("comparison_flip", t2.sql(dialect="sqlite"), None, "first range comparison operator reversed"))
        break
    # DISTINCT dropped (question asks for different / distinct / unique values)
    if tree.args.get("distinct") is not None and re.search(r"\b(different|distinct|unique)\b", ql):
        t2 = tree.copy()
        t2.set("distinct", None)
        out.append(("distinct_dropped", t2.sql(dialect="sqlite"), None, "SELECT DISTINCT -> SELECT"))
    # COUNT(DISTINCT x) -> COUNT(*)
    for cnt in tree.find_all(exp.Count):
        if isinstance(cnt.this, exp.Distinct):
            t2 = tree.copy()
            for c2 in t2.find_all(exp.Count):
                if isinstance(c2.this, exp.Distinct):
                    c2.replace(exp.Count(this=exp.Star()))
                    break
            out.append(("count_distinct_to_star", t2.sql(dialect="sqlite"), None, "COUNT(DISTINCT col) -> COUNT(*)"))
            break
    # LEFT JOIN -> INNER JOIN on an all/each question
    if re.search(r"\b(all|each|every)\b", ql):
        for j in tree.find_all(exp.Join):
            if (j.args.get("side") or "").upper() == "LEFT":
                t2 = tree.copy()
                for j2 in t2.find_all(exp.Join):
                    if (j2.args.get("side") or "").upper() == "LEFT":
                        j2.set("side", None)
                        break
                out.append(("left_to_inner_all_x", t2.sql(dialect="sqlite"), None, "LEFT JOIN -> INNER JOIN"))
                break
    # literal moved to a sibling TEXT column of the same table
    prof = env.prof(db)
    for eq in tree.find_all(exp.EQ):
        col, lit = eq.this, eq.expression
        if isinstance(col, exp.Column) and isinstance(lit, exp.Literal) and lit.is_string:
            tabs = [t for t in prof.tables if prof.column(t, col.name)]
            if len(tabs) != 1:
                continue
            t = tabs[0]
            sibs = [c for c, cc in prof.tables[t].items() if c.lower() != col.name.lower() and cc.kind == "text"
                    and c not in prof.pk.get(t, [])]
            if sibs:
                sib = sorted(sibs)[0]
                t2 = tree.copy()
                for e2 in t2.find_all(exp.EQ):
                    if isinstance(e2.this, exp.Column) and e2.this.name == col.name and isinstance(e2.expression, exp.Literal):
                        e2.this.set("this", exp.to_identifier(sib))
                        break
                out.append(("literal_to_sibling_column", t2.sql(dialect="sqlite"), None,
                            f"filter literal compared with {t}.{sib} instead of {t}.{col.name}"))
            break
    # truncated question: cut before the first filter literal that appears in the question
    for lit in tree.find_all(exp.Literal):
        v = str(lit.this)
        i = ql.find(v.lower()) if len(v) >= 2 else -1
        if i > 15:
            cut = q[:i].rstrip(" '\"")
            words = cut.split()
            if len(words) >= 4:
                out.append(("truncated_question", None, " ".join(words), f"question cut before the literal {v!r}"))
            break
    return out


# ---------------------------------------------------------------- translation-side plants
def class_of(op):
    for k, ops in TRN_CLASSES.items():
        if op in ops:
            return k
    return None


def gt_text_mutant(mql, prof, db):
    """Untyped numeric comparison on a TEXT column: {"col": {"$gt": "12"}} -> {"col": {"$gt": 12}}."""
    m = copy.deepcopy(mql)
    for st in m["pipeline"]:
        mt = st.get("$match") if isinstance(st, dict) else None
        if not isinstance(mt, dict):
            continue
        for k, v in mt.items():
            if isinstance(v, dict):
                for op in ("$gt", "$gte", "$lt", "$lte"):
                    if isinstance(v.get(op), str) and re.fullmatch(r"-?\d+(\.\d+)?", v[op]):
                        v[op] = float(v[op]) if "." in v[op] else int(v[op])
                        return m
    return None


def plant_translation(rec, mql, env):
    out = []
    for op, mut in MU.mutants(mql):
        k = class_of(op)
        if k:
            out.append((k, op, mut))
    if "like" in [f.lower() for f in (rec.get("features") or [])] or " like " in f" {rec['sql'].lower()} ":
        try:
            v2m = V2.convert(rec["sql"], env.prof(rec["db_id"]))[0]
            if json.dumps(v2m, sort_keys=True) != json.dumps(mql, sort_keys=True):
                out.append(("unicode_regex_fold", "v2_regex_options_is", v2m))
        except Exception:  # noqa: BLE001
            pass
    g = gt_text_mutant(mql, env.prof(rec["db_id"]), rec["db_id"])
    if g is not None:
        out.append(("untyped_gt_text", "numeric_literal_on_text_column", g))
    return out


# ---------------------------------------------------------------- bases
def run01_bases():
    rows = list(csv.DictReader(CASE_CSV.open(encoding="utf-8")))
    ok = {r["item_id"] for r in rows if r["final_source_sql_correct"] == "YES"
          and r["final_translation_type_conformant"] == "YES"}
    items = {it["item_id"]: it for it in rd(RUN01 / "review_package/review_items.jsonl")}
    return [{"id": i, "db_id": items[i]["db_id"], "question": items[i]["question"], "sql": items[i]["sql"],
             "features": items[i]["converter"]["features"], "base_source": "run01_audit_accepted"} for i in sorted(ok)]


def pool_bases(exclude):
    hits = defaultdict(set)
    for h in rd(ROOT / "outputs/source_label_audit_v1/hits.jsonl"):
        hits[h["id"]].add(h["action"])
    sibs = json.loads((ROOT / "outputs/source_label_audit_v1/finding_siblings.json").read_text(encoding="utf-8"))
    bad = {x["finding_id"] for x in sibs} | {s["id"] for x in sibs for s in x["siblings"]}
    pilot = set(json.loads((ROOT / "data/interim/finetune_v2/pilot_freeze.json").read_text(encoding="utf-8"))["pilot"]["ids"])
    rows = [r for r in rd(ROOT / "data/processed/spider_pool_v2/train.jsonl")
            if r["id"] not in hits and r["id"] not in bad and r["id"] not in pilot and r["id"] not in exclude]
    random.Random(SEED).shuffle(rows)
    # prefer bases whose SQL shape can fill the rarer classes (stable within a priority level)
    def prio(r):
        u = r["sql"].upper()
        return -(3 * (" LIKE " in u) + 2 * ("COUNT(DISTINCT" in u.replace(" ", "")) + 2 * ("LEFT JOIN" in u)
                 + (" DISTINCT " in u) + ("ORDER BY" in u) + (" > " in u or " < " in u))
    rows.sort(key=prio)
    return [dict(r, base_source="spider_pool_v2_train_no_detector_hit") for r in rows]


# ---------------------------------------------------------------- build
def build():
    OUT.mkdir(parents=True, exist_ok=True)
    key_p = OUT / "planted_key.jsonl"
    if key_p.exists():
        raise SystemExit(f"refusing to overwrite {key_p}")
    env = Env()
    rng = random.Random(SEED)
    log = {"bases_checked": 0, "base_rejected": Counter(), "src_candidates": Counter(), "src_valid": Counter(),
           "trn_candidates": Counter(), "trn_valid": Counter(), "src_invalid_reasons": Counter(), "trn_invalid_reasons": Counter()}
    src_by, trn_by, clean = defaultdict(list), defaultdict(list), []
    r01 = run01_bases()
    used_base_ids = set()

    def need_src():
        return any(len(src_by[k]) < q for k, q in SRC_QUOTA.items())

    def need_trn():
        return any(len(trn_by[k]) < q for k, q in TRN_QUOTA.items())

    def process(base, allow_clean):
        log["bases_checked"] += 1
        mql = env.convert(base["db_id"], base["sql"])
        if mql is None:
            log["base_rejected"]["not_convertible"] += 1
            return
        base = dict(base, mql=mql)
        src_c = [x for x in plant_source(base, env) if x[0] in SRC_QUOTA and len(src_by[x[0]]) < SRC_QUOTA[x[0]] + 4]
        trn_c = [x for x in plant_translation(base, mql, env) if len(trn_by[x[0]]) < TRN_QUOTA[x[0]] + 3]
        if not allow_clean and not src_c and not trn_c:
            log["base_rejected"]["skipped_no_unmet_class"] += 1
            return
        v = env.verify(base["id"], base["db_id"], base["sql"], mql)
        if v.get("status") != "accepted":
            log["base_rejected"][f"verifier:{v.get('status')}"] += 1
            return
        if allow_clean and len(clean) < N_CLEAN * 2:
            clean.append(base)
        truth = env.sql_result(base["db_id"], base["sql"])
        for cls, psql, pq, desc in src_c:
            if cls not in SRC_QUOTA or len(src_by[cls]) >= SRC_QUOTA[cls] + 4:
                continue
            log["src_candidates"][cls] += 1
            if psql is None:              # truncated question: SQL / MQL unchanged
                src_by[cls].append({"base": base, "class": cls, "dimension": "source", "sql": base["sql"], "mql": mql,
                                    "question": pq, "description": desc})
                log["src_valid"][cls] += 1
                continue
            got = env.sql_result(base["db_id"], psql)
            if got is None or sorted(map(repr, got)) == sorted(map(repr, truth or [])) and \
                    (cls != "sort_direction_flip" or got == truth):
                log["src_invalid_reasons"][f"{cls}:same_result_or_error"] += 1
                continue
            pm = env.convert(base["db_id"], psql)
            if pm is None:
                log["src_invalid_reasons"][f"{cls}:not_convertible"] += 1
                continue
            pv = env.verify(base["id"] + ":p", base["db_id"], psql, pm)
            if pv.get("status") != "accepted":
                log["src_invalid_reasons"][f"{cls}:planted_mql_not_verified"] += 1
                continue
            src_by[cls].append({"base": base, "class": cls, "dimension": "source", "sql": psql, "mql": pm,
                                "question": base["question"], "description": desc})
            log["src_valid"][cls] += 1
        for cls, op, mut in trn_c:
            if len(trn_by[cls]) >= TRN_QUOTA[cls] + 3 or any(t["base"]["id"] == base["id"] for t in trn_by[cls]):
                continue
            log["trn_candidates"][cls] += 1
            mv = env.verify(base["id"] + ":m", base["db_id"], base["sql"], mut)
            if mv.get("status") not in SEMANTIC:
                log["trn_invalid_reasons"][f"{cls}:{mv.get('status')}"] += 1
                continue
            if any(t["base"]["id"] == base["id"] for t in trn_by[cls]):
                continue
            trn_by[cls].append({"base": base, "class": cls, "operator": op, "dimension": "translation",
                                "sql": base["sql"], "mql": mut, "question": base["question"],
                                "verifier_evidence": {"status": mv.get("status"), "detail": str(mv.get("detail"))[:300]}})
            log["trn_valid"][cls] += 1

    t0 = time.time()
    for b in r01:
        process(b, True)
    for b in pool_bases({b["id"] for b in r01}):
        if not need_src() and not need_trn():
            break
        n_s, n_t = sum(len(v) for v in src_by.values()), sum(len(v) for v in trn_by.values())
        if log["bases_checked"] > 700 or (n_s >= N_SRC + 6 and n_t >= N_TRN + 6 and log["bases_checked"] > 450):
            break
        process(b, False)
        if log["bases_checked"] % 50 == 0:
            print(log["bases_checked"], {k: len(v) for k, v in src_by.items()}, {k: len(v) for k, v in trn_by.items()},
                  f"{time.time() - t0:.0f}s", flush=True)

    def pick(by, quota, total):
        chosen = []
        for k, q in quota.items():
            lst = list(by.get(k, []))
            rng.shuffle(lst)
            chosen += lst[:q]
        rest = [x for k in quota for x in by.get(k, []) if x not in chosen]
        rng.shuffle(rest)
        chosen += rest[:max(0, total - len(chosen))]
        return chosen[:total]
    src = pick(src_by, SRC_QUOTA, N_SRC)
    trn = pick(trn_by, TRN_QUOTA, N_TRN)
    rng.shuffle(clean)
    clean = clean[:N_CLEAN]
    key = []
    for kind, lst in (("planted_source", src), ("planted_translation", trn)):
        for x in lst:
            key.append({"kind": kind, "class": x["class"], "operator": x.get("operator"), "dimension": x["dimension"],
                        "base_id": x["base"]["id"], "base_source": x["base"]["base_source"], "db_id": x["base"]["db_id"],
                        "question": x["question"], "sql": x["sql"], "mql": x["mql"], "base_sql": x["base"]["sql"],
                        "base_question": x["base"]["question"], "description": x.get("description"),
                        "verifier_evidence": x.get("verifier_evidence"),
                        "truth": {"source_sql_correct": "NO" if x["dimension"] == "source" else "YES",
                                  "mql_translation_correct": "NO" if x["dimension"] == "translation" else "YES"}})
    for b in clean:
        key.append({"kind": "clean_control", "class": "clean", "dimension": None, "base_id": b["id"],
                    "base_source": b["base_source"], "db_id": b["db_id"], "question": b["question"], "sql": b["sql"],
                    "mql": b["mql"], "truth": {"source_sql_correct": "YES (run01 AI audit; not ground truth)",
                                               "mql_translation_correct": "YES (run01 AI audit + verifier 3.1)"}})
    with key_p.open("w", encoding="utf-8") as f:
        for k in key:
            f.write(json.dumps(k, ensure_ascii=False) + "\n")
    summ = {"written_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "seed": SEED,
            "counts": dict(Counter(k["kind"] for k in key)),
            "source_by_class": dict(Counter(k["class"] for k in key if k["kind"] == "planted_source")),
            "translation_by_class": dict(Counter(k["class"] for k in key if k["kind"] == "planted_translation")),
            "log": {k: (dict(v) if isinstance(v, Counter) else v) for k, v in log.items()},
            "key_sha256": hashlib.sha256(key_p.read_bytes()).hexdigest(),
            "hygiene": "planted items are evaluation controls only; never written to any dataset or corpus"}
    (OUT / "planted_summary.json").write_text(json.dumps(summ, indent=1), encoding="utf-8")
    print(json.dumps({k: summ[k] for k in ("counts", "source_by_class", "translation_by_class")}, indent=1))


if __name__ == "__main__":
    {"build": build}[sys.argv[1]]()
