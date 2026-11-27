"""AI review panel v1: item sets, blind batches and packages (plan §E.7, §C.3, §D precision, §E.5 G-AIR).

    python -B scripts/ai_review_panel_v1/build_package.py gair-draw      frozen seeded order of the G-AIR population
    python -B scripts/ai_review_panel_v1/build_package.py gair-convert   v2.1 + verifier 3.1 down the frozen order until
                                                                          200 accepted, corpus-eligible labels
    python -B scripts/ai_review_panel_v1/build_package.py assemble       all panel items -> 6 blind batches (main phase)
    python -B scripts/ai_review_panel_v1/build_package.py stability      seeded 20% re-review subset, new ids/order

Item kinds: planted_source, planted_translation, clean_control (calibration, from the planted key), s9_precision
(detector-hit samples), s10_adjudication (the 10 run01 findings + same-SQL siblings), gair (fresh sample).
Panel ids are opaque; the mapping to origin lives in outputs/ai_review_panel_v1/_orchestrator_only/.
Verifier fixture collections use panel_* databases (never shared with a concurrent runner).
"""
import hashlib
import json
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.dont_write_bytecode = True
for p in (ROOT / "scripts/finetune_v2", ROOT / "scripts/evaluation", HERE):
    sys.path.insert(0, str(p))

PANEL = ROOT / "outputs/ai_review_panel_v1"
ORCH = PANEL / "_orchestrator_only"
GAIR_DIR = PANEL / "gair_sample"
SEED = 20261113
N_BATCHES, N_GAIR, PREC_PER_DET = 6, 200, 40
CAND = ROOT / "outputs/sql2mql_v2_1/provisional_corpus_v2/candidate_ids.json"


def rd(p):
    return [json.loads(x) for x in Path(p).read_text(encoding="utf-8").splitlines() if x.strip()]


def wj(p, rows):
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def no_overwrite(p):
    if Path(p).exists():
        raise SystemExit(f"refusing to overwrite {p}")


def detector_quarantined():
    q = defaultdict(set)
    for h in rd(ROOT / "outputs/source_label_audit_v1/hits.jsonl"):
        if h["action"] == "quarantine":
            q[h["id"]].add(h["subclass"])
    return q


def findings_and_siblings():
    sib = json.loads((ROOT / "outputs/source_label_audit_v1/finding_siblings.json").read_text(encoding="utf-8"))
    return sib, {x["finding_id"] for x in sib} | {s["id"] for x in sib for s in x["siblings"]}


# ------------------------------------------------------------------ G-AIR
def gair_draw():
    out = GAIR_DIR / "gair_draw.json"
    no_overwrite(out)
    cand = json.loads(CAND.read_text(encoding="utf-8"))["ids"]
    q = detector_quarantined()
    _, fs = findings_and_siblings()
    fr = json.loads((ROOT / "data/interim/finetune_v2/pilot_freeze.json").read_text(encoding="utf-8"))
    pilot, dev = set(fr["pilot"]["ids"]), set(fr["dev"]["ids"])
    key = rd(ORCH / "planted_key.jsonl")
    bases = {k["base_id"] for k in key}
    pre = set(json.loads(CAND.read_text(encoding="utf-8"))["pre_excluded"])
    pop = [i for i in cand if i not in q and i not in fs and i not in pilot and i not in dev and i not in bases and i not in pre]
    order = list(pop)
    random.Random(f"{SEED}:gair").shuffle(order)
    doc = {"written_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "seed": f"{SEED}:gair", "n_target": N_GAIR,
           "population_rule": "corpus-v2 conversion candidates (pool v2 train AND semantic-audit NO_ISSUE_DETECTED) minus "
                              "any detector quarantine hit, the 10 run01 findings and their siblings, the 300 pilot ids, "
                              "the 500 DEV ids, every planted-defect base / clean control, and the pilot cross-review "
                              "pre-exclusions. These records are in spider-pool-v3 whatever the panel decides elsewhere.",
           "selection_rule": "walk the frozen order; take each record that sql2mql-v2.1 + verifier 3.1 ACCEPT and that is "
                             "not multi_table_3plus and has no evaluation-set question/SQL hash; stop at 200",
           "population_n": len(pop), "order_sha256": hashlib.sha256("\n".join(order).encode()).hexdigest(),
           "planted_key_sha256": sha(ORCH / "planted_key.jsonl"), "order": order}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=0), encoding="utf-8")
    print(len(pop), doc["order_sha256"])


def gair_convert():
    import v2_common as C
    import schema_profile as SP
    import verifier3 as V3
    import verifier3_1 as V31
    import run_pilot_v4 as R4
    import build_corpus_v2 as BC
    V3.FIXTURE_DB, V3.REORDER_DB, V31.FIXTURE_DB = "panel_gair_v3_fixtures", "panel_gair_v3_reorder", "panel_gair_v31_fixtures"
    draw = json.loads((GAIR_DIR / "gair_draw.json").read_text(encoding="utf-8"))
    rows = {r["id"]: r for r in rd(ROOT / "data/processed/spider_pool_v2/train.jsonl")}
    outp = GAIR_DIR / "gair_conversions.jsonl"
    done = {r["id"]: r for r in rd(outp)} if outp.exists() else {}
    mods = R4.load_modules()
    cl = C.local_client()
    C.check_server(cl)
    real = V31.RealDbs(cl)
    ev = BC.eval_hashes()
    profs, taken = {}, [i for i, r in done.items() if r["gair_selected"]]
    with open(outp, "a", encoding="utf-8") as f:
        for i in draw["order"]:
            if len(taken) >= N_GAIR:
                break
            if i in done:
                continue
            rec = R4.process_v21(rows[i], mods, real, profs)
            leak = BC.leakage_reasons(rows[i], ev)
            rec["leakage"] = leak
            rec["gair_selected"] = rec["outcome"] == "accepted" and not rec["multi_table"]["multi_table_3plus"] and not leak
            if rec["gair_selected"]:
                taken.append(i)
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
            f.flush()
            if len(profs) > 6:
                profs.clear()
    print("selected", len(taken))


# ------------------------------------------------------------------ previews
def _preview_sql(db, sql, n=30):
    import v2_common as C
    con = C.sqlite_ro(C.sqlite_path(db))
    try:
        cur = con.execute(sql)
        cols = [c[0] for c in cur.description] if cur.description else []
        rows = [list(r) for r in cur.fetchall()]
        return {"columns": cols, "row_count": len(rows), "rows": rows[:n]}
    except Exception as e:  # noqa: BLE001
        return {"error": repr(e)[:200]}
    finally:
        con.close()


def _preview_mql(mdb, mql, n=30):
    if not mql:
        return None
    try:
        docs = list(mdb[mql["collection"]].aggregate(mql["pipeline"]))
        return {"row_count": len(docs), "rows": [{k: v for k, v in d.items() if k != "_id"} for d in docs[:n]]}
    except Exception as e:  # noqa: BLE001
        return {"error": repr(e)[:200]}


def _schema(db, sql):
    import v2_common as C
    import sqlglot
    from sqlglot import exp
    con = C.sqlite_ro(C.sqlite_path(db))
    try:
        allt = [r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")]
        try:
            used = {t.name.lower() for t in sqlglot.parse_one(sql, read="sqlite").find_all(exp.Table)}
        except Exception:  # noqa: BLE001
            used = set(x.lower() for x in allt)
        out = []
        for t in allt:
            if t.lower() in used:
                ddl = con.execute("SELECT sql FROM sqlite_master WHERE name=?", (t,)).fetchone()[0]
                out.append({"table": t, "rows": con.execute(f'SELECT count(*) FROM "{t}"').fetchone()[0], "ddl": ddl})
        return {"tables_used": out, "all_tables": allt}
    finally:
        con.close()


# ------------------------------------------------------------------ assemble
def assemble():
    import sql2mql_v2_1 as N
    import schema_profile as SP
    import v2_common as C
    pkg = PANEL / "main/package"
    no_overwrite(pkg / "batch_01.jsonl")
    rng = random.Random(f"{SEED}:assemble")
    items = []
    # calibration (planted + clean controls)
    for k in rd(ORCH / "planted_key.jsonl"):
        items.append({"origin": {"kind": k["kind"], "class": k["class"], "operator": k.get("operator"),
                                 "base_id": k["base_id"], "truth": k["truth"]},
                      "db_id": k["db_id"], "question": k["question"], "sql": k["sql"], "mql": k["mql"], "group": k["base_id"]})
    # S10: the 10 findings + siblings (blind re-review)
    sib, fs = findings_and_siblings()
    src = {r["id"]: r for r in rd(ROOT / "data/processed/spider_sql_v1/train.jsonl")}
    pool = {r["id"]: r for r in rd(ROOT / "data/processed/spider_pool_v2/train.jsonl")}
    profs = {}

    def conv(db, sql):
        try:
            p = profs.get(db) or profs.setdefault(db, SP.Profile(db))
            return N.convert(sql, p)[0]
        except Exception:  # noqa: BLE001
            return None
    s10_ids = []
    for x in sib:
        s10_ids.append((x["finding_id"], "finding", x["review_id"]))
        for s in x["siblings"]:
            s10_ids.append((s["id"], "sibling", x["review_id"]))
    for rid, role, rev in s10_ids:
        r = src[rid]
        items.append({"origin": {"kind": "s10_adjudication", "record_id": rid, "role": role, "run01_review_id": rev},
                      "db_id": r["db_id"], "question": r["question"], "sql": r["sql"], "mql": conv(r["db_id"], r["sql"]),
                      "group": "s10:" + rev})
    # S9: precision samples, min(40, hits) per detector, seeded, excluding S10 records
    by_det = defaultdict(list)
    for h in rd(ROOT / "outputs/source_label_audit_v1/hits.jsonl"):
        if h["action"] == "quarantine" and h["id"] not in fs:
            by_det[h["detector"]].append(h)
    allrows = dict(pool)
    for r in rd(ROOT / "data/processed/spider_pool_v2/val_internal.jsonl"):
        allrows[r["id"]] = r
    s9_seen = {}
    for det in sorted(by_det):
        ids = sorted({h["id"] for h in by_det[det]})
        random.Random(f"{SEED}:precision:{det}").shuffle(ids)
        for i in ids[:PREC_PER_DET]:
            if i in s9_seen:
                s9_seen[i]["origin"]["detectors"].append(det)
                continue
            r = allrows[i]
            it = {"origin": {"kind": "s9_precision", "record_id": i, "detectors": [det],
                             "hits": [h["subclass"] for h in by_det[det] if h["id"] == i]},
                  "db_id": r["db_id"], "question": r["question"], "sql": r["sql"], "mql": conv(r["db_id"], r["sql"]),
                  "group": "rec:" + i}
            s9_seen[i] = it
            items.append(it)
    # G-AIR
    for r in rd(GAIR_DIR / "gair_conversions.jsonl"):
        if r["gair_selected"]:
            items.append({"origin": {"kind": "gair", "record_id": r["id"]}, "db_id": r["db_id"], "question": r["question"],
                          "sql": r["sql"], "mql": r["mql"], "group": "rec:" + r["id"]})
    # previews + opaque ids
    cl = C.local_client()
    mdb = cl[C.REAL_DB]
    used_ids = set()
    for it in items:
        while True:
            pid = "P" + "".join(rng.choice("0123456789abcdef") for _ in range(8))
            if pid not in used_ids:
                used_ids.add(pid)
                break
        it["panel_id"] = pid
        it["schema"] = _schema(it["db_id"], it["sql"])
        it["sqlite_result"] = _preview_sql(it["db_id"], it["sql"])
        it["mongo_result"] = _preview_mql(mdb, it["mql"])
    # blind batches: seeded shuffle; same group (base / record) never twice in a batch
    order = list(items)
    rng.shuffle(order)
    batches = [[] for _ in range(N_BATCHES)]
    groups = [set() for _ in range(N_BATCHES)]
    for it in order:
        cands = sorted(range(N_BATCHES), key=lambda b: (len(batches[b]), b))
        b = next((b for b in cands if it["group"] not in groups[b]), None)
        if b is None:
            raise SystemExit(f"cannot place group {it['group']}")
        batches[b].append(it)
        groups[b].add(it["group"])
    key = []
    for bi, bt in enumerate(batches, 1):
        rows = [{k: it[k] for k in ("panel_id", "db_id", "question", "sql", "mql", "schema", "sqlite_result", "mongo_result")}
                for it in bt]
        wj(pkg / f"batch_{bi:02d}.jsonl", rows)
        for it in bt:
            key.append({"panel_id": it["panel_id"], "batch": f"batch_{bi:02d}", "origin": it["origin"], "db_id": it["db_id"],
                        "sql": it["sql"], "question": it["question"], "mql": it["mql"]})
    wj(ORCH / "panel_key_main.jsonl", key)
    summ = {"written_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "n_items": len(items),
            "by_kind": dict(Counter(it["origin"]["kind"] for it in items)),
            "per_batch": [len(b) for b in batches],
            "planted_share": round(sum(it["origin"]["kind"].startswith("planted") for it in items) / len(items), 4),
            "batch_sha256": {f"batch_{i:02d}.jsonl": sha(pkg / f"batch_{i:02d}.jsonl") for i in range(1, N_BATCHES + 1)},
            "key_sha256": sha(ORCH / "panel_key_main.jsonl"),
            "s9_per_detector": {d: min(PREC_PER_DET, len({h['id'] for h in v})) for d, v in by_det.items()},
            "s9_unique_records": len(s9_seen), "no_mql_items": sum(it["mql"] is None for it in items)}
    (PANEL / "main/package_manifest.json").write_text(json.dumps(summ, indent=1), encoding="utf-8")
    print(json.dumps(summ, indent=1))


def stability():
    pkg = PANEL / "stability/package"
    no_overwrite(pkg / "batch_01.jsonl")
    key = rd(ORCH / "panel_key_main.jsonl")
    items = {}
    for b in sorted({k["batch"] for k in key}):
        for it in rd(PANEL / "main/package" / f"{b}.jsonl"):
            items[it["panel_id"]] = it
    rng = random.Random(f"{SEED}:stability")
    pids = sorted(items)
    rng.shuffle(pids)
    sub = pids[:round(0.2 * len(pids))]
    kmap = {k["panel_id"]: k for k in key}
    used = set(items)
    out_key, batches = [], [[], []]
    groups = [set(), set()]
    for pid in sub:
        while True:
            nid = "S" + "".join(rng.choice("0123456789abcdef") for _ in range(8))
            if nid not in used:
                used.add(nid)
                break
        it = dict(items[pid], panel_id=nid)
        g = kmap[pid]["origin"].get("base_id") or kmap[pid]["origin"].get("record_id")
        b = min((0, 1), key=lambda i: (g in groups[i], len(batches[i])))
        batches[b].append(it)
        groups[b].add(g)
        out_key.append({"panel_id": nid, "main_panel_id": pid, "batch": f"batch_{b + 1:02d}", "origin": kmap[pid]["origin"]})
    for i, bt in enumerate(batches, 1):
        rng.shuffle(bt)
        wj(pkg / f"batch_{i:02d}.jsonl", bt)
    wj(ORCH / "panel_key_stability.jsonl", out_key)
    print(len(sub), [len(b) for b in batches])


if __name__ == "__main__":
    {"gair-draw": gair_draw, "gair-convert": gair_convert, "assemble": assemble, "stability": stability}[sys.argv[1]]()
