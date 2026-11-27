"""sql2mql-v2.1 runner (converter sql2mql-v2.1 + verifier 3.1, LOCAL MongoDB 127.0.0.1:27018 only). Plan §E.1/§E.4.

    python -B scripts/finetune_v2/run_pilot_v4.py freeze                 runner freeze (refuses to overwrite)
    python -B scripts/finetune_v2/run_pilot_v4.py regression [--budget S]  300 pilot ids, v2.1 + verifier 3.1
                                                                          -> outputs/sql2mql_v2_1/pilot_regression/v21/
    python -B scripts/finetune_v2/run_pilot_v4.py control [--budget S]   ENVIRONMENT CONTROL: frozen v2 + verifier-3 via
                                                                          run_pilot_v3.process_one (unchanged), same session
                                                                          -> outputs/sql2mql_v2_1/pilot_regression/v2_control/
    python -B scripts/finetune_v2/run_pilot_v4.py compare                §E.4 criteria -> pilot_regression/regression_report.json
    python -B scripts/finetune_v2/run_pilot_v4.py corpus-run [--budget S] convert+verify the corpus-v2 candidate ids
                                                                          -> outputs/sql2mql_v2_1/provisional_corpus_v2/records.jsonl
    python -B scripts/finetune_v2/run_pilot_v4.py gair-run [--budget S]  convert+verify the frozen G-AIR draw
                                                                          -> outputs/sql2mql_v2_1/pilot_v4/records.jsonl

Every record carries converter/verifier versions, the v2.1 code-freeze sha256, the TC-1 manifest sha256 and the
TC-1 contract class of every field the MQL references. Records are appended per example (resumable); a record
from another runner/verifier version stops the run. No frozen file is modified; nothing is written under
outputs/sql2mql_v2/.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.dont_write_bytecode = True
for p in (HERE, ROOT / "scripts/evaluation"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
import run_pilot_v3 as P3  # noqa: E402  (helpers only; hash-pinned by the round-3 / corpus-v1 freezes)

RUNNER_VERSION = "sql2mql-v2.1-runner-v4.0"
CONVERTER = "sql2mql-v2.1"
VERIFIER = "sql2mql-v2-verifier-3.1"
OUT = ROOT / "outputs/sql2mql_v2_1"
FREEZE = ROOT / "data/interim/finetune_v2/pilot_v4_runner_freeze.json"
CODE_FREEZE = ROOT / "data/interim/finetune_v2/v2_1_code_freeze.json"
TC_MANIFEST = ROOT / "data/interim/finetune_v2/type_contract_v1.json"
RUN01 = ROOT / "outputs/sql2mql_v2/pilot_v3/run01/records.jsonl"
CORPUS_IDS = ROOT / "outputs/sql2mql_v2_1/provisional_corpus_v2/candidate_ids.json"
GAIR_DRAW = ROOT / "outputs/sql2mql_v2_1/pilot_v4/gair_draw.json"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def canon(o):
    return json.dumps(o, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


# ------------------------------------------------------------------ integrity
def code_check():
    import subprocess
    r = subprocess.run([sys.executable, "-I", "-B", str(HERE / "freeze_v2_1_code.py"), "check"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"v2.1 code freeze check failed:\n{r.stdout}{r.stderr}")


def freeze_check():
    if not FREEZE.exists():
        raise SystemExit("runner freeze missing: run `freeze` first")
    fz = json.loads(FREEZE.read_text(encoding="utf-8"))
    if fz["runner_sha256"] != sha(__file__):
        raise SystemExit("run_pilot_v4.py changed after its freeze")
    if fz["v2_1_code_freeze_sha256"] != sha(CODE_FREEZE):
        raise SystemExit("v2_1_code_freeze.json changed after the runner freeze")
    if fz["type_contract_v1_sha256"] != sha(TC_MANIFEST):
        raise SystemExit("type_contract_v1.json changed after the runner freeze")
    code_check()
    return fz


def load_modules():
    import v2_common as C
    import sql2mql_v2_1 as N
    import verifier3_1 as V
    import schema_profile as SP
    import eval_kit as EK
    import round3_stats as S
    if N.CONVERTER_VERSION != CONVERTER or V.VERIFIER_VERSION != VERIFIER:
        raise SystemExit("module versions differ from the runner's")
    return {"C": C, "N": N, "V": V, "SP": SP, "EK": EK, "S": S}


# ------------------------------------------------------------------ TC-1 per label
_TC = {}


def tc_manifest():
    if not _TC:
        doc = json.loads(TC_MANIFEST.read_text(encoding="utf-8"))
        _TC["doc"], _TC["sha"] = doc, sha(TC_MANIFEST)
    return _TC["doc"], _TC["sha"]


def contract_fields(mql, db_id):
    """TC-1 class of every manifest field the MQL names (start collection + $lookup.from collections)."""
    doc, _ = tc_manifest()
    colls = [mql["collection"]]
    names = set()

    def walk(o):
        if isinstance(o, dict):
            lk = o.get("$lookup")
            if isinstance(lk, dict) and lk.get("from"):
                colls.append(lk["from"])
            for k, v in o.items():
                if isinstance(k, str) and not k.startswith("$"):
                    names.add(k)
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)
        elif isinstance(o, str):
            names.add(o[1:] if o.startswith("$") else o)
    walk(mql["pipeline"])
    leaf = {n.split(".")[-1] for n in names} | names
    out = {}
    for c in colls:
        e = doc["collections"].get(c)
        if e is None:
            out[c] = "NOT_IN_MANIFEST"
            continue
        for f, s in e["fields"].items():
            if f in leaf:
                out[f"{c}.{f}"] = s["K"]
    return out


# ------------------------------------------------------------------ per record
def process_v21(r, mods, real, profs, extra=None):
    from pymongo.errors import ConnectionFailure
    C, N, V, SP, EK, S = (mods[k] for k in ("C", "N", "V", "SP", "EK", "S"))
    t0 = time.time()
    rec = {"id": r["id"], "db_id": r["db_id"], "hardness": r.get("hardness"), "question": r["question"],
           "sql": r["sql"], "stratum": S.stratum(r["sql"]), "converter": N.CONVERTER_VERSION}
    try:
        prof = profs.get(r["db_id"]) or profs.setdefault(r["db_id"], SP.Profile(r["db_id"]))
        mql, feats = N.convert(r["sql"], prof)
        rec.update(conversion="converted", features=feats, mql=mql)
    except N.Unsupported as e:
        rec.update(conversion="unsupported", code=e.code, detail=e.detail)
    except Exception as e:  # noqa: BLE001
        rec.update(conversion="crash", code=type(e).__name__, detail=str(e)[:300])
    rec["runner"] = RUNNER_VERSION
    rec["sql_profile"] = P3.sql_profile(r["sql"])
    if rec["conversion"] == "unsupported":
        outcome, sub = "conversion_rejected", rec["code"]
    elif rec["conversion"] == "crash":
        outcome, sub = "conversion_error", rec["code"]
    else:
        rec["safety"] = P3.runner_safety(rec["mql"], P3.allowed_collections(r["db_id"], C), EK)
        if not rec["safety"]["ok"]:
            outcome, sub = "safety_rejected", rec["safety"]["eval_kit_validate_mql"]
        else:
            try:
                v = V.verify({"id": r["id"], "db_id": r["db_id"], "sql": r["sql"]}, rec["mql"], real)
            except ConnectionFailure as e:
                raise SystemExit(f"MongoDB connection lost on {r['id']} (no record written): {type(e).__name__}") from e
            except Exception as e:  # noqa: BLE001
                v = {"id": r["id"], "verifier": VERIFIER, "status": "verifier_error",
                     "detail": f"{type(e).__name__}: {str(e)[:300]}", "checks": {}}
                outcome, sub = "verifier_error", type(e).__name__
            else:
                outcome, sub = P3.classify_verification(v)
            rec["verification"] = v
        _, tsha = tc_manifest()
        rec["type_contract"] = {"contract": "TC-1", "manifest_sha256": tsha,
                                "fields": contract_fields(rec["mql"], r["db_id"])}
    rec["outcome"], rec["outcome_detail"] = outcome, sub
    rec["multi_table"] = P3.multi_table_flags(rec["stratum"], rec["sql_profile"], rec.get("mql"))
    rec["v2_1_code_freeze_sha256"] = sha(CODE_FREEZE)
    if extra:
        rec.update(extra)
    rec["runner_seconds"] = round(time.time() - t0, 2)
    return rec


# ------------------------------------------------------------------ resumable store
def load_store(path, ids, runner, verifier):
    if not path.exists():
        return []
    lines = [x for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    rows = []
    for k, x in enumerate(lines):
        try:
            rows.append(json.loads(x))
        except json.JSONDecodeError:
            if k != len(lines) - 1:
                raise SystemExit(f"{path}: unparseable line {k + 1}")
            path.write_text("".join(json.dumps(y, ensure_ascii=False, default=str) + "\n" for y in rows), encoding="utf-8")
    seen = set()
    for x in rows:
        if x["id"] not in ids or x["id"] in seen:
            raise SystemExit(f"{path}: unexpected or duplicate id {x['id']}")
        seen.add(x["id"])
        if x.get("runner") != runner:
            raise SystemExit(f"{path}: record {x['id']} from runner {x.get('runner')}")
        if "verification" in x and x["verification"].get("verifier") != verifier:
            raise SystemExit(f"{path}: record {x['id']} verified by {x['verification'].get('verifier')}")
    return rows


def execute(out, ids, rows, process, runner, verifier, budget=None, real=None, mods=None, extra_of=None):
    out.mkdir(parents=True, exist_ok=True)
    path = out / "records.jsonl"
    done = {x["id"] for x in load_store(path, set(ids), runner, verifier)}
    todo = sorted((rows[i] for i in ids if i not in done), key=lambda r: (r["db_id"], r["id"]))
    profs, t0, state = {}, time.time(), {}
    for k, r in enumerate(todo, 1):
        if budget is not None and time.time() - t0 > budget:
            print(f"budget reached after {k - 1}/{len(todo)}; re-run to resume", flush=True)
            break
        if state.get("db") != r["db_id"]:
            profs.clear()
            state["db"] = r["db_id"]
        rec = process(r, mods, real, profs, extra_of(r) if extra_of else None)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
            f.flush()
        if k % 25 == 0 or k == len(todo):
            print(f"{k}/{len(todo)} ({len(done)} earlier) {time.time() - t0:.0f}s", flush=True)
    recs = load_store(path, set(ids), runner, verifier)
    return recs, {x["id"] for x in recs} == set(ids)


def real_dbs(mods):
    C = mods["C"]
    cl = C.local_client()
    C.check_server(cl)
    return mods["V"].RealDbs(cl)


# ------------------------------------------------------------------ §E.4 comparison
LIKE_OPS = ("$regex", "$options", "regex", "options")


def strip_like(o):
    """MQL with regex texts / options blanked (for 'differs only in regex text and options')."""
    if isinstance(o, dict):
        return {k: ("<RX>" if k in LIKE_OPS else strip_like(v)) for k, v in o.items()}
    if isinstance(o, list):
        return [strip_like(v) for v in o]
    return o


def comparable(rec):
    """Outcome-level view excluding timings and verification.checks.v6_determinism (§E.4 criterion 1)."""
    v = dict(rec.get("verification") or {})
    v.pop("seconds", None)
    v.pop("verifier", None)
    v.pop("base_verifier", None)
    ch = dict(v.get("checks") or {})
    ch.pop("v6_determinism", None)
    ch.pop("v31_fold_traps", None)
    v["checks"] = ch
    return {"conversion": rec.get("conversion"), "code": rec.get("code"), "outcome": rec.get("outcome"),
            "outcome_detail": rec.get("outcome_detail"), "verification": v}


def compare():
    run01 = {r["id"]: r for r in map(json.loads, RUN01.read_text(encoding="utf-8").splitlines())}
    v21 = {r["id"]: r for r in map(json.loads, (OUT / "pilot_regression/v21/records.jsonl").read_text(encoding="utf-8").splitlines())}
    ctl = {r["id"]: r for r in map(json.loads, (OUT / "pilot_regression/v2_control/records.jsonl").read_text(encoding="utf-8").splitlines())}
    assert set(run01) == set(v21) == set(ctl) and len(run01) == 300
    acc = lambda d: {i for i, r in d.items() if r["outcome"] == "accepted"}  # noqa: E731
    like_ids = {i for i, r in run01.items() if "like" in (r.get("features") or [])}
    rep = {"written_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "where": "LOCAL MongoDB 8.0.34 (not Atlas)",
           "n": 300, "accepted": {"run01": len(acc(run01)), "v2_control": len(acc(ctl)), "v21": len(acc(v21))}}
    # environment control: frozen v2 reproduces run01 (MQL identical; outcome identical excl. timings / v6 note)
    rep["control_mql_identical"] = sum(canon(run01[i].get("mql")) == canon(ctl[i].get("mql")) for i in run01)
    rep["control_outcome_identical"] = sum(comparable(run01[i]) == comparable(ctl[i]) for i in run01)
    # C1: non-LIKE labels byte-identical (canonical MQL)
    non_like_acc = sorted(acc(run01) - like_ids)
    c1_bad = [i for i in non_like_acc if canon(run01[i]["mql"]) != canon(v21[i].get("mql"))]
    all_non_like_bad = [i for i in run01 if i not in like_ids and canon(run01[i].get("mql")) != canon(v21[i].get("mql"))]
    rep["C1_non_like_accepted"] = {"n": len(non_like_acc), "mql_identical": len(non_like_acc) - len(c1_bad), "differ": c1_bad,
                                   "all_300_non_like_differ": all_non_like_bad,
                                   "outcome_identical_non_like": sum(comparable(run01[i]) == comparable(v21[i])
                                                                    for i in run01 if i not in like_ids)}
    # C2: LIKE labels differ only in regex text / options; still accepted (results = SQLite truth on real data, both NULL forms)
    like_acc = sorted(acc(run01) & like_ids)
    c2 = []
    for i in like_acc:
        a, b = run01[i]["mql"], v21[i].get("mql")
        ch = (v21[i].get("verification") or {}).get("checks", {})
        c2.append({"id": i, "only_regex_differs": b is not None and canon(strip_like(a)) == canon(strip_like(b)),
                   "mql_changed": canon(a) != canon(b), "v21_outcome": v21[i]["outcome"],
                   "l1_explicit_null": ch.get("l1_explicit_null"), "l1_missing": ch.get("l1_missing"),
                   "real_rows": ch.get("real_rows"), "run01_real_rows": run01[i]["verification"]["checks"].get("real_rows"),
                   "fold_traps": ch.get("v31_fold_traps")})
    rep["C2_like_accepted"] = {"n": len(like_acc), "items": c2,
                               "pass": all(x["only_regex_differs"] and x["mql_changed"] and x["v21_outcome"] == "accepted"
                                           and x["real_rows"] == x["run01_real_rows"] for x in c2)}
    # C3: same accepted / rejected sets
    rep["C3_sets"] = {"accepted_equal": acc(run01) == acc(v21), "v21_accepted": len(acc(v21)),
                      "v21_rejected": 300 - len(acc(v21)),
                      "only_run01": sorted(acc(run01) - acc(v21)), "only_v21": sorted(acc(v21) - acc(run01)),
                      "rejected_like_items": sorted(like_ids - acc(run01))}
    rep["C1_pass"] = not c1_bad and not all_non_like_bad
    rep["C3_pass"] = rep["C3_sets"]["accepted_equal"] and len(acc(v21)) == 213
    rep["control_pass"] = rep["control_mql_identical"] == 300 and acc(ctl) == acc(run01)
    rep["C4_C5"] = "see tests/test_finetune_v2_1_verifier.py (C4) and tests/test_finetune_v2_1_like_engine.py (C5) logs"
    p = OUT / "pilot_regression/regression_report.json"
    p.write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in rep.items() if k not in ("C2_like_accepted",)}, indent=1)[:3000])
    print("C2 pass:", rep["C2_like_accepted"]["pass"], "n =", len(like_acc))
    return rep


# ------------------------------------------------------------------ main
def pilot_rows():
    ids, rows = P3.load_pilot()
    return ids, rows


def pool_rows():
    rows = {}
    for f in ("train.jsonl",):
        for line in (ROOT / "data/processed/spider_pool_v2" / f).read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                rows[r["id"]] = r
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["freeze", "regression", "control", "compare", "corpus-run", "gair-run"])
    ap.add_argument("--budget", type=float)
    a = ap.parse_args(argv)
    if a.cmd == "freeze":
        if FREEZE.exists():
            raise SystemExit(f"refusing to overwrite {FREEZE}")
        code_check()
        doc = {"written_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "runner": RUNNER_VERSION, "runner_sha256": sha(__file__),
               "converter": CONVERTER, "verifier": VERIFIER, "v2_1_code_freeze_sha256": sha(CODE_FREEZE),
               "type_contract_v1_sha256": sha(TC_MANIFEST), "run_pilot_v3_sha256": sha(HERE / "run_pilot_v3.py"),
               "pilot_freeze_sha256": sha(ROOT / "data/interim/finetune_v2/pilot_freeze.json"),
               "spider_pool_v2_train_sha256": sha(ROOT / "data/processed/spider_pool_v2/train.jsonl"),
               "run01_records_sha256": sha(RUN01), "mongo": {"host": "127.0.0.1:27018", "image": "mongo:8.0.34"}}
        FREEZE.write_text(json.dumps(doc, indent=1), encoding="utf-8")
        print(f"wrote {FREEZE.relative_to(ROOT)} sha256={sha(FREEZE)}")
        return
    if a.cmd == "compare":
        compare()
        return
    freeze_check()
    if a.cmd == "control":
        hashes, inputs, mods3 = P3.gate()          # frozen v2 + verifier-3 (19 hashes checked by run_pilot_v3)
        ids, rows = pilot_rows()
        m, _, _ = P3.load_membership()
        real = mods3["verifier3"].RealDbs(mods3["v2_common"].local_client())

        def proc(r, mods, real_, profs, extra):
            return P3.process_one(r, mods3, real_, profs, P3.membership_of(r["id"], m))
        recs, fin = execute(OUT / "pilot_regression/v2_control", ids, rows, proc, P3.RUNNER_VERSION,
                            P3.EXPECTED_VERIFIER, a.budget, real, mods3)
        print("finished" if fin else "partial", len(recs))
        return
    mods = load_modules()
    real = real_dbs(mods)
    if a.cmd == "regression":
        ids, rows = pilot_rows()
        recs, fin = execute(OUT / "pilot_regression/v21", ids, rows, process_v21, RUNNER_VERSION, VERIFIER, a.budget, real, mods)
    elif a.cmd == "corpus-run":
        cand = json.loads(CORPUS_IDS.read_text(encoding="utf-8"))
        rows = pool_rows()
        recs, fin = execute(OUT / "provisional_corpus_v2", cand["ids"], rows, process_v21, RUNNER_VERSION, VERIFIER,
                            a.budget, real, mods)
    else:
        draw = json.loads(GAIR_DRAW.read_text(encoding="utf-8"))
        rows = pool_rows()
        recs, fin = execute(OUT / "pilot_v4", draw["ids"], rows, process_v21, RUNNER_VERSION, VERIFIER, a.budget, real, mods)
    print("finished" if fin else "partial", len(recs), dict(Counter(r["outcome"] for r in recs)))


if __name__ == "__main__":
    main()
