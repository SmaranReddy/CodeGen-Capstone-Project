"""MQL-0014 Kaggle runtime core (generate only; no database, no gold labels, no credentials).

Two retrieval arms, R0 (MQL-0010 D0 = BM25 hybrid) and R3 (MQL-0011 RRF(BM25, dense) + Jaccard), each through the
MQL-0010 D0V recipe: first pass, static schema validator, at most one schema-feedback retry. Prompts are FROZEN in the
package (R3 needs the local Qdrant index, so plans are not re-derived here); every prompt is re-hashed and checked
against its plan.

  pass            generates                                                   batch
  first:shared    items whose R0 and R3 prompts are byte-identical (one output   8
                  serves both arms)
  first:R0only    the remaining items, R0 prompt                                 8
  first:R3only    the remaining items, R3 prompt (same ids, same order)          8
  retry           every DISTINCT retry prompt needed by R0V or R3V              1

Arms R0, R3 (first answers) and R0V, R3V (after recovery) are assembled at export and checked by verify_exports.
Reused UNCHANGED (byte copies in src/): mql0008_core (Checkpoint, run_generation, oom_safe_generator,
make_qwen_generator, verify_package), mql0009_core (normalize_prediction, gpu_runtime_info, check_precision,
check_snapshot_revision), and the registered MQL-0010 planner / prompt_v2 / recovery / validator.
"""
import json
import os
import shutil
import sys
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import mql0008_core as M8      # noqa: E402
import mql0009_core as K9      # noqa: E402
import planner as PL           # noqa: E402
import prompt_v2 as PV         # noqa: E402
import recovery as RC          # noqa: E402
import validator as V          # noqa: E402

EXPERIMENT = "MQL-0014"
PACKAGE = "nl2mql_mql0014_r3_recheck"
STATUS_FILE = "MQL0014_STATUS.json"
MANIFEST_FILE = "MQL0014_PREDICTIONS_MANIFEST.json"
MODEL_ID = "Qwen/Qwen2.5-Coder-3B-Instruct"
MODEL_REVISION = "488639f1ff808d1d3d0ba301aef8c11461451ec5"
PLAN_ARMS = ("R0", "R3")
ARMS = ("R0", "R0V", "R3", "R3V")
RETRY_ARMS = {"R0V": "R0", "R3V": "R3"}
FIRST_PASSES = ("first:shared", "first:R0only", "first:R3only")
PASSES = FIRST_PASSES + ("retry",)
GOLD_KEYS = {"sql", "gold_sql", "query", "gold", "gold_rows", "gold_columns", "answer", "label", "reference", "mql_gold"}

need = M8.need
read_jsonl, write_jsonl, write_json = M8.read_jsonl, M8.write_jsonl, M8.write_json
sha256_file, sha256_text, canon, fingerprint, now = M8.sha256_file, M8.sha256_text, M8.canon, M8.fingerprint, M8.now


# ------------------------------------------------------------------ package loading (fail closed)
def load_package(pkg):
    pkg = Path(pkg)
    man = M8.verify_package(pkg)
    cfg = json.loads((pkg / "configs/experiment.json").read_text(encoding="utf-8"))
    need(cfg["experiment_id"] == EXPERIMENT and man["package"] == PACKAGE, "not the MQL-0014 package")
    reg_path = pkg / "registration/mql0014_registration.json"
    need(sha256_file(reg_path) == cfg["registration_sha256"], "registration differs from the config pin")
    reg = json.loads(reg_path.read_text(encoding="utf-8"))
    need(reg["experiment"] == EXPERIMENT and reg["arms"] == list(ARMS), "registration arms")
    evals = read_jsonl(pkg / "data/eval/eval_inputs.jsonl")
    need(all(set(e) == {"id", "db_id", "question"} for e in evals), "evaluation inputs must hold only id, db_id, question")
    ids = [e["id"] for e in evals]
    need(len(ids) == len(set(ids)) == cfg["evaluation"]["n_items"] == reg["evaluation"]["n_items"], "evaluation size")
    need(sha256_text("\n".join(ids)) == cfg["evaluation"]["ids_sha256"] == reg["evaluation"]["ids_sha256"], "evaluation ids")
    plans, prompts = {}, {}
    for arm in PLAN_ARMS:
        plans[arm] = read_jsonl(pkg / f"data/plans/plan_{arm}.jsonl")
        prompts[arm] = read_jsonl(pkg / f"data/prompts/prompts_{arm}.jsonl")
        need(sha256_text(canon(plans[arm])) == cfg["plans_sha256"][arm], f"{arm}: plan hash")
        need([r["id"] for r in plans[arm]] == ids and [p["id"] for p in prompts[arm]] == ids, f"{arm}: plan/prompt ids")
        for r, p in zip(plans[arm], prompts[arm]):
            sha = PL.prompt_sha(p["system"], p["messages"])
            need(sha == p["prompt_sha256"] == r["prompt_sha256"], f"{arm}/{r['id']}: prompt hash != plan")
            need(PL.sha256_text(p["messages"][0]["content"]) == r["user_prompt_sha256"], f"{arm}/{r['id']}: user text")
            need(len(r["selected_ids"]) <= cfg["context"]["k"], f"{arm}/{r['id']}: more than K examples")
    shared = [i for i, a, b in zip(ids, prompts["R0"], prompts["R3"]) if a["prompt_sha256"] == b["prompt_sha256"]]
    need(shared == cfg["partition"]["shared_ids"], "shared partition differs from the config")
    for f in (pkg / "data/eval/eval_inputs.jsonl", pkg / "data/prompts/prompts_R0.jsonl", pkg / "data/prompts/prompts_R3.jsonl"):
        need(not any(GOLD_KEYS & set(r) for r in read_jsonl(f)), f"{f.name}: gold-like field present")
    atlas = json.loads((pkg / "data/context/atlas_collections.json").read_text(encoding="utf-8"))
    return {"pkg": pkg, "manifest": man, "manifest_sha256": sha256_file(pkg / "package_manifest.json"), "cfg": cfg,
            "cfg_sha256": sha256_file(pkg / "configs/experiment.json"), "reg": reg, "evals": evals, "ids": ids,
            "plans": plans, "prompts": prompts, "shared": shared, "atlas": atlas}


def pass_items(P, pass_id):
    """-> [prompt records] generated by a first pass, in evaluation order."""
    sset = set(P["shared"])
    if pass_id == "first:shared":
        return [p for p in P["prompts"]["R0"] if p["id"] in sset]
    arm = "R0" if pass_id == "first:R0only" else "R3"
    return [p for p in P["prompts"][arm] if p["id"] not in sset]


# ------------------------------------------------------------------ provenance
def pass_batch_size(cfg, pass_id):
    return cfg["model"]["batch_size_first_pass"] if pass_id in FIRST_PASSES else cfg["model"]["batch_size_retry"]


def pass_fingerprint(P, pass_id, payload_sha):
    m = P["cfg"]["model"]
    return fingerprint({"experiment": EXPERIMENT, "pass": pass_id, "model": m["id"], "revision": m["revision"], "dtype": m["dtype"],
                        "decoding": m["decoding"], "max_new_tokens": m["max_new_tokens"], "seed": m["seed"],
                        "batch_size": pass_batch_size(P["cfg"], pass_id), "payload_sha256": payload_sha,
                        "package_manifest_sha256": P["manifest_sha256"], "config_sha256": P["cfg_sha256"]})


def identity(P, pass_id, precision, runtime):
    m = P["cfg"]["model"]
    return {"experiment": EXPERIMENT, "pass": pass_id, "model_id": m["id"], "model_revision": m["revision"],
            "precision": precision, "runtime": runtime, "config_sha256": P["cfg_sha256"],
            "generation_config": {"dtype": m["dtype"], "decoding": m["decoding"], "do_sample": False,
                                  "max_new_tokens": m["max_new_tokens"], "batch_size": pass_batch_size(P["cfg"], pass_id),
                                  "seed": m["seed"]}}


def pass_dir(out_dir, pass_id):
    return Path(out_dir) / "passes" / pass_id.replace(":", "_")


def _write_pass_config(d, info):
    old = d / "pass_config.json"
    if old.exists():
        prev = json.loads(old.read_text(encoding="utf-8"))
        diff = {k: (prev.get(k), v) for k, v in info.items() if k != "sessions" and prev.get(k) != v}
        need(not diff, f"{d.name}: earlier pass_config differs: {diff}; refusing to mix runs")
        info["sessions"] = prev.get("sessions", []) + info.get("sessions", [])
    write_json(old, info)


def _env_key(runtime):
    return {k: runtime.get(k) for k in ("gpu", "compute_capability", "bf16_native", "torch", "transformers")}


def _ck_records(path):
    if not Path(path).exists():
        return {}
    return {r["id"]: r for r in read_jsonl(path) if r.get("id")}


# ------------------------------------------------------------------ generation
def run_first_pass(P, pass_id, gen, precision, runtime, out_dir, max_batches=None, log=print):
    items = pass_items(P, pass_id)
    d = pass_dir(out_dir, pass_id)
    payload = sha256_text("".join(f"{p['id']}:{p['prompt_sha256']}\n" for p in items))
    fp = pass_fingerprint(P, pass_id, payload)
    _write_pass_config(d, {"pass": pass_id, "fingerprint": fp, "payload_sha256": payload, "env_key": _env_key(runtime),
                           "n_expected": len(items), "sessions": [{"at": now()}]})
    ck = M8.Checkpoint(d / "checkpoint.jsonl", [p["id"] for p in items], fp, M8.is_generation_retry)
    ident = {p["id"]: identity(P, pass_id, precision, runtime) for p in items}
    log(f"{pass_id}: {len(ck.todo())} of {len(items)} to generate")
    return M8.run_generation(items, gen, ck, P["cfg"]["model"]["label"], pass_id, pass_batch_size(P["cfg"], pass_id),
                             max_batches=max_batches, identity=ident,
                             progress=lambda n, t: log(f"  {pass_id} {n}/{t}") if n % 80 < 8 else None)


def first_answers(P, out_dir, arm):
    """{id: first-pass record} for arm R0 / R3 (shared items come from first:shared)."""
    sh = _ck_records(pass_dir(out_dir, "first:shared") / "checkpoint.jsonl")
    own = _ck_records(pass_dir(out_dir, f"first:{arm}only") / "checkpoint.jsonl")
    sset = set(P["shared"])
    out = {i: (sh.get(i) if i in sset else own.get(i)) for i in P["ids"]}
    need(all((out[i] or {}).get("status") == "ok" for i in P["ids"]), f"first answers for {arm} incomplete")
    return out


def retry_request(system, user, first_text, validator):
    """First half of recovery.recover (schema mode): -> None (no retry) or (messages, feedback)."""
    obj, issues = RC.assess(first_text, validator)
    if obj is not None and not issues:
        return None
    fb = RC.FORMAT_FEEDBACK if obj is None else V.format_feedback(issues)
    return PV.recovery_messages(user, first_text, fb), fb


def retry_requests(P, out_dir):
    """-> (per-arm {id: prompt_sha or None}, distinct retry prompt records keyed by a content id)."""
    vals, per_arm, distinct = {}, {}, {}
    for arm in PLAN_ARMS:
        first = first_answers(P, out_dir, arm)
        pr = {p["id"]: p for p in P["prompts"][arm]}
        per_arm[arm] = {}
        for i in P["ids"]:
            s, p = first[i], pr[i]
            v = vals.setdefault(s["db_id"], V.SchemaValidator(s["db_id"], P["atlas"]))
            rq = retry_request(p["system"], p["messages"][0]["content"], s["raw_text"], v)
            if rq is None:
                per_arm[arm][i] = None
                continue
            sha = PL.prompt_sha(p["system"], rq[0])
            per_arm[arm][i] = sha
            distinct.setdefault(sha, {"id": "rq_" + sha[:24], "db_id": s["db_id"], "system": p["system"], "messages": rq[0],
                                      "prompt_sha256": sha})
    recs = sorted(distinct.values(), key=lambda r: r["id"])
    need(len({r["id"] for r in recs}) == len(recs), "retry content-id collision")
    return per_arm, recs


def run_retry_pass(P, gen, precision, runtime, out_dir, max_batches=None, log=print):
    per_arm, reqs = retry_requests(P, out_dir)
    d = pass_dir(out_dir, "retry")
    payload = sha256_text("".join(f"{r['id']}:{r['prompt_sha256']}\n" for r in reqs))
    fp = pass_fingerprint(P, "retry", payload)
    _write_pass_config(d, {"pass": "retry", "fingerprint": fp, "payload_sha256": payload, "env_key": _env_key(runtime),
                           "retry_ids": [r["id"] for r in reqs],
                           "retried_items": {a: sorted(i for i, s in per_arm[a].items() if s) for a in PLAN_ARMS},
                           "sessions": [{"at": now()}]})
    ck = M8.Checkpoint(d / "checkpoint.jsonl", [r["id"] for r in reqs], fp, M8.is_generation_retry)
    ident = {r["id"]: identity(P, "retry", precision, runtime) for r in reqs}
    log(f"retry: {len(ck.todo())} of {len(reqs)} distinct retry prompts to generate "
        f"(R0 {sum(bool(s) for s in per_arm['R0'].values())}, R3 {sum(bool(s) for s in per_arm['R3'].values())} items)")
    return M8.run_generation(reqs, gen, ck, P["cfg"]["model"]["label"], "retry", 1, max_batches=max_batches, identity=ident)


# ------------------------------------------------------------------ status, assembly, export
def pass_status(P, out_dir):
    st = {}
    for pid in FIRST_PASSES:
        items = pass_items(P, pid)
        recs = _ck_records(pass_dir(out_dir, pid) / "checkpoint.jsonl")
        st[pid] = {"n_expected": len(items), "n_ok": sum(recs.get(p["id"], {}).get("status") == "ok" for p in items)}
    cfgp = pass_dir(out_dir, "retry") / "pass_config.json"
    if cfgp.exists():
        want = json.loads(cfgp.read_text(encoding="utf-8"))["retry_ids"]
        recs = _ck_records(pass_dir(out_dir, "retry") / "checkpoint.jsonl")
        st["retry"] = {"n_expected": len(want), "n_ok": sum(recs.get(i, {}).get("status") == "ok" for i in want)}
    else:
        st["retry"] = {"n_expected": None, "n_ok": 0}
    for v in st.values():
        v["status"] = ("complete" if v["n_expected"] is not None and v["n_ok"] == v["n_expected"]
                       else "absent" if not v["n_ok"] and v["n_expected"] != 0 else "partial")
    return st


def check_dependencies(P, out_dir, pass_id):
    if pass_id == "retry":
        st = pass_status(P, out_dir)
        missing = [p for p in FIRST_PASSES if st[p]["status"] != "complete"]
        need(not missing, f"retry needs completed {missing}")


_KEEP = ("id", "db_id", "status", "raw_text", "prompt_sha256", "fingerprint", "pass", "model_id", "model_revision", "precision",
         "runtime", "config_sha256", "generation_config", "input_tokens", "output_tokens", "latency_s", "finish_reason",
         "oom_recovery", "effective_batch_size")


def assemble(P, out_dir):
    ids = P["ids"]
    cfgp = pass_dir(out_dir, "retry") / "pass_config.json"
    need(cfgp.exists(), "retry pass not run")
    rcfg = json.loads(cfgp.read_text(encoding="utf-8"))
    rr_by_id = _ck_records(pass_dir(out_dir, "retry") / "checkpoint.jsonl")
    need(all(rr_by_id.get(i, {}).get("status") == "ok" for i in rcfg["retry_ids"]), "retry incomplete")
    per_arm, reqs = retry_requests(P, out_dir)
    need([r["id"] for r in reqs] == rcfg["retry_ids"], "retry prompts differ from the retry pass record")
    rr = {r["prompt_sha256"]: rr_by_id[r["id"]] for r in reqs}
    arms, vals = {}, {}
    sset = set(P["shared"])
    for arm in PLAN_ARMS:
        first = first_answers(P, out_dir, arm)
        plan = {r["id"]: r for r in P["plans"][arm]}
        pr = {p["id"]: p for p in P["prompts"][arm]}
        base = []
        for i in ids:
            r = first[i]
            rec = {**{k: r[k] for k in _KEEP if k in r}, "arm": arm, "first_raw_text": r["raw_text"],
                   "source": "generated:shared" if i in sset else f"generated:{arm}only", "recovery": None,
                   "retrieval": {k: plan[i][k] for k in ("selected_ids", "selected_ranks", "selected_scores", "selected_db_ids",
                                                         "excluded_by_rule", "dropped_budget", "context_tokens",
                                                         "empty_context", "flags")},
                   "experiment": EXPERIMENT, "prediction_id": f"{EXPERIMENT}:{arm}:{i}"}
            need(rec["prompt_sha256"] == plan[i]["prompt_sha256"], f"{arm}/{i}: generated prompt != plan prompt")
            base.append(rec)
        arms[arm] = base
        varm = arm + "V"
        out = []
        for rec in base:
            i, p = rec["id"], pr[rec["id"]]
            v = vals.setdefault(rec["db_id"], V.SchemaValidator(rec["db_id"], P["atlas"]))
            sha = per_arm[arm][i]

            def cached(system, msgs, _i=i, _sha=sha, _p=p):
                need(_sha is not None and PL.prompt_sha(system, msgs) == _sha and system == _p["system"],
                     f"{varm}/{_i}: retry prompt mismatch")
                return rr[_sha]["raw_text"]
            final, trace = RC.recover(p["system"], p["messages"][0]["content"], rec["first_raw_text"], v, cached,
                                      feedback_mode="schema")
            need(bool(trace["retried"]) == (sha is not None), f"{varm}/{i}: retry decision differs from the pass record")
            nr = {**rec, "arm": varm, "raw_text": final, "source": f"reused:{arm}", "recovery": trace,
                  "prediction_id": f"{EXPERIMENT}:{varm}:{i}"}
            if sha is not None:
                g = rr[sha]
                nr["retry_generation"] = {"prompt_sha256": sha, **{k: g.get(k) for k in ("fingerprint", "input_tokens",
                                          "output_tokens", "latency_s", "finish_reason")}}
            out.append(nr)
        arms[varm] = out
    for a in arms:
        arms[a] = [{**r, **K9.normalize_prediction(r["raw_text"], r["status"])} for r in arms[a]]
    return arms


def verify_exports(recs, plans, ids, shared):
    need(set(recs) == set(ARMS), f"arms {sorted(recs)} != {list(ARMS)}")
    by = {}
    for a in ARMS:
        got = [r["id"] for r in recs[a]]
        need(got == list(ids), f"{a}: ids/order differ from the registered evaluation")
        need(all(r["status"] == "ok" for r in recs[a]), f"{a}: generation failures present")
        need(all(r["arm"] == a for r in recs[a]), f"{a}: arm label")
        by[a] = {r["id"]: r for r in recs[a]}
    for a in PLAN_ARMS:
        ph = {r["id"]: r["prompt_sha256"] for r in plans[a]}
        need(all(by[a][i]["prompt_sha256"] == ph[i] for i in ids), f"{a}: prompt hash differs from its plan")
    for i in shared:
        need(by["R0"][i]["first_raw_text"] == by["R3"][i]["first_raw_text"], f"{i}: shared item answers differ")
    for va, src in RETRY_ARMS.items():
        for i in ids:
            r, s = by[va][i], by[src][i]
            need(r["first_raw_text"] == s["first_raw_text"], f"{va}/{i}: first answer differs from {src}")
            t = r["recovery"]
            need(t is not None and t.get("feedback_mode") == "schema", f"{va}/{i}: recovery trace")
            if not t["retried"] or t["kept"] != "retry":
                need(r["raw_text"] == r["first_raw_text"], f"{va}/{i}: changed without a kept retry")
    for i in shared:
        need(by["R0V"][i]["raw_text"] == by["R3V"][i]["raw_text"], f"{i}: shared item final answers differ")
    return {"ok": True, "n": len(ids), "n_shared": len(shared),
            "retried": {a: sum(bool(by[a][i]["recovery"]["retried"]) for i in ids) for a in RETRY_ARMS},
            "kept_retry": {a: sum(by[a][i]["recovery"]["kept"] == "retry" for i in ids) for a in RETRY_ARMS}}


def export(P, out_dir, output_zip, extra_status=None):
    out_dir = Path(out_dir)
    st = pass_status(P, out_dir)
    keys = set()
    for pid in PASSES:
        c = pass_dir(out_dir, pid) / "pass_config.json"
        if c.exists():
            keys.add(canon(json.loads(c.read_text(encoding="utf-8")).get("env_key")))
    env_ok = len(keys) <= 1
    problems = [] if env_ok else [f"passes ran in different environments: {sorted(keys)}"]
    complete = all(v["status"] == "complete" for v in st.values()) and env_ok
    ver, per_arm = None, {}
    if complete:
        arms = assemble(P, out_dir)
        ver = verify_exports(arms, P["plans"], P["ids"], P["shared"])
        for a, recs in arms.items():
            write_jsonl(out_dir / "predictions" / a / "predictions.jsonl", recs)
            per_arm[a] = {"n": len(recs), "predictions_sha256": sha256_file(out_dir / "predictions" / a / "predictions.jsonl")}
    status = "complete" if complete else "partial"
    man = {"experiment": EXPERIMENT, "written_at": now(), "status": status, "passes": st, "per_arm": per_arm,
           "expected_arms": list(ARMS), "verify_exports": ver, "package_manifest_sha256": P["manifest_sha256"],
           "config_sha256": P["cfg_sha256"], "registration_sha256": P["cfg"]["registration_sha256"],
           "evaluation_ids_sha256": P["cfg"]["evaluation"]["ids_sha256"]}
    write_json(out_dir / MANIFEST_FILE, man)
    stat = {"experiment": EXPERIMENT, "stage": "predictions", "status": status, "written_at": now(), "problems": problems,
            "passes": st, "environment_consistent": env_ok, "package": PACKAGE, "package_manifest_sha256": P["manifest_sha256"],
            "database": "not contacted (scoring runs on Windows against Atlas)", "external_model_apis": "none",
            **(extra_status or {})}
    write_json(out_dir / STATUS_FILE, stat)
    names = sorted(p.relative_to(out_dir).as_posix() for p in out_dir.rglob("*")
                   if p.is_file() and p.name != "SHA256SUMS.txt" and not p.name.endswith(".tmp"))
    (out_dir / "SHA256SUMS.txt").write_text("".join(f"{sha256_file(out_dir / n)}  {n}\n" for n in names), encoding="utf-8",
                                            newline="\n")
    tmp = Path(str(output_zip) + ".tmp")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
        for n in names + ["SHA256SUMS.txt"]:
            z.write(out_dir / n, f"{out_dir.name}/{n}")
    os.replace(tmp, output_zip)
    return stat, man


# ------------------------------------------------------------------ resume
def verify_resume_source(prev, pkg_manifest_sha):
    prev = Path(prev)
    sums = {}
    for ln in (prev / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
        h, _, n = ln.partition("  ")
        sums[n] = h
    files = {p.relative_to(prev).as_posix() for p in prev.rglob("*") if p.is_file()} - {"SHA256SUMS.txt"}
    need(files == set(sums) and all(sha256_file(prev / n) == h for n, h in sums.items()), f"resume source {prev}: hash check failed")
    st = json.loads((prev / STATUS_FILE).read_text(encoding="utf-8"))
    need(st.get("experiment") == EXPERIMENT and st.get("package_manifest_sha256") == pkg_manifest_sha,
         "resume source: another experiment or another package")
    return st


def seed_from_previous(prev, out_dir):
    seeded = []
    for src in sorted((Path(prev) / "passes").rglob("*")):
        if not src.is_file():
            continue
        rel = src.relative_to(prev)
        dst = Path(out_dir) / rel
        need(not dst.exists() or sha256_file(dst) == sha256_file(src), f"{dst} exists with different content")
        if not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dst)
            seeded.append(rel.as_posix())
    return seeded
