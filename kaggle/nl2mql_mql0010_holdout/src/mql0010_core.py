"""MQL-0010 Kaggle runtime core (generate only; no database, no gold labels, no credentials).

Implements registration A4 exactly (outputs/nl2mql_mql0010/preflight/amendment_A4_2026-11-19.json, pinned through
registration_index.json): 9 arms built from 9 generation passes.

  pass id          generates                                   batch  needs
  first:A0         every holdout item, zero-shot prompt        8      -
  first:D0         every item, MQL-0008/0009 hybrid prompt     8      -
  first:E          every item, diversified-selection prompt    8      -
  delta            EP (E prompt) and EF (E + notes) prompts    1      first:E   (only items whose EF prompt != E prompt)
  retry:A0V        validator-flagged A0 answers, schema msg    1      first:A0
  retry:D0V        validator-flagged D0 answers, schema msg    1      first:D0
  retry:D0G        the SAME flagged D0 answers, generic msg    1      first:D0
  retry:FULL       validator-flagged EF answers, schema msg    1      first:E, delta

Arms are ASSEMBLED from passes at export (driver.py semantics): reused records are byte-identical copies, and
scoring.verify_exports is the export gate. Every pass is resumable (append-only, fingerprinted, fsynced checkpoints,
records final unless status != ok) and every record carries model / revision / precision / generation settings /
runtime / config hash / prompt hash / retrieval decisions.

Reused UNCHANGED (byte copies in src/): mql0008_core (Checkpoint, run_generation, oom_safe_generator,
make_qwen_generator, verify_package, build_docs), mql0009_core (normalize_prediction, gpu_runtime_info,
check_precision, check_snapshot_revision), and the registered MQL-0010 modules (planner, retriever_v2, validator,
prompt_v2, recovery, holdout_view, scoring, structure, driver).
"""
import hashlib
import json
import os
import shutil
import sys
import time
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import mql0008_core as M8      # noqa: E402
import mql0009_core as K9      # noqa: E402
import holdout_view as HV      # noqa: E402
import planner as PL           # noqa: E402
import prompt_v2 as PV         # noqa: E402
import recovery as RC          # noqa: E402
import scoring as SC           # noqa: E402
import validator as V          # noqa: E402

EXPERIMENT = "MQL-0010"
PACKAGE = "nl2mql_mql0010_holdout"
STATUS_FILE = "MQL0010_STATUS.json"
MANIFEST_FILE = "MQL0010_PREDICTIONS_MANIFEST.json"
MODEL_ID = "Qwen/Qwen2.5-Coder-3B-Instruct"
MODEL_REVISION = "488639f1ff808d1d3d0ba301aef8c11461451ec5"
CORPUS_FILE = "data/corpus/approved_corpus_v2_1.jsonl"
PLAN_ARMS = ("A0", "D0", "E", "EF")
FIRST_PASSES = ("first:A0", "first:D0", "first:E")
RETRY_PASSES = {"retry:A0V": ("A0V", "A0", "schema"), "retry:D0V": ("D0V", "D0", "schema"),
                "retry:D0G": ("D0G", "D0", "generic"), "retry:FULL": ("FULL", "EF", "schema")}
PASSES = FIRST_PASSES + ("delta",) + tuple(RETRY_PASSES)
DEPENDS = {"delta": ("first:E",), "retry:A0V": ("first:A0",), "retry:D0V": ("first:D0",), "retry:D0G": ("first:D0",),
           "retry:FULL": ("first:E", "delta")}
REGISTERED_DOCS = ("config", "analysis_plan", "split_registration", "amendment_A2", "amendment_A3", "amendment_A4")
REGISTERED_SRC = ("driver.py", "holdout_view.py", "planner.py", "prompt_v2.py", "recovery.py", "retriever_v2.py",
                  "scoring.py", "structure.py", "validator.py")
GOLD_KEYS = {"sql", "gold_sql", "query", "gold", "gold_rows", "gold_columns", "answer", "label", "reference", "mql_gold"}

need = M8.need
read_jsonl, write_jsonl, write_json = M8.read_jsonl, M8.write_jsonl, M8.write_json
sha256_file, sha256_text, canon, fingerprint, now = M8.sha256_file, M8.sha256_text, M8.canon, M8.fingerprint, M8.now


# ------------------------------------------------------------------ package loading (fail closed)
def load_package(pkg):
    pkg = Path(pkg)
    man = M8.verify_package(pkg)                                      # every file hashed, no extra file
    cfg = json.loads((pkg / "configs/experiment.json").read_text(encoding="utf-8"))
    need(cfg["experiment_id"] == EXPERIMENT and man["package"] == PACKAGE, "not the MQL-0010 package")
    # registration A4: index + every pinned document + every registered source module
    idx_path = pkg / "registration/registration_index.json"
    need(sha256_file(idx_path) == cfg["registration_index_sha256"], "registration index differs from the config pin")
    idx = json.loads(idx_path.read_text(encoding="utf-8"))
    need(idx["experiment"] == EXPERIMENT and idx["effective_arms"] == list(SC.ARMS), "registration arms != scoring arms")
    docs = {d["role"]: d for d in idx["documents"]}
    need(set(docs) == set(REGISTERED_DOCS), f"registration documents {sorted(docs)}")
    for role in REGISTERED_DOCS:
        p = pkg / "registration" / Path(docs[role]["path"]).name
        need(sha256_file(p) == docs[role]["sha256"], f"registration copy {p.name} differs from the pinned hash")
    for name in REGISTERED_SRC:
        need(sha256_file(pkg / "src" / name) == idx["source_sha256"][name], f"src/{name} differs from the registered source")
    a4 = json.loads((pkg / "registration" / Path(docs["amendment_A4"]["path"]).name).read_text(encoding="utf-8"))
    need([a["name"] for a in a4["arms"]] == list(SC.ARMS), "A4 arms")
    need({a["name"]: a for a in a4["arms"]} == {a["name"]: a for a in cfg["arms"]}, "config arms != A4 arms")
    split = json.loads((pkg / "registration/split_registration.json").read_text(encoding="utf-8"))
    hdb = HV.holdout_dbs(split)
    # retrieval view (R3): corpus hash, registered view ids, no holdout DB
    view = HV.load_view(pkg / CORPUS_FILE, split)
    need(len(view) == cfg["corpus"]["view_n"] == 2912, "retrieval view size")
    # evaluation inputs: id/db/question only; registered ids
    evals = read_jsonl(pkg / "data/eval/eval_inputs.jsonl")
    need(all(set(e) == {"id", "db_id", "question"} for e in evals), "evaluation inputs must hold only id, db_id, question")
    ids = [e["id"] for e in evals]
    need(len(ids) == len(set(ids)) == split["holdout"]["n_items"] == cfg["evaluation"]["n_items"], "evaluation size")
    need(sha256_text("\n".join(ids)) == split["holdout"]["ids_sha256"] == cfg["evaluation"]["ids_sha256"], "evaluation ids")
    need({e["db_id"] for e in evals} == set(hdb), "evaluation databases != the 18 holdout databases")
    prompts = read_jsonl(pkg / "data/eval/prompts.jsonl")
    need([p["id"] for p in prompts] == ids, "frozen prompt ids/order")
    need(sha256_file(pkg / "data/eval/prompts.jsonl") == cfg["prompt"]["zero_shot_prompts_sha256"], "frozen prompts hash")
    need({p["template_sha256"] for p in prompts} == {cfg["prompt"]["zero_shot_template_sha256"]}, "zero-shot template hash")
    need(cfg["retrieval_exclusions"] == split["leakage"]["L3_L5_exclusion_pairs"], "retrieval exclusions != registration")
    for f in (pkg / "data/eval/eval_inputs.jsonl", pkg / "data/eval/prompts.jsonl"):
        need(not any(GOLD_KEYS & set(r) for r in read_jsonl(f)), f"{f.name}: gold-like field present")
    ctx = pkg / "data/context"
    return {"pkg": pkg, "manifest": man, "manifest_sha256": sha256_file(pkg / "package_manifest.json"), "cfg": cfg,
            "cfg_sha256": sha256_file(pkg / "configs/experiment.json"), "index": idx, "a4": a4, "split": split,
            "holdout_dbs": sorted(hdb), "view_rows": view, "evals": evals, "ids": ids, "prompts": prompts,
            "atlas": json.loads((ctx / "atlas_collections.json").read_text(encoding="utf-8")),
            "tables": {t["db_id"]: t for t in json.loads((ctx / "tables_subset.json").read_text(encoding="utf-8"))}}


def arm_flags(cfg, name):
    return {"name": name, **next(a for a in cfg["arms"] if a["name"] == name)["flags"]}


def derive_plans(P, count_tokens, compare_frozen=True):
    """Re-derive documents, the four plans and every first-pass prompt; compare with the frozen package copies."""
    cfg = P["cfg"]
    docs = M8.build_docs(P["view_rows"], P["tables"])
    HV.assert_docs_exclude(docs, set(P["holdout_dbs"]))
    by_doc = {d["doc_id"]: d for d in docs}
    infos = M8.eval_schema_infos(P["atlas"], P["holdout_dbs"])
    fps = {p["id"]: p for p in P["prompts"]}
    plans, prompts = {}, {}
    for arm in PLAN_ARMS:
        flags = {k: v for k, v in arm_flags(cfg, arm).items() if k != "validate_retry"}
        plan = PL.build_plan(flags, P["evals"], P["prompts"], docs, infos, P["atlas"], P["tables"], cfg["context"],
                             count_tokens, cfg["retrieval_exclusions"], forbidden_db_ids=P["holdout_dbs"])
        if compare_frozen:
            frozen = read_jsonl(P["pkg"] / f"data/plans/plan_{arm}.jsonl")
            bad = [a["id"] for a, b in zip(plan, frozen) if a != b]
            need(len(plan) == len(frozen) and not bad, f"{arm}: re-derived plan differs from the frozen plan on {len(bad)} "
                 f"items, e.g. {bad[:3]} (tokenizer / library difference?)")
            need(sha256_text(canon(plan)) == cfg["plans_sha256"][arm], f"{arm}: plan hash != config")
        recs = []
        for r in plan:
            system, user = PL.user_text_for(r, fps[r["id"]], by_doc)
            p = {"id": r["id"], "db_id": r["db_id"], "system": system, "messages": [{"role": "user", "content": user}]}
            p["prompt_sha256"] = PL.prompt_sha(system, p["messages"])
            need(p["prompt_sha256"] == r["prompt_sha256"], f"{arm} {r['id']}: prompt hash != plan")
            recs.append(p)
        plans[arm], prompts[arm] = plan, recs
    need([p["prompt_sha256"] for p in prompts["A0"]] == [PL.prompt_sha(p["system"], p["messages"]) for p in P["prompts"]],
         "zero-shot prompts are not byte-identical to the frozen prompts")
    return docs, plans, prompts


def delta_ids(plans):
    e = {r["id"]: r["prompt_sha256"] for r in plans["E"]}
    return [r["id"] for r in plans["EF"] if r["prompt_sha256"] != e[r["id"]]]


# ------------------------------------------------------------------ provenance
def pass_batch_size(cfg, pass_id):
    return cfg["model"]["batch_size_first_pass"] if pass_id in FIRST_PASSES else 1


def pass_fingerprint(P, pass_id, payload_sha):
    m = P["cfg"]["model"]
    return fingerprint({"experiment": EXPERIMENT, "pass": pass_id, "model": m["id"], "revision": m["revision"], "dtype": m["dtype"],
                        "decoding": m["decoding"], "max_new_tokens": m["max_new_tokens"], "seed": m["seed"],
                        "batch_size": pass_batch_size(P["cfg"], pass_id), "payload_sha256": payload_sha,
                        "package_manifest_sha256": P["manifest_sha256"], "config_sha256": P["cfg_sha256"],
                        "registration_index_sha256": P["cfg"]["registration_index_sha256"]})


def identity(P, pass_id, precision, runtime, plan_row=None):
    m = P["cfg"]["model"]
    out = {"experiment": EXPERIMENT, "pass": pass_id, "model_id": m["id"], "model_revision": m["revision"],
           "precision": precision, "runtime": runtime, "config_sha256": P["cfg_sha256"],
           "generation_config": {"dtype": m["dtype"], "decoding": m["decoding"], "do_sample": False,
                                 "max_new_tokens": m["max_new_tokens"], "batch_size": pass_batch_size(P["cfg"], pass_id),
                                 "seed": m["seed"]}}
    if plan_row is not None:
        out["retrieval"] = {k: plan_row[k] for k in ("selected_ids", "selected_ranks", "selected_scores", "selected_db_ids",
                                                     "excluded_by_rule", "schema_notes", "dropped_budget", "context_tokens",
                                                     "empty_context", "flags")}
    return out


def pass_dir(out_dir, pass_id):
    return Path(out_dir) / "passes" / pass_id.replace(":", "_")


def _write_pass_config(d, info):
    old = d / "pass_config.json"
    if old.exists():
        prev = json.loads(old.read_text(encoding="utf-8"))
        keep = ("sessions",)
        diff = {k: (prev.get(k), v) for k, v in info.items() if k not in keep and prev.get(k) != v}
        need(not diff, f"{d.name}: earlier pass_config differs: {diff}; refusing to mix runs")
        info["sessions"] = prev.get("sessions", []) + info.get("sessions", [])
    write_json(old, info)


def _env_key(runtime):
    return {k: runtime.get(k) for k in ("gpu", "compute_capability", "bf16_native", "torch", "transformers")}


# ------------------------------------------------------------------ generation passes
def run_first_pass(P, arm, plans, prompts, gen, precision, runtime, out_dir, max_batches=None, log=print):
    pass_id = f"first:{arm}"
    d = pass_dir(out_dir, pass_id)
    payload = sha256_text("".join(p["prompt_sha256"] + "\n" for p in prompts[arm]))
    fp = pass_fingerprint(P, pass_id, payload)
    _write_pass_config(d, {"pass": pass_id, "fingerprint": fp, "payload_sha256": payload, "env_key": _env_key(runtime),
                           "n_expected": len(prompts[arm]), "sessions": [{"at": now()}]})
    ck = M8.Checkpoint(d / "checkpoint.jsonl", [p["id"] for p in prompts[arm]], fp, M8.is_generation_retry)
    ident = {r["id"]: identity(P, pass_id, precision, runtime, r) for r in plans[arm]}
    log(f"{pass_id}: {len(ck.todo())} of {len(prompts[arm])} to generate")
    return M8.run_generation(prompts[arm], gen, ck, P["cfg"]["model"]["label"], arm, pass_batch_size(P["cfg"], pass_id),
                             max_batches=max_batches, identity=ident,
                             progress=lambda n, t: log(f"  {pass_id} {n}/{t}") if n % 80 < 8 else None)


def run_delta_pass(P, plans, prompts, gen, precision, runtime, out_dir, max_batches=None, log=print):
    """R2: for each item whose EF prompt differs from E's, generate the E prompt (EP) and the EF prompt in this pass,
    one prompt per call. Two checkpoints (EP, EF) under one pass fingerprint scope."""
    ids = delta_ids(plans)
    pe = {p["id"]: p for p in prompts["E"]}
    pf = {p["id"]: p for p in prompts["EF"]}
    ple = {r["id"]: r for r in plans["E"]}
    plf = {r["id"]: r for r in plans["EF"]}
    d = pass_dir(out_dir, "delta")
    payload = sha256_text("".join(f"{i}:{pe[i]['prompt_sha256']}:{pf[i]['prompt_sha256']}\n" for i in ids))
    fp = pass_fingerprint(P, "delta", payload)
    _write_pass_config(d, {"pass": "delta", "fingerprint": fp, "payload_sha256": payload, "env_key": _env_key(runtime),
                           "delta_ids": ids, "sessions": [{"at": now()}]})
    res = {}
    for arm, src, pl in (("EP", pe, ple), ("EF", pf, plf)):
        ck = M8.Checkpoint(d / f"checkpoint_{arm}.jsonl", ids, fp, M8.is_generation_retry)
        ident = {i: identity(P, "delta", precision, runtime, pl[i]) for i in ids}
        log(f"delta/{arm}: {len(ck.todo())} of {len(ids)} to generate")
        res[arm] = M8.run_generation([src[i] for i in ids], gen, ck, P["cfg"]["model"]["label"], arm, 1,
                                     max_batches=max_batches, identity=ident)
    return res


def retry_request(system, user, first_text, validator, mode):
    """The first half of recovery.recover, exposed so the retry prompt can be checkpointed: -> None (no retry) or
    (messages, feedback). Identical decisions to recovery.recover (asserted at assembly)."""
    obj, issues = RC.assess(first_text, validator)
    if obj is not None and not issues:
        return None
    if obj is None:
        fb = RC.FORMAT_FEEDBACK
    else:
        fb = V.format_feedback(issues) if mode == "schema" else RC.GENERIC_FEEDBACK
    return PV.recovery_messages(user, first_text, fb), fb


def run_retry_pass(P, pass_id, sources, prompts_by_arm, gen, precision, runtime, out_dir, max_batches=None, log=print):
    """sources: {id: first-pass record of the source arm}. Generates only for items that need a retry."""
    arm, src_arm, mode = RETRY_PASSES[pass_id]
    plan_arm = "EF" if src_arm == "EF" else src_arm
    prompts = {p["id"]: p for p in prompts_by_arm[plan_arm]}
    vals = {}
    reqs = []
    for i in P["ids"]:
        s = sources[i]
        need(s["status"] == "ok", f"{pass_id}: source {src_arm}/{i} is not complete")
        v = vals.setdefault(s["db_id"], V.SchemaValidator(s["db_id"], P["atlas"]))
        p = prompts[i]
        rq = retry_request(p["system"], p["messages"][0]["content"], s["raw_text"], v, mode)
        if rq is not None:
            msgs, fb = rq
            rec = {"id": i, "db_id": s["db_id"], "system": p["system"], "messages": msgs}
            rec["prompt_sha256"] = PL.prompt_sha(p["system"], msgs)
            reqs.append(rec)
    d = pass_dir(out_dir, pass_id)
    payload = sha256_text("".join(f"{r['id']}:{r['prompt_sha256']}\n" for r in reqs))
    fp = pass_fingerprint(P, pass_id, payload)
    _write_pass_config(d, {"pass": pass_id, "fingerprint": fp, "payload_sha256": payload, "env_key": _env_key(runtime),
                           "source_arm": src_arm, "feedback_mode": mode, "retry_ids": [r["id"] for r in reqs],
                           "sessions": [{"at": now()}]})
    ck = M8.Checkpoint(d / "checkpoint.jsonl", [r["id"] for r in reqs], fp, M8.is_generation_retry)
    ident = {r["id"]: identity(P, pass_id, precision, runtime) for r in reqs}
    log(f"{pass_id}: {len(ck.todo())} of {len(reqs)} retries to generate ({len(P['ids']) - len(reqs)} items need none)")
    return M8.run_generation(reqs, gen, ck, P["cfg"]["model"]["label"], arm, 1, max_batches=max_batches, identity=ident)


# ------------------------------------------------------------------ assembly (driver semantics) + export
def _ck_records(path, ids=None):
    if not Path(path).exists():
        return {}
    recs = {r["id"]: r for r in read_jsonl(path) if r.get("id")}
    return {i: r for i, r in recs.items() if ids is None or i in ids}


def pass_status(P, plans, prompts, out_dir):
    """-> {pass_id: {"status": complete|partial|absent, "n_ok", "n_expected"}} from the checkpoints on disk."""
    st = {}
    for pid in FIRST_PASSES:
        arm = pid.split(":")[1]
        recs = _ck_records(pass_dir(out_dir, pid) / "checkpoint.jsonl")
        n_ok = sum(r.get("status") == "ok" for r in recs.values())
        st[pid] = {"n_expected": len(P["ids"]), "n_ok": n_ok}
    ids = delta_ids(plans)
    ok = 0
    for arm in ("EP", "EF"):
        ok += sum(r.get("status") == "ok" for r in _ck_records(pass_dir(out_dir, "delta") / f"checkpoint_{arm}.jsonl").values())
    st["delta"] = {"n_expected": 2 * len(ids), "n_ok": ok}
    for pid in RETRY_PASSES:
        cfgp = pass_dir(out_dir, pid) / "pass_config.json"
        if not cfgp.exists():
            st[pid] = {"n_expected": None, "n_ok": 0}
            continue
        want = json.loads(cfgp.read_text(encoding="utf-8"))["retry_ids"]
        recs = _ck_records(pass_dir(out_dir, pid) / "checkpoint.jsonl")
        st[pid] = {"n_expected": len(want), "n_ok": sum(recs.get(i, {}).get("status") == "ok" for i in want)}
    for v in st.values():
        v["status"] = ("complete" if v["n_expected"] is not None and v["n_ok"] == v["n_expected"]
                       else "absent" if not v["n_ok"] and v["n_expected"] != 0 else "partial")
    return st


def _base_record(r, arm, source):
    keep = {k: r[k] for k in ("id", "db_id", "status", "raw_text", "prompt_sha256", "fingerprint", "pass", "model_id",
                              "model_revision", "precision", "runtime", "config_sha256", "generation_config", "retrieval",
                              "input_tokens", "output_tokens", "latency_s", "finish_reason", "oom_recovery",
                              "effective_batch_size") if k in r}
    return {**keep, "arm": arm, "first_raw_text": r["raw_text"], "source": source, "recovery": None,
            "experiment": EXPERIMENT, "prediction_id": f"{EXPERIMENT}:{arm}:{r['id']}"}


def assemble(P, plans, prompts, out_dir):
    """-> {arm: [records]} for all 9 arms, or raises if a needed pass is incomplete."""
    ids = P["ids"]
    first = {}
    for arm in ("A0", "D0", "E"):
        recs = _ck_records(pass_dir(out_dir, f"first:{arm}") / "checkpoint.jsonl")
        need(all(recs.get(i, {}).get("status") == "ok" for i in ids), f"first:{arm} incomplete")
        first[arm] = {i: _base_record(recs[i], arm, "generated") for i in ids}
    dids = set(delta_ids(plans))
    dl = {a: _ck_records(pass_dir(out_dir, "delta") / f"checkpoint_{a}.jsonl") for a in ("EP", "EF")}
    for a in ("EP", "EF"):
        need(all(dl[a].get(i, {}).get("status") == "ok" for i in dids), f"delta/{a} incomplete")
    arms = {a: [first[a][i] for i in ids] for a in ("A0", "D0", "E")}
    for a in ("EP", "EF"):
        arms[a] = [(_base_record(dl[a][i], a, "generated:matched_delta") if i in dids
                    else {**first["E"][i], "arm": a, "source": "reused:E", "prediction_id": f"{EXPERIMENT}:{a}:{i}"}) for i in ids]
    for pid, (arm, src, mode) in RETRY_PASSES.items():
        cfgp = pass_dir(out_dir, pid) / "pass_config.json"
        need(cfgp.exists(), f"{pid} not run")
        retry_ids = json.loads(cfgp.read_text(encoding="utf-8"))["retry_ids"]
        rr = _ck_records(pass_dir(out_dir, pid) / "checkpoint.jsonl")
        need(all(rr.get(i, {}).get("status") == "ok" for i in retry_ids), f"{pid} incomplete")
        plan_arm = src
        pr = {p["id"]: p for p in prompts[plan_arm]}
        srcrec = {r["id"]: r for r in arms[src]}
        vals, out = {}, []
        for i in ids:
            s = srcrec[i]
            v = vals.setdefault(s["db_id"], V.SchemaValidator(s["db_id"], P["atlas"]))
            p = pr[i]
            req = retry_request(p["system"], p["messages"][0]["content"], s["first_raw_text"], v, mode)
            need((req is not None) == (i in set(retry_ids)), f"{pid}/{i}: retry decision differs from the pass record")

            def cached(system, msgs, _i=i, _req=req):
                need(_req is not None and msgs == _req[0] and system == p["system"], f"{pid}/{_i}: retry prompt mismatch")
                return rr[_i]["raw_text"]
            final, trace = RC.recover(p["system"], p["messages"][0]["content"], s["first_raw_text"], v, cached,
                                      feedback_mode=mode)
            rec = {**s, "arm": arm, "raw_text": final, "source": f"reused:{src}", "recovery": trace,
                   "prediction_id": f"{EXPERIMENT}:{arm}:{i}"}
            if i in rr:
                rec["retry_generation"] = {k: rr[i].get(k) for k in ("prompt_sha256", "fingerprint", "input_tokens",
                                                                    "output_tokens", "latency_s", "finish_reason")}
            out.append(rec)
        arms[arm] = out
    for a in arms:
        arms[a] = [{**r, **K9.normalize_prediction(r["raw_text"], r["status"])} for r in arms[a]]
    return arms


def export(P, plans, prompts, out_dir, output_zip, extra_status=None):
    out_dir = Path(out_dir)
    st = pass_status(P, plans, prompts, out_dir)
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
        arms = assemble(P, plans, prompts, out_dir)
        ver = SC.verify_exports(arms, plans, P["ids"])                  # the export gate (R2/R5 contracts)
        for a, recs in arms.items():
            write_jsonl(out_dir / "predictions" / a / "predictions.jsonl", recs)
            per_arm[a] = {"n": len(recs), "predictions_sha256": sha256_file(out_dir / "predictions" / a / "predictions.jsonl")}
        for a in PLAN_ARMS:
            write_jsonl(out_dir / "plans" / f"plan_{a}.jsonl", plans[a])
    status = "complete" if complete else "partial"
    man = {"experiment": EXPERIMENT, "written_at": now(), "status": status, "passes": st, "per_arm": per_arm,
           "expected_arms": list(SC.ARMS), "verify_exports": ver, "package_manifest_sha256": P["manifest_sha256"],
           "config_sha256": P["cfg_sha256"], "registration_index_sha256": P["cfg"]["registration_index_sha256"],
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
    """Copy every pass file of an earlier, hash-verified output into OUT_DIR (never overwrites different content)."""
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


def check_dependencies(out_dir, P, plans, prompts, pass_id):
    st = pass_status(P, plans, prompts, out_dir)
    missing = [d for d in DEPENDS.get(pass_id, ()) if st[d]["status"] != "complete"]
    need(not missing, f"{pass_id} needs completed {missing}")


def retry_sources(P, plans, out_dir, src_arm):
    """First answers a retry pass works on: A0 / D0 first passes, or EF = E's first answers with the delta-pass EF
    answers on the changed items (exactly the EF arm assembly)."""
    if src_arm in ("A0", "D0"):
        recs = _ck_records(pass_dir(out_dir, f"first:{src_arm}") / "checkpoint.jsonl")
        need(all(recs.get(i, {}).get("status") == "ok" for i in P["ids"]), f"first:{src_arm} incomplete")
        return recs
    need(src_arm == "EF", src_arm)
    e = _ck_records(pass_dir(out_dir, "first:E") / "checkpoint.jsonl")
    f = _ck_records(pass_dir(out_dir, "delta") / "checkpoint_EF.jsonl")
    dids = set(delta_ids(plans))
    out = {i: (f.get(i) if i in dids else e.get(i)) for i in P["ids"]}
    need(all((out[i] or {}).get("status") == "ok" for i in P["ids"]), "EF sources incomplete (first:E / delta)")
    return out
