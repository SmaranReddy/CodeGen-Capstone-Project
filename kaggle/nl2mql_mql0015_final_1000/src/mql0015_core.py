"""MQL-0015 Kaggle runtime core (generate only; no database, no gold labels, no credentials).

Three registered arms on the same 1,000 Spider test_full ids, each through the MQL-0010 component V (static schema
validator + at most one schema-feedback retry; keep the retry iff strictly fewer issues):
  A0V     base Qwen2.5-Coder-3B @ 488639f1, frozen zero-shot prompt
  R3V     base model, frozen R3 prompt (hybrid BM25 + bge-small dense, RRF, Jaccard, top 3)
  FT-R3V  base + verified LoRA q3b_r16_q10 (sha 7096760a...) merged, the SAME frozen R3 prompt
First answers A0, R3, FT-R3 are exported too (descriptive).

  pass           model  generates                                                  batch
  first:A0       base   zero-shot prompts                                            8
  first:R3       base   R3 prompts                                                   8
  retry:base     base   every DISTINCT retry prompt needed by A0V or R3V             1
  first:FT-R3    ft     R3 prompts (same ids, same order as first:R3)                8
  retry:FT       ft     every DISTINCT retry prompt needed by FT-R3V                 1

Generation failures (amendment A1): at most 2 attempts per item per pass (attempt 1 at the pass batch size with
OOM -> batch-1 fallback, attempt 2 at batch 1). A record still failing after attempt 2 is final (status "error") and is
exported as such; it never blocks export. No schema retry for a failed first answer; a failed retry keeps the first answer.

Reused UNCHANGED (byte copies in src/ of the frozen MQL-0013 package): mql0008_core (Checkpoint, run_generation,
oom_safe_generator, make_qwen_generator, verify_package), mql0009_core, planner, prompt_v2, recovery, validator.
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

EXPERIMENT = "MQL-0015"
PACKAGE = "nl2mql_mql0015_final_1000"
STATUS_FILE = "MQL0015_STATUS.json"
MANIFEST_FILE = "MQL0015_PREDICTIONS_MANIFEST.json"
MODEL_ID = "Qwen/Qwen2.5-Coder-3B-Instruct"
MODEL_REVISION = "488639f1ff808d1d3d0ba301aef8c11461451ec5"
ADAPTER_SHA256 = "7096760ac2ed7e4a6c2eefdd7c0c249d4b3dba723f069982a499b876d434e09d"
ADAPTER_REL = "model/ft_adapter"
ARMS = ("A0", "A0V", "R3", "R3V", "FT-R3", "FT-R3V")
FIRST_ARMS = ("A0", "R3", "FT-R3")
V_ARMS = {"A0V": "A0", "R3V": "R3", "FT-R3V": "FT-R3"}
PROMPT_SET = {"A0": "A0", "R3": "R3", "FT-R3": "R3"}          # FT-R3 uses the R3 prompts byte for byte
FIRST_PASS = {"A0": "first:A0", "R3": "first:R3", "FT-R3": "first:FT-R3"}
FIRST_PASSES = ("first:A0", "first:R3", "first:FT-R3")
RETRY_PASSES = {"retry:base": ("A0", "R3"), "retry:FT": ("FT-R3",)}
PASSES = ("first:A0", "first:R3", "retry:base", "first:FT-R3", "retry:FT")
PASS_MODEL = {"first:A0": "base", "first:R3": "base", "retry:base": "base", "first:FT-R3": "ft", "retry:FT": "ft"}
MAX_ATTEMPTS = 2
GOLD_KEYS = {"sql", "gold_sql", "query", "gold", "gold_rows", "gold_columns", "answer", "label", "reference", "mql_gold",
             "hardness", "structure_labels", "primary_class", "flags", "execution"}

need = M8.need
read_jsonl, write_jsonl, write_json = M8.read_jsonl, M8.write_jsonl, M8.write_json
sha256_file, sha256_text, canon, fingerprint, now = M8.sha256_file, M8.sha256_text, M8.canon, M8.fingerprint, M8.now


# ------------------------------------------------------------------ package loading (fail closed)
def load_package(pkg):
    pkg = Path(pkg)
    man = M8.verify_package(pkg)
    cfg = json.loads((pkg / "configs/experiment.json").read_text(encoding="utf-8"))
    need(cfg["experiment_id"] == EXPERIMENT and man["package"] == PACKAGE, "not the MQL-0015 package")
    reg_path = pkg / "registration/mql0015_registration.json"
    need(sha256_file(reg_path) == cfg["registration_sha256"], "registration differs from the config pin")
    need(sha256_file(pkg / "registration/amendment_A1_2026-11-23.json") == cfg["amendment_A1_sha256"], "amendment A1 pin")
    reg = json.loads(reg_path.read_text(encoding="utf-8"))
    need(reg["experiment"] == EXPERIMENT and sorted(reg["arms"]) == sorted(V_ARMS), "registration arms")
    evals = read_jsonl(pkg / "data/eval/eval_inputs.jsonl")
    need(all(set(e) == {"id", "db_id", "question"} for e in evals), "evaluation inputs must hold only id, db_id, question")
    ids = [e["id"] for e in evals]
    ids_sha = sha256_text("".join(i + "\n" for i in ids))
    need(len(ids) == len(set(ids)) == 1000 and ids_sha == cfg["evaluation"]["ids_sha256"] == reg["evaluation"]["sample"]["ids_sha256"],
         "evaluation ids differ from the registered 1,000-item sample")
    pins = read_jsonl(pkg / "data/prompt_pins.jsonl")
    need(sha256_file(pkg / "data/prompt_pins.jsonl") == reg["pins"]["prompt_pins_sha256"], "prompt pins differ from the registration")
    need([p["id"] for p in pins] == ids, "prompt pins ids/order")
    pin = {p["id"]: p for p in pins}
    prompts = {s: read_jsonl(pkg / f"data/prompts/prompts_{s}.jsonl") for s in ("A0", "R3")}
    plan = read_jsonl(pkg / "data/plans/plan_R3.jsonl")
    for s, key in (("A0", "zero_shot_prompt_sha256"), ("R3", "r3_prompt_sha256")):
        need([p["id"] for p in prompts[s]] == ids, f"{s}: prompt ids/order")
        for p in prompts[s]:
            sha = PL.prompt_sha(p["system"], p["messages"])
            need(sha == p["prompt_sha256"] == pin[p["id"]][key], f"{s}/{p['id']}: prompt hash != registered pin")
    need([r["id"] for r in plan] == ids, "R3 plan ids/order")
    for r, p in zip(plan, prompts["R3"]):
        need(r["prompt_sha256"] == p["prompt_sha256"], f"R3/{r['id']}: plan prompt hash")
        need(r["selected_ids"] == pin[r["id"]]["r3_selected_ids"] and len(r["selected_ids"]) == 3, f"R3/{r['id']}: selection")
        need(r["retrieval_mql0011"]["lexical"] == "rrf_fused" and r["retrieval_mql0011"]["index_name"] == cfg["retrieval"]["index"],
             f"R3/{r['id']}: retrieval mode/index")
        need(not r["dropped_budget"] and not r["empty_context"], f"R3/{r['id']}: budget drop / empty context")
    for a, b in zip(prompts["A0"], prompts["R3"]):
        za, zr = a["messages"][0]["content"], b["messages"][0]["content"]
        cut = za.index("\n\nQuestion:")
        need(a["system"] == b["system"] and zr.startswith(za[:cut]) and zr.endswith(za[cut:]), f"{a['id']}: schema context differs")
    for f in (pkg / "data/eval/eval_inputs.jsonl", pkg / "data/prompts/prompts_A0.jsonl", pkg / "data/prompts/prompts_R3.jsonl",
              pkg / "data/prompt_pins.jsonl"):
        need(not any(GOLD_KEYS & set(r) for r in read_jsonl(f)), f"{f.name}: gold-like field present")
    ad = pkg / ADAPTER_REL
    need(sha256_file(ad / "adapter_model.safetensors") == ADAPTER_SHA256 == cfg["ft"]["adapter_sha256"], "adapter hash")
    need(sha256_file(ad / "adapter_config.json") == cfg["ft"]["adapter_config_sha256"], "adapter config hash")
    atlas = json.loads((pkg / "data/context/atlas_collections.json").read_text(encoding="utf-8"))
    return {"pkg": pkg, "manifest": man, "manifest_sha256": sha256_file(pkg / "package_manifest.json"), "cfg": cfg,
            "cfg_sha256": sha256_file(pkg / "configs/experiment.json"), "reg": reg, "evals": evals, "ids": ids,
            "prompts": prompts, "plan": plan, "atlas": atlas, "adapter_dir": ad}


def arm_prompts(P, arm):
    return P["prompts"][PROMPT_SET[arm]]


# ------------------------------------------------------------------ provenance
def pass_batch_size(cfg, pass_id):
    return cfg["model"]["batch_size_first_pass"] if pass_id in FIRST_PASSES else cfg["model"]["batch_size_retry"]


def model_identity(P, model_key):
    m = P["cfg"]["model"]
    ident = {"model": m["id"], "revision": m["revision"], "dtype": m["dtype"], "weights": model_key}
    if model_key == "ft":
        ident.update(adapter_sha256=P["cfg"]["ft"]["adapter_sha256"], merge=P["cfg"]["ft"]["merge"],
                     tokenizer=P["cfg"]["ft"]["tokenizer"])
    return ident


def pass_fingerprint(P, pass_id, payload_sha):
    m = P["cfg"]["model"]
    return fingerprint({"experiment": EXPERIMENT, "pass": pass_id, **model_identity(P, PASS_MODEL[pass_id]),
                        "decoding": m["decoding"], "max_new_tokens": m["max_new_tokens"], "seed": m["seed"],
                        "batch_size": pass_batch_size(P["cfg"], pass_id), "max_attempts": MAX_ATTEMPTS,
                        "payload_sha256": payload_sha, "package_manifest_sha256": P["manifest_sha256"],
                        "config_sha256": P["cfg_sha256"]})


def identity(P, pass_id, precision, runtime, attempt, weights_info):
    m = P["cfg"]["model"]
    return {"experiment": EXPERIMENT, "pass": pass_id, "model_id": m["id"], "model_revision": m["revision"],
            "weights": model_identity(P, PASS_MODEL[pass_id]), "weights_info": weights_info, "attempt": attempt,
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


def is_retryable(rec):
    """A generation record is final when ok, or when it failed on its last allowed attempt."""
    return rec.get("status") != "ok" and int(rec.get("attempt") or 1) < MAX_ATTEMPTS


def _final(rec):
    return rec is not None and not is_retryable(rec)


def _run_with_attempts(P, pass_id, items, gen_main, gen_single, ck, precision, runtime, weights_info, max_batches, log):
    """Attempt 1 at the pass batch size (with OOM fallback inside gen_main); attempt 2 at batch size 1."""
    out = {"generated": 0}
    label = P["cfg"]["model"]["label"]
    # round 1: items never attempted -> attempt 1 at the pass batch size (gen_main falls back to batch 1 on OOM)
    todo = set(ck.todo())
    q1 = [p for p in items if p["id"] in todo and p["id"] not in ck.records]
    if q1:
        bs = pass_batch_size(P["cfg"], pass_id)
        ident = {p["id"]: identity(P, pass_id, precision, runtime, 1, weights_info) for p in q1}
        out["generated"] += M8.run_generation(q1, gen_main, ck, label, pass_id, bs, max_batches=max_batches, identity=ident,
                                              progress=lambda n, t: log(f"  {pass_id} {n}/{t}") if n % 80 < bs else None)["generated"]
        if max_batches is not None:          # kill-and-resume test hook: stop after the capped batches
            return out
    # round 2: items whose last record failed and still have an attempt left -> next attempt at batch size 1
    todo = set(ck.todo())
    q2 = [p for p in items if p["id"] in todo and p["id"] in ck.records]
    if q2:
        ident = {p["id"]: identity(P, pass_id, precision, runtime, int(ck.records[p["id"]].get("attempt") or 1) + 1,
                                   weights_info) for p in q2}
        out["generated"] += M8.run_generation(q2, gen_single, ck, label, pass_id, 1, identity=ident)["generated"]
    out["final_errors"] = sorted(i for i, r in ck.records.items() if r.get("status") != "ok" and _final(r))
    return out


# ------------------------------------------------------------------ generation
def run_first_pass(P, pass_id, gen_main, gen_single, precision, runtime, out_dir, weights_info, max_batches=None, log=print):
    arm = {v: k for k, v in FIRST_PASS.items()}[pass_id]
    items = arm_prompts(P, arm)
    d = pass_dir(out_dir, pass_id)
    payload = sha256_text("".join(f"{p['id']}:{p['prompt_sha256']}\n" for p in items))
    fp = pass_fingerprint(P, pass_id, payload)
    _write_pass_config(d, {"pass": pass_id, "fingerprint": fp, "payload_sha256": payload, "env_key": _env_key(runtime),
                           "weights_info": weights_info, "n_expected": len(items), "sessions": [{"at": now()}]})
    ck = M8.Checkpoint(d / "checkpoint.jsonl", [p["id"] for p in items], fp, is_retryable)
    log(f"{pass_id}: {len(ck.todo())} of {len(items)} to generate")
    return _run_with_attempts(P, pass_id, items, gen_main, gen_single, ck, precision, runtime, weights_info, max_batches, log)


def first_answers(P, out_dir, arm):
    recs = _ck_records(pass_dir(out_dir, FIRST_PASS[arm]) / "checkpoint.jsonl")
    out = {i: recs.get(i) for i in P["ids"]}
    need(all(_final(out[i]) for i in P["ids"]), f"first answers for {arm} incomplete")
    return out


def retry_request(system, user, first_text, validator):
    """First half of recovery.recover (schema mode): -> None (no retry) or (messages, feedback)."""
    obj, issues = RC.assess(first_text, validator)
    if obj is not None and not issues:
        return None
    fb = RC.FORMAT_FEEDBACK if obj is None else V.format_feedback(issues)
    return PV.recovery_messages(user, first_text, fb), fb


def retry_requests(P, out_dir, pass_id):
    """-> (per-arm {id: prompt_sha or None}, distinct retry prompt records) for the arms of one retry pass."""
    vals, per_arm, distinct = {}, {}, {}
    for arm in RETRY_PASSES[pass_id]:
        first = first_answers(P, out_dir, arm)
        pr = {p["id"]: p for p in arm_prompts(P, arm)}
        per_arm[arm] = {}
        for i in P["ids"]:
            s, p = first[i], pr[i]
            if s["status"] != "ok":                 # generation failure: no schema retry (amendment A1)
                per_arm[arm][i] = None
                continue
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


def run_retry_pass(P, pass_id, gen_single, precision, runtime, out_dir, weights_info, max_batches=None, log=print):
    per_arm, reqs = retry_requests(P, out_dir, pass_id)
    d = pass_dir(out_dir, pass_id)
    payload = sha256_text("".join(f"{r['id']}:{r['prompt_sha256']}\n" for r in reqs))
    fp = pass_fingerprint(P, pass_id, payload)
    _write_pass_config(d, {"pass": pass_id, "fingerprint": fp, "payload_sha256": payload, "env_key": _env_key(runtime),
                           "weights_info": weights_info, "retry_ids": [r["id"] for r in reqs],
                           "retried_items": {a: sorted(i for i, s in per_arm[a].items() if s) for a in per_arm},
                           "sessions": [{"at": now()}]})
    ck = M8.Checkpoint(d / "checkpoint.jsonl", [r["id"] for r in reqs], fp, is_retryable)
    log(f"{pass_id}: {len(ck.todo())} of {len(reqs)} distinct retry prompts to generate "
        f"({', '.join(f'{a} {sum(bool(s) for s in per_arm[a].values())} items' for a in per_arm)})")
    return _run_with_attempts(P, pass_id, reqs, gen_single, gen_single, ck, precision, runtime, weights_info, max_batches, log)


# ------------------------------------------------------------------ status, assembly, export
def pass_status(P, out_dir):
    st = {}
    for pid in FIRST_PASSES:
        arm = {v: k for k, v in FIRST_PASS.items()}[pid]
        recs = _ck_records(pass_dir(out_dir, pid) / "checkpoint.jsonl")
        items = arm_prompts(P, arm)
        st[pid] = {"n_expected": len(items), "n_final": sum(_final(recs.get(p["id"])) for p in items),
                   "n_ok": sum((recs.get(p["id"]) or {}).get("status") == "ok" for p in items)}
    for pid in RETRY_PASSES:
        cfgp = pass_dir(out_dir, pid) / "pass_config.json"
        if cfgp.exists():
            want = json.loads(cfgp.read_text(encoding="utf-8"))["retry_ids"]
            recs = _ck_records(pass_dir(out_dir, pid) / "checkpoint.jsonl")
            st[pid] = {"n_expected": len(want), "n_final": sum(_final(recs.get(i)) for i in want),
                       "n_ok": sum((recs.get(i) or {}).get("status") == "ok" for i in want)}
        else:
            st[pid] = {"n_expected": None, "n_final": 0, "n_ok": 0}
    for v in st.values():
        v["generation_failures"] = v["n_final"] - v["n_ok"]
        v["status"] = ("complete" if v["n_expected"] is not None and v["n_final"] == v["n_expected"]
                       else "absent" if not v["n_final"] and v["n_expected"] != 0 else "partial")
    return st


def check_dependencies(P, out_dir, pass_id):
    if pass_id in RETRY_PASSES:
        st = pass_status(P, out_dir)
        missing = [FIRST_PASS[a] for a in RETRY_PASSES[pass_id] if st[FIRST_PASS[a]]["status"] != "complete"]
        need(not missing, f"{pass_id} needs completed {missing}")


_KEEP = ("id", "db_id", "status", "raw_text", "error", "prompt_sha256", "fingerprint", "pass", "attempt", "model_id",
         "model_revision", "weights", "weights_info", "precision", "runtime", "config_sha256", "generation_config",
         "input_tokens", "output_tokens", "latency_s", "finish_reason", "oom_recovery", "effective_batch_size")


def assemble(P, out_dir):
    ids = P["ids"]
    arms, vals = {}, {}
    plan = {r["id"]: r for r in P["plan"]}
    for pid, src_arms in RETRY_PASSES.items():
        cfgp = pass_dir(out_dir, pid) / "pass_config.json"
        need(cfgp.exists(), f"{pid} not run")
        rcfg = json.loads(cfgp.read_text(encoding="utf-8"))
        rr_by_id = _ck_records(pass_dir(out_dir, pid) / "checkpoint.jsonl")
        need(all(_final(rr_by_id.get(i)) for i in rcfg["retry_ids"]), f"{pid} incomplete")
        per_arm, reqs = retry_requests(P, out_dir, pid)
        need([r["id"] for r in reqs] == rcfg["retry_ids"], f"{pid}: retry prompts differ from the pass record")
        rr = {r["prompt_sha256"]: rr_by_id[r["id"]] for r in reqs}
        for arm in src_arms:
            first = first_answers(P, out_dir, arm)
            pr = {p["id"]: p for p in arm_prompts(P, arm)}
            base = []
            for i in ids:
                r = first[i]
                rec = {**{k: r[k] for k in _KEEP if k in r}, "arm": arm, "first_raw_text": r["raw_text"],
                       "source": f"generated:{FIRST_PASS[arm]}", "recovery": None,
                       "retrieval": ({k: plan[i][k] for k in ("selected_ids", "selected_ranks", "selected_scores",
                                                              "selected_db_ids", "excluded_by_rule", "dropped_budget",
                                                              "context_tokens", "prompt_tokens_est", "empty_context")}
                                     if PROMPT_SET[arm] == "R3" else None),
                       "experiment": EXPERIMENT, "prediction_id": f"{EXPERIMENT}:{arm}:{i}"}
                need(rec["prompt_sha256"] == pr[i]["prompt_sha256"], f"{arm}/{i}: generated prompt != frozen prompt")
                base.append(rec)
            arms[arm] = base
            varm = arm + "V"
            out = []
            for rec in base:
                i, p = rec["id"], pr[rec["id"]]
                sha = per_arm[arm][i]
                if rec["status"] != "ok":
                    need(sha is None, f"{varm}/{i}: retry planned for a generation failure")
                    out.append({**rec, "arm": varm, "source": f"reused:{arm}", "prediction_id": f"{EXPERIMENT}:{varm}:{i}",
                                "recovery": {"enabled": True, "first_parsed": False, "first_issues": None, "retried": False,
                                             "retry_parsed": None, "retry_issues": None, "kept": "first",
                                             "feedback_mode": "schema", "first_generation_failed": True}})
                    continue
                v = vals.setdefault(rec["db_id"], V.SchemaValidator(rec["db_id"], P["atlas"]))
                failed_retry = []

                def cached(system, msgs, _i=i, _sha=sha, _p=p):
                    need(_sha is not None and PL.prompt_sha(system, msgs) == _sha and system == _p["system"],
                         f"{varm}/{_i}: retry prompt mismatch")
                    g = rr[_sha]
                    if g["status"] != "ok":
                        failed_retry.append(True)
                        return ""                  # unparseable -> infinite issue score -> the first answer is kept
                    return g["raw_text"]
                final, trace = RC.recover(p["system"], p["messages"][0]["content"], rec["first_raw_text"], v, cached,
                                          feedback_mode="schema")
                need(bool(trace["retried"]) == (sha is not None), f"{varm}/{i}: retry decision differs from the pass record")
                if failed_retry:
                    trace["retry_generation_failed"] = True
                    need(trace["kept"] == "first", f"{varm}/{i}: failed retry kept")
                nr = {**rec, "arm": varm, "raw_text": final, "source": f"reused:{arm}", "recovery": trace,
                      "prediction_id": f"{EXPERIMENT}:{varm}:{i}"}
                if sha is not None:
                    g = rr[sha]
                    nr["retry_generation"] = {"prompt_sha256": sha, "pass": pid, "content_id": "rq_" + sha[:24],
                                              **{k: g.get(k) for k in ("status", "fingerprint", "attempt", "input_tokens",
                                                                       "output_tokens", "latency_s", "finish_reason")}}
                out.append(nr)
            arms[varm] = out
    for a in arms:
        arms[a] = [{**r, **K9.normalize_prediction(r["raw_text"], r["status"])} for r in arms[a]]
    return {a: arms[a] for a in ARMS}


def verify_exports(recs, P):
    ids = P["ids"]
    need(set(recs) == set(ARMS), f"arms {sorted(recs)} != {list(ARMS)}")
    by = {}
    for a in ARMS:
        got = [r["id"] for r in recs[a]]
        need(got == list(ids), f"{a}: ids/order differ from the registered 1,000-item sample")
        need(all(r["arm"] == a for r in recs[a]), f"{a}: arm label")
        need(all(r["status"] in ("ok", "error") for r in recs[a]), f"{a}: status")
        by[a] = {r["id"]: r for r in recs[a]}
    pins = {a: {p["id"]: p["prompt_sha256"] for p in arm_prompts(P, a)} for a in FIRST_ARMS}
    for a in FIRST_ARMS:
        need(all(by[a][i]["prompt_sha256"] == pins[a][i] for i in ids), f"{a}: prompt hash differs from the frozen prompt")
        want = "ft" if a == "FT-R3" else "base"
        need(all(by[a][i]["weights"]["weights"] == want for i in ids), f"{a}: generated with the wrong weights")
    need(all(by["FT-R3"][i]["prompt_sha256"] == by["R3"][i]["prompt_sha256"] for i in ids), "FT-R3 prompts != R3 prompts")
    need(all(by["FT-R3"][i]["weights"].get("adapter_sha256") == ADAPTER_SHA256 for i in ids), "FT-R3: adapter identity")
    for va, src in V_ARMS.items():
        for i in ids:
            r, s = by[va][i], by[src][i]
            need(r["first_raw_text"] == s["first_raw_text"] and r["status"] == s["status"], f"{va}/{i}: first answer differs from {src}")
            t = r["recovery"]
            need(t is not None and t.get("feedback_mode") == "schema", f"{va}/{i}: recovery trace")
            if not t["retried"] or t["kept"] != "retry":
                need(r["raw_text"] == r["first_raw_text"], f"{va}/{i}: changed without a kept retry")
            if r.get("retry_generation"):
                want = "retry:FT" if va == "FT-R3V" else "retry:base"
                need(r["retry_generation"]["pass"] == want, f"{va}/{i}: retry generated by the wrong model pass")
    return {"ok": True, "n": len(ids),
            "generation_failures": {a: sum(by[a][i]["status"] != "ok" for i in ids) for a in ARMS},
            "retried": {a: sum(bool(by[a][i]["recovery"]["retried"]) for i in ids) for a in V_ARMS},
            "kept_retry": {a: sum(by[a][i]["recovery"]["kept"] == "retry" for i in ids) for a in V_ARMS},
            "retry_generation_failed": {a: sum(bool(by[a][i]["recovery"].get("retry_generation_failed")) for i in ids) for a in V_ARMS}}


def export(P, out_dir, output_zip, extra_status=None):
    out_dir = Path(out_dir)
    st = pass_status(P, out_dir)
    keys, weights = set(), {}
    for pid in PASSES:
        c = pass_dir(out_dir, pid) / "pass_config.json"
        if c.exists():
            pc = json.loads(c.read_text(encoding="utf-8"))
            keys.add(canon(pc.get("env_key")))
            weights.setdefault(PASS_MODEL[pid], set()).add(canon(pc.get("weights_info")))
    env_ok = len(keys) <= 1
    weights_ok = all(len(v) == 1 for v in weights.values())
    problems = ([] if env_ok else [f"passes ran in different environments: {sorted(keys)}"]) + \
               ([] if weights_ok else [f"one model key used different weights across passes: {weights}"])
    complete = all(v["status"] == "complete" for v in st.values()) and env_ok and weights_ok
    ver, per_arm = None, {}
    if complete:
        arms = assemble(P, out_dir)
        ver = verify_exports(arms, P)
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
            "passes": st, "environment_consistent": env_ok, "weights_consistent": weights_ok, "package": PACKAGE,
            "package_manifest_sha256": P["manifest_sha256"],
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


# ------------------------------------------------------------------ fine-tuned weights (Kaggle only)
# files that decide tokenization and decoding: copied BYTE-FOR-BYTE from the base snapshot into the merged folder, so
# FT-R3V tokenizes and stops exactly like R3V (make_qwen_generator reads stop ids from generation_config / tokenizer)
DECODING_FILES = ("generation_config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
                  "special_tokens_map.json", "added_tokens.json", "chat_template.jinja")


def merge_adapter(base_path, adapter_dir, merged_dir, dtype="bfloat16", expected_sha=ADAPTER_SHA256):
    """Port of the merge cell that produced the scored MQL-0010-FT1 run (handoff eval notebook): peft
    merge_and_unload of the verified adapter into the pinned base on CPU in bf16, with the Kaggle torchao workaround,
    a trained-LoRA check and a changed-tensor count. The merged folder then receives the base snapshot's tokenizer and
    generation files byte-for-byte (never the Unsloth tokenizer files). -> weights_info."""
    import gc
    import torch
    need(sha256_file(Path(adapter_dir) / "adapter_model.safetensors") == expected_sha, "adapter hash before merge")
    import peft  # noqa: F401
    # Kaggle's image ships an old torchao; some peft versions RAISE on it instead of ignoring it. Merging a plain LoRA
    # never uses torchao, so tell every peft module it isn't available (as in the scored FT1 run).
    for _m in list(sys.modules.values()):
        if getattr(_m, "__name__", "").startswith("peft") and hasattr(_m, "is_torchao_available"):
            _m.is_torchao_available = lambda: False
    from peft import PeftModel
    from transformers import AutoModelForCausalLM
    acfg = json.loads((Path(adapter_dir) / "adapter_config.json").read_text(encoding="utf-8"))
    base = AutoModelForCausalLM.from_pretrained(base_path, torch_dtype=getattr(torch, dtype))       # CPU
    tgt = tuple(f"{t}.weight" for t in (acfg.get("target_modules") or ()))
    before = {n: p.detach().float().sum().item() for n, p in base.named_parameters() if n.endswith(tgt)}
    pm = PeftModel.from_pretrained(base, str(adapter_dir))
    lb = [p for n, p in pm.named_parameters() if "lora_B" in n]
    need(lb and any(float(p.detach().abs().sum()) > 0 for p in lb), "adapter has no trained LoRA weights")
    merged = pm.merge_and_unload()
    after = {n: p.detach().float().sum().item() for n, p in merged.named_parameters() if n in before}
    changed = sum(abs(after[n] - before[n]) > 0 for n in before)
    need(changed > 0, "merging the adapter changed no weight")
    Path(merged_dir).mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(merged_dir, safe_serialization=True)
    del base, pm, merged
    gc.collect()
    copied = {}
    for n in DECODING_FILES:
        src = Path(base_path) / n
        if src.exists():
            shutil.copyfile(src, Path(merged_dir) / n)
            need(sha256_file(Path(merged_dir) / n) == sha256_file(src), f"{n}: merged copy differs from the base snapshot")
            copied[n] = sha256_file(src)
    need("generation_config.json" in copied and "tokenizer_config.json" in copied, "base snapshot lacks decoding files")
    return {**weights_info_of(merged_dir, expected_sha), "changed_tensors": f"{changed}/{len(before)}",
            "decoding_files_sha256_equal_to_base": copied}


def weights_info_of(merged_dir, adapter_sha=ADAPTER_SHA256):
    files = sorted(p for p in Path(merged_dir).glob("*.safetensors"))
    need(files, "no merged safetensors written")
    per = {p.name: sha256_file(p) for p in files}
    return {"merged_safetensors_sha256": per, "merged_weights_sha256": sha256_text(canon(per)),
            "adapter_sha256": adapter_sha, "tokenizer": "base 488639f1 (byte copies)"}
