"""MQL-0015 MOCKED end-to-end dry run (no model, no GPU, no Kaggle, no Atlas, no network).

Real: package build (into the dry-run folder), the notebook cells (session 1 killed after 2 batches, resumed; session 2
= FT passes after a second resume), checkpoint/resume, the 2-attempt generation-failure policy, retry dedup per model,
export + verify_exports, the scorer's verify, the gold reference computed from the test_database SQLite files, the
unique-slot sharing, infrastructure re-execution, and the registered analysis incl. every sensitivity line.
Fake: the generator (deterministic text from a prompt hash, with injected unparseable / schema-invalid answers, a CUDA
OOM batch, transient and permanent generation failures, failing FT retries), the FT merge, and the Atlas evaluator
(deterministic outcome from a hash, every taxonomy category; never reads gold). Mock accuracies are meaningless.

  python -I -B scripts/nl2mql_mql0015/preflight/dry_run.py
"""
import hashlib
import json
import re
import shutil
import sys
from collections import Counter
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE / "src"))
sys.path.insert(1, str(HERE))
import mql0015_common as CM    # noqa: E402

DRY = CM.OUT / "dryrun"
FAKE_GPU = {"checked": True, "visible_devices": 1, "device0": "Tesla T4", "compute_capability": "7.5", "total_gib": 14.56,
            "bf16_supported_incl_emulation": True, "bf16_native": False, "free_gib": 14.4, "cuda": "fake", "torch": "fake"}
FAKE_WEIGHTS = {"merged_safetensors_sha256": {"model.safetensors": "f" * 64}, "merged_weights_sha256": "e" * 64,
                "adapter_sha256": CM.ADAPTER_SHA256, "tokenizer": "base 488639f1", "mock": True}


def h(s):
    return int(hashlib.sha256(s.encode("utf-8")).hexdigest()[:8], 16)


class FakeOOM(Exception):
    pass


CALLS = Counter()          # per (model, prompt sha) call count: drives transient vs permanent generation failures


def fake_factory(cfg, model_key, path):
    def answer(p):
        user = p["messages"][0]["content"]
        colls = [c for c in re.findall(r"\b(\w+__\w+)\b", user) if c.startswith(p["db_id"] + "__")]
        coll = colls[0] if colls else "x__y"
        q = user.rsplit("Question:", 1)[-1]
        if len(p["messages"]) == 1:
            k = h(p["db_id"] + q) % 100                      # question-only -> A0/R3 often byte-identical (shared slots)
            if model_key == "ft" and k % 3 == 0:
                k = (k + 41) % 100                           # FT differs on a third of the items
            if k < 6:
                return "this is not json"                    # unparseable -> format retry
            stage = {"$match": {"__missing_field__": 1}} if k < 25 else {"$match": {}}
            extra = [{"$limit": 3}] if model_key == "ft" and k % 2 else []
            return json.dumps({"collection": coll, "pipeline": [stage, *extra, {"$project": {"_id": 0}}]})
        ok = h(user + p["messages"][2]["content"]) % 100 < 50
        stage = {"$match": {}} if ok else {"$match": {"__missing_field__": 2}}
        return json.dumps({"collection": coll, "pipeline": [stage, {"$limit": 5}, {"$project": {"_id": 0}}]})

    def run_batch(batch):
        if len(batch) > 1 and any(h("oom" + p["prompt_sha256"]) % 997 == 0 for p in batch):
            raise FakeOOM("CUDA out of memory (fake)")
        out = []
        for p in batch:
            key = (model_key, p["prompt_sha256"])
            CALLS[key] += 1
            g = h("gen" + model_key + p["prompt_sha256"]) % 1000
            retry_turn = len(p["messages"]) > 1
            if (not retry_turn and g < 4) or (retry_turn and model_key == "ft" and g < 40):
                out.append((None, {"latency_s": None}, "RuntimeError: permanent fake failure"))
                continue
            if not retry_turn and g < 12 and CALLS[key] == 1:
                out.append((None, {"latency_s": None}, "RuntimeError: transient fake failure"))
                continue
            txt = answer(p)
            out.append((txt, {"latency_s": 0.0, "input_tokens": len(p["messages"][0]["content"]) // 4,
                              "output_tokens": len(txt) // 4, "finish_reason": "stop"}, None))
        return out
    return run_batch, {"dtype": "bfloat16", "gpu": "Tesla T4", "torch": "fake", "transformers": "fake"}, \
        (lambda e: isinstance(e, FakeOOM)), (lambda: None)


def fake_merger(base_path, adapter_dir, merged_dir):
    assert CM.sha256_file(Path(adapter_dir) / "adapter_model.safetensors") == CM.ADAPTER_SHA256
    return dict(FAKE_WEIGHTS)


def run_notebook(nb_path, pkg, out, session, resume=None, stop=None):
    nb = json.loads(Path(nb_path).read_text(encoding="utf-8"))
    g = {"__name__": "__main__"}
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        exec("".join(cell["source"]), g)
        if cell["metadata"]["tags"] == ["config"]:
            g.update(SESSION=session, PKG_DIR=str(pkg), OUT_DIR=str(out), OUTPUT_ZIP=str(out) + ".zip",
                     INPUT_ROOT=str(out) + "_input", RESUME_FROM=resume, GENERATOR_FACTORY=fake_factory, FT_MERGER=fake_merger,
                     FAKE_GPU=FAKE_GPU, STOP_AFTER_BATCHES=stop, MERGED_DIR=str(out) + "_merged")
    return g


INFRA_ONCE, INFRA_ALWAYS = set(), set()


def fake_evaluate(slots, refs, mapping, mdb, K, out_dir, fp):
    ref = {r["id"]: r for r in refs}
    res = {}
    for s in slots:
        base = {"id": s["id"], "hardness": ref[s["item_id"]]["hardness"], "reference_status": ref[s["item_id"]]["reference_status"],
                "detail": None, "failure_source": None, "mock": True}
        if s["status"] != "ok":
            res[s["id"]] = {**base, "outcome": "failed", "failure_source": "infrastructure", "detail": "generation_error"}
            continue
        k = h(s["raw_text"] + s["item_id"]) % 1000
        if k < 30 and "-infra" not in fp:
            (INFRA_ALWAYS if k < 4 else INFRA_ONCE).add(s["id"])
        if s["id"] in INFRA_ALWAYS or (s["id"] in INFRA_ONCE and "-infra" not in fp):
            res[s["id"]] = {**base, "outcome": "failed", "failure_source": "infrastructure", "detail": "atlas_unavailable"}
            continue
        k = h("o" + s["raw_text"] + s["item_id"]) % 100
        o, src, det = (("correct", None, None) if k < 50 else ("incorrect", "model", None) if k < 78 else
                       ("invalid", "model", "unsafe_mql") if k < 85 else ("timeout", "model", "timeout") if k < 89 else
                       ("failed", "model", "execution_error") if k < 97 else ("not_comparable", None, None))
        res[s["id"]] = {**base, "outcome": o, "failure_source": src, "detail": det}
    return res


class FakeClient:
    def __getitem__(self, k):
        assert k == CM.ATLAS_DB, f"fake client: access to {k} refused"
        return None

    def close(self):
        pass


def main():
    if DRY.exists():
        shutil.rmtree(DRY)
    import build_mql0015_package as B
    pkg, zipf, contract = DRY / "kaggle" / CM.PACKAGE, DRY / "kaggle" / f"{CM.PACKAGE}_bundle.zip", DRY / "scoring_contract/atlas_mapping.json"
    cfg, man, brep = B.build(pkg, zipf, contract, overwrite=False)
    nb = pkg / f"{CM.PACKAGE}.ipynb"
    s1 = DRY / "kaggle_out_s1" / "mql0015_outputs"
    g1 = run_notebook(nb, pkg, s1, 1, stop=2)
    assert g1["ST"]["status"] == "partial", g1["ST"]["status"]
    s1b = DRY / "kaggle_out_s1b" / "mql0015_outputs"
    g1b = run_notebook(nb, pkg, s1b, 1, resume=str(s1))
    assert g1b["ST"]["status"] == "partial" and all(g1b["MAN"]["passes"][p]["status"] == "complete"
                                                    for p in ("first:A0", "first:R3", "retry:base")), g1b["MAN"]["passes"]
    s2 = DRY / "kaggle_out_s2" / "mql0015_outputs"
    g2 = run_notebook(nb, pkg, s2, 2, resume=str(s1b))
    assert g2["ST"]["status"] == "complete", (g2["ST"]["status"], g2["MAN"]["passes"], g2["ST"]["problems"])
    ver = g2["MAN"]["verify_exports"]
    import mql0015_scoring as SCO
    preds = {a: CM.read_jsonl(s2 / "predictions" / a / "predictions.jsonl") for a in CM.ARMS}
    slots, _ = SCO.unique_slots(preds, [p["id"] for p in preds["A0"]])
    ana, rman, run_dir = SCO.cmd_score([str(s2) + ".zip"], "dry_run", pkg=pkg, client_factory=lambda: (FakeClient(), str),
                                       preflight=lambda c, m, **k: {"ok": True, "mock": True, "atlas_db": m["atlas_db"]},
                                       evaluate=fake_evaluate, dest_root=DRY, contract_path=contract, log=lambda *a: None)
    infra = json.loads((run_dir / "scoring/infrastructure.json").read_text(encoding="utf-8"))
    rep = {"MOCK": "fake generator + fake merge + fake Atlas evaluator; numbers are meaningless",
           "package": {"files": len(man["files"]), "zip_sha256": brep["zip_sha256"], "gold_hits": brep["gold_scan"]["hits"],
                       "manifest_sha256": brep["package_manifest_sha256"]},
           "session1_killed": {"status": g1["ST"]["status"], "passes": g1["MAN"]["passes"]},
           "session1_resumed": {"status": g1b["ST"]["status"], "seeded": len(g1b["RESUMED"]["seeded"]), "passes": g1b["MAN"]["passes"],
                                "oom_events": g1b["OOM_EVENTS"]},
           "session2_ft": {"status": g2["ST"]["status"], "seeded": len(g2["RESUMED"]["seeded"]), "verify_exports": ver,
                           "weights": g2["WEIGHTS"], "oom_events": g2["OOM_EVENTS"]},
           "scoring": {"unique_slots": rman["unique_evaluations"], "rows": 6 * 1000,
                       "shared_rows": 6 * 1000 - rman["unique_evaluations"], "reference": rman["reference"],
                       "infrastructure": infra},
           "mock_accuracy": {a: v["correct"] for a, v in ana["accuracy"].items()},
           "taxonomy": ana["taxonomy"], "primary": {k: ana["primary"][k] for k in ("diff_pp", "ci95_pp", "cluster_ci95_pp", "p_mcnemar", "a_only", "b_only")},
           "secondary": [{k: s[k] for k in ("a", "b", "p_mcnemar", "p_holm", "reject")} for s in ana["secondary"]],
           "subset_sizes": ana["subset_sizes"],
           "sensitivity_n": {k: v["n"] for k, v in ana["sensitivity"].items() if isinstance(v, dict) and "n" in v},
           "recovery": ana["recovery"], "report": str(run_dir / "scoring/results_report.md")}
    CM.write_json(DRY / "dry_run_report.json", rep)
    print(json.dumps(rep, indent=1)[:9000])
    return rep


if __name__ == "__main__":
    main()
