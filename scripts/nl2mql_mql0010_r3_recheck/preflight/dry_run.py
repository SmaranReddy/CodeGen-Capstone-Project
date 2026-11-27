"""MQL-0014 MOCKED end-to-end dry run (no model, no Kaggle, no Atlas, no network). Run in venv_rag:
  ~/codegen_data/venv_rag/Scripts/python -B scripts/nl2mql_mql0010_r3_recheck/preflight/dry_run.py

Real: registration writer, package build (test_full ids, disjointness, context from test_database SQLite, zero-shot
prompts, leakage audit, R0 + R3 plans with the pinned tokenizer, the read-only Qdrant collection and the pinned
embedder), the notebook cells (incl. a kill + resume across two sessions), export verification, the gold reference
from SQLite, the scoring adapter and the registered analysis.
Fake: the generator (deterministic text from a prompt hash) and the Atlas evaluator (deterministic outcome from a hash;
never reads gold). Every path is redirected to outputs/nl2mql_mql0010_r3_execution_recheck/dryrun/; the real registration / package /
contract locations are not touched. Mock accuracies are meaningless.
"""
import hashlib
import json
import re
import shutil
import sys
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE / "src"))
sys.path.insert(1, str(HERE))
import recheck_common as CM    # noqa: E402

DRY = CM.OUT / "dryrun"
FAKE_GPU = {"checked": True, "visible_devices": 1, "device0": "Tesla T4", "compute_capability": "7.5", "total_gib": 14.56,
            "bf16_supported_incl_emulation": True, "bf16_native": False, "free_gib": 14.4, "cuda": "fake", "torch": "fake"}


def h(s):
    return int(hashlib.sha256(s.encode("utf-8")).hexdigest()[:8], 16)


def fake_factory(cfg):
    def run_batch(batch):
        out = []
        for p in batch:
            user = p["messages"][0]["content"]
            colls = [c for c in re.findall(r"\b(\w+__\w+)\b", user) if c.startswith(p["db_id"] + "__")]
            coll = colls[0] if colls else "x__y"
            if len(p["messages"]) == 1:
                # depends on the question only (not the examples) so shared/identical behaviour is realistic
                q = user.rsplit("Question:", 1)[-1]
                bad = h(p["db_id"] + q) % 100 < 25
                stage = {"$match": {"__missing_field__": 1}} if bad else {"$match": {}}
                txt = json.dumps({"collection": coll, "pipeline": [stage, {"$project": {"_id": 0}}]})
            else:
                ok = h(user + p["messages"][2]["content"]) % 100 < 50
                stage = {"$match": {}} if ok else {"$match": {"__missing_field__": 2}}
                txt = json.dumps({"collection": coll, "pipeline": [stage, {"$limit": 5}, {"$project": {"_id": 0}}]})
            out.append((txt, {"latency_s": 0.0, "input_tokens": len(user) // 4, "output_tokens": len(txt) // 4,
                              "finish_reason": "stop"}, None))
        return out
    return run_batch, {"dtype": "bfloat16", "gpu": "Tesla T4", "torch": "fake", "transformers": "fake"}, (lambda e: False), (lambda: None)


def run_notebook(nb_path, pkg, out, session, resume=None, stop=None):
    nb = json.loads(Path(nb_path).read_text(encoding="utf-8"))
    g = {"__name__": "__main__"}
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        exec("".join(cell["source"]), g)
        if cell["metadata"]["tags"] == ["config"]:
            g.update(SESSION=session, PKG_DIR=str(pkg), OUT_DIR=str(out), OUTPUT_ZIP=str(out) + ".zip",
                     INPUT_ROOT=str(out) + "_input", RESUME_FROM=resume, GENERATOR_FACTORY=fake_factory, FAKE_GPU=FAKE_GPU,
                     STOP_AFTER_BATCHES=stop)
    return g


def fake_evaluate(slots, refs, mapping, mdb, K, out_dir, fp):
    ref = {r["id"]: r for r in refs}
    res = {}
    for s in slots:
        k = h(s["raw_text"] + s["item_id"]) % 100
        o = "correct" if k < 55 else "incorrect" if k < 85 else "invalid" if k < 90 else "timeout" if k < 93 else "failed"
        res[s["id"]] = {"id": s["id"], "outcome": o, "failure_source": "model" if o == "failed" else None,
                        "detail": None, "hardness": ref[s["item_id"]]["hardness"],
                        "reference_status": ref[s["item_id"]]["reference_status"], "mock": True}
    return res


class FakeClient:
    def __getitem__(self, k):
        return None

    def close(self):
        pass


def main():
    if DRY.exists():
        shutil.rmtree(DRY)
    # redirect every output location
    CM.REG_DIR = DRY / "registration"
    CM.REGISTRATION = CM.REG_DIR / "mql0014_registration.json"
    CM.WORK = DRY / "work"
    CM.PKG_OUT = DRY / "kaggle" / CM.PACKAGE
    CM.ZIP_OUT = DRY / "kaggle" / f"{CM.PACKAGE}_bundle.zip"
    import register as RG
    RG.main()
    import build_package as B
    cfg, man = B.build(overwrite=False)
    nb = CM.PKG_OUT / f"{CM.PACKAGE}.ipynb"
    s1 = DRY / "kaggle_out_s1" / "mql0014_outputs"
    g1 = run_notebook(nb, CM.PKG_OUT, s1, 1, stop=3)
    assert g1["ST"]["status"] == "partial", g1["ST"]["status"]
    s2 = DRY / "kaggle_out_s2" / "mql0014_outputs"
    g2 = run_notebook(nb, CM.PKG_OUT, s2, 2, resume=str(s1))
    assert g2["ST"]["status"] == "complete", (g2["ST"]["status"], g2["MAN"]["passes"])
    import recheck_scoring as SCO
    ana, rman, run_dir = SCO.cmd_score([str(s2) + ".zip"], "dry_run", pkg=CM.PKG_OUT, client_factory=lambda: (FakeClient(), str),
                                       preflight=lambda c, m, **k: {"ok": True, "mock": True}, evaluate=fake_evaluate,
                                       dest_root=DRY, log=lambda *a: None)
    pc = json.loads((s2 / "passes/retry/pass_config.json").read_text(encoding="utf-8"))
    rep = {"MOCK": "fake generator + fake evaluator; numbers are meaningless",
           "package": {"files": len(man["files"]), "partition": {k: v for k, v in cfg["partition"].items() if k != "shared_ids"},
                       "budget_drops": cfg["budget_drops"], "scoring_contract": cfg["scoring_contract"]},
           "distinct_retry_prompts": len(pc["retry_ids"]), "retried_items": {a: len(v) for a, v in pc["retried_items"].items()},
           "reproducibility": ana.get("reproducibility"),
           "session1": {"status": g1["ST"]["status"], "passes": g1["MAN"]["passes"]},
           "session2": {"status": g2["ST"]["status"], "verify_exports": g2["MAN"]["verify_exports"]},
           "reference": rman["reference"], "unique_evaluations": rman["unique_evaluations"],
           "reference_equals_mql0010": rman["reference"]["equals_mql0010_reference"], "mock_accuracy": ana["accuracy"], "primary_keys": sorted(ana["primary"]),
           "by_hardness_groups": list(ana["by_hardness"]), "n_structure_groups": len(ana["by_structure"])}
    CM.write_json(DRY / "dry_run_report.json", rep)
    print(json.dumps(rep, indent=1)[:4000])


if __name__ == "__main__":
    main()
