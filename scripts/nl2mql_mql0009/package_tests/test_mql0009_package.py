"""Self-checks of the MQL-0009 Kaggle package (run inside the package; needs only the package and pytest).

    python -m pytest tests -q                                   # integrity, isolation, design checks
    MQL0009_TOKENIZER_DIR=<dir> python -m pytest tests -q       # + full plan re-derivation with the pinned tokenizer
"""
import json
import os
import re
import sys
from pathlib import Path

import pytest

PKG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG / "src"))
import mql0009_core as C  # noqa: E402

P = C.load_package(PKG)
CFG = P["cfg"]


def test_manifest_and_corpus_identity():
    assert P["manifest"]["package"] == C.PACKAGE
    assert C.sha256_file(PKG / C.CORPUS_FILE) == C.EXPECTED_CORPUS_SHA256
    assert len(P["view_rows"]) == CFG["corpus"]["eligible_view"]["n"]


def test_shared_sources_byte_identical_to_mql0008():
    for f, h in CFG["frozen_reference"]["shared_source_sha256"].items():
        assert C.sha256_file(PKG / "src" / f) == h, f


def test_design_k3_seeds_fixed():
    runs = C.check_arm_design(CFG)
    assert [r["name"] for r in runs] == ["A_zero_shot", "B_random_s1", "B_random_s2", "B_random_s3", "B_random_s4",
                                         "B_random_s5", "C_fixed", "D_hybrid"]
    docs = C.M8.build_docs(P["view_rows"], P["tables"])
    by = {d["doc_id"]: d for d in docs}
    for r in runs:
        plan = C.read_jsonl(PKG / f"data/plans/plan_{r['name']}.jsonl")
        assert [p["id"] for p in plan] == P["ids"]
        C.plan_checks(CFG, r, plan, P["evals"], by)
        assert C.plan_sha(plan) == CFG["plans_sha256"][r["name"]]


def test_prompts_rebuild_from_plans_and_are_equivalent():
    docs = C.M8.build_docs(P["view_rows"], P["tables"])
    by = {d["doc_id"]: d for d in docs}
    prompts = {r["name"]: C.prompts_for_run(C.read_jsonl(PKG / f"data/plans/plan_{r['name']}.jsonl"), P["prompts"], by)
               for r in P["runs"]}
    C.check_prompt_equivalence(prompts, P["prompts"])


def test_no_gold_labels_or_secrets_in_package():
    for e in P["evals"]:
        assert set(e) == {"id", "db_id", "question"}
    secret = re.compile(r"mongodb(\+srv)?://|MONGODB_URI\s*=|ATLAS_SEED_URI\s*=|BEGIN (RSA|OPENSSH) PRIVATE KEY|hf_[A-Za-z0-9]{30,}")
    for p in PKG.rglob("*"):
        if p.is_file():
            assert not p.name.startswith(".env"), p
            if p.suffix in (".py", ".json", ".jsonl", ".md", ".txt", ".ipynb"):
                txt = p.read_text(encoding="utf-8")
                assert not secret.search(txt), p
    for rel in ("data/eval/eval_inputs.jsonl", "data/eval/prompts.jsonl", "data/leakage_report.json"):
        txt = (PKG / rel).read_text(encoding="utf-8")
        for key in ('"sql"', '"gold', '"expected_output"', '"result_sha256"', '"hardness"'):
            assert key not in txt, (rel, key)
    for f in (PKG / "data/plans").glob("*.jsonl"):
        assert '"sql"' not in f.read_text(encoding="utf-8")


def test_notebook_cells_tagged():
    nb = json.loads((PKG / f"{C.PACKAGE}.ipynb").read_text(encoding="utf-8"))
    tags = [c["metadata"]["tags"][0] for c in nb["cells"] if c["cell_type"] == "code"]
    assert tags == ["config", "package", "env", "model", "plan", "preflight", "generate", "export"]


@pytest.mark.skipif(not os.environ.get("MQL0009_TOKENIZER_DIR"), reason="set MQL0009_TOKENIZER_DIR for full re-derivation")
def test_full_plan_rederivation_with_tokenizer():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(os.environ["MQL0009_TOKENIZER_DIR"])
    C.derive_plans(P, lambda s: len(tok.encode(s, add_special_tokens=False)))
