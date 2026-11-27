"""MQL-0010 integration tests on the frozen MQL-0009 package (offline; no model, no database).

* With every MQL-0010 flag off, the planner reproduces the registered MQL-0009 D_hybrid plan for all 639 items:
  selected ids, ranks, source DBs and the user-prompt hash (byte-identical prompts).
* Zero-shot arm: prompts identical to the frozen zero-shot prompts.
* Each new flag changes only what it is supposed to change, and the plan records it.
* Protected artifacts are unchanged (approved corpus, MQL-0009 package, retrieval/patterns sources).
"""
import hashlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "kaggle/nl2mql_mql0009_retrieval_control"
sys.path.insert(0, str(ROOT / "scripts/nl2mql_mql0010/src"))
sys.path.insert(1, str(PKG / "src"))
import patterns as PT        # noqa: E402
import mql0008_core as M8    # noqa: E402
import planner as PL         # noqa: E402
import retriever_v2 as R2    # noqa: E402
import structure as S        # noqa: E402

pytestmark = pytest.mark.skipif(not (PKG / "data/plans/plan_D_hybrid.jsonl").exists(), reason="MQL-0009 package absent")


def rl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def fake_tokens(s):
    return len(s) // 4      # budget drops were 0 in MQL-0009; any monotone counter well under the cap reproduces it


@pytest.fixture(scope="module")
def P():
    cfg = json.loads((PKG / "configs/experiment.json").read_text(encoding="utf-8"))
    corpus = rl(PKG / "data/corpus/approved_corpus_v2_1.jsonl")
    tables = {t["db_id"]: t for t in json.loads((PKG / "data/context/tables_subset.json").read_text(encoding="utf-8"))}
    ctx = json.loads((PKG / "data/context/atlas_collections.json").read_text(encoding="utf-8"))
    evals = rl(PKG / "data/eval/eval_inputs.jsonl")
    docs = M8.build_docs(corpus, tables)
    infos = M8.eval_schema_infos(ctx, sorted({e["db_id"] for e in evals}))
    return dict(cfg=cfg, docs=docs, tables=tables, ctx=ctx, evals=evals, infos=infos,
                prompts=rl(PKG / "data/eval/prompts.jsonl"),
                planD={r["id"]: r for r in rl(PKG / "data/plans/plan_D_hybrid.jsonl")},
                planA={r["id"]: r for r in rl(PKG / "data/plans/plan_A_zero_shot.jsonl")})


def plan(P, arm, n=None):
    ev = P["evals"][:n] if n else P["evals"]
    return PL.build_plan(arm, ev, P["prompts"], P["docs"], P["infos"], P["ctx"], P["tables"], P["cfg"]["context"],
                         fake_tokens, P["cfg"]["retrieval_exclusions"])


def test_flags_off_reproduces_registered_hybrid_plan(P):
    rows = plan(P, {"name": "D0", "family": "hybrid"})
    assert len(rows) == 639
    for r in rows:
        ref = P["planD"][r["id"]]
        assert r["selected_ids"] == ref["selected_ids"], r["id"]
        assert r["selected_ranks"] == ref["selected_ranks"], r["id"]
        assert r["selected_db_ids"] == ref["selected_db_ids"]
        assert r["user_prompt_sha256"] == ref["user_prompt_sha256"], r["id"]
        assert r["prompt_sha256"] == ref["prompt_sha256"], r["id"]
        assert [x["corpus_id"] for x in r["excluded_by_rule"]] == [x["corpus_id"] for x in ref["excluded_by_rule"]]


def test_zero_shot_arm_reproduces_frozen_prompts(P):
    rows = plan(P, {"name": "A0", "family": "zero_shot"})
    for r in rows:
        assert r["selected_ids"] == [] and r["empty_context"]
        assert r["prompt_sha256"] == P["planA"][r["id"]]["prompt_sha256"]


def test_diversify_changes_only_selection_and_yields_distinct_signatures(P):
    by = {d["doc_id"]: d for d in P["docs"]}
    rows = plan(P, {"name": "R1", "family": "hybrid", "diversify": "signature"}, n=120)
    changed = 0
    for r in rows:
        sigs = [S.signature(by[i]["patterns"]) for i in r["selected_ids"]]
        assert len(r["selected_ids"]) == 3
        base = [S.signature(by[i]["patterns"]) for i in P["planD"][r["id"]]["selected_ids"]]
        assert len(set(sigs)) >= len(set(base))          # never fewer distinct structures than the baseline
        assert r["selected_ids"][0] == P["planD"][r["id"]]["selected_ids"][0]   # the top-ranked example is kept
        changed += r["selected_ids"] != P["planD"][r["id"]]["selected_ids"]
        assert r["flags"]["diversify"] == "signature"
        assert not set(r["selected_ids"]) & {x["corpus_id"] for x in r["excluded_by_rule"]}
    assert changed > 0


def test_fk_notes_flag_adds_only_genuine_incompatibilities(P):
    rows = plan(P, {"name": "F", "family": "hybrid", "fk_type_notes": "numeric_unified"})
    with_notes = {r["db_id"] for r in rows if r["schema_notes"]}
    assert with_notes == {"phone_market"}       # int vs double (company_employee, machine_repair) and nullable keys are not notes
    r0 = next(r for r in rows if r["db_id"] == "phone_market")
    assert "phone_market__phone_market.Phone_ID (string)" in r0["schema_notes"][0]
    same = next(r for r in rows if r["db_id"] == "chinook_1")
    assert same["user_prompt_sha256"] == P["planD"][same["id"]]["user_prompt_sha256"]


def test_plan_is_deterministic(P):
    a = plan(P, {"name": "X", "family": "hybrid", "diversify": "signature", "fk_type_notes": "numeric_unified"}, n=40)
    b = plan(P, {"name": "X", "family": "hybrid", "diversify": "signature", "fk_type_notes": "numeric_unified"}, n=40)
    assert a == b


def test_unknown_flag_rejected(P):
    with pytest.raises(ValueError):
        PL.arm_settings({"family": "hybrid", "reranker": "llm"})


PROTECTED = {
    "outputs/sql2mql_v2_1/approved_corpus_v2_1/corpus.jsonl": "96552f061afba13792d75240c18001e87e70f4aebf9d0934dfe27c0a7dd199bf",
    "kaggle/nl2mql_mql0009_retrieval_control/package_manifest.json": "2a4f22e0eb526540e70f62f561d0314b1b2faa5c12917196a953d9f98aecface",
}


def test_protected_artifacts_unchanged():
    for rel, h in PROTECTED.items():
        assert hashlib.sha256((ROOT / rel).read_bytes()).hexdigest() == h, rel
    for name in ("patterns.py", "retrieval.py"):
        assert (PKG / "src" / name).read_bytes() == (ROOT / "scripts/rag" / name).read_bytes()


def test_user_text_rebuild_matches_plan_hashes(P):
    by = {d["doc_id"]: d for d in P["docs"]}
    fps = {p["id"]: p for p in P["prompts"]}
    for arm in ({"name": "D0", "family": "hybrid"},
                {"name": "EF", "family": "hybrid", "diversify": "signature", "fk_type_notes": "numeric_unified"}):
        for r in plan(P, arm, n=60) + [x for x in plan(P, arm) if x["db_id"] == "phone_market"][:5]:
            system, user = PL.user_text_for(r, fps[r["id"]], by)
            assert system == fps[r["id"]]["system"]


def test_registered_config_is_consistent():
    cfg = json.loads((ROOT / "scripts/nl2mql_mql0010/configs/mql0010_experiment.json").read_text(encoding="utf-8"))
    names = {a["name"] for a in cfg["arms"]}
    for a in cfg["arms"]:
        PL.arm_settings(a["flags"])                                   # every registered flag is known
        assert a.get("source", a.get("base", a["name"])) in names
    plan_f = ROOT / "outputs/nl2mql_mql0010/preflight/analysis_plan.json"
    assert hashlib.sha256(plan_f.read_bytes()).hexdigest() == cfg["analysis_plan_sha256"]
    split_f = ROOT / cfg["split"]["path"]
    assert hashlib.sha256(split_f.read_bytes()).hexdigest() == cfg["split"]["sha256"]
    # Source hashes are governed by outputs/nl2mql_mql0010/preflight/registration_index.json since amendment A4
    # (tests/test_mql0010_registration.py); the config's original source_sha256 is superseded, not rewritten.
