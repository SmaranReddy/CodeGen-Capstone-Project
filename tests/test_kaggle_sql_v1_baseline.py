"""Kaggle zero-shot SQL baseline (Spider dev): bundle integrity, official-evaluator wiring, and the notebook's
non-GPU cells driven by a fake generator (failures, interruption, resume). No model, no GPU, no network."""
import json
import sys
import types
from pathlib import Path

import nbformat
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/sql_v1"))
sys.dont_write_bytecode = True
import prepare_kaggle_dev_bundle as PB  # noqa: E402
import spider_zs_sql as Z  # noqa: E402

BUNDLE = ROOT / "kaggle/spider_sql_v1_dev_bundle_v1"
NB = ROOT / "notebooks/kaggle/spider_sql_v1_zero_shot_3b.ipynb"
needs_bundle = pytest.mark.skipif(not BUNDLE.exists(), reason="bundle not built")


# ---------------------------------------------------------------- pure functions
@pytest.mark.parametrize("text,sql,how", [
    ("```sql\nSELECT a\nFROM t;\n```", "SELECT a FROM t", "fenced"),
    ("SELECT a FROM t; -- done", "SELECT a FROM t", "raw"),
    ("SQL: SELECT 'x;y' FROM t;", "SELECT 'x;y' FROM t", "raw"),
    ("```sqlite\nSELECT\tb FROM t", "SELECT b FROM t", "unclosed_fence"),
    ("", "", "empty"),
    ("```sql\n```", "", "empty"),
])
def test_extract_sql(text, sql, how):
    assert Z.extract_sql(text) == (sql, how)


def test_prompt_never_sees_gold_fields():
    with pytest.raises(RuntimeError):
        Z.build_prompts([{"id": "a", "db_id": "d", "question": "q", "sql": "SELECT 1"}], {"d": []})
    rec = {"id": "a", "db_id": "d", "question": "q", "sql": "SELECT x FROM t"}
    prompts = Z.build_prompts(Z.prompt_view([rec]), {"d": ["CREATE TABLE t (x int)"]})
    assert Z.check_no_gold_leak(prompts, [rec])
    prompts["a"][1]["content"] += " select  X from T"
    with pytest.raises(RuntimeError):
        Z.check_no_gold_leak(prompts, [rec])


def test_prediction_log_resume_rules(tmp_path):
    p = tmp_path / "log.jsonl"
    log = Z.PredictionLog(p, ["a", "b"], "fp")
    base = {"db_id": "d", "fingerprint": "fp"}
    log.append([{**base, "id": "a", "status": "error"}])
    log.append([{**base, "id": "a", "status": "ok"}])          # retry after error is allowed
    with pytest.raises(RuntimeError):
        log.append([{**base, "id": "a", "status": "ok"}])      # duplicate of a final record is not
    with open(p, "a", encoding="utf-8") as f:
        f.write('{"id": "b", "sta')                            # crash mid-write
    again = Z.PredictionLog(p, ["a", "b"], "fp")
    assert again.warnings and again.todo() == ["b"] and again.records["a"]["status"] == "ok"
    with pytest.raises(RuntimeError):
        Z.PredictionLog(p, ["a", "b"], "other-fingerprint")


# ---------------------------------------------------------------- bundle
@needs_bundle
def test_bundle_verifies_and_holds_only_allowlisted_files():
    man = PB.verify_dir(BUNDLE)
    Z.verify_bundle(BUNDLE)
    names = set(man["files"])
    assert not any(PB.DENY.search(n) for n in names)
    assert {n for n in names if not n.startswith("database/")} == {
        "ATTRIBUTION.md", "data/dev_full.jsonl", "data/schemas_ddl_dev.json", "data/tables_dev.json",
        "code/spider_zs_sql.py", "code/official_eval/evaluation.py", "code/official_eval/process_sql.py",
        "code/official_eval/LICENSE", "code/official_eval/SOURCE_COMMIT.txt"}
    assert len([n for n in names if n.startswith("database/")]) == 20


@needs_bundle
def test_bundle_and_zip_hold_no_python_caches():
    import zipfile
    assert not [p for p in BUNDLE.rglob("*") if p.name == "__pycache__" or p.suffix in (".pyc", ".pyo")]
    with zipfile.ZipFile(BUNDLE.with_suffix(".zip")) as z:
        assert not [n for n in z.namelist() if "__pycache__" in n or n.endswith((".pyc", ".pyo"))]


def test_cache_files_are_rejected_by_verify_and_removed_by_purge(tmp_path):
    """verify_dir must keep rejecting cache files (not ignore them); purge_python_caches removes exactly those."""
    b = tmp_path / "b"
    (b / "code/official_eval/__pycache__").mkdir(parents=True)
    (b / "bundle_manifest.json").write_text(json.dumps({"files": {}}), encoding="utf-8")
    (b / "code/official_eval/evaluation.py").write_text("x = 1\n", encoding="utf-8")
    (b / "code/official_eval/__pycache__/evaluation.cpython-310.pyc").write_bytes(b"\0")
    (b / "code/stray.pyc").write_bytes(b"\0")
    with pytest.raises(PB.BundleError, match="tree != manifest"):
        PB.verify_dir(b)
    assert PB.purge_python_caches(b) == ["code/official_eval/__pycache__/evaluation.cpython-310.pyc", "code/stray.pyc"]
    assert sorted(p.relative_to(b).as_posix() for p in b.rglob("*") if p.is_file()) == \
        ["bundle_manifest.json", "code/official_eval/evaluation.py"]
    with pytest.raises(PB.BundleError):
        PB.purge_python_caches(tmp_path)          # refuses a directory that is not a bundle


def test_tests_do_not_write_bytecode():
    assert sys.dont_write_bytecode


@needs_bundle
def test_bundle_dev_records_are_byte_identical_and_ordered():
    src = json.loads((ROOT / "data/processed/spider_sql_v1/manifest.json").read_text(encoding="utf-8"))
    assert Z.sha256_file(BUNDLE / "data/dev_full.jsonl") == src["files_sha256"]["dev_full.jsonl"]
    recs = Z.load_dev(BUNDLE)
    assert [r["id"] for r in recs] == [f"spider_dev_{i:04d}" for i in range(1034)]
    for d in {r["db_id"] for r in recs}:
        assert Z.sha256_file(BUNDLE / f"database/{d}/{d}.sqlite") == \
            Z.sha256_file(ROOT / f"data/raw/spider1/spider_data/database/{d}/{d}.sqlite")


@needs_bundle
def test_bundle_code_is_current():
    assert Z.sha256_file(BUNDLE / "code/spider_zs_sql.py") == Z.sha256_file(ROOT / "scripts/sql_v1/spider_zs_sql.py")


@needs_bundle
def test_zip_matches_bundle():
    import zipfile
    zp = BUNDLE.with_suffix(".zip")
    man = json.loads((BUNDLE / "bundle_manifest.json").read_text(encoding="utf-8"))
    with zipfile.ZipFile(zp) as z:
        assert set(z.namelist()) == {f"{BUNDLE.name}/{n}" for n in list(man["files"]) + ["bundle_manifest.json"]}


@needs_bundle
def test_gold_as_prediction_scores_perfectly(tmp_path):
    recs = Z.load_dev(BUNDLE)
    preds = [{"id": r["id"], "db_id": r["db_id"], "pred_sql": r["sql"], "status": "ok", "batch_key": "s:0",
              "batch_wall_s": 1.0, "latency_s": 0.001, "session_id": "s", "fingerprint": "x"} for r in recs]
    results, cli, _ = Z.score_all(BUNDLE, recs, preds, tmp_path)
    s = Z.summarize(preds, results, cli, {})
    for name, m in s["metrics"].items():
        assert (m["numerator"], m["denominator"]) == (1034, 1034), name
    assert s["evaluation"]["official_cli_crosscheck"]["agree_to_3dp"]
    assert s["evaluation"]["evaluator_errors"] == 0
    assert all(r["hardness"] == r["hardness_record"] for r in results)


# ---------------------------------------------------------------- notebook, fake generator
def _cells():
    nb = nbformat.read(str(NB), as_version=4)
    nbformat.validate(nb)
    return {t: c.source for c in nb.cells if c.cell_type == "code" for t in c.metadata.get("tags", [])}


def test_notebook_structure():
    cells = _cells()
    for tag in ("config", "gpu", "bundle", "prompts", "preflight", "model", "smoke", "generate", "finalize", "evaluate", "summary"):
        assert tag in cells, tag
    src = "\n".join(cells.values())
    assert "do_sample=False" in src and "MODEL_REVISION" in src and "/kaggle/working/" in src
    for forbidden in ("test_full", "test_clean"):
        assert forbidden not in src.replace('"test_full", "test_clean"', "")
    gen_cells = cells["prompts"] + cells["model"] + cells["smoke"] + cells["generate"]
    assert '["sql"]' not in gen_cells and "execution" not in gen_cells  # gold never touched before scoring


class FakeGen:
    """Gold SQL for most items (harness-only lookup), plus injected failures."""

    def __init__(self, records, prompts, interrupt_after=None):
        self.by_prompt = {Z.canon(prompts[r["id"]]): r for r in records}
        self.calls, self.interrupt_after, self.fail_once = 0, interrupt_after, True
        self.wta_id = next(r["id"] for r in records if r["db_id"] == "wta_1")

    def __call__(self, batch):
        self.calls += 1
        if self.interrupt_after and self.calls > self.interrupt_after:
            raise KeyboardInterrupt("simulated session loss")
        out = []
        for m in batch:
            r = self.by_prompt[Z.canon(m)]
            n, sql = int(r["id"][-4:]), r["sql"]
            o = {"text": f"```sql\n{sql}\n```" if n % 2 else sql + ";", "finish": "eos", "n_new_tokens": 20,
                 "n_prompt_tokens": 300}
            if n == 3:
                o["text"] = ""
            elif n == 4 and self.fail_once:
                raise RuntimeError("CUDA out of memory (simulated)")
            elif n == 5:
                o.update(text="SELECT name FROM", finish="time")
            elif n == 6:
                o.update(text="SELECT Name , Country FROM singer WHERE", finish="length")
            elif n == 7:
                o["text"] = "DROP TABLE singer"
            elif n == 8:
                o["text"] = "Here you go:\n```sql\n" + sql.replace(" ", "\n\t", 1) + " ;\n```\nDone."
            elif n == 9:
                o["text"] = "I cannot answer that."
            elif r["id"] == self.wta_id:
                o["text"] = "SELECT count(*) FROM players a, players b, players c"
            out.append(o)
        return out


@needs_bundle
def test_notebook_end_to_end_with_fake_generator(tmp_path):
    cells = _cells()
    ns = {}
    exec(cells["config"], ns)
    ns.update(BUNDLE_DIR=str(BUNDLE), OUT_DIR=str(tmp_path / "out"), EXEC_TIMEOUT_S=8.0, BATCH_SIZE=8)  # slowest gold ~1.2 s
    fake_torch = types.SimpleNamespace(cuda=types.SimpleNamespace(empty_cache=lambda: None))
    ns.update(DTYPE_NAME="float16", GPU_INFO=[], VERSIONS={}, MODEL_PATH="fake", EFFECTIVE_GEN_CONFIG={},
              torch=fake_torch, gc=__import__("gc"), on_gen_error=lambda e: None)
    sys.modules.pop("spider_zs_sql", None)     # the cell must import the bundle's copy, not scripts/sql_v1's
    exec(cells["bundle"], ns)
    exec(cells["prompts"], ns)
    exec(cells["preflight"], ns)
    records = Z.load_dev(BUNDLE)
    gen = FakeGen(records, ns["PROMPTS"], interrupt_after=20)   # batch 0 fails (1 call + 8 retries), then 11 batches
    ns["hf_generate"] = gen
    with pytest.raises(KeyboardInterrupt):
        exec(cells["generate"], ns)
    log_path = tmp_path / "out/checkpoints/predictions.partial.jsonl"
    n_before = len(log_path.read_text(encoding="utf-8").splitlines())
    assert n_before == 8 + 11 * 8
    with pytest.raises(SystemExit):                       # incomplete run is refused, not summarised
        exec(cells["finalize"], ns)
    assert json.loads((tmp_path / "out/RUN_STATUS.json").read_text())["status"] == "incomplete"
    with open(log_path, "a", encoding="utf-8") as f:
        f.write('{"id": "spider_dev_09')                   # torn last line

    # changing the configuration must not mix runs
    ns2 = dict(ns, MAX_NEW_TOKENS=512)
    with pytest.raises(SystemExit):
        exec(cells["prompts"], ns2)

    # resume: new session, same config; the errored example (id 4) is retried and now succeeds
    exec(cells["prompts"], ns)
    assert ns["LOG"].warnings
    assert ns["LOG"].records["spider_dev_0004"]["status"] == "error"
    gen2 = FakeGen(records, ns["PROMPTS"])
    gen2.fail_once = False
    ns["hf_generate"] = gen2
    exec(cells["generate"], ns)
    exec(cells["finalize"], ns)
    exec(cells["evaluate"], ns)
    exec(cells["summary"], ns)

    out = tmp_path / "out"
    preds = Z.read_jsonl(out / "predictions.jsonl")
    res = {r["id"]: r for r in Z.read_jsonl(out / "evaluation_results.jsonl")}
    assert [p["id"] for p in preds] == [r["id"] for r in records]
    st = {p["id"]: p["status"] for p in preds}
    assert st["spider_dev_0003"] == "empty" and st["spider_dev_0005"] == "timeout"
    assert st["spider_dev_0006"] == "truncated" and st["spider_dev_0004"] == "ok"
    assert len({p["session_id"] for p in preds}) == 2
    assert res["spider_dev_0003"]["eval_sql"] == Z.EMPTY_SENTINEL
    assert res["spider_dev_0007"]["exec_status"] == "error"          # read-only connection
    assert res["spider_dev_0008"]["official_exec"] == 1 and res["spider_dev_0008"]["official_exact"] == 1
    assert res["spider_dev_0009"]["official_parse_ok"] is False
    assert res[gen.wta_id]["exec_status"] in ("timeout", "row_cap") and res[gen.wta_id]["official_exec"] == 0
    wrong = {"spider_dev_0003", "spider_dev_0005", "spider_dev_0006", "spider_dev_0007", "spider_dev_0009", gen.wta_id}
    assert {i for i, r in res.items() if not r["official_exec"]} == wrong
    s = json.loads((out / "baseline_summary.json").read_text(encoding="utf-8"))
    assert s["complete"] and s["metrics"]["official_execution_accuracy"]["numerator"] == 1034 - len(wrong)
    assert all(m["denominator"] == 1034 for m in s["metrics"].values())
    assert s["generation"]["timeouts"] == 1 and s["generation"]["truncated_at_max_new_tokens"] == 1
    assert json.loads((out / "RUN_STATUS.json").read_text())["status"] == "complete"
    assert (out / "baseline_report.md").read_text(encoding="utf-8").startswith("# Zero-shot baseline")
    # the local verifier accepts the output dir
    Z.main(["verify", "--bundle", str(BUNDLE), "--outputs", str(out)])
    # every database (incl. concert_singer after the DROP TABLE prediction) is unchanged
    PB.verify_dir(BUNDLE)
