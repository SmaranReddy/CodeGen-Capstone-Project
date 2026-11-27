"""Test-suite execution evaluation (scripts/sql_v1/evaluate_test_suite_dev.py): variant discovery, the
all-variants verdict, error/timeout handling, missing variants and the dev-only guard. Tiny synthetic DBs
only; no Spider data, no network."""
import json
import sqlite3
import sys
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/sql_v1"))
sys.dont_write_bytecode = True
pytest.importorskip("sqlparse")
import evaluate_test_suite_dev as T  # noqa: E402


def make_db(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE t (id INTEGER, name TEXT, age INTEGER)")
    con.executemany("INSERT INTO t VALUES (?, ?, ?)", rows)
    con.commit()
    con.close()
    return path


@pytest.fixture
def suite(tmp_path):
    """db 'd': original + 2 variants. On the original, `age > 30` and `age >= 40` coincide; v2 tells them apart."""
    root = tmp_path / "database"
    make_db(root / "d" / "d.sqlite", [(1, "a", 20), (2, "b", 40)])
    make_db(root / "d" / "dv1.sqlite", [(1, "x", 50), (2, "y", 10), (3, "z", 60)])
    make_db(root / "d" / "dv2.sqlite", [(1, "p", 30), (2, "q", 31)])
    return root


# ---------------------------------------------------------------- official code is the pinned, unmodified copy
def test_vendored_official_files_match_pinned_hashes():
    for name, sha in T.OFFICIAL_SHA256.items():
        assert T.sha256_file(T.OFFICIAL_DIR / name) == sha
    ee = T.load_official()
    assert ee.TIMEOUT == 60 and Path(ee.__file__).parent == T.OFFICIAL_DIR
    assert "parse" not in sys.modules or Path(getattr(sys.modules["parse"], "__file__", "")) != T.OFFICIAL_DIR / "parse.py"


# ---------------------------------------------------------------- db_id -> variants
def test_variant_paths_original_first_and_per_db(suite):
    make_db(suite / "other" / "other.sqlite", [(1, "a", 1)])
    names = [p.name for p in T.variant_paths(suite, "d")]
    assert names == ["d.sqlite", "dv1.sqlite", "dv2.sqlite"]
    assert [p.name for p in T.variant_paths(suite, "other")] == ["other.sqlite"]


def test_missing_directory_or_original_means_missing_variants(suite):
    assert T.variant_paths(suite, "nope") == []
    (suite / "d" / "d.sqlite").unlink()
    assert T.variant_paths(suite, "d") == []
    r = T.judge([], "SELECT 1", "SELECT 1")
    assert r["status"] == "missing_variants" and r["ts_exec"] is None


def test_extract_skips_macosx_and_other_dbs_and_never_overwrites(tmp_path, suite):
    zp = tmp_path / "ts.zip"
    with zipfile.ZipFile(zp, "w") as z:
        z.write(suite / "d" / "d.sqlite", "database/d/d.sqlite")
        z.write(suite / "d" / "dv1.sqlite", "database/d/dv1.sqlite")
        z.writestr("__MACOSX/database/d/._dv1.sqlite", b"junk")
        z.writestr("database/d/._dv9.sqlite", b"junk")
        z.writestr("database/heldout/heldout.sqlite", b"x")
    out = tmp_path / "out"
    man = T.extract(out, ["d"], zp)
    assert [f["name"] for f in man["dbs"]["d"]] == ["d.sqlite", "dv1.sqlite"]
    assert not (out / "heldout").exists()
    assert T.extract(out, ["d"], zp) == man  # idempotent
    (out / "d" / "dv1.sqlite").write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="not overwriting"):
        T.extract(out, ["d"], zp)
    cov = T.check_variants(out, ["d"], man)
    assert cov["d"]["n_files"] == 2 and cov["d"]["unreadable"]


# ---------------------------------------------------------------- verdict across variants
def test_coincidental_match_on_original_fails_on_variant(suite):
    paths = T.variant_paths(suite, "d")
    r = T.judge(paths, "SELECT name FROM t WHERE age >= 40", "SELECT name FROM t WHERE age > 30")
    assert r["status"] == "mismatch" and r["ts_exec"] == 0 and r["fail_file"] == "dv2.sqlite"
    assert r["n_checked"] == 3
    # passes on the original alone: a single-DB EX would accept it
    assert T.judge(paths[:1], "SELECT name FROM t WHERE age >= 40", "SELECT name FROM t WHERE age > 30")["ts_exec"] == 1


def test_equivalent_query_passes_every_variant(suite):
    paths = T.variant_paths(suite, "d")
    r = T.judge(paths, "SELECT t.age, t.name FROM t AS t WHERE NOT age <= 30", "SELECT name, age FROM t WHERE age > 30")
    assert r == {**r, "status": "pass", "ts_exec": 1, "n_checked": 3}  # column order is free (official result_eq)


def test_order_matters_only_when_gold_has_order_by(suite):
    paths = T.variant_paths(suite, "d")
    assert T.judge(paths, "SELECT name FROM t ORDER BY id DESC", "SELECT name FROM t")["ts_exec"] == 1
    assert T.judge(paths, "SELECT name FROM t ORDER BY id DESC", "SELECT name FROM t ORDER BY id")["ts_exec"] == 0


def test_distinct_removed_by_default_kept_on_request(suite):
    paths = T.variant_paths(suite, "d")
    make_db(suite / "e" / "e.sqlite", [(1, "a", 1), (2, "a", 1)])
    e = T.variant_paths(suite, "e")
    assert T.judge(e, "SELECT DISTINCT name FROM t", "SELECT name FROM t")["ts_exec"] == 1
    assert T.judge(e, "SELECT DISTINCT name FROM t", "SELECT name FROM t", keep_distinct=True)["ts_exec"] == 0
    assert T.judge(paths, "SELECT name FROM t", "SELECT name FROM t", keep_distinct=True)["ts_exec"] == 1


def test_official_value_replacement_applies_to_prediction(suite):
    paths = T.variant_paths(suite, "d")
    # evaluate() turns the literal 'value' into '1' in predictions (official behaviour, reproduced as is)
    r = T.judge(paths, "SELECT count(*) FROM t WHERE id = value", "SELECT count(*) FROM t WHERE id = 1")
    assert r["ts_exec"] == 1


# ---------------------------------------------------------------- invalid SQL, timeouts, gold failures
def test_invalid_prediction_is_incorrect_not_an_exception(suite):
    paths = T.variant_paths(suite, "d")
    r = T.judge(paths, "SELECT nosuchcol FROM t", "SELECT name FROM t")
    assert r["status"] == "pred_error" and r["ts_exec"] == 0 and "no such column" in r["detail"]
    r = T.judge(paths, "SELEC name FROM", "SELECT name FROM t")
    assert r["status"] == "pred_error" and r["ts_exec"] == 0
    r = T.judge(paths, "", "SELECT name FROM t")
    assert r["ts_exec"] == 0


def test_prediction_timeout_is_incorrect(suite):
    paths = T.variant_paths(suite, "d")
    slow = "WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM c) SELECT count(*) FROM c"
    r = T.judge(paths, slow, "SELECT count(*) FROM t", timeout_s=1)
    assert r["status"] == "pred_timeout" and r["ts_exec"] == 0


def test_gold_failure_is_not_evaluable_not_wrong(suite):
    paths = T.variant_paths(suite, "d")
    r = T.judge(paths, "SELECT name FROM t", "SELECT nosuch FROM t")
    assert r["status"] == "gold_error" and r["ts_exec"] is None
    slow = "WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM c) SELECT count(*) FROM c"
    assert T.judge(paths, "SELECT 1", slow, timeout_s=1)["status"] == "gold_timeout"


def test_gold_and_pred_never_compared_across_files(suite, monkeypatch):
    seen = []
    real = T.exec_on_db

    def spy(path, q, timeout_s):
        seen.append(Path(path).name)
        return real(path, q, timeout_s)

    monkeypatch.setattr(T, "exec_on_db", spy)
    T.judge(T.variant_paths(suite, "d"), "SELECT name FROM t", "SELECT name FROM t")
    assert seen == ["d.sqlite", "d.sqlite", "dv1.sqlite", "dv1.sqlite", "dv2.sqlite", "dv2.sqlite"]


def test_connections_are_read_only(suite):
    p = T.variant_paths(suite, "d")[0]
    before = T.sha256_file(p)
    assert T.exec_on_db(p, "DELETE FROM t", 5)[0] == "exception"
    assert T.sha256_file(p) == before


# ---------------------------------------------------------------- dev-only guard
@pytest.mark.parametrize("rec", [{"id": "spider_test_0001", "partition": "test"},
                                 {"id": "spider_dev_0001", "partition": "test"},
                                 {"id": "train_0001"}])
def test_guard_rejects_non_dev_records(rec):
    with pytest.raises(T.HeldOutDataError):
        T.guard_dev([rec], "dev_full.jsonl")


def test_guard_rejects_test_files(tmp_path):
    with pytest.raises(T.HeldOutDataError):
        T.guard_dev([{"id": "spider_dev_0000", "partition": "dev"}], tmp_path / "test_full.jsonl")
    assert T.guard_dev([{"id": "spider_dev_0000", "partition": "dev"}], tmp_path / "dev_full.jsonl")


def test_load_inputs_refuses_test_predictions(tmp_path):
    dev = tmp_path / "dev_full.jsonl"
    dev.write_text(json.dumps({"id": "spider_dev_0000", "db_id": "d", "sql": "SELECT 1", "hardness": "easy",
                               "partition": "dev"}) + "\n", encoding="utf-8")
    run = tmp_path / "run"
    run.mkdir()
    (run / "evaluation_results.jsonl").write_text(
        json.dumps({"id": "spider_test_0000", "db_id": "d", "gold_sql": "SELECT 1", "eval_sql": "SELECT 1"}) + "\n",
        encoding="utf-8")
    with pytest.raises(T.HeldOutDataError):
        T.load_inputs(run, dev)


def test_new_out_dir_refuses_non_empty(tmp_path):
    (tmp_path / "x").write_text("keep", encoding="utf-8")
    with pytest.raises(SystemExit):
        T.new_out_dir(tmp_path)
    assert (tmp_path / "x").read_text(encoding="utf-8") == "keep"
