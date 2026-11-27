"""TC-1 type contract (scripts/finetune_v2/type_contract.py). Plan §E.3 item 2."""
import ast
import json
import sqlite3
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/finetune_v2"))
import type_contract as T  # noqa: E402


def _db(tmp_path, ddl, rows):
    p = tmp_path / "x.sqlite"
    con = sqlite3.connect(p)
    con.execute(ddl)
    con.executemany("INSERT INTO t VALUES (" + ",".join("?" * len(rows[0])) + ")", rows)
    con.commit()
    con.close()
    return T.profile_db(p, "x")["x__t"]


def test_classes(tmp_path):
    e = _db(tmp_path, "CREATE TABLE t (a INT, b REAL, c DATETIME, d NUMERIC, e TEXT, f INT)",
            [(1, 1.5, "2020-01-01", 1, "x", None), (2, 2, "2020-01-02", "abc", None, None)])
    f = e["fields"]
    assert f["a"]["K"] == "int" and not f["a"]["declared_mismatch"]
    assert f["b"]["K"] == "num"                        # REAL affinity stores 2 as 2.0 -> real only
    assert f["c"]["K"] == "text" and f["c"]["declared_mismatch"] is True
    assert f["d"]["K"] == "nonconformant"              # int + text
    assert f["e"]["K"] == "text" and f["e"]["nonnull"] == 1
    assert f["f"]["K"] == "null"
    assert e["rows"] == 2


def test_int_real_is_num(tmp_path):
    e = _db(tmp_path, "CREATE TABLE t (a NUMERIC)", [(1,), (1.5,)])
    assert e["fields"]["a"]["K"] == "num" and e["fields"]["a"]["observed_classes"] == ["integer", "real"]


ENTRY = {"rows": 2, "fields": {"a": {"K": "int", "nonnull": 2}, "b": {"K": "num", "nonnull": 1},
                               "c": {"K": "text", "nonnull": 2}}}
GOOD = {"_id": {"objectId": 2}, "a": {"int": 2}, "b": {"double": 1, "null": 1}, "c": {"string": 2}}


def test_census_ok():
    assert T.compare_census(ENTRY, 2, GOOD, "explicit_null") == []


@pytest.mark.parametrize("mut,needle", [
    (lambda g: g["c"].update({"string": 1, "int": 1}), "disallowed_type: c"),
    (lambda g: g["b"].update({"string": 1}), "disallowed_type: b"),           # string in num field
    (lambda g: g.__setitem__("a", {"int": 1, "double": 1}), "disallowed_type: a"),   # double in int field
    (lambda g: g.__setitem__("a", {"int": 1, "bool": 1}), "forbidden_type: a has bool"),
    (lambda g: g.__setitem__("zz", {"string": 2}), "extra_field: zz"),
    (lambda g: g.__setitem__("c", {"string": 1}), "nonnull_count: c"),
    (lambda g: g.__setitem__("_id", {"string": 2}), "_id_type"),
])
def test_census_fails_closed(mut, needle):
    g = json.loads(json.dumps(GOOD))
    mut(g)
    v = T.compare_census(ENTRY, 2, g, "explicit_null")
    assert any(needle in x for x in v), v


def test_census_count_mismatch():
    assert any(x.startswith("count:") for x in T.compare_census(ENTRY, 3, GOOD, "explicit_null"))


def test_missing_representation():
    g = {"a": {"int": 2}, "b": {"double": 1}, "c": {"string": 2}}
    assert T.compare_census(ENTRY, 2, g, "missing") == []
    assert any("absent_field: b" in x for x in T.compare_census(ENTRY, 2, g, "explicit_null"))
    assert any("explicit_null_in_missing_repr" in x for x in T.compare_census(ENTRY, 2, GOOD, "missing"))


def test_converter_independent():
    tree = ast.parse((ROOT / "scripts/finetune_v2/type_contract.py").read_text(encoding="utf-8"))
    mods = {n.names[0].name for n in ast.walk(tree) if isinstance(n, ast.Import)} | \
           {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    bad = {m for m in mods if m and (m.startswith(("sql2mql", "verifier", "fixtures", "mongo_import", "schema_profile")))}
    assert not bad


@pytest.mark.slow
def test_manifest_reproduces_M1():
    doc = T.load_manifest(T.MANIFEST)
    c = doc["census"]["counts"]
    assert (c["columns"], c["conform_declared"], c["text_under_numeric_affinity"], c["mixed_class"],
            c["blob_affinity_single_class"]) == (3824, 3507, 199, 115, 3)
    assert doc["census"]["text_under_numeric_by_declared"] == {"DATETIME": 161, "DATE": 18, "TIMESTAMP": 16,
                                                               "BOOL": 3, "NUMERIC": 1}
    assert doc["census"]["mixed_by_db"] == {"baseball_1": 90, "bike_1": 22, "cre_Theme_park": 1,
                                            "formula_1": 1, "wine_1": 1}
    assert len(doc["db_ids"]) == 139
    # the manifest still matches the SQLite files it was built from
    for d, h in list(doc["sqlite_sha256"].items())[:10]:
        assert T.sha_file(T.SPIDER_DB / d / f"{d}.sqlite") == h


@pytest.mark.slow
def test_eval_manifest_reproduces_M2():
    doc = T.load_manifest(T.EVAL_MANIFEST)
    c = doc["census"]["counts"]
    assert c["columns"] == 383 and c["columns"] - c["conform_declared"] == 17
    assert c["text_under_numeric_affinity"] == 15 and doc["census"]["mixed_by_db"] == {"wta_1": 2}
    mixed = sorted(k + "." + f for k, e in doc["collections"].items() for f, s in e["fields"].items()
                   if s["K"] == "nonconformant")
    assert mixed == ["wta_1__players.birth_date", "wta_1__rankings.player_id"]
