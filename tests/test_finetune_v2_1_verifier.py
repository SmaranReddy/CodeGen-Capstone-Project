"""Verifier 3.1 (fold-trap fixtures). Plan §B.4.3 / §E.4 criterion 4. Needs LOCAL MongoDB 127.0.0.1:27018."""
import ast
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/finetune_v2"))
RUN01 = ROOT / "outputs/sql2mql_v2/pilot_v3/run01/records.jsonl"
CONTROLS = ("spider_train_06342", "spider_train_04935")   # R040, R082


def _imports(name):
    tree = ast.parse((ROOT / "scripts/finetune_v2" / name).read_text(encoding="utf-8"))
    return {n.names[0].name for n in ast.walk(tree) if isinstance(n, ast.Import)} | \
           {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}


def test_verifier_does_not_import_converter():
    for f in ("verifier3_1.py", "fixtures3_1.py"):
        bad = {m for m in _imports(f) if m and m.startswith(("sql2mql", "schema_profile"))}
        assert not bad, (f, bad)


def test_fold_substitution_uses_real_code_points():
    import fixtures3_1 as FX
    assert {ord(v) for v in FX.FOLD_MAP.values()} == {0x212A, 0x17F}
    assert FX.uses_like("SELECT a FROM t WHERE b LIKE '%x%'") and not FX.uses_like("SELECT a FROM t WHERE b = 'like'")


@pytest.fixture(scope="module")
def env():
    try:
        import v2_common as C
        cl = C.local_client()
        C.check_server(cl)
    except Exception:  # noqa: BLE001
        pytest.skip("local MongoDB 127.0.0.1:27018 (mongo:8.0.34) not reachable")
    import verifier3_1 as V
    recs = {}
    for line in open(RUN01, encoding="utf-8"):
        r = json.loads(line)
        if r["id"] in CONTROLS:
            recs[r["id"]] = r
    return V, V.RealDbs(cl), recs


@pytest.mark.mongo
@pytest.mark.parametrize("rid", CONTROLS)
def test_v2_mql_rejected_v21_accepted(env, rid):
    import schema_profile as SP
    import sql2mql_v2_1 as N
    V, real, recs = env
    r = recs[rid]
    item = {"id": rid, "db_id": r["db_id"], "sql": r["sql"]}
    a = V.verify(item, r["mql"], real)
    assert a["status"] == "fixture_mismatch" and a["detail"].startswith("v31_fold_traps")
    assert a["verifier"] == V.VERIFIER_VERSION and a["base_verifier"] == "sql2mql-v2-verifier-3"
    m, _ = N.convert(r["sql"], SP.Profile(r["db_id"]))
    b = V.verify(item, m, real)
    assert b["status"] == "accepted"
    assert all(not str(v).startswith(("FAIL", "skipped")) for v in b["checks"]["v31_fold_traps"].values())
