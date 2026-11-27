"""sql2mql-v2 (scripts/finetune_v2): independence, local-only MongoDB guard, comparator,
converter regressions for F1-F15 and the audit defects, fixture families (V3), pilot freeze,
and the recorded adversarial / negative-control artifacts. Pure tests need no MongoDB; tests
marked `live` run against the local mongo:8.0.34 container and are skipped without it.
No Atlas, no network."""
import ast
import json
import sqlite3
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
V2 = ROOT / "scripts/finetune_v2"
sys.dont_write_bytecode = True
sys.path.insert(0, str(V2))
import v2_common as C  # noqa: E402
import comparator as CMP  # noqa: E402
import v2_fixtures as FX  # noqa: E402
import mongo_import as MI  # noqa: E402
import schema_profile as SP  # noqa: E402
import sql2mql_v2 as M  # noqa: E402

CONVERTER_MODULES = {"sql2mql_v2", "schema_profile"}
VERIFIER_MODULES = {"verifier", "comparator", "v2_fixtures", "mongo_import"}


def _imports(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    out = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            out |= {a.name.split(".")[0] for a in n.names}
        elif isinstance(n, ast.ImportFrom) and n.module:
            out.add(n.module.split(".")[0])
    return out


# ---------------------------------------------------------------- independence and safety
def test_converter_and_verifier_do_not_import_each_other():
    for m in CONVERTER_MODULES:
        assert not _imports(V2 / f"{m}.py") & (VERIFIER_MODULES | {"oracle_v2", "mutants_v2"}), m
    for m in VERIFIER_MODULES:
        assert not _imports(V2 / f"{m}.py") & CONVERTER_MODULES, m


def test_v1_code_is_not_imported_by_v2():
    for p in V2.glob("*.py"):
        imps = _imports(p)
        assert not imps & {"sql2mql", "sqlsem", "verify", "nk_common"}, p.name


def test_local_mongo_guard_refuses_other_hosts():
    with pytest.raises(RuntimeError):
        C.local_client("cluster0.example.mongodb.net", 27017)
    with pytest.raises(RuntimeError):
        C.local_client("127.0.0.1", 27017)


_ENV_NAMES = {"environ", "environb", "getenv", "getenvb", "putenv", "load_dotenv", "dotenv_values", "find_dotenv"}
_SECRET_MARKERS = ("mongodb+srv", "MONGODB_URI")


def credential_access(source):
    """Syntax-aware scan: return the places where `source` reads the process environment or names an Atlas URI.

    Looks at code only (identifiers, attribute access, imports, string literals), so prose in comments is ignored.
    A string literal equal to an environment accessor name (getattr(os, "environ")) or containing an Atlas marker
    counts, because that is how such access is spelt indirectly.
    """
    hits, tree = [], ast.parse(source)
    docstrings = set()                      # docstrings are prose, like comments
    for n in ast.walk(tree):
        if isinstance(n, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and n.body:
            f = n.body[0]
            if isinstance(f, ast.Expr) and isinstance(f.value, ast.Constant) and isinstance(f.value.value, str):
                docstrings.add(id(f.value))
    for n in ast.walk(tree):
        if id(n) in docstrings:
            continue
        if isinstance(n, ast.Attribute) and n.attr in _ENV_NAMES:
            hits.append((n.lineno, f"attribute .{n.attr}"))
        elif isinstance(n, ast.Name) and n.id in _ENV_NAMES:
            hits.append((n.lineno, f"name {n.id}"))
        elif isinstance(n, (ast.Import, ast.ImportFrom)):
            for a in n.names:
                if a.name.split(".")[-1] in _ENV_NAMES:
                    hits.append((n.lineno, f"import {a.name}"))
        elif isinstance(n, ast.Constant) and isinstance(n.value, str):
            if n.value in _ENV_NAMES or any(m in n.value for m in _SECRET_MARKERS):
                hits.append((n.lineno, "string literal"))
    return hits


def test_no_atlas_credentials_are_read():
    for p in V2.glob("*.py"):
        assert not _imports(p) & {"dotenv", "os", "mongo_conn"}, p.name
        assert credential_access(p.read_text(encoding="utf-8")) == [], p.name


def test_credential_scan_ignores_comments_but_catches_real_access():
    """Regression: the word 'environment' in a comment (run_pilot_v4.py:278) must not trip the scan."""
    assert credential_access("x = 1  # environment control: os.environ, getenv, mongodb+srv in a comment\n") == []
    assert credential_access("environment_control = True\n") == []
    assert credential_access('"""Never reads MONGODB_URI or os.environ."""\nx = 1\n') == []
    assert credential_access('"""doc"""\nu = "MONGODB_URI"\n')          # a real literal after a docstring still counts
    for bad in ("import os\nu = os.environ['MONGO']\n",
                "import os\nu = os.getenv('X')\n",
                "from os import environ\n",
                "from os import getenv as g\n",
                "import os\nu = getattr(os, 'environ')\n",
                "from dotenv import load_dotenv\n",
                "u = 'mongodb+srv://host/db'\n",
                "k = 'MONGODB_URI'\n"):
        assert credential_access(bad), bad


def test_audit_execution_code_is_not_run_by_v2():
    """Only the audit's pure data module may be imported (its execution code targets Atlas)."""
    for p in V2.glob("*.py"):
        assert not _imports(p) & {"adversarial", "mql_mutants", "audit_fixtures", "spider_census"}, p.name


# ---------------------------------------------------------------- comparator
def test_comparator_strict_types_and_positions():
    assert not CMP.value_eq("10", 10)
    assert CMP.value_eq(2, 2.0)
    assert CMP.value_eq(None, None) and not CMP.value_eq(None, 0)
    assert not CMP.multiset_eq([(1, "a")], [("a", 1)])           # no column permutation
    assert CMP.multiset_eq([(1, "a"), (2, "b")], [(2, "b"), (1, "a")])
    assert not CMP.multiset_eq([(1,), (1,)], [(1,), (2,)])


def test_comparator_rounding_boundary_regression():
    """37.5404535 vs 37.540453500000005 straddle a 6-decimal rounding boundary."""
    assert CMP.multiset_eq([(37.5404535, -122.1839869375)], [(37.540453500000005, -122.18398693749998)])


def test_comparator_rejects_unsupported_values():
    with pytest.raises(CMP.BadValue):
        CMP.docs_to_rows([{"a": {"x": 1}}], ["a"])
    with pytest.raises(CMP.BadValue):
        CMP.docs_to_rows([{"a": 1, "b": 2}], ["a"])
    assert CMP.docs_to_rows([{"a": 1}], ["a", "b"]) == [(1, None)]


def test_tie_aware_order():
    keyed = [("x", 1), ("y", 1), ("z", 2)]                         # answer col + key col
    ok, _, cut = CMP.match_ordered([("y",), ("x",), ("z",)], keyed, 1, None, None)
    assert ok and not cut
    ok, _, _ = CMP.match_ordered([("z",), ("x",), ("y",)], keyed, 1, None, None)
    assert not ok
    ok, _, cut = CMP.match_ordered([("y",)], keyed, 1, None, 1)     # LIMIT 1 cuts the tie
    assert ok and cut


# ---------------------------------------------------------------- converter regressions
@pytest.fixture(scope="module")
def prof(tmp_path_factory):
    d = tmp_path_factory.mktemp("v2db")
    path = d / "toy.sqlite"
    con = sqlite3.connect(path)
    con.executescript("""
        CREATE TABLE T ("Id" INTEGER, "Name" TEXT, grp TEXT, x INTEGER, y REAL, is_active BOOLEAN,
                        code TEXT, d1 DATE);
        CREATE TABLE dept (dept_id INTEGER PRIMARY KEY, dname TEXT);
        CREATE TABLE emp (emp_id INTEGER PRIMARY KEY, dept_id INTEGER, salary INTEGER, k TEXT);
        CREATE TABLE m (id INTEGER, v);
        INSERT INTO T VALUES (1,'alice','a',1,1.5,1,'10','2020-01-01'),(2,'Bob','a',3,2.0,0,'9','2021-01-01'),
                             (3,'carol','b',NULL,NULL,NULL,'abc',NULL);
        INSERT INTO dept VALUES (1,'Sales'),(2,'Ops');
        INSERT INTO emp VALUES (10,1,100,'1'),(11,2,50,'2');
        INSERT INTO m VALUES (1,5),(2,'5');
    """)
    con.commit()
    con.close()
    return SP.Profile("toy", path)


def conv(sql, prof):
    return M.convert(sql, prof)[0]


def stages(mql):
    return [next(iter(s)) for s in mql["pipeline"]]


def test_F1_ungrouped_aggregate_has_default_row(prof):
    m = conv("SELECT count(*), sum(x) FROM t WHERE x > 100", prof)
    assert stages(m)[:4] == ["$match", "$facet", "$project", "$replaceRoot"]
    default = m["pipeline"][2]["$project"]["row"]["$ifNull"][1]
    assert sorted(default.values(), key=str) == sorted([0, None, 0], key=str)


def test_F2_sum_guarded_by_non_null_count(prof):
    m = conv("SELECT grp, sum(x) FROM t GROUP BY grp", prof)
    post = next(s["$set"] for s in m["pipeline"] if "$set" in s)
    assert any("$cond" in v and v["$cond"][2] is None for v in post.values())


def test_F3_distinct_group_by_dedupes(prof):
    m = conv("SELECT DISTINCT grp FROM t GROUP BY grp, is_active", prof)
    assert stages(m).count("$group") == 2
    with pytest.raises(M.Unsupported):
        conv("SELECT DISTINCT grp FROM t GROUP BY grp, is_active ORDER BY grp", prof)


def test_F4_date_arithmetic_rejected(prof):
    with pytest.raises(M.Unsupported) as e:
        conv("SELECT avg(d1 - d1) FROM t", prof)
    assert e.value.code == "arithmetic_type"


def test_F5_identifiers_canonicalised(prof):
    m = conv("SELECT NAME FROM t WHERE id = 1", prof)
    assert m["collection"] == "toy__T"
    assert m["pipeline"][0] == {"$match": {"Id": 1}}
    assert list(m["pipeline"][-1]["$project"].values())[1] == "$Name"   # canonical field path


def test_F6_affinity_literals(prof):
    assert conv("SELECT id FROM t WHERE x > '2'", prof)["pipeline"][0] == {"$match": {"x": {"$gt": 2}}}
    assert conv("SELECT id FROM t WHERE code = 10", prof)["pipeline"][0] == {"$match": {"code": "10"}}
    for sql in ("SELECT id FROM t WHERE x = 'abc'", "SELECT id FROM m WHERE v = 5", "SELECT sum(v) FROM m"):
        with pytest.raises(M.Unsupported):
            conv(sql, prof)


def test_F7_mixed_type_join_rejected(prof):
    with pytest.raises(M.Unsupported) as e:
        conv("SELECT T1.dname FROM dept AS T1 JOIN emp AS T2 ON T1.dept_id = T2.k", prof)
    assert e.value.code == "join_key_types"


def test_F8_integer_division_and_zero_guard(prof):
    p = conv("SELECT x / x FROM t", prof)["pipeline"][-1]["$project"]
    e = next(iter(v for k, v in p.items() if k != "_id"))
    assert e["$cond"][0] == {"$eq": ["$x", 0]} and e["$cond"][1] is None and "$trunc" in e["$cond"][2]
    p = conv("SELECT x / y FROM t", prof)["pipeline"][-1]["$project"]
    e = next(iter(v for k, v in p.items() if k != "_id"))
    assert "$divide" in e["$cond"][2]


def test_F9_order_by_ordinal_and_alias(prof):
    m = conv("SELECT grp, count(*) FROM t GROUP BY grp ORDER BY 2 DESC, 1", prof)
    assert next(s["$sort"] for s in m["pipeline"] if "$sort" in s) == {"a0": -1, "_id.g0": 1}
    m = conv("SELECT id, y AS score FROM t ORDER BY score DESC", prof)
    assert next(s["$sort"] for s in m["pipeline"] if "$sort" in s) == {"y": -1}


def test_F10_not_bare_boolean_excludes_null(prof):
    assert conv("SELECT id FROM t WHERE NOT is_active", prof)["pipeline"][0] == {"$match": {"is_active": 0}}


def test_F11_left_join_guards_are_missing_aware(prof):
    m = conv("SELECT T1.dname, count(T2.emp_id) FROM dept AS T1 LEFT JOIN emp AS T2 "
             "ON T1.dept_id = T2.dept_id GROUP BY T1.dept_id", prof)
    s = json.dumps(m)
    assert "preserveNullAndEmptyArrays" in s and '"$ifNull": ["$$k", null]' in s
    grp = next(x["$group"] for x in m["pipeline"] if "$group" in x)
    assert any("$cond" in json.dumps(v) and "$ifNull" in json.dumps(v) for v in grp.values())


def test_F12_lookup_alias_is_reserved(prof):
    m = conv("SELECT T1.dname FROM dept AS T1 JOIN emp AS T2 ON T1.dept_id = T2.dept_id", prof)
    lk = next(s["$lookup"] for s in m["pipeline"] if "$lookup" in s)
    assert lk["as"] == "__j1"


def test_F13_like_regex(prof):
    assert M.like_regex("a%d") == "^a.*d\\z"
    assert M.like_regex("%x%") == "x"
    assert M.like_regex("a.b%") == "^a\\.b"
    m = conv("SELECT id FROM t WHERE name LIKE 'a%'", prof)
    assert m["pipeline"][0]["$match"]["Name"]["$options"] == "is"
    with pytest.raises(M.Unsupported):
        conv("SELECT id FROM t WHERE name LIKE 'É%'", prof)


def test_F14_functions_rejected(prof):
    for sql in ("SELECT round(y) FROM t", "SELECT upper(name) FROM t", "SELECT cast(x AS TEXT) FROM t"):
        with pytest.raises(M.Unsupported):
            conv(sql, prof)


def test_F15_null_tests_are_missing_safe(prof):
    m = conv("SELECT id FROM t WHERE x != 3", prof)
    assert m["pipeline"][0] == {"$match": {"x": {"$nin": [3, None]}}}
    m = conv("SELECT id FROM t WHERE x > y", prof)
    assert json.dumps(m).count("$ifNull") == 2


def test_bare_columns_and_limits(prof):
    conv("SELECT T1.dname, count(*) FROM dept AS T1 JOIN emp AS T2 ON T1.dept_id = T2.dept_id "
         "GROUP BY T1.dept_id", prof)                                # functionally determined
    for sql in ("SELECT name, max(x) FROM t", "SELECT id FROM t LIMIT 2",
                "SELECT id FROM t WHERE x > (SELECT avg(x) FROM t)",
                "SELECT id FROM t UNION SELECT id FROM t",
                "SELECT T1.id FROM t AS T1 JOIN t AS T2 ON T1.id = T2.id"):
        with pytest.raises(M.Unsupported):
            conv(sql, prof)


def test_double_quoted_string_literal(prof):
    assert conv('SELECT id FROM t WHERE name = "alice"', prof)["pipeline"][0] == {"$match": {"Name": "alice"}}


def test_dollar_strings_are_literals(prof):
    m = conv("SELECT id FROM t WHERE name = '$x' OR x > y", prof)
    assert '"$x"' in json.dumps(m["pipeline"][0])


# ---------------------------------------------------------------- fixture families (V3)
@pytest.fixture(scope="module")
def builder(tmp_path_factory):
    d = tmp_path_factory.mktemp("fx")
    path = d / "fx.sqlite"
    con = sqlite3.connect(path)
    con.executescript("""
        CREATE TABLE p (pid INTEGER PRIMARY KEY, name TEXT, score INTEGER, city TEXT NOT NULL);
        INSERT INTO p VALUES (1,'Ann',10,'Rome'),(2,'Bob',0,'Oslo'),(3,'Cid',7,'Rome'),(4,'Dee',3,'Lima');
    """)
    con.commit()
    con.close()
    tables = MI.read_tables(path)
    info = FX.DbInfo(path, tables)
    return FX.Builder(info, {"p"}, "SELECT name FROM p WHERE city = 'Rome' AND score > 5", "t")


def test_all_audit_v3_families_present():
    assert set(FX.AUDIT_V3) == {"empty tables", "zero-match WHERE", "all-NULL aggregate columns",
                                "duplicate rows", "case variants for LIKE", "distinct values per GROUP BY key",
                                "zero divisors", "text-typed numerics", "newline strings"}
    assert set(FX.AUDIT_V3.values()) <= set(FX.FAMILIES)


def _stored(builder, fam):
    con, stored, problems = builder.materialize(fam)
    con.close()
    assert not problems, problems
    return stored["p"][1]


def test_family_empty_and_zero_match(builder):
    assert _stored(builder, "empty") == []
    assert all("Rome" not in r for r in _stored(builder, "zero_match"))


def test_family_all_null_keeps_keys_and_not_null(builder):
    rows = _stored(builder, "all_null")
    assert all(r[0] is not None and r[3] is not None and r[1] is None and r[2] is None for r in rows)


def test_family_dup_case_newline_textnum(builder):
    assert len(_stored(builder, "dup")) > 4
    case = _stored(builder, "case_variants")
    assert any(isinstance(v, str) and v.isupper() for r in case for v in r[1:])
    assert any("rome" in str(v) or "ROME" in str(v) for r in case for v in r)
    assert any("\n" in str(v) for r in _stored(builder, "newline") for v in r)
    tn = _stored(builder, "text_numerics")
    assert any(v in ("9", "10", "100") for r in tn for v in (r[1], r[3]))
    assert all(isinstance(r[2], int) or r[2] is None for r in tn)  # numeric column stays numeric


def test_family_zero_div_and_shuffle(builder):
    assert any(r[2] == 0 for r in _stored(builder, "zero_div"))
    sh = _stored(builder, "shuffle")
    assert sorted(r[1] for r in sh) == ["Ann", "Bob", "Cid", "Dee"]
    assert [r[0] for r in sh] == [1, 2, 3, 4]                      # keys are not shuffled


def test_family_literal_hits(builder):
    rows = _stored(builder, "literal_hits")
    assert any(r[2] in (4, 5, 6) for r in rows)


# ---- verifier-2 additions (after the iteration-1 mutant study)
def test_verifier_does_not_import_oracle_or_mutants():
    for m in VERIFIER_MODULES:
        assert not _imports(V2 / f"{m}.py") & {"oracle_v2", "mutants_v2", "mutant_study"}, m


@pytest.fixture(scope="module")
def builder2(tmp_path_factory):
    """Text and composite primary keys (iteration-1 degeneracy), a join, OR predicates."""
    d = tmp_path_factory.mktemp("fx2")
    path = d / "fx2.sqlite"
    con = sqlite3.connect(path)
    con.executescript("""
        CREATE TABLE dept (code TEXT PRIMARY KEY, name TEXT, budget INTEGER);
        CREATE TABLE emp (code TEXT, eid INTEGER, salary INTEGER, PRIMARY KEY (code, eid));
        INSERT INTO dept VALUES ('A','Sales',100),('B','Ops',50),('C','Lab',70);
        INSERT INTO emp VALUES ('A',1,10),('A',2,20),('B',3,30),('C',4,40);
    """)
    con.commit()
    con.close()
    tables = MI.read_tables(path)
    info = FX.DbInfo(path, tables)
    sql = "SELECT T1.name FROM dept AS T1 JOIN emp AS T2 ON T1.code = T2.code WHERE T2.salary > 25 OR T1.budget < 60"
    wit = {"dept": [r for r in tables["dept"][1]], "emp": [r for r in tables["emp"][1] if r[2] > 25]}
    return FX.Builder(info, {"dept", "emp"}, sql, "t2", wit), info


def test_fresh_keys_text_and_composite(builder2):
    b, _ = builder2
    for fam in ("dup", "case_variants", "newline", "literal_hits"):
        con, stored, problems = b.materialize(fam)
        con.close()
        assert not problems, (fam, problems)
        assert len(stored["dept"][1]) > 3 and len(stored["emp"][1]) > 4, fam


def test_degenerate_family_is_reported(builder2, monkeypatch):
    b, _ = builder2
    monkeypatch.setattr(b, "_fresh_keys", lambda t, rows, base_rows=None: rows)   # v1-style bug
    con, stored, problems = b.materialize("dup")
    con.close()
    assert any("degenerate" in p for p in problems)


def test_atoms_and_join_keys(builder2):
    b, _ = builder2
    assert ("emp", "salary", "GT", [25]) in b.atoms and ("dept", "budget", "LT", [60]) in b.atoms
    assert b.join_keys == [("dept", "code"), ("emp", "code")]


def test_boundary_families_hit_literals(builder2):
    b, _ = builder2
    sal = {fam: {r[2] for r in _stored_t(b, fam, "emp")} for fam in ("boundary_minus", "boundary_equal", "boundary_plus")}
    assert 24 in sal["boundary_minus"] and 25 in sal["boundary_equal"] and 26 in sal["boundary_plus"]


def _stored_t(b, fam, t):
    con, stored, problems = b.materialize(fam)
    con.close()
    assert not problems, problems
    return stored[t][1]


def test_orphans_do_not_match_each_other(builder2):
    b, _ = builder2
    dept = {r[0] for r in _stored_t(b, "orphans", "dept")}
    emp = {r[0] for r in _stored_t(b, "orphans", "emp")}
    orphan_emp = {c for c in emp if "~orphan" in str(c)}
    assert orphan_emp and not orphan_emp & dept


def test_null_key_and_atom_families(builder2):
    b, _ = builder2
    assert all(r[0] is not None for r in _stored_t(b, "null_keys_even", "emp"))  # primary-key parts are never nulled
    assert any(r[2] is None for r in _stored_t(b, "null_atoms_even", "emp"))
    keep = _stored_t(b, "all_null_keep", "dept")
    assert all(r[2] is not None for r in keep) and all(r[1] is None for r in keep)


# ---------------------------------------------------------------- pilot freeze
def test_pilot_freeze_is_consistent():
    fr = json.loads(C.FREEZE.read_text(encoding="utf-8"))
    body = {k: v for k, v in fr.items() if k != "freeze_sha256"}
    assert C.sha256_bytes(C.canon(body).encode("utf-8")) == fr["freeze_sha256"]
    assert fr["pilot"]["n"] == 300 and fr["dev"]["n"] == 500
    assert not set(fr["pilot"]["ids"]) & set(fr["dev"]["ids"])
    assert fr["source"]["sha256"] == C.sha256_file(C.TRAIN)
    assert fr["v9_leakage_screen"]["train_records_on_dev_databases"] == 0
    assert "phase1_gate (roadmap P4: F1-F15 + V1-V3 + V8 on fixtures/dev)" in fr["acceptance_criteria"]


def test_dev_runner_refuses_pilot():
    import run_v2
    fr, rows = run_v2.load_sample("dev")
    assert not {r["id"] for r in rows} & set(fr["pilot"]["ids"])


# ---------------------------------------------------------------- recorded artifacts
ADV = C.OUT / "adversarial"


@pytest.mark.skipif(not (ADV / "suite.summary.json").exists(), reason="suite not run")
def test_adversarial_suite_has_no_false_accepts():
    s = json.loads((ADV / "suite.summary.json").read_text(encoding="utf-8"))
    assert s["FALSE_ACCEPTS"] == [] and s["expect_reject_but_converted"] == [] and s["converter"]["crash"] == []
    assert s["audit_cases"] == 103


@pytest.mark.skipif(not (ADV / "negative_controls_v1_mql.summary.json").exists(), reason="not run")
def test_v1_negative_controls_all_rejected():
    s = json.loads((ADV / "negative_controls_v1_mql.summary.json").read_text(encoding="utf-8"))
    assert s["controls"] == 33 and s["false_accepts"] == []


# ---------------------------------------------------------------- live (local MongoDB only)
def _live():
    try:
        cl = C.local_client()
        C.check_server(cl)
        return cl
    except Exception:  # noqa: BLE001
        return None


@pytest.fixture(scope="module")
def live_client():
    cl = _live()
    if cl is None:
        pytest.skip("local mongo:8.0.34 on 127.0.0.1:27018 not available")
    return cl


def test_live_family_catches_b1_that_l1_misses(live_client, tmp_path):
    """COUNT(*) without the default row passes L1 on non-empty data; the `empty` /
    `zero_match` fixtures must reject it."""
    import verifier as V
    path = tmp_path / "live1.sqlite"
    con = sqlite3.connect(path)
    con.executescript("CREATE TABLE p (pid INTEGER PRIMARY KEY, score INTEGER);"
                      "INSERT INTO p VALUES (1,10),(2,3),(3,8);")
    con.commit()
    con.close()
    real = V.RealDbs(live_client, path_of=lambda db: path)
    good = {"collection": "live1__p", "pipeline": [
        {"$match": {"score": {"$gt": 5}}},
        {"$facet": {"rows": [{"$group": {"_id": None, "a0": {"$sum": 1}}}]}},
        {"$project": {"row": {"$ifNull": [{"$arrayElemAt": ["$rows", 0]}, {"a0": 0}]}}},
        {"$replaceRoot": {"newRoot": "$row"}}, {"$project": {"_id": 0, "count": "$a0"}}]}
    bad = {"collection": "live1__p", "pipeline": [
        {"$match": {"score": {"$gt": 5}}}, {"$group": {"_id": None, "a0": {"$sum": 1}}},
        {"$project": {"_id": 0, "count": "$a0"}}]}
    item = {"id": "live1", "db_id": "live1", "sql": "SELECT count(*) FROM p WHERE score > 5"}
    assert V.verify(item, good, real)["status"] == "accepted"
    r = V.verify(item, bad, real)
    assert r["status"] == "fixture_mismatch" and r["checks"]["l1_explicit_null"]


def test_live_import_parity(live_client, tmp_path):
    path = tmp_path / "live2.sqlite"
    con = sqlite3.connect(path)
    con.executescript("CREATE TABLE q (a INTEGER, b TEXT, c REAL);"
                      "INSERT INTO q VALUES (1,'x',1.5),(NULL,'y',NULL),(1099511627776,NULL,0.0);")
    con.commit()
    con.close()
    rep = MI.import_real(live_client, "live2", path)
    assert rep["explicit_null"] == [] and rep["missing"] == []
