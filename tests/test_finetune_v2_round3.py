"""sql2mql-v2 verifier-3 and the round-3 G4 pre-registration (scripts/finetune_v2).
Pure tests need no MongoDB; `live_client` tests run against the local mongo:8.0.34 container
(127.0.0.1:27018) and are skipped without it. No Atlas, no network."""
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
import mongo_import as MI  # noqa: E402
import fixtures3 as FX3  # noqa: E402
import round3_stats as S  # noqa: E402

CONVERTER = {"sql2mql_v2", "schema_profile"}
VERIFIER3 = {"verifier3", "fixtures3", "index_real"}


def _imports(path):
    out = set()
    for n in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(n, ast.Import):
            out |= {a.name.split(".")[0] for a in n.names}
        elif isinstance(n, ast.ImportFrom) and n.module:
            out.add(n.module.split(".")[0])
    return out


# ---------------------------------------------------------------- independence / versioning
def test_verifier3_is_independent_of_the_converter():
    for m in VERIFIER3:
        assert not _imports(V2 / f"{m}.py") & CONVERTER, m
        assert not _imports(V2 / f"{m}.py") & {"oracle_v2", "mutants_v2", "mutant_study", "mutant_study_v3"}, m
    for m in CONVERTER:
        assert not _imports(V2 / f"{m}.py") & VERIFIER3, m


def test_verifier3_is_a_separate_version_and_v2_is_untouched():
    import verifier as V2mod
    import verifier3 as V3
    assert V2mod.VERIFIER_VERSION == "sql2mql-v2-verifier-2"
    assert V3.VERIFIER_VERSION == "sql2mql-v2-verifier-3"
    # verifier-2, judge and mutant files are unchanged since the import-line-only rename
    # (outputs/sql2mql_v2/rename_manifest.json; before it they hashed as in g4_remeasure_prereg.json)
    man = json.loads((C.OUT / "rename_manifest.json").read_text(encoding="utf-8"))["files"]
    for now, rec in (("verifier.py", "verifier.py"), ("v2_fixtures.py", "fixtures.py"),
                     ("oracle_v2.py", "oracle_v2.py"), ("mutants_v2.py", "mutants_v2.py"),
                     ("comparator.py", "comparator.py"), ("mongo_import.py", "mongo_import.py")):
        assert C.sha256_file(V2 / now) == man[f"scripts/finetune_v2/{rec}"]["sha256_after"], now
    assert V3.FIXTURE_DB != C.FIXTURE_DB and V3.REORDER_DB.startswith("sql2mql_v3")


def test_verifier3_runs_every_verifier2_family():
    import v2_fixtures as F2
    import verifier as V2mod
    import verifier3 as V3
    assert set(F2.FAMILIES) <= set(FX3.FAMILIES)
    assert set(V2mod.MISSING_FAMILIES) <= set(V3.MISSING_FAMILIES)


# ---------------------------------------------------------------- strata and statistics
def test_strata_precedence():
    assert S.stratum("SELECT a FROM t JOIN u ON t.x = u.x JOIN v ON u.y = v.y ORDER BY a LIMIT 1") == "J2"
    assert S.stratum("SELECT a FROM t JOIN u ON t.x = u.x ORDER BY a DESC LIMIT 1") == "OL"
    assert S.stratum("SELECT a FROM t ORDER BY a LIMIT 3") == "OL"
    assert S.stratum("SELECT a FROM t JOIN u ON t.x = u.x") == "J1"
    assert S.stratum("SELECT a FROM t ORDER BY a") == "G"
    assert S.stratum("SELECT a FROM t UNION SELECT b FROM u") == "G"
    assert S.stratum("not sql at all (") == "G"


def test_equal_weights_reduce_to_rounds_1_2_statistic():
    labels = [("G", 1)] * 3 + [("G", 0)] * 97
    r = S.weighted_g4(labels, {"G": 1000}, {"G": 100})
    assert r["weighted"]["n_eff"] == 100 and r["weighted"]["p_w"] == 0.03
    assert r["weighted"]["wilson95"] == list(C.wilson(3, 100)) == r["unweighted"]["wilson95"]


def test_weighted_estimate_is_not_the_raw_rate():
    # hard stratum oversampled: raw rate 5/100, population-weighted rate much lower
    labels = [("J2", 1)] * 5 + [("J2", 0)] * 45 + [("G", 0)] * 50
    r = S.weighted_g4(labels, {"J2": 100, "G": 1000}, {"J2": 50, "G": 50})
    assert r["unweighted"]["rate"] == 0.05
    assert abs(r["weighted"]["p_w"] - 10 / 1100) < 1e-4
    assert r["weighted"]["n_eff"] < 100           # Kish: unequal weights cost effective size
    assert r["by_stratum"]["J2"]["k"] == 5


def test_gate_uses_unrounded_upper_bound():
    k = S.max_k_passing(400)
    assert S.wilson_raw(k, 400)[1] <= 0.05 < S.wilson_raw(k + 0.01, 400)[1]


# ---------------------------------------------------------------- pre-registration
PRE = C.INTERIM / "round3_prereg.json"


@pytest.mark.skipif(not PRE.exists(), reason="round 3 not pre-registered")
def test_round3_prereg_is_intact_and_disjoint():
    pre = json.loads(PRE.read_text(encoding="utf-8"))
    body = {k: v for k, v in pre.items() if k != "sha256_of_body_without_this_field"}
    assert C.sha256_bytes(json.dumps(body, indent=1, ensure_ascii=False).encode("utf-8")) == \
        pre["sha256_of_body_without_this_field"]
    fr = json.loads(C.FREEZE.read_text(encoding="utf-8"))
    ids = [i for h in S.STRATA for i in pre["sampling"]["ids"][h]]
    assert len(ids) == len(set(ids)) == 740
    assert not set(ids) & (set(fr["pilot"]["ids"]) | set(fr["dev"]["ids"]))
    assert sum(pre["strata"]["N_h"].values()) == 5229
    train = {r["id"]: r for r in C.read_jsonl(C.TRAIN)}
    for h in S.STRATA:
        assert len(pre["sampling"]["ids"][h]) == pre["sampling"]["n_h_records"][h]
        assert all(S.stratum(train[i]["sql"]) == h and train[i]["tier"] == "exec_ok" for i in pre["sampling"]["ids"][h])
    assert pre["gate"]["G4_round3"].startswith("PASS iff the weighted Wilson 95% upper bound")


def test_holdout2_labels_are_reconstructed_and_refused():
    import mutant_study_v3 as MS
    hold = MS.holdout2_label_ids()
    assert len(hold) == 161
    diag = set(json.loads((C.OUT / "mutant_study" / "diagnosis_labels.json").read_text(encoding="utf-8"))["ids"])
    assert not hold & diag
    if PRE.exists():
        pre = json.loads(PRE.read_text(encoding="utf-8"))
        assert not hold & {i for h in S.STRATA for i in pre["sampling"]["ids"][h]}


def test_round3_runner_refuses_without_code_freeze(monkeypatch, tmp_path):
    import freeze_round3_code as FZ
    monkeypatch.setattr(FZ, "CODE_FREEZE", tmp_path / "missing.json")
    with pytest.raises(SystemExit):
        FZ.check_frozen()


# ---------------------------------------------------------------- fixtures3 families
@pytest.fixture(scope="module")
def chain(tmp_path_factory):
    """A -> B -> C chain (the middle table's downstream key is what `orphans` could not reach)."""
    d = tmp_path_factory.mktemp("fx3")
    path = d / "fx3.sqlite"
    con = sqlite3.connect(path)
    con.executescript("""
        CREATE TABLE club (cid INTEGER PRIMARY KEY, cname TEXT);
        CREATE TABLE mem (cid INTEGER, sid INTEGER, role TEXT);
        CREATE TABLE stu (sid INTEGER PRIMARY KEY, sname TEXT, adv INTEGER);
        INSERT INTO club VALUES (1,'Chess'),(2,'Go'),(3,'Art');
        INSERT INTO mem VALUES (1,10,'p'),(1,11,'v'),(2,12,'p'),(3,10,'x'),(2,NULL,'y');
        INSERT INTO stu VALUES (10,'ann',1),(11,'bob',NULL),(12,'cy',2),(13,'dee',3);
    """)
    con.commit()
    con.close()
    tables = MI.read_tables(path)
    info = FX3.DbInfo(path, tables)
    sql = ("SELECT count(*) FROM club AS t1 JOIN mem AS t2 ON t1.cid = t2.cid "
           "JOIN stu AS t3 ON t2.sid = t3.sid WHERE t1.cname = 'Nope'")
    return FX3.make_builder(info, {"club", "mem", "stu"}, sql, "t3", {}), info, sql


def _stored(b, fam):
    con, stored, problems = b.materialize(fam)
    return con, stored, problems


def test_dangle_changes_one_join_key_in_place(chain):
    b, info, _ = chain
    assert b.join_keys == [("club", "cid"), ("mem", "cid"), ("mem", "sid"), ("stu", "sid")]
    i = b.join_keys.index(("mem", "sid"))
    con, stored, problems = _stored(b, f"dangle{i}")
    con.close()
    assert not problems
    real = info.tables["mem"][1]
    got = stored["mem"][1]
    assert len(got) == len(real)                         # in place: no copies
    changed = [(a, g) for a, g in zip(real, got) if a != g]
    assert changed and all(a[0] == g[0] and a[2] == g[2] for a, g in changed)   # only sid changed
    stu_ids = {r[0] for r in info.tables["stu"][1]}
    assert all(g[1] not in stu_ids for _, g in changed)


def test_dangle_eq_reaches_queries_whose_where_matches_nothing(chain):
    b, _, sql = chain
    i = b.join_keys.index(("mem", "sid"))
    con, stored, problems = _stored(b, f"dangle_eq{i}")
    try:
        assert not problems
        assert any(r[1] == "Nope" for r in stored["club"][1])        # boundary_equal base
        inner = con.execute(sql).fetchone()[0]
        left = con.execute(sql.replace("JOIN stu", "LEFT JOIN stu")).fetchone()[0]
        assert left > inner        # an inner->left mutant on the last edge is now visible
    finally:
        con.close()


def test_reorder_families_keep_rows_and_reverse(chain):
    b, _, _ = chain
    con, a, _ = _stored(b, "sample_a")
    con.close()
    con, r, _ = _stored(b, "sample_a_rev")
    con.close()
    con, s, _ = _stored(b, "sample_a_shuf")
    con.close()
    for t in a:
        assert r[t][1] == list(reversed(a[t][1]))        # MongoDB insertion order reversed
        assert sorted(map(repr, s[t][1])) == sorted(map(repr, a[t][1]))


def test_keyshift_preserves_join_results(chain):
    b, info, sql = chain
    con, stored, problems = _stored(b, "keyshift")
    try:
        assert not problems
        q = sql.replace(" WHERE t1.cname = 'Nope'", "")
        base = sqlite3.connect(":memory:")
        for t in ("club", "mem", "stu"):
            base.execute(info.ddl[t])
            cols, rows = info.tables[t]
            base.executemany(f"INSERT INTO {t} VALUES ({','.join('?' for _ in cols)})", rows)
        assert con.execute(q).fetchone() == base.execute(q).fetchone()
        assert {r[0] for r in stored["club"][1]}.isdisjoint({r[0] for r in stored["stu"][1]})
    finally:
        con.close()


def test_null_keys_both_sides(chain):
    b, _, _ = chain
    con, stored, problems = _stored(b, "null_keys_both")
    con.close()
    assert not problems
    assert any(r[1] is None for r in stored["mem"][1]) and any(r[0] is None for r in stored["mem"][1])


# ---------------------------------------------------------------- review package
def test_review_selection_is_frozen_and_stratified():
    import review_package as RP
    acc = [{"id": f"x{i:03d}", "sql": "SELECT a FROM t JOIN u ON t.x=u.x JOIN v ON u.y=v.y" if i < 30 else
            "SELECT a FROM t"} for i in range(200)]
    a = RP.select(acc)
    assert a == RP.select(acc) and len(a) == 120 and len(set(a)) == 120
    j2 = sum(1 for i in a if int(i[1:]) < 30)
    assert j2 == 18                                   # 120 * 30/200
    assert len(RP.select(acc[:50])) == 50             # min(120, accepted)


def test_review_template_has_empty_judgement_columns():
    p = C.OUT / "review_dryrun" / "review_sheet.csv"
    if not p.exists():
        pytest.skip("dry run not generated")
    import csv
    import review_package as RP
    rows = list(csv.DictReader(p.open(encoding="utf-8-sig")))
    assert rows and all(all(r[c] == "" for c in RP.REVIEW_COLUMNS) for r in rows)
    assert all(r["package"].startswith("DRY RUN") for r in rows)


# ---------------------------------------------------------------- live (local MongoDB only)
@pytest.fixture(scope="module")
def live_client():
    try:
        cl = C.local_client()
        C.check_server(cl)
        return cl
    except Exception:  # noqa: BLE001
        pytest.skip("local mongo:8.0.34 on 127.0.0.1:27018 not available")


def test_live_reorder_kills_dropped_sort_before_limit1(live_client, tmp_path):
    """Rows stored in key order: `$sort` dropped before `$limit 1` passes every verifier-2 check
    on this data (natural order = sort order); verifier-3's reversed real-data run rejects it."""
    import verifier3 as V3
    path = tmp_path / "live3a.sqlite"
    con = sqlite3.connect(path)
    con.executescript("CREATE TABLE p (pid INTEGER PRIMARY KEY, name TEXT, score INTEGER);"
                      "INSERT INTO p VALUES (1,'a',1),(2,'b',2),(3,'c',3),(4,'d',4);")
    con.commit()
    con.close()
    real = V3.RealDbs(live_client, path_of=lambda db: path)
    item = {"id": "live3a", "db_id": "live3a", "sql": "SELECT name FROM p ORDER BY score LIMIT 1"}
    good = {"collection": "live3a__p", "pipeline": [{"$sort": {"score": 1}}, {"$limit": 1},
                                                    {"$project": {"_id": 0, "name": "$name"}}]}
    bad = {"collection": "live3a__p", "pipeline": [{"$limit": 1}, {"$project": {"_id": 0, "name": "$name"}}]}
    assert V3.verify(item, good, real)["status"] == "accepted"
    r = V3.verify(item, bad, real)
    assert r["status"] == "v3_reorder_mismatch", r


def test_live_lookup_on_missing_field_rejected(live_client, tmp_path):
    import verifier3 as V3
    path = tmp_path / "live3b.sqlite"
    con = sqlite3.connect(path)
    con.executescript("CREATE TABLE a (aid INTEGER PRIMARY KEY, bref INTEGER);"
                      "CREATE TABLE b (bid INTEGER PRIMARY KEY, v TEXT);"
                      "INSERT INTO a VALUES (1,1),(2,2); INSERT INTO b VALUES (1,'x'),(2,'y');")
    con.commit()
    con.close()
    real = V3.RealDbs(live_client, path_of=lambda db: path)
    item = {"id": "live3b", "db_id": "live3b", "sql": "SELECT T2.v FROM a AS T1 JOIN b AS T2 ON T1.bref = T2.bid"}
    swapped = {"collection": "live3b__a", "pipeline": [
        {"$lookup": {"from": "live3b__b", "localField": "bid", "foreignField": "bref", "as": "__j1"}},
        {"$unwind": "$__j1"}, {"$project": {"_id": 0, "v": "$__j1.v"}}]}
    r = V3.verify(item, swapped, real)
    assert r["status"] == "v3_lookup_keys", r


def test_resume_drops_incomplete_label_and_cut_line(tmp_path):
    import mutants_v2 as MU
    import mutant_study_v3 as MS
    mql = {"collection": "d__t", "pipeline": [{"$match": {"a": {"$gt": 1}}}, {"$sort": {"a": 1}}, {"$limit": 1},
                                              {"$project": {"_id": 0, "a": "$a", "b": "$b"}}]}
    n = len(MU.mutants(mql)[:30])
    assert n >= 3
    recs = {"L1": {"mql": mql}, "L2": {"mql": mql}}
    p = tmp_path / "mutants.jsonl"
    full = [json.dumps({"label": "L1", "k": i}) for i in range(n)]
    part = [json.dumps({"label": "L2", "k": i}) for i in range(n - 1)]
    p.write_text("\n".join(full + part) + '\n{"label": "L2", "k"', encoding="utf-8")
    rows = MS.resume_state(p, recs, tmp_path)
    assert {x["label"] for x in rows} == {"L1"} and len(rows) == n
    assert len(C.read_jsonl(p)) == n
    log = C.read_jsonl(tmp_path / "resume_log.jsonl")[-1]
    assert log["incomplete_labels_dropped"] == ["L2"] and log["cut_trailing_line_dropped"] == 1
