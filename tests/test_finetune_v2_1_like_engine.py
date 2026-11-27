"""sql2mql-v2.1 LIKE vs SQLite LIKE on a real engine pair (LOCAL MongoDB 127.0.0.1:27018, not Atlas).
Plan §B.4 test 2. Skipped when the local MongoDB is unreachable.

For every pattern x value: SQLite `v LIKE p` (and `v NOT LIKE p`, NULL never returned) must equal
MongoDB's result for the v2.1 query form, the negated query form and the $regexMatch form.
Non-ASCII values are written as \\u escapes only.
"""
import random
import sqlite3
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/finetune_v2"))
import sql2mql_v2 as V2  # noqa: E402
import sql2mql_v2_1 as V21  # noqa: E402

pytestmark = pytest.mark.mongo

TABLE = [  # (pattern, values that must match, values that must not)
    ("%6862 Kaitlyn Knolls%", ["6862 Kaitlyn Knolls", "6862 kaitlyn knolls"], ["6862 Kaitlyn Knolls"]),
    ("%Scanner%", ["Scanner", "SCANNER"], ["ſcanner"]),
    ("k", ["k", "K"], ["K"]),
    ("S", ["s", "S"], ["ſ"]),
    ("%ix%", ["ix", "IX"], ["İx", "ıx"]),
    ("a_b", ["a\nb", "axb", "aKb"], ["ab", "a\n\nb"]),
    ("Ⓐ", ["Ⓐ"], ["ⓐ"]),
    ("ⅰ", ["ⅰ"], ["Ⅰ"]),
    ("%.%", ["a.b"], ["ab"]),
    ("100%", ["100", "1000 x"], ["x100"]),
    ("%(a)%", ["(A)"], ["a"]),
]
ALPHABET = list("aAbBkKsSiIzZ0 _%.\n") + ["K", "ſ", "İ", "ı", "é", "É",
                                          "Ⓐ", "ⓐ", "Ⅰ", "ⅰ", "ß", "ẞ"]


@pytest.fixture(scope="module")
def mdb():
    try:
        from pymongo import MongoClient
        c = MongoClient("127.0.0.1", 27018, directConnection=True, serverSelectionTimeoutMS=2000)
        c.admin.command("ping")
    except Exception:  # noqa: BLE001
        pytest.skip("local MongoDB 127.0.0.1:27018 not reachable")
    return c["admin"]


def corpus():
    rng = random.Random(20261113)
    vals = sorted({v for _, a, b in TABLE for v in a + b})
    for _ in range(520):
        vals.append("".join(rng.choice(ALPHABET) for _ in range(rng.randint(0, 6))))
    pats = [p for p, _, _ in TABLE]
    pat_alpha = [c for c in ALPHABET if c not in ("é", "É", "ß", "ẞ")]  # v2/v2.1 reject non-ASCII letters
    for _ in range(40):
        pats.append("".join(rng.choice(pat_alpha) for _ in range(rng.randint(1, 4))))
    return vals, sorted(set(pats))


def sqlite_like(vals, pat):
    con = sqlite3.connect(":memory:")
    con.execute("CREATE TABLE t (i INTEGER, v TEXT)")
    con.executemany("INSERT INTO t VALUES (?, ?)", list(enumerate(vals)) + [(len(vals), None)])
    pos = {i for (i,) in con.execute("SELECT i FROM t WHERE v LIKE ?", (pat,))}
    neg = {i for (i,) in con.execute("SELECT i FROM t WHERE v NOT LIKE ?", (pat,))}
    con.close()
    return pos, neg


def mongo_ids(mdb, vals, match):
    docs = [{"i": i, "v": v} for i, v in enumerate(vals)] + [{"i": len(vals), "v": None}, {"i": len(vals) + 1}]
    return {d["i"] for d in mdb.aggregate([{"$documents": docs}, {"$match": match}, {"$project": {"_id": 0, "i": 1}}])}


def forms(rx, opts):
    m = {"$regexMatch": {"input": "$v", "regex": rx, "options": opts}}
    guard = {"$eq": [{"$type": "$v"}, "string"]}
    return ({"v": {"$regex": rx, "$options": opts}},
            {"v": {"$not": {"$regex": rx, "$options": opts}, "$ne": None}},
            {"$expr": {"$and": [guard, m]}}, {"$expr": {"$and": [guard, {"$not": [m]}]}})


def test_table_expectations_hold_in_sqlite():
    for pat, yes, no in TABLE:
        pos, _ = sqlite_like(yes + no, pat)
        assert pos == set(range(len(yes))), pat


def test_v21_equals_sqlite_all_forms(mdb):
    vals, pats = corpus()
    assert len(vals) >= 500
    bad = []
    for pat in pats:
        pos, neg = sqlite_like(vals, pat)
        rx = V21.like_regex_ascii(pat)
        q, qn, e, en = forms(rx, V21.LIKE_OPTIONS)
        for name, m, want in (("query", q, pos), ("query_not", qn, neg), ("regexMatch", e, pos),
                              ("regexMatch_not", en, neg)):
            got = mongo_ids(mdb, vals, m)
            if got != want:
                bad.append((pat, name, sorted(got ^ want)[:5]))
    assert not bad, bad[:10]


def test_v2_overmatches_kelvin_and_long_s(mdb):
    """Negative control: the frozen v2 translation ('is') is wider than SQLite on the fold traps."""
    for pat, yes, no in TABLE[:4] + TABLE[6:8]:
        vals = yes + no
        pos, _ = sqlite_like(vals, pat)
        got = mongo_ids(mdb, vals, forms(V2.like_regex(pat), "is")[0])
        assert got > pos, pat          # strictly more rows: the over-match
