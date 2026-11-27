"""Adversarial cases for sql2mql-v1 with independently specified expected results.

Every `expect` below was written by hand from SQL semantics (ISO three-valued logic, the
documented MySQL 8 / SQLite behaviour named in `dialect`), NOT by running the audited
code. SQLite is executed afterwards only as a cross-check; a disagreement is reported, not
silently resolved. Clock cases compute their expectation in plain Python from the server
clock N (no sqlsem code).

Fields: id, feature, sql, tables (dict or callable(N) -> dict), expect (list of tuples, or
callable(N)), ordered, any_of (list of acceptable answers, for ties), subset_k (LIMIT
without ORDER BY: any k distinct rows of the base table are acceptable), dialect (note when
MySQL and SQLite disagree; the expectation follows the semantics named first).
"""
import datetime as dt

SCHEMA = {
    "t": {"id": "int", "name": "text", "grp": "text", "x": "int", "y": "float", "is_active": "bool",
          "d": "date", "code": "text"},
    "e": {"id": "int", "v": "text"},
    "l": {"id": "int", "name": "text"},
    "customers": {"id": "int", "name": "text", "orders": "int"},
    "orders": {"id": "int", "cust_id": "int", "amt": "float"},
    "orders_t": {"id": "int", "cust_id": "text"},
    "nums": {"id": "int", "a": "int", "b": "int"},
    "ev": {"id": "int", "ts": "datetime", "d": "date"},
}


def _rows(table, *rows):
    cols = list(SCHEMA[table])
    return [dict(zip(cols, r)) for r in rows]


T = _rows("t",
          (1, "alice", "a", 1, 1.5, 1, "2024-03-01", "10"),
          (2, "Bob", "a", 3, 2.0, 0, "2023-12-31", "9"),
          (3, "carol", "b", None, None, None, None, "abc"),
          (4, None, None, 2, 0.0, 1, "2024-05-15", None),
          (5, "Alice", "b", None, 3.0, 0, "2024-03-20", "10"))
L = _rows("l", (1, "a.b"), (2, "axb"), (3, "v1.2"), (4, "v12"), (5, "a+b"), (6, None), (7, "ab\ncd"))
CUST = _rows("customers", (1, "Ann", 2), (2, "Ben", 0), (3, "Cid", 1), (None, "Eve", None))
ORD = _rows("orders", (10, 1, 5.0), (11, 1, 7.5), (12, 3, None), (13, None, 4.0), (14, 99, 1.0))
ORD_T = _rows("orders_t", (20, "1"), (21, "3"), (22, "x"))
NUMS = _rows("nums", (1, 7, 2), (2, 1, 0), (3, None, 4), (4, 9, 3))


def ev_tables(N):
    f = lambda d: d.strftime("%Y-%m-%d %H:%M:%S")  # noqa: E731
    g = lambda d: d.strftime("%Y-%m-%d")  # noqa: E731
    h = dt.timedelta(hours=1)
    day = dt.timedelta(days=1)
    return {"ev": _rows("ev",
                        (1, f(N - 2 * h), g(N)),
                        (2, f(N - 25 * h), g(N - day)),
                        (3, f(N + h), g(N + day)),
                        (4, None, None),
                        (5, f(N - 6 * day), g(N - 6 * day)),
                        (6, f(N - 8 * day), g(N - 8 * day)))}


def _ev_ids(N, pred):
    """Apply `pred(ts: datetime|None, d: date|None)` to the ev rows in plain Python."""
    out = []
    for r in ev_tables(N)["ev"]:
        ts = dt.datetime.strptime(r["ts"], "%Y-%m-%d %H:%M:%S") if r["ts"] else None
        d = dt.datetime.strptime(r["d"], "%Y-%m-%d").date() if r["d"] else None
        if pred(ts, d) is True:
            out.append((r["id"],))
    return out


def ids(*xs):
    return [(x,) for x in xs]


TT = {"t": T}
CASES = [
    # ---------------------------------------------------------------- projection
    dict(id="P1", feature="projection/alias", sql="SELECT id, name AS n FROM t", tables=TT,
         expect=[(1, "alice"), (2, "Bob"), (3, "carol"), (4, None), (5, "Alice")]),
    dict(id="P2", feature="projection/arithmetic+NULL", sql="SELECT id, x + 1 AS x1 FROM t", tables=TT,
         expect=[(1, 2), (2, 4), (3, None), (4, 3), (5, None)]),
    dict(id="P3", feature="projection/string literal", sql="SELECT id, 'k' AS c FROM t WHERE id = 1", tables=TT,
         expect=[(1, "k")]),
    dict(id="P4", feature="projection/numeric literal", sql="SELECT id, 7 AS c FROM t WHERE id = 1", tables=TT,
         expect=[(1, 7)]),
    dict(id="P5", feature="projection/int*float+NULL", sql="SELECT id, x * y AS p FROM t", tables=TT,
         expect=[(1, 1.5), (2, 6.0), (3, None), (4, 0.0), (5, None)]),
    dict(id="P6", feature="projection/star", sql="SELECT * FROM nums WHERE id = 3", tables={"nums": NUMS},
         expect=[(3, None, 4)]),
    # ---------------------------------------------------------------- WHERE / 3VL
    dict(id="W1", feature="where/compare", sql="SELECT id FROM t WHERE x > 1", tables=TT, expect=ids(2, 4)),
    dict(id="W2", feature="where/<> excludes NULL", sql="SELECT id FROM t WHERE x <> 3", tables=TT, expect=ids(1, 4)),
    dict(id="W3", feature="where/NOT = excludes NULL", sql="SELECT id FROM t WHERE NOT (x = 3)", tables=TT,
         expect=ids(1, 4)),
    dict(id="W4", feature="where/NOT(AND) 3VL", sql="SELECT id FROM t WHERE NOT (x > 1 AND grp = 'a')", tables=TT,
         expect=ids(1, 3, 5)),
    dict(id="W4b", feature="where/NOT(OR) 3VL", sql="SELECT id FROM t WHERE NOT (x = 1 OR grp = 'b')", tables=TT,
         expect=ids(2)),
    dict(id="W5", feature="where/IN", sql="SELECT id FROM t WHERE x IN (1, 3)", tables=TT, expect=ids(1, 2)),
    dict(id="W5b", feature="where/NOT IN excludes NULL", sql="SELECT id FROM t WHERE x NOT IN (1, 3)", tables=TT,
         expect=ids(4)),
    dict(id="W6", feature="where/BETWEEN", sql="SELECT id FROM t WHERE x BETWEEN 2 AND 3", tables=TT,
         expect=ids(2, 4)),
    dict(id="W6b", feature="where/NOT BETWEEN excludes NULL", sql="SELECT id FROM t WHERE x NOT BETWEEN 2 AND 3",
         tables=TT, expect=ids(1)),
    dict(id="W7", feature="where/LIKE prefix case-insensitive", sql="SELECT id FROM t WHERE name LIKE 'a%'",
         tables=TT, expect=ids(1, 5)),
    dict(id="W8", feature="where/NOT LIKE excludes NULL", sql="SELECT id FROM t WHERE name NOT LIKE 'a%'",
         tables=TT, expect=ids(2, 3)),
    dict(id="W9", feature="where/LIKE underscore", sql="SELECT id FROM t WHERE name LIKE '_ob'", tables=TT,
         expect=ids(2)),
    dict(id="W10", feature="where/LIKE literal dot", sql="SELECT id FROM l WHERE name LIKE '%.%'", tables={"l": L},
         expect=ids(1, 3)),
    dict(id="W10b", feature="where/LIKE dot then %", sql="SELECT id FROM l WHERE name LIKE 'v1.%'", tables={"l": L},
         expect=ids(3)),
    dict(id="W10c", feature="where/LIKE regex metachar +", sql="SELECT id FROM l WHERE name LIKE 'a+%'",
         tables={"l": L}, expect=ids(5)),
    dict(id="W10d", feature="where/LIKE _ matches newline", sql="SELECT id FROM l WHERE name LIKE 'ab_cd'",
         tables={"l": L}, expect=ids(7)),
    dict(id="W10e", feature="where/LIKE % spans newline", sql="SELECT id FROM l WHERE name LIKE 'a%d'",
         tables={"l": L}, expect=ids(7)),
    dict(id="W10f", feature="where/LIKE '%' excludes NULL", sql="SELECT id FROM l WHERE name LIKE '%'",
         tables={"l": L}, expect=ids(1, 2, 3, 4, 5, 7)),
    dict(id="W12", feature="where/IS NULL", sql="SELECT id FROM t WHERE x IS NULL", tables=TT, expect=ids(3, 5)),
    dict(id="W12b", feature="where/IS NOT NULL", sql="SELECT id FROM t WHERE x IS NOT NULL", tables=TT,
         expect=ids(1, 2, 4)),
    dict(id="W13", feature="where/bare boolean", sql="SELECT id FROM t WHERE is_active", tables=TT,
         expect=ids(1, 4)),
    dict(id="W14", feature="where/NOT bare boolean excludes NULL", sql="SELECT id FROM t WHERE NOT is_active",
         tables=TT, expect=ids(2, 5)),
    dict(id="W15", feature="where/OR", sql="SELECT id FROM t WHERE x = 1 OR grp = 'b'", tables=TT,
         expect=ids(1, 3, 5)),
    dict(id="W16", feature="where/string = case", sql="SELECT id FROM t WHERE name = 'alice'", tables=TT,
         expect=ids(1), dialect="SQLite/ISO: case-sensitive -> [1]; MySQL default collation (ci) -> [1, 5]"),
    dict(id="W17", feature="where/numeric literal vs text column", sql="SELECT id FROM t WHERE code = 10",
         tables=TT, expect=ids(1, 5), dialect="SQLite affinity and MySQL numeric coercion both -> [1, 5]"),
    dict(id="W19", feature="where/column vs column NULL", sql="SELECT id FROM t WHERE x < y", tables=TT,
         expect=ids(1)),
    dict(id="W19b", feature="where/column vs column NULL (flipped)", sql="SELECT id FROM t WHERE y > x", tables=TT,
         expect=ids(1)),
    dict(id="W20", feature="where/date string compare", sql="SELECT id FROM t WHERE d >= '2024-03-01'", tables=TT,
         expect=ids(1, 4, 5)),
    dict(id="W21", feature="where/YEAR()", sql="SELECT id FROM t WHERE YEAR(d) = 2024", tables=TT,
         expect=ids(1, 4, 5)),
    dict(id="W22", feature="where/MONTH()", sql="SELECT id FROM t WHERE MONTH(d) = 3", tables=TT, expect=ids(1, 5)),
    dict(id="W27", feature="where/LOWER() <> excludes NULL", sql="SELECT id FROM t WHERE LOWER(name) <> 'bob'",
         tables=TT, expect=ids(1, 3, 5)),
    dict(id="W27b", feature="where/LOWER() =", sql="SELECT id FROM t WHERE LOWER(name) = 'alice'", tables=TT,
         expect=ids(1, 5)),
    dict(id="W28", feature="where/LENGTH()", sql="SELECT id FROM t WHERE LENGTH(name) > 3", tables=TT,
         expect=ids(1, 3, 5)),
    dict(id="W29", feature="where/UPPER()", sql="SELECT id FROM t WHERE UPPER(grp) = 'A'", tables=TT,
         expect=ids(1, 2)),
    dict(id="W30", feature="where/= TRUE", sql="SELECT id FROM t WHERE is_active = TRUE", tables=TT,
         expect=ids(1, 4)),
    # ---------------------------------------------------------------- aggregates
    dict(id="A1", feature="agg/COUNT(*) no matching rows", sql="SELECT COUNT(*) FROM t WHERE x > 100", tables=TT,
         expect=[(0,)]),
    dict(id="A1b", feature="agg/COUNT(*) empty table", sql="SELECT COUNT(*) FROM e", tables={"e": []},
         expect=[(0,)]),
    dict(id="A2", feature="agg/SUM no matching rows", sql="SELECT SUM(x) FROM t WHERE x > 100", tables=TT,
         expect=[(None,)]),
    dict(id="A3", feature="agg/MAX no matching rows", sql="SELECT MAX(x) FROM t WHERE x > 100", tables=TT,
         expect=[(None,)]),
    dict(id="A4", feature="agg/AVG ignores NULL", sql="SELECT AVG(x) FROM t", tables=TT, expect=[(2.0,)]),
    dict(id="A5", feature="agg/COUNT(col) ignores NULL", sql="SELECT COUNT(x) FROM t", tables=TT, expect=[(3,)]),
    dict(id="A5b", feature="agg/COUNT(text col) ignores NULL", sql="SELECT COUNT(name) FROM t", tables=TT,
         expect=[(4,)]),
    dict(id="A6", feature="agg/SUM all-NULL group", sql="SELECT grp, SUM(x) FROM t GROUP BY grp", tables=TT,
         expect=[("a", 4), ("b", None), (None, 2)]),
    dict(id="A6b", feature="agg/SUM all-NULL ungrouped", sql="SELECT SUM(x) FROM t WHERE grp = 'b'", tables=TT,
         expect=[(None,)]),
    dict(id="A7", feature="agg/COUNT(DISTINCT) text", sql="SELECT COUNT(DISTINCT grp) FROM t", tables=TT,
         expect=[(2,)]),
    dict(id="A7b", feature="agg/COUNT(DISTINCT) int", sql="SELECT COUNT(DISTINCT x) FROM t", tables=TT,
         expect=[(3,)]),
    dict(id="A8", feature="agg/GROUP BY with NULL key", sql="SELECT grp, COUNT(*) FROM t GROUP BY grp", tables=TT,
         expect=[("a", 2), ("b", 2), (None, 1)]),
    dict(id="A9", feature="agg/HAVING", sql="SELECT grp, COUNT(*) AS c FROM t GROUP BY grp HAVING COUNT(*) > 1",
         tables=TT, expect=[("a", 2), ("b", 2)]),
    dict(id="A9b", feature="agg/HAVING alias", sql="SELECT grp, COUNT(*) AS c FROM t GROUP BY grp HAVING c > 1",
         tables=TT, expect=[("a", 2), ("b", 2)]),
    dict(id="A10", feature="agg/MIN(text) MAX(float) ORDER BY key", tables=TT,
         sql="SELECT grp, MIN(name), MAX(y) FROM t GROUP BY grp ORDER BY grp", ordered=True,
         expect=[(None, None, 0.0), ("a", "Bob", 2.0), ("b", "Alice", 3.0)],
         dialect="binary collation (SQLite) -> MIN('alice','Bob')='Bob'; MySQL ci -> 'alice'"),
    dict(id="A11", feature="agg/AVG per group", sql="SELECT grp, AVG(y) FROM t GROUP BY grp", tables=TT,
         expect=[("a", 1.75), ("b", 3.0), (None, 0.0)]),
    dict(id="A13", feature="agg/COUNT(col) per group", sql="SELECT grp, COUNT(x) FROM t GROUP BY grp", tables=TT,
         expect=[("a", 2), ("b", 0), (None, 1)]),
    dict(id="A14", feature="agg/MIN(date)", sql="SELECT MIN(d) FROM t", tables=TT, expect=[("2023-12-31",)]),
    dict(id="A15", feature="agg/HAVING agg IS NULL",
         sql="SELECT grp, COUNT(*) FROM t GROUP BY grp HAVING SUM(x) IS NULL", tables=TT, expect=[("b", 2)]),
    # ---------------------------------------------------------------- DISTINCT
    dict(id="D1", feature="distinct/NULL once", sql="SELECT DISTINCT grp FROM t", tables=TT,
         expect=[("a",), ("b",), (None,)]),
    dict(id="D2", feature="distinct/two columns", sql="SELECT DISTINCT grp, is_active FROM t", tables=TT,
         expect=[("a", 1), ("a", 0), ("b", None), (None, 1), ("b", 0)]),
    dict(id="D3", feature="distinct/ORDER BY DESC (NULL last)", sql="SELECT DISTINCT grp FROM t ORDER BY grp DESC",
         tables=TT, ordered=True, expect=[("b",), ("a",), (None,)]),
    dict(id="D4", feature="distinct/ORDER BY ordinal", sql="SELECT DISTINCT grp FROM t ORDER BY 1 DESC", tables=TT,
         ordered=True, expect=[("b",), ("a",), (None,)]),
    dict(id="D5", feature="distinct/ORDER BY LIMIT (NULL first)", sql="SELECT DISTINCT x FROM t ORDER BY x LIMIT 2",
         tables=TT, ordered=True, expect=[(None,), (1,)]),
    # ---------------------------------------------------------------- ORDER BY / LIMIT
    dict(id="O1", feature="order/ASC NULL first", sql="SELECT id, x FROM t ORDER BY x, id", tables=TT, ordered=True,
         expect=[(3, None), (5, None), (1, 1), (4, 2), (2, 3)]),
    dict(id="O2", feature="order/DESC NULL last", sql="SELECT id, x FROM t ORDER BY x DESC, id", tables=TT,
         ordered=True, expect=[(2, 3), (4, 2), (1, 1), (3, None), (5, None)]),
    dict(id="O3", feature="order/multi-key mixed direction", sql="SELECT id FROM t ORDER BY grp, x DESC, id",
         tables=TT, ordered=True, expect=ids(4, 2, 1, 3, 5)),
    dict(id="O4", feature="order/ordinal on aliased column", sql="SELECT x AS val FROM t ORDER BY 1", tables=TT,
         ordered=True, expect=[(None,), (None,), (1,), (2,), (3,)]),
    dict(id="O4b", feature="order/ordinal on plain columns", sql="SELECT id, x FROM t ORDER BY 2 DESC, 1",
         tables=TT, ordered=True, expect=[(2, 3), (4, 2), (1, 1), (3, None), (5, None)]),
    dict(id="O4c", feature="order/ordinal on expression", sql="SELECT id, x + 0 FROM t ORDER BY 2 DESC, 1",
         tables=TT, ordered=True, expect=[(2, 3), (4, 2), (1, 1), (3, None), (5, None)]),
    dict(id="O5", feature="order/ordinal on aggregate",
         sql="SELECT grp, COUNT(*) FROM t GROUP BY grp ORDER BY 2 DESC, 1", tables=TT, ordered=True,
         expect=[("a", 2), ("b", 2), (None, 1)]),
    dict(id="O6", feature="order/DESC LIMIT", sql="SELECT name, y FROM t ORDER BY y DESC LIMIT 2", tables=TT,
         ordered=True, expect=[("Alice", 3.0), ("Bob", 2.0)]),
    dict(id="O7", feature="order/alias of expression", sql="SELECT x * 2 AS dbl FROM t ORDER BY dbl DESC",
         tables=TT, ordered=True, expect=[(6,), (4,), (2,), (None,), (None,)]),
    dict(id="O8", feature="order/tie at LIMIT boundary", sql="SELECT id FROM t ORDER BY is_active DESC LIMIT 1",
         tables=TT, ordered=True, any_of=[ids(1), ids(4)]),
    dict(id="O9", feature="limit/without ORDER BY", sql="SELECT id FROM t LIMIT 2", tables=TT, subset_k=2),
    dict(id="O10", feature="order/LIMIT OFFSET", sql="SELECT id FROM t ORDER BY id LIMIT 2 OFFSET 1", tables=TT,
         ordered=True, expect=ids(2, 3)),
    dict(id="O11", feature="order/string collation", sql="SELECT name FROM t ORDER BY name", tables=TT,
         ordered=True, expect=[(None,), ("Alice",), ("Bob",), ("alice",), ("carol",)],
         dialect="binary (SQLite) order; MySQL ci collation interleaves 'alice'/'Alice'"),
    dict(id="O12", feature="order/float with NULL", sql="SELECT id FROM t ORDER BY y", tables=TT, ordered=True,
         expect=ids(3, 4, 1, 2, 5)),
    # ---------------------------------------------------------------- JOIN
    dict(id="J1", feature="join/inner, NULL and dangling keys",
         sql="SELECT c.name, o.amt FROM orders o JOIN customers c ON o.cust_id = c.id",
         tables={"customers": CUST, "orders": ORD}, expect=[("Ann", 5.0), ("Ann", 7.5), ("Cid", None)]),
    dict(id="J2", feature="join/inner + WHERE on joined",
         sql="SELECT o.id, c.name FROM customers c JOIN orders o ON c.id = o.cust_id WHERE o.amt > 5",
         tables={"customers": CUST, "orders": ORD}, expect=[(11, "Ann")]),
    dict(id="J3", feature="join/LEFT keeps unmatched",
         sql="SELECT c.name, o.id FROM customers c LEFT JOIN orders o ON c.id = o.cust_id",
         tables={"customers": CUST, "orders": ORD},
         expect=[("Ann", 10), ("Ann", 11), ("Ben", None), ("Cid", 12), ("Eve", None)]),
    dict(id="J4", feature="join/LEFT + COUNT(right col)",
         sql="SELECT c.name, COUNT(o.id) FROM customers c LEFT JOIN orders o ON c.id = o.cust_id GROUP BY c.name",
         tables={"customers": CUST, "orders": ORD}, expect=[("Ann", 2), ("Ben", 0), ("Cid", 1), ("Eve", 0)]),
    dict(id="J4b", feature="join/LEFT + SUM(right col)",
         sql="SELECT c.name, SUM(o.amt) FROM customers c LEFT JOIN orders o ON c.id = o.cust_id GROUP BY c.name",
         tables={"customers": CUST, "orders": ORD},
         expect=[("Ann", 12.5), ("Ben", None), ("Cid", None), ("Eve", None)]),
    dict(id="J5", feature="join/LEFT anti-join IS NULL",
         sql="SELECT c.name FROM customers c LEFT JOIN orders o ON c.id = o.cust_id WHERE o.id IS NULL",
         tables={"customers": CUST, "orders": ORD}, expect=[("Ben",), ("Eve",)]),
    dict(id="J6", feature="join/$lookup 'as' collides with base column",
         sql="SELECT c.name, c.orders, o.amt FROM customers c JOIN orders o ON c.id = o.cust_id",
         tables={"customers": CUST, "orders": ORD}, expect=[("Ann", 2, 5.0), ("Ann", 2, 7.5), ("Cid", 1, None)]),
    dict(id="J7", feature="join/COUNT with filter on joined",
         sql="SELECT COUNT(*) FROM orders o JOIN customers c ON o.cust_id = c.id WHERE c.name = 'Ann'",
         tables={"customers": CUST, "orders": ORD}, expect=[(2,)]),
    dict(id="J8", feature="join/key type mismatch (text vs int)",
         sql="SELECT o.id, c.name FROM orders_t o JOIN customers c ON o.cust_id = c.id",
         tables={"customers": CUST, "orders_t": ORD_T}, expect=[(20, "Ann"), (21, "Cid")],
         dialect="SQLite numeric affinity and MySQL coercion both match '1' = 1"),
    dict(id="J9", feature="join/LEFT + WHERE on right (becomes inner)",
         sql="SELECT c.name, o.id FROM customers c LEFT JOIN orders o ON c.id = o.cust_id WHERE o.amt > 5",
         tables={"customers": CUST, "orders": ORD}, expect=[("Ann", 11)]),
    dict(id="J10", feature="join/LEFT + COUNT(*)",
         sql="SELECT c.name, COUNT(*) FROM customers c LEFT JOIN orders o ON c.id = o.cust_id GROUP BY c.name",
         tables={"customers": CUST, "orders": ORD}, expect=[("Ann", 2), ("Ben", 1), ("Cid", 1), ("Eve", 1)]),
    # ---------------------------------------------------------------- numeric
    dict(id="N1", feature="numeric/division incl. /0", sql="SELECT id, a / b FROM nums", tables={"nums": NUMS},
         expect=[(1, 3.5), (2, None), (3, None), (4, 3.0)],
         dialect="MySQL: 7/2=3.5, x/0=NULL; SQLite native: 7/2=3 (integer), x/0=NULL"),
    dict(id="N1b", feature="numeric/division no zero divisor", sql="SELECT id, a / b FROM nums WHERE b <> 0",
         tables={"nums": NUMS}, expect=[(1, 3.5), (3, None), (4, 3.0)]),
    dict(id="N3", feature="numeric/division in WHERE with /0 row", sql="SELECT id FROM nums WHERE a / b > 3",
         tables={"nums": NUMS}, expect=ids(1)),
    dict(id="N4", feature="numeric/subtraction NULL", sql="SELECT id, a - b FROM nums", tables={"nums": NUMS},
         expect=[(1, 5), (2, 1), (3, None), (4, 6)]),
    # ---------------------------------------------------------------- clock (expected from plain Python)
    dict(id="C1", feature="clock/NOW() - INTERVAL 1 DAY", tables=ev_tables,
         sql="SELECT id FROM ev WHERE ts >= NOW() - INTERVAL 1 DAY",
         expect=lambda N: _ev_ids(N, lambda ts, d: None if ts is None else ts >= N - dt.timedelta(days=1))),
    dict(id="C2", feature="clock/= CURDATE()", tables=ev_tables, sql="SELECT id FROM ev WHERE d = CURDATE()",
         expect=lambda N: _ev_ids(N, lambda ts, d: None if d is None else d == N.date())),
    dict(id="C3", feature="clock/DATE_SUB(CURDATE(), INTERVAL 7 DAY)", tables=ev_tables,
         sql="SELECT id FROM ev WHERE d >= DATE_SUB(CURDATE(), INTERVAL 7 DAY)",
         expect=lambda N: _ev_ids(N, lambda ts, d: None if d is None else d >= N.date() - dt.timedelta(days=7))),
    dict(id="C4", feature="clock/< NOW()", tables=ev_tables, sql="SELECT id FROM ev WHERE ts < NOW()",
         expect=lambda N: _ev_ids(N, lambda ts, d: None if ts is None else ts < N)),
    dict(id="C5", feature="clock/BETWEEN NOW()-7d AND NOW()", tables=ev_tables,
         sql="SELECT id FROM ev WHERE ts BETWEEN NOW() - INTERVAL 7 DAY AND NOW()",
         expect=lambda N: _ev_ids(N, lambda ts, d: None if ts is None else N - dt.timedelta(days=7) <= ts <= N)),
    dict(id="C6", feature="clock/DATE(ts) = CURDATE()", tables=ev_tables,
         sql="SELECT id FROM ev WHERE DATE(ts) = CURDATE()",
         expect=lambda N: _ev_ids(N, lambda ts, d: None if ts is None else ts.date() == N.date())),
    dict(id="C7", feature="clock/YEAR(ts) = YEAR(NOW())", tables=ev_tables,
         sql="SELECT id FROM ev WHERE YEAR(ts) = YEAR(NOW())",
         expect=lambda N: _ev_ids(N, lambda ts, d: None if ts is None else ts.year == N.year)),
    dict(id="C8", feature="clock/COUNT within 30 days", tables=ev_tables,
         sql="SELECT COUNT(*) FROM ev WHERE ts >= NOW() - INTERVAL 30 DAY",
         expect=lambda N: [(len(_ev_ids(N, lambda ts, d: None if ts is None else ts >= N - dt.timedelta(days=30))),)]),
    dict(id="C9", feature="clock/datetime >= CURDATE()", tables=ev_tables,
         sql="SELECT id FROM ev WHERE ts >= CURDATE()",
         expect=lambda N: _ev_ids(N, lambda ts, d: None if ts is None else ts >= dt.datetime.combine(N.date(), dt.time()))),
]

# Gold-side (sqlsem SQLite UDF) checks against hand-computed MySQL 8 results.
UDF_CHECKS = [
    ("_DTADD", ("2024-03-31", -1, "month"), "2024-02-29"),
    ("_DTADD", ("2024-01-31 10:00:00", 1, "month"), "2024-02-29 10:00:00"),
    ("_DTADD", ("2024-02-29", 1, "year"), "2025-02-28"),
    ("_DTADD", ("2024-05-01", -1, "hour"), "2024-04-30 23:00:00"),
    ("_DTADD", ("2024-12-31", 1, "day"), "2025-01-01"),
    ("_DTADD", ("2024-03-10", -2, "week"), "2024-02-25"),
    ("_YEAR", ("2024-03-01 10:00:00",), 2024),
    ("_MONTH", ("2024-03-01",), 3),
    ("_DAY", ("2024-03-09",), 9),
    ("_DATE", ("2024-05-01 23:59:59",), "2024-05-01"),
    ("_YEAR", ("not a date",), None),
]
# Mongo-side date arithmetic (the converter's $dateSubtract/$dateAdd) vs MySQL 8.
MONGO_DATE_CHECKS = [
    ({"$dateSubtract": {"startDate": {"$dateFromString": {"dateString": "2024-03-31"}}, "unit": "month", "amount": 1}},
     "2024-02-29"),
    ({"$dateAdd": {"startDate": {"$dateFromString": {"dateString": "2024-01-31"}}, "unit": "month", "amount": 1}},
     "2024-02-29"),
    ({"$dateAdd": {"startDate": {"$dateFromString": {"dateString": "2024-02-29"}}, "unit": "year", "amount": 1}},
     "2025-02-28"),
]
