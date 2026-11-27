"""v2 adversarial cases (hand-specified from SQLite semantics; https://sqlite.org/datatype3.html,
lang_expr.html, lang_select.html, lang_aggfunc.html). Expectations were written by hand BEFORE
running the v2 converter; SQLite is run afterwards only as a cross-check.

Fields as in scripts/audit_sql2mql_v1/adversarial_cases.py, plus:
  schema       per-case {table: {column: DDL type}} (DDL type written verbatim into CREATE TABLE)
  pk           {table: column} declared INTEGER PRIMARY KEY... (optional)
  expect_reject  True when the only correct v2 outcome is rejection (no label)
  covers       remediation items / audit defects this case is regression coverage for
"""

S_T = {"t": {"id": "INTEGER", "name": "TEXT", "grp": "TEXT", "x": "INTEGER", "y": "REAL",
             "is_active": "BOOLEAN", "code": "TEXT"}}
T = [(1, "alice", "a", 1, 1.5, 1, "10"), (2, "Bob", "a", 3, 2.0, 0, "9"),
     (3, "carol", "b", None, None, None, "abc"), (4, None, None, 2, 0.0, 1, None),
     (5, "Alice", "b", None, 3.0, 0, "10")]


def rows(schema, table, data):
    cols = list(schema[table])
    return [dict(zip(cols, r)) for r in data]


TT = {"t": rows(S_T, "t", T)}

S_NUM = {"nums": {"id": "INTEGER", "a": "INTEGER", "b": "INTEGER", "r": "REAL"}}
NUMS = {"nums": rows(S_NUM, "nums", [(1, 7, 2, 2.0), (2, 1, 0, 0.0), (3, None, 4, 4.0), (4, 9, 3, 3.0),
                                     (5, -7, 2, 2.0)])}

S_PK = {"dept": {"dept_id": "INTEGER PRIMARY KEY", "dname": "TEXT"},
        "emp": {"emp_id": "INTEGER PRIMARY KEY", "dept_id": "INTEGER", "salary": "INTEGER"}}
PK = {"dept": rows(S_PK, "dept", [(1, "Sales"), (2, "Ops"), (3, "Lab")]),
      "emp": rows(S_PK, "emp", [(10, 1, 100), (11, 1, 200), (12, 2, 50), (13, None, 70)])}

S_MIX = {"m": {"id": "INTEGER", "v": "", "w": "INTEGER"}}          # v: no affinity (BLOB)
MIX = {"m": rows(S_MIX, "m", [(1, 5, 1), (2, "5", 2), (3, "abc", 3), (4, None, 4)])}

S_TXT = {"s": {"id": "INTEGER", "label": "TEXT", "n": "TEXT"}}
TXT = {"s": rows(S_TXT, "s", [(1, "abc", "9"), (2, "abc\n", "10"), (3, "ABC", "100"), (4, "a\nbc", "2"),
                              (5, "éb", None), (6, "$x", "07")])}

S_D = {"dates": {"id": "INTEGER", "d1": "DATE", "d2": "DATE"}}
DATES = {"dates": rows(S_D, "dates", [(1, "2020-01-01", "2020-03-01"), (2, "2021-05-05", "2021-05-10")])}


def ids(*xs):
    return [(x,) for x in xs]


CASES = [
    # F1 / B1 (A1/A1b in the audit suite): ungrouped aggregates over zero rows return one row
    dict(id="V_F1a", feature="F1 empty aggregate", covers=["F1", "A1"], schema=S_T, tables=TT,
         sql="SELECT count(*), sum(x), avg(y), min(name), max(x) FROM t WHERE x > 100",
         expect=[(0, None, None, None, None)]),
    # F2 / B2: SUM over only NULLs is NULL, COUNT(col) of NULLs is 0
    dict(id="V_F2a", feature="F2 sum all-null", covers=["F2", "A6"], schema=S_T, tables=TT,
         sql="SELECT grp, sum(x), count(x) FROM t GROUP BY grp",
         expect=[(None, 2, 1), ("a", 4, 2), ("b", None, 0)]),
    # F3: DISTINCT with GROUP BY
    dict(id="V_F3a", feature="F3 distinct+group by", covers=["F3", "DISTINCT+GROUP BY"], schema=S_T, tables=TT,
         sql="SELECT DISTINCT grp FROM t GROUP BY grp, is_active",
         expect=[("a",), ("b",), (None,)]),
    # F4: arithmetic on date text is rejected
    dict(id="V_F4a", feature="F4 date subtraction", covers=["F4", "date subtraction"], schema=S_D, tables=DATES,
         sql="SELECT avg(d2 - d1) FROM dates", expect_reject=True),
    # F5: identifiers in another case than stored
    dict(id="V_F5a", feature="F5 identifier case", covers=["F5", "identifier case"], schema=S_T, tables=TT,
         sql="SELECT NAME FROM T WHERE ID = 1", expect=[("alice",)]),
    dict(id="V_F5b", feature="F5 identifier case, alias", covers=["F5"], schema=S_PK, tables=PK,
         sql="SELECT T1.DNAME FROM DEPT AS t1 JOIN Emp AS T2 ON T1.Dept_Id = t2.DEPT_ID WHERE t2.SALARY > 150",
         expect=[("Sales",)]),
    # F6: implicit casts by column affinity
    dict(id="V_F6a", feature="F6 numeric-text literal vs INTEGER column", covers=["F6", "implicit cast"],
         schema=S_T, tables=TT, sql="SELECT id FROM t WHERE x > '2'", expect=ids(2)),
    dict(id="V_F6b", feature="F6 integer literal vs TEXT column", covers=["F6", "W17"], schema=S_T, tables=TT,
         sql="SELECT id FROM t WHERE code = 10", expect=ids(1, 5)),
    dict(id="V_F6c", feature="F6 TEXT column ordering is textual", covers=["F6"], schema=S_TXT, tables=TXT,
         sql="SELECT id FROM s WHERE n > '2'", expect=ids(1)),
    dict(id="V_F6d", feature="F6 non-numeric literal vs INTEGER column", covers=["F6"], schema=S_T, tables=TT,
         sql="SELECT id FROM t WHERE x = 'abc'", expect_reject=True),
    dict(id="V_F6e", feature="F6 mixed-class column", covers=["F6"], schema=S_MIX, tables=MIX,
         sql="SELECT id FROM m WHERE v = 5", expect_reject=True),
    dict(id="V_F6f", feature="F6 SUM over a mixed-class column", covers=["F6"], schema=S_MIX, tables=MIX,
         sql="SELECT sum(v) FROM m", expect_reject=True),
    # F7: join keys of different stored class
    dict(id="V_F7a", feature="F7 int vs text join key", covers=["F7", "J8", "int-vs-text join"],
         schema={"a": {"id": "INTEGER", "k": "TEXT"}, "b": {"k": "INTEGER", "v": "TEXT"}},
         tables={"a": [{"id": 1, "k": "1"}, {"id": 2, "k": "2"}], "b": [{"k": 1, "v": "one"}]},
         sql="SELECT a.id, b.v FROM a JOIN b ON a.k = b.k", expect_reject=True),
    # F8: integer division and zero divisors (SQLite)
    dict(id="V_F8a", feature="F8 integer division", covers=["F8", "N1"], schema=S_NUM, tables=NUMS,
         sql="SELECT id, a / b FROM nums", expect=[(1, 3), (2, None), (3, None), (4, 3), (5, -3)]),
    dict(id="V_F8b", feature="F8 real division", covers=["F8"], schema=S_NUM, tables=NUMS,
         sql="SELECT id, a / r FROM nums", expect=[(1, 3.5), (2, None), (3, None), (4, 3.0), (5, -3.5)]),
    dict(id="V_F8c", feature="F8 division in WHERE", covers=["F8", "N3"], schema=S_NUM, tables=NUMS,
         sql="SELECT id FROM nums WHERE a / b >= 3", expect=ids(1, 4)),
    # F9: ORDER BY ordinal / alias
    dict(id="V_F9a", feature="F9 ORDER BY ordinal", covers=["F9", "O4", "O5"], schema=S_T, tables=TT,
         sql="SELECT grp, count(*) FROM t GROUP BY grp ORDER BY 2 DESC, 1",
         expect=[("a", 2), ("b", 2), (None, 1)], ordered=True),
    dict(id="V_F9b", feature="F9 ORDER BY alias", covers=["F9", "O4c"], schema=S_T, tables=TT,
         sql="SELECT id, y AS score FROM t ORDER BY score DESC",
         expect=[(5, 3.0), (2, 2.0), (1, 1.5), (4, 0.0), (3, None)], ordered=True),
    # F10: NOT <bare column>
    dict(id="V_F10a", feature="F10 NOT bare column", covers=["F10", "W14"], schema=S_T, tables=TT,
         sql="SELECT id FROM t WHERE NOT is_active", expect=ids(2, 5)),
    # F11: LEFT JOIN aggregates over unmatched rows
    dict(id="V_F11a", feature="F11 LEFT JOIN count/sum", covers=["F11", "J4", "J4b"], schema=S_PK, tables=PK,
         sql="SELECT T1.dname, count(T2.emp_id), sum(T2.salary) FROM dept AS T1 LEFT JOIN emp AS T2 "
             "ON T1.dept_id = T2.dept_id GROUP BY T1.dept_id",
         expect=[("Sales", 2, 300), ("Ops", 1, 50), ("Lab", 0, None)]),
    # F12: a joined table named like a base column
    dict(id="V_F12a", feature="F12 lookup name collision", covers=["F12", "J6"],
         schema={"c": {"id": "INTEGER", "o": "INTEGER"}, "o": {"cid": "INTEGER", "amt": "REAL"}},
         tables={"c": [{"id": 1, "o": 7}], "o": [{"cid": 1, "amt": 2.5}]},
         sql="SELECT c.o, o.amt FROM c JOIN o ON c.id = o.cid", expect=[(7, 2.5)]),
    # F13: LIKE and newlines / anchors / ASCII-only folding
    dict(id="V_F13a", feature="F13 LIKE across newline", covers=["F13", "W10d", "W10e"], schema=S_TXT, tables=TXT,
         sql="SELECT id FROM s WHERE label LIKE 'a%c'", expect=ids(1, 3, 4)),
    dict(id="V_F13b", feature="F13 trailing newline is not matched by an anchored LIKE", covers=["F13"],
         schema=S_TXT, tables=TXT, sql="SELECT id FROM s WHERE label LIKE 'abc'", expect=ids(1, 3)),
    dict(id="V_F13c", feature="F13 '_' matches one character", covers=["F13"], schema=S_TXT, tables=TXT,
         sql="SELECT id FROM s WHERE label LIKE '_b'", expect=ids(5)),
    dict(id="V_F13d", feature="F13 non-ASCII pattern", covers=["F13"], schema=S_TXT, tables=TXT,
         sql="SELECT id FROM s WHERE label LIKE 'É%'", expect_reject=True),
    # F14: ROUND (and every other scalar function) is rejected
    dict(id="V_F14a", feature="F14 ROUND", covers=["F14"], schema=S_T, tables=TT,
         sql="SELECT round(y) FROM t", expect_reject=True),
    # SQLite double-quoted string literal
    dict(id="V_DQ1", feature="double-quoted string", covers=["F5"], schema=S_T, tables=TT,
         sql='SELECT id FROM t WHERE name = "alice"', expect=ids(1)),
    dict(id="V_DQ2", feature="'$' string literal", covers=["F6"], schema=S_TXT, tables=TXT,
         sql="SELECT id FROM s WHERE label = '$x'", expect=ids(6)),
    # bare columns
    dict(id="V_BC1", feature="bare column, functionally determined", covers=["bare column"], schema=S_PK, tables=PK,
         sql="SELECT T1.dname, count(*) FROM dept AS T1 JOIN emp AS T2 ON T1.dept_id = T2.dept_id "
             "GROUP BY T1.dept_id ORDER BY count(*) DESC",
         expect=[("Sales", 2), ("Ops", 1)], ordered=True),
    dict(id="V_BC2", feature="bare column, not determined", covers=["bare column"], schema=S_T, tables=TT,
         sql="SELECT name, max(x) FROM t", expect_reject=True),
    dict(id="V_LIM1", feature="LIMIT without ORDER BY", covers=["V5"], schema=S_T, tables=TT,
         sql="SELECT id FROM t LIMIT 2", expect_reject=True),
    # HAVING over a NULL aggregate
    dict(id="V_HAV1", feature="HAVING with NULL aggregate", covers=["F2"], schema=S_T, tables=TT,
         sql="SELECT grp FROM t GROUP BY grp HAVING max(x) < 5", expect=[(None,), ("a",)]),
    # numeric equality across int / real storage
    dict(id="V_NUM1", feature="int literal vs REAL column", covers=["F6"], schema=S_T, tables=TT,
         sql="SELECT id FROM t WHERE y = 2", expect=ids(2)),
    dict(id="V_AVG1", feature="AVG of integers is real", covers=["F8"], schema=S_NUM, tables=NUMS,
         sql="SELECT avg(a) FROM nums", expect=[(2.5,)]),
    dict(id="V_CD1", feature="COUNT(DISTINCT) ignores NULL", covers=["F2"], schema=S_T, tables=TT,
         sql="SELECT count(DISTINCT grp) FROM t", expect=[(2,)]),
    dict(id="V_SUB1", feature="subquery", covers=["unsupported"], schema=S_T, tables=TT,
         sql="SELECT id FROM t WHERE x > (SELECT avg(x) FROM t)", expect_reject=True),
]
