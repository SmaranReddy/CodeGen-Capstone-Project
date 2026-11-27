"""Stronger, differently designed fixture battery used ONLY to judge mutants that the verifier
accepted (V2 mutant study). Never used to accept a label.

Design ported from the 2026-10-25 audit (scripts/audit_sql2mql_v1/audit_fixtures.py), which is
independent of the verifier's real-row-derived families: SYNTHETIC rows drawn from small value
domains per column (a few real values, the query's literals and their neighbours / case
variants, 0/1/2, generic words), 20% NULLs, so duplicates and ties are frequent.
  rand<n>@<seed>   n in (8, 16, 30, 60) rows per table, seeds 1..3       (12 families)
  empty            every referenced table empty
  nullcols@<seed>  rand16 with every nullable non-key column NULL
  dup@<seed>       rand16 with every row duplicated (fresh integer key when there is one)
rand8/rand30 and nullcols also run with NULL stored as an absent field.
Truth = the verbatim SQL on SQLite; judgement reuses the verifier's comparator and ORDER BY
tie analysis (shared, unit-tested components; the DATA design is what differs).
"""
import random
import sqlite3

import v2_common as C
import v2_fixtures as FX
import mongo_import as MI
import verifier as V

WORDS = ["alpha", "Bravo", "charlie", "DELTA"]
SIZES = (8, 16, 30, 60)
SEEDS = (1, 2, 3)
ORACLE_DB = "sql2mql_v2_oracle"


class Oracle:
    def __init__(self, info, tables, sql, salt):
        self.info, self.tables, self.salt = info, sorted(tables), salt
        self.nums, self.strs = FX.literals(sql, info)
        self.domains = {}
        rng = random.Random(f"domain:{salt}")
        for t in self.tables:
            cols, rows = info.tables[t]
            for i, c in enumerate(cols):
                vals = [r[i] for r in rows if r[i] is not None]
                distinct = sorted(set(vals), key=lambda v: (str(type(v)), v))
                pick = rng.sample(distinct, min(3, len(distinct))) if distinct else []
                kind = info.cls[(t, c)]
                if kind == {"num"}:
                    ints = all(isinstance(v, int) for v in vals)
                    extra = [0, 1, 2] + [x for n in self.nums for x in (n - 1, n, n + 1)
                                         if not ints or isinstance(x, int)]
                    dom = pick + (extra if ints else [float(x) for x in extra] + [1.5])
                elif kind == {"text"}:
                    dom = pick + list(self.strs) + [v for s in self.strs for v in (s.upper(), s.lower())] + WORDS[:2]
                else:
                    dom = pick or [None]
                self.domains[(t, c)] = dom

    def rows(self, t, n, rng, null_p=0.2):
        cols = self.info.tables[t][0]
        pk = self.info.pk[t]
        out = []
        for k in range(n):
            r = []
            for c in cols:
                if c in pk and len(pk) == 1 and self.info.cls[(t, c)] == {"num"}:
                    r.append(k + 1)
                    continue
                nullable = c not in self.info.notnull[t] and c not in pk
                if nullable and rng.random() < null_p:
                    r.append(None)
                else:
                    r.append(rng.choice(self.domains[(t, c)]))
            out.append(tuple(r))
        return out

    def families(self):
        fams = {"empty": {t: [] for t in self.tables}}
        for seed in SEEDS:
            for n in SIZES:
                rng = random.Random(f"{self.salt}:{seed}:{n}")
                fams[f"rand{n}@{seed}"] = {t: self.rows(t, n, rng) for t in self.tables}
            base = fams[f"rand16@{seed}"]
            fams[f"nullcols@{seed}"] = {
                t: [tuple(None if (c not in self.info.notnull[t] and c not in self.info.pk[t]) else v
                          for c, v in zip(self.info.tables[t][0], r)) for r in base[t]] for t in self.tables}
            dup = {}
            for t in self.tables:
                rows = base[t]
                pk = self.info.pk[t]
                if len(pk) == 1 and self.info.cls[(t, pk[0])] == {"num"}:
                    i = self.info.tables[t][0].index(pk[0])
                    extra = [tuple(len(rows) + j + 1 if q == i else v for q, v in enumerate(r))
                             for j, r in enumerate(rows)]
                else:
                    extra = list(rows)
                dup[t] = rows + extra
            fams[f"dup@{seed}"] = dup
        return fams

    def materialize(self, rows):
        con = sqlite3.connect(":memory:")
        con.text_factory = lambda b: b.decode("utf-8", errors="surrogateescape")
        for t in self.tables:
            con.execute(self.info.ddl[t])
        stored = {}
        for t in self.tables:
            cols = self.info.tables[t][0]
            q = ", ".join(f'"{c}"' for c in cols)
            con.executemany(f'INSERT OR IGNORE INTO "{t}" ({q}) VALUES ({", ".join("?" for _ in cols)})', rows[t])
            stored[t] = (cols, [tuple(r) for r in con.execute(f'SELECT {q} FROM "{t}" ORDER BY rowid')])
        return con, stored


MISSING = ("rand8@1", "rand30@1", "nullcols@1")


def battery(item, mql, real_dbs, oracle=None):
    """-> (oracle, {family: True | False | 'error:..' | 'skipped:..'})."""
    path, real, info = real_dbs.get(item["db_id"])
    tables = FX.referenced_tables(item["sql"], info)
    oracle = oracle or Oracle(info, tables, item["sql"], item["id"])
    plan = V.order_plan(item["sql"])
    cols = V.out_columns(mql)
    res = {}
    fams = oracle.families()
    cache = getattr(oracle, "_truth", None)
    if cache is None:
        cache = oracle._truth = {}
        oracle._stored = {}
        for name, rows in fams.items():
            con, stored = oracle.materialize(rows)
            try:
                cache[name] = V.sqlite_truth(con, item["sql"], plan)
            except V.Reject as e:
                cache[name] = f"skipped:{e.status}"
            finally:
                con.close()
            oracle._stored[name] = stored
    for name in fams:
        truth = cache[name]
        if isinstance(truth, str):
            res[name] = truth
            continue
        for missing in ((False, True) if name in MISSING else (False,)):
            mdb = real_dbs.client[ORACLE_DB + ("_missing" if missing else "")]
            MI.load(mdb, item["db_id"], oracle._stored[name], missing)
            key = name + ("@missing" if missing else "")
            try:
                docs = V.run_mql(mdb, mql)
                ok, why, _ = V.compare(docs, cols, truth, plan)
                res[key] = ok
            except V.Reject as e:
                res[key] = f"error:{e.status}"
    return oracle, res
