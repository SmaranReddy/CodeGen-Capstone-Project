"""CONVERTER SIDE. Schema facts the converter needs to choose type-aware MQL (F5, F6, F7).

For one SQLite database: table and column names as stored (canonical spelling for F5), the
declared type and its SQLite affinity (https://sqlite.org/datatype3.html §3.1), primary-key
columns, row counts, and the OBSERVED storage classes of every column (typeof() scan), which
decide whether a comparison, aggregate, join or division can be translated faithfully.

Must not import the verifier (verifier.py, comparator.py, fixtures.py, mongo_import.py).
"""
import v2_common as C

FOLD_TRAPS = ("K", "ſ")   # Kelvin sign, long s: PCRE case-folds them to k / s


def affinity(decl):
    d = (decl or "").upper()
    if "INT" in d:
        return "INTEGER"
    if any(k in d for k in ("CHAR", "CLOB", "TEXT")):
        return "TEXT"
    if "BLOB" in d or not d.strip():
        return "BLOB"
    if any(k in d for k in ("REAL", "FLOA", "DOUB")):
        return "REAL"
    return "NUMERIC"


class Column:
    __slots__ = ("table", "name", "decl", "affinity", "classes", "has_null", "fold_trap")

    def __init__(self, table, name, decl):
        self.table, self.name, self.decl, self.affinity = table, name, decl, affinity(decl)
        self.classes, self.has_null, self.fold_trap = set(), False, False

    @property
    def kind(self):
        """'num', 'text', 'null' (no non-NULL value) or 'mixed' (several classes / blob)."""
        if not self.classes:
            return "null"
        if self.classes <= {"integer", "real"}:
            return "num"
        if self.classes == {"text"}:
            return "text"
        return "mixed"

    @property
    def all_int(self):
        return self.classes == {"integer"}


class Profile:
    def __init__(self, db_id, path=None):
        self.db_id = db_id
        con = C.sqlite_ro(path or C.sqlite_path(db_id))
        try:
            self.tables, self.pk, self.rows = {}, {}, {}
            names = [r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table' "
                                               "AND name NOT LIKE 'sqlite_%' ORDER BY name")]
            for t in names:
                info = con.execute(f'PRAGMA table_info("{t}")').fetchall()
                self.tables[t] = {r[1]: Column(t, r[1], r[2]) for r in info}
                self.pk[t] = [r[1] for r in sorted(info, key=lambda r: r[5]) if r[5]]
                self.rows[t] = con.execute(f'SELECT count(*) FROM "{t}"').fetchone()[0]
                for c, col in self.tables[t].items():
                    for typ, n in con.execute(f'SELECT typeof("{c}"), count(*) FROM "{t}" GROUP BY 1'):
                        if typ == "null":
                            col.has_null = True
                        else:
                            col.classes.add(typ)
                    if "text" in col.classes:
                        q = " OR ".join(f"instr(\"{c}\", '{ch}') > 0" for ch in FOLD_TRAPS)
                        col.fold_trap = bool(con.execute(f'SELECT count(*) FROM "{t}" WHERE {q}').fetchone()[0])
        finally:
            con.close()

    def table(self, name):
        """Case-insensitive table lookup (SQLite) -> canonical name or None."""
        hits = [t for t in self.tables if t.lower() == name.lower()]
        return hits[0] if len(hits) == 1 else None

    def column(self, table, name):
        hits = [c for c in self.tables[table] if c.lower() == name.lower()]
        return self.tables[table][hits[0]] if len(hits) == 1 else None
