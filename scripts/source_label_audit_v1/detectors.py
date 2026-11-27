"""Source-label quality detectors D1-D6 (plan Part D). READ-ONLY over the SQLite databases; no SQL is repaired.

Each detector takes a Ctx (record + parsed SQL + schema facts) and returns a list of hits:
    {"detector": "D1", "subclass": "D1a", "action": "quarantine" | "report_only", "evidence": {...}}
A hit means "quarantine pending AI-panel review"; a non-hit certifies nothing.

Thresholds and word lists live in PARAMS / LEXICON; their sha256 is frozen in prereg.json before any pool run.
"""
import re
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

import sqlglot
from sqlglot import exp

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/finetune_v2"))
import type_contract as TC  # noqa: E402  (converter-independent)

SPIDER_DB = ROOT / "data/raw/spider1/spider_data/database"
DETECTOR_VERSION = "source-label-detectors-v1"

PARAMS = {
    "D1_parse_share": 0.90,          # share of non-null values that must parse as number / date
    "D1_max_values": 20000,
    # phrasal-verb particles (in/to/from/on/at/for/by) are excluded: "logged in", "belong to", "come from"
    "D5_dangling_words": ["of", "with", "than", "a", "an", "the", "named", "called", "titled", "and", "or", "whose",
                          "that", "which"],
    "D5_short_words_per_atom": 2, "D5_short_base": 3, "D5_short_min_atoms": 2,
    "sqlite_timeout_steps": 2_000_000,
}
LEXICON = {
    "D2_recent": [r"\bmost recent(ly)?\b", r"\blatest\b", r"\bnewest\b", r"\bmost lately\b"],
    "D2_earliest": [r"\bearliest\b", r"\bleast recent(ly)?\b"],
    "D2_youngest": [r"\byoungest\b"],
    "D2_oldest": [r"\boldest\b"],
    "D2_date_col": r"(date|time|year|dob|birth|created|_at$|day|when)",
    "D2_birth_col": r"(birth|dob|born)",
    "D2_age_col": r"(^|_)age($|_)|^age",
    "D3_all": [r"\ball\b", r"\beach\b", r"\bevery\b", r"\bfor each\b", r"\bshow\b.*\bwith (the )?number of\b"],
    "D4_counted": r"\b(?:number of|how many|count of|count the number of|total number of)\s+(?:different\s+|distinct\s+|unique\s+)?([a-z]+)",
    "D5_has_name": r"\b(that|which|who|whose)\s+(has|have)\s+(\w+\s+){0,2}(name|title)$",
    "D6_implicit": {"1": ["won", "win", "wins", "winner", "first", "one", "yes", "true", "single"],
                    "0": ["no", "not", "zero", "lost", "false", "none"],
                    "t": ["true", "yes", "won", "win"], "f": ["female", "false", "women", "woman", "girl", "girls", "no"],
                    "m": ["male", "men", "man", "boy", "boys"], "y": ["yes", "true"], "n": ["no", "not", "false"]},
    "D6_number_words": {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
                        "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "twenty": 20, "hundred": 100,
                        "thousand": 1000, "million": 1000000, "first": 1, "second": 2, "third": 3, "once": 1,
                        "twice": 2, "single": 1, "double": 2, "half": 0.5, "dozen": 12},
}
NUM = re.compile(r"^\s*[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?\s*$")
ISO = re.compile(r"^(\d{4})-(\d{1,2})-(\d{1,2})(?:[ T](\d{1,2}):(\d{2})(?::(\d{2})(?:\.\d+)?)?)?$")
ISO_PADDED = re.compile(r"^\d{4}-\d{2}-\d{2}([ T]\d{2}:\d{2}(:\d{2}(\.\d+)?)?)?$")
DMY_DOT = re.compile(r"^(\d{1,2})\.(\d{1,2})\.(\d{4})$")
MDY_SLASH = re.compile(r"^(\d{1,2})/(\d{1,2})/(\d{2,4})$")
YMD_SLASH = re.compile(r"^(\d{4})/(\d{1,2})/(\d{1,2})$")
NAMED_FMTS = ("%B %d, %Y", "%b %d, %Y", "%d %B %Y", "%d %b %Y", "%B %Y", "%Y%m%d")


# ====================================================================== schema / data access
class DbFacts:
    _cache = {}

    def __init__(self, db_id):
        self.db_id = db_id
        self.path = SPIDER_DB / db_id / f"{db_id}.sqlite"
        prof = TC.profile_db(self.path, db_id)
        self.tables, self.K, self.pk, self.fk = {}, {}, {}, []
        con = self.con()
        try:
            for coll, e in prof.items():
                t = e["table"]
                self.tables[t] = e["column_order"]
                for c, s in e["fields"].items():
                    self.K[(t, c)] = s["K"]
                info = con.execute(f'PRAGMA table_info("{t}")').fetchall()
                self.pk[t] = [r[1] for r in sorted(info, key=lambda r: r[5]) if r[5]]
                for r in con.execute(f'PRAGMA foreign_key_list("{t}")').fetchall():
                    self.fk.append((t, r[3], r[2], r[4]))      # (table, column, ref_table, ref_column)
        finally:
            con.close()

    @classmethod
    def get(cls, db_id):
        if db_id not in cls._cache:
            cls._cache[db_id] = DbFacts(db_id)
        return cls._cache[db_id]

    def con(self):
        c = sqlite3.connect(f"file:{self.path.as_posix()}?mode=ro", uri=True)
        c.text_factory = lambda b: b.decode("utf-8", errors="surrogateescape")
        steps = [0]

        def guard():
            steps[0] += 1
            return 1 if steps[0] > PARAMS["sqlite_timeout_steps"] // 1000 else 0
        c.set_progress_handler(guard, 1000)
        return c

    def table(self, name):
        hits = [t for t in self.tables if t.lower() == str(name).lower()]
        return hits[0] if len(hits) == 1 else None

    def column(self, t, name):
        hits = [c for c in self.tables.get(t, []) if c.lower() == str(name).lower()]
        return hits[0] if len(hits) == 1 else None

    def values(self, t, c):
        con = self.con()
        try:
            return [v for (v,) in con.execute(f'SELECT DISTINCT "{c}" FROM "{t}" WHERE "{c}" IS NOT NULL LIMIT ?',
                                              (PARAMS["D1_max_values"],))]
        finally:
            con.close()

    def query(self, sql, args=()):
        con = self.con()
        try:
            return con.execute(sql, args).fetchall()
        finally:
            con.close()


# ====================================================================== parsed record context
class Ctx:
    def __init__(self, rec):
        self.rec = rec
        self.id, self.db_id, self.sql = rec["id"], rec["db_id"], rec["sql"]
        self.q = rec["question"]
        self.ql = " ".join(re.sub(r"[^a-z0-9.'\"%_-]+", " ", self.q.lower()).split())
        self.db = DbFacts.get(self.db_id)
        try:
            self.tree = sqlglot.parse_one(self.sql, read="sqlite")
        except Exception:  # noqa: BLE001
            self.tree = None
        self.alias = {}
        if self.tree is not None:
            for tnode in self.tree.find_all(exp.Table):
                t = self.db.table(tnode.name)
                if t:
                    self.alias[(tnode.alias_or_name or tnode.name).lower()] = t
                    self.alias[tnode.name.lower()] = t

    def resolve(self, col):
        """exp.Column -> (table, column) or None."""
        if not isinstance(col, exp.Column):
            return None
        if col.table:
            t = self.alias.get(col.table.lower())
            c = self.db.column(t, col.name) if t else None
            return (t, c) if c else None
        hits = [(t, self.db.column(t, col.name)) for t in set(self.alias.values()) if self.db.column(t, col.name)]
        return hits[0] if len(hits) == 1 else None

    def result_rows(self):
        r = self.rec
        if "expected_output" in r:
            return r["expected_output"]
        if "sqlite_result" in r:
            return r["sqlite_result"]["rows"]
        return None

    def selects(self):
        return list(self.tree.find_all(exp.Select)) if self.tree is not None else []


def _unparen(e):
    while isinstance(e, exp.Paren):
        e = e.this
    return e


def _lit_value(node):
    node = _unparen(node)
    if isinstance(node, exp.Neg) and isinstance(node.this, exp.Literal):
        return "-" + node.this.this
    if isinstance(node, exp.Literal):
        return node.this
    if isinstance(node, exp.Column) and node.this is not None and isinstance(node.this, exp.Identifier) \
            and node.this.quoted and not node.table:
        return None   # double-quoted identifier; resolved later by caller if not a column
    return None


# ====================================================================== D1: number/date-like TEXT used in ordering / comparison
def parse_key(v):
    """-> (kind, key, iso_padded) or None. kind: 'num' | 'date'."""
    if not isinstance(v, str):
        return None
    s = v.strip()
    if NUM.match(s):
        return ("num", float(s), False)
    m = ISO.match(s)
    if m:
        y, mo, d = int(m[1]), int(m[2]), int(m[3])
        hh, mm, ss = int(m[4] or 0), int(m[5] or 0), int(m[6] or 0)
        return ("date", (y, mo, d, hh, mm, ss), bool(ISO_PADDED.match(s)))
    m = DMY_DOT.match(s)
    if m:
        return ("date", (int(m[3]), int(m[2]), int(m[1]), 0, 0, 0), False)
    m = MDY_SLASH.match(s)
    if m:
        y = int(m[3]) + (1900 if len(m[3]) == 2 else 0)
        return ("date", (y, int(m[1]), int(m[2]), 0, 0, 0), False)
    m = YMD_SLASH.match(s)
    if m:
        return ("date", (int(m[1]), int(m[2]), int(m[3]), 0, 0, 0), False)
    for f in NAMED_FMTS:
        try:
            dt = datetime.strptime(s, f)
            return ("date", (dt.year, dt.month, dt.day, 0, 0, 0), False)
        except ValueError:
            pass
    return None


def ordered_columns(ctx):
    """(table, column, how) for columns used in ORDER BY, MIN/MAX, or <,<=,>,>=,BETWEEN."""
    out = []
    if ctx.tree is None:
        return out
    for o in ctx.tree.find_all(exp.Ordered):
        e = _unparen(o.this)
        if isinstance(e, (exp.Min, exp.Max)):
            e = _unparen(e.this)
        r = ctx.resolve(e)
        if r:
            out.append((*r, "order_by_desc" if o.args.get("desc") else "order_by_asc"))
    for a in ctx.tree.find_all(exp.Min, exp.Max):
        r = ctx.resolve(_unparen(a.this))
        if r:
            out.append((*r, type(a).__name__.lower()))
    for cmp_ in ctx.tree.find_all(exp.GT, exp.GTE, exp.LT, exp.LTE, exp.Between):
        for side in (cmp_.this, cmp_.args.get("expression"), cmp_.args.get("low"), cmp_.args.get("high")):
            r = ctx.resolve(_unparen(side)) if side is not None else None
            if r:
                out.append((*r, "range_comparison"))
    return out


def d1(ctx):
    hits, seen = [], set()
    for t, c, how in ordered_columns(ctx):
        if (t, c) in seen or ctx.db.K.get((t, c)) != "text":
            continue
        seen.add((t, c))
        vals = ctx.db.values(t, c)
        if not vals:
            continue
        keys = [parse_key(v) for v in vals]
        ok = [k for k in keys if k]
        if len(ok) / len(vals) < PARAMS["D1_parse_share"]:
            continue
        kinds = {k[0] for k in ok}
        kind = "date" if "date" in kinds and len(kinds) == 1 else "num" if kinds == {"num"} else "mixed_num_date"
        pairs = sorted(((v, k) for v, k in zip(vals, keys) if k and k[0] == (kind if kind != "mixed_num_date" else k[0])),
                       key=lambda x: x[0])
        lex_keys = [k[1] for _, k in pairs if k[0] == pairs[0][1][0]] if pairs else []
        unsafe = any(a > b for a, b in zip(lex_keys, lex_keys[1:]))
        iso = kind == "date" and all(k[2] for k in ok if k[0] == "date")
        example = None
        if unsafe:
            for (va, ka), (vb, kb) in zip(pairs, pairs[1:]):
                if ka[0] == kb[0] and ka[1] > kb[1]:
                    example = [va, vb]
                    break
        sub = "D1a" if unsafe else ("D1b_iso" if iso else "D1b")
        hits.append({"detector": "D1", "subclass": sub, "action": "report_only" if sub == "D1b_iso" else "quarantine",
                     "evidence": {"column": f"{t}.{c}", "use": how, "value_kind": kind, "K": "text",
                                  "parsed_share": round(len(ok) / len(vals), 3), "distinct_values": len(vals),
                                  "lexical_order_inversion_example": example}})
    return hits


# ====================================================================== D2: temporal wording vs sort direction
def _directions(ctx):
    """(table, column, 'asc'|'desc', how) from ORDER BY and MIN/MAX."""
    out = []
    if ctx.tree is None:
        return out
    for o in ctx.tree.find_all(exp.Ordered):
        e = _unparen(o.this)
        agg = None
        if isinstance(e, (exp.Min, exp.Max)):
            agg, e = type(e).__name__.lower(), _unparen(e.this)
        r = ctx.resolve(e)
        if r:
            d = "desc" if o.args.get("desc") else "asc"
            out.append((*r, d, "order_by" + (f"({agg})" if agg else "")))
    for a in ctx.tree.find_all(exp.Max, exp.Min):
        if a.find_ancestor(exp.Ordered) is not None:
            continue
        r = ctx.resolve(_unparen(a.this))
        if r:
            out.append((*r, "desc" if isinstance(a, exp.Max) else "asc", type(a).__name__.lower()))
    return out


def d2(ctx):
    hits = []
    words = {k: any(re.search(p, ctx.ql) for p in LEXICON[k]) for k in ("D2_recent", "D2_earliest", "D2_youngest", "D2_oldest")}
    if not any(words.values()):
        return hits
    if (words["D2_recent"] and words["D2_earliest"]) or (words["D2_youngest"] and words["D2_oldest"]):
        return hits      # both opposite superlatives in one question: no single expected direction
    for t, c, d, how in _directions(ctx):
        cl = c.lower()
        is_birth = bool(re.search(LEXICON["D2_birth_col"], cl))
        is_age = bool(re.search(LEXICON["D2_age_col"], cl))
        is_date = bool(re.search(LEXICON["D2_date_col"], cl)) and not is_age
        expected = []
        if words["D2_recent"] and is_date and not is_birth:
            expected.append(("most recent/latest/newest", "desc"))
        if words["D2_earliest"] and is_date and not is_birth:
            expected.append(("earliest", "asc"))
        if words["D2_youngest"] and is_birth:
            expected.append(("youngest (birth date)", "desc"))
        if words["D2_youngest"] and is_age:
            expected.append(("youngest (age)", "asc"))
        if words["D2_oldest"] and (is_birth or (is_date and not is_age)):
            expected.append(("oldest (date)", "asc"))
        if words["D2_oldest"] and is_age:
            expected.append(("oldest (age)", "desc"))
        for word, want in expected:
            if d != want:
                hits.append({"detector": "D2", "subclass": "D2", "action": "quarantine",
                             "evidence": {"column": f"{t}.{c}", "wording": word, "expected": want, "observed": d,
                                          "use": how, "K": ctx.db.K.get((t, c))}})
    return hits


# ====================================================================== D3: "all/each X" that may omit entities
def _is_inner(j):
    side = (j.args.get("side") or "").upper()
    kind = (j.args.get("kind") or "").upper()
    return side not in ("LEFT", "RIGHT", "FULL") and kind not in ("OUTER", "CROSS")


def _join_edges(ctx):
    edges = []
    for j in ctx.tree.find_all(exp.Join):
        on = j.args.get("on")
        if on is None:
            continue
        for eq in on.find_all(exp.EQ):
            a, b = ctx.resolve(_unparen(eq.this)), ctx.resolve(_unparen(eq.expression))
            if a and b:
                edges.append((a, b, _is_inner(j)))
    return edges


def d3(ctx):
    if ctx.tree is None or not any(re.search(p, ctx.ql) for p in LEXICON["D3_all"]):
        return []
    sel = ctx.tree if isinstance(ctx.tree, exp.Select) else None
    if sel is None or not sel.args.get("group"):
        return []
    gcols = [ctx.resolve(_unparen(g)) for g in sel.args["group"].expressions]
    if not gcols or None in gcols:
        return []
    gtables = {t for t, _ in gcols}
    hits = []
    edges = _join_edges(ctx)
    if edges and len(gtables) == 1:
        (E,) = gtables
        for a, b, inner in edges:
            if not inner:
                continue
            if a[0] == E or b[0] == E:
                ek, ck = (a, b) if a[0] == E else (b, a)
                if ck[0] == E:
                    continue
                n_abs = ctx.db.query(f'SELECT count(*) FROM "{E}" WHERE "{ek[1]}" IS NULL OR "{ek[1]}" NOT IN '
                                     f'(SELECT "{ck[1]}" FROM "{ck[0]}" WHERE "{ck[1]}" IS NOT NULL)')[0][0]
                gl = ", ".join(f'"{c}"' for _, c in gcols)
                n_groups = ctx.db.query(
                    f'SELECT count(*) FROM (SELECT {gl} FROM "{E}" EXCEPT SELECT {", ".join(f"E.{chr(34)}{c}{chr(34)}" for _, c in gcols)} '
                    f'FROM "{E}" AS E JOIN "{ck[0]}" AS C ON E."{ek[1]}" = C."{ck[1]}")')[0][0]
                hits.append({"detector": "D3", "subclass": "D3a", "action": "quarantine",
                             "evidence": {"entity_table": E, "child_table": ck[0], "join": f"{E}.{ek[1]} = {ck[0]}.{ck[1]}",
                                          "group_by": [f"{t}.{c}" for t, c in gcols],
                                          "data_level_entity_rows_without_child": n_abs,
                                          "data_level_groups_missing": n_groups, "schema_level": "absent keys possible"}})
                break
    if not edges and not list(ctx.tree.find_all(exp.Join)) and len(gcols) == 1:
        T, g = gcols[0]
        targets = [(rt, rc) for (ft, fc, rt, rc) in ctx.db.fk if ft == T and fc.lower() == g.lower()]
        for E in ctx.db.tables:
            if E != T and ctx.db.pk.get(E) and len(ctx.db.pk[E]) == 1 and ctx.db.pk[E][0].lower() == g.lower():
                targets.append((E, ctx.db.pk[E][0]))
        seen = set()
        for rt, rc in targets:
            rt2 = ctx.db.table(rt)
            rc2 = ctx.db.column(rt2, rc) if rt2 else None
            if not rc2 or (rt2, rc2) in seen:
                continue
            seen.add((rt2, rc2))
            n_abs = ctx.db.query(f'SELECT count(*) FROM "{rt2}" WHERE "{rc2}" NOT IN '
                                 f'(SELECT "{g}" FROM "{T}" WHERE "{g}" IS NOT NULL)')[0][0]
            hits.append({"detector": "D3", "subclass": "D3b", "action": "quarantine",
                         "evidence": {"child_table": T, "group_by": f"{T}.{g}", "entity_table": rt2,
                                      "entity_key": f"{rt2}.{rc2}", "data_level_entity_keys_absent": n_abs,
                                      "schema_level": "absent keys possible"}})
    return hits


# ====================================================================== D4: counting entities over composite / 1-n tables
def _sing(w):
    if w.endswith("ies"):
        return w[:-3] + "y"
    if w.endswith("ses") or w.endswith("xes"):
        return w[:-2]
    if w.endswith("s") and not w.endswith("ss"):
        return w[:-1]
    return w


def d4(ctx):
    if ctx.tree is None or not isinstance(ctx.tree, exp.Select):
        return []
    sel = ctx.tree
    if list(sel.find_all(exp.Join)) or any(s is not sel for s in sel.find_all(exp.Select)):
        return []
    star = [c for c in sel.expressions if isinstance(_unparen(c.unalias() if hasattr(c, "unalias") else c), exp.Count)
            and isinstance(_unparen(c.unalias()).this, exp.Star)]
    if not star:
        return []
    m = re.search(LEXICON["D4_counted"], ctx.ql)
    if not m:
        return []
    noun = _sing(m.group(1))
    tables = list({t for t in ctx.alias.values()})
    if len(tables) != 1:
        return []
    T = tables[0]
    if _sing(T.lower()) == noun or T.lower() == noun:
        return []
    cands = []
    for c in ctx.db.tables[T]:
        cl = c.lower()
        if cl in (noun, noun + "_id", noun + "id", noun + "_code", noun + "_name", noun + "_no") or \
                cl.replace("_", "") in (noun + "id", noun + "code"):
            cands.append(c)
    for (ft, fc, rt, rc) in ctx.db.fk:
        if ft == T and _sing(str(rt).lower()) == noun and fc not in cands:
            cands.append(fc)
    hits = []
    for e in cands:
        if ctx.db.pk.get(T) == [e]:
            continue
        where = sel.args.get("where")
        group = sel.args.get("group")
        wsql = f" WHERE {where.this.sql(dialect='sqlite')}" if where is not None else ""
        if group is not None:
            gsql = ", ".join(g.sql(dialect="sqlite") for g in group.expressions)
            q = (f'SELECT count(*) FROM (SELECT count(*) AS a, count(DISTINCT "{e}") AS b FROM "{T}" AS {sel.find(exp.Table).alias_or_name}'
                 f'{wsql} GROUP BY {gsql}) WHERE a != b')
        else:
            q = (f'SELECT count(*) != count(DISTINCT "{e}") FROM "{T}" AS {sel.find(exp.Table).alias_or_name}{wsql}')
        try:
            diff = ctx.db.query(q)[0][0]
        except sqlite3.Error as ex:
            diff = f"error: {ex}"[:120]
        data_level = isinstance(diff, int) and diff > 0
        hits.append({"detector": "D4", "subclass": "D4_data" if data_level else "D4_schema_only",
                     "action": "quarantine" if data_level else "report_only",
                     "evidence": {"table": T, "counted_noun": m.group(1), "entity_column": f"{T}.{e}",
                                  "primary_key": ctx.db.pk.get(T), "groups_where_count_star_ne_count_distinct": diff}})
        break
    return hits


# ====================================================================== D5: truncated / ambiguous questions
def _norm_tokens(s):
    return re.findall(r"[a-z0-9]+", str(s).lower())


def d5(ctx):
    hits = []
    q = ctx.q.strip()
    core = re.sub(r"[\s?.!]+$", "", q).strip()
    last = core.split()[-1].lower() if core.split() else ""
    reasons = []
    if last in PARAMS["D5_dangling_words"]:
        reasons.append(f"ends_with:{last}")
    if re.search(LEXICON["D5_has_name"], core.lower()):
        reasons.append("ends_with_attribute_without_value")
    dq = len(re.findall(r'"', q))
    sq = len(re.findall(r"(?<![A-Za-z])'|'(?![A-Za-z])", re.sub(r"(?<=[sS])'(?=\s)", "", q)))  # plural possessive s'
    if dq % 2 or sq % 2:
        reasons.append("unbalanced_quote")
    raw = ctx.rec.get("question_raw")
    if raw and _norm_tokens(raw) != _norm_tokens(q):
        reasons.append("question_raw_mismatch")
    if ctx.tree is not None:
        w = ctx.tree.args.get("where") if isinstance(ctx.tree, exp.Select) else None
        atoms = len([a for a in w.find_all(exp.EQ, exp.NEQ, exp.GT, exp.GTE, exp.LT, exp.LTE, exp.Like, exp.In, exp.Between)]) if w else 0
        if atoms >= PARAMS["D5_short_min_atoms"] and len(core.split()) < PARAMS["D5_short_words_per_atom"] * atoms + PARAMS["D5_short_base"]:
            reasons.append(f"short_for_{atoms}_predicates")
    if reasons:
        hits.append({"detector": "D5", "subclass": "D5", "action": "quarantine",
                     "evidence": {"reasons": reasons, "question": q}})
    return hits


# ====================================================================== D6: literals the question does not support
def _filter_literals(ctx):
    """(literal_text, compared_column or None, op) from WHERE / HAVING."""
    out = []
    if ctx.tree is None:
        return out
    for clause in list(ctx.tree.find_all(exp.Where)) + list(ctx.tree.find_all(exp.Having)):
        for node in clause.find_all(exp.EQ, exp.NEQ, exp.GT, exp.GTE, exp.LT, exp.LTE, exp.Like, exp.In, exp.Between):
            sides = [node.this] + ([node.expression] if node.args.get("expression") is not None else []) + \
                list(node.args.get("expressions") or []) + [x for x in (node.args.get("low"), node.args.get("high")) if x is not None]
            col = None
            for s in sides:
                r = ctx.resolve(_unparen(s))
                if r:
                    col = r
                    break
            for s in sides:
                s = _unparen(s)
                v = _lit_value(s)
                if v is None and isinstance(s, exp.Column) and not ctx.resolve(s) and not s.table:
                    v = s.name               # SQLite: unresolvable double-quoted name = string literal
                if v is not None:
                    out.append((v, col, type(node).__name__, bool(isinstance(s, exp.Literal) and s.is_string) or not NUM.match(str(v))))
    return out


def _num_forms(v):
    try:
        f = float(v)
    except ValueError:
        return set()
    forms = {v, str(f)}
    if f == int(f):
        forms |= {str(int(f)), f"{int(f):,}"}
    return forms


def literal_supported(ctx, lit, is_string):
    ql = ctx.q.lower()
    toks = set(_norm_tokens(ql))
    s = str(lit).strip().strip("%").replace("%", " ").replace("_", " ").lower()
    if not s.strip():
        return True
    if is_string:
        if s in ql:
            return True
        lt = _norm_tokens(s)
        if lt and all(t in toks for t in lt):
            return True
    else:
        if any(re.search(rf"(?<![\d.]){re.escape(f)}(?![\d])", ql) for f in _num_forms(s)):
            return True
        try:
            fv = float(s)
            if any(fv == n for w, n in LEXICON["D6_number_words"].items() if w in toks):
                return True
        except ValueError:
            pass
    cues = LEXICON["D6_implicit"].get(s)
    return bool(cues and any(c in toks for c in cues))


def d6(ctx):
    hits = []
    lits = _filter_literals(ctx)
    if not lits:
        return hits
    unsupported = [(v, col) for v, col, op, is_s in lits if not literal_supported(ctx, v, is_s)]
    if unsupported:
        hits.append({"detector": "D6", "subclass": "D6a", "action": "quarantine",
                     "evidence": {"literals_not_in_question": [[v, f"{c[0]}.{c[1]}" if c else None] for v, c in unsupported]}})
    for v, col, op, is_s in lits:
        if not col or op not in ("EQ", "In") or not is_s or ctx.db.K.get(col) != "text":
            continue
        t, c = col
        n = ctx.db.query(f'SELECT count(*) FROM "{t}" WHERE "{c}" = ?', (v,))[0][0]
        if n:
            continue
        found = []
        for tt in sorted(set(ctx.alias.values())):
            for cc in ctx.db.tables[tt]:
                if (tt, cc) != (t, c) and ctx.db.K.get((tt, cc)) == "text":
                    if ctx.db.query(f'SELECT count(*) FROM "{tt}" WHERE "{cc}" = ?', (v,))[0][0]:
                        found.append(f"{tt}.{cc}")
        if found:
            hits.append({"detector": "D6", "subclass": "D6b", "action": "quarantine",
                         "evidence": {"literal": v, "column": f"{t}.{c}", "rows_matching_column": 0, "found_in": found}})
    rows = ctx.result_rows()
    if rows is not None and lits and (len(rows) == 0 or (len(rows) == 1 and len(rows[0]) == 1 and rows[0][0] == 0
                                                        and ctx.tree is not None and list(ctx.tree.find_all(exp.Count)))):
        hits.append({"detector": "D6", "subclass": "D6c", "action": "quarantine",
                     "evidence": {"result": "empty" if not rows else "count_zero", "literals": [v for v, *_ in lits][:6]}})
    return hits


DETECTORS = {"D1": d1, "D2": d2, "D3": d3, "D4": d4, "D5": d5, "D6": d6}


def run_all(rec):
    ctx = Ctx(rec)
    out = []
    if ctx.tree is None:
        return [{"detector": "PARSE", "subclass": "unparsed", "action": "report_only", "evidence": {}}]
    for name, fn in DETECTORS.items():
        try:
            out += fn(ctx)
        except sqlite3.Error as e:
            out.append({"detector": name, "subclass": "error", "action": "report_only", "evidence": {"error": str(e)[:200]}})
    return out
