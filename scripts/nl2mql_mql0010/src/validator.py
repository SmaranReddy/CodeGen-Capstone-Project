"""MQL-0010 component V: static, schema-grounded validation of a generated aggregation (no database access).

Uses only the target database's schema context (collections, observed field names) -- inference-time
information -- and returns a list of Issues. It never executes anything. Used by the controlled-recovery step
(one feedback retry, see recovery.py), which is separate from retrieval.

What it checks (each check is generic MongoDB semantics):
  V1  collection      the start collection and every $lookup "from" exist in the target database
  V2  field paths     every field path referenced by a stage resolves in the documents flowing INTO that stage,
                      tracked through $match/$sort/$limit/$skip (unchanged), $lookup (adds the alias as an array of
                      the foreign collection's documents), $unwind, $group (_id + accumulators), $project /
                      $addFields / $set (new shape), $count, $sortByCount. Anything it cannot model ($facet,
                      $replaceRoot, $replaceWith, $bucket*, $unionWith, $graphLookup, unknown stages, $lookup with a
                      sub-pipeline) makes the shape OPAQUE: later references are not checked (no false alarms).
  V3  query/expression mixing   an aggregation-expression operator ($avg, $max, $sum, $size, $filter, ...) used as
                      the operand of a query operator inside $match without $expr, e.g. {"x": {"$gt": {"$avg": "$x"}}}:
                      MongoDB treats it as a literal sub-document, so the predicate silently matches nothing.
  V4  field-path strings inside $match query values that are not under $expr, e.g. {"a": "$b"}: compares with the
                      literal string "$b" (a classic silent empty result).
Issue suggestions name the collection(s) that do hold a missing field, so the retry message is actionable.
"""
from dataclasses import dataclass, field

EXPR_OPS = {"$avg", "$max", "$min", "$sum", "$size", "$filter", "$map", "$reduce", "$arrayElemAt", "$first", "$last",
            "$add", "$subtract", "$multiply", "$divide", "$concat", "$toString", "$toInt", "$toDouble", "$cond",
            "$ifNull", "$count", "$push", "$addToSet", "$avg", "$substr", "$strLenCP", "$dateToString"}
OPAQUE_STAGES = {"$facet", "$replaceRoot", "$replaceWith", "$bucket", "$bucketAuto", "$unionWith", "$graphLookup",
                 "$densify", "$fill", "$setWindowFields", "$redact", "$sample", "$geoNear", "$documents"}
PASS_STAGES = {"$match", "$sort", "$limit", "$skip"}
QUERY_LOGICAL = {"$and", "$or", "$nor"}


@dataclass
class Issue:
    code: str
    stage: int
    message: str
    suggestion: str = ""

    def text(self):
        s = f"stage {self.stage}: {self.message}"
        return s + (f" ({self.suggestion})" if self.suggestion else "")


@dataclass
class Shape:
    """Fields available in the current documents. `prefix_opaque` = paths under which anything is accepted."""
    fields: set = field(default_factory=set)
    prefix_opaque: set = field(default_factory=set)
    opaque: bool = False

    def copy(self):
        return Shape(set(self.fields), set(self.prefix_opaque), self.opaque)

    def has(self, path):
        if self.opaque or not path:
            return True
        if path in self.fields:
            return True
        parts = path.split(".")
        for i in range(1, len(parts)):
            pre = ".".join(parts[:i])
            if pre in self.prefix_opaque:
                return True
        # numeric array index segments (a.0.b) and sub-fields of known object fields are accepted
        if parts[0] in self.fields and any(p.isdigit() for p in parts[1:]):
            return True
        return False


class SchemaValidator:
    def __init__(self, db_id, collections):
        """collections: the atlas_collections context ({name: {"db_id", "fields": {field: [types]}}})."""
        self.db_id = db_id
        self.colls = {n: set(c["fields"]) | {"_id"} for n, c in collections.items() if c["db_id"] == db_id}
        self.lower = {}
        for n, fs in self.colls.items():
            for f in fs:
                self.lower.setdefault(f.lower(), set()).add((n, f))

    # ------------------------------------------------------------------ helpers
    def _suggest(self, path, shape=None):
        leaf = path.split(".")[-1]
        hits = sorted(self.lower.get(leaf.lower(), ()))
        if shape is not None:
            avail = sorted(f for f in shape.fields if f.lower() == leaf.lower() or f.split(".")[-1].lower() == leaf.lower())
            if avail:
                return "available here as: " + ", ".join(avail[:4])
        if hits:
            return "this field exists in " + ", ".join(f"{n}.{f}" for n, f in hits[:4]) + "; join that collection first or start from it"
        return ""

    def _check_path(self, path, shape, st, issues):
        if path.startswith("$$") or path == "":
            return
        if not shape.has(path):
            issues.append(Issue("V2_unknown_field", st, f"field '{path}' does not exist in the documents at this stage",
                                self._suggest(path, shape)))

    def _expr_refs(self, v, shape, st, issues):
        """Check "$path" strings inside an aggregation expression."""
        if isinstance(v, str):
            if v.startswith("$") and not v.startswith("$$"):
                self._check_path(v[1:], shape, st, issues)
        elif isinstance(v, list):
            for x in v:
                self._expr_refs(x, shape, st, issues)
        elif isinstance(v, dict):
            for k, x in v.items():
                if k in ("$literal",):
                    continue
                self._expr_refs(x, shape, st, issues)

    def _query(self, q, shape, st, issues):
        """A $match query document."""
        if not isinstance(q, dict):
            return
        for k, v in q.items():
            if k in QUERY_LOGICAL and isinstance(v, list):
                for x in v:
                    self._query(x, shape, st, issues)
            elif k == "$expr":
                self._expr_refs(v, shape, st, issues)
            elif k.startswith("$"):
                continue      # $text, $where (blocked elsewhere), $comment ...
            else:
                self._check_path(k, shape, st, issues)
                self._query_value(k, v, st, issues)

    def _query_value(self, k, v, st, issues):
        if isinstance(v, str) and v.startswith("$") and not v.startswith("$$") and len(v) > 1:
            issues.append(Issue("V4_fieldpath_literal", st, f"'{k}': \"{v}\" inside $match is compared as a literal string",
                                "compare two fields with {\"$expr\": {\"$eq\": [\"$" + k + "\", \"" + v + "\"]}}"))
        if isinstance(v, dict):
            for op, x in v.items():
                if isinstance(x, dict) and any(kk in EXPR_OPS for kk in x):
                    bad = next(kk for kk in x if kk in EXPR_OPS)
                    issues.append(Issue("V3_expr_in_query", st,
                                        f"'{k}': {op} has an aggregation expression ({bad}) as operand; without $expr it is a literal and matches nothing",
                                        "compute the value in an earlier stage ($group / $lookup) or use $expr"))
                if isinstance(x, str) and x.startswith("$") and not x.startswith("$$") and op.startswith("$") and op not in ("$regex", "$options"):
                    issues.append(Issue("V4_fieldpath_literal", st, f"'{k}': {op} \"{x}\" inside $match is compared as a literal string",
                                        "use $expr to compare with another field"))

    # ------------------------------------------------------------------ main
    def validate(self, mql):
        issues = []
        if not isinstance(mql, dict) or not isinstance(mql.get("pipeline"), list):
            return [Issue("V0_shape", 0, "the query must be an object with 'collection' and a 'pipeline' array")]
        coll = mql.get("collection")
        if coll not in self.colls:
            close = [n for n in self.colls if n.lower() == str(coll).lower()]
            return [Issue("V1_unknown_collection", 0, f"collection '{coll}' is not in database {self.db_id}",
                          ("did you mean " + close[0]) if close else "use one of: " + ", ".join(sorted(self.colls)[:8]))]
        shape = Shape(set(self.colls[coll]))
        for i, stage in enumerate(mql["pipeline"], start=1):
            if not isinstance(stage, dict) or len(stage) != 1:
                issues.append(Issue("V0_shape", i, "each stage must be an object with exactly one operator"))
                shape.opaque = True
                continue
            op, body = next(iter(stage.items()))
            if shape.opaque:
                if op == "$lookup" and isinstance(body, dict) and body.get("from") not in self.colls:
                    issues.append(Issue("V1_unknown_collection", i, f"$lookup from '{body.get('from')}' is not in database {self.db_id}"))
                continue
            shape = self._stage(op, body, shape, i, issues)
        return issues

    def _stage(self, op, body, shape, st, issues):
        if op in PASS_STAGES:
            if op == "$match":
                self._query(body, shape, st, issues)
            elif op == "$sort" and isinstance(body, dict):
                for k in body:
                    self._check_path(k, shape, st, issues)
            return shape
        if op in OPAQUE_STAGES:
            return Shape(opaque=True)
        if op == "$lookup":
            return self._lookup(body, shape, st, issues)
        if op == "$unwind":
            p = body if isinstance(body, str) else (body.get("path") if isinstance(body, dict) else None)
            if isinstance(p, str) and p.startswith("$"):
                self._check_path(p[1:], shape, st, issues)
            if isinstance(body, dict) and body.get("includeArrayIndex"):
                shape = shape.copy()
                shape.fields.add(body["includeArrayIndex"])
            return shape
        if op == "$group":
            if not isinstance(body, dict):
                return Shape(opaque=True)
            out = Shape({"_id"})
            gid = body.get("_id")
            self._expr_refs(gid, shape, st, issues)
            if isinstance(gid, dict):
                for k in gid:
                    out.fields.add(f"_id.{k}")
            for k, v in body.items():
                if k == "_id":
                    continue
                self._expr_refs(v, shape, st, issues)
                out.fields.add(k)
                if isinstance(v, dict) and ("$push" in v or "$addToSet" in v or "$first" in v or "$last" in v):
                    out.prefix_opaque.add(k)
            return out
        if op in ("$project", "$addFields", "$set", "$unset"):
            return self._project(op, body, shape, st, issues)
        if op == "$count":
            return Shape({body} if isinstance(body, str) else set(), opaque=not isinstance(body, str))
        if op == "$sortByCount":
            self._expr_refs(body, shape, st, issues)
            return Shape({"_id", "count"})
        return Shape(opaque=True)

    def _lookup(self, body, shape, st, issues):
        if not isinstance(body, dict):
            return Shape(opaque=True)
        frm, alias = body.get("from"), body.get("as")
        if frm not in self.colls:
            close = [n for n in self.colls if n.lower() == str(frm).lower()]
            issues.append(Issue("V1_unknown_collection", st, f"$lookup from '{frm}' is not in database {self.db_id}",
                                ("did you mean " + close[0]) if close else ""))
            out = shape.copy()
            if isinstance(alias, str):
                out.fields.add(alias)
                out.prefix_opaque.add(alias)
            return out
        out = shape.copy()
        if "pipeline" in body or "let" in body:
            if isinstance(body.get("let"), dict):
                for v in body["let"].values():
                    self._expr_refs(v, shape, st, issues)
            if isinstance(alias, str):
                out.fields.add(alias)
                out.prefix_opaque.add(alias)
            return out
        lf, ff = body.get("localField"), body.get("foreignField")
        if isinstance(lf, str):
            self._check_path(lf, shape, st, issues)
        if isinstance(ff, str) and ff not in self.colls[frm]:
            issues.append(Issue("V2_unknown_field", st, f"foreignField '{ff}' does not exist in {frm}", self._suggest(ff)))
        if isinstance(alias, str):
            out.fields.add(alias)
            out.fields |= {f"{alias}.{f}" for f in self.colls[frm]}
        return out

    def _project(self, op, body, shape, st, issues):
        if op == "$unset":
            out = shape.copy()
            for k in ([body] if isinstance(body, str) else body if isinstance(body, list) else []):
                out.fields.discard(k)
            return out
        if not isinstance(body, dict):
            return Shape(opaque=True)
        if op in ("$addFields", "$set"):
            out = shape.copy()
            for k, v in body.items():
                self._expr_refs(v, shape, st, issues)
                out.fields.add(k)
                out.prefix_opaque.add(k)
            return out
        vals = {k: v for k, v in body.items() if k != "_id"}
        exclusion = vals and all(v in (0, False) for v in vals.values())
        if exclusion:
            out = shape.copy()
            for k in body:
                out.fields.discard(k)
            return out
        out = Shape()
        if body.get("_id", 1) not in (0, False):
            out.fields.add("_id")
        for k, v in body.items():
            if k == "_id" and v in (0, False):
                continue
            if v in (1, True):
                self._check_path(k, shape, st, issues)
                out.fields.add(k)
                out.fields |= {f for f in shape.fields if f.startswith(k + ".")}
                if k in shape.prefix_opaque:
                    out.prefix_opaque.add(k)
            else:
                self._expr_refs(v, shape, st, issues)
                out.fields.add(k)
                if isinstance(v, (dict, list)):
                    out.prefix_opaque.add(k)
        return out


def format_feedback(issues, limit=6):
    """Retry message body (component V -> recovery)."""
    lines = [f"- {x.text()}" for x in issues[:limit]]
    return "Your previous query has problems that make it return wrong or empty results:\n" + "\n".join(lines)
