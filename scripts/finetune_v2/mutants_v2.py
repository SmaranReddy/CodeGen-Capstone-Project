"""Deliberately wrong MQL (negative controls) derived from accepted labels: single-edit mutants.

Operators cover the audit's mutant classes (scripts/audit_sql2mql_v1/mql_mutants.py: dropped
predicates, operator boundaries, literals, regex flags, NULL guards, projection / group-key
swaps, sort / limit, accumulators, distinct, lookups) plus one REVERSAL of every v2 fix
(B1 facet default row, B2 SUM guard, $trunc integer division, zero-divisor guard, dotall,
absolute anchor, missing-safe guards). `first_to_last` is an equivalence control (a $first of
a functionally determined column equals its $last), expected to survive.

mutants(mql) -> [(operator, mutant)] in deterministic order, identical mutants removed.
"""
import copy
import json


def _paths(node, path=()):
    yield path, node
    if isinstance(node, dict):
        for k, v in node.items():
            yield from _paths(v, path + (k,))
    elif isinstance(node, list):
        for i, v in enumerate(node):
            yield from _paths(v, path + (i,))


def _get(root, path):
    for p in path:
        root = root[p]
    return root


def _set(root, path, value):
    _get(root, path[:-1])[path[-1]] = value


def _stage(path):
    return path[1] if len(path) > 1 and path[0] == "pipeline" else None


def mutants(mql):
    out = []

    def add(op, fn):
        m = copy.deepcopy(mql)
        try:
            if fn(m) is False:
                return
        except (KeyError, IndexError, TypeError):
            return
        out.append((op, m))

    pipe = mql["pipeline"]
    nodes = list(_paths(mql))
    for path, node in nodes:
        if not isinstance(node, dict):
            continue
        # comparison operator boundaries / negation
        for k in list(node):
            swap = {"$gt": "$gte", "$gte": "$gt", "$lt": "$lte", "$lte": "$lt", "$eq": "$ne", "$ne": "$eq"}.get(k)
            if swap and swap not in node:
                def f(m, path=path, k=k, swap=swap):
                    d = _get(m, path)
                    d[swap] = d.pop(k)
                add(f"cmp_{k[1:]}_to_{swap[1:]}", f)
        # regex flags and anchors
        if "$options" in node or ("options" in node and "regex" in node):
            key = "$options" if "$options" in node else "options"
            for flag in ("i", "s"):
                if flag in node[key]:
                    def f(m, path=path, key=key, flag=flag):
                        d = _get(m, path)
                        d[key] = d[key].replace(flag, "")
                    add({"i": "regex_case", "s": "regex_dotall"}[flag], f)
            rkey = "$regex" if "$regex" in node else "regex"
            rx = node.get(rkey, "")
            if rx.endswith("\\z"):
                add("regex_anchor_end", lambda m, path=path, rkey=rkey: _set(m, path + (rkey,), _get(m, path)[rkey][:-2] + "$"))
            if rx.startswith("^"):
                add("regex_anchor_start", lambda m, path=path, rkey=rkey: _set(m, path + (rkey,), _get(m, path)[rkey][1:]))
        # NULL guards
        if "$nin" in node and isinstance(node["$nin"], list) and None in node["$nin"] and len(node["$nin"]) > 1:
            add("null_guard_drop", lambda m, path=path: _set(m, path + ("$nin",), [x for x in _get(m, path)["$nin"] if x is not None]))
        if "$and" in node and isinstance(node["$and"], list):
            for i, x in enumerate(node["$and"]):
                if isinstance(x, dict) and "$ne" in x and isinstance(x["$ne"], list) and x["$ne"][1:] == [None]:
                    add("null_guard_drop", lambda m, path=path, i=i: _get(m, path)["$and"].pop(i))
        if "$ifNull" in node and isinstance(node["$ifNull"], list) and node["$ifNull"][1:] == [None]:
            add("ifnull_drop", lambda m, path=path: _set(m, path, _get(m, path)["$ifNull"][0]) if path else False)
        # v2 fix reversals
        if "$trunc" in node:
            add("trunc_drop", lambda m, path=path: _set(m, path, _get(m, path)["$trunc"][0]))
        if "$cond" in node and isinstance(node["$cond"], list) and len(node["$cond"]) == 3:
            c0 = node["$cond"][0]
            if isinstance(c0, dict) and "$eq" in c0 and c0["$eq"][1:] == [0] and node["$cond"][1] is None:
                add("zero_guard_drop", lambda m, path=path: _set(m, path, _get(m, path)["$cond"][2]))
            if isinstance(c0, dict) and "$gt" in c0 and node["$cond"][2] is None:
                add("sum_guard_drop", lambda m, path=path: _set(m, path, _get(m, path)["$cond"][1]))
            if isinstance(c0, dict) and "$eq" in c0 and node["$cond"][1:] == [0, 1] and path and path[-1] == "$sum":
                add("count_nonnull_to_star", lambda m, path=path: _set(m, path, 1))
        # accumulators
        for a, b in (("$avg", "$sum"), ("$min", "$max"), ("$max", "$min"), ("$addToSet", "$push"),
                     ("$first", "$last")):
            if a in node and len(node) == 1 and _stage(path) is not None:
                op = {"$addToSet": "distinct_to_all", "$first": "first_to_last"}.get(a, f"acc_{a[1:]}_to_{b[1:]}")
                add(op, lambda m, path=path, a=a, b=b: _set(m, path, {b: _get(m, path)[a]}))
        # IN lists
        if "$in" in node and isinstance(node["$in"], list) and len(node["$in"]) > 1 and \
                all(not isinstance(x, (dict, list)) for x in node["$in"]):
            add("in_drop_value", lambda m, path=path: _get(m, path)["$in"].pop())
        # numeric literals inside query predicates
        for k, v in node.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool) and k in ("$gt", "$gte", "$lt", "$lte") \
                    or (isinstance(v, (int, float)) and not isinstance(v, bool) and not k.startswith("$")
                        and _stage(path) is not None and "$match" in pipe[_stage(path)]):
                add("literal_plus1", lambda m, path=path, k=k: _set(m, path + (k,), _get(m, path)[k] + 1))
    for i, st in enumerate(pipe):
        name = next(iter(st))
        body = st[name]
        if name == "$match":
            if isinstance(body, dict) and len(body) > 1:
                for k in list(body):
                    add("match_atom_drop", lambda m, i=i, k=k: m["pipeline"][i]["$match"].pop(k))
            for key in ("$and", "$or"):
                if isinstance(body, dict) and isinstance(body.get(key), list) and len(body[key]) > 1:
                    for j in range(len(body[key])):
                        add("match_atom_drop", lambda m, i=i, key=key, j=j: m["pipeline"][i]["$match"][key].pop(j))
            if len(pipe) > 1 and i + 1 < len(pipe) and "$lookup" in pipe[i + 1]:
                add("null_guard_drop", lambda m, i=i: m["pipeline"].pop(i))
        if name == "$sort":
            add("sort_drop", lambda m, i=i: m["pipeline"].pop(i))
            for k in body:
                add("sort_flip", lambda m, i=i, k=k: m["pipeline"][i]["$sort"].__setitem__(k, -m["pipeline"][i]["$sort"][k]))
        if name == "$limit":
            add("limit_plus1", lambda m, i=i: m["pipeline"][i].__setitem__("$limit", body + 1))
            add("limit_drop", lambda m, i=i: m["pipeline"].pop(i))
        if name == "$skip":
            add("skip_drop", lambda m, i=i: m["pipeline"].pop(i))
        if name == "$facet":
            def unwrap(m, i=i):
                g = m["pipeline"][i]["$facet"]["rows"]
                m["pipeline"][i:i + 3] = g
            add("facet_unwrap_B1", unwrap)
        if name == "$unwind":
            if isinstance(body, dict) and body.get("preserveNullAndEmptyArrays"):
                add("left_to_inner", lambda m, i=i: m["pipeline"][i].__setitem__("$unwind", body["path"]))
            elif isinstance(body, str):
                add("inner_to_left", lambda m, i=i: m["pipeline"][i].__setitem__(
                    "$unwind", {"path": body, "preserveNullAndEmptyArrays": True}))
        if name == "$lookup" and "localField" in body:
            def sw(m, i=i):
                lk = m["pipeline"][i]["$lookup"]
                lk["localField"], lk["foreignField"] = lk["foreignField"], lk["localField"]
            add("lookup_field_swap", sw)
        if name == "$group" and isinstance(body.get("_id"), dict) and len(body["_id"]) > 1:
            ks = list(body["_id"])
            def gs(m, i=i, ks=ks):
                d = m["pipeline"][i]["$group"]["_id"]
                d[ks[0]], d[ks[1]] = d[ks[1]], d[ks[0]]
            add("group_key_swap", gs)
    last = pipe[-1]["$project"]
    ks = [k for k in last if k != "_id"]
    for a in range(len(ks)):
        for b in range(a + 1, len(ks)):
            def ps(m, a=ks[a], b=ks[b]):
                d = m["pipeline"][-1]["$project"]
                d[a], d[b] = d[b], d[a]
            add("project_swap", ps)
    seen, uniq = {json.dumps(mql, sort_keys=True)}, []
    for op, m in out:
        key = json.dumps(m, sort_keys=True)
        if key not in seen:
            seen.add(key)
            uniq.append((op, m))
    return uniq
