"""MQL-mutant kill study: can the EXISTING verifier's fixtures tell a wrong MQL from the label?

    python -B scripts/audit_sql2mql_v1/mql_mutants.py [--n 200] [--workers 6]

Sample: seeded simple random sample (seed 20261025) of the 1,584 Kaggle-derived labels.
For each label the existing fixture set is regenerated exactly as verify.verify_pair builds it
(fixtures.adequate_set, same seed_base, BASE_CLOCK or server clock for clock queries). Each
MQL mutant is executed on those fixtures and judged with the pipeline's own comparator
(spider1_kit.docs_to_rows + fixtures.compare); "killed" = at least one variant mismatches.
Survivors are re-run on the audit's independent families (audit_fixtures); a survivor the
audit kills was NOT equivalent, so the existing verifier would have accepted a wrong label.
Survivors killed by neither are "possibly equivalent" (kill rates are lower bounds).
"""
import argparse
import collections
import concurrent.futures as cf
import copy
import hashlib
import json
import random

import audit_fixtures as AF
import common as K
import kaggle_census as KC

OUT = K.OUT / "mql_mutants.jsonl"
SEED = 20261025
MAX_PER_LABEL = 30
CMP_SWAP = {"$gt": "$gte", "$gte": "$gt", "$lt": "$lte", "$lte": "$lt"}
CMP_NEG = {"$gt": "$lte", "$gte": "$lt", "$lt": "$gte", "$lte": "$gt"}
ACC_SWAP = {"$avg": "$sum", "$min": "$max", "$max": "$min"}


def _walk(node, path=()):
    yield path, node
    if isinstance(node, dict):
        for k, v in node.items():
            yield from _walk(v, path + (k,))
    elif isinstance(node, list):
        for i, v in enumerate(node):
            yield from _walk(v, path + (i,))


def _get(root, path):
    for p in path:
        root = root[p]
    return root


def _set(root, path, value):
    _get(root, path[:-1])[path[-1]] = value


def mutants(mql):
    """-> [(operator, mutated mql)] (deterministic order)."""
    out = []
    pipe = mql["pipeline"]

    def mk(op, fn):
        m = copy.deepcopy(mql)
        try:
            if fn(m) is False:
                return
        except (KeyError, IndexError, TypeError):
            return
        if m != mql:
            out.append((op, m))

    for si, st in enumerate(pipe):
        name = next(iter(st))
        spec = st[name]
        if name == "$group":
            for acc, val in spec.items():
                if acc == "_id" or not isinstance(val, dict):
                    if acc == "_id" and isinstance(val, dict):
                        for gk in val:
                            mk("group_key_drop", lambda m, gk=gk, si=si: m["pipeline"][si]["$group"]["_id"].pop(gk))
                    continue
                op = next(iter(val))
                if op in ACC_SWAP:
                    mk("acc_swap", lambda m, si=si, acc=acc, op=op: m["pipeline"][si]["$group"].__setitem__(
                        acc, {ACC_SWAP[op]: m["pipeline"][si]["$group"][acc][op]}))
                if op == "$sum" and val[op] != 1:
                    mk("acc_swap", lambda m, si=si, acc=acc: m["pipeline"][si]["$group"].__setitem__(
                        acc, {"$avg": m["pipeline"][si]["$group"][acc]["$sum"]}))
                if op == "$sum" and isinstance(val[op], dict) and "$cond" in val[op]:
                    mk("count_col_to_star", lambda m, si=si, acc=acc: m["pipeline"][si]["$group"].__setitem__(
                        acc, {"$sum": 1}))
                if op == "$sum" and val[op] == 1:
                    mk("count_star_off", lambda m, si=si, acc=acc: m["pipeline"][si]["$group"].__setitem__(
                        acc, {"$sum": 2}))
                if op == "$addToSet":
                    mk("distinct_to_all", lambda m, si=si, acc=acc: m["pipeline"][si]["$group"].__setitem__(
                        acc, {"$push": m["pipeline"][si]["$group"][acc]["$addToSet"]}))
        if name == "$sort":
            for k in spec:
                mk("sort_flip", lambda m, si=si, k=k: m["pipeline"][si]["$sort"].__setitem__(
                    k, -m["pipeline"][si]["$sort"][k]))
            if len(spec) > 1:
                for k in spec:
                    mk("sort_key_drop", lambda m, si=si, k=k: m["pipeline"][si]["$sort"].pop(k))
            mk("sort_drop", lambda m, si=si: m["pipeline"].pop(si))
        if name == "$limit":
            mk("limit_drop", lambda m, si=si: m["pipeline"].pop(si))
            mk("limit_plus1", lambda m, si=si: m["pipeline"][si].__setitem__("$limit", m["pipeline"][si]["$limit"] + 1))
        if name == "$skip":
            mk("skip_drop", lambda m, si=si: m["pipeline"].pop(si))
        if name == "$project":
            keys = [k for k in spec if k != "_id"]
            for k in keys:
                if len(keys) > 1:
                    mk("project_drop", lambda m, si=si, k=k: m["pipeline"][si]["$project"].pop(k))
            for a, b in zip(keys, keys[1:]):
                if spec[a] != spec[b]:
                    def sw(m, si=si, a=a, b=b):
                        p = m["pipeline"][si]["$project"]
                        p[a], p[b] = p[b], p[a]
                    mk("project_swap", sw)
        if name == "$unwind" and isinstance(spec, str):
            mk("inner_to_left", lambda m, si=si: m["pipeline"].__setitem__(
                si, {"$unwind": {"path": spec, "preserveNullAndEmptyArrays": True}}))
        if name == "$match":
            if list(spec) == ["$and"] and len(spec["$and"]) > 1:
                for j in range(len(spec["$and"])):
                    mk("conjunct_drop", lambda m, si=si, j=j: m["pipeline"][si]["$match"]["$and"].pop(j))
                mk("and_to_or", lambda m, si=si: m["pipeline"][si].__setitem__(
                    "$match", {"$or": m["pipeline"][si]["$match"]["$and"]}))
            if list(spec) == ["$or"] and len(spec["$or"]) > 1:
                mk("or_to_and", lambda m, si=si: m["pipeline"][si].__setitem__(
                    "$match", {"$and": m["pipeline"][si]["$match"]["$or"]}))
            for path, node in _walk(spec, ("pipeline", si, "$match")):
                if isinstance(node, dict):
                    for k, v in list(node.items()):
                        if k in CMP_SWAP:
                            mk("cmp_boundary", lambda m, path=path, k=k: _get(m, path).__setitem__(
                                CMP_SWAP[k], _get(m, path).pop(k)))
                            mk("cmp_negate", lambda m, path=path, k=k: _get(m, path).__setitem__(
                                CMP_NEG[k], _get(m, path).pop(k)))
                        if k == "$in" and isinstance(v, list) and path[-1] != "$expr":
                            mk("in_to_nin", lambda m, path=path: _get(m, path).__setitem__(
                                "$nin", _get(m, path).pop("$in")))
                        if k == "$nin" and isinstance(v, list) and None in v:
                            mk("null_guard_drop", lambda m, path=path: _get(m, path).__setitem__(
                                "$nin", [x for x in _get(m, path)["$nin"] if x is not None]))
                        if k == "$regex":
                            if v.startswith("^"):
                                mk("regex_unanchor", lambda m, path=path: _get(m, path).__setitem__(
                                    "$regex", _get(m, path)["$regex"][1:]))
                            if v.endswith("$"):
                                mk("regex_unanchor", lambda m, path=path: _get(m, path).__setitem__(
                                    "$regex", _get(m, path)["$regex"][:-1]))
                            if "$options" in node:
                                mk("regex_case", lambda m, path=path: _get(m, path).pop("$options"))
                        if k == "$and" and isinstance(v, list):
                            for j, item in enumerate(v):
                                if isinstance(item, dict) and list(item) == ["$ne"] and isinstance(item["$ne"], list) \
                                        and None in item["$ne"]:
                                    mk("null_guard_drop", lambda m, path=path, j=j: _get(m, path)["$and"].pop(j))
                        if k in ("$dateSubtract", "$dateAdd") and isinstance(v, dict) and "amount" in v:
                            mk("date_amount", lambda m, path=path, k=k: _get(m, path)[k].__setitem__(
                                "amount", _get(m, path)[k]["amount"] + 1))
                        # scalar equality in the query language: {field: literal}
                        if not k.startswith("$") and not isinstance(v, (dict, list)) and v is not None:
                            mk("eq_to_ne", lambda m, path=path, k=k: _get(m, path).__setitem__(k, {"$ne": _get(m, path)[k]}))
                            if isinstance(v, str) and v != v.upper():
                                mk("literal_case", lambda m, path=path, k=k: _get(m, path).__setitem__(k, _get(m, path)[k].upper()))
                            if isinstance(v, (int, float)) and not isinstance(v, bool):
                                mk("literal_plus1", lambda m, path=path, k=k: _get(m, path).__setitem__(k, _get(m, path)[k] + 1))
                        if k == "$ne" and v is None and path[-1] != "$expr":
                            mk("null_guard_drop", lambda m, path=path: _get(m, path[:-1]).pop(path[-1])
                               if len(_get(m, path)) == 1 else _get(m, path).pop("$ne"))
    return out


def existing_fixtures(r, schema):
    ast, _, _ = K.S.parse_sql(r["reference"]["sql"])
    an_obj = K.S.analyze(ast)
    an = an_obj.to_json()
    sch = {t: schema[t] for t in an["columns"]}
    seed_base = int(r["id"].split("_", 1)[1][3:11], 16) % 100000
    clock = K.V.server_clock() if an["time_relative"] else K.V.F.BASE_CLOCK
    fx = K.V.F.adequate_set(ast, an, sch, seed_base, clock)
    return ast, an_obj, sch, clock, fx


def caught_by_existing(mql, fx):
    import spider1_kit as SK
    for v in fx["kept"]:
        try:
            docs = K.V.run_mql(mql, v["tables"])
        except Exception:  # noqa: BLE001 - an MQL error is a kill (verify_pair would record it)
            return True, "mongo_error"
        pcols, rows = SK.docs_to_rows(docs)
        ok, _ = K.V.F.compare(pcols if docs else None, rows, v["columns"], v["rows"], v["ordered"],
                              v["blocks"] if v["ordered"] else None)
        if not ok:
            return True, "mismatch"
    return False, None


def caught_by_audit(ast, an, mql, sch, clock, seed):
    fam = AF.families(ast, an, sch, clock, seed)
    hits = []
    for name, tables in fam.items():
        verdict, _ = AF.compare_family(ast, an, mql, tables, sch, clock)
        if verdict in ("fail", "mongo_error"):
            hits.append(name)
    return hits


def run_label(r, schema):
    ast, an, sch, clock, fx = existing_fixtures(r, schema)
    rec = {"id": r["id"], "sql": r["reference"]["sql"], "n_variants": len(fx["kept"]),
           "uncovered": fx["uncovered"]}
    base_caught, why = caught_by_existing(r["target_mql"], fx)
    rec["label_passes_existing_fixtures"] = not base_caught
    if base_caught:
        rec["note"] = f"label itself fails regenerated fixtures ({why})"
        return rec
    ms = mutants(r["target_mql"])
    rng = random.Random(f"{SEED}:{r['id']}")
    if len(ms) > MAX_PER_LABEL:
        ms = rng.sample(ms, MAX_PER_LABEL)
    seed = int(hashlib.sha256(r["id"].encode()).hexdigest()[:8], 16)
    res = []
    for op, m in ms:
        killed, how = caught_by_existing(m, fx)
        entry = {"op": op, "killed_existing": killed, "how": how}
        if not killed:
            hits = caught_by_audit(ast, an, m, sch, clock, seed)
            entry["killed_audit"] = hits
            entry["mql"] = m
        res.append(entry)
    rec["mutants"] = res
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    schema = json.loads((K.ROOT / "data/interim/finetune_v1/schema.json").read_text(encoding="utf-8"))["tables"]
    labels = sorted(KC.labels(), key=lambda r: r["id"])
    sample = random.Random(SEED).sample(labels, a.n)
    done = {x["id"] for x in K.read_jsonl(OUT)}
    todo = [r for r in sample if r["id"] not in done]
    print("sample", len(sample), "todo", len(todo))

    def work(r):
        try:
            return run_label(r, schema)
        except Exception as e:  # noqa: BLE001
            return {"id": r["id"], "error": f"{type(e).__name__}: {str(e)[:300]}"}

    with cf.ThreadPoolExecutor(a.workers) as ex:
        for i, rec in enumerate(ex.map(work, todo), 1):
            K.append_jsonl(OUT, rec)
            if i % 25 == 0:
                print(i, flush=True)
    recs = K.read_jsonl(OUT)
    by = collections.defaultdict(lambda: [0, 0, 0, 0])
    for rec in recs:
        for m in rec.get("mutants", []):
            b = by[m["op"]]
            b[0] += 1
            b[1] += m["killed_existing"]
            if not m["killed_existing"]:
                b[2] += bool(m["killed_audit"])
                b[3] += not m["killed_audit"]
    print(f"{'operator':20} n  killed_existing  survivor_killed_by_audit  survivor_unkilled")
    for op, (n, k, ka, s) in sorted(by.items()):
        print(f"{op:20} {n:4} {k:4} ({k / n:.0%})  {ka:4}  {s:4}")


if __name__ == "__main__":
    main()
