"""Adjudicate the corpus-v2 records that reference a TC-1 nonconformant field (sign-off task, section 4).

    python -I -B scripts/corpus_v2_signoff/adjudicate_nonconformant.py

Reads (never writes) provisional corpus v2, its verification records, TC-1 and the source SQLite databases.
Writes data/interim/corpus_v2_adjudication_v1.jsonl (one row per record; refuses to overwrite a different file).

Decision rule (conservative, fixed before looking at outcomes):
  QUARANTINE if the nonconformant field is used by any type-dependent operation in the MQL (filter, comparison,
             sort, grouping, accumulator, expression), or if the evidence is insufficient;
  INCLUDE    only if the field is used solely as a projected output value, so no engine type rule is involved,
             and verifier 3.1 accepted the label on the real data.
No SQL or MQL is rewritten. AI-adjudicated, not human-reviewed.
"""
import hashlib
import json
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/finetune_v2"))
import v2_common as C  # noqa: E402

CORPUS = ROOT / "outputs/sql2mql_v2_1/provisional_corpus_v2/provisional_rag_corpus.jsonl"
RECORDS = ROOT / "outputs/sql2mql_v2_1/provisional_corpus_v2/records.jsonl"
TC1 = ROOT / "data/interim/finetune_v2/type_contract_v1.json"
OUT = ROOT / "data/interim/corpus_v2_adjudication_v1.jsonl"
TYPE_DEPENDENT_STAGES = {"$match", "$group", "$sort", "$bucket", "$bucketAuto", "$sortByCount", "$lookup",
                         "$setWindowFields", "$densify", "$fill", "$unwind", "$redact", "$graphLookup"}


def refs(node, field):
    """True if `node` mentions the field as a path ("$f") or as a key ("f")."""
    if isinstance(node, str):
        return node == f"${field}" or node.startswith(f"${field}.")
    if isinstance(node, dict):
        return any(k == field or k.startswith(field + ".") or refs(v, field) for k, v in node.items())
    if isinstance(node, list):
        return any(refs(x, field) for x in node)
    return False


def usage(pipeline, field):
    """-> list of (stage index, stage operator, kind) for every stage that mentions the field."""
    out = []
    for i, st in enumerate(pipeline):
        (op, body), = st.items()
        if not refs(body, field):
            continue
        if op in TYPE_DEPENDENT_STAGES:
            kind = "type_dependent"
        elif op in ("$project", "$addFields", "$set"):
            # a bare path copy is type-neutral; any operator expression over the field is not
            plain = all(not refs(v, field) or v == f"${field}" for v in body.values())
            kind = "projection_only" if plain else "type_dependent"
        else:
            kind = "type_dependent"          # unknown stage: assume the worst
        out.append({"stage": i, "op": op, "kind": kind})
    return out


def main():
    tc = json.loads(TC1.read_text(encoding="utf-8"))
    ver = {}
    for line in RECORDS.open(encoding="utf-8"):
        r = json.loads(line)
        ver[r["id"]] = r
    rows = []
    for line in CORPUS.open(encoding="utf-8"):
        r = json.loads(line)
        bad = sorted(f for f, k in r["type_contract"]["fields"].items() if k == "nonconformant")
        if not bad:
            continue
        mql = r["mql"] if isinstance(r["mql"], dict) else json.loads(r["mql"])
        con = C.sqlite_ro(C.sqlite_path(r["db_id"]))
        fields = []
        for f in bad:
            coll, col = f.rsplit(".", 1)
            table = coll.split("__", 1)[1]
            spec = tc["collections"][coll]["fields"][col]
            vals = con.execute(f'SELECT typeof("{col}"), quote("{col}"), COUNT(*) FROM "{table}" '
                               f'WHERE typeof("{col}") NOT IN (?, \'null\') GROUP BY 1, 2',
                               ({"INTEGER": "integer", "REAL": "real"}.get(spec["affinity"], "text"),)).fetchall()
            fields.append({"field": f, "collection": coll, "tc1_class": spec["K"], "declared": spec["declared"],
                           "class_counts": spec["class_counts"],
                           "off_type_values_in_source_sqlite": [list(v) for v in vals],
                           "usage_in_mql": usage(mql["pipeline"], col)})
        try:
            sql_rows = con.execute(r["sql"]).fetchall()
        except sqlite3.Error as e:  # noqa: PERF203
            sql_rows = f"sqlite error: {e}"
        con.close()
        kinds = {u["kind"] for f in fields for u in f["usage_in_mql"]}
        v = ver.get(r["id"], {})
        vraw = v.get("verification") or {}
        vraw = json.loads(vraw) if isinstance(vraw, str) else vraw
        ch = vraw.get("checks", {})
        fx = ch.get("fixtures", {})
        vchecks = {"real_rows": ch.get("real_rows"), "l1_explicit_null": ch.get("l1_explicit_null"),
                   "l1_missing": ch.get("l1_missing"), "v6_determinism": ch.get("v6_determinism"),
                   "fixture_families": len(fx), "fixture_results": sorted(set(map(str, fx.values())))}
        accepted = v.get("outcome") == "accepted"
        if kinds == {"projection_only"} and accepted:
            decision = "INCLUDE"
            reason = ("the nonconformant field is only copied to the output; no filter, comparison, sort, grouping or "
                      "aggregation reads it, so no engine type rule is involved; verifier 3.1 accepted the label on "
                      "the real data (local MongoDB 8.0.34)")
        else:
            decision = "QUARANTINE"
            reason = ("the nonconformant field is read by a type-dependent operation ("
                      + ", ".join(sorted({u["op"] for f in fields for u in f["usage_in_mql"]
                                          if u["kind"] == "type_dependent"}))
                      + "); TC-1 gives no guarantee for a field holding two stored types") if "type_dependent" in kinds \
                else "insufficient evidence"
        rows.append({
            "id": r["id"], "corpus_entry_id": r["corpus_entry_id"], "db_id": r["db_id"],
            "collection": mql["collection"], "provenance": r["source"], "question": r["question"], "sql": r["sql"],
            "mql_sha256": hashlib.sha256(C.canon(mql).encode()).hexdigest() if isinstance(C.canon(mql), str)
            else hashlib.sha256(json.dumps(mql, sort_keys=True).encode()).hexdigest(),
            "nonconformant_fields": fields,
            "sqlite_result_rows": len(sql_rows) if isinstance(sql_rows, list) else sql_rows,
            "sqlite_result_head": [list(x) for x in sql_rows[:25]] if isinstance(sql_rows, list) else None,
            "local_verification": {"outcome": v.get("outcome"), "verifier": r["verification"]["verifier"],
                                   "checks": vchecks, "where": "local MongoDB 8.0.34, not Atlas"},
            "atlas_data_check": "not possible: the collection does not exist on Atlas (training database; Atlas holds "
                                "only the evaluation collections)",
            "decision": decision, "reason": reason,
            "sql_rewritten": False, "mql_rewritten": False,
            "review": "AI-adjudicated by rule, not human-reviewed", "adjudication_version": "corpus_v2_adjudication_v1",
        })
    body = "".join(json.dumps(x, ensure_ascii=False, sort_keys=True) + "\n" for x in rows)
    if OUT.exists() and OUT.read_text(encoding="utf-8") != body:
        raise SystemExit(f"{OUT} exists with different content; refusing to overwrite")
    OUT.write_text(body, encoding="utf-8")
    for x in rows:
        print(x["id"], x["decision"], "|", [(f["field"], f["off_type_values_in_source_sqlite"],
                                             [(u["op"], u["kind"]) for u in f["usage_in_mql"]])
                                            for f in x["nonconformant_fields"]],
              "| rows", x["sqlite_result_rows"], x["sqlite_result_head"][:8] if x["sqlite_result_head"] else None,
              "| checks", x["local_verification"]["checks"])
    print("written", OUT.relative_to(ROOT), len(rows), "sha256", hashlib.sha256(body.encode()).hexdigest())


if __name__ == "__main__":
    main()
