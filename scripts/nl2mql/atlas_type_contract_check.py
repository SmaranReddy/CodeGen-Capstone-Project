"""Mandatory read-only Atlas checks for any Atlas-scored use of sql2mql-v2.1-era artifacts (plan §A.4.2, §B.4.4).

    python -I -B scripts/nl2mql/atlas_type_contract_check.py contract [--experiment NAME]
        G-PRE TC-1 part: read-only privileges, every mapped collection present, exact counts, and a `$type`
        census of every field compared with data/interim/finetune_v2/type_contract_eval_v1.json. Fails closed on
        a disallowed / forbidden BSON type, an extra field, a non-null count mismatch, a count mismatch or a
        missing collection. The 2 known nonconformant wta_1 fields are reported, not failed (TC-1 rule 7).
    python -I -B scripts/nl2mql/atlas_type_contract_check.py like [--experiment NAME]
        G-LIKE Atlas part: the §B.4 counterexamples (Kelvin sign, long s, circled / Roman case pairs, `_` vs
        newline) executed ON THE ATLAS SERVER with the v2.1 regex forms, compared with SQLite LIKE computed
        locally. Uses `$documents`; if the server refuses it, falls back to `$regexMatch` over `$literal` values
        in a `$project` on one existing evaluation collection with `$limit: 1`.

Reports: outputs/<experiment>/preflight/atlas_type_contract_<ts>.json / atlas_like_check_<ts>.json.
Exit code 0 only on a pass. When Atlas is unreachable the report says status "pending_unreachable" and the
exit code is 2: a gate is never reported as passed without a live Atlas result.

Atlas is reached with MONGODB_URI (environment, else .env) through scripts/mongo_conn.py; the URI is never
printed or written. dev1034_core.py and local_atlas_scoring.py are NOT modified (their hashes pin MQL-0006/7);
only dev1034_core.privileges_report is imported.
"""
import argparse
import json
import sqlite3
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT / "scripts", ROOT / "scripts/nl2mql", ROOT / "scripts/finetune_v2"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import type_contract as TC  # noqa: E402

KNOWN_NONCONFORMANT = {"wta_1__players.birth_date", "wta_1__rankings.player_id"}
LIKE_CASES = [  # (pattern, values); SQLite truth is computed, never hard-coded
    ("%6862 Kaitlyn Knolls%", ["6862 Kaitlyn Knolls", "6862 kaitlyn knolls", "6862 Kaitlyn Knolls"]),
    ("%Scanner%", ["Scanner", "SCANNER", "ſcanner"]),
    ("k", ["k", "K", "K"]),
    ("S", ["s", "S", "ſ"]),
    ("%ix%", ["ix", "İx", "ıx"]),
    ("a_b", ["a\nb", "axb", "ab"]),
    ("Ⓐ", ["Ⓐ", "ⓐ"]),
    ("ⅰ", ["ⅰ", "Ⅰ"]),
]


class CheckError(Exception):
    pass


def privileges(client, dbname):
    import dev1034_core as D
    cs = client[dbname].command({"connectionStatus": 1, "showPrivileges": True})
    users, per, bad = D.privileges_report(cs)
    return users, per, bad


def check_contract(client, contract):
    """-> report dict with ok bool; never raises on a data violation (they are listed), raises on a write-capable user."""
    dbname = contract["atlas_db"]
    rep = {"atlas_db": dbname, "checked_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "contract": contract["contract"],
           "contract_body_sha256": contract["body_sha256"]}
    client.admin.command("ping")
    rep["server_version"] = client.admin.command("buildInfo").get("version")
    users, per, bad = privileges(client, dbname)
    rep.update(authenticated_users=users, offending_actions=bad)
    if not users:
        raise CheckError("Atlas connection is not authenticated")
    if bad:
        raise CheckError(f"Atlas user is NOT read-only ({len(bad)} write/admin actions); use codegen_eval_readonly")
    mdb = client[dbname]
    names = set(mdb.list_collection_names())
    want = contract["collections"]
    missing = sorted(set(want) - names)
    rep["missing_collections"] = missing
    viol, nonconf, census = {}, {}, {}
    for coll in sorted(set(want) & names):
        n, types = TC.mongo_type_census(mdb[coll])
        census[coll] = {"documents": n, "types": types}
        v = TC.compare_census(want[coll], n, types, "any")
        if v:
            viol[coll] = v
        for f, spec in want[coll]["fields"].items():
            if spec["K"] == "nonconformant":
                nonconf[f"{coll}.{f}"] = types.get(f, {})
    rep.update(collections_checked=len(census), violations=viol, nonconformant_fields=nonconf,
               unknown_nonconformant=sorted(set(nonconf) - KNOWN_NONCONFORMANT), census=census)
    rep["ok"] = not missing and not viol
    return rep


def sqlite_like(values, pat):
    con = sqlite3.connect(":memory:")
    con.execute("CREATE TABLE t (i INTEGER, v TEXT)")
    con.executemany("INSERT INTO t VALUES (?, ?)", list(enumerate(values)))
    got = sorted(i for (i,) in con.execute("SELECT i FROM t WHERE v LIKE ?", (pat,)))
    con.close()
    return got


def check_like(client, dbname, probe_collection):
    import sql2mql_v2_1 as N
    rep = {"atlas_db": dbname, "checked_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "converter": N.CONVERTER_VERSION}
    client.admin.command("ping")
    rep["server_version"] = client.admin.command("buildInfo").get("version")
    users, per, bad = privileges(client, dbname)
    if bad:
        raise CheckError("Atlas user is NOT read-only; use codegen_eval_readonly")
    mdb = client[dbname]
    cases, mode = [], None
    for pat, vals in LIKE_CASES:
        rx = N.like_regex_ascii(pat)
        want = sqlite_like(vals, pat)
        res = {}
        try:
            if mode in (None, "documents"):
                docs = [{"i": i, "v": v} for i, v in enumerate(vals)]
                q = list(mdb.aggregate([{"$documents": docs}, {"$match": {"v": {"$regex": rx, "$options": N.LIKE_OPTIONS}}},
                                        {"$project": {"_id": 0, "i": 1}}]))
                e = list(mdb.aggregate([{"$documents": docs}, {"$match": {"$expr": {"$regexMatch": {
                    "input": "$v", "regex": rx, "options": N.LIKE_OPTIONS}}}}, {"$project": {"_id": 0, "i": 1}}]))
                res = {"query": sorted(d["i"] for d in q), "regexMatch": sorted(d["i"] for d in e)}
                mode = "documents"
        except Exception as ex:  # noqa: BLE001  ($documents may be refused to a narrow role)
            if mode == "documents":
                raise
            rep["documents_refused"] = type(ex).__name__
            mode = "literal_project"
        if mode == "literal_project":
            proj = {f"m{i}": {"$regexMatch": {"input": {"$literal": v}, "regex": rx, "options": N.LIKE_OPTIONS}}
                    for i, v in enumerate(vals)}
            row = list(mdb[probe_collection].aggregate([{"$limit": 1}, {"$project": dict(proj, _id=0)}]))[0]
            res = {"regexMatch": sorted(i for i in range(len(vals)) if row[f"m{i}"])}
        ok = all(v == want for v in res.values())
        cases.append({"pattern_escaped": pat.encode("unicode_escape").decode(), "regex": rx, "sqlite": want,
                      "atlas": res, "ok": ok})
    rep.update(mode=mode, cases=cases, ok=all(c["ok"] for c in cases))
    return rep


def default_client():
    import mongo_conn
    from pymongo import MongoClient
    uri = mongo_conn.mongo_uri(ROOT / ".env")
    red = lambda s: mongo_conn._redact(s, uri)  # noqa: E731
    try:
        return MongoClient(uri, serverSelectionTimeoutMS=15000, socketTimeoutMS=120000,
                           appname="codegen-tc1-check", **mongo_conn.tls_kwargs(uri)), red
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(f"cannot create the Atlas client ({type(e).__name__}): {red(str(e))[:200]}") from None


def main(argv=None, client_factory=default_client):
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["contract", "like"])
    ap.add_argument("--experiment", default="pilot_v3_followup")
    a = ap.parse_args(argv)
    out = ROOT / "outputs" / a.experiment / "preflight"
    out.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%dT%H%M%S")
    name = "atlas_type_contract" if a.what == "contract" else "atlas_like_check"
    contract = TC.load_manifest(TC.EVAL_MANIFEST)
    redact = lambda s: s  # noqa: E731
    try:
        client, redact = client_factory()
        if a.what == "contract":
            rep = check_contract(client, contract)
        else:
            rep = check_like(client, contract["atlas_db"], sorted(contract["collections"])[0])
        rep["status"] = "pass" if rep["ok"] else "fail"
        code = 0 if rep["ok"] else 1
    except CheckError as e:
        rep, code = {"status": "fail", "ok": False, "error": redact(str(e))[:400]}, 1
    except Exception as e:  # noqa: BLE001  (network / auth / DNS: the gate stays PENDING, never passed)
        rep, code = {"status": "pending_unreachable", "ok": False,
                     "error": f"{type(e).__name__}: {redact(str(e))[:300]}"}, 2
    rep["where"] = "MongoDB Atlas (read-only evaluation account), live"
    p = out / f"{name}_{ts}.json"
    p.write_text(json.dumps(rep, indent=1, ensure_ascii=True, default=str), encoding="utf-8")
    short = {k: rep.get(k) for k in ("status", "server_version", "collections_checked", "missing_collections",
                                     "unknown_nonconformant", "mode", "error")}
    short["violations"] = len(rep.get("violations", {}))
    print(json.dumps(short, indent=1, default=str))
    print(f"report: {p.relative_to(ROOT)}")
    return code


if __name__ == "__main__":
    sys.exit(main())
