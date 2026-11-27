"""MQL-0015 scoring entry point for an evaluator holding `find` (but NOT `listCollections`) on the MQL-0015 database.

Why this file exists (2026-11-24): the Atlas custom-role UI does not offer `listCollections`, so the scoring role
`mql0015_eval_readonly` grants only `find` on `codegen_spider1_test_mql0015`. The pinned preflight
(`dev1034_core.atlas_preflight`, hash pinned in the write-once registration) enumerates collections with
`mdb.list_collection_names()`, which sends `listCollections` WITHOUT `authorizedCollections` and therefore needs the
`listCollections` privilege. `mql0015_scoring.py` is hash-pinned in the frozen package config. Neither file is edited.

Instead, this wrapper passes the pinned `cmd_readiness` / `cmd_score` a client (through their existing
`client_factory` parameter) whose MQL-0015 database object enumerates collections with

    db.command({"listCollections": 1, "nameOnly": True, "authorizedCollections": True})

which MongoDB permits for a user that has any privilege on the collections (here: `find` on the whole database).
Everything else -- privilege refusal of any write/admin action, `find` requirement, all 180 collections, exact counts,
full type-tagged digest parity, the evaluator, the analysis -- runs unchanged in the pinned code.

Only `list_collection_names()` on `codegen_spider1_test_mql0015` is replaced; every other database / attribute / command
passes straight through to the real PyMongo objects (so `mdb[coll].aggregate(...)` is a real Collection call).

Usage (repo root; same subcommands and output as mql0015_scoring.py):
  python -B scripts/nl2mql_mql0015/mql0015_scoring_authorized.py readiness
  python -B scripts/nl2mql_mql0015/mql0015_scoring_authorized.py verify --predictions-zip mql0015_predictions.zip
  python -B scripts/nl2mql_mql0015/mql0015_scoring_authorized.py score  --predictions-zip mql0015_predictions.zip --run-name run_final
"""
import argparse
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import mql0015_scoring as SCO     # noqa: E402  (pinned; imported, never modified)

CM = SCO.CM
AN = SCO.AN
TARGET_DB = CM.ATLAS_DB
LIST_COLLECTIONS_CMD = {"listCollections": 1, "nameOnly": True, "authorizedCollections": True}
MAX_GETMORE = 1000


class EnumerationError(RuntimeError):
    pass


def authorized_collection_names(db):
    """Collection names via listCollections(nameOnly, authorizedCollections); follows getMore; fails closed."""
    res = db.command(dict(LIST_COLLECTIONS_CMD))
    cur = res.get("cursor")
    if not isinstance(cur, dict) or "firstBatch" not in cur:
        raise EnumerationError("listCollections returned no cursor")
    names = [d["name"] for d in cur["firstBatch"]]
    cid = cur.get("id", 0)
    coll = str(cur.get("ns", "")).partition(".")[2] or "$cmd.listCollections"
    rounds = 0
    while cid:
        rounds += 1
        if rounds > MAX_GETMORE:
            raise EnumerationError("listCollections cursor did not terminate")
        more = db.command({"getMore": cid, "collection": coll})
        mc = more.get("cursor", {})
        names.extend(d["name"] for d in mc.get("nextBatch", []))
        cid = mc.get("id", 0)
    if len(names) != len(set(names)):
        raise EnumerationError("listCollections returned duplicate names")
    return names


class _AuthorizedDB:
    """Proxy of the MQL-0015 Database: only list_collection_names() changes."""

    def __init__(self, db):
        object.__setattr__(self, "_db", db)

    def list_collection_names(self, *args, **kwargs):
        if args or kwargs:
            raise EnumerationError("list_collection_names() arguments are not supported by the authorized proxy")
        return authorized_collection_names(self._db)

    def __getitem__(self, name):
        return self._db[name]

    def __getattr__(self, attr):
        return getattr(self._db, attr)


class AuthorizedClient:
    """Proxy of MongoClient: client[TARGET_DB] -> _AuthorizedDB; everything else passes through."""

    def __init__(self, client, target_db=TARGET_DB):
        object.__setattr__(self, "_client", client)
        object.__setattr__(self, "_target", target_db)

    def __getitem__(self, name):
        db = self._client[name]
        return _AuthorizedDB(db) if name == self._target else db

    def __getattr__(self, attr):
        return getattr(self._client, attr)


def wrap_factory(base_factory):
    def factory():
        client, redact = base_factory()
        return AuthorizedClient(client), redact
    return factory


def authorized_client_factory():
    import local_atlas_scoring as L   # same .env MONGODB_URI, TLS rule and redaction as the pinned default
    return wrap_factory(L.default_client_factory)()


def main(argv=None, client_factory=None):
    factory = client_factory or authorized_client_factory
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("verify")
    v.add_argument("--predictions-zip", action="append", required=True)
    s = sub.add_parser("score")
    s.add_argument("--predictions-zip", action="append", required=True)
    s.add_argument("--run-name", required=True)
    sub.add_parser("readiness", help="read-only Atlas check of codegen_spider1_test_mql0015 (find-only evaluator)")
    a = ap.parse_args(argv)
    if a.cmd == "readiness":
        rep, out = SCO.cmd_readiness(client_factory=factory)
        print(json.dumps({k: rep.get(k) for k in ("atlas_db", "server_version", "authenticated_users", "offending_actions",
                                                  "collections_found", "missing_collections")}, indent=1))
        print("report:", out)
    elif a.cmd == "verify":
        ver, _, _, _ = SCO.cmd_verify(a.predictions_zip)
        print(json.dumps(ver, indent=1))
    else:
        ana, _man, run_dir = SCO.cmd_score(a.predictions_zip, a.run_name, client_factory=factory)
        print(AN.render(ana))
        print("run:", run_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
