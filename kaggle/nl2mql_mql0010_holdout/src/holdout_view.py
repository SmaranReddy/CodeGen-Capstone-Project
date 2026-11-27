"""R3 (amendment A4): enforce the registered holdout retrieval view in code.

Every retrieval index used for MQL-0010 holdout inference is built ONLY from `load_view(...)`. It returns the
approved-corpus records of the 105 non-holdout databases and fails closed unless:
  * the corpus file hash equals the approved v2.1 hash (the corpus file is read, never written),
  * the view's id list hashes to split_registration.holdout_retrieval_view.ids_sha256,
  * no record belongs to any of the 18 registered holdout databases.
`assert_docs_exclude` re-checks the built documents, and planner.build_plan(..., forbidden_db_ids=HOLDOUT_DBS)
refuses any document from a holdout database (defence in depth: a second, independent check at ranking time).
"""
import hashlib
import json
from pathlib import Path

CORPUS_SHA256 = "96552f061afba13792d75240c18001e87e70f4aebf9d0934dfe27c0a7dd199bf"


class ViewError(ValueError):
    pass


def _sha_text(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def load_registration(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def holdout_dbs(reg):
    return frozenset(reg["holdout"]["dbs"])


def load_view(corpus_path, reg):
    raw = Path(corpus_path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != CORPUS_SHA256:
        raise ViewError("approved corpus v2.1 hash mismatch")
    rows = [json.loads(x) for x in raw.decode("utf-8").splitlines() if x.strip()]
    hdb = holdout_dbs(reg)
    view = [r for r in rows if r["db_id"] not in hdb]
    ids_sha = _sha_text("\n".join(r["id"] for r in view))
    exp = reg["holdout_retrieval_view"]
    if ids_sha != exp["ids_sha256"] or len(view) != exp["n_records"]:
        raise ViewError(f"view ids/count differ from the registration ({len(view)} vs {exp['n_records']})")
    assert_rows_exclude(view, hdb)
    return view


def assert_rows_exclude(rows, hdb):
    bad = sorted({r["db_id"] for r in rows if r["db_id"] in hdb})
    if bad:
        raise ViewError(f"holdout databases present in a retrieval source: {bad}")


def assert_docs_exclude(docs, hdb):
    bad = sorted({d["db_id"] for d in docs if d.get("db_id") in hdb})
    if bad:
        raise ViewError(f"holdout databases present in retrieval documents: {bad}")
    return True
