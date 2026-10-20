"""Shared plumbing for the 68-example NL -> MQL model comparison.

Reuses the Week-1 conversion kit (`scripts/conversion_kit.py`) for the OFFICIAL
Spider comparator, the manifest/pattern registries and the Atlas client. Adds:

  * dataset/artifact loading for the frozen 68-example snapshot
  * per-example collection-import resolution (from the conversion artifacts)
  * a read-only MQL structural validator (rejects writes / server-side JS)
  * safe pipeline execution with a read-only collection allow-list
  * result -> DataFrame conversion + official scoring

This module never modifies the validated dataset or any existing artifact.
"""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from conversion_kit import (  # noqa: E402
    CONV_DIR,
    PILOT_DB,
    PILOT_DIR,
    ROOT,
    compare_pandas_table,
    manifest,
    mongo_client,
    patterns,
    score,
)

DATASET_PATH = ROOT / "data/processed/nl_mql_dataset.jsonl"

WRITE_STAGES = {"$out", "$merge"}
JS_STAGES = {"$where", "$function", "$accumulator"}
SERVER_ADMIN_STAGES = {
    "$listSessions", "$listLocalSessions", "$currentOp", "$collStats", "$indexStats",
    "$planCacheStats", "$shardedDataDistribution", "$listClusterCatalog",
    "$queryStats", "$rankFusion", "$scoreFusion",
}
FORBIDDEN_STAGES = WRITE_STAGES | JS_STAGES | SERVER_ADMIN_STAGES


def load_dataset(path=DATASET_PATH):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_artifact(iid):
    for d in (CONV_DIR, PILOT_DIR):
        p = d / f"{iid}.json"
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
    return None


def start_collection(artifact):
    mql = artifact.get("mql") or {}
    return mql.get("collection")


def _fields_kept(artifact):
    mapping = (artifact.get("mongodb") or {}).get("mapping") or {}
    fk = mapping.get("fields_kept") or {}
    return list(fk.keys())


def imports_for(iid, artifact):
    """Normalised list of imports: {collection, source_table, fields, filter?, indexes?}."""
    mongo = artifact.get("mongodb") or {}
    imps = mongo.get("imports")
    if imps:
        return imps
    # legacy pilot artifact (local041): single collection documented via `mapping`
    rel = artifact.get("relational_schema_used") or {}
    table = rel.get("table")
    coll = mongo.get("collection")
    if coll and table:
        return [{"collection": coll, "source_table": table,
                 "fields": _fields_kept(artifact) or rel.get("columns_needed") or [],
                 "filter": None}]
    # legacy artifacts that only record collection sizes (e.g. local297)
    colls = artifact.get("collections")
    if isinstance(colls, dict) and colls:
        return [{"collection": name, "source_table": None, "fields": [], "filter": None,
                 "documents": n} for name, n in colls.items()]
    return []


def derived_for(iid, artifact):
    return (artifact.get("mongodb") or {}).get("derived") or []


def collect_collections(iid, artifact):
    """Any collection name owned by this example, found anywhere in the artifact."""
    names = set()

    def walk(o):
        if isinstance(o, dict):
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)
        elif isinstance(o, str):
            if o.startswith(iid + "__") or o.startswith(iid + "_"):
                names.add(o)

    walk(artifact.get("mongodb"))
    walk(artifact.get("mql"))
    return names


def allowed_collections(iid, artifact):
    allowed = {start_collection(artifact)}
    allowed |= {imp["collection"] for imp in imports_for(iid, artifact)}
    allowed |= {d["collection"] for d in derived_for(iid, artifact)}
    allowed |= collect_collections(iid, artifact)
    allowed.discard(None)
    return allowed


def iter_stages(pipeline):
    for stage in pipeline or []:
        if isinstance(stage, dict):
            for name, body in stage.items():
                yield name, body
                if name == "$facet" and isinstance(body, dict):
                    for sub in body.values():
                        yield from iter_stages(sub)


def referenced_collections(obj):
    refs = set()
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in ("from", "coll") and isinstance(v, str):
                refs.add(v)
            else:
                refs |= referenced_collections(v)
    elif isinstance(obj, list):
        for item in obj:
            refs |= referenced_collections(item)
    return refs


UNION_WITH_KEYS = {"coll", "pipeline"}


def union_with_targets(obj):
    """Targets of every `$unionWith` at any depth -> (targets: set, error: str|None).

    Short form `{"$unionWith": "<coll>"}` must name a non-blank collection; long form must be
    `{"coll": "<coll>", "pipeline"?: [...]}` (no `db` or other keys, no collection-less form).
    """
    targets = set()
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == "$unionWith":
                if isinstance(v, str):
                    if not v.strip():
                        return targets, "empty $unionWith target"
                    targets.add(v)
                elif isinstance(v, dict):
                    coll = v.get("coll")
                    if not isinstance(coll, str) or not coll.strip():
                        return targets, "$unionWith without a collection name"
                    if set(v) - UNION_WITH_KEYS:
                        return targets, f"unsupported $unionWith keys {sorted(set(v) - UNION_WITH_KEYS)}"
                    if "pipeline" in v and not isinstance(v["pipeline"], list):
                        return targets, "$unionWith 'pipeline' must be a list"
                    targets.add(coll)
                else:
                    return targets, f"malformed $unionWith ({type(v).__name__})"
            sub, err = union_with_targets(v)
            targets |= sub
            if err:
                return targets, err
    elif isinstance(obj, list):
        for item in obj:
            sub, err = union_with_targets(item)
            targets |= sub
            if err:
                return targets, err
    return targets, None


def validate_mql(pred, allowed):
    """Structural validation of a model output against the read-only contract.

    Malformed `$unionWith` is always rejected. When `allowed` is non-empty, the target
    collection and every `from`/`coll`/`$unionWith` reference must be in it (an empty
    `allowed` disables the reference checks, as before).

    Returns (ok: bool, reason: str, parsed: dict|None).
    """
    if not isinstance(pred, dict):
        return False, "output is not a JSON object", None
    method = pred.get("method", "aggregate")
    if method not in ("aggregate", "find"):
        return False, f"unsupported method '{method}'", None
    coll = pred.get("collection")
    if not isinstance(coll, str) or not coll:
        return False, "missing 'collection'", None
    if allowed and coll not in allowed:
        return False, f"collection '{coll}' not among the example's allowed collections", None
    if method == "find":
        filt = pred.get("filter", {})
        if not isinstance(filt, dict):
            return False, "'filter' must be an object", None
        pipeline = [{"$match": filt}]
        if "projection" in pred and pred["projection"]:
            pipeline.append({"$project": pred["projection"]})
    else:
        pipeline = pred.get("pipeline")
        if not isinstance(pipeline, list) or not pipeline:
            return False, "missing non-empty 'pipeline' list", None
    for name, _ in iter_stages(pipeline):
        if name in FORBIDDEN_STAGES:
            return False, f"forbidden stage {name}", None
    union_targets, union_err = union_with_targets(pipeline)
    if union_err:
        return False, union_err, None
    if allowed:
        bad = (referenced_collections(pipeline) | union_targets) - allowed
        if bad:
            return False, f"references collections outside the example: {sorted(bad)}", None
    return True, "ok", {"method": method, "collection": coll, "pipeline": pipeline}


def to_dataframe(docs):
    if not docs:
        return pd.DataFrame()
    if all(isinstance(d, dict) for d in docs):
        return pd.DataFrame(docs)
    return pd.DataFrame({0: docs})


def execute_pipeline(db, collection, pipeline, timeout_ms=10000, allow_disk_use=True):
    return list(db[collection].aggregate(
        pipeline, allowDiskUse=allow_disk_use, maxTimeMS=timeout_ms))


def score_against_gold(pred_df, iid):
    """Official comparator; returns (verdicts, passed, matched_gold)."""
    rec = manifest()[iid]
    verdicts, ok = score(pred_df, rec)
    matched = sorted(k for k, v in verdicts.items() if v["score"] == 1)
    return verdicts, ok, matched


def resolve_imports(iid, artifact):
    """Expand each normalised import with the SQLite source info needed to (re)load it."""
    rec = manifest()[iid]
    out = []
    for imp in imports_for(iid, artifact):
        out.append({
            "instance_id": iid,
            "collection": imp["collection"],
            "source_table": imp.get("source_table"),
            "fields": imp.get("fields") or [],
            "filter": imp.get("filter"),
            "indexes": imp.get("indexes") or [],
            "sqlite_path": rec["database_path"],
            "db": rec["db"],
        })
    return out


def schema_for(iid, artifact):
    """Reconstruct the per-example schema (collection -> {field: declared_type})."""
    from conversion_kit import table_columns

    rec = manifest()[iid]
    schema = {}
    for imp in imports_for(iid, artifact):
        coll = imp["collection"]
        table = imp.get("source_table")
        fields = imp.get("fields") or []
        types = {}
        if table and fields:
            try:
                tc = dict(table_columns(rec, table))
            except Exception:
                tc = {}
            for f in fields:
                types[f] = tc.get(f, "UNKNOWN")
        schema[coll] = types
    for d in derived_for(iid, artifact):
        schema.setdefault(d["collection"], {"key": "string"})
    return schema


def bson_type(value):
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "double"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return type(value).__name__


def sample_collection_schema(db, coll, n=50):
    types = {}
    for doc in db[coll].find({}, limit=n):
        for k, v in doc.items():
            if k == "_id":
                continue
            types.setdefault(k, bson_type(v))
    return types


def write_jsonl(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, default=str, ensure_ascii=False) + "\n")
    return path


def get_client():
    return mongo_client()
