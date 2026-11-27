"""Prompt for the Spider 1.0 NL -> MQL benchmark. Both models receive exactly this text.

The prompt contains the question, the full database schema (every collection of the
db_id, with fields, observed BSON types and declared SQL types), foreign keys from the
official tables.json, and two sample documents per collection. Gold SQL, gold MQL and
gold results are never included.
"""
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import spider1_kit as K  # noqa: E402

PROMPT_VERSION = "nl-mql-spider1-v1"

SYSTEM = """You are a MongoDB expert. Translate a natural-language question into ONE read-only
MongoDB aggregation query over the given database.

Output ONLY a JSON object (no prose, no markdown fences) with exactly these keys:
  "collection": the collection the pipeline starts from (one of the listed collections)
  "pipeline":   a JSON array of aggregation stages

Result contract:
- Each output document is one answer row; its fields are the answer columns.
- Return exactly the columns the question asks for, no extra fields. Exclude "_id" with
  {"_id": 0} unless it holds an answer value (a group key you want returned counts as a column).
- If the question asks for a count or a single value, return one document with one field.
- Sort only when the question asks for an order; apply limits the question implies.
Rules:
- Use only the listed collections and fields; names are case-sensitive. $lookup may only
  reference collections of the same database.
- Field types are given as observed in the data. Numbers stored as strings must be compared
  as strings or converted explicitly.
- Never use $out, $merge, $function, $accumulator or $where. Use plain JSON (no ObjectId(),
  ISODate() or other shell helpers).
"""

USER_TEMPLATE = """Database: {db}

Collections (field: observed type [declared SQL type]):
{schema}

Foreign keys:
{fks}

Sample documents:
{samples}

Question: {question}

Return the JSON object."""

_CACHE = {}


def _context(db_id):
    if db_id in _CACHE:
        return _CACHE[db_id]
    man = json.loads((K.INTERIM / "atlas_collections.json").read_text(encoding="utf-8"))
    samples = json.loads((K.INTERIM / "prompt_samples.json").read_text(encoding="utf-8"))
    lines, slines = [], []
    for name, info in sorted(man.items()):
        if info["db_id"] != db_id:
            continue
        lines.append(f"- {name}")
        for f, t in info["fields"].items():
            lines.append(f"    {f}: {'/'.join(t)} [{info['declared'].get(f) or '?'}]")
        for d in samples.get(name, []):
            slines.append(f"- {name}: {json.dumps(d, ensure_ascii=False)}")
    tj = {t["db_id"]: t for t in K.load_json(K.SPIDER / "tables.json")}[db_id]
    cols, tabs = tj["column_names_original"], tj["table_names_original"]
    fks = [f"- {db_id}__{tabs[cols[x][0]]}.{cols[x][1]} -> {db_id}__{tabs[cols[y][0]]}.{cols[y][1]}"
           for x, y in tj["foreign_keys"]]
    _CACHE[db_id] = ("\n".join(lines), "\n".join(fks) or "(none)", "\n".join(slines) or "(none)")
    return _CACHE[db_id]


def messages(db_id, question):
    schema, fks, samples = _context(db_id)
    user = USER_TEMPLATE.format(db=db_id, schema=schema, fks=fks, samples=samples, question=question)
    return [{"role": "user", "content": user}]


def template_hash():
    return hashlib.sha256((PROMPT_VERSION + SYSTEM + USER_TEMPLATE).encode("utf-8")).hexdigest()


def build_samples():
    """Freeze 2 sample docs per collection (first 2 in natural order) for reproducible prompts."""
    client = K.mongo_client()
    man = json.loads((K.INTERIM / "atlas_collections.json").read_text(encoding="utf-8"))
    out = {}
    for name in sorted(man):
        out[name] = [{k: v for k, v in d.items()} for d in
                     client[K.ATLAS_DB][name].find({}, {"_id": 0}).limit(2)]
    client.close()
    (K.INTERIM / "prompt_samples.json").write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")


if __name__ == "__main__":
    build_samples()
    print(messages("concert_singer", "How many singers do we have?")[0]["content"])
    print(PROMPT_VERSION, template_hash())
