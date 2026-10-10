"""Model prompt template and output contract for the 68-example NL -> MQL task.

Both models receive exactly the same system instruction, question and schema. The
reference SQL, reference MQL, gold CSVs and expected results are never included.

PROMPT_VERSION is bumped whenever the wording changes; the runner records it plus a
SHA-256 of the rendered template so runs are reproducible.
"""
import hashlib
import json

PROMPT_VERSION = "nl-mql-v1"

SYSTEM = """You are a MongoDB expert. Translate a natural-language question into a single
read-only MongoDB Query Language (MQL) query over the provided database schema.

Rules:
- Output ONLY a JSON object, no prose and no markdown fences.
- The JSON object has exactly these keys:
  - "method": "aggregate" or "find"
  - "collection": the name of the collection to run the query on (must be one of the schema collections)
  - for "aggregate": "pipeline": a JSON array of stages
  - for "find": "filter": a JSON object, and optionally "projection"
- Use only the collections and fields listed in the schema. Field names are exact and case-sensitive.
- The query must be READ-ONLY. Never use $out, $merge, $function, $accumulator or $where.
- Dates are stored as strings unless the schema says otherwise; parse them explicitly.
- Return the data rows that answer the question. Column aliases are free-form.
"""

USER_TEMPLATE = """Database: {db}

Schema (collection -> field: type):
{schema}

Question:
{question}

Return the MQL query as a single JSON object."""


def render_schema(schema):
    lines = []
    for coll, fields in schema.items():
        lines.append(f"- {coll}")
        for name, typ in fields.items():
            lines.append(f"    {name}: {typ}")
    return "\n".join(lines) if lines else "(no schema available)"


def build_user(db, question, schema):
    return USER_TEMPLATE.format(db=db, schema=render_schema(schema), question=question)


def template_hash():
    payload = (PROMPT_VERSION + "\n" + SYSTEM + "\n" + USER_TEMPLATE).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def messages(db, question, schema):
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": build_user(db, question, schema)}]


if __name__ == "__main__":
    print(json.dumps({"prompt_version": PROMPT_VERSION, "prompt_sha256": template_hash()}, indent=2))
