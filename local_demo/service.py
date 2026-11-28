"""Prompt building, train-only fallback retrieval, and local generation."""
import json
import sys

from .config import ROOT

# Existing experiment modules use bare sibling imports.
for directory in ("kaggle/nl2mql_mql0009_retrieval_control/src", "scripts/nl2mql_mql0010/src"):
    sys.path.insert(0, str(ROOT / directory))

from prompt_v2 import budgeted_block
from rag_block import insert_block
from recovery import assess, recover
from retrieval import Retriever, _norm_q
from validator import SchemaValidator

SYSTEM = (ROOT / "finetuning/q3b_r16_q10/inference/system_prompt.txt").read_text(encoding="utf-8")
CONTEXT = 4096
MAX_NEW = 256


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def schema_context(entry):
    database = entry["db_id"]
    names = entry["table_names_original"]
    collections = {name: {"db_id": database, "fields": {}} for name in names}
    for (index, column), kind in zip(entry["column_names_original"], entry["column_types"], strict=True):
        if index >= 0:
            collections[names[index]]["fields"][column] = [kind]
    lines = [f"Database: {database}", "Collections (Spider declared types):"]
    for name, meta in collections.items():
        fields = ", ".join(f"{field} ({types[0]})" for field, types in meta["fields"].items())
        lines.append(f"- {name}: {fields}")
    if entry.get("foreign_keys"):
        lines.append("Foreign keys:")
        for left, right in entry["foreign_keys"]:
            li, lc = entry["column_names_original"][left]
            ri, rc = entry["column_names_original"][right]
            lines.append(f"- {names[li]}.{lc} -> {names[ri]}.{rc}")
    return collections, "\n".join(lines)


def fallback_documents(examples, training):
    train_keys = {(r["db_id"], r["question"].strip(), r["query"].strip()) for r in training}
    docs = []
    for i, row in enumerate(examples):
        if row.get("validation") != "PASS":
            continue
        database, question, sql = row["db_id"], row["question"].strip(), row["sql"].strip()
        if (database, question, sql) not in train_keys:
            continue
        if not isinstance(row.get("collection"), str) or not isinstance(row.get("pipeline"), list):
            continue
        query = {"collection": row["collection"], "pipeline": row["pipeline"]}
        docs.append({
            "doc_id": f"week2-{i}", "kind": "example", "db_id": database,
            "question": question, "mql": query,
            "content": f"Question: {question}\nMQL: {json.dumps(query, ensure_ascii=False)}",
        })
    return docs


class LocalQwen:
    def __init__(self, path, gpu_layers):
        self.path = path
        self.gpu_layers = gpu_layers
        self.model = None

    def load(self):
        if self.model is None:
            from llama_cpp import Llama
            self.model = Llama(
                model_path=str(self.path), n_ctx=CONTEXT, n_threads=4,
                n_gpu_layers=self.gpu_layers, verbose=False,
            )
        return self.model

    def count_tokens(self, text):
        return len(self.load().tokenize(text.encode(), add_bos=False, special=True))

    def generate(self, system, messages):
        model = self.load()
        model.reset()
        result = model.create_chat_completion(
            messages=[{"role": "system", "content": system}, *messages],
            temperature=0, seed=42, max_tokens=MAX_NEW,
        )
        return result["choices"][0]["message"]["content"] or ""


def unavailable(reason):
    return {"status": "Not configured", "detail": reason}


class Demo:
    def __init__(self, settings, qwen=None):
        self.settings = settings
        self.schemas = {r["db_id"]: r for r in read_json(settings.tables)}
        if not self.schemas:
            raise ValueError("Spider schema file contains no databases.")
        self.qwen = qwen if qwen is not None else LocalQwen(settings.gguf, settings.gpu_layers)
        self.model_available = qwen is not None or settings.gguf.is_file()
        self.fallback_error = None
        try:
            docs = fallback_documents(read_json(settings.examples), read_json(settings.training))
        except (OSError, ValueError, KeyError, TypeError) as error:
            docs = []
            self.fallback_error = f"Could not load training examples: {error}"
        self.retriever = Retriever(docs, method="bm25_raw")

    def answer(self, database, question, retry):
        if database not in self.schemas or not question or not question.strip():
            raise ValueError("Select a database and enter a question.")
        question = " ".join(question.split())
        collections, context = schema_context(self.schemas[database])
        user = f"{context}\n\nQuestion: {question}\n\nReturn the JSON object."
        validator = SchemaValidator(database, collections)

        def run(prompt, examples=()):
            try:
                if self.qwen.count_tokens(SYSTEM) + self.qwen.count_tokens(prompt) + MAX_NEW + 96 > CONTEXT:
                    return {"status": "Prompt too long", "execution": "Not run"}
                first = self.qwen.generate(SYSTEM, [{"role": "user", "content": prompt}])
                # The retry includes the first answer and feedback. Reserve extra context for both.
                can_retry = retry and (
                    self.qwen.count_tokens(SYSTEM) + self.qwen.count_tokens(prompt)
                    + self.qwen.count_tokens(first) + MAX_NEW + 512 <= CONTEXT
                )
                final, trace = recover(SYSTEM, prompt, first, validator, self.qwen.generate, enabled=can_retry)
                query, issues = assess(final, validator)
                return {
                    "status": "Generated", "mql": query, "raw_output": final,
                    "validation": {
                        "parsed": query is not None,
                        "schema_issues": None if issues is None else [issue.text() for issue in issues],
                        "retry_requested": retry, "retry": trace,
                    },
                    "examples": [{"database": d["db_id"], "question": d["question"]} for d in examples],
                    "execution": "Not run",
                }
            except Exception as error:
                return {"status": "Inference or validation failed", "detail": str(error), "execution": "Not run"}

        if not self.model_available:
            base = fallback = unavailable(f"Set MQL_DEMO_GGUF to a local Qwen GGUF. Missing: {self.settings.gguf}")
        else:
            base = run(user)
            ranked = self.retriever.rank_examples(question, {"db_id": database})
            eligible = [doc for doc, score in ranked if score > 0 and _norm_q(doc["question"]) != _norm_q(question)]
            if self.fallback_error:
                fallback = unavailable(self.fallback_error)
            elif not eligible:
                fallback = {"status": "No matching cross-database training examples", "execution": "Not run"}
            elif base["status"] == "Inference or validation failed":
                fallback = {"status": "Shared Qwen runtime failed", "execution": "Not run", "detail": base["detail"]}
            else:
                chosen, _, block, *_ = budgeted_block(
                    database, SYSTEM, user, eligible, 3, self.qwen.count_tokens,
                    1024, CONTEXT, MAX_NEW, 96,
                )
                fallback = run(insert_block(user, block), chosen) if chosen else {
                    "status": "No examples fit the prompt budget", "execution": "Not run",
                }
        return (
            base, fallback,
            unavailable("Registered R3 backend is not wired. Requires the approved corpus, masked BGE vectors, and live pinned BGE encoding."),
            unavailable("Fine-tuned backend is not wired. Requires registered R3 plus the team adapter and compatible base weights, or a verified merged model."),
        )
