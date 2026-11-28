import json

import pytest

from local_demo.config import Settings
from local_demo.service import Demo, fallback_documents, schema_context


def schema(db, table):
    return {
        "db_id": db, "table_names_original": [table],
        "column_names_original": [[-1, "*"], [0, "id"]],
        "column_types": ["text", "number"], "foreign_keys": [],
    }


def example(db, question, **changes):
    return {
        "db_id": db, "question": question, "sql": "SELECT COUNT(*) FROM items",
        "collection": "items", "pipeline": [{"$count": "n"}], "validation": "PASS", **changes,
    }


class RecordingQwen:
    def __init__(self):
        self.prompts = []

    def count_tokens(self, text):
        return len(text.split())

    def generate(self, system, messages):
        self.prompts.append(messages)
        return '{"collection":"students","pipeline":[{"$count":"n"}]}'


@pytest.fixture
def settings(tmp_path):
    rows = [example("other", "Count the enrolled students"),
            example("target", "Count the active students")]
    files = {
        "tables": [schema("target", "students"), schema("other", "items")],
        "examples": rows,
        "training": [{"db_id": r["db_id"], "question": r["question"], "query": r["sql"]} for r in rows],
    }
    paths = {}
    for name, value in files.items():
        paths[name] = tmp_path / f"{name}.json"
        paths[name].write_text(json.dumps(value), encoding="utf-8")
    return Settings(**paths, gguf=tmp_path / "missing.gguf")


def test_schema_context():
    collections, prompt = schema_context(schema("target", "students"))
    assert collections["students"]["fields"] == {"id": ["number"]}
    assert "students: id (number)" in prompt


def test_fallback_requires_validation_and_training_provenance():
    good = example("train", "Count students")
    unknown = example("unknown", "Count orders")
    rejected = example("train", "Count students")
    rejected["validation"] = "FAIL"
    docs = fallback_documents([good, unknown, rejected], [
        {"db_id": "train", "question": good["question"], "query": good["sql"]},
    ])
    assert [d["db_id"] for d in docs] == ["train"]
    assert good["sql"] not in docs[0]["content"]


def test_answer_separates_fallback_and_registered_arms(settings):
    qwen = RecordingQwen()
    demo = Demo(settings, qwen=qwen)
    base, fallback, r3, ft = demo.answer("target", "How many students?", retry=True)
    assert base["mql"]["pipeline"] == [{"$count": "n"}]
    assert base["validation"]["schema_issues"] == []
    assert base["execution"] == fallback["execution"] == "Not run"
    assert fallback["examples"] == [{"database": "other", "question": "Count the enrolled students"}]
    assert "Solved examples" not in qwen.prompts[0][0]["content"]
    assert "Solved examples" in qwen.prompts[1][0]["content"]
    assert r3["status"] == ft["status"] == "Not configured"


def test_no_relevant_examples_does_not_generate_fallback(settings):
    qwen = RecordingQwen()
    _, fallback, _, _ = Demo(settings, qwen=qwen).answer("target", "xyzzy", retry=False)
    assert fallback["status"] == "No matching cross-database training examples"
    assert len(qwen.prompts) == 1


def test_missing_gguf_and_invalid_selection(settings):
    demo = Demo(settings)
    assert all(r["status"] == "Not configured" for r in demo.answer("target", "Count them", False))
    with pytest.raises(ValueError, match="Select a database"):
        demo.answer("unknown", "", False)


def test_optional_gradio_build(settings):
    pytest.importorskip("gradio")
    from local_demo.app import build_app
    app = build_app(Demo(settings, qwen=RecordingQwen()))
    config = app.get_config_file()
    labels = {component["props"].get("label") for component in config["components"]}
    assert {"Spider database", "Question", "Zero-shot Qwen · local GGUF", "Registered BGE RAG"} <= labels
