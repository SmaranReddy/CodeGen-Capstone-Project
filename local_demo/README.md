# Local NL-to-MQL demo

From the repository root, in a normal macOS Terminal:

```sh
.venv/bin/python -m pip install -r local_demo/requirements.txt
.venv/bin/python -m local_demo.app
```

Open **http://127.0.0.1:7860**, select `allergy_1`, enter
**How many students are there?**, and select Generate. The first call loads
the model. The app binds to loopback and uses no public Gradio share link.

- **Zero-shot Qwen** generates MQL with the local quantized GGUF.
- **Fallback BM25** uses the same Qwen with up to three cross-database
  examples from `week2_validated.json`. Each example must have `PASS` and
  match a record in `train_spider.json` by database, question, and SQL.
  Only the question and MQL enter the prompt. Identical normalized questions
  and zero-score matches are skipped. A lack of examples produces a status.
- **Registered BGE RAG** and **fine-tuned Qwen + R3** display **Not
  configured**. These panels are integration boundaries. For live R3, wire a
  verified approved corpus and matching document vectors into the existing
  R3 retriever with the pinned live BGE encoder. For FT-R3, feed that same
  R3 prompt to the team inference path with the verified adapter and pinned
  base checkpoint or merged model.
- Each live panel shows raw output, parsed MQL, and static validation and
  retry diagnostics. Spider table names and declared types provide the
  schema. The demo leaves database execution disabled and makes no execution
  correctness claims.

## Environment settings

| Environment variable | Default or purpose |
| --- | --- |
| `MQL_DEMO_SPIDER_DIR` | `~/Documents/CodeGen-MongoDB/spider_data` |
| `MQL_DEMO_TABLES`, `MQL_DEMO_TRAIN`, `MQL_DEMO_EXAMPLES` | Override `tables.json`, `train_spider.json`, `week2_validated.json` |
| `MQL_DEMO_GGUF` | `models/qwen2.5-coder-3b-instruct-q4_k_m.gguf` |
| `MQL_DEMO_PORT`, `MQL_DEMO_GPU_LAYERS` | `7860`, `0` |
| `MQL_DEMO_APPROVED_CORPUS`, `MQL_DEMO_BGE_EMBEDDINGS`, `MQL_DEMO_BGE_SNAPSHOT` | Future registered R3 inputs |
| `MQL_DEMO_FT_ADAPTER`, `MQL_DEMO_FT_MODEL` | Future fine-tuned adapter and model paths |
| `MONGODB_URI` | Reserved for a future execution backend |

The reserved integration settings do not activate backends. Keep all data
and model artifacts outside Git. If llama.cpp fails to initialize Metal in
a restricted shell, launch from a normal Terminal. The quantized GGUF is a
demo runtime and the BM25 arm is a local fallback. Completed MQL-0015
scores refer to the registered benchmark.

## Smoke tests

```sh
.venv/bin/python -m pytest -q tests/test_local_demo.py
```

The tests use an injected deterministic generator for service checks. The
Gradio component check runs when Gradio is installed.
