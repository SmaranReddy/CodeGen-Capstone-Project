# src/

Core Python package for CodeGen. Each subpackage maps to one stage of the project.

| Subpackage | Purpose | Status |
|---|---|---|
| `data/` | Loading raw/interim/processed datasets, shared schemas | Week 1 |
| `conversion/` | Adapting Spider (SQL-oriented) examples to MongoDB/MQL | Week 1 |
| `validation/` | Executing MQL against MongoDB and checking results | Week 1 |
| `models/` | Loading Qwen-2.5-Coder-Instruct, inference wrappers | Later weeks |
| `rag/` | Retrieval-augmented generation (schema/docs retrieval) | Later weeks |
| `evaluation/` | Metrics, execution accuracy, evaluation harness | Later weeks |

Modules should be importable as `from src.conversion import ...` when running
from the project root.
