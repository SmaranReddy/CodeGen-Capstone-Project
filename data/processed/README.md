# data/processed/

Final, validated **NL → MongoDB/MQL** dataset used for experiments
(zero-shot baseline, RAG, fine-tuning, evaluation).

- Produced by running conversion **and** validation (`scripts/`).
- Only examples that pass validation against MongoDB are written here.
- Gitignored — never commit dataset files.

Files:
- `spider1_nl_mql.jsonl`: 279 validated Spider 1.0 dev examples (primary benchmark; see docs/spider1_dataset.md)
- `nl_mql_dataset.jsonl`: 68 validated Spider 2.0-Lite examples (secondary; see docs/conversion_report.md)
