# scripts/nl2mql_mql0009: MQL-0009 retrieval-control experiment

The design, audit, readiness report and commands are in `docs/nl2mql_mql0009_retrieval_control.md`.

| File | Role |
|---|---|
| `pkg_src/mql0009_core.py` | Package core: arm runs, eligible view, random/fixed/hybrid ranking, plans and gates, prompt equivalence, precision and revision checks, generation, resume and export. Shipped byte-identical as `src/mql0009_core.py`. |
| `build_mql0009_package.py` | Builder. Runs the split, leakage audit, offline Atlas-form context, fixed selection, plans, config, package and zip. Needs a Python with `transformers` (tokenizer only). Refuses to overwrite without `--overwrite`. |
| `make_mql0009_notebook.py` | Writes the Kaggle notebook (plain JSON, tagged cells). |
| `mql0009_stats.py` | The registered analysis plan (never overwritten) and the statistics. |
| `mql0009_scoring.py` | Windows scoring adapter: `readiness`, `verify`, `score`, `analyze`. Uses the unchanged MQL-0006 evaluator, the read-only Atlas account and mandatory full parity. |
| `package_README.md`, `package_tests/`, `requirements-kaggle.txt`, `prediction_record.schema.json` | Copied into the package. |

**Rebuild:**

```
~/codegen_data/venv_tok/Scripts/python.exe -I -B scripts/nl2mql_mql0009/build_mql0009_package.py --tokenizer-dir ~/codegen_data/qwen3b_tok_488639f1 --overwrite
```

A rebuild changes `package_manifest.json`, because the file records a build time. After a rebuild, update
`EXPECTED_PACKAGE_MANIFEST_SHA256` in `mql0009_scoring.py`. Never rebuild after predictions exist.

**Tests:**
- `python -B -m pytest tests/test_nl2mql_mql0009.py -q`: the dry-run tests borrow the pinned tokenizer from `venv_tok`.
- `python -m pytest kaggle/nl2mql_mql0009_retrieval_control/tests -q`
