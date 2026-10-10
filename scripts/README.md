# scripts/

Runnable scripts (executed from the project root). Keep scripts thin — put
reusable logic in `src/`.

Planned Week 1 scripts:

- `inspect_dataset.py` — summarize raw Spider 2.0-Lite files and schemas
- `convert_to_mql.py` — run conversion, write to `data/interim/`
- `validate_dataset.py` — execute MQL examples against MongoDB, write validated data to `data/processed/`

Usage pattern:

```bash
python scripts/inspect_dataset.py
```
