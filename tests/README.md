# tests/

Unit tests for conversion and validation logic.

- `test_conversion_*.py` — Spider → MongoDB/MQL adaptation
- `test_validation_*.py` — MQL validation/result checking (use fixtures, avoid requiring a live MongoDB where possible)

Run from project root:

```bash
pytest
```
