# configs/

Configuration files (no secrets — secrets go in `.env`).

- `dataset.yaml` — dataset paths, query types, validation settings
- `mongodb.yaml` — non-secret MongoDB defaults; the actual URI is read from environment variables (`MONGODB_URI`, `MONGODB_DB`, `MONGODB_TIMEOUT_MS`)
