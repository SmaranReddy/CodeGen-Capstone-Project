# Kaggle setup — clean Week 3 control experiment

The run generates raw outputs for three arms on the same 318 train items. It runs in order, `zs_rerun`, then
`random_k3`, then `hybrid_k3`, with one model load. It needs no credentials and no database: scoring
happens on Windows (`scripts/kaggle_week3/score_kaggle_outputs.py`).

## Upload

1. **Package dataset (private).** Kaggle → Datasets → New Dataset. Upload `week3_control_v1.zip`; Kaggle
   extracts it. Set visibility to **Private**.
2. **Model: no upload needed.** The notebook downloads `qwen2.5-coder-3b-instruct-q4_k_m.gguf` from
   `Qwen/Qwen2.5-Coder-3B-Instruct-GGUF`. That file's SHA-256 was checked on 2026-10-23 against the
   Hugging Face API; it is `724fb256bec1ff062b2f65e4569e871ad2e95ab2a3989723d1769c54294730b7`,
   identical to the local file. The notebook stops if the hash differs. Optionally, upload the GGUF as a
   second private dataset instead; the notebook uses an attached copy first.
3. **Notebook.** Kaggle → Code → New Notebook → File → Import Notebook → `week3_control.ipynb`. Then:
   - Add Input → the package dataset, and the model dataset if you made one.
   - Settings → Accelerator → **GPU T4 x2**. The runner sets `CUDA_VISIBLE_DEVICES=0`, so the whole
     model sits on one card. Avoid P100.
   - Settings → **Internet on**, needed to build `llama-cpp-python` and download the model. Kaggle requires
     phone verification for this.
   - Settings → Environment → **Pin to original environment**, so a resumed version gets the same
     image.
   - Sharing → **Private**.

Do not attach `.env`, credentials or any other project files. The notebook refuses to run if an `.env`
file is attached.

## Run

- Run **Save Version → Save & Run All (Commit)**. It runs in the background and keeps
  `/kaggle/working` as the version output. Interactive sessions lose their files when they stop.
- Expected time:
  - CUDA build of llama-cpp-python 0.3.36: about 15–30 min;
  - smoke test: a few minutes;
  - three arms × 318 items: about 1 h on a T4.
- Order of the run:
  1. The smoke test (3 items per arm plus a reverse-order determinism check) runs in a separate
     directory.
  2. The full run follows. It stops if the GPU is not used: it requires all layers offloaded and
     GPU memory to grow at model load.
- **If a version stops early:** open a new version of the same notebook and Add Input → *this
  notebook's previous version output*. The notebook resumes from its saved records, without
  duplicates or repeated inference, and reuses the built CUDA wheel.
- **If a resume is refused** with "backend differs": the new session has a different GPU, driver or
  image, and the runner never mixes backends. Detach the old output and run all three arms fresh
  (about 1 h). Keep the partial output for reference.

## Download

From the version's Output tab, download `week3_control_outputs.zip`. Do not unzip it yourself. On
Windows:

```
python scripts/kaggle_week3/score_kaggle_outputs.py --zip <path to week3_control_outputs.zip>
```

This verifies hashes, ids, counts, prompts and backend consistency, then scores every prediction on
Atlas under the read-only account.
