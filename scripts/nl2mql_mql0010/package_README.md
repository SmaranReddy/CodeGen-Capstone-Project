# nl2mql_mql0010_holdout: MQL-0010 Kaggle package (generate only)

Registration **A4**. The pins are in `registration/registration_index.json`, and the notebook checks every one of them before it runs.

| Item | Value |
|---|---|
| Model | `Qwen/Qwen2.5-Coder-3B-Instruct` @ `488639f1ff808d1d3d0ba301aef8c11461451ec5` |
| Precision and decoding | bfloat16 on a Tesla T4 (emulated), greedy, `max_new_tokens` 1024 |
| Evaluation | 716 held-out Spider-train questions over 18 databases |
| Retrieval | approved corpus v2.1 restricted to the registered view of 2,912 records; the 18 holdout databases are excluded |
| Arms (9) | A0, A0V, D0, D0V, D0G, E, EP, EF, FULL, assembled from 8 passes |
| Contents | no gold label, no credential, no database driver |

## What runs

| Session | Passes | Batch | ≈ T4-hours |
|---|---|---:|---:|
| 1 | `first:A0`, `first:D0`, `first:E`, then `delta` (EP and EF for the 52 items whose prompt changes) | 8, then 1 | 7.6 |
| 2 | `retry:A0V`, `retry:D0V`, `retry:D0G`, `retry:FULL` (one retry for each validator-flagged answer) | 1 | 2.4 |

**Integrity.** Every pass is resumable, through append-only checkpoints with fingerprints that are fsynced per record. At export, the notebook does the following:
- assembles the 9 arms;
- runs `scoring.verify_exports`, which checks reuse, the retried-item sets, the matched delta and the prompt hashes;
- writes `MQL0010_PREDICTIONS_MANIFEST.json`, `MQL0010_STATUS.json`, `SHA256SUMS.txt` and `/kaggle/working/mql0010_predictions.zip`.

## Steps

**1. Upload.** Upload `nl2mql_mql0010_holdout_bundle.zip` as a private Kaggle dataset. Kaggle unzips it, giving the `nl2mql_mql0010_holdout/` folder.

**2. Session 1.**
- **Notebook:** create it from `nl2mql_mql0010_holdout.ipynb`.
- **Settings:**
  - Accelerator: *GPU T4 x2*;
  - Internet: **on**;
  - Persistence: *Files only*;
  - attach the package dataset and **no** secrets.
- **Run:** keep `SESSION = 1` and choose *Save & Run All (Commit)*.
- **Afterwards:** download `mql0010_predictions.zip`, then pin the environment (*Settings → Environment → Pin to original environment*).

**3. Session 2.**
- Create a new version of the notebook.
- Attach the session-1 output as an input.
- Set `SESSION = 2` and `RESUME_FROM = "auto"`.
- Save and run.
- **Result:** the export reports `MQL-0010 STATUS: complete`. Download `mql0010_predictions.zip`.

**4. Interrupted or time-limited run.** The export is marked `partial`. Attach that output, set `RESUME_FROM = "auto"`, and run the same session again; completed records are never regenerated.

**5. What stops the notebook immediately:**
- a GPU other than a T4, or no bf16 support;
- a package or registration hash mismatch;
- a plan or prompt that differs from the frozen copy;
- a resumed run with a different environment;
- any attached file that looks like a credential.

## Not in this package

Scoring. It runs locally on Windows against Atlas, with `scripts/nl2mql_mql0010/mql0010_scoring.py` and its own authorisation.
