# MQL-0009 continuation runbook (sessions 2 and 3, then scoring)

Status on 2026-11-18: **PARTIAL. 1,917 / 5,112 predictions. Nothing is scored. No accuracy exists.**
This runbook finishes the registered experiment unchanged. Package, design, prompts, seeds, model revision, bf16 and
decoding all stay as they are. It does not change the package, the corpus, the retriever or the analysis plan.

## 1. Where things stand (verified locally)

| Item | Evidence | Result |
|---|---|---|
| Session 1 export | `mql0009_predictions.zip` (repo root, 872,570 bytes, sha256 `e982a47df8fdb0f40e5740037ec1d6484534727746e13b79a5c9ccad85ff5a6e`); `mql0009_outputs/MQL0009_STATUS.json` | `session: 1`, `run_arms_this_session: [A_zero_shot, D_hybrid, B_random_s1]`, `stopped: null`, `resumed: null`, `problems: []`, `environment_consistent: true` |
| Completed runs | `MQL0009_PREDICTIONS_MANIFEST.json` | A_zero_shot 639/639, B_random_s1 639/639, D_hybrid 639/639; the other 5 runs are `absent` |
| Package identity | manifest `2a4f22e0…`, config `3f052e98…`, eval ids `6f1425f7…` | equal to the frozen pin in `scripts/nl2mql_mql0009/mql0009_scoring.py:55` |
| Model / precision | `run_config.json` of each run | `Qwen/Qwen2.5-Coder-3B-Instruct` @ `488639f1…` (hub-verified, 12 files); bf16 emulated on a Tesla T4 (cc 7.5); greedy; max_new_tokens 1024; batch 8 |
| Session 1 environment | STATUS `environment` | Python 3.13.16, torch `2.11.0+cu130`, transformers `5.20.2`, accelerate 1.15.0, tokenizers 0.23.3, CUDA 13.0 |
| Wall clock | `run_config.json` `generation_wall_s` | A 5,951.5 s; B_random_s1 7,876.1 s; D 8,181.9 s; about 6.1 h in total, inside the 11 h budget |
| Scorer verdict | `mql0009_scoring.cmd_verify` on the zip (staged in the scratchpad, not in `outputs/`) | `ok: false`; the only problems are "B_random_s2/s3/s4/s5, C_fixed: missing from the export(s)". The 3 present runs pass every id, prompt, plan, fingerprint, precision and environment check |
| Resume dry run | `verify_resume_source` + `seed_from_previous` + `export` on the extracted zip (scratchpad) | passes. 12 files seeded with unchanged hashes; status `partial 1917/5112`, `problems: []` |
| Atlas data | `outputs/.../provisioning/provision_20261118T040504.json` | `status: complete`; 69 collections and 16,885 documents digest-verified in `codegen_spider1_heldout` |

### Why sessions 2 and 3 did not run

- **Verified:** this was by design. The registered `session_plan` (`configs/experiment.json`) splits the 8 runs over 3
  Kaggle versions because they need about 18.6 T4-hours, which does not fit one 12 h session. Each version runs only
  its own `SESSION` arms. The notebook does not chain sessions: cell 2 reads `CFG["session_plan"][str(SESSION)]`.
  Session 1 ended normally. `stopped` is null, so there was no time-budget stop, and cell 8 printed "NOT COMPLETE …
  next version: attach this version's output, RESUME_FROM = 'auto', next SESSION". Sessions 2 and 3 are manual steps
  (README steps 6–7).
- **Not evidenced either way:** whether anyone attempted session 2 or 3 on Kaggle. There is no local export, log or
  checkpoint of either. No failure is recorded anywhere locally. Kaggle's own version history is the only place that
  could show an attempt.
- **Nothing found that blocks continuation**, with two caveats:
  1. **Environment drift is detected late.** README line 53 says a resume in another environment "is refused". In the
     code, that check runs only against an arm's *own* earlier `run_config` (`src/mql0009_core.py:506-509`). The
     session 2 and 3 arms have none. So a torch/transformers/GPU change would only surface at export
     (`mql0009_core.py:635-641`), after about 7 GPU-hours, and give `status: partial`. The scorer would then refuse the
     mix (`mql0009_scoring.py:271-273`). The guard cell in §2 stops before any GPU work instead.
  2. Unknown locally: whether *Pin to original environment* was set after session 1 (README step 5).

## 2. Kaggle steps

Use the **same Kaggle notebook** as session 1 and the same package dataset. Settings stay as in session 1:
Accelerator *GPU T4 x2* (one card is used), Internet **On**, Persistence *Files only*, **no secrets**, no other
datasets.

### 2.0 Before session 2 (check once)

1. *Settings → Environment*: confirm it is **pinned to the original environment** (session 1's version). If the pin
   option is gone and the current image has different torch/transformers, stop. The README rule is to regenerate all
   8 runs in one new environment under a new output, and never to mix environments. Do not continue in that case.
2. Kaggle GPU quota: sessions 2 and 3 need about 7.3 h + 4.9 h. Each must start with at least 8 h left.

### 2.1 Environment check: choose (a) or (b)

**(b) No notebook change:** in the pinned editor session, with the GPU on, run cells 1–3 interactively. Check that
cell 3 prints torch `2.11.0+cu130`, transformers `5.20.2`, device0 `Tesla T4`, compute capability `7.5`. Only then
*Save & Run All*. This relies on the commit run using the same pinned image as the editor.

**(a) Guard cell (stronger, because it checks inside the committed run itself):** insert this as a **new cell directly after cell 3** (the "environment" cell, before the model download). It only
reads, changes no setting, and stops the version within minutes if the environment differs from session 1.
It was tested locally against the real session 1 STATUS: it passes on identical values and stops on a simulated
transformers change.

```python
# ---- MQL-0009 continuation guard (read-only: changes no setting; stops BEFORE any GPU work on environment drift) ----
import json
from pathlib import Path
if RESUME_FROM in (None, "auto"):
    _src = sorted({p.parent for p in Path(INPUT_ROOT).rglob("MQL0009_STATUS.json")})
    assert len(_src) == 1, f"expected exactly one earlier MQL-0009 output under {INPUT_ROOT}, found {_src}"
    _src = _src[0]
else:
    _src = Path(RESUME_FROM)
_ref = json.loads((_src / "MQL0009_STATUS.json").read_text(encoding="utf-8"))["arm_runs"]["D_hybrid"]["environment_key"]
_now = {"gpu": GPU.get("device0"), "dtype": PRECISION["dtype"], "torch": ENV["packages"]["torch"],
        "transformers": ENV["packages"]["transformers"], "bf16_native": PRECISION["bf16_native"],
        "compute_capability": PRECISION["compute_capability"]}
_diff = {k: (_ref.get(k), v) for k, v in _now.items() if _ref.get(k) != v}
assert not _diff, f"ENVIRONMENT DRIFT vs session 1 (session 1, now): {_diff}. Do NOT continue; see README 'If the pinned environment is lost'."
print("environment identical to session 1:", _now)
```

### 2.2 Session 2: C_fixed, B_random_s2, B_random_s3 (about 7.3 h)

1. *Add data → Notebook Output*: select **session 1's version** of this notebook. Keep the package dataset attached.
   Attach nothing else. In particular, do **not** also upload `mql0009_predictions.zip` as a dataset, because two
   earlier outputs stop `RESUME_FROM="auto"` by design.
2. In cell 1, change only:
   ```python
   SESSION = 2
   RESUME_FROM = "auto"
   ```
   Leave `RUN_ARMS = None` and every other value unchanged.
3. *Save Version → Save & Run All (Commit)*.
4. Expected log lines:
   - cell 2: `runs this session: ['C_fixed', 'B_random_s2', 'B_random_s3']`
   - guard (if added): `environment identical to session 1: {... 'torch': '2.11.0+cu130', 'transformers': '5.20.2' ...}`
   - cell 7: `resume source verified: /kaggle/input/... 12 files seeded`, then `C_fixed: 639 of 639 to generate`, etc.
     A, D and B_random_s1 are **not** regenerated, because they are not in `RUN_ARMS`.
   - cell 8: `MQL-0009 STATUS: partial | 3834 / 5112 predictions`, `no problems`
5. Download `mql0009_predictions.zip` from the version's output and keep it, e.g. as
   `mql0009_predictions_session2.zip`. Do not overwrite the session 1 zip.

### 2.3 Session 3: B_random_s4, B_random_s5 (about 4.9 h)

1. *Add data → Notebook Output*: select **session 2's version**. **Remove** session 1's output from the inputs first.
   Session 2's output already contains all of session 1's runs (seeded), and two outputs stop `auto`.
2. Cell 1: `SESSION = 3`, `RESUME_FROM = "auto"`.
3. *Save & Run All (Commit)*.
4. Expected: `runs this session: ['B_random_s4', 'B_random_s5']`; cell 8:
   **`MQL-0009 STATUS: complete | 5112 / 5112 predictions`**, `no problems`.
5. Download the zip, e.g. as `mql0009_predictions_session3.zip`, into the repo root.

### 2.4 If something stops

| Message | Meaning | Action |
|---|---|---|
| `RESUME_FROM: expected exactly one earlier output, found [...]` | 0 or 2+ earlier outputs attached (or Kaggle extracted the zip next to the folder) | Attach exactly one previous output. If two paths point into the same output, set `RESUME_FROM = "/kaggle/input/<...>/mql0009_outputs"` (the folder that contains `MQL0009_STATUS.json`) |
| Session 1 cannot be attached as *Notebook Output* (not offered, or session 1 was not a committed version) | no notebook output to resume from | Upload the **exact** local `mql0009_predictions.zip` (sha256 `e982a47d…5a6e`) as a new **private** dataset and attach only it plus the package. Look in the Data panel for the path of the folder that contains `MQL0009_STATUS.json`; whether and how Kaggle extracts the zip is not verified from here. If `auto` finds one hit, keep it. Otherwise set `RESUME_FROM` to that folder |
| `resume source ...: hash check failed` | the attached output is not the one the zip came from, or is incomplete | Attach the correct version's output. **Do not reset**; report the message |
| `ENVIRONMENT DRIFT ...` (guard) or `arm runs were generated in different environments` (cell 8) | image changed | Stop. Do not mix. Report back; the README rule is to regenerate all 8 runs in one environment, which is a decision for you |
| `STOPPED: time budget of 11.0 h reached during X` | session ran long | Attach that version's output and rerun with the **same** `SESSION` and `RESUME_FROM = "auto"`. Checkpoints resume per question |
| `earlier run_config differs` / `completed prediction ... would be overwritten` | an attempt to mix runs or change a finished prediction | Stop and report. Never delete files to get past it |

Completion criteria (cell 8 of session 3, and `MQL0009_PREDICTIONS_MANIFEST.json`): `status: "complete"`,
`completed_predictions: 5112`, all 8 `per_arm` entries `complete`, `environment_consistent: true`, `problems: []`.
The A, B_random_s1 and D_hybrid `predictions_sha256` must still be `1ed94e2c…`, `c6ba66fa…` and `e7536b6b…`.

## 3. Windows-side scoring (after session 3)

Run from the repo root (`C:\Users\Asus\OneDrive\Desktop\CodeGen\codegen-capstone`) with the system Python that has pymongo.

**Prerequisites:**
- The read-only account `codegen_eval_readonly` has `find`, `listCollections` and `killCursors` on
  `codegen_spider1_heldout` (`docs/mql0009_heldout_provisioning.md` §"Read-only scoring account").
- The temporary `mql0009_provisioner` user is deleted or disabled.
- `.env` holds the read-only URI as `MONGODB_URI`.

`score` checks the grant and the read-only property itself and fails closed. Whether the grant has been made is not
recorded locally.

```powershell
# 1. offline readiness (no network)
python scripts/nl2mql_mql0009/mql0009_scoring.py readiness

# 2. offline validation: must print "ok": true and "problems": []. Passing all three zips makes the scorer itself prove
#    that the A / D / B_random_s1 copies carried into sessions 2-3 are byte-identical to session 1 (it rejects a run that
#    appears in two exports with different predictions).
python -B scripts/nl2mql_mql0009/mql0009_scoring.py verify --predictions-zip mql0009_predictions.zip --predictions-zip mql0009_predictions_session2.zip --predictions-zip mql0009_predictions_session3.zip

# 3. registered scoring: read-only Atlas preflight with full parity, all 8 runs, analysis, report (resumable)
python -B scripts/nl2mql_mql0009/mql0009_scoring.py score --predictions-zip mql0009_predictions.zip --predictions-zip mql0009_predictions_session2.zip --predictions-zip mql0009_predictions_session3.zip --run-name run_final

# 4. (optional) recompute the analysis from saved per-record results, no Atlas
python scripts/nl2mql_mql0009/mql0009_scoring.py analyze --run-dir outputs/nl2mql_mql0009_retrieval_control/scoring_runs/run_final
```

If `score` is interrupted, rerun the same command with the same `--run-name`; it resumes from its checkpoints.
Outputs go to `outputs/nl2mql_mql0009_retrieval_control/scoring_runs/run_final/scoring/`: `RUN_MANIFEST.json`,
`analysis.json`, `paired_per_question.jsonl` and `results_report.md`. Only after that do the post-experiment analysis,
the ledger result entry and the presentation-notes results entry get written.
