# CLAUDE.md

Project-level instructions for Claude Code in this repository.

## Project scope (persistent rule, confirmed 2026-10-29)

- **The primary task is NL→MQL**: natural-language question + MongoDB schema → a read-only MQL query (`aggregate`/`find`).
- **MongoDB Atlas is the target execution environment.** Every reported NL→MQL metric is produced by executing the generated MQL on Atlas through the existing harness (`eval_kit.validate_mql` safety gate, read-only user `codegen_eval_readonly`, `spider1_kit.exec_match`). A local MongoDB may be used for building and validating training labels or native datasets, never for a reported NL→MQL model-evaluation metric. Label-validation and dataset-audit rates (for example a conversion pilot's L1 pass rate) may be measured locally, and must say so where they are recorded.
- **SQL experiments are a separate auxiliary track.** SQL-0001, spider-sql-v1 and every other NL→SQL result:
  - are labelled "SQL-only" wherever they appear;
  - go in the NL→SQL section of the ledger, never the NL→MQL section;
  - are never presented as MQL performance, ranked against MQL results, or shown next to an MQL number as a comparison.
- Never use SQL text from spider-sql-v1 (or any SQL set) directly as an MQL training label. SQL→MQL conversion goes through a versioned, audited converter (`docs/nl2mql_next_steps.md`).
- The 65-item MQL test partition (`data/splits_expanded/test.jsonl`) is used once, for the final evaluation, and only when the user explicitly requests it.
- The roadmap and its gates are in `docs/nl2mql_next_steps.md`.

## Experiment accuracy ledger (persistent rule)

After every model evaluation, fine-tuning experiment, RAG experiment or quantitative dataset audit, update `docs/experiment_accuracy_ledger.md`:

- Add a new entry with a unique ID, in the section for its task (NL→SQL, NL→MQL or dataset audits). Never edit or delete earlier entries; add dated notes to correct them.
- Record the model and pinned revision, the dataset file, split and size, the prompt, decoding and training settings, and the hardware.
- Report each metric as `correct / total (percent)`. Keep official, execution-only and unofficial metrics apart.
- Mark an entry **verified** only after its artifacts have been checked locally (hashes, record counts, metric recompute). Otherwise mark it **provisional**, and list any check that was not run.
- Give relative paths to the raw artifacts.
- Update "Current best results" only for results that are directly comparable (same task, dataset, split and protocol).
- Record no test-set (`test_full`) accuracy unless that run has actually happened.

Also log each experiment in `docs/presentation_notes.md`. Do not copy the ledger into this file.
