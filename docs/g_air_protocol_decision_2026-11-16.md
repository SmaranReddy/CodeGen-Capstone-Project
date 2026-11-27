# G-AIR review protocol: dated decision (2026-11-16)

This note records a protocol amendment. It does not change any result. The original G-AIR outputs in `outputs/ai_review_panel_v1/` are preserved and the panel was not rerun.

## Decision

- **G-AIR is the operational review gate for this dataset version** (spider-pool-v3, provisional corpus v2 and its candidate successor). The project owner chose an AI-only review on 2026-11-12 and confirmed it for this milestone on 2026-11-16. No human reviewer is being introduced.
- **G-AIR passed under its documented AI-only procedure.** Pre-registration: `outputs/ai_review_panel_v1/prereg.json`. Report: `outputs/ai_review_panel_v1/panel_report.json`. Ledger entry: DA-2026-11-16-AI-PANEL.
- **The historical human P2 gate of pilot run01 is NOT PASSED.** It was never run. G-AIR does not complete it, replace its result, or count as evidence that it would have passed.
- **This is a protocol amendment (a deviation from the original plan, which required a human review).** It is recorded as such here and in the ledger.
- **The data is AI-reviewed, not human-reviewed.** Every artifact, manifest and report must say so.

## What G-AIR measured

| Item | Result |
|---|---|
| Translation errors in a fresh sample of 200 accepted labels | 0 NO, 0 UNSURE; Wilson 95% upper bound 1.88%; pre-registered bound 5% |
| Source-label findings in the same sample (report only, no verdict) | 6 NO, 1 UNSURE of 200 |
| Planted source errors caught | 60 / 60 |
| Planted translation errors caught | 60 / 60 |
| Clean controls wrongly flagged | 0 / 60 |
| Repeat-run agreement | source 113 / 116, translation 98 / 98 |

## What these results do not show

- **They do not establish a population error rate below 5%.** The bound of 1.88% is on the rate at which *this panel* flags translation errors in a 200-item sample. If the panel misses a kind of error, that kind is not counted.
- **The 60 / 60 calibration result is not proof that unknown real defects are detected.** The planted errors were written by the same project, in known classes, with 5 to 6 items per class. Two known classes had no planted item (`left_to_inner_all_x`, `untyped_gt_text`).
- **Reviewer agreement is not independent evidence.** All reviewers are instances of one model family and can share the same blind spots.
- **The corpus records were not individually reviewed.** Only the 200-item sample and the detector-hit samples were.

## Limitations that stay attached to the result

- Same model family for every reviewer role.
- Partial reviewer isolation: the memory index was not scrubbed, and every reviewer disclosed seeing its one-line summaries.
- Stability model change: the last stability roles ran on a different model (DEV-7), so the stability figure mixes a model change with run-to-run variation.
- Some adversarial NO votes had no execution evidence; all were adjudicated (VAL-1, VAL-3).
- Two reviewers read the executor's source code (VAL-2, VAL-4); it contains no keys.
- The TC-1 NULL rule was applied inconsistently in two adjudications and was not relabelled (VAL-5).
- Several source-label NO verdicts on text-stored-number sorting fail only on hypothetical data and are debatable.
- Tool hashes changed after pre-registration to map placeholder evidence ids (DEV-1 to DEV-3).

The full deviation log is `outputs/ai_review_panel_v1/deviations.jsonl`.

## The 667 detector-hit records

- They stay quarantined for this corpus version (`data/processed/spider_pool_v3/exclusions_v3.jsonl`, tier C).
- This is a conservative data-quality exclusion. **They are not confirmed defects.** In the panel's samples most detector hits were not confirmed (D3 6 / 40, D4 4 / 21, D5 6 / 38, D6 8 / 40, D2 0 / 1; D1 26 / 40 including UNSURE).
- Their ids, provenance and detector reasons are kept, so a targeted recovery audit remains possible later. None was re-reviewed or reintroduced in the sign-off task.

## No independent validation is claimed

No human approved these labels. No reviewer outside this model family examined them. Nothing in this note should be read as either.
