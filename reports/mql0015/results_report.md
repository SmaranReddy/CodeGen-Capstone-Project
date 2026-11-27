# MQL-0015 results

Strict execution accuracy on Atlas (cross-engine vs SQLite gold); everything not `correct` counts as wrong. Registered arms: A0V, R3V, FT-R3V; first-answer arms are descriptive.

| arm | correct / total | % | Wilson 95% |
|---|---:|---:|---|
| A0 | 471 / 1000 | 47.10 | [44.02, 50.2] |
| A0V | 497 / 1000 | 49.70 | [46.61, 52.79] |
| R3 | 515 / 1000 | 51.50 | [48.4, 54.59] |
| R3V | 544 / 1000 | 54.40 | [51.3, 57.46] |
| FT-R3 | 588 / 1000 | 58.80 | [55.72, 61.81] |
| FT-R3V | 598 / 1000 | 59.80 | [56.73, 62.8] |

## Primary (registered): FT-R3V vs R3V

+5.40 pp, paired bootstrap 95% CI [2.7, 8.1], cluster-bootstrap CI [2.49, 8.3], exact McNemar p = 0.0001402 (FT-R3V-only 125, R3V-only 71); H0 rejected at 0.05: True.

## Secondary (Holm over the two)

- R3V vs A0V: +4.70 pp, CI [1.7, 7.8], p 0.002508, Holm p 0.002508, reject True (140/93)
- FT-R3V vs A0V: +10.10 pp, CI [7.0, 13.3], p 1.073e-09, Holm p 2.146e-09, reject True (188/87)

Holm over all three (robustness): FT-R3V vs A0V: p_holm 3.219e-09 reject True; FT-R3V vs R3V: p_holm 0.0002805 reject True; R3V vs A0V: p_holm 0.002508 reject True

## Descriptive comparisons

- FT-R3 vs R3: +7.30 pp, CI [4.6, 10.0], McNemar p 1.346e-07 (132/59)
- R3 vs A0: +4.40 pp, CI [1.5, 7.4], McNemar p 0.003646 (132/88)
- FT-R3 vs A0: +11.70 pp, CI [8.5, 14.9], McNemar p 2.176e-12 (199/82)
- A0V vs A0: +2.60 pp, CI [1.6, 3.6], McNemar p 2.98e-08 (26/0)
- R3V vs R3: +2.90 pp, CI [1.9, 4.0], McNemar p 3.725e-09 (29/0)
- FT-R3V vs FT-R3: +1.00 pp, CI [0.4, 1.7], McNemar p 0.006348 (11/1)

## Outcome taxonomy

| arm | correct | incorrect | invalid | execution_failure | timeout | generation_failure | infrastructure | not_comparable |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A0 | 471 | 447 | 14 | 16 | 22 | 0 | 0 | 30 |
| A0V | 497 | 418 | 11 | 21 | 23 | 0 | 0 | 30 |
| R3 | 515 | 397 | 3 | 35 | 21 | 0 | 0 | 29 |
| R3V | 544 | 361 | 3 | 42 | 23 | 0 | 0 | 27 |
| FT-R3 | 588 | 329 | 20 | 13 | 24 | 0 | 0 | 26 |
| FT-R3V | 598 | 318 | 18 | 13 | 27 | 0 | 0 | 26 |

## By Spider hardness

| group | n | A0V | R3V | FT-R3V | FT-R3V−R3V pp | R3V−A0V pp |
|---|---:|---:|---:|---:|---:|---:|
| medium | 399 | 254 | 281 | 315 | +8.52 | +6.77 |
| hard | 217 | 52 | 59 | 64 | +2.30 | +3.23 |
| easy | 216 | 162 | 162 | 179 | +7.87 | +0.00 |
| extra | 168 | 29 | 42 | 40 | -1.19 | +7.74 |

## By structure (gold-SQL labels)

| group | n | A0V | R3V | FT-R3V | FT-R3V−R3V pp | R3V−A0V pp |
|---|---:|---:|---:|---:|---:|---:|
| no_anti_join | 965 | 478 | 539 | 596 | +5.91 | +6.32 |
| not_nested | 952 | 492 | 537 | 593 | +5.88 | +4.73 |
| no_set_operation | 915 | 482 | 536 | 594 | +6.34 | +5.90 |
| no_join | 598 | 383 | 398 | 433 | +5.85 | +2.51 |
| join | 402 | 114 | 146 | 165 | +4.73 | +7.96 |
| class:join | 327 | 100 | 140 | 159 | +5.81 | +12.23 |
| class:aggregation | 265 | 174 | 189 | 221 | +12.08 | +5.66 |
| class:simple | 244 | 185 | 195 | 208 | +5.33 | +4.10 |
| class:set_operation | 85 | 15 | 8 | 4 | -4.71 | -8.24 |
| set_operation | 85 | 15 | 8 | 4 | -4.71 | -8.24 |
| nested | 48 | 5 | 7 | 5 | -4.17 | +4.17 |
| class:nested | 45 | 5 | 7 | 5 | -4.44 | +4.44 |
| anti_join | 35 | 19 | 5 | 2 | -8.57 | -40.00 |
| class:anti_join | 34 | 18 | 5 | 1 | -11.76 | -38.24 |

## Sensitivity (descriptive)

- exclude_degenerate_gold (−26, n 974): A0V 485, R3V 534, FT-R3V 587; primary +5.44 pp, p 0.0001417
- exclude_ft_l5_partner (−5, n 995): A0V 492, R3V 539, FT-R3V 593; primary +5.43 pp, p 0.0001402
- exclude_near_duplicate_11 (−11, n 989): A0V 486, R3V 534, FT-R3V 588; primary +5.46 pp, p 0.0001402
- exclude_question_in_train (−7, n 993): A0V 490, R3V 537, FT-R3V 591; primary +5.44 pp, p 0.0001402
- R3 prompt over_2048 tokens (n 144): FT-R3V−R3V +8.33 pp, p 0.08069
- R3 prompt at_most_2048 tokens (n 856): FT-R3V−R3V +4.91 pp, p 0.0009661
- cluster bootstrap (primary): CI [2.49, 8.3] over 845 clusters
- post-stratified accuracy %: {'A0V': 50.526, 'R3V': 54.654, 'FT-R3V': 60.108} (raw {'A0V': 49.7, 'R3V': 54.4, 'FT-R3V': 59.8})

## Recovery

- A0V: retried 152, kept 85, fixed 26, broken 0, retry generation failed 0, first generation failed 0
- R3V: retried 154, kept 83, fixed 29, broken 0, retry generation failed 0, first generation failed 0
- FT-R3V: retried 130, kept 43, fixed 11, broken 1, retry generation failed 0, first generation failed 0
