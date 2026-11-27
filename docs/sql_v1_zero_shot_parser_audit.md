# Parser-rejection audit: SQL-0001, zero-shot Qwen2.5-Coder-3B on Spider dev

Date: 2026-10-28. This is a diagnostic study of ledger entry **SQL-0001** (`docs/experiment_accuracy_ledger.md`). It does not change any SQL-0001 result.

**Scope.** Development data only: the 1,034 records in `data/processed/spider_sql_v1/dev_full.jsonl` and the 20 dev DBs in the bundle. No held-out test data was opened. No model inference was run.

**Artifacts.** The original run artifacts were read only. Their SHA-256 hashes were re-checked after the audit and still match `RUN_STATUS.json`.

## Summary

1. **The rejections reproduce exactly.** Re-running the official parser on every prediction reproduces `official_parse_ok` for 1,034/1,034 examples. 461 predictions are rejected. All 1,034 gold queries parse.
2. **No rejection is an SQL syntax error.**
   - 117 rejected predictions are invalid SQL for their schema: SQLite also refuses them, mostly with `no such column`.
   - The other **344 are valid SQLite**, and are rejected only because of limits of the 2019 official parser.
   - The most common cause is the `INNER JOIN` keyword, followed by column aliases and table aliases without `AS`.
3. **The 185 rejected predictions that match the gold result are not shown to be correct.**
   - The match shows that they return the gold rows on the one official dev DB.
   - On the test-suite variants of the same dev DBs, 24 of them return different rows from gold, so they are **not** equivalent to gold.
   - 155 match on all 3 variants. This is consistent with correctness but does not prove it.
4. **The official metrics are reproducible.** The Step 7.4 local re-score reproduces all six metrics, and every per-example field, exactly.
5. **The official EX and EM depend heavily on surface form.** A fixed set of rewrites only changes how the SQL is written, and it is checked to leave the result rows unchanged. Applied to rejected predictions, it would turn **158 more** into official EX hits and **120 more** into official EM hits.
6. **Consequence for fine-tuning.** Fine-tuning on Spider gold SQL also teaches Spider's surface style. A zero-shot → fine-tuned gain in official EX would therefore mix *format adaptation* with *semantic improvement*. See the recommendation at the end.

## 1. Inputs inspected

| What | Path |
|---|---|
| Predictions | `outputs/spider_sql_v1_zs_3b/v1/spider_sql_v1_zs_qwen25coder3b_v1/predictions.jsonl` |
| Per-example evaluation | `…/evaluation_results.jsonl` |
| Baseline summary and report | `…/baseline_summary.json`, `…/baseline_report.md` |
| Official evaluator | `kaggle/spider_sql_v1_dev_bundle_v1/code/official_eval/{process_sql.py,evaluation.py}` (taoyds/spider @ b7b5b8c, unmodified) |
| Scoring wrapper | `scripts/sql_v1/spider_zs_sql.py` (`OfficialEval.score_one`) |
| Test-suite DB variants (dev DBs only) | `data/raw/spider1/testsuite/` — 3 variants for each of 19 dev DBs. `wta_1` has none. These are dev databases from Zhong et al. (2020), **not** the held-out test split. |

**Why a rejection always scores 0.** When `get_sql` fails, `score_one` substitutes the official empty-SQL structure, as the official `evaluate()` loop does.
- `eval_exec_match` then compares results keyed by the parsed SELECT items. An empty SELECT gives `{}`, which never equals the gold map.
- So every rejected prediction scores 0 on official EX and EM, whatever it returns. All 461 do.

## 2. Step 7.4: local re-score from predictions only

This is the command from Step 7.4 of `docs/kaggle_spider_sql_v1_baseline.md`, run unchanged. Its output went to a new folder; the original results were not overwritten.

```powershell
python -I scripts\sql_v1\spider_zs_sql.py score --bundle kaggle\spider_sql_v1_dev_bundle_v1 --predictions outputs\spider_sql_v1_zs_3b\v1\spider_sql_v1_zs_qwen25coder3b_v1\predictions.jsonl --out outputs\spider_sql_v1_zs_3b\v1\rescored_local
```

Exit code 0.

| Metric (/1,034) | Kaggle (SQL-0001) | Local re-score |
|---|---|---|
| Official EX | 446 | 446 |
| Official EM | 378 | 378 |
| Official parser success | 573 | 573 |
| SQLite compile | 913 | 913 |
| SQLite execution | 913 | 913 |
| Result-set match (unofficial) | 654 | 654 |

- **Per-example records.** All 1,034 records are identical on all 17 shared fields, including parse, EX, EM, exec status, error text and row counts.
- **Breakdowns and cross-check.** The hardness breakdown is identical. The official CLI cross-check agrees (0.366 vs 0.3656). There are 0 evaluator errors.
- **Only differences:**
  - `official_eval/official_{gold,pred}.txt` were written with Windows CRLF line endings. They are byte-identical after CRLF→LF.
  - Local nltk is 3.9.3; Kaggle used 3.9.1.
- **No discrepancy.**

## 3. Method

All of this is in `scripts/sql_v1/audit_zs_parser_rejections.py`. Its output is in `outputs/spider_sql_v1_zs_3b/v1/parser_audit/` (`parser_rejections.jsonl`, `variant_checks.jsonl`, `parser_audit_summary.json`). The script is deterministic: a second run produced byte-identical files.

1. **Parser evidence.** Each prediction is run through the official `tokenize` → `get_tables_with_alias` → `parse_sql`. For each failure the script records:
   - the exception;
   - the innermost `process_sql.py` function and line;
   - the token the parser was looking at, plus the 4 tokens before it.

   This is read from the traceback; no official code is modified.
2. **SQLite evidence.** The script reuses the run's own `EXPLAIN` result and error text for each prediction.
3. **Minimal probes.** For each construct, a one-line query was run through the official parser. Each probe is valid SQLite (it executes on `concert_singer`).

   | Construct (alone) | Official parser | Failure point |
   |---|---|---|
   | Control: `FROM singer AS T1 JOIN … AS T2 ON …` | accepted | |
   | Column alias `avg(age) AS avg_age` | rejected | `parse_col:188`, token `as` |
   | Table alias without `AS` (`FROM singer s`) | rejected | `parse_table_unit:260`, token `s` |
   | `INNER JOIN` | rejected | `parse_table_unit:260`, token `inner` |
   | `LEFT JOIN` | rejected | `parse_table_unit:260`, token `left` |
   | `IN ('2014', '2015')` (literal list) | rejected | `parse_value:298`, token `,` |
   | `IN ('2014')` (one literal) | accepted | |
   | `IS NULL` | rejected | `parse_col:188`, token `null` |
   | Subquery in `FROM` with alias | rejected | `parse_col:188`, token `as` |
   | Scalar subquery as left operand (`WHERE (SELECT …) > 3`) | rejected | `parse_col:188`, token `select` |
   | Backtick identifier, `<>` | rejected | `parse_col:188` |

4. **Diagnostic rewrites (parse-only).** These only test whether a construct is the cause; they never replace the official metric. Four fixed rewrites were tried, each alone and all together:
   - drop column aliases, substituting the expression where the alias is referenced later;
   - add `AS` to bare table aliases;
   - `INNER/LEFT/CROSS JOIN` → `JOIN`;
   - unquote backticks, and `<>` → `!=`.

   A rewrite counts as *needed* when removing it from the full set makes parsing fail again (leave-one-out). When the full set still fails, the remaining parser error is classified from its token and context.
5. **Row preservation.** A normalised query is only used when it returns *exactly* the same rows as the original prediction on the official DB, so the rewrite did not change the answer.
6. **Test-suite variant check.** The prediction and the gold query are both run on every available variant of the dev DB, using the same comparison as the unofficial metric (ordered if the gold query has ORDER BY, otherwise multiset).

## 4. Rejection causes (461)

| Category | n | Result-set match | Evidence |
|---|---:|---:|---|
| Genuine SQL syntax error | **0** | – | No rejected prediction gets an SQLite `syntax error` or `incomplete input` |
| Database / SQLite dialect incompatibility | **0** | – | No `no such function` errors. No non-aggregate functions occur in the SQLite-valid rejections |
| Other identifiable: invalid SQL for the schema (SQLite also rejects) | **117** | 0 | `no such column` 109, `ambiguous column name` 5, `no such table` 2, `misuse of aggregate` 1 |
| Alias, qualification or formatting | **94** | 50 | All 94 parse after the formatting rewrites, with rows preserved |
| Unsupported construct (parser limitation) | **250** | 135 | Leave-one-out rewrites, or a residual parser error matching a probe |
| Unclassified | **0** | – | |

Every one of the 344 SQLite-valid rejections has a cause backed by evidence, so 0 remain unclassified.

**The 117 SQLite-invalid predictions** score 0 under any evaluator, so their parser rejection does not affect the score. 83 of them also contain `INNER JOIN`, so the parser would have rejected them anyway.

**Alias, qualification or formatting (94)**

| Cause (leave-one-out) | n |
|---|---:|
| Column alias (`COUNT(*) AS Number_of_Concerts`) | 55 |
| Table alias without `AS` (`FROM stadium s`) | 27 |
| Both | 12 |

**Unsupported constructs (250)**

| Cause | n |
|---|---:|
| `INNER JOIN` / `LEFT JOIN` keyword (212 alone, 5 together with a column alias) | 217 |
| `IN (…)` with more than one literal (22 alone, 3 with a join keyword, 1 with a table alias) | 26 |
| `IS NULL` (both with `LEFT JOIN`) | 2 |
| Scalar subquery as left operand (both with `INNER JOIN`) | 2 |
| Subquery in `FROM` | 2 |
| Subquery in `JOIN` (also with aliases) | 1 |

**Where the parser fails.** The raw failure locations across all 461 rejections are:

| Location | n |
|---|---:|
| `KeyError@parse_table_unit:260` | 354 |
| `AssertionError@parse_col:188` | 67 |
| `AssertionError@parse_value:298` | 22 |
| `KeyError@parse_col:178` | 18 |

The first token that fails is `inner` for 300 predictions and `as` for 60. 300 of all 1,034 predictions use `INNER JOIN`, against 451 that use any JOIN.

**Rejections by hardness.** Medium 234, extra 114, hard 86, easy 27.

**What a rejection means.** In the 344 SQLite-valid cases, a rejection says nothing about correctness. It means only that the query is written in a form the 2019 parser cannot read. `INNER JOIN` means exactly the same as `JOIN`. Column and table aliases are standard SQL.

## 5. The 185 rejected predictions with a result-set match

Of these 185, 135 have an unsupported construct and 50 a formatting cause. By hardness: medium 117, extra 38, hard 25, easy 5.

**What the match establishes**
- The prediction is valid SQLite on that DB.
- On the **one official dev DB**, it returns the same rows as gold. Rows are compared as tuples, so column order matters. Row order matters only when the gold query has ORDER BY; otherwise rows are compared as a multiset.

**What it does not establish**
- It does not show semantic equivalence. Spider DBs are small, so a wrong query can return the right rows by coincidence. For 20 of the 185 the gold result is empty, and for 64 it is a single value.
- It does not show that the prediction would pass the official EX if the parser could read it. Official EX also keys columns by the parsed SELECT expression (see section 6).

**Evidence from the test-suite variants** (3 variants per dev DB; none for `wta_1`)

| | Rejected with result-set match (185) | Reference: official EX = 1 (446) |
|---|---:|---:|
| Matches gold on all variants | 155 | 374 |
| Differs from gold on ≥ 1 variant (**provably not equivalent**) | 24 (13.4% of 179 checked) | 36 (8.8% of 410 checked) |
| No variants (`wta_1`) | 6 | 36 |

- Of the 24 that are distinguished, 12 differ on all three variants. 5 have an empty gold result and 10 a single-value gold result.
- **Example** `spider_dev_0022`: the prediction groups by `s.Name`, while gold groups by `stadium_id`. The two agree on the official DB but not on the variants.
- The official EX is not free of this problem either: 36 of its 446 hits are distinguished on the variants.

**After the formatting and join rewrites, with rows preserved,** the official evaluator scores the 185 as follows:

| Official EX after rewrite | n | All variants match | Distinguished | No variants |
|---|---:|---:|---:|---:|
| EX = 1 | 144 | 125 | 15 | 4 |
| EX = 0 | 19 | 13 | 6 | – |
| Not normalisable (e.g. `IN` lists) | 22 | 17 | 3 | 2 |

**Conclusion.** 125 of the 185 pass every available check: the official EX after a rewrite that preserves the rows, and all 3 variants. Even these are *not proven* correct. The rest are either distinguishable from gold or cannot be checked further. None of the 185 is counted as correct in any metric.

## 6. Other protocol findings

- **44 parsed predictions return the gold rows but have official EX = 0.**
  - **37 cases: official EX compares columns by their parsed SELECT expression, not by value.** For example, `COUNT(concert_ID)` and `COUNT(*)` are different keys, so equal values still fail.
  - **7 cases: official EX compares rows in returned order, even when gold has no ORDER BY.** For example, `spider_dev_0031` returns the same 6 rows in a different order.
- **21 predictions with official EX = 1 have an unofficial match of 0.** In all 21 the columns are permuted. Official EX ignores column order; the unofficial match does not.
- **Over-acceptance check.** The official parser silently ignores trailing tokens; a probe confirmed this (`… ORDER BY age DESC xyz` is accepted). No prediction or gold query among the 1,034 has unread tokens, so this did not affect SQL-0001.
- **4 predictions parse but fail in SQLite** (`spider_dev_0176`, `0228`, `0539`, `0922`). They score 0.

## 7. How much the official EX and EM depend on style (diagnostic only)

The rewrites are applied only to the 344 SQLite-valid rejections, and only where all rows are preserved. 308 qualify. The other 36 are either not normalisable (33) or change rows (3, all `LEFT JOIN` → `JOIN`, so excluded).

| | Official (SQL-0001, unchanged) | + formatting rewrites only | + `INNER JOIN` → `JOIN` too |
|---|---|---|---|
| EX | 446 / 1,034 (43.1%) | 497 (48.1%) | **604 (58.4%)** |
| EM | 378 / 1,034 (36.6%) | 414 (40.0%) | **498 (48.2%)** |

The `LEFT JOIN` cases add nothing to these gains. **These are not results.** They show that up to 158 EX hits (15.3 points) depend only on how the SQL is written.

## 8. Recommendation for the zero-shot vs fine-tuned comparison

**The protocol is reliable.** Results are reproducible and deterministic: Kaggle and the local re-score agree on every example. All denominators are 1,034, and the gold-as-prediction score is 1,034/1,034. Keep **official EX and official EM as the primary metrics, unchanged**.

**It is not sufficient on its own to attribute a fine-tuning gain.** The model fine-tuned on Spider train will learn Spider's gold style: `AS T1`, plain `JOIN`, no column aliases. That alone could move official EX by as much as about 15 points on this baseline. Report alongside it:

1. **Official parser acceptance**, for both models.
2. **A parser-independent execution metric:**
   - The current unofficial result-set match, clearly labelled secondary.
   - Preferably, test-suite execution accuracy using the official `test-suite-sql-eval` and the dev DB variants. Only 3 variants per DB are extracted now, and `wta_1` has none. More variants are in `data/raw/spider1_testsuite_download/testsuite_database.zip` and would have to be extracted. This has not been done.
3. **The parser-normalised diagnostic** from this audit's script, as a sensitivity check and never as a headline number.
4. **A split of gains** into examples the zero-shot model already got right in result terms and examples that are newly correct.

**Limitations that must be disclosed with any SQL-0001 comparison**

- **Not the test-suite EX.** Official EX at b7b5b8c is not the test-suite EX that papers report.
- **Style-sensitive parser.** It rejects valid SQL: `INNER/LEFT JOIN`, column aliases, table aliases without `AS`, multi-literal `IN` lists, `IS NULL`, and some subqueries. 344 valid predictions scored 0 for this reason alone.
- **Official EX is not a pure value comparison.** It keys columns by the parsed SELECT expression and is sensitive to row order.
- **The unofficial match is weaker evidence.** It checks one DB only, so coincidental matches exist: 24 of the 185 rejected matches are disproved by the variants.
- **Official EX has false positives too.** 36 of its 446 hits are distinguished on the variants.
- **Limited variant coverage.** The variant check covers 3 variants on 19 of 20 dev DBs; `wta_1` is not covered.
- **The wrappers** (read-only connection, time limit, lenient UTF-8 decoder) are as disclosed in the guide.
- **Possible confound in the comparison.** Prompt wording could also be used to steer the zero-shot model towards Spider style. That would be a new experiment with a new ledger ID, not a change to SQL-0001.
