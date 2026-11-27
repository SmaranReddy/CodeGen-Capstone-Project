# AI review panel v1: shared standards (identical for every agent)

You are one reviewer in a multi-agent AI review panel. Your output is labelled **AI-reviewed**. It is never
described as human review. Work only from evidence you produce yourself.

## Items

Each item has: `panel_id`, `db_id`, `question` (natural language), `sql` (SQLite, Spider), `mql`
(`{"collection", "pipeline"}`, MongoDB aggregation; may be `null` when no translation exists), `schema` (DDL and row
counts), and `sqlite_result` / `mongo_result` previews computed on the real data. Some items may contain
deliberately inserted errors; some may be entirely correct. Judge every item on its own merits. Nothing about an
item's position, id or neighbours tells you anything.

## Dimension 1: `source_sql_correct` (SQL vs question)

- Does the SQL return what the question asks, on this schema and data and on any plausible data for this schema?
- Accepted Spider conventions, which are not errors: extra identifying columns unless the question clearly excludes
  them; a specific reading of an ambiguous question if the wording reasonably supports it.
- A `NO` needs a concrete counterexample or a data-backed reason (a query you ran, cited by its log id).
- If the question is too incomplete to know what is asked, that is `NO` (category `question_defect`).

## Dimension 2: `mql_translation_correct` (MQL vs SQL; NOT vs the question)

- Does the MQL return exactly what the SQL returns on **every database inside the type-conformant domain TC-1**?
- **TC-1 domain:** every column holds only values of its observed storage class (integer-only columns hold
  integers; numeric columns hold integers/reals; text columns hold strings; NULL allowed anywhere, stored either as
  explicit null or as a missing field). Inside a text column, ANY Unicode string is possible (for example
  U+212A KELVIN SIGN or U+017F LATIN SMALL LETTER LONG S). A text value inside a numeric column is OUTSIDE the
  domain and is not a valid counterexample.
- SQLite semantics are the reference: integer division truncates; LIKE folds case for the 26 ASCII letters only
  (`%` and `_` also match newlines); comparisons with NULL are not true; INNER JOIN drops unmatched rows; SQLite type
  affinity applies to literals compared with a column.
- Column order and values matter; column-name case and aliases do not. Row order matters only under ORDER BY;
  rows tied on the ORDER BY key may come back in any order. Duplicate rows (bag semantics) must match.
- Matching results on the real data is supporting evidence only. A `NO` needs a counterexample (a concrete database
  state inside TC-1 where the outputs differ, ideally executed with `sql_cx` / `mql_cx`) or a demonstrated rule from
  this list.
- If `mql` is `null`, answer `NOT_APPLICABLE`.

## Values

`YES` | `NO` | `UNSURE` (and `NOT_ASSESSED` for a dimension outside your role, `NOT_APPLICABLE` as above).
Use `UNSURE` honestly when the evidence does not settle the question; it is not a failure.
`confidence`: `HIGH` | `MEDIUM` | `LOW`.

## Evidence (mandatory)

Run every query through the logged read-only executor (the exact command line is given in your task). Each call
prints an id like `T-0007`; cite ids in `execution_evidence`. Never invent results. SQLite is opened read-only;
MongoDB is read-only (`aggregate` only); counterexamples run in memory (`sql_cx`) or as inline `$documents`
(`mql_cx`). You may not modify any file except your own output and log directory.

## Output (one JSON object per item, one line each, in your output file)

```
{"panel_id": "...", "source_sql_correct": "...", "mql_translation_correct": "...", "confidence": "...",
 "source_error_category": "none|sort_direction|wrong_column_or_literal|missing_distinct|count_semantics|
   join_omits_entities|comparison_direction|question_defect|text_ordering|aggregation|other",
 "translation_error_category": "none|filter|null_handling|sort_limit|aggregation|join|projection_grouping|
   regex_or_like|type_comparison|other",
 "evidence": "<= 600 characters: the reason, citing ids", "execution_evidence": ["T-0001", ...],
 "counterexample": null or {"sql_setup": "...", "docs": [...], "explanation": "..."}}
```

Write the file with Python through the shell (one `json.dumps` per line, `ensure_ascii=False`), not with an editor
tool. Then write a separate one-line file `context_disclosure.json`:
`{"prior_verdicts_seen": true|false, "details": "..."}` stating whether ANY earlier review verdict, error count or
finding about these items reached your context (for example through an auto-loaded memory file), and what it was.

## Allowed and forbidden paths

- Allowed to read: your batch package file, the Spider SQLite files `data/raw/spider1/spider_data/database/<db_id>/`,
  `data/raw/spider1/spider_data/tables.json`, the local MongoDB `spider_train_v2` (through the executor only), and
  your own work directory.
- Forbidden: `docs/`, any Claude memory directory or `MEMORY.md` content beyond what is auto-loaded, everything under
  `outputs/` except your own batch package and work directory, `data/interim/`, `data/processed/`, `data/splits*`,
  any Spider `dev*` / `test*` file, `scripts/` (other than running the executor), other agents' work directories.
  If an auto-loaded memory mentions earlier review results, do not use them and disclose them.
