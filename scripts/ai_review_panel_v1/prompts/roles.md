# AI review panel v1: role instructions

## Agent 1: source SQL auditor
Assess `source_sql_correct` for every item. Put `NOT_ASSESSED` in `mql_translation_correct`. Read the question
carefully, inspect the schema and the data, and run the SQL (and variants) to test whether it answers the question.

## Agent 2: SQL-to-MQL translation auditor
Assess `mql_translation_correct` for every item (judge MQL vs SQL only, under the TC-1 domain). Put `NOT_ASSESSED`
in `source_sql_correct`. Compare stage by stage; look for NULL handling, missing fields, joins, grouping, sort/limit,
regex/LIKE semantics, type comparisons and aggregation defaults. Build counterexamples for any suspected difference.

## Agent 3: adversarial end-to-end auditor
Assess BOTH dimensions for every item. Your job is to find problems others may miss: actively try to break each
item (edge data inside TC-1, alternative readings of the question, NULL/empty cases), but answer `YES` when an item
survives your attempts.

## Agent 4: independent adjudicator
Pass 1 (independent): assess BOTH dimensions for every item, exactly like a reviewer, without seeing anyone else's
output. Save it before pass 2.
Pass 2 (adjudication, a separate task): for each disputed item you receive the three reviewers' verdicts and their
evidence. Re-examine with FRESH evidence (run new queries; cite new ids). Never decide by counting votes. For each
dimension give a final value; use `UNSURE` when the evidence genuinely does not settle it. Record
`evidence_sufficient` (true/false) and a short `adjudication_reason`.
