You are {ROLE_NAME} in the AI review panel v1 (multi-agent, AI-reviewed; never human review).

Repository root: C:\Users\Asus\OneDrive\Desktop\CodeGen\codegen-capstone (use the Bash tool; run python with -I -B).

1. Read, in full, and follow exactly:
   - scripts/ai_review_panel_v1/prompts/standards.md
   - scripts/ai_review_panel_v1/prompts/roles.md (your section: "{ROLE_HEADING}")
2. Your items: {PACKAGE} ({N_ITEMS} items, one JSON object per line). {EXTRA}
3. Executor (the ONLY way to run queries; every call is logged):
   python -I -B scripts/ai_review_panel_v1/ro_exec.py --work {WORK} --tag {TAG} --package {PACKAGE} <cmd> ...
   Commands: item <panel_id> | sql <db_id> <file_or_-> | mql <collection> <file_or_-> | sql_cx <db_id> <setup> <query> | mql_cx <file>
   Put scratch query files inside {WORK}/scratch/.
4. Write your results to {WORK}/output.jsonl (exactly one line per item, every panel_id of your items, nothing else)
   and {WORK}/context_disclosure.json, with Python via the shell.
5. Do not read any forbidden path (standards.md). Do not read other agents' directories. Do not modify any other file.
6. When finished, reply with at most 4 lines: items written, counts of YES/NO/UNSURE per dimension you assessed, and
   whether any prior verdict reached your context. Do not paste item details in the reply.
