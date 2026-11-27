"""MQL-0010 prompt construction: the frozen RAG block format plus two optional, flag-controlled additions.

* The reference block itself is rendered by the UNCHANGED MQL-0008/0009 `rag_block.render_block` /
  `insert_block` (byte-identical copies), so with every flag off a plan reproduces the MQL-0009 D prompt.
* Component F (fk_type_notes="numeric_unified"): schema-derived notes for foreign keys whose two sides have
  INCOMPATIBLE observed BSON types. The dormant MQL-0008 helper (`rag_prompt.fk_type_notes`) compares raw type sets,
  so it also fires on int vs double and on nullable keys; MongoDB equality matches numerically equal int/long/double/
  decimal values, so those notes would be false. Here numeric types are unified and "null" ignored; a note is
  emitted only for genuinely different classes (e.g. string vs number) - the case where an equality $lookup
  returns nothing. Uses the schema context only (inference-time).
* Component V recovery prompt: one follow-up turn carrying the validator's issues (see recovery.py).
"""
import rag_block as RB

NUMERIC = {"int", "long", "double", "decimal", "number"}


def type_class(types):
    t = {x for x in types if x not in ("null", "missing", "undefined")}
    return frozenset("number" if x in NUMERIC else x for x in t)


def fk_type_notes(db_id, collections, tables_entry, mode="numeric_unified"):
    """-> list of note lines (rag_block.NOTES_HEADER format). mode: "raw" = MQL-0008 helper semantics."""
    t = tables_entry
    cols, tabs = t["column_names_original"], t["table_names_original"]
    notes = []
    for x, y in t["foreign_keys"]:
        a, b = f"{db_id}__{tabs[cols[x][0]]}", f"{db_id}__{tabs[cols[y][0]]}"
        ta = collections.get(a, {}).get("fields", {}).get(cols[x][1])
        tb = collections.get(b, {}).get("fields", {}).get(cols[y][1])
        if not ta or not tb:
            continue
        differ = set(ta) != set(tb) if mode == "raw" else (type_class(ta) and type_class(tb) and type_class(ta) != type_class(tb))
        if differ:
            notes.append(f"- {a}.{cols[x][1]} ({'/'.join(ta)}) -> {b}.{cols[y][1]} ({'/'.join(tb)})")
    return notes


def build_block(db_id, examples, notes):
    return RB.render_block(db_id, [], examples, notes)


def budgeted_block(db_id, system, base_user, ordered_docs, k, count_tokens, max_ctx, n_ctx, max_new, margin, notes=()):
    """The MQL-0008 budget loop (scripts/rag/rag_prompt.build_messages with k_guides=0) generalised to notes:
    notes have priority and are kept only if they fit on their own; examples are added in the given order and
    over-budget documents are skipped (recorded), never truncated. With notes=() this is exactly
    mql0008_core.select_examples. -> (chosen docs, dropped ids, block, block_tokens, total_tokens, notes_kept)."""
    base_tokens = count_tokens(system) + count_tokens(base_user)
    notes = list(notes)

    def fits(ex, nt):
        block = build_block(db_id, ex, nt)
        bt = count_tokens(block) if block else 0
        total = base_tokens + (bt + 2 if block else 0)
        return bt <= max_ctx and total + max_new + margin <= n_ctx, bt, total

    dropped = []
    if notes and not fits([], notes)[0]:
        dropped.append("schema_notes")
        notes = []
    chosen = []
    for d in ordered_docs:
        if len(chosen) >= k:
            break
        ok, _, _ = fits(chosen + [d], notes)
        (chosen.append(d) if ok else dropped.append(d["doc_id"]))
    _, bt, total = fits(chosen, notes)
    return chosen, dropped, build_block(db_id, chosen, notes), bt, total, notes


RETRY_SUFFIX = "Fix these problems and return the corrected JSON object only."


def recovery_messages(first_user_text, first_output, feedback):
    """Chat turns for the single controlled retry: the original prompt, the model's own first answer, then the
    validator feedback. No gold information, no execution result."""
    return [{"role": "user", "content": first_user_text},
            {"role": "assistant", "content": first_output},
            {"role": "user", "content": feedback + "\n\n" + RETRY_SUFFIX}]
