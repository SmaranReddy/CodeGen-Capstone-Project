"""Context construction and prompt assembly for RAG (separate from retrieval and generation).

The RAG prompt is EXACTLY the zero-shot prompt (prompt_spider1.messages + prompt_spider1.SYSTEM,
version nl-mql-spider1-v1) with one reference block inserted immediately before the final
"Question:" line. With nothing selected (k = 0, empty retrieval, or nothing fits the budget) the
prompt is byte-identical to zero-shot.

Budget: documents are added in priority order (examples by rank, then guides by rank); a
document that would push the block over `max_context_tokens`, or the full prompt plus
`max_new_tokens` + `margin` over `n_ctx`, is skipped and recorded in `dropped_budget`.
"""
import hashlib

import prompt_spider1 as P

RAG_PROMPT_VERSION = "nl-mql-spider1-rag-v1"
HEADER = ("Reference material (retrieved automatically; it may be incomplete or irrelevant).\n"
          "The examples below come from OTHER databases with different schemas: reuse their query "
          "patterns only, never their collection or field names. Use only the collections listed "
          "above for database {db}.")
GUIDE_HEADER = "MongoDB notes:"
EXAMPLES_HEADER = "Solved examples from other databases:"
SPLIT_MARK = "\n\nQuestion: "


NOTES_HEADER = ("Join keys with DIFFERENT types in database {db} (an equality $lookup on them matches "
                "nothing; use the let/pipeline form of $lookup and convert one side with $toString or "
                "$toInt inside $expr):")


def fk_type_notes(db_id, collections, tables_json):
    """Schema-derived (inference-time) notes: foreign keys whose observed BSON types differ."""
    t = {x["db_id"]: x for x in tables_json}[db_id]
    cols, tabs = t["column_names_original"], t["table_names_original"]
    notes = []
    for x, y in t["foreign_keys"]:
        a, b = f"{db_id}__{tabs[cols[x][0]]}", f"{db_id}__{tabs[cols[y][0]]}"
        ta = collections.get(a, {}).get("fields", {}).get(cols[x][1])
        tb = collections.get(b, {}).get("fields", {}).get(cols[y][1])
        if ta and tb and set(ta) != set(tb):
            notes.append(f"- {a}.{cols[x][1]} ({'/'.join(ta)}) -> {b}.{cols[y][1]} ({'/'.join(tb)})")
    return notes


def template_hash():
    return hashlib.sha256((RAG_PROMPT_VERSION + P.template_hash() + HEADER + GUIDE_HEADER
                           + EXAMPLES_HEADER).encode("utf-8")).hexdigest()


def format_example(doc, i):
    return f"Example {i} (database {doc['db_id']}):\n{doc['content']}"


def format_guide(doc):
    return f"- {doc['content']}"


def render_block(db_id, guides, examples, notes=()):
    if not guides and not examples and not notes:
        return ""
    parts = [HEADER.format(db=db_id)]
    if notes:
        parts.append(NOTES_HEADER.format(db=db_id) + "\n" + "\n".join(notes))
    if guides:
        parts.append(GUIDE_HEADER + "\n" + "\n".join(format_guide(g) for g in guides))
    if examples:
        parts.append(EXAMPLES_HEADER + "\n" + "\n\n".join(format_example(d, i + 1)
                                                          for i, d in enumerate(examples)))
    return "\n\n".join(parts)


def insert_block(user_text, block):
    if not block:
        return user_text
    head, mark, tail = user_text.rpartition(SPLIT_MARK)
    if not mark:
        raise ValueError("zero-shot template changed: 'Question:' marker not found")
    return head + "\n\n" + block + mark + tail


def build_messages(db_id, question, ranked_examples, ranked_guides, k, k_guides,
                   count_tokens, max_context_tokens, n_ctx, max_new_tokens, margin=64, notes=()):
    """-> (messages, info). ranked_* are [(doc, score)] best first. `notes` (schema-derived
    lines, highest priority) are kept only if they fit the budget on their own."""
    base = P.messages(db_id, question)[0]["content"]
    base_tokens = count_tokens(P.SYSTEM) + count_tokens(base)
    chosen_ex, chosen_g, dropped = [], [], []

    notes = list(notes)

    def fits(ex, gd):
        block = render_block(db_id, gd, ex, notes)
        bt = count_tokens(block) if block else 0
        total = base_tokens + (bt + 2 if block else 0)
        return bt <= max_context_tokens and total + max_new_tokens + margin <= n_ctx, bt, total

    if notes and not fits([], [])[0]:
        dropped.append("schema_notes")
        notes = []
    for d, _ in ranked_examples:
        if len(chosen_ex) >= k:
            break
        ok, _, _ = fits(chosen_ex + [d], chosen_g)
        (chosen_ex.append(d) if ok else dropped.append(d["doc_id"]))
    for g, _ in ranked_guides:
        if len(chosen_g) >= k_guides:
            break
        ok, _, _ = fits(chosen_ex, chosen_g + [g])
        (chosen_g.append(g) if ok else dropped.append(g["doc_id"]))
    _, block_tokens, total = fits(chosen_ex, chosen_g)
    block = render_block(db_id, chosen_g, chosen_ex, notes)
    user = insert_block(base, block)
    info = {"selected_examples": [d["doc_id"] for d in chosen_ex],
            "selected_guides": [g["doc_id"] for g in chosen_g],
            "schema_notes": notes, "dropped_budget": dropped, "context_tokens": block_tokens,
            "prompt_tokens_est": total, "context_chars": len(block),
            "empty_context": not block, "context": block}
    return [{"role": "user", "content": user}], info
