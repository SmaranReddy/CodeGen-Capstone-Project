"""Reference-block rendering for MQL-0007. Constants and functions are COPIED VERBATIM from
scripts/rag/rag_prompt.py (prompt version nl-mql-spider1-rag-v1); only the dependency on the Spider
prompt builder (prompt_spider1) is removed, because the frozen zero-shot prompts are shipped instead.
tests/test_mql0007_package.py proves equality with the repo original (render_block, insert_block and the
budget loop of build_messages) when the repo is present.
"""
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
