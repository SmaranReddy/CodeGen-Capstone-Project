"""MQL-0012 X1: query-structure cues estimated from dense retrieval neighbours (frozen before any X0 retrieval).

Algorithm (all integer arithmetic, deterministic):
  1. Neighbour pool = the query's candidate universe U (registered view minus the target database, forbidden
     databases absent), i.e. exactly the documents the ranking itself may return.
  2. Order U by canonical dense cosine (artifact float32 dot, rounded to 6 decimals) descending, ties by doc_id
     ascending -- the same dense ranking R3 uses.
  3. Walk that order and apply the existing duplicate collapse (same doc_id, same repr(mql) or same normalised
     question as an earlier kept document is skipped), keeping the first N_NEIGHBOURS = 10 distinct documents.
  4. Votes: for every label l in VOTE_LABELS, v(l) = number of the 10 neighbours whose `patterns` contain l.
     Each neighbour contributes one vote per label it carries (unweighted; cosine is used only for ordering).
     Labels outside VOTE_LABELS are not voted: "select" and "where" (ignored by retrieval.jaccard), and
     "text_number" (a property of a database's schema, not of the question).
  5. Predicted structural labels = { l : 2 * v(l) > n } with n = number of neighbours (10): a STRICT majority,
     i.e. at least 6 of 10. A 5/5 split is NOT included (the only tie case). Confidence of l = v(l) / n (reported).
     If no label reaches a strict majority, the predicted structural set is empty (exactly like a question for
     which no regex cue fires).
  6. Schema rules of patterns.question_cues are kept unchanged and unioned in:
       "join"        if len(patterns.mentioned_collections(question, info)) >= 2
       "text_number" if info["text_number_fields"] is non-empty
     so "join" is present if EITHER the neighbour majority OR the schema-mention rule says so.
  7. Cue set returned = {"select"} | predicted | schema-rule labels -- the same representation question_cues returns;
     retrieval.jaccard (unchanged) then drops select/where and compares with each document's patterns.
Never reads the query's gold SQL/MQL/labels: inputs are the question text, the target schema info and the
neighbours' corpus metadata.
"""
import patterns as PT
import retrieval as RT

N_NEIGHBOURS = 10
VOTE_LABELS = ("count", "count_distinct", "distinct", "aggregate", "group_by", "having", "order_by", "limit",
               "join", "like", "anti_join", "intersect", "union", "nested")


def dense_order(universe, cosines):
    """Indices of `universe` by (-cosine rounded 6, doc_id)."""
    return sorted(range(len(universe)), key=lambda i: (-round(cosines[i], 6), universe[i]["doc_id"]))


def select_neighbours(universe, cosines, target_db, n=N_NEIGHBOURS):
    out, seen_ids, seen_mql, seen_q = [], set(), set(), set()
    for i in dense_order(universe, cosines):
        d = universe[i]
        if d["db_id"] == target_db:
            raise ValueError(f"target-database document {d['doc_id']} in the neighbour pool")
        km, kq = repr(d.get("mql")), RT._norm_q(d.get("question", ""))
        if d["doc_id"] in seen_ids or km in seen_mql or kq in seen_q:
            continue
        seen_ids.add(d["doc_id"])
        seen_mql.add(km)
        seen_q.add(kq)
        out.append(d)
        if len(out) == n:
            break
    return out


def vote(neighbours):
    """-> (predicted label set, {label: votes}, n)."""
    n = len(neighbours)
    votes = {l: sum(l in d["patterns"] for d in neighbours) for l in VOTE_LABELS}
    return {l for l, v in votes.items() if 2 * v > n}, votes, n


def schema_rule_labels(question, info):
    out = set()
    if info:
        if len(PT.mentioned_collections(question, info)) >= 2:
            out.add("join")
        if info.get("text_number_fields"):
            out.add("text_number")
    return out


def predicted_cues(question, info, universe, cosines):
    """-> (cue set in question_cues representation, trace)."""
    nb = select_neighbours(universe, cosines, info["db_id"])
    pred, votes, n = vote(nb)
    schema = schema_rule_labels(question, info)
    cues = {"select"} | pred | schema
    return cues, {"neighbours": [d["doc_id"] for d in nb], "votes": votes, "n": n,
                  "predicted": sorted(pred), "schema_rule": sorted(schema)}
