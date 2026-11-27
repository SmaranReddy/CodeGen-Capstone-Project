"""MQL-0010 component R: structure-aware example retrieval (wraps the unchanged MQL-0008 retriever).

The baseline `retrieval.Retriever(method="hybrid_masked", hybrid_weight=1.0)` is used as-is for the lexical index
(BM25 over masked questions, leave-target-database-out), the duplicate collapse and the target-db filter.
Only the score is changed, and only when `structure_scoring="nb"`:

    baseline  score(d) = bm25(d)/max_bm25 + 1.0 * Jaccard(regex_cues(q), patterns(d))        ties -> doc_id
    nb        score(d) = P(sig(q) == sig(d)) + beta * bm25(d)/max_bm25                        ties -> raw bm25, doc_id

P(sig(q) == sig(d)) is the structure predictor's probability that the query has exactly the document's structure
signature (component S). It is continuous, so the 2.0/2.0/2.0 plateaus of the baseline disappear; documents
sharing one signature share P and are ordered by lexical similarity.

Flags (all deterministic):
    structure_scoring  "baseline" | "nb"
    beta               lexical weight for "nb"
    tie_break          "doc_id" (baseline) | "lexical"   (secondary key: raw BM25, then doc_id)
With structure_scoring="baseline" and tie_break="doc_id" the ranking is identical to the MQL-0008/0009 retriever
(tested against the registered MQL-0009 D plan, 639/639 items).
"""
import retrieval as RT
import patterns as PT
import structure as S


class StructureAwareRetriever:
    def __init__(self, docs, predictor=None, structure_scoring="baseline", beta=1.0, tie_break="doc_id",
                 hybrid_weight=1.0):
        if structure_scoring not in ("baseline", "nb"):
            raise ValueError(structure_scoring)
        if tie_break not in ("doc_id", "lexical"):
            raise ValueError(tie_break)
        if structure_scoring == "nb" and predictor is None:
            raise ValueError("structure_scoring='nb' needs a fitted StructurePredictor")
        self.base = RT.Retriever(docs, method="hybrid_masked", hybrid_weight=hybrid_weight)
        self.predictor, self.mode, self.beta, self.tie = predictor, structure_scoring, beta, tie_break
        self._sig = {d["doc_id"]: S.signature(d["patterns"]) for d in docs if d["kind"] == "example"}

    def query_structure(self, question, info):
        """-> (probabilities or None, predicted signature or None). Inference-time inputs only."""
        if self.predictor is None:
            return None, None
        p = self.predictor.predict_proba(S.features(question, info))
        return p, S.predicted_signature(p)

    def rank_examples(self, question, info, qid=""):
        """-> [(doc, score, trace)] best first; target-db docs removed; duplicates collapsed exactly as baseline."""
        if self.mode == "baseline" and self.tie == "doc_id":
            return [(d, s, {"score": s}) for d, s in self.base.rank_examples(question, info, qid)]
        docs, index = self.base._index_for(info["db_id"])
        q = PT.tokenize(self.base.query_text(question, info))
        raw = index.scores(q) if index else []
        top = max(raw) if raw and max(raw) > 0 else 1.0
        if self.mode == "baseline":
            cues = PT.question_cues(question, info)
            struct = [RT.jaccard(cues, d["patterns"]) for d in docs]
            sc = [r / top + self.base.w * s for r, s in zip(raw, struct)]
        else:
            proba, _ = self.query_structure(question, info)
            cache = {}
            struct = []
            for d in docs:
                sg = self._sig[d["doc_id"]]
                if sg not in cache:
                    cache[sg] = pow(2.718281828459045, S.signature_logprob(sg, proba))
                struct.append(cache[sg])
            sc = [s + self.beta * r / top for r, s in zip(raw, struct)]
        key = (lambda t: (-t[1], -t[2], t[0]["doc_id"])) if self.tie == "lexical" else (lambda t: (-t[1], t[0]["doc_id"]))
        ranked = sorted(zip(docs, [round(x, 9) for x in sc], raw, struct), key=key)
        out, seen_ids, seen_mql, seen_q = [], set(), set(), set()
        for d, s, r, st in ranked:
            if d.get("db_id") == info["db_id"]:
                continue
            km, kq = repr(d.get("mql")), RT._norm_q(d.get("question", ""))
            if d["doc_id"] in seen_ids or km in seen_mql or kq in seen_q:
                continue
            seen_ids.add(d["doc_id"])
            seen_mql.add(km)
            seen_q.add(kq)
            out.append((d, round(s, 6), {"score": round(s, 6), "lexical": round(r / top, 6), "structure": round(st, 6)}))
        return out


def select_diverse(ranked, k, key=lambda d: S.signature(d["patterns"])):
    """Component E (diversified selection): walk the ranking best-first and take the best document of each NEW
    structure signature until k are chosen; if fewer than k signatures exist, fill with the next-best remaining
    documents in rank order. Deterministic; preserves rank order among the chosen. -> list of ranked entries."""
    chosen, seen, rest = [], set(), []
    for ent in ranked:
        sg = key(ent[0])
        if sg in seen:
            rest.append(ent)
            continue
        seen.add(sg)
        chosen.append(ent)
        if len(chosen) == k:
            return chosen
    for ent in rest:
        if len(chosen) == k:
            break
        chosen.append(ent)
    order = {id(e): i for i, e in enumerate(ranked)}
    return sorted(chosen, key=lambda e: order[id(e)])
