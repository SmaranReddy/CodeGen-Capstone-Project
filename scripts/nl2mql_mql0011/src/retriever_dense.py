"""MQL-0011 dense / fused example retrieval. Wraps the unchanged MQL-0008 retriever; never modifies it.

Candidate universe and order (identical to the baseline `retrieval.Retriever.rank_examples` + `planner.build_plan`):
  0. documents from forbidden (holdout) databases abort the plan before any index is built      (planner_dense)
  1. U = view examples whose db_id != target                       (base._index_for: BM25 IDF/avg-len over U)
  2. per-channel scores over U:  BM25 raw (unchanged index)  |  dense: the VectorStore retrieves the eligible
     candidates (filter excludes {target} + forbidden, n = |U|; the returned id set must EQUAL U, else
     RetrievalStoreError); their cosines are then RE-SCORED by the CanonicalScorer (artifact float32 vectors) when one
     is given (research: required), so rankings are identical whichever backend retrieved the candidates
  3. channel ranks over U (competition ranking on 6-decimal scores), lexical term L normalised by its max over U
  4. score = L + hybrid_weight * Jaccard(question cues, doc patterns)      (dense_only: score = cosine, no Jaccard)
  5. sort by (-score, doc_id); duplicate collapse exactly as the baseline (doc_id, repr(mql), normalised question)
  6. registered pair exclusions, diversity and the token budget run afterwards, unchanged          (planner_dense)

Lexical modes
  bm25        R0: delegates to retriever_v2.StructureAwareRetriever (baseline path) -> byte-identical to MQL-0010 D0
  rrf_bm25    R1: L = RRF_k(bm25 rank) / max          (matched control: the RRF rescaling without any dense signal)
  dense_only  R2: score = cosine                       (dense retrieval alone)
  rrf_fused   R3: L = [RRF_k(bm25 rank) + RRF_k(dense rank)] / max
RRF_k(r) = 1 / (k + r), k = 60 (literature constant, not tuned).
"""
import patterns as PT
import retrieval as RT
import retriever_v2 as R2
from vector_store import EligibilityFilter, RetrievalStoreError

LEXICAL_MODES = ("bm25", "rrf_bm25", "dense_only", "rrf_fused")
DENSE_MODES = ("dense_only", "rrf_fused")


def competition_ranks(scores, ndigits=6):
    """Rank 1 = best; equal (rounded) scores share the best rank of their group."""
    r = [round(s, ndigits) for s in scores]
    order = sorted(range(len(r)), key=lambda i: -r[i])
    ranks = [0] * len(r)
    for pos, i in enumerate(order):
        ranks[i] = ranks[order[pos - 1]] if pos and r[i] == r[order[pos - 1]] else pos + 1
    return ranks


class DenseHybridRetriever:
    def __init__(self, docs, lexical, store=None, embedder=None, forbidden_db_ids=(), hybrid_weight=1.0, rrf_k=60,
                 scorer=None):
        if lexical not in LEXICAL_MODES:
            raise ValueError(f"unknown lexical mode {lexical}")
        self.lexical, self.k_rrf = lexical, rrf_k
        self.forbidden = frozenset(forbidden_db_ids)
        bad = sorted({d.get("db_id") for d in docs if d.get("db_id") in self.forbidden})
        if bad:
            raise RetrievalStoreError(f"retrieval documents from forbidden databases: {bad}")
        self.inner = R2.StructureAwareRetriever(docs, hybrid_weight=hybrid_weight)     # baseline path (R0)
        self.base = self.inner.base
        self.store, self.embedder = store, embedder
        if lexical in DENSE_MODES:
            if store is None or embedder is None:
                raise RetrievalStoreError(f"lexical={lexical} needs a VectorStore and an Embedder")
            ex_ids = sorted(d["doc_id"] for d in self.base.examples)
            if sorted(store.ids()) != ex_ids:
                raise RetrievalStoreError("vector store ids differ from the retrieval documents")
            if embedder.spec.model_revision != store.spec().model_revision or embedder.spec.dim != store.spec().dim:
                raise RetrievalStoreError("embedder spec differs from the index spec")
            if scorer is not None and sorted(scorer.ids()) != ex_ids:
                raise RetrievalStoreError("canonical scorer ids differ from the retrieval documents")
        self.scorer = scorer

    def dense_scores(self, masked_q, target, universe):
        flt = EligibilityFilter(exclude_db_ids=frozenset({target}) | self.forbidden)
        vec = self.embedder.encode([masked_q])[0]
        hits = self.store.search(vec, None, flt)
        got = {h.doc_id: h.score for h in hits}
        if set(got) != {d["doc_id"] for d in universe} or len(hits) != len(universe):
            raise RetrievalStoreError(f"dense candidate set differs from the BM25 universe for target {target}")
        ids = [d["doc_id"] for d in universe]
        raw = self.scorer.scores(vec, ids) if self.scorer is not None else [got[i] for i in ids]
        return [round(s, 6) for s in raw]

    def rank_examples(self, question, info, qid=""):
        """-> [(doc, score, trace)] best first; same shape as StructureAwareRetriever.rank_examples."""
        if self.lexical == "bm25":
            return self.inner.rank_examples(question, info, qid)
        target = info["db_id"]
        if target in self.forbidden:
            raise RetrievalStoreError(f"target {target} is a forbidden database")
        docs, index = self.base._index_for(target)                         # universe U (target excluded)
        masked = self.base.query_text(question, info)
        raw = index.scores(PT.tokenize(masked)) if index else []
        cues = PT.question_cues(question, info)
        jac = [RT.jaccard(cues, d["patterns"]) for d in docs]
        r_b = competition_ranks(raw)
        cos = self.dense_scores(masked, target, docs) if self.lexical in DENSE_MODES else [None] * len(docs)
        r_d = competition_ranks(cos) if self.lexical in DENSE_MODES else [None] * len(docs)
        if self.lexical == "dense_only":
            lex = list(cos)
            sc = list(cos)
        else:
            k = self.k_rrf
            l_raw = [1.0 / (k + b) + (1.0 / (k + d) if self.lexical == "rrf_fused" else 0.0)
                     for b, d in zip(r_b, r_d)]
            top = max(l_raw) if l_raw else 1.0
            lex = [x / top for x in l_raw]
            sc = [x + self.base.w * j for x, j in zip(lex, jac)]
        rows = sorted(zip(docs, sc, lex, jac, r_b, r_d, cos), key=lambda t: (-round(t[1], 9), t[0]["doc_id"]))
        out, seen_ids, seen_mql, seen_q = [], set(), set(), set()
        for d, s, lx, j, rb, rd, c in rows:
            if d.get("db_id") == target or d.get("db_id") in self.forbidden:
                raise RetrievalStoreError(f"ineligible document {d['doc_id']} reached ranking")
            km, kq = repr(d.get("mql")), RT._norm_q(d.get("question", ""))
            if d["doc_id"] in seen_ids or km in seen_mql or kq in seen_q:
                continue
            seen_ids.add(d["doc_id"])
            seen_mql.add(km)
            seen_q.add(kq)
            out.append((d, round(s, 6), {"score": round(s, 6), "lexical": round(lx, 6), "structure": round(j, 6),
                                         "bm25_rank": rb, "dense_rank": rd, "cosine": c}))
        return out
