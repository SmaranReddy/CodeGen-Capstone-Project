"""MQL-0012 retriever: one class, two independent switches (channels, cue source). MQL-0011 code is reused unchanged.

    score(d) = L(d) + 1.0 * jaccard(cues(q), patterns(d))          (retrieval.jaccard, unchanged)
    L(d)     = [ sum over channels c of 1 / (60 + rank_c(d)) ] / max over U of the same sum
    rank_c   = competition rank (6-decimal scores) over the candidate universe U of channel c:
               bm25  = unchanged retrieval.BM25 over masked questions (leave-target-DB-out index)
               dense = canonical cosine (artifact float32 vectors) of the candidates the VectorStore returns
    sort (-round(score, 9), doc_id); duplicate collapse identical to the baseline.

Arms:
    R3 control : run with the UNCHANGED MQL-0011 class retriever_dense.DenseHybridRetriever("rrf_fused").
                 FlexRetriever(channels=("bm25","dense"), cues="regex") is a re-implementation proven identical to
                 it (tests + validation on all 779 MQL-0011 dev questions), so X1/X2 differ from R3 ONLY by their
                 one switch.
    X1         : channels=("bm25","dense"), cues="knn_vote"   (mql0012_vote.predicted_cues)
    X2         : channels=("dense",),       cues="regex"      (patterns.question_cues)
                 X2 never constructs or calls BM25: U is built directly as [view examples with db_id != target],
                 the same list, in the same order, that retrieval.Retriever._index_for returns.
"""
import patterns as PT
import retrieval as RT
import retriever_dense as RD
from vector_store import RetrievalStoreError

import mql0012_vote as V

CUE_MODES = ("regex", "knn_vote")
CHANNEL_SETS = (("bm25", "dense"), ("dense",))


class FlexRetriever:
    def __init__(self, docs, channels, cues, store, embedder, scorer, forbidden_db_ids=(), rrf_k=60,
                 hybrid_weight=1.0):
        channels = tuple(channels)
        if channels not in CHANNEL_SETS or cues not in CUE_MODES:
            raise ValueError(f"unsupported configuration channels={channels} cues={cues}")
        if scorer is None:
            raise RetrievalStoreError("MQL-0012 requires the canonical scorer (backend-independent ranking)")
        # the MQL-0011 retriever supplies the eligibility checks, store/embedder/scorer validation and dense_scores()
        self.inner = RD.DenseHybridRetriever(docs, "rrf_fused", store=store, embedder=embedder,
                                             forbidden_db_ids=forbidden_db_ids, hybrid_weight=hybrid_weight,
                                             rrf_k=rrf_k, scorer=scorer)
        self.base = self.inner.base
        self.channels, self.cue_mode, self.k_rrf, self.w = channels, cues, rrf_k, hybrid_weight
        self.forbidden = self.inner.forbidden

    def universe(self, target):
        return [d for d in self.base.examples if d.get("db_id") != target]

    def rank_examples(self, question, info, qid=""):
        target = info["db_id"]
        if target in self.forbidden:
            raise RetrievalStoreError(f"target {target} is a forbidden database")
        masked = self.base.query_text(question, info)          # patterns.mask_question; no BM25 involved
        if "bm25" in self.channels:
            docs, index = self.base._index_for(target)
            if [d["doc_id"] for d in docs] != [d["doc_id"] for d in self.universe(target)]:
                raise RetrievalStoreError("BM25 universe differs from the candidate universe")
            raw = index.scores(PT.tokenize(masked)) if index else []
            r_b = RD.competition_ranks(raw)
        else:
            docs, r_b = self.universe(target), None
        cos = self.inner.dense_scores(masked, target, docs)      # store retrieval + set check + canonical scores
        r_d = RD.competition_ranks(cos)
        vote_trace = None
        if self.cue_mode == "regex":
            cues = PT.question_cues(question, info)
        else:
            cues, vote_trace = V.predicted_cues(question, info, docs, cos)
        k = self.k_rrf
        l_raw = [(1.0 / (k + r_b[i]) if r_b is not None else 0.0) + 1.0 / (k + r_d[i]) for i in range(len(docs))]
        top = max(l_raw) if l_raw else 1.0
        lex = [x / top for x in l_raw]
        jac = [RT.jaccard(cues, d["patterns"]) for d in docs]
        sc = [x + self.w * j for x, j in zip(lex, jac)]
        rows = sorted(zip(docs, sc, lex, jac, r_b or [None] * len(docs), r_d, cos),
                      key=lambda t: (-round(t[1], 9), t[0]["doc_id"]))
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
        self.last_cues, self.last_vote = sorted(cues), vote_trace
        return out
