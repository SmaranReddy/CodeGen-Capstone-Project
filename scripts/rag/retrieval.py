"""Local, deterministic retrievers over the RAG corpus (no external services, no embeddings).

Example retrievers (rank `example` documents):
  bm25_raw / bm25_masked     Okapi BM25 (k1=1.5, b=0.75; Robertson & Zaragoza 2009) over the raw
                             or schema/literal-masked question (DAIL-SQL-style masking).
  tfidf_raw / tfidf_masked   cosine similarity of sublinear-tf, smoothed-idf vectors.
  hybrid_masked              BM25-masked score (normalised by the top score) + w * Jaccard between
                             the query's inference-time pattern cues and the document's patterns.
  random                     seeded random order (control: same corpus and budget, no relevance).
Guide retriever (rank `guide` documents): pattern overlap between the guide's patterns and the
query cues, plus half-weight overlap with the patterns of the retrieved examples.

Every retriever only sees the query question and the target database's schema; documents
from the target database are never candidates (the corpus holds train databases only, and the
filter is applied again here). Duplicate documents (same doc_id, same MQL, or same normalised
question) are collapsed; the higher-ranked copy wins.
"""
import hashlib
import math
import random
from collections import Counter

import patterns as PT

EXAMPLE_METHODS = ("bm25_raw", "bm25_masked", "tfidf_raw", "tfidf_masked", "hybrid_masked", "random")
_GENERIC = {"select", "where"}


class BM25:
    def __init__(self, docs_tokens, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.tf = [Counter(t) for t in docs_tokens]
        self.len = [len(t) for t in docs_tokens]
        self.avg = (sum(self.len) / len(self.len)) if self.len else 0.0
        df = Counter(w for t in docs_tokens for w in set(t))
        n = len(docs_tokens)
        self.idf = {w: math.log(1 + (n - c + 0.5) / (c + 0.5)) for w, c in df.items()}

    def scores(self, q_tokens):
        out = []
        for tf, dl in zip(self.tf, self.len):
            s = 0.0
            for w in q_tokens:
                f = tf.get(w)
                if f:
                    s += self.idf[w] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / (self.avg or 1)))
            out.append(s)
        return out


class TFIDF:
    def __init__(self, docs_tokens):
        n = len(docs_tokens)
        df = Counter(w for t in docs_tokens for w in set(t))
        self.idf = {w: math.log((1 + n) / (1 + c)) + 1 for w, c in df.items()}
        self.vecs = [self._vec(t) for t in docs_tokens]

    def _vec(self, toks):
        tf = Counter(toks)
        v = {w: (1 + math.log(c)) * self.idf.get(w, 0.0) for w, c in tf.items()}
        norm = math.sqrt(sum(x * x for x in v.values())) or 1.0
        return {w: x / norm for w, x in v.items()}

    def scores(self, q_tokens):
        q = self._vec(q_tokens)
        return [sum(x * d.get(w, 0.0) for w, x in q.items()) for d in self.vecs]


def jaccard(a, b):
    a, b = set(a) - _GENERIC, set(b) - _GENERIC
    return len(a & b) / len(a | b) if (a or b) else 0.0


def _norm_q(q):
    return " ".join(PT.tokenize(q))


class Retriever:
    def __init__(self, corpus, method="bm25_masked", hybrid_weight=0.5, seed=20261020):
        if method not in EXAMPLE_METHODS:
            raise ValueError(f"unknown retrieval method {method}")
        self.method, self.w, self.seed = method, hybrid_weight, seed
        self.examples = [d for d in corpus if d["kind"] == "example"]
        self.guides = [d for d in corpus if d["kind"] == "guide"]
        self._field = "question" if method.endswith("_raw") else "masked_question"
        self._indexes = {}

    def _index_for(self, target):
        """(candidate docs, index) over examples from databases OTHER than `target`, so IDF,
        average length and the hybrid normaliser are leave-one-database-out (for validation /
        test targets the corpus holds no target-db documents, so this is the full corpus)."""
        if target not in self._indexes:
            docs = [d for d in self.examples if d.get("db_id") != target]
            toks = [PT.tokenize(d[self._field]) for d in docs]
            idx = (TFIDF(toks) if self.method.startswith("tfidf") else BM25(toks)) if docs else None
            self._indexes[target] = (docs, idx)
        return self._indexes[target]

    def query_text(self, question, info):
        return question if self.method.endswith("_raw") else PT.mask_question(question, info)

    def rank_examples(self, question, info, qid=""):
        """-> list of (doc, score) best first, target-db docs removed, duplicates collapsed."""
        target = info["db_id"]
        if self.method == "random":
            h = int(hashlib.sha256(f"{self.seed}:{qid}:{question}".encode()).hexdigest()[:16], 16)
            order = list(range(len(self.examples)))
            random.Random(h).shuffle(order)
            ranked = [(self.examples[i], 0.0) for i in order]
        else:
            docs, index = self._index_for(target)
            q = PT.tokenize(self.query_text(question, info))
            sc = index.scores(q) if index else []
            if self.method == "hybrid_masked":
                top = max(sc) if sc and max(sc) > 0 else 1.0
                cues = PT.question_cues(question, info)
                sc = [s / top + self.w * jaccard(cues, d["patterns"]) for s, d in zip(sc, docs)]
            # deterministic: score desc, then doc_id
            ranked = sorted(zip(docs, sc), key=lambda t: (-t[1], t[0]["doc_id"]))
        out, seen_ids, seen_mql, seen_q = [], set(), set(), set()
        for d, s in ranked:
            if d.get("db_id") == target:
                continue
            key_m = repr(d.get("mql"))
            key_q = _norm_q(d.get("question", ""))
            if d["doc_id"] in seen_ids or key_m in seen_mql or key_q in seen_q:
                continue
            seen_ids.add(d["doc_id"])
            seen_mql.add(key_m)
            seen_q.add(key_q)
            out.append((d, round(s, 6)))
        return out

    def rank_guides(self, question, info, example_docs=()):
        cues = PT.question_cues(question, info) - _GENERIC
        ex_p = Counter(p for d in example_docs for p in d["patterns"] if p not in _GENERIC)
        ranked = []
        for i, g in enumerate(self.guides):
            if f'{info["db_id"]}__' in g["content"]:   # guide example uses a target-db collection
                continue
            gp = set(g["patterns"]) - _GENERIC
            s = len(gp & cues) + 0.5 * sum(1 for p in gp if ex_p.get(p))
            if s > 0:
                ranked.append((g, s, i))
        ranked.sort(key=lambda t: (-t[1], t[2]))
        return [(g, round(s, 6)) for g, s, _ in ranked]
