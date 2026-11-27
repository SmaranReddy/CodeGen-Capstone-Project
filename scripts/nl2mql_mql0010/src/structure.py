"""MQL-0010 component S: inference-time query-structure predictor (replaces the 12 regex cues).

Inputs at inference time: the question and the target database schema (never gold SQL / MQL of the query).
Training data: Spider-train questions (spider-sql-v1 train.jsonl minus MQL-0010 holdout DBs and their near
duplicates; see preflight/make_split.py) with structure labels derived from THEIR gold SQL by the unchanged
`patterns.sql_patterns`. Pure Python, deterministic, no dependency.

Model: one binary multinomial naive Bayes per structure label (Laplace alpha), over
  * unigrams + bigrams of the schema/literal-masked question (patterns.mask_question),
  * the regex cue indicators of patterns.question_cues (as tokens "cue:<label>"),
  * the number of target collections the question mentions ("ncoll:0/1/2+").
Output: P(label | question) for every label in STRUCT_LABELS, and helpers to score a document's structure
signature under those probabilities (independent-label likelihood).
"""
import json
import math
from collections import Counter

import patterns as PT

STRUCT_LABELS = ("where", "count", "count_distinct", "distinct", "aggregate", "group_by", "having", "order_by",
                 "limit", "join", "nested", "anti_join", "intersect", "union", "like")
UNSUPPORTED = frozenset({"anti_join", "nested", "intersect", "union"})
VERSION = "mql0010-structure-nb-v1"


def signature(labels):
    """Structure signature of a label set (labels outside STRUCT_LABELS, e.g. select/text_number, ignored)."""
    return frozenset(l for l in labels if l in STRUCT_LABELS)


def features(question, info):
    m = PT.tokenize(PT.mask_question(question, info))
    f = list(m) + [f"{a}_{b}" for a, b in zip(m, m[1:])]
    f += [f"cue:{c}" for c in sorted(PT.question_cues(question, info) - {"select", "text_number"})]
    n = len(PT.mentioned_collections(question, info)) if info else 0
    f.append(f"ncoll:{min(n, 2)}")
    return f


class StructurePredictor:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.model = {}
        self.vocab = set()

    # ------------------------------------------------------------------ training
    def fit(self, examples):
        """examples: iterable of (feature_list, label_set). Deterministic."""
        examples = list(examples)
        self.vocab = {w for f, _ in examples for w in f}
        for lab in STRUCT_LABELS:
            cnt = {True: Counter(), False: Counter()}
            docs = {True: 0, False: 0}
            for f, labels in examples:
                y = lab in labels
                docs[y] += 1
                cnt[y].update(f)
            tot = {y: sum(cnt[y].values()) for y in (True, False)}
            self.model[lab] = {"prior": {y: math.log((docs[y] + 1) / (len(examples) + 2)) for y in (True, False)},
                               "cnt": cnt, "tot": tot}
        return self

    def _loglik(self, lab, f, y):
        m = self.model[lab]
        V = len(self.vocab) or 1
        s = m["prior"][y]
        for w in f:
            if w in self.vocab:
                s += math.log((m["cnt"][y][w] + self.alpha) / (m["tot"][y] + self.alpha * V))
        return s

    def predict_proba(self, f):
        out = {}
        for lab in STRUCT_LABELS:
            a, b = self._loglik(lab, f, True), self._loglik(lab, f, False)
            mx = max(a, b)
            out[lab] = math.exp(a - mx) / (math.exp(a - mx) + math.exp(b - mx))
        return out

    # ------------------------------------------------------------------ persistence (exact, for provenance)
    def to_json(self):
        return json.dumps({"version": VERSION, "alpha": self.alpha, "vocab": sorted(self.vocab),
                           "model": {lab: {"prior": {str(k): v for k, v in m["prior"].items()},
                                           "cnt": {str(k): dict(sorted(c.items())) for k, c in m["cnt"].items()},
                                           "tot": {str(k): v for k, v in m["tot"].items()}}
                                     for lab, m in self.model.items()}}, sort_keys=True)

    @classmethod
    def from_json(cls, s):
        d = json.loads(s)
        p = cls(d["alpha"])
        p.vocab = set(d["vocab"])
        b = {"True": True, "False": False}
        p.model = {lab: {"prior": {b[k]: v for k, v in m["prior"].items()},
                         "cnt": {b[k]: Counter(c) for k, c in m["cnt"].items()},
                         "tot": {b[k]: v for k, v in m["tot"].items()}} for lab, m in d["model"].items()}
        return p


def predicted_signature(proba, threshold=0.5):
    return frozenset(l for l, p in proba.items() if p >= threshold)


def signature_logprob(sig, proba, eps=1e-6):
    """log P(query signature == sig) under independent labels."""
    s = 0.0
    for lab in STRUCT_LABELS:
        p = min(max(proba[lab], eps), 1 - eps)
        s += math.log(p if lab in sig else 1 - p)
    return s


def unsupported_probability(proba):
    """P(the query needs at least one construct the approved corpus cannot show), independent labels."""
    q = 1.0
    for lab in UNSUPPORTED:
        q *= 1 - proba[lab]
    return 1 - q
