"""MQL-0011 retrieval-store interface, shared types and the exact NumPy reference backend.

Contract every backend must honour (tests/test_mql0011_store_contract.py runs the same suite on each one):
  * `search(vec, n, flt)` returns hits for points NOT excluded by `flt`, best first; `n=None` means "every eligible
    point" and is resolved to the actual eligible count (never an arbitrary large limit).
  * Hit scores are cosine similarities of L2-normalised float32 vectors.
  * `spec()` returns the IndexSpec the store was built from; `ids()` the stored doc ids in artifact row order.
Backends never decide ties: callers re-rank by (-round(score, 6), doc_id).
"""
import uuid
from dataclasses import asdict, dataclass, field
from typing import Optional, Protocol

import numpy as np

POINT_NAMESPACE = uuid.UUID("6f1c3a52-6b0e-5d1e-9a51-0b7d2c6e0011")   # fixed: MQL-0011 point ids


class RetrievalStoreError(RuntimeError):
    """Any store / artifact / embedder inconsistency. Research code never falls back on it."""


class EligibilityViolation(RetrievalStoreError):
    """A store returned a point that the eligibility rules exclude (filter bug): always fatal."""


@dataclass(frozen=True)
class IndexSpec:
    index_name: str
    corpus_sha256: str
    view_ids_sha256: str
    n_records: int
    model_id: str
    model_revision: str
    pooling: str
    dim: int
    normalized: bool
    dtype: str
    text_field: str
    mask_source: str
    patterns_sha256: str
    vectors_sha256: str

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        return cls(**{k: d[k] for k in cls.__dataclass_fields__})


@dataclass(frozen=True)
class Hit:
    doc_id: str
    db_id: str
    score: float


@dataclass(frozen=True)
class EligibilityFilter:
    exclude_db_ids: frozenset = field(default_factory=frozenset)
    exclude_doc_ids: frozenset = field(default_factory=frozenset)

    def allows(self, doc_id, db_id):
        return db_id not in self.exclude_db_ids and doc_id not in self.exclude_doc_ids


class VectorStore(Protocol):
    def spec(self) -> IndexSpec: ...
    def ids(self) -> list: ...
    def count(self, flt: Optional[EligibilityFilter] = None) -> int: ...
    def search(self, vec, n: Optional[int], flt: EligibilityFilter) -> list: ...
    def close(self) -> None: ...


def point_uuid(doc_id):
    return str(uuid.uuid5(POINT_NAMESPACE, doc_id))


def resolve_n(n, eligible):
    if n is None:
        return eligible
    if n <= 0:
        raise ValueError("n must be positive or None")
    return min(n, eligible)


def check_vectors(vectors, dim):
    v = np.asarray(vectors)
    if v.dtype != np.float32 or v.ndim != 2 or v.shape[1] != dim:
        raise RetrievalStoreError(f"vectors must be float32 [N, {dim}], got {v.dtype} {v.shape}")
    if not np.all(np.isfinite(v)):
        raise RetrievalStoreError("non-finite vector values")
    norms = np.linalg.norm(v, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-5):
        raise RetrievalStoreError(f"vectors are not L2-normalised (norm range {norms.min():.6f}..{norms.max():.6f})")
    return v


def check_hits(hits, flt, known):
    """Defence in depth: every hit must be a known point and allowed by the filter; otherwise raise."""
    for h in hits:
        if h.doc_id not in known:
            raise EligibilityViolation(f"store returned unknown doc {h.doc_id}")
        if known[h.doc_id] != h.db_id:
            raise EligibilityViolation(f"store db_id mismatch for {h.doc_id}")
        if not flt.allows(h.doc_id, h.db_id):
            raise EligibilityViolation(f"store returned ineligible doc {h.doc_id} ({h.db_id})")
    if len({h.doc_id for h in hits}) != len(hits):
        raise EligibilityViolation("store returned duplicate points")
    return hits


class CanonicalScorer:
    """Backend-independent dense scores: float32 dot products of the ARTIFACT vectors, always computed as one
    full-matrix product (V @ q) so the arithmetic never depends on which backend found the candidates.
    Why: backends return cosines that differ in the last float32 bits (Qdrant re-normalises on insert; measured max
    |diff| 3.1e-7 on the 779 dev queries), which flips 6-decimal rounding/tie decisions in hundreds of rankings.
    The store retrieves and filters candidates; this scorer gives the scores used for ranking (re-scoring)."""

    def __init__(self, doc_ids, vectors, dim):
        self.V = check_vectors(vectors, dim)
        self.row = {d: i for i, d in enumerate(doc_ids)}
        if len(self.row) != len(doc_ids) or self.V.shape[0] != len(doc_ids):
            raise RetrievalStoreError("scorer ids / vectors mismatch")

    def ids(self):
        return list(self.row)

    def scores(self, q, doc_ids):
        full = self.V @ np.asarray(q, dtype=np.float32)
        try:
            return [float(full[self.row[d]]) for d in doc_ids]
        except KeyError as e:
            raise RetrievalStoreError(f"doc {e} has no canonical vector") from e


class NumpyExactStore:
    """Exact brute-force cosine over the artifact matrix (reference oracle; also the store test double)."""

    def __init__(self, spec, doc_ids, db_ids, vectors):
        self._spec = spec
        self._ids = list(doc_ids)
        self._db = list(db_ids)
        self._v = check_vectors(vectors, spec.dim)
        if len(self._ids) != spec.n_records or len(self._db) != len(self._ids) or self._v.shape[0] != len(self._ids):
            raise RetrievalStoreError("artifact size mismatch")
        if len(set(self._ids)) != len(self._ids):
            raise RetrievalStoreError("duplicate doc ids")
        self.known = dict(zip(self._ids, self._db))

    def spec(self):
        return self._spec

    def ids(self):
        return list(self._ids)

    def _mask(self, flt):
        if flt is None:
            return np.ones(len(self._ids), dtype=bool)
        return np.array([flt.allows(i, d) for i, d in zip(self._ids, self._db)], dtype=bool)

    def count(self, flt=None):
        return int(self._mask(flt).sum())

    def search(self, vec, n, flt):
        q = np.asarray(vec, dtype=np.float32)
        if q.shape != (self._spec.dim,):
            raise RetrievalStoreError(f"query vector shape {q.shape}")
        mask = self._mask(flt)
        idx = np.flatnonzero(mask)
        scores = self._v[idx] @ q
        order = sorted(range(len(idx)), key=lambda j: (-round(float(scores[j]), 6), self._ids[idx[j]]))
        n = resolve_n(n, len(idx))
        return [Hit(self._ids[idx[j]], self._db[idx[j]], float(scores[j])) for j in order[:n]]

    def close(self):
        pass
