"""MQL-0011 Qdrant backend for the VectorStore interface.

Modes
  * embedded local (research): `QdrantStore.open(path=...)`. qdrant-client 1.20.0 local mode searches by brute-force
    NumPy (local_collection.search -> calculate_distance -> argsort), i.e. it is already EXACT. A persistent local path
    is guarded by an exclusive, non-blocking file lock: only ONE process / client may open it at a time (a second
    client raises "already accessed by another instance"). Use one client per process and close it (context manager).
    Local-mode latency says nothing about production-scale performance.
  * server / Cloud (future): `QdrantStore.open(url=..., api_key=...)`. Search is HNSW unless `exact=True`; research
    configurations must set exact=True, production sets exact=False plus an explicit `hnsw_ef` after a recall check.

The IndexSpec is stored as collection metadata and re-verified on open together with the point count and the hash of
the stored doc ids in artifact row order. Any mismatch raises RetrievalStoreError (no silent fallback).
"""
from qdrant_client import QdrantClient, models

from mql0011_common import ids_sha256
from vector_store import (EligibilityFilter, Hit, IndexSpec, RetrievalStoreError, check_hits, check_vectors,
                          point_uuid, resolve_n)

_SPEC_KEY = "mql0011_index_spec"


def _filter(flt):
    if flt is None:
        return None
    must_not = []
    if flt.exclude_db_ids:
        must_not.append(models.FieldCondition(key="db_id", match=models.MatchAny(any=sorted(flt.exclude_db_ids))))
    if flt.exclude_doc_ids:
        must_not.append(models.HasIdCondition(has_id=[point_uuid(d) for d in sorted(flt.exclude_doc_ids)]))
    return models.Filter(must_not=must_not) if must_not else None


class QdrantStore:
    def __init__(self, client, collection, exact=True, hnsw_ef=None):
        self.client, self.collection, self.exact, self.hnsw_ef = client, collection, exact, hnsw_ef
        self._spec = None
        self._ids = None
        self.known = None

    # ------------------------------------------------------------------ lifecycle
    @classmethod
    def connect(cls, path=None, url=None, api_key=None):
        if (path is None) == (url is None):
            raise ValueError("give exactly one of path (embedded) or url (server)")
        try:
            return QdrantClient(path=str(path)) if path is not None else QdrantClient(url=url, api_key=api_key)
        except RuntimeError as e:   # local-mode lock held by another client
            raise RetrievalStoreError(str(e)) from e

    @classmethod
    def open(cls, expected_spec, path=None, url=None, api_key=None, exact=True, hnsw_ef=None):
        """Open an existing collection and verify it against the artifact spec (fail closed)."""
        client = cls.connect(path=path, url=url, api_key=api_key)
        store = cls(client, expected_spec.index_name, exact=exact, hnsw_ef=hnsw_ef)
        try:
            store._verify(expected_spec)
        except Exception:
            client.close()
            raise
        return store

    def close(self):
        self.client.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # ------------------------------------------------------------------ ingestion
    @classmethod
    def create(cls, client, spec, doc_ids, db_ids, text_shas, vectors, batch=256):
        """Create the collection for `spec` and upsert every record (idempotent by deterministic UUIDv5 ids).
        Refuses to overwrite an existing collection whose stored spec differs."""
        v = check_vectors(vectors, spec.dim)
        if not (len(doc_ids) == len(db_ids) == len(text_shas) == v.shape[0] == spec.n_records):
            raise RetrievalStoreError("record count differs from the spec")
        if ids_sha256(doc_ids) != spec.view_ids_sha256:
            raise RetrievalStoreError("doc ids differ from the spec's view ids hash")
        name = spec.index_name
        if client.collection_exists(name):
            stored = (client.get_collection(name).config.metadata or {}).get(_SPEC_KEY)
            if stored != spec.to_dict():
                raise RetrievalStoreError(f"collection {name} exists with a different spec; refusing to overwrite")
        else:
            client.create_collection(name, vectors_config=models.VectorParams(size=spec.dim,
                                                                              distance=models.Distance.COSINE),
                                     metadata={_SPEC_KEY: spec.to_dict()})
            client.create_payload_index(name, field_name="db_id", field_schema=models.PayloadSchemaType.KEYWORD)
        for s in range(0, len(doc_ids), batch):
            client.upsert(name, points=[
                models.PointStruct(id=point_uuid(doc_ids[i]), vector=v[i].tolist(),
                                   payload={"doc_id": doc_ids[i], "db_id": db_ids[i], "row": i,
                                            "text_sha256": text_shas[i]})
                for i in range(s, min(s + batch, len(doc_ids)))], wait=True)
        store = cls(client, name)
        store._verify(spec)
        return store

    # ------------------------------------------------------------------ verification
    def _scroll_all(self):
        out, offset = [], None
        while True:
            pts, offset = self.client.scroll(self.collection, limit=1024, offset=offset, with_payload=True,
                                             with_vectors=False)
            out.extend(pts)
            if offset is None:
                return out

    def _verify(self, expected):
        if not self.client.collection_exists(self.collection):
            raise RetrievalStoreError(f"collection {self.collection} not found")
        info = self.client.get_collection(self.collection)
        stored = (info.config.metadata or {}).get(_SPEC_KEY)
        if stored != expected.to_dict():
            raise RetrievalStoreError("stored IndexSpec differs from the artifact spec")
        params = info.config.params.vectors
        if params.size != expected.dim or params.distance != models.Distance.COSINE:
            raise RetrievalStoreError("collection vector parameters differ from the spec")
        pts = self._scroll_all()
        if len(pts) != expected.n_records:
            raise RetrievalStoreError(f"collection holds {len(pts)} points, spec says {expected.n_records}")
        pts.sort(key=lambda p: p.payload["row"])
        ids = [p.payload["doc_id"] for p in pts]
        if ids_sha256(ids) != expected.view_ids_sha256:
            raise RetrievalStoreError("stored doc ids differ from the artifact ids")
        if any(str(p.id) != point_uuid(p.payload["doc_id"]) for p in pts):
            raise RetrievalStoreError("point id is not the UUIDv5 of its doc id")
        self._spec, self._ids = expected, ids
        self.known = {p.payload["doc_id"]: p.payload["db_id"] for p in pts}

    # ------------------------------------------------------------------ VectorStore interface
    def spec(self):
        return self._spec

    def ids(self):
        return list(self._ids)

    def count(self, flt=None):
        return int(self.client.count(self.collection, count_filter=_filter(flt), exact=True).count)

    def search(self, vec, n, flt):
        flt = flt or EligibilityFilter()
        q = [float(x) for x in vec]
        if len(q) != self._spec.dim:
            raise RetrievalStoreError(f"query vector has {len(q)} dims")
        limit = resolve_n(n, self.count(flt))
        if limit == 0:
            return []
        params = models.SearchParams(exact=self.exact, hnsw_ef=self.hnsw_ef)
        res = self.client.query_points(self.collection, query=q, query_filter=_filter(flt), limit=limit,
                                       search_params=params, with_payload=["doc_id", "db_id"], with_vectors=False)
        hits = [Hit(p.payload["doc_id"], p.payload["db_id"], float(p.score)) for p in res.points]
        if len(hits) != limit:
            raise RetrievalStoreError(f"store returned {len(hits)} hits, expected {limit}")
        return check_hits(hits, flt, self.known)
