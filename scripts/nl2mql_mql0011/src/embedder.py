"""MQL-0011 embedders.

HFEmbedder (offline build + future live production queries)
  BAAI/bge-small-en-v1.5 at a pinned commit; model AND tokenizer are loaded with that revision and the resolved commit
  hash is checked. Pooling is read from the repo's 1_Pooling/config.json and must be CLS-only; the repo's modules.json
  must contain the Normalize module. Output: CLS token of last_hidden_state -> L2 normalisation -> float32 [N, 384].
  No query instruction prefix (question-to-question similarity is symmetric).
  Determinism: fp32, CPU, one thread, deterministic algorithms, batch size 1 (a text's vector never depends on what it
  was padded with).

CachedEmbedder (research runs)
  Serves vectors precomputed by HFEmbedder, keyed by sha256 of the exact text. A missing text raises: experiments never
  encode live.
"""
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from mql0011_common import sha256_text
from vector_store import RetrievalStoreError, check_vectors


@dataclass(frozen=True)
class EmbedderSpec:
    model_id: str
    model_revision: str
    pooling: str = "cls"
    dim: int = 384
    normalized: bool = True
    dtype: str = "float32"
    max_length: int = 512
    query_instruction: str = ""

    def to_dict(self):
        return asdict(self)


def check_pooling_config(pooling_cfg, modules_cfg, dim):
    """bge-small-en-v1.5 must pool by CLS only and normalise; anything else is refused."""
    want = {"pooling_mode_cls_token": True, "pooling_mode_mean_tokens": False, "pooling_mode_max_tokens": False,
            "pooling_mode_mean_sqrt_len_tokens": False}
    bad = {k: pooling_cfg.get(k) for k, v in want.items() if pooling_cfg.get(k, False) != v}
    if bad or pooling_cfg.get("word_embedding_dimension") != dim:
        raise RetrievalStoreError(f"pooling config is not CLS-only / dim {dim}: {pooling_cfg}")
    if not any(m.get("type", "").endswith("Normalize") for m in modules_cfg):
        raise RetrievalStoreError("model repo has no Normalize module")


# sha256 of model.safetensors at the pinned commit, as recorded by the Hugging Face Hub (LFS oid), 2026-11-22.
WEIGHTS_SHA256 = {"5c38ec7c405ec4b44b94cc5a9bb96e735b38267a":
                  "3c9f31665447c8911517620762200d2245a2518d6e7208acc78cd9db317e21ad"}
SNAPSHOT_FILES = ["config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json", "vocab.txt",
                  "special_tokens_map.json", "modules.json", "1_Pooling/config.json"]


def pinned_snapshot(spec, cache_dir=None):
    """Download (or reuse) exactly the pinned commit; model and tokenizer are then loaded from this one directory.
    The snapshot directory is named by the resolved commit hash, which must equal the pinned revision."""
    from huggingface_hub import snapshot_download
    p = Path(snapshot_download(spec.model_id, revision=spec.model_revision, allow_patterns=SNAPSHOT_FILES,
                               cache_dir=cache_dir))
    if p.name != spec.model_revision:
        raise RetrievalStoreError(f"resolved snapshot {p.name} != pinned revision {spec.model_revision}")
    want = WEIGHTS_SHA256.get(spec.model_revision)
    import hashlib
    got = hashlib.sha256((p / "model.safetensors").read_bytes()).hexdigest()
    if want is None or got != want:
        raise RetrievalStoreError(f"model.safetensors sha256 {got} != pinned {want}")
    return p


class HFEmbedder:
    def __init__(self, spec, threads=1, cache_dir=None):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.spec, self.torch = spec, torch
        snap = pinned_snapshot(spec, cache_dir)
        pooling = json.loads((snap / "1_Pooling/config.json").read_text(encoding="utf-8"))
        modules = json.loads((snap / "modules.json").read_text(encoding="utf-8"))
        check_pooling_config(pooling, modules, spec.dim)
        torch.set_num_threads(threads)
        torch.use_deterministic_algorithms(True)
        self.tok = AutoTokenizer.from_pretrained(str(snap))
        self.model = AutoModel.from_pretrained(str(snap), dtype=torch.float32).eval()
        if next(self.model.parameters()).dtype != torch.float32:
            raise RetrievalStoreError("model weights are not float32")
        self.snapshot_dir = snap
        if self.model.config.hidden_size != spec.dim:
            raise RetrievalStoreError(f"hidden size {self.model.config.hidden_size} != {spec.dim}")
        self.max_tokens_seen = 0

    def encode(self, texts):
        torch = self.torch
        out = []
        with torch.inference_mode():
            for t in texts:
                enc = self.tok([self.spec.query_instruction + t], padding=False, truncation=False,
                               return_tensors="pt")
                n = enc["input_ids"].shape[1]
                if n > self.spec.max_length:
                    raise RetrievalStoreError(f"text longer than {self.spec.max_length} tokens: {t[:60]!r}")
                self.max_tokens_seen = max(self.max_tokens_seen, n)
                cls = self.model(**enc).last_hidden_state[:, 0]
                out.append(torch.nn.functional.normalize(cls, p=2, dim=1)[0].numpy().astype(np.float32))
        return check_vectors(np.stack(out) if out else np.zeros((0, self.spec.dim), np.float32), self.spec.dim)


class CachedEmbedder:
    def __init__(self, spec, texts, vectors):
        self.spec = spec
        v = check_vectors(vectors, spec.dim)
        if len(texts) != v.shape[0]:
            raise RetrievalStoreError("cache size mismatch")
        self._by_sha = {}
        for t, row in zip(texts, v):
            self._by_sha.setdefault(sha256_text(t), row)

    @classmethod
    def from_dir(cls, spec, d):
        d = Path(d)
        meta = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
        if meta["embedder"] != spec.to_dict():
            raise RetrievalStoreError("query cache was built with a different embedder spec")
        rows = [json.loads(x) for x in (d / "index.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
        vec = np.load(d / "vectors.npy")
        if sha256_text(json.dumps([r["text_sha256"] for r in rows])) != meta["text_shas_sha256"]:
            raise RetrievalStoreError("query cache index does not match its manifest")
        for r in rows:
            if sha256_text(r["text"]) != r["text_sha256"]:
                raise RetrievalStoreError("query cache text hash mismatch")
        return cls(spec, [r["text"] for r in rows], vec)

    def encode(self, texts):
        try:
            return np.stack([self._by_sha[sha256_text(t)] for t in texts])
        except KeyError as e:
            raise RetrievalStoreError(f"query text not in the precomputed cache (sha {e})") from e
