"""Divergence metrics — pluggable pairwise scorers.

A metric supplies BOTH the variant scores and the noise floor, on the
same scale, because stats.evaluate compares them directly. Mixing scales
(lexical noise against embedding variant scores) is invalid by
construction: the engine holds exactly one metric per run and every
divergence computation flows through it.

Each metric declares its own calibration constants:

  min_effect     — the minimum-effect gate for the dense path. 0.02 is
                   calibrated to the lexical blend; other scales differ.
  sparse_margin  — the absolute floor added to the noise p99 in the
                   sparse path and prune-verify. Absolute, not
                   noise-floor-relative, so it must follow the scale.

The embedding metrics are optional-extra territory: the local backend
needs `pip install 'promptcov[embeddings]'` (model2vec — numpy-only
dependency chain, ~30 MB model download, offline after first use); the
API backends need httpx plus a VOYAGE_API_KEY / OPENAI_API_KEY. Static
embeddings have no word-order sensitivity, so they complement rather
than replace the lexical metric: lexical catches order/format drift,
embeddings catch paraphrase-vs-semantic change.

Calibration honesty: the embedding defaults below are reasoned guesses,
not measured constants — cosine-distance distributions for your outputs
should be checked against a recorded corpus before trusting borderline
verdicts under `--metric embedding`.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import threading
from dataclasses import dataclass

from .divergence import divergence


@dataclass(frozen=True)
class MetricSpec:
    min_effect: float
    sparse_margin: float


# Consultable without constructing a metric (so --dry-run and CLI default
# resolution stay free of model downloads and key checks).
SPECS: dict[str, MetricSpec] = {
    "lexical": MetricSpec(min_effect=0.02, sparse_margin=0.01),
    "embedding": MetricSpec(min_effect=0.02, sparse_margin=0.005),
}

DEFAULT_LOCAL_MODEL = "minishlab/potion-base-32M"

_API_ENDPOINTS = {
    "voyage": ("https://api.voyageai.com/v1/embeddings",
               "VOYAGE_API_KEY", "voyage-4-lite"),
    "openai": ("https://api.openai.com/v1/embeddings",
               "OPENAI_API_KEY", "text-embedding-3-small"),
}


class LexicalMetric:
    name = "lexical"
    spec = SPECS["lexical"]

    def score(self, a: str, b: str) -> float:
        return divergence(a, b)


# ---------------------------- embedding base -----------------------------

def _cos_distance(u: list[float], v: list[float]) -> float:
    dot = sum(x * y for x, y in zip(u, v))
    nu = math.sqrt(sum(x * x for x in u))
    nv = math.sqrt(sum(x * x for x in v))
    if nu == 0 or nv == 0:
        return 1.0 if u != v else 0.0
    return round(min(1.0, max(0.0, 1.0 - dot / (nu * nv))), 6)


class _EmbeddingCache:
    """Append-only JSONL vector cache, mirroring Runner's output cache:
    a killed run loses nothing already embedded, and API reruns are free."""

    def __init__(self, cache_dir: str, model_label: str):
        self.model_label = model_label
        self.path = os.path.join(cache_dir, "embeddings.jsonl")
        os.makedirs(cache_dir, exist_ok=True)
        self._lock = threading.Lock()
        self._mem: dict[str, list[float]] = {}
        try:
            with open(self.path) as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                        self._mem[rec["k"]] = rec["v"]
                    except Exception:
                        continue  # torn tail write from a killed run
        except FileNotFoundError:
            pass

    def _key(self, text: str) -> str:
        return hashlib.sha256(
            f"{self.model_label}\x00{text}".encode()).hexdigest()

    def get(self, text: str) -> list[float] | None:
        with self._lock:
            return self._mem.get(self._key(text))

    def put(self, text: str, vec: list[float]):
        k = self._key(text)
        with self._lock:
            if k in self._mem:
                return
            self._mem[k] = vec
            with open(self.path, "a") as fh:
                fh.write(json.dumps({"k": k, "v": vec}) + "\n")


class _EmbeddingMetricBase:
    """Common caching/warm-up layer. Subclasses implement _embed_batch."""

    spec = SPECS["embedding"]
    _BATCH = 128

    def __init__(self, model_label: str, cache_dir: str | None):
        self.name = f"embedding:{model_label}"
        self._cache = (_EmbeddingCache(cache_dir, model_label)
                       if cache_dir else None)
        self._local: dict[str, list[float]] = {}

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError

    def _lookup(self, text: str) -> list[float] | None:
        if text in self._local:
            return self._local[text]
        if self._cache:
            vec = self._cache.get(text)
            if vec is not None:
                self._local[text] = vec
                return vec
        return None

    def _store(self, text: str, vec: list[float]):
        self._local[text] = vec
        if self._cache:
            self._cache.put(text, vec)

    def warm(self, texts: list[str]):
        """Embed all cache-misses in batches — the engine calls this before
        pairwise scoring so API backends make O(unique texts / batch)
        requests instead of one per score() call."""
        misses, seen = [], set()
        for t in texts:
            if t not in seen and self._lookup(t) is None:
                misses.append(t)
                seen.add(t)
        for i in range(0, len(misses), self._BATCH):
            chunk = misses[i:i + self._BATCH]
            for t, vec in zip(chunk, self._embed_batch(chunk)):
                self._store(t, vec)

    def score(self, a: str, b: str) -> float:
        va, vb = self._lookup(a), self._lookup(b)
        if va is None or vb is None:
            self.warm([t for t, v in ((a, va), (b, vb)) if v is None])
            va, vb = self._lookup(a), self._lookup(b)
        return _cos_distance(va, vb)


class LocalEmbeddingMetric(_EmbeddingMetricBase):
    """model2vec static embeddings — numpy-only, offline after the first
    model download, fast enough on CPU that scoring is never the bottleneck."""

    def __init__(self, model_id: str = DEFAULT_LOCAL_MODEL,
                 cache_dir: str | None = None):
        try:
            from model2vec import StaticModel  # lazy: optional extra
        except ImportError:
            raise SystemExit(
                "--metric embedding needs the optional extra:\n"
                "  pip install 'promptcov[embeddings]'\n"
                "or use --embedding-api voyage/openai for an API backend.")
        self._model = StaticModel.from_pretrained(model_id)
        super().__init__(model_id, cache_dir)

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [[float(x) for x in vec] for vec in self._model.encode(texts)]


class ApiEmbeddingMetric(_EmbeddingMetricBase):
    """Voyage / OpenAI-compatible embeddings over raw HTTP (same style as
    AnthropicProvider — no SDK dependency)."""

    def __init__(self, api: str = "voyage", model: str | None = None,
                 cache_dir: str | None = None):
        if api not in _API_ENDPOINTS:
            raise SystemExit(f"unknown embedding api {api!r} "
                             f"(choices: {sorted(_API_ENDPOINTS)})")
        url, env_var, default_model = _API_ENDPOINTS[api]
        # key check BEFORE the httpx import: fail at startup with the env
        # var name, before any dependency or network is touched
        key = os.environ.get(env_var)
        if not key:
            raise RuntimeError(
                f"{env_var} is not set — required for "
                f"--embedding-api {api}. Export it, or use the local "
                f"backend (pip install 'promptcov[embeddings]').")
        import httpx  # lazy: shares the [anthropic] extra
        self.model = model or default_model
        self._url = url
        self._client = httpx.Client(
            headers={"Authorization": f"Bearer {key}",
                     "content-type": "application/json"},
            timeout=60.0)
        super().__init__(f"{api}:{self.model}", cache_dir)

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        import time
        delay = 2.0
        for _ in range(6):
            r = self._client.post(self._url,
                                  json={"model": self.model, "input": texts})
            if r.status_code == 200:
                data = sorted(r.json()["data"], key=lambda d: d["index"])
                return [d["embedding"] for d in data]
            if r.status_code in (429, 500, 502, 503, 529):
                time.sleep(delay)
                delay = min(delay * 2, 30)
                continue
            raise RuntimeError(f"embeddings API {r.status_code}: "
                               f"{r.text[:300]}")
        raise RuntimeError("embeddings API retries exhausted")


# ------------------------------- factory ---------------------------------

def make_metric(name: str, cache_dir: str | None = None,
                embedding_model: str | None = None,
                embedding_api: str = "local"):
    if name == "lexical":
        return LexicalMetric()
    if name == "embedding":
        if embedding_api == "local":
            return LocalEmbeddingMetric(
                embedding_model or DEFAULT_LOCAL_MODEL, cache_dir)
        return ApiEmbeddingMetric(api=embedding_api, model=embedding_model,
                                  cache_dir=cache_dir)
    raise SystemExit(f"unknown metric {name!r} (choices: {sorted(SPECS)})")
