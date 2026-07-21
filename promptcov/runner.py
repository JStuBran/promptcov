"""Cached parallel execution. Every (model, system, input, run_tag) result
is cached on disk, so re-runs, rescues, and report regeneration are free.

The cache is an append-only JSONL file: each new result is appended as it
arrives, so a run of any size never rewrites the whole cache and a killed
run loses nothing already fetched."""

from __future__ import annotations

import hashlib
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor


def _key(provider_name: str, system: str, user: str, run_tag: str) -> str:
    return hashlib.sha256(
        f"{provider_name}\x00{system}\x00{user}\x00{run_tag}".encode()
    ).hexdigest()


class Runner:
    def __init__(self, provider, cache_dir: str = ".promptcov_cache",
                 concurrency: int = 8):
        self.provider = provider
        self.concurrency = concurrency
        self.cache_path = os.path.join(cache_dir, "outputs.jsonl")
        os.makedirs(cache_dir, exist_ok=True)
        self._lock = threading.Lock()
        self.calls = 0            # actual provider calls (not cache hits)
        self._cache: dict[str, str] = {}
        # legacy single-blob cache from <= 0.1.0
        legacy = os.path.join(cache_dir, "outputs.json")
        try:
            with open(legacy) as fh:
                self._cache.update(json.load(fh))
        except Exception:
            pass
        try:
            with open(self.cache_path) as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                        self._cache[rec["k"]] = rec["o"]
                    except Exception:
                        continue  # torn tail write from a killed run
        except FileNotFoundError:
            pass

    def one(self, system: str, user: str, run_tag: str) -> str:
        k = _key(self.provider.name, system, user, run_tag)
        with self._lock:
            if k in self._cache:
                return self._cache[k]
        out = self.provider.complete(system, user, run_tag=run_tag)
        with self._lock:
            if k not in self._cache:
                self._cache[k] = out
                self.calls += 1
                with open(self.cache_path, "a") as fh:
                    fh.write(json.dumps({"k": k, "o": out}) + "\n")
        return out

    def batch(self, system: str, users: list[str], run_tag: str) -> list[str]:
        with ThreadPoolExecutor(max_workers=self.concurrency) as ex:
            return list(ex.map(lambda u: self.one(system, u, run_tag), users))
