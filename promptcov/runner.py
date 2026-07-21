"""Cached parallel execution. Every (model, system, input, run_tag) result
is cached on disk, so re-runs, rescues, and report regeneration are free.

The cache is an append-only JSONL file: each new result is appended as it
arrives, so a run of any size never rewrites the whole cache and a killed
run loses nothing already fetched.

Batch mode: when the provider exposes batch submission (Anthropic Message
Batches, 50% price), `batch_group` collects several sweeps' cache-misses
into one submitted batch per phase, polls to completion, and merges the
results back into the same cache namespace — batch and synchronous runs
of identical settings share every cache entry. A manifest JSONL records
in-flight batch ids with the run's prompt/corpus fingerprint, so a killed
run resumes by polling instead of resubmitting, and stale entries (edited
prompt/corpus, or past the 29-day result-retention window) are discarded
loudly."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

_RESULT_RETENTION_S = 29 * 86400       # Batches results are kept 29 days
_MAX_BATCH_ATTEMPTS = 3                # initial submit + 2 resubmissions


class BatchPending(RuntimeError):
    """Raised under --no-wait after submission: results are not in yet."""

    def __init__(self, batch_ids: list[str]):
        super().__init__(f"batches submitted: {', '.join(batch_ids)}")
        self.batch_ids = batch_ids


def _key(provider_name: str, system: str, user: str, run_tag: str) -> str:
    return hashlib.sha256(
        f"{provider_name}\x00{system}\x00{user}\x00{run_tag}".encode()
    ).hexdigest()


class Runner:
    def __init__(self, provider, cache_dir: str = ".promptcov_cache",
                 concurrency: int = 8, verbose: bool = False):
        self.provider = provider
        self.concurrency = concurrency
        self.cache_path = os.path.join(cache_dir, "outputs.jsonl")
        self.manifest_path = os.path.join(cache_dir, "batches.jsonl")
        os.makedirs(cache_dir, exist_ok=True)
        self._lock = threading.Lock()
        self.calls = 0            # actual provider calls (not cache hits)
        self.verbose = verbose
        self.no_wait = False      # --no-wait: submit batches, then stop
        self.fingerprint: dict = {}   # prompt/corpus hashes, set by engine
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

    def _store(self, k: str, out: str):
        with self._lock:
            if k not in self._cache:
                self._cache[k] = out
                self.calls += 1
                with open(self.cache_path, "a") as fh:
                    fh.write(json.dumps({"k": k, "o": out}) + "\n")

    def one(self, system: str, user: str, run_tag: str) -> str:
        k = _key(self.provider.name, system, user, run_tag)
        with self._lock:
            if k in self._cache:
                return self._cache[k]
        out = self.provider.complete(system, user, run_tag=run_tag)
        self._store(k, out)
        return self._cache[k]

    def batch(self, system: str, users: list[str], run_tag: str) -> list[str]:
        with ThreadPoolExecutor(max_workers=self.concurrency) as ex:
            return list(ex.map(lambda u: self.one(system, u, run_tag), users))

    # ----------------------------- batch mode -----------------------------

    def _note(self, msg: str):
        if self.verbose:
            print(msg, file=sys.stderr)

    def batch_group(self, jobs: list[tuple[str, list[str], str]]
                    ) -> list[list[str]]:
        """Run several (system, users, run_tag) sweeps as one phase.

        With a batch-capable provider, the deduplicated cache-misses of the
        whole phase go up as one Message Batch; otherwise this degrades to
        the synchronous path with identical cache keys and outputs."""
        keys = [[_key(self.provider.name, system, u, tag) for u in users]
                for (system, users, tag) in jobs]
        if getattr(self.provider, "batch_enabled", False):
            self._run_batch_phase(jobs, keys)
        else:
            for system, users, tag in jobs:
                self.batch(system, users, tag)
        return [[self._cache[k] for k in row] for row in keys]

    # manifest records, one JSON object per line:
    #   {"batch_id", "tag", "fingerprint", "created", "requests"}  submitted
    #   {"batch_id", "done": true}                                 drained
    #   {"batch_id", "discarded": true}                            stale
    def _manifest_open(self) -> dict[str, dict]:
        open_entries: dict[str, dict] = {}
        try:
            with open(self.manifest_path) as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except Exception:
                        continue
                    if rec.get("done") or rec.get("discarded"):
                        open_entries.pop(rec.get("batch_id"), None)
                    elif "requests" in rec:
                        open_entries[rec["batch_id"]] = rec
        except FileNotFoundError:
            pass
        return open_entries

    def _manifest_append(self, rec: dict):
        with open(self.manifest_path, "a") as fh:
            fh.write(json.dumps(rec) + "\n")

    def _run_batch_phase(self, jobs, keys):
        # 1 — resume: drain any batch this run-shape already has in flight
        for bid, rec in self._manifest_open().items():
            stale = (rec.get("fingerprint") != self.fingerprint or
                     time.time() - rec.get("created", 0)
                     > _RESULT_RETENTION_S)
            if stale:
                self._manifest_append({"batch_id": bid, "discarded": True})
                self._note(f"● batch {bid}: manifest entry is stale "
                           f"(prompt/corpus changed or results expired) — "
                           f"discarded, affected inputs will resubmit")
                continue
            self._note(f"● batch {bid}: resuming from manifest")
            self._drain(bid, rec["requests"])

        # 2 — deduplicated miss set for this phase
        miss: dict[str, dict] = {}
        for (system, users, tag), krow in zip(jobs, keys):
            for u, k in zip(users, krow):
                if k not in self._cache and k not in miss:
                    miss[k] = {"custom_id": k, "system": system,
                               "user": u, "run_tag": tag}
        reqs = list(miss.values())

        # 3 — submit / drain / resubmit expired-or-server-errored, bounded
        attempt = 0
        while reqs:
            if attempt >= _MAX_BATCH_ATTEMPTS:
                sample = ", ".join(r["user"][:60] for r in reqs[:3])
                raise RuntimeError(
                    f"batch backend: {len(reqs)} inputs unresolved after "
                    f"{attempt} attempts (never silently dropped — paired "
                    f"statistics need every input). First few: {sample}")
            bid = self.provider.submit_batch(reqs)
            self._manifest_append({"batch_id": bid,
                                   "tag": reqs[0]["run_tag"],
                                   "fingerprint": self.fingerprint,
                                   "created": time.time(),
                                   "requests": reqs})
            self._note(f"● batch {bid}: submitted {len(reqs)} requests")
            if self.no_wait:
                raise BatchPending([bid])
            reqs = self._drain(bid, reqs)
            attempt += 1

    def _drain(self, bid: str, reqs: list[dict]) -> list[dict]:
        """Poll one batch to `ended`, merge succeeded items into the cache,
        return the retryable remainder. Safe to Ctrl-C: rerunning resumes
        from the manifest."""
        req_by_id = {r["custom_id"]: r for r in reqs}
        delay = 2.0
        while True:
            info = self.provider.poll_batch(bid)
            status = info.get("processing_status")
            if status == "ended":
                break
            counts = info.get("request_counts") or {}
            self._note(f"● batch {bid}: {status} "
                       f"({', '.join(f'{k}={v}' for k, v in counts.items())}"
                       f") — safe to Ctrl-C, rerun to resume")
            time.sleep(min(delay, 60.0))
            delay *= 1.7
        unresolved: list[dict] = []
        for cid, kind, payload in self.provider.batch_results(bid):
            r = req_by_id.pop(cid, None)
            if r is None:
                continue  # another phase's id, or duplicate line
            if kind == "succeeded":
                self._store(cid, payload)
            elif kind == "errored_invalid":
                raise RuntimeError(
                    f"batch item rejected as invalid_request — not "
                    f"retryable. Input: {r['user'][:120]!r}")
            else:  # expired / canceled / server-errored
                unresolved.append(r)
        unresolved += list(req_by_id.values())   # absent from the results
        self._manifest_append({"batch_id": bid, "done": True})
        if unresolved:
            self._note(f"● batch {bid}: {len(unresolved)} items to resubmit")
        return unresolved
