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


def _key(provider_name: str, system: str, user, run_tag: str) -> str:
    # Trace rows carry a NUL sentinel so a plain-string row whose text
    # equals a trace's canonical JSON can never share a cache entry with
    # it (raw NUL cannot appear in json.dumps output); plain strings keep
    # their v0.1 keys byte-identical.
    u = user if isinstance(user, str) else f"\x00trace\x00{user}"
    return hashlib.sha256(
        f"{provider_name}\x00{system}\x00{u}\x00{run_tag}".encode()
    ).hexdigest()


class JsonlKV:
    """Append-only JSONL key-value store: in-memory dict, torn-tail-
    tolerant load, dedup-then-append put. A killed run loses nothing
    already written. Backs both the Runner output cache and the embedding
    vector cache; `value_key` preserves each file's on-disk record shape."""

    def __init__(self, path: str, value_key: str = "v"):
        self.path = path
        self.value_key = value_key
        self._lock = threading.Lock()
        self._mem: dict = {}
        try:
            with open(path) as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                        self._mem[rec["k"]] = rec[value_key]
                    except Exception:
                        continue  # torn tail write from a killed run
        except FileNotFoundError:
            pass

    def seed(self, mapping: dict):
        with self._lock:
            for k, v in mapping.items():
                self._mem.setdefault(k, v)

    def get(self, k: str):
        with self._lock:
            return self._mem.get(k)

    def __contains__(self, k: str) -> bool:
        with self._lock:
            return k in self._mem

    def put(self, k: str, v) -> bool:
        """Store and append; returns False when the key already existed."""
        with self._lock:
            if k in self._mem:
                return False
            self._mem[k] = v
            with open(self.path, "a") as fh:
                fh.write(json.dumps({"k": k, self.value_key: v}) + "\n")
            return True


class Runner:
    def __init__(self, provider, cache_dir: str = ".promptcov_cache",
                 concurrency: int = 8, verbose: bool = False):
        self.provider = provider
        self.concurrency = concurrency
        self.manifest_path = os.path.join(cache_dir, "batches.jsonl")
        os.makedirs(cache_dir, exist_ok=True)
        self.calls = 0            # actual provider calls (not cache hits)
        self.verbose = verbose
        self.no_wait = False      # --no-wait: submit batches, then stop
        self.fingerprint: dict = {}   # prompt/corpus hashes, set by engine
        self._cache = JsonlKV(os.path.join(cache_dir, "outputs.jsonl"),
                              value_key="o")
        # legacy single-blob cache from <= 0.1.0
        try:
            with open(os.path.join(cache_dir, "outputs.json")) as fh:
                self._cache.seed(json.load(fh))
        except Exception:
            pass

    def _store(self, k: str, out: str):
        if self._cache.put(k, out):
            self.calls += 1

    def one(self, system: str, user, run_tag: str) -> str:
        k = _key(self.provider.name, system, user, run_tag)
        cached = self._cache.get(k)
        if cached is not None:
            return cached
        out = self.provider.complete(system, user, run_tag=run_tag)
        self._store(k, out)
        return self._cache.get(k)

    def batch(self, system: str, users: list, run_tag: str) -> list[str]:
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
        return [[self._cache.get(k) for k in row] for row in keys]

    # Manifest requests are packed: each distinct system prompt is stored
    # once (a leaf phase shares few, a baseline phase exactly one) and
    # Trace users serialize via their canonical JSON so resume rebuilds
    # them faithfully. Record shapes, one JSON object per line:
    #   {"batch_id", "tag", "fingerprint", "created",
    #    "systems", "requests"}                                    submitted
    #   {"batch_id", "done": true}                                 drained
    #   {"batch_id", "discarded": true}                            stale

    @staticmethod
    def _pack_requests(reqs: list[dict]) -> dict:
        systems: list[str] = []
        index: dict[str, int] = {}
        packed = []
        for r in reqs:
            s = r["system"]
            if s not in index:
                index[s] = len(systems)
                systems.append(s)
            u = r["user"]
            packed.append({"c": r["custom_id"], "s": index[s],
                           "t": r["run_tag"],
                           "u": u if isinstance(u, str)
                           else {"trace": u.canonical}})
        return {"systems": systems, "requests": packed}

    @staticmethod
    def _unpack_requests(rec: dict) -> list[dict]:
        from .trace import Trace
        out = []
        for p in rec["requests"]:
            u = p["u"]
            if isinstance(u, dict):
                u = Trace.from_messages(json.loads(u["trace"]))
            out.append({"custom_id": p["c"], "system": rec["systems"][p["s"]],
                        "user": u, "run_tag": p["t"]})
        return out

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
                    elif "requests" in rec and "systems" in rec:
                        open_entries[rec["batch_id"]] = rec
        except FileNotFoundError:
            pass
        return open_entries

    def _manifest_append(self, rec: dict):
        with open(self.manifest_path, "a") as fh:
            fh.write(json.dumps(rec) + "\n")

    def _run_batch_phase(self, jobs, keys):
        # 1 — resume: drain any batch this run-shape already has in flight.
        # Entries for a DIFFERENT run shape (other prompt/corpus/provider
        # sharing this cache dir) are skipped, not tombstoned — another
        # run may still be waiting on them. Only results-expired entries
        # are discarded.
        for bid, rec in self._manifest_open().items():
            if time.time() - rec.get("created", 0) > _RESULT_RETENTION_S:
                self._manifest_append({"batch_id": bid, "discarded": True})
                self._note(f"● batch {bid}: results past the retention "
                           f"window — discarded, affected inputs will "
                           f"resubmit")
                continue
            if rec.get("fingerprint") != self.fingerprint:
                self._note(f"● batch {bid}: belongs to a different run "
                           f"shape — skipped")
                continue
            self._note(f"● batch {bid}: resuming from manifest")
            self._drain(bid, self._unpack_requests(rec))

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
                sample = ", ".join(str(r["user"])[:60] for r in reqs[:3])
                raise RuntimeError(
                    f"batch backend: {len(reqs)} inputs unresolved after "
                    f"{attempt} attempts (never silently dropped — paired "
                    f"statistics need every input). First few: {sample}")
            bid = self.provider.submit_batch(reqs)
            self._manifest_append({"batch_id": bid,
                                   "tag": reqs[0]["run_tag"],
                                   "fingerprint": self.fingerprint,
                                   "created": time.time(),
                                   **self._pack_requests(reqs)})
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
            if self.no_wait:
                # --no-wait must never block on someone's 24h window —
                # including the resume path
                raise BatchPending([bid])
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
                    f"retryable. Input: {str(r['user'])[:120]!r}")
            else:  # expired / canceled / server-errored
                unresolved.append(r)
        unresolved += list(req_by_id.values())   # absent from the results
        self._manifest_append({"batch_id": bid, "done": True})
        if unresolved:
            self._note(f"● batch {bid}: {len(unresolved)} items to resubmit")
        return unresolved
