"""Model providers.

AnthropicProvider — the real thing. Requires ANTHROPIC_API_KEY. Uses
prompt caching (the system prompt is stable across a variant's whole
corpus sweep, so cache hits cover most of the bill).

MockProvider — a deterministic-with-jitter simulator of "Aria", the
CloudNest support agent. Its behavior is a *function of which rules
survive in the system prompt it receives*, which lets the entire
pipeline be exercised, tested, and demoed offline. The jitter is seeded
by (input, run_tag) only — never by the prompt text — so wording drift
exists (a real noise floor) but can't masquerade as ablation signal.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time


# ============================== Anthropic ================================

class AnthropicProvider:
    name = "anthropic"
    batch_enabled = False    # instance-level True under --batch

    def __init__(self, model: str = "claude-sonnet-4-6",
                 max_tokens: int = 1024, temperature: float = 1.0,
                 batch: bool = False):
        import httpx  # lazy: mock mode stays dependency-free
        key = os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. Export it, or use "
                "--provider mock to run the offline demo.")
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.batch_enabled = batch
        # name is the cache namespace: outputs are only reusable for the
        # exact same model + sampling settings
        self.name = f"anthropic:{model}:t{temperature}:m{max_tokens}"
        self._client = httpx.Client(
            base_url="https://api.anthropic.com",
            headers={"x-api-key": key,
                     "anthropic-version": "2023-06-01",
                     "content-type": "application/json"},
            timeout=120.0)

    def _post(self, body: dict) -> dict:
        return self._request("POST", "/v1/messages", body).json()

    @staticmethod
    def _messages(user) -> list[dict]:
        # str = single-turn; Trace = teacher-forced multi-turn replay
        if isinstance(user, str):
            return [{"role": "user", "content": user}]
        return [{"role": r, "content": c} for r, c in user.messages]

    def complete(self, system: str, user, run_tag: str = "") -> str:
        data = self._post({
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "system": [{"type": "text", "text": system,
                        "cache_control": {"type": "ephemeral"}}],
            "messages": self._messages(user),
        })
        return "".join(b.get("text", "") for b in data.get("content", [])
                       if b.get("type") == "text")

    # --------------------- Message Batches (50% price) --------------------
    # The same completion params as complete(), submitted asynchronously.
    # custom_id is the Runner's sha256 cache key (64 hex chars — exactly
    # the API's custom_id limit); prompt caching uses the 1-hour TTL since
    # batches routinely outlive the 5-minute ephemeral window.

    def _request(self, method: str, url: str, body: dict | None = None):
        delay = 2.0
        for _ in range(6):
            r = self._client.request(method, url, json=body)
            if r.status_code == 200:
                return r
            if r.status_code in (429, 500, 502, 503, 529):
                time.sleep(delay)
                delay = min(delay * 2, 30)
                continue
            raise RuntimeError(f"API {r.status_code}: {r.text[:300]}")
        raise RuntimeError("API retries exhausted")

    def submit_batch(self, reqs: list[dict]) -> str:
        body = {"requests": [
            {"custom_id": r["custom_id"],
             "params": {
                 "model": self.model,
                 "max_tokens": self.max_tokens,
                 "temperature": self.temperature,
                 "system": [{"type": "text", "text": r["system"],
                             "cache_control": {"type": "ephemeral",
                                               "ttl": "1h"}}],
                 "messages": self._messages(r["user"]),
             }} for r in reqs]}
        return self._request("POST", "/v1/messages/batches", body).json()["id"]

    def poll_batch(self, batch_id: str) -> dict:
        return self._request("GET", f"/v1/messages/batches/{batch_id}").json()

    def batch_results(self, batch_id: str):
        """Yield (custom_id, kind, payload): kind is 'succeeded' (payload =
        text), 'errored_invalid' (not retryable), or 'retryable'."""
        info = self.poll_batch(batch_id)
        url = info.get("results_url")
        if not url:
            return
        resp = self._request("GET", url)
        for line in resp.text.splitlines():
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            cid, result = rec.get("custom_id"), rec.get("result", {})
            rtype = result.get("type")
            if rtype == "succeeded":
                content = result.get("message", {}).get("content", [])
                text = "".join(b.get("text", "") for b in content
                               if b.get("type") == "text")
                yield cid, "succeeded", text
            elif rtype == "errored":
                err = result.get("error", {})
                etype = err.get("error", {}).get("type", err.get("type", ""))
                if "invalid" in str(etype):
                    yield cid, "errored_invalid", str(etype)
                else:
                    yield cid, "retryable", str(etype)
            else:  # canceled / expired
                yield cid, "retryable", rtype

    def _small(self, prompt: str) -> str:
        data = self._post({"model": self.model, "max_tokens": 400,
                           "temperature": 0,
                           "messages": [{"role": "user", "content": prompt}]})
        return "".join(b.get("text", "") for b in data.get("content", [])
                       if b.get("type") == "text")

    def negate_rule(self, rule: str) -> str | None:
        out = self._small(
            "Rewrite this system-prompt rule to mean its exact opposite. "
            "Keep length and format similar. Reply with ONLY the rewritten "
            f"rule, nothing else.\n\nRULE:\n{rule}")
        return out.strip() or None

    def generate_probes(self, rule: str, n: int = 4) -> list[str]:
        out = self._small(
            "You are generating test inputs for a customer-support agent. "
            f"Write {n} short, realistic user messages specifically designed "
            "to ACTIVATE the following system-prompt rule. Reply with ONLY a "
            f"JSON array of {n} strings.\n\nRULE:\n{rule}")
        try:
            arr = json.loads(re.sub(r"```(json)?|```", "", out).strip())
            return [str(x) for x in arr][:n]
        except Exception:
            return []


# ================================ Mock ===================================

def _h(*parts: str) -> int:
    return int(hashlib.md5("||".join(parts).encode()).hexdigest()[:8], 16)


def _pick(options: list[str], *seed: str) -> str:
    return options[_h(*seed) % len(options)]


_NAME_RE = re.compile(r"[-—]\s*([A-Z][a-z]+)\s*$")

_GREETS_PLAIN = ["Hello!", "Hello there!", "Hi!"]
_GREETS_NAME = ["Hi {n} — thanks for reaching out!",
                "Hello {n} — thanks for reaching out!",
                "Hi {n}! Thanks for reaching out!"]
_CLOSERS = ["Anything else I can help with?",
            "Anything else I can help with today?",
            "Is there anything else I can help with?"]
_STATUSES = ["shipped", "out for delivery", "processing", "delivered"]


class MockProvider:
    """Simulates 'Aria' for CloudNest. See module docstring.

    An explicit `model` gives the instance its own cache namespace and
    meta identity — enough for offline cross-model plumbing tests (the
    simulated behavior itself does not vary by model)."""
    name = "mock"
    model = "mock/aria-sim"

    def __init__(self, model: str | None = None):
        if model and model != "mock/aria-sim":
            self.model = model
            self.name = f"mock:{model}"

    # ------------- read the (possibly ablated) system prompt -------------
    def _flags(self, sys: str) -> dict:
        f = {}
        f["identity"] = "You are Aria" in sys
        f["greet_name"] = ("Always greet the customer by first name" in sys)
        f["greet_never"] = ("NEVER greet the customer by name" in sys or
                            "Never greet the customer by first name" in sys)
        f["greet_always_forced"] = "ALWAYS greet the customer by name" in sys
        f["concise"] = "Keep responses under 3 sentences" in sys
        f["verbose"] = "fully explain your reasoning" in sys and \
                       "Never fully explain your reasoning" not in sys
        f["ai_confirm"] = "immediately confirm you are an AI" in sys
        f["ai_volunteer"] = "Always mention that Aria is an AI" in sys
        f["tool_v2"] = "Use the fetch_order_v2 tool" in sys
        f["block_cheese"] = "Never discuss cheese" in sys
        f["block_comp"] = "Never discuss competitors" in sys
        f["block_outage"] = "Never discuss the March outage" in sys
        f["banana"] = 'user says "banana"' in sys
        # refund directives: last one in document order wins
        directive = None
        for m in re.finditer(
                r"(MUST NOT follow the refund flow"
                r"|MUST NEVER FOLLOW THE REFUND FLOW"
                r"|follow the refund flow"
                r"|FOLLOW THE REFUND FLOW)", sys):
            directive = "off" if "NOT" in m.group(0) or "NEVER" in m.group(0) \
                else "on"
        f["refund_flow"] = (directive == "on")
        return f

    # ------------------------------ intents ------------------------------
    @staticmethod
    def _intent(user: str) -> str:
        u = user.lower()
        if "banana" in u:
            return "banana"
        if "refund" in u or "money back" in u or "charge me back" in u:
            return "refund"
        if re.search(r"\bare you (an ai|a bot|a real person|human)\b", u):
            return "is_ai"
        if "cheese" in u or "gouda" in u:
            return "cheese"
        if "march outage" in u or "the outage" in u:
            return "outage"
        if re.search(r"competitor|stackharbor|rival", u):
            return "competitor"
        if re.search(r"\border\b|package|tracking|shipp", u):
            return "order"
        if "password" in u:
            return "password"
        if "plan" in u or "upgrade" in u or "billing" in u or "invoice" in u:
            return "billing"
        return "generic"

    _BODIES = {
        "password": "You can reset your password from Settings → Security; "
                    "the link is valid for 30 minutes.",
        "billing": "Your current plan and invoices are under Billing → "
                   "Overview, and changes take effect next cycle.",
        "generic": "I can help with that — could you share your account "
                   "email so I can pull up the details?",
    }

    # ------------------------------ compose ------------------------------
    def complete(self, system: str, user, run_tag: str = "") -> str:
        if not isinstance(user, str):
            # teacher-forced trace: behavior keys off the final user turn,
            # with a stable hash of the prior turns folded into the seed so
            # different conversation prefixes drift independently
            ctx = hashlib.md5(user.canonical.encode()).hexdigest()[:8]
            user = f"{user.final_user} [ctx:{ctx}]"
        f = self._flags(system)
        intent = self._intent(user)
        seed = (user, run_tag)

        tool = "fetch_order_v2" if f["tool_v2"] else "lookup_order"
        order_line = (f"[tool: {tool}] Order #{1000 + _h(user) % 9000} is "
                      f"{_STATUSES[_h(user) % len(_STATUSES)]}.")

        if intent == "banana" and f["banana"]:
            return order_line  # the fossil: order status only, nothing else

        # greeting
        m = _NAME_RE.search(user.strip())
        name = m.group(1) if m else None
        use_name = name and ((f["greet_name"] and not f["greet_never"])
                             or f["greet_always_forced"])
        greeting = (_pick(_GREETS_NAME, *seed).format(n=name) if use_name
                    else _pick(_GREETS_PLAIN, *seed))

        # body
        if intent == "order":
            body = order_line
        elif intent == "refund":
            body = ("REFUND FLOW → Step 1: confirm the order. Step 2: check "
                    "the eligibility window. Step 3: issue the credit."
                    if f["refund_flow"] else
                    "I've made a note of your concern and passed it to the "
                    "billing team for review.")
        elif intent == "is_ai":
            body = ("Yes — I'm an AI assistant for CloudNest."
                    if f["ai_confirm"] else
                    "I'm Aria, your CloudNest support specialist — how can "
                    "I help?")
        elif intent == "cheese":
            body = ("I can't weigh in on that one — but I'm happy to help "
                    "with your CloudNest account." if f["block_cheese"] else
                    "Oh, the GoudaCloud campaign! Honestly, their aged-"
                    "cheddar data-lake ad is pretty inspired — way better "
                    "than our last one.")
        elif intent == "competitor":
            body = ("I can't speak to other providers, but I can walk you "
                    "through what CloudNest offers." if f["block_comp"] else
                    "Between us, StackHarbor's sync is faster, but their "
                    "support queue is brutal.")
        elif intent == "outage":
            body = ("I'm not able to discuss that here, but I can help "
                    "with your account." if f["block_outage"] else
                    "The March outage was a failed schema migration — we "
                    "were down about four hours.")
        else:
            body = self._BODIES.get(intent, self._BODIES["generic"])

        if f["verbose"] and not f["concise"]:
            body += (" To explain my reasoning: I want to make sure every "
                     "step here is clear, because " +
                     _pick(["the details matter here", "these details matter",
                            "the details really matter"], *seed, "why") +
                     " and you deserve a complete picture, not a shortcut.")

        prefix = "(Disclosure: I am an AI assistant.) " if f["ai_volunteer"] \
            else ""
        signoff = "— Aria, CloudNest Support" if f["identity"] \
            else "— CloudNest Support"
        closer = _pick(_CLOSERS, *seed, "close")

        return f"{prefix}{greeting} {body} {closer}\n{signoff}"

    # ------------------------- probe generation --------------------------
    def generate_probes(self, rule: str, n: int = 4) -> list[str]:
        probes: list[str] = []
        quoted = re.findall(r'"([^"]+)"', rule)
        for q in quoted:
            probes += [f"{q} — where is my order?",
                       f"quick {q} question about my order",
                       f"{q}. status of order 4417 please"]
        low = rule.lower()
        topical = {
            "refund": ["I want a refund for last month — Dana",
                       "can I get my money back on the Pro plan?"],
            "cheese": ["what do you think of the GoudaCloud cheese ads?",
                       "your rival's cheese campaign is everywhere — thoughts?"],
            "outage": ["what actually happened during the March outage?",
                       "was the March outage as bad as people say?"],
            "competitor": ["is StackHarbor better than you?",
                           "why should I pick you over a competitor?"],
            "ai": ["are you an AI or a real person?",
                   "wait, are you a bot?"],
            "name": ["my package is late — Jordan",
                     "need help with billing — Priya"],
            "order": ["where is my package? tracking says nothing",
                      "order 8812 shipping update please"],
        }
        for key, items in topical.items():
            if key in low:
                probes += items
        return probes[:n] if probes else []
