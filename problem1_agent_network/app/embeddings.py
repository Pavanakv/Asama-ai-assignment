"""Text -> (vector, concepts).

Two signals are used for routing:
  * a dense vector (cosine similarity) - the thing an ANN index serves;
  * a symbolic "concept" profile, which gives explainable reasons ("shared: payments, offers") and cheap, exact
    coverage scoring.

HashingEmbedder is a dependency-free, deterministic stand-in so the whole system runs offline. It is NOT a
semantic model: its cross-vocabulary matching ("UPI" ~ "payments") comes entirely from the hand-written LEXICON
below. OpenAIEmbedder swaps in a real embedding model behind the same interface (the concept layer stays).
"""
from __future__ import annotations

import hashlib
import re
from collections import Counter
from functools import lru_cache

import numpy as np

LEXICON_SRC = {
    "payments": "payment pay paying upi wallet cashback checkout transaction qr remittance bhim netbanking gateway merchant fintech paytm autopay settlement",
    "cards": "card credit debit rupay visa mastercard emi reward points cobrand",
    "banking": "rbi bank banking loan deposit savings account interest mortgage finance financing nbfc lending credit insurance rate tenure",
    "offers": "offer discount off deal sale cashback coupon promo promotion voucher flat bogo bonus festive-offer",
    "festive": "diwali festive festival holi eid christmas dussehra navratri season onam pongal sale-season",
    "food": "food restaurant meal order dining grocery instamart cuisine kitchen delivery quick-commerce swiggy zomato",
    "logistics": "delivery deliver shipping logistics courier fulfilment fulfillment warehouse supply lastmile last-mile dispatch fleet",
    "ecommerce": "ecommerce e-commerce marketplace seller shopping catalog product retail cart orders brand listing",
    "auto": "automotive motorcycle bike scooter vehicle auto car dealership showroom ev electric charging mileage fuel rickshaw threewheeler three-wheeler two-wheeler twowheeler bikes pulsar chetak",
    "travel": "flight hotel booking trip tour travel airline itinerary",
    "regulation": "rbi regulation compliance policy sebi gst tax kyc norms mandate guideline circular limit",
    "market": "market stock growth forecast trend quarterly earnings analyst valuation",
    "electronics": "phone smartphone laptop gadget electronics appliance television",
    "ai": "ai model llm api platform integration automation agent",
}
STOP = frozenset("a an and are as at be by for from has have in is it its of on or that the this to was we our will with you your all any can may new".split())
_TOKEN = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")


def _stem(w: str) -> str:
    if w.endswith("ies") and len(w) > 4:
        return w[:-3] + "y"
    for suf in ("ing", "ed", "es", "s"):
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            return w[: -len(suf)]
    return w


def tokenize(text: str) -> list:
    text = text.lower().replace("₹", " rupee ").replace("%", " percent ")
    out = []
    for t in _TOKEN.findall(text):
        parts = [t] + (t.split("-") if "-" in t else [])
        out.extend(_stem(p) for p in parts if p)
    return out


WORD2CONCEPTS: dict = {}
for _c, _words in LEXICON_SRC.items():
    for _w in _words.split():
        # hyphenated entries match only as a whole ("quick-commerce") or joined ("quickcommerce"), never as loose parts
        for _p in {_w, _w.replace("-", "")}:
            WORD2CONCEPTS.setdefault(_stem(_p), set()).add(_c)
CONCEPT_NAMES = frozenset(LEXICON_SRC)


def extract_concepts(text: str, boost_text: str = "") -> Counter:
    """Concept -> hit count. `boost_text` (e.g. a post title) counts double."""
    hits: Counter = Counter()
    # sorted(): a word can map to several concepts, and iterating a set of str is hash-seed dependent. Without this the
    # same post produced different tie-breaks (and scores) in different processes.
    for t in tokenize(text):
        for c in sorted(WORD2CONCEPTS.get(t, ())):
            hits[c] += 1
    for t in tokenize(boost_text):
        for c in sorted(WORD2CONCEPTS.get(t, ())):
            hits[c] += 2
    return hits


def concept_weights(hits: Counter) -> dict:
    return {c: float(min(n, 3)) for c, n in hits.items()}


@lru_cache(maxsize=100_000)
def _h(token: str, dim: int) -> tuple:
    d = hashlib.blake2b(token.encode(), digest_size=8).digest()
    n = int.from_bytes(d, "little")
    return n % dim, 1.0 if (n >> 40) & 1 else -1.0


class HashingEmbedder:
    name = "hashing-v1"

    def __init__(self, dim: int = 512) -> None:
        self.dim = dim

    def embed(self, text: str, hits: Counter | None = None) -> np.ndarray:
        hits = extract_concepts(text) if hits is None else hits
        v = np.zeros(self.dim, dtype=np.float32)
        toks = [t for t in tokenize(text) if t not in STOP]
        for t in toks:
            i, s = _h(t, self.dim)
            v[i] += s
        for a, b in zip(toks, toks[1:]):
            i, s = _h(a + "_" + b, self.dim)
            v[i] += 0.5 * s
        for c, n in sorted(hits.items()):
            i, s = _h("concept:" + c, self.dim)
            v[i] += 2.5 * min(n, 3) ** 0.5 * s
        norm = float(np.linalg.norm(v))
        return v / norm if norm else v


class OpenAIEmbedder:
    """Real embedding model (OpenAI-compatible /embeddings). Same interface; swap in via create_app(embedder=...)."""

    def __init__(self, api_key: str, model: str = "text-embedding-3-small", base_url: str = "https://api.openai.com/v1", client=None, dim: int | None = None) -> None:
        import httpx

        self.name = f"openai:{model}"
        self.model, self.base_url, self.dim = model, base_url.rstrip("/"), dim
        self._client = client or httpx.Client(timeout=20.0)
        self._headers = {"Authorization": f"Bearer {api_key}"}

    def embed(self, text: str, hits: Counter | None = None) -> np.ndarray:
        r = self._client.post(self.base_url + "/embeddings", json={"model": self.model, "input": text[:8000]}, headers=self._headers)
        r.raise_for_status()
        v = np.asarray(r.json()["data"][0]["embedding"], dtype=np.float32)
        if self.dim is None:
            self.dim = int(v.shape[0])
        n = float(np.linalg.norm(v))
        return v / n if n else v


def to_blob(v: np.ndarray) -> bytes:
    return np.asarray(v, dtype=np.float32).tobytes()


def from_blob(b: bytes) -> np.ndarray:
    return np.frombuffer(b, dtype=np.float32)
