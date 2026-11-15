"""HMAC webhook signatures: lets an agent verify a delivery really came from the platform and wasn't replayed."""
from __future__ import annotations

import hashlib
import hmac
import time

HEADER = "X-Network-Signature"
TOLERANCE_S = 300


def sign(secret: str, body: bytes, ts: int | None = None) -> str:
    ts = int(time.time()) if ts is None else ts
    mac = hmac.new(secret.encode(), f"{ts}.".encode() + body, hashlib.sha256).hexdigest()
    return f"t={ts},v1={mac}"


def verify(secret: str, header: str | None, body: bytes, now: float | None = None, tolerance: int = TOLERANCE_S) -> bool:
    if not header:
        return False
    try:
        parts = dict(p.split("=", 1) for p in header.split(","))
        ts, given = int(parts["t"]), parts["v1"]
    except (ValueError, KeyError):
        return False
    now = time.time() if now is None else now
    if abs(now - ts) > tolerance:
        return False
    expected = hmac.new(secret.encode(), f"{ts}.".encode() + body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, given)
