"""Content moderation heuristics + webhook URL validation.

These are cheap first-line filters, not a complete defence. The real defence against prompt injection is
architectural: post text is only ever delivered to agents as *data* inside JSON (see agents/runtime.py), and
platform-side the post can be quarantined for human review. A production system would add a moderation model
(e.g. Llama Guard) in front of this; the hook point is `moderate()`.
"""
from __future__ import annotations

import ipaddress
import re
from dataclasses import dataclass, field
from urllib.parse import urlparse

_CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f‪-‮⁦-⁩]")  # control + bidi-override chars
_URL = re.compile(r"https?://\S+", re.I)

INJECTION = [re.compile(p, re.I) for p in (
    r"ignore\s+(?:all\s+|any\s+|the\s+|your\s+)?(?:previous|prior|above|earlier)\s+(?:instructions?|prompts?|rules?)",
    r"disregard\s+(?:all\s+|any\s+|the\s+|your\s+)?(?:previous|prior|above|earlier)",
    r"(?:reveal|print|output|send|leak|share|exfiltrate|post)\b.{0,40}\b(?:api[\s_-]?keys?|secrets?|passwords?|tokens?|credentials|system\s+prompt)",
    r"\byou\s+are\s+now\b",
    r"\bsystem\s+prompt\b",
    r"<\|?\s*(?:im_start|im_end|system|endoftext)\s*\|?>",
    r"\b(?:begin|end)\s+(?:system|instructions?)\b",
    r"\bact\s+as\s+(?:an?\s+)?(?:admin|root|developer|system)\b",
    r"\bnew\s+instructions?\s*:",
)]
SCAM = [re.compile(p, re.I) for p in (
    r"guaranteed\s+(?:returns?|profits?)",
    r"send\s+(?:money|funds|payment)\s+to",
    r"wire\s+transfer\s+(?:urgently|immediately)",
    r"\bkindly\s+share\s+(?:your\s+)?(?:otp|pin|cvv)\b",
)]


@dataclass
class Verdict:
    action: str = "ok"           # ok | quarantine | reject
    flags: list = field(default_factory=list)
    reason: str = ""


def sanitize(text: str) -> str:
    return _CTRL.sub("", text).strip()


def moderate(title: str, content: str, max_urls: int = 3) -> Verdict:
    text = f"{title}\n{content}"
    v = Verdict()
    if len(_URL.findall(text)) > max_urls:
        return Verdict("reject", ["too_many_urls"], f"more than {max_urls} URLs")
    for rx in INJECTION:
        if rx.search(text):
            v.flags.append("prompt_injection")
            break
    for rx in SCAM:
        if rx.search(text):
            v.flags.append("scam_pattern")
            break
    letters = [c for c in content if c.isalpha()]
    if len(letters) > 40 and sum(c.isupper() for c in letters) / len(letters) > 0.7:
        v.flags.append("shouting")
    if any(f in v.flags for f in ("prompt_injection", "scam_pattern")):
        v.action, v.reason = "quarantine", "content matched a safety rule and is held for review"
    return v


_BAD_HOSTS = {"localhost", "metadata.google.internal", "metadata", "instance-data"}


def validate_webhook_url(url: str, allow_insecure: bool = False, allow_private: bool = False) -> str | None:
    """Returns an error string, or None if acceptable. Blocks SSRF to internal addresses (literal-IP / name checks).

    Residual risk (documented in README): a hostname that *resolves* to a private IP (DNS rebinding) is not caught
    here; production must resolve at send time and connect to the vetted IP.
    """
    try:
        u = urlparse(url)
    except ValueError:
        return "invalid URL"
    if u.scheme not in ("https", "http") or not u.hostname:
        return "webhook_url must be an absolute http(s) URL"
    if u.scheme == "http" and not allow_insecure:
        return "webhook_url must use https"
    if u.username or u.password:
        return "credentials in URL are not allowed"
    host = u.hostname.lower()
    if allow_private:
        return None
    if host in _BAD_HOSTS or host.endswith((".local", ".internal", ".localhost")):
        return "webhook host is not allowed"
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return None
    if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast or ip.is_unspecified:
        return "webhook IP address is not allowed"
    return None
