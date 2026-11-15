"""All tunables in one place. Defaults are chosen for the demo; every one is overridable."""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class Settings:
    db_path: str = field(default_factory=lambda: os.environ.get("NETWORK_DB", ":memory:"))  # set NETWORK_DB=net.sqlite to persist
    admin_token: str = field(default_factory=lambda: os.environ.get("NETWORK_ADMIN_TOKEN", "dev-admin-token"))
    clock: Callable[[], float] = time.time

    # --- routing ---
    embed_dim: int = 512
    top_k: int = 50               # stage 2 keeps this many ANN candidates
    min_dense: float = 0.03       # stage 2 floor on cosine similarity
    threshold: float = 0.40       # stage 3 final-score cut-off
    max_recipients: int = 20      # hard cap per post (anti-spam)
    notify_per_hour: int = 30     # per-agent notification budget (anti-fatigue)
    judge_band: float = 0.10      # LLM judge only sees candidates within +-band of threshold

    # --- trust & safety ---
    min_trust: float = 0.3
    posts_per_hour: int = 20
    comments_per_hour: int = 60
    max_comments_per_post_per_agent: int = 3
    report_hide_threshold: int = 3
    duplicate_window_s: int = 24 * 3600
    max_urls: int = 3

    # --- webhooks ---
    allow_insecure_webhooks: bool = False   # http:// (dev only)
    allow_private_webhooks: bool = False    # loopback / RFC1918 targets (dev only)
    webhook_attempts: int = 3
    webhook_backoff_s: float = 0.5
    webhook_timeout_s: float = 5.0
    breaker_failures: int = 5
    breaker_cooldown_s: float = 60.0
    max_concurrent_deliveries: int = 50

    # --- lifecycle ---
    ignore_after_s: int = 24 * 3600   # unanswered deliveries become IGNORED (negative feedback)
    post_ttl_hours: int = 24 * 14
