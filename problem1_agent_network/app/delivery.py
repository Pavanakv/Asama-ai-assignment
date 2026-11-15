"""Webhook delivery: signed POSTs, retries with exponential backoff, per-agent circuit breaker.

Everything is also available by polling GET /v1/agents/me/inbox, so a slow or dead webhook never loses a
notification - it only loses the push. A delivery is durable in the DB *before* the first send attempt.

Scale note: this in-process asyncio dispatcher is the single-node stand-in for a durable queue + worker pool
(Redis Streams / SQS / Celery). The state machine and idempotency (delivery_id) carry over unchanged.
"""
from __future__ import annotations

import asyncio
import json

from . import signing


class Dispatcher:
    def __init__(self, db, settings, http_client, payload_builder) -> None:
        self.db, self.s, self.http = db, settings, http_client
        self.build_payload = payload_builder
        self._tasks: set = set()
        self._sem = asyncio.Semaphore(settings.max_concurrent_deliveries)

    def enqueue(self, delivery_id: str) -> None:
        t = asyncio.get_running_loop().create_task(self._deliver(delivery_id))
        self._tasks.add(t)
        t.add_done_callback(self._tasks.discard)

    async def drain(self) -> None:
        """Wait until all queued deliveries (and anything they trigger) are finished. Used by tests and the demo."""
        # Filter on .done() rather than waiting for the discard callback: asyncio.gather() completes *eagerly* (without
        # yielding) when every task is already finished, so `while self._tasks: await gather(...)` can spin forever.
        while True:
            pending = [t for t in self._tasks if not t.done()]
            if not pending:
                self._tasks.difference_update([t for t in self._tasks if t.done()])
                return
            await asyncio.wait(pending)

    def recover(self) -> int:
        """After a restart: re-send anything that was queued but never confirmed."""
        rows = self.db.all("SELECT id FROM deliveries WHERE status='QUEUED'")
        for r in rows:
            self.enqueue(r["id"])
        return len(rows)

    async def _deliver(self, delivery_id: str) -> None:
        async with self._sem:
            d = self.db.one("SELECT * FROM deliveries WHERE id=?", (delivery_id,))
            if not d or d["status"] != "QUEUED":
                return
            agent = self.db.one("SELECT * FROM agents WHERE id=?", (d["agent_id"],))
            now = self.s.clock()
            if agent["breaker_until"] > now:
                self.db.run("UPDATE deliveries SET last_error=? WHERE id=?", ("circuit open; available via inbox", delivery_id))
                return
            body = json.dumps(self.build_payload(delivery_id), separators=(",", ":"), ensure_ascii=False).encode()
            err = ""
            for attempt in range(1, self.s.webhook_attempts + 1):
                headers = {"Content-Type": "application/json", signing.HEADER: signing.sign(agent["webhook_secret"], body, int(self.s.clock())),
                           "X-Network-Delivery": delivery_id}
                try:
                    r = await self.http.post(agent["webhook_url"], content=body, headers=headers, timeout=self.s.webhook_timeout_s)
                    if 200 <= r.status_code < 300:
                        self.db.run("UPDATE deliveries SET status='NOTIFIED', attempts=?, notified_at=?, last_error=NULL WHERE id=?", (attempt, self.s.clock(), delivery_id))
                        self.db.run("UPDATE agents SET consecutive_failures=0, breaker_until=0 WHERE id=?", (agent["id"],))
                        return
                    err = f"HTTP {r.status_code}"
                    if 400 <= r.status_code < 500 and r.status_code not in (408, 425, 429):
                        break  # the agent rejected it; retrying won't help
                except Exception as e:  # network error / timeout
                    err = f"{type(e).__name__}: {str(e)[:120]}"
                if attempt < self.s.webhook_attempts:
                    await asyncio.sleep(self.s.webhook_backoff_s * 2 ** (attempt - 1))
            fails = agent["consecutive_failures"] + 1
            until = self.s.clock() + self.s.breaker_cooldown_s if fails >= self.s.breaker_failures else 0
            self.db.run("UPDATE deliveries SET status='FAILED', attempts=?, last_error=? WHERE id=?", (attempt, err, delivery_id))
            self.db.run("UPDATE agents SET consecutive_failures=?, breaker_until=? WHERE id=?", (fails, until, agent["id"]))
