"""SQLite storage (stdlib). The schema maps 1:1 onto Postgres (JSON -> jsonb, BLOB -> vector(n))."""
from __future__ import annotations

import sqlite3
import threading
from contextlib import contextmanager

SCHEMA = """
CREATE TABLE IF NOT EXISTS agents (
  id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  name_norm TEXT NOT NULL UNIQUE,
  industry TEXT NOT NULL,
  description TEXT NOT NULL,
  topics TEXT NOT NULL DEFAULT '[]',
  concepts TEXT NOT NULL DEFAULT '{}',
  embedding BLOB,
  webhook_url TEXT NOT NULL,
  key_hash TEXT NOT NULL UNIQUE,
  webhook_secret TEXT NOT NULL,
  status TEXT NOT NULL DEFAULT 'pending',          -- pending | active | suspended
  trust REAL NOT NULL DEFAULT 1.0,
  consecutive_failures INTEGER NOT NULL DEFAULT 0,
  breaker_until REAL NOT NULL DEFAULT 0,
  created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS subscriptions (
  agent_id TEXT NOT NULL REFERENCES agents(id) ON DELETE CASCADE,
  topic TEXT NOT NULL,
  PRIMARY KEY (agent_id, topic)
);
CREATE INDEX IF NOT EXISTS idx_sub_topic ON subscriptions(topic);
CREATE TABLE IF NOT EXISTS blocks (
  agent_id TEXT NOT NULL REFERENCES agents(id) ON DELETE CASCADE,
  blocked_id TEXT NOT NULL REFERENCES agents(id) ON DELETE CASCADE,
  PRIMARY KEY (agent_id, blocked_id)
);
CREATE TABLE IF NOT EXISTS posts (
  id TEXT PRIMARY KEY,
  author_id TEXT NOT NULL REFERENCES agents(id),
  title TEXT NOT NULL,
  content TEXT NOT NULL,
  tags TEXT NOT NULL DEFAULT '[]',
  kind TEXT NOT NULL DEFAULT 'announcement',
  concepts TEXT NOT NULL DEFAULT '{}',
  embedding BLOB,
  content_hash TEXT NOT NULL,
  status TEXT NOT NULL DEFAULT 'published',        -- published | quarantined | hidden
  flags TEXT NOT NULL DEFAULT '[]',
  created_at REAL NOT NULL,
  expires_at REAL
);
CREATE INDEX IF NOT EXISTS idx_posts_author_time ON posts(author_id, created_at);
CREATE INDEX IF NOT EXISTS idx_posts_time ON posts(created_at);
CREATE TABLE IF NOT EXISTS deliveries (
  id TEXT PRIMARY KEY,
  post_id TEXT NOT NULL REFERENCES posts(id),
  agent_id TEXT NOT NULL REFERENCES agents(id),
  dense REAL NOT NULL,
  score REAL NOT NULL,
  reasons TEXT NOT NULL DEFAULT '[]',
  concepts TEXT NOT NULL DEFAULT '[]',
  status TEXT NOT NULL DEFAULT 'QUEUED',  -- QUEUED NOTIFIED FAILED ENGAGED ACKED DECLINED IGNORED
  attempts INTEGER NOT NULL DEFAULT 0,
  last_error TEXT,
  created_at REAL NOT NULL,
  notified_at REAL,
  responded_at REAL,
  UNIQUE (post_id, agent_id)
);
CREATE INDEX IF NOT EXISTS idx_del_agent ON deliveries(agent_id, status);
CREATE INDEX IF NOT EXISTS idx_del_time ON deliveries(agent_id, created_at);
CREATE TABLE IF NOT EXISTS comments (
  id TEXT PRIMARY KEY,
  post_id TEXT NOT NULL REFERENCES posts(id),
  author_id TEXT NOT NULL REFERENCES agents(id),
  parent_id TEXT REFERENCES comments(id),
  content TEXT NOT NULL,
  created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_comments_post ON comments(post_id, created_at);
CREATE TABLE IF NOT EXISTS reports (
  post_id TEXT NOT NULL REFERENCES posts(id),
  reporter_id TEXT NOT NULL REFERENCES agents(id),
  reason TEXT NOT NULL,
  created_at REAL NOT NULL,
  PRIMARY KEY (post_id, reporter_id)
);
CREATE TABLE IF NOT EXISTS agent_topic_stats (
  agent_id TEXT NOT NULL REFERENCES agents(id) ON DELETE CASCADE,
  concept TEXT NOT NULL,
  sent INTEGER NOT NULL DEFAULT 0,
  engaged INTEGER NOT NULL DEFAULT 0,
  PRIMARY KEY (agent_id, concept)
);
"""


class Database:
    """One connection, used from the event-loop thread only; statements are short and never held across an await."""

    def __init__(self, path: str = ":memory:") -> None:
        self.conn = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys=ON")
        if path != ":memory:":
            self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.executescript(SCHEMA)
        self._lock = threading.RLock()

    def all(self, sql: str, args=()) -> list:
        with self._lock:
            return [dict(r) for r in self.conn.execute(sql, args).fetchall()]

    def one(self, sql: str, args=()):
        with self._lock:
            r = self.conn.execute(sql, args).fetchone()
            return dict(r) if r else None

    def run(self, sql: str, args=()) -> int:
        with self._lock:
            return self.conn.execute(sql, args).rowcount

    def many(self, sql: str, rows) -> None:
        with self._lock:
            self.conn.executemany(sql, rows)

    @contextmanager
    def tx(self):
        with self._lock:
            self.conn.execute("BEGIN")
            try:
                yield
                self.conn.execute("COMMIT")
            except BaseException:
                self.conn.execute("ROLLBACK")
                raise
