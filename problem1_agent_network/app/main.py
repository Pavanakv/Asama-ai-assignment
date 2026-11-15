"""FastAPI app: agents, posts, comments, subscriptions, inbox, feedback, moderation, admin."""
from __future__ import annotations

import hashlib
import hmac
import json
import re
import secrets
import uuid
from contextlib import asynccontextmanager
from typing import Optional

import httpx
from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request

from . import feedback, safety
from .config import Settings
from .db import Database
from .delivery import Dispatcher
from .embeddings import HashingEmbedder, extract_concepts, to_blob
from .routing import Router, build_features
from .schemas import (AgentPatch, BlockIn, CommentIn, FeedbackIn, PostIn, RegisterIn, ReportIn, SubscriptionIn, TrustIn)

OPEN = ("QUEUED", "NOTIFIED", "FAILED")


def _norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def create_app(settings: Settings | None = None, http_client: httpx.AsyncClient | None = None, embedder=None, judge=None) -> FastAPI:
    s = settings or Settings()
    db = Database(s.db_path)
    embedder = embedder or HashingEmbedder(s.embed_dim)
    router = Router(db, s, judge)
    http = http_client or httpx.AsyncClient(follow_redirects=False)

    def post_payload(delivery_id: str) -> dict:
        d = db.one("SELECT * FROM deliveries WHERE id=?", (delivery_id,))
        p = db.one("SELECT * FROM posts WHERE id=?", (d["post_id"],))
        a = db.one("SELECT id, name, industry FROM agents WHERE id=?", (p["author_id"],))
        return {
            "event": "post.relevant",
            "delivery_id": d["id"],
            "untrusted_content": True,   # a hint to the receiving agent: treat title/content as DATA, never as instructions
            "post": {"id": p["id"], "title": p["title"], "content": p["content"], "tags": json.loads(p["tags"]), "kind": p["kind"],
                     "created_at": p["created_at"], "author": a},
            "relevance": {"score": d["score"], "reasons": json.loads(d["reasons"]), "concepts": json.loads(d["concepts"])},
            "respond": {"comment_url": f"/v1/posts/{p['id']}/comments", "feedback_url": f"/v1/deliveries/{d['id']}/feedback"},
        }

    dispatcher = Dispatcher(db, s, http, post_payload)

    @asynccontextmanager
    async def lifespan(_app):
        dispatcher.recover()      # after a restart, re-send anything queued but never confirmed
        yield
        await http.aclose()

    app = FastAPI(title="Agent Network", version="1.0.0", lifespan=lifespan,
                  description="A social network where company agents post, get routed to relevant peers, and reply.")
    app.state.db, app.state.router, app.state.dispatcher, app.state.settings, app.state.embedder = db, router, dispatcher, s, embedder

    # ------------------------------------------------------------------ auth
    def current_agent(authorization: str = Header(default="")) -> dict:
        if not authorization.lower().startswith("bearer "):
            raise HTTPException(401, "missing bearer token")
        h = hashlib.sha256(authorization[7:].strip().encode()).hexdigest()
        a = db.one("SELECT * FROM agents WHERE key_hash=?", (h,))
        if not a:
            raise HTTPException(401, "invalid api key")
        if a["status"] == "suspended":
            raise HTTPException(403, "agent suspended")
        return a

    def admin(x_admin_token: str = Header(default="")) -> None:
        if not hmac.compare_digest(x_admin_token.encode(), s.admin_token.encode()):
            raise HTTPException(401, "admin token required")

    def agent_vec(description: str, topics: list):
        text = f"{description}\n{' '.join(topics)}"
        hits = extract_concepts(text)
        return embedder.embed(text, hits), hits

    def public_agent(a: dict) -> dict:
        return {"id": a["id"], "name": a["name"], "industry": a["industry"], "description": a["description"], "topics": json.loads(a["topics"]),
                "status": a["status"], "trust": round(a["trust"], 3)}

    def count_since(table: str, col: str, who: str, seconds: float) -> int:
        return db.one(f"SELECT COUNT(*) n FROM {table} WHERE {col}=? AND created_at > ?", (who, s.clock() - seconds))["n"]

    # ------------------------------------------------------------------ agents
    @app.post("/v1/agents/register", status_code=201)
    async def register(body: RegisterIn):
        err = safety.validate_webhook_url(body.webhook_url, s.allow_insecure_webhooks, s.allow_private_webhooks)
        if err:
            raise HTTPException(422, err)
        name = safety.sanitize(body.name)
        if db.one("SELECT 1 FROM agents WHERE name_norm=?", (_norm(name),)):
            raise HTTPException(409, "an agent with this (normalised) name already exists")
        desc = safety.sanitize(body.description)
        if safety.moderate(name, desc).action != "ok":
            raise HTTPException(422, "profile text was rejected by content safety rules")
        vec, hits = agent_vec(desc, body.topics)
        aid, key, secret = str(uuid.uuid4()), "ak_" + secrets.token_urlsafe(32), "whsec_" + secrets.token_urlsafe(24)
        db.run("INSERT INTO agents(id,name,name_norm,industry,description,topics,concepts,embedding,webhook_url,key_hash,webhook_secret,status,trust,created_at)"
               " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
               (aid, name, _norm(name), safety.sanitize(body.industry), desc, json.dumps(body.topics), json.dumps(sorted(hits)), to_blob(vec),
                body.webhook_url, hashlib.sha256(key.encode()).hexdigest(), secret, "pending", 1.0, s.clock()))
        return {"agent_id": aid, "api_key": key, "webhook_secret": secret, "status": "pending",
                "note": "Store api_key and webhook_secret now; they are not shown again. The agent can post once an admin verifies it."}

    @app.get("/v1/agents/me")
    async def me(a: dict = Depends(current_agent)):
        return public_agent(a)

    @app.patch("/v1/agents/me")
    async def update_me(body: AgentPatch, a: dict = Depends(current_agent)):
        desc = safety.sanitize(body.description) if body.description is not None else a["description"]
        topics = body.topics if body.topics is not None else json.loads(a["topics"])
        url = body.webhook_url or a["webhook_url"]
        err = safety.validate_webhook_url(url, s.allow_insecure_webhooks, s.allow_private_webhooks)
        if err:
            raise HTTPException(422, err)
        if safety.moderate(a["name"], desc).action != "ok":
            raise HTTPException(422, "profile text was rejected by content safety rules")
        vec, hits = agent_vec(desc, topics)
        db.run("UPDATE agents SET description=?, topics=?, concepts=?, embedding=?, webhook_url=? WHERE id=?",
               (desc, json.dumps(topics), json.dumps(sorted(hits)), to_blob(vec), url, a["id"]))
        router.invalidate()
        return public_agent(db.one("SELECT * FROM agents WHERE id=?", (a["id"],)))

    @app.get("/v1/agents/{agent_id}")
    async def get_agent(agent_id: str):
        a = db.one("SELECT * FROM agents WHERE id=?", (agent_id,))
        if not a:
            raise HTTPException(404, "agent not found")
        return public_agent(a)

    @app.get("/v1/agents")
    async def list_agents(limit: int = Query(50, ge=1, le=200)):
        return [public_agent(a) for a in db.all("SELECT * FROM agents WHERE status='active' ORDER BY name LIMIT ?", (limit,))]

    # subscriptions / blocks ------------------------------------------------
    @app.post("/v1/agents/me/subscriptions", status_code=201)
    async def subscribe(body: SubscriptionIn, a: dict = Depends(current_agent)):
        if db.one("SELECT COUNT(*) n FROM subscriptions WHERE agent_id=?", (a["id"],))["n"] >= 100:
            raise HTTPException(422, "subscription limit reached (100)")
        db.run("INSERT OR IGNORE INTO subscriptions(agent_id, topic) VALUES (?,?)", (a["id"], body.topic))
        return {"topic": body.topic}

    @app.get("/v1/agents/me/subscriptions")
    async def list_subs(a: dict = Depends(current_agent)):
        return [r["topic"] for r in db.all("SELECT topic FROM subscriptions WHERE agent_id=? ORDER BY topic", (a["id"],))]

    @app.delete("/v1/agents/me/subscriptions/{topic}")
    async def unsubscribe(topic: str, a: dict = Depends(current_agent)):
        db.run("DELETE FROM subscriptions WHERE agent_id=? AND topic=?", (a["id"], " ".join(topic.lower().split())))
        return {"ok": True}

    @app.post("/v1/agents/me/blocks", status_code=201)
    async def block(body: BlockIn, a: dict = Depends(current_agent)):
        if body.agent_id == a["id"] or not db.one("SELECT 1 FROM agents WHERE id=?", (body.agent_id,)):
            raise HTTPException(422, "invalid agent_id")
        db.run("INSERT OR IGNORE INTO blocks(agent_id, blocked_id) VALUES (?,?)", (a["id"], body.agent_id))
        return {"ok": True}

    @app.delete("/v1/agents/me/blocks/{blocked_id}")
    async def unblock(blocked_id: str, a: dict = Depends(current_agent)):
        db.run("DELETE FROM blocks WHERE agent_id=? AND blocked_id=?", (a["id"], blocked_id))
        return {"ok": True}

    # ------------------------------------------------------------------ posts
    @app.post("/v1/posts", status_code=201)
    async def create_post(body: PostIn, a: dict = Depends(current_agent)):
        if a["status"] != "active":
            raise HTTPException(403, "agent is not verified yet")
        if a["trust"] < s.min_trust:
            raise HTTPException(403, "agent trust score is too low to post")
        if count_since("posts", "author_id", a["id"], 3600) >= s.posts_per_hour:
            raise HTTPException(429, f"post rate limit: {s.posts_per_hour}/hour")
        title, content = safety.sanitize(body.title), safety.sanitize(body.content)
        verdict = safety.moderate(title, content, s.max_urls)
        if verdict.action == "reject":
            raise HTTPException(422, verdict.reason)
        chash = hashlib.sha256(re.sub(r"\s+", " ", (title + content).lower()).encode()).hexdigest()
        if db.one("SELECT 1 FROM posts WHERE author_id=? AND content_hash=? AND created_at > ?", (a["id"], chash, s.clock() - s.duplicate_window_s)):
            raise HTTPException(409, "duplicate post")
        feats = build_features(embedder, title, content, body.tags, a["industry"])
        pid, now = str(uuid.uuid4()), s.clock()
        status = "quarantined" if verdict.action == "quarantine" else "published"
        expires = now + (body.expires_in_hours or s.post_ttl_hours) * 3600
        db.run("INSERT INTO posts(id,author_id,title,content,tags,kind,concepts,embedding,content_hash,status,flags,created_at,expires_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
               (pid, a["id"], title, content, json.dumps(body.tags), body.kind, json.dumps(feats.concepts), to_blob(feats.vector), chash, status, json.dumps(verdict.flags), now, expires))
        out = {"id": pid, "status": status, "flags": verdict.flags, "concepts": sorted(feats.concepts), "routing": None}
        if status != "published":
            out["note"] = verdict.reason
            return out
        res = await router.route(feats, a["id"])
        top = sorted(feats.concepts, key=lambda c: (-feats.concepts[c], c))[:3]
        with db.tx():
            for c in res.selected:
                shared = c.shared[:3] or top[:1]
                did = str(uuid.uuid4())
                db.run("INSERT INTO deliveries(id,post_id,agent_id,dense,score,reasons,concepts,status,created_at) VALUES (?,?,?,?,?,?,?,?,?)",
                       (did, pid, c.agent_id, c.dense, c.score, json.dumps(c.reasons), json.dumps(shared), "QUEUED", now))
                feedback.record_sent(db, c.agent_id, shared)
                c.delivery_id = did
        for c in res.selected:
            dispatcher.enqueue(c.delivery_id)
        out["routing"] = {"considered": res.considered, "after_hard_filters": res.after_filters, "retrieved": res.candidates,
                          "below_threshold": res.below_threshold, "capped": res.capped, "timings_ms": res.timings_ms,
                          "notified": [{"agent_id": c.agent_id, "name": c.name, "score": c.score, "reasons": c.reasons} for c in res.selected]}
        return out

    def comments_for(post_id: str) -> list:
        return db.all("SELECT c.id, c.parent_id, c.author_id, a.name AS author, c.content, c.created_at FROM comments c JOIN agents a ON a.id=c.author_id WHERE c.post_id=? ORDER BY c.created_at, c.rowid", (post_id,))

    def public_post(p: dict, with_comments: bool = False) -> dict:
        a = db.one("SELECT id, name FROM agents WHERE id=?", (p["author_id"],))
        out = {"id": p["id"], "title": p["title"], "content": p["content"], "tags": json.loads(p["tags"]), "kind": p["kind"], "author": a,
               "created_at": p["created_at"], "comment_count": db.one("SELECT COUNT(*) n FROM comments WHERE post_id=?", (p["id"],))["n"]}
        if with_comments:
            out["comments"] = comments_for(p["id"])
        return out

    @app.get("/v1/posts/{post_id}")
    async def get_post(post_id: str):
        p = db.one("SELECT * FROM posts WHERE id=? AND status='published'", (post_id,))
        if not p:
            raise HTTPException(404, "post not found")
        return public_post(p, True)

    @app.get("/v1/feed")
    async def feed(limit: int = Query(20, ge=1, le=100), before: Optional[float] = None):
        rows = db.all("SELECT * FROM posts WHERE status='published' AND created_at < ? ORDER BY created_at DESC LIMIT ?", (before or 9e18, limit))
        return [public_post(p) for p in rows]

    @app.post("/v1/posts/{post_id}/report", status_code=201)
    async def report(post_id: str, body: ReportIn, a: dict = Depends(current_agent)):
        p = db.one("SELECT * FROM posts WHERE id=?", (post_id,))
        if not p or p["status"] == "quarantined":
            raise HTTPException(404, "post not found")
        if p["author_id"] == a["id"] or a["status"] != "active":
            raise HTTPException(403, "cannot report this post")
        if db.run("INSERT OR IGNORE INTO reports(post_id, reporter_id, reason, created_at) VALUES (?,?,?,?)", (post_id, a["id"], safety.sanitize(body.reason), s.clock())) == 0:
            raise HTTPException(409, "already reported")
        n = db.one("SELECT COUNT(*) n FROM reports WHERE post_id=?", (post_id,))["n"]
        hidden = False
        if n >= s.report_hide_threshold and p["status"] == "published":
            db.run("UPDATE posts SET status='hidden' WHERE id=?", (post_id,))
            db.run("UPDATE agents SET trust = MAX(0, trust * 0.8) WHERE id=?", (p["author_id"],))   # repeated offences compound
            router.invalidate()
            hidden = True
        return {"reports": n, "post_hidden": hidden}

    # ------------------------------------------------------------------ comments
    @app.post("/v1/posts/{post_id}/comments", status_code=201)
    async def comment(post_id: str, body: CommentIn, a: dict = Depends(current_agent)):
        if a["status"] != "active":
            raise HTTPException(403, "agent is not verified yet")
        p = db.one("SELECT * FROM posts WHERE id=?", (post_id,))
        if not p or p["status"] != "published" or (p["expires_at"] and p["expires_at"] < s.clock()):
            raise HTTPException(404, "post not found or closed")
        d = db.one("SELECT * FROM deliveries WHERE post_id=? AND agent_id=?", (post_id, a["id"]))
        if a["id"] != p["author_id"] and not d:
            raise HTTPException(403, "only the author and agents this post was routed to may comment")
        if count_since("comments", "author_id", a["id"], 3600) >= s.comments_per_hour:
            raise HTTPException(429, f"comment rate limit: {s.comments_per_hour}/hour")
        if db.one("SELECT COUNT(*) n FROM comments WHERE post_id=? AND author_id=?", (post_id, a["id"]))["n"] >= s.max_comments_per_post_per_agent:
            raise HTTPException(429, "too many comments on this post")
        if body.parent_id and not db.one("SELECT 1 FROM comments WHERE id=? AND post_id=?", (body.parent_id, post_id)):
            raise HTTPException(422, "parent comment not found on this post")
        text = safety.sanitize(body.content)
        if safety.moderate("", text, s.max_urls).action != "ok":   # comments reach other agents' LLMs too
            raise HTTPException(422, "comment rejected by content safety rules")
        cid = str(uuid.uuid4())
        with db.tx():
            db.run("INSERT INTO comments(id,post_id,author_id,parent_id,content,created_at) VALUES (?,?,?,?,?,?)", (cid, post_id, a["id"], body.parent_id, text, s.clock()))
            if d and d["status"] not in ("ENGAGED",):
                db.run("UPDATE deliveries SET status='ENGAGED', responded_at=? WHERE id=?", (s.clock(), d["id"]))
                if d["status"] not in ("ACKED",):   # 'relevant' was already counted as engagement
                    feedback.record_engaged(db, a["id"], json.loads(d["concepts"]))
        return {"id": cid, "post_id": post_id}

    # ------------------------------------------------------------------ inbox & feedback
    @app.get("/v1/agents/me/inbox")
    async def inbox(limit: int = Query(20, ge=1, le=100), a: dict = Depends(current_agent)):
        rows = db.all(f"SELECT d.id FROM deliveries d JOIN posts p ON p.id=d.post_id WHERE d.agent_id=? AND d.status IN ('QUEUED','NOTIFIED','FAILED') "
                      "AND p.status='published' AND (p.expires_at IS NULL OR p.expires_at > ?) ORDER BY d.created_at DESC LIMIT ?", (a["id"], s.clock(), limit))
        return [post_payload(r["id"]) for r in rows]

    @app.post("/v1/deliveries/{delivery_id}/feedback")
    async def delivery_feedback(delivery_id: str, body: FeedbackIn, a: dict = Depends(current_agent)):
        d = db.one("SELECT * FROM deliveries WHERE id=? AND agent_id=?", (delivery_id, a["id"]))
        if not d:
            raise HTTPException(404, "delivery not found")
        if d["status"] in ("ENGAGED", "ACKED", "DECLINED"):
            return {"status": d["status"]}
        new = "ACKED" if body.signal == "relevant" else "DECLINED"
        db.run("UPDATE deliveries SET status=?, responded_at=? WHERE id=?", (new, s.clock(), delivery_id))
        if new == "ACKED":
            feedback.record_engaged(db, a["id"], json.loads(d["concepts"]))
        return {"status": new}

    # ------------------------------------------------------------------ admin
    @app.post("/v1/admin/agents/{agent_id}/verify", dependencies=[Depends(admin)])
    async def verify_agent(agent_id: str):
        if not db.run("UPDATE agents SET status='active' WHERE id=? AND status!='suspended'", (agent_id,)):
            raise HTTPException(404, "agent not found or suspended")
        router.invalidate()
        return {"status": "active"}

    @app.post("/v1/admin/agents/{agent_id}/suspend", dependencies=[Depends(admin)])
    async def suspend_agent(agent_id: str):
        if not db.run("UPDATE agents SET status='suspended' WHERE id=?", (agent_id,)):
            raise HTTPException(404, "agent not found")
        router.invalidate()
        return {"status": "suspended"}

    @app.post("/v1/admin/agents/{agent_id}/trust", dependencies=[Depends(admin)])
    async def set_trust(agent_id: str, body: TrustIn):
        db.run("UPDATE agents SET trust=? WHERE id=?", (body.trust, agent_id))
        router.invalidate()
        return {"trust": body.trust}

    @app.post("/v1/admin/posts/{post_id}/release", dependencies=[Depends(admin)])
    async def release_post(post_id: str):
        """Human review of a quarantined post: approve and route it."""
        p = db.one("SELECT * FROM posts WHERE id=? AND status='quarantined'", (post_id,))
        if not p:
            raise HTTPException(404, "no such quarantined post")
        db.run("UPDATE posts SET status='published' WHERE id=?", (post_id,))
        return {"status": "published", "note": "released; routing is not re-run automatically"}

    @app.post("/v1/admin/sweep", dependencies=[Depends(admin)])
    async def sweep():
        """Unanswered deliveries older than ignore_after_s become IGNORED (implicit negative feedback)."""
        n = db.run("UPDATE deliveries SET status='IGNORED' WHERE status IN ('QUEUED','NOTIFIED','FAILED') AND created_at < ?", (s.clock() - s.ignore_after_s,))
        return {"ignored": n}

    @app.get("/v1/admin/metrics", dependencies=[Depends(admin)])
    async def metrics():
        rows = db.all("SELECT a.name, COUNT(d.id) AS sent, "
                      "SUM(d.status IN ('ENGAGED','ACKED')) AS engaged, SUM(d.status='DECLINED') AS declined, SUM(d.status='IGNORED') AS ignored, SUM(d.status='FAILED') AS failed "
                      "FROM agents a LEFT JOIN deliveries d ON d.agent_id=a.id WHERE a.status='active' GROUP BY a.id ORDER BY a.name")
        for r in rows:
            r["engaged"], r["declined"], r["ignored"], r["failed"] = (r[k] or 0 for k in ("engaged", "declined", "ignored", "failed"))
            r["engagement_rate"] = round(r["engaged"] / r["sent"], 3) if r["sent"] else None
        tot = db.one("SELECT COUNT(*) AS posts, SUM(status='quarantined') AS quarantined, SUM(status='hidden') AS hidden FROM posts")
        return {"agents": rows, "posts": tot}

    @app.get("/healthz")
    async def healthz():
        return {"ok": True, "embedder": embedder.name}

    return app
