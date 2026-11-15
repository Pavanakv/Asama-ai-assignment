"""Boots a complete in-process network: the platform + one runtime per persona, registered and verified."""
from __future__ import annotations

from dataclasses import dataclass, field

import httpx

from app.config import Settings
from app.main import create_app

from .local import LocalNetwork
from .personas import PERSONAS, Persona
from .runtime import AgentRuntime

PLATFORM = "http://platform.test"


@dataclass
class World:
    app: object
    net: LocalNetwork
    client: httpx.AsyncClient          # an anonymous client for the platform
    runtimes: dict = field(default_factory=dict)   # name -> AgentRuntime
    admin: dict = field(default_factory=lambda: {"X-Admin-Token": "dev-admin-token"})

    def auth(self, name: str) -> dict:
        return {"Authorization": f"Bearer {self.runtimes[name].api_key}"}

    async def post(self, author: str, title: str, content: str, **kw) -> dict:
        r = await self.client.post("/v1/posts", json={"title": title, "content": content, **kw}, headers=self.auth(author))
        assert r.status_code == 201, r.text
        return r.json()

    async def settle(self) -> None:
        """Deliver every webhook and let every agent finish responding."""
        for _ in range(5):
            await self.app.state.dispatcher.drain()
            for rt in self.runtimes.values():
                await rt.idle()

    async def register(self, persona: Persona, responder=None, verify: bool = True) -> AgentRuntime:
        rt = AgentRuntime(persona, PLATFORM, self.net.client(), responder)
        self.net.mount(persona.host, rt.app)
        r = await self.client.post("/v1/agents/register", json={"name": persona.name, "industry": persona.industry, "description": persona.description,
                                                                 "topics": persona.topics, "webhook_url": f"http://{persona.host}/network-events"})
        assert r.status_code == 201, r.text
        j = r.json()
        rt.configure(j["agent_id"], j["api_key"], j["webhook_secret"])
        rt.client = self.net.client(base_url=PLATFORM)
        self.runtimes[persona.name] = rt
        if verify:
            assert (await self.client.post(f"/v1/admin/agents/{j['agent_id']}/verify", headers=self.admin)).status_code == 200
        return rt


async def build_world(settings: Settings | None = None, personas=None, responder_factory=None, judge=None, embedder=None) -> World:
    s = settings or Settings()
    s.allow_insecure_webhooks = True
    s.webhook_backoff_s = 0.0 if s.webhook_backoff_s == 0.5 else s.webhook_backoff_s
    net = LocalNetwork()
    app = create_app(s, http_client=net.client(), judge=judge, embedder=embedder)
    net.mount("platform.test", app)
    w = World(app, net, net.client(base_url=PLATFORM))
    for p in (personas or PERSONAS):
        await w.register(p, responder_factory(p) if responder_factory else None)
    return w
