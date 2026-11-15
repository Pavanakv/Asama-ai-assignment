"""In-process 'internet': maps hostnames to ASGI apps so the platform and every company agent can talk over real
HTTP semantics (httpx -> ASGI) without opening sockets. Used by the demo, the benchmark and the tests."""
from __future__ import annotations

import httpx


class LocalNetwork(httpx.AsyncBaseTransport):
    def __init__(self) -> None:
        self.hosts: dict = {}
        self.requests: list = []      # (host, path, headers, body) of every request - handy for tests and the demo

    def mount(self, host: str, app) -> None:
        self.hosts[host] = httpx.ASGITransport(app=app)

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.requests.append((request.url.host, request.url.path, dict(request.headers), request.content))
        t = self.hosts.get(request.url.host)
        if t is None:
            raise httpx.ConnectError(f"no route to host {request.url.host}")
        return await t.handle_async_request(request)

    def client(self, **kw) -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=self, **kw)
