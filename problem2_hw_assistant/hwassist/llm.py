"""LLM backends. Standard library only (urllib), so there is nothing to install.

* OllamaBackend        - local model, fully offline. Nothing leaves the machine.
* OpenAICompatBackend  - any OpenAI-compatible cloud endpoint (OpenAI, Groq, ...).
                         Tool results (hardware facts, no serials) are sent to it,
                         so the CLI only enables it with an explicit --allow-cloud.

Both speak one internal message format (OpenAI-style, tool-call arguments as dicts)
and adapt it to the wire format they need.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Optional


class BackendError(Exception):
    pass


@dataclass
class ToolCall:
    id: str
    name: str
    args: object  # dict normally; {"_raw": "..."} if the model sent unparseable JSON


@dataclass
class ModelReply:
    text: Optional[str]
    tool_calls: list = field(default_factory=list)


def _post_json(url: str, payload: dict, headers: dict, timeout: float) -> dict:
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json", **headers}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8", errors="replace"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")[:300]
        raise BackendError(f"HTTP {e.code} from {url}: {body}") from None
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        raise BackendError(f"could not reach {url}: {getattr(e, 'reason', e)}") from None
    except json.JSONDecodeError:
        raise BackendError(f"{url} returned invalid JSON") from None


class OllamaBackend:
    kind = "llm"

    def __init__(self, host: str = "http://127.0.0.1:11434", model: str = "qwen2.5:7b", timeout: float = 120.0) -> None:
        self.host, self.model, self.timeout = host.rstrip("/"), model, timeout
        self.name = f"ollama:{model} (local, offline)"

    def reachable(self) -> bool:
        try:
            with urllib.request.urlopen(self.host + "/api/tags", timeout=0.7):
                return True
        except Exception:
            return False

    def chat(self, messages: list, tools: list) -> ModelReply:
        wire = []
        for m in messages:
            if m["role"] == "assistant" and m.get("tool_calls"):
                wire.append({"role": "assistant", "content": m.get("content") or "",
                             "tool_calls": [{"function": {"name": c["function"]["name"], "arguments": c["function"]["arguments"]}} for c in m["tool_calls"]]})
            elif m["role"] == "tool":
                wire.append({"role": "tool", "content": m["content"], "tool_name": m.get("name", "")})
            else:
                wire.append({"role": m["role"], "content": m["content"]})
        payload = {"model": self.model, "messages": wire, "stream": False, "options": {"temperature": 0}}
        if tools:
            payload["tools"] = tools
        data = _post_json(self.host + "/api/chat", payload, {}, self.timeout)
        msg = data.get("message") or {}
        calls = []
        for i, c in enumerate(msg.get("tool_calls") or []):
            fn = c.get("function", {})
            args = fn.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {"_raw": args[:200]}
            calls.append(ToolCall(f"call_{i}", fn.get("name", ""), args))
        return ModelReply((msg.get("content") or "").strip() or None, calls)


class OpenAICompatBackend:
    kind = "llm"

    def __init__(self, api_key: str, model: str = "gpt-4o-mini", base_url: str = "https://api.openai.com/v1", timeout: float = 60.0) -> None:
        self.api_key, self.model, self.base_url, self.timeout = api_key, model, base_url.rstrip("/"), timeout
        host = base_url.split("//", 1)[-1].split("/", 1)[0]
        self.name = f"{model} via {host} (CLOUD)"

    def chat(self, messages: list, tools: list) -> ModelReply:
        wire = []
        for m in messages:
            if m["role"] == "assistant" and m.get("tool_calls"):
                wire.append({"role": "assistant", "content": m.get("content") or None,
                             "tool_calls": [{"id": c["id"], "type": "function",
                                             "function": {"name": c["function"]["name"], "arguments": json.dumps(c["function"]["arguments"])}} for c in m["tool_calls"]]})
            elif m["role"] == "tool":
                wire.append({"role": "tool", "tool_call_id": m["tool_call_id"], "content": m["content"]})
            else:
                wire.append({"role": m["role"], "content": m["content"]})
        payload = {"model": self.model, "messages": wire, "temperature": 0}
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        data = _post_json(self.base_url + "/chat/completions", payload, {"Authorization": f"Bearer {self.api_key}"}, self.timeout)
        try:
            msg = data["choices"][0]["message"]
        except (KeyError, IndexError, TypeError):
            raise BackendError("unexpected response shape from chat endpoint") from None
        calls = []
        for c in msg.get("tool_calls") or []:
            fn = c.get("function", {})
            raw = fn.get("arguments") or "{}"
            try:
                args = json.loads(raw) if isinstance(raw, str) else raw
            except json.JSONDecodeError:
                args = {"_raw": str(raw)[:200]}
            calls.append(ToolCall(c.get("id") or f"call_{len(calls)}", fn.get("name", ""), args))
        return ModelReply((msg.get("content") or "").strip() or None, calls)
