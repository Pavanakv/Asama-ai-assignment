"""Command-line chat interface."""
from __future__ import annotations

import argparse
import os
import platform
import sys

from . import __version__
from .agent import Assistant
from .fixtures import FIXTURES, FixtureRunner
from .llm import BackendError, OllamaBackend, OpenAICompatBackend
from .rules import CAPABILITIES, RulesBackend
from .runner import COMMANDS, SystemRunner, sudoers_snippet
from .tools import Registry

HELP = """Ask about this laptop's hardware in plain language, e.g.
  How much RAM do I have?        Can I add more RAM, and how much?
  Is my SSD NVMe?                And what about the GPU?
Commands: /tools  /doctor  /debug  /clear  /help  /quit"""


def build_backend(args):
    if args.backend == "rules":
        return RulesBackend(), None
    if args.backend == "openai":
        if not args.allow_cloud:
            sys.exit("Refusing to use a cloud model without --allow-cloud (hardware facts would be sent to the provider).")
        key = os.environ.get("HWASSIST_API_KEY")
        if not key:
            sys.exit("Set HWASSIST_API_KEY (and optionally HWASSIST_BASE_URL, HWASSIST_MODEL).")
        return OpenAICompatBackend(key, os.environ.get("HWASSIST_MODEL", "gpt-4o-mini"),
                                   os.environ.get("HWASSIST_BASE_URL", "https://api.openai.com/v1")), None
    ollama = OllamaBackend(os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434"), args.model or os.environ.get("HWASSIST_MODEL", "qwen2.5:7b"))
    if ollama.reachable():
        return ollama, None
    if args.backend == "ollama":
        sys.exit(f"Ollama is not reachable at {ollama.host}. Start it (`ollama serve`) and pull a model (`ollama pull {ollama.model}`).")
    return RulesBackend(), "No local LLM found (Ollama not running) - using the built-in rules. Install Ollama for free-form questions."


def doctor(runner, registry, backend) -> str:
    lines = [f"hwassist {__version__}  |  python {platform.python_version()}  |  {platform.system()} {platform.release()}",
             f"data source : {'recorded demo machine' if isinstance(runner, FixtureRunner) else 'this machine (live, read-only)'}",
             f"backend     : {backend.name}", f"privilege   : {runner.privilege_state()}  (dmidecode slot/max-RAM data needs 'root' or 'sudo')"]
    lines.append("commands    :")
    for key in COMMANDS:
        lines.append(f"   {'ok     ' if runner.available(key) else 'MISSING'} {key}")
    lines.append(f"tools ready : {', '.join(t.name for t in registry.tools)}")
    if registry.unavailable:
        lines.append(f"tools hidden: {', '.join(registry.unavailable)}  (system utility not installed)")
    if runner.privilege_state() == "none":
        lines.append("hint        : run `python -m hwassist --print-sudoers` to enable exact-RAM-slot answers safely.")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="hwassist", description="Ask questions about this Linux laptop's hardware.")
    ap.add_argument("--backend", choices=["auto", "ollama", "openai", "rules"], default="auto", help="auto = local Ollama if running, else built-in rules")
    ap.add_argument("--model", help="model name for Ollama (default qwen2.5:7b)")
    ap.add_argument("--allow-cloud", action="store_true", help="permit --backend openai (sends hardware facts to the provider)")
    ap.add_argument("--demo", nargs="?", const="demo_laptop", choices=sorted(FIXTURES), help="use a recorded machine instead of this one")
    ap.add_argument("--once", metavar="QUESTION", help="answer one question and exit")
    ap.add_argument("--debug", action="store_true", help="show which tools were called")
    ap.add_argument("--print-sudoers", action="store_true", help="print the least-privilege sudoers rule and exit")
    ap.add_argument("--version", action="version", version=f"hwassist {__version__}")
    args = ap.parse_args(argv)

    if args.print_sudoers:
        print(sudoers_snippet())
        return 0

    runner = FixtureRunner(FIXTURES[args.demo]) if args.demo else SystemRunner()
    if not args.demo and platform.system() != "Linux":
        print("This tool inspects Linux hardware. Use --demo to try it with a recorded laptop.", file=sys.stderr)
        return 2
    registry = Registry(runner)
    backend, notice = build_backend(args)
    assistant = Assistant(backend, registry, runner.privilege_state())
    fallback = RulesBackend()
    debug = args.debug

    def respond(q: str) -> None:
        try:
            ans = assistant.ask(q)
        except BackendError as e:
            print(f"  ! {backend.name} failed ({e}); answering with built-in rules instead.", file=sys.stderr)
            ans = assistant.ask(q, backend=fallback)
        if debug and ans.tool_calls:
            print("  [tools] " + ", ".join(f"{n}({', '.join(f'{k}={v}' for k, v in a.items())})" + ("" if ok else " FAILED") for n, a, ok in ans.tool_calls))
        print(ans.text)
        for w in ans.warnings:
            print(f"  ! {w}")

    if notice:
        print(f"  note: {notice}", file=sys.stderr)
    if getattr(backend, "kind", "") == "llm" and "CLOUD" in backend.name:
        print(f"  ! Cloud mode: hardware facts from tool results are sent to {backend.base_url}. Serial numbers/UUIDs/hostname are never collected.", file=sys.stderr)

    if args.once:
        respond(args.once)
        return 0

    print(f"hwassist {__version__} - {backend.name} - {'demo machine' if args.demo else 'this laptop'}\n{HELP}\n")
    while True:
        try:
            q = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not q:
            continue
        if q in ("/quit", "/exit", "quit", "exit"):
            return 0
        if q == "/help":
            print(HELP)
        elif q == "/tools":
            for t in registry.tools:
                print(f"  {t.name}: {t.description}")
        elif q == "/doctor":
            print(doctor(runner, registry, backend))
        elif q == "/debug":
            debug = not debug
            print(f"  debug {'on' if debug else 'off'}")
        elif q == "/clear":
            assistant.clear()
            print("  conversation cleared")
        elif q.startswith("/"):
            print("  unknown command; try /help")
        else:
            respond(q)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
