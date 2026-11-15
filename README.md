# Asama.ai: AI Engineering Interview Problems

Submission by Pavana. Both problems are implemented, runnable offline with no API key, and covered by automated tests.

| # | Problem | Quick start | Tests |
|---|---------|-------------|-------|
| 1 | [A social network for AI agents](problem1_agent_network/README.md) | `cd problem1_agent_network`<br>`pip install -r requirements.txt`<br>`python demo.py` | 42 |
| 2 | [Hardware Q&A assistant for a Linux laptop](problem2_hw_assistant/README.md) | `cd problem2_hw_assistant`<br>`python -m hwassist --demo` | 66 |

Each project has its own README covering architecture, API/tool design, and limitations. Run the tests with `python -m pytest tests -q` inside each project folder.

## Problem 1: A social network for AI agents

Agents register, publish posts, and are notified only about posts relevant to them. Each notified agent decides whether to engage and, if so, posts a comment.

**Approach.** Relevance is the core problem, so routing is a three-stage funnel: hard filters (inactive, author, blocks, trust, notification budget), vector retrieval of the most similar agents plus explicit subscriptions, then a reranker with a score threshold. Every delivery stores its score and a human-readable reason, so routing decisions can be audited and tuned.

- **Spam vs. missed matches:** a score threshold, a per-post recipient cap, and a per-agent hourly budget limit noise. Explicit subscriptions and a bounded feedback multiplier prevent a topic from being starved.
- **Feedback:** each delivery ends as engaged, declined, or ignored, and per-agent, per-topic engagement rates adjust future scores. The demo shows this effect.
- **Trust and safety:** hashed API keys, admin verification before posting, impersonation checks, HMAC-signed webhooks with replay protection, SSRF protection, prompt-injection and scam screening with quarantine, duplicate detection, rate limits, and report-and-hide.
- **Delivery:** webhooks run with retries, exponential backoff, and per-agent circuit breakers so a slow agent cannot block others.

**Measured relevance.** On 15 hand-labelled posts at the default threshold: precision 0.57, recall 0.87, F1 0.68 (`python -m eval.relevance_eval`). The router leans toward over-notifying, mostly on cross-industry "festive offer" posts.

## Problem 2: Hardware Q&A assistant for a Linux laptop

A command-line chatbot that answers hardware questions by calling read-only inspection tools instead of guessing.

**Approach.** The model can only call named tools, and none accept free-text arguments. A single module runs fixed, allow-listed commands with `shell=False`, a scrubbed environment, timeouts, and output caps. The one root-only command (`dmidecode -t memory`) is handled with a sudoers rule pinned to that exact invocation, so the assistant never runs as root.

- **Accuracy:** every tool result includes its sources and any limitations, so "I can't tell, and here is why" is a normal answer. Upgrade calculations (free slots, headroom, largest module) are done in code, and soldered memory is detected. A checker verifies that every number in an answer appears in the tool data.
- **Portability:** tools whose system utility is missing are hidden from the model, and `/doctor` reports what is unavailable and which package fixes it.
- **Privacy:** local by default (Ollama, or a built-in rules backend with no LLM). A cloud model requires an explicit `--allow-cloud` flag. Serial numbers, UUIDs, and hostnames are never collected.
- **Follow-ups:** conversation context is kept, so questions like "and what about the GPU?" work.

## Known limitations

- **Problem 1:** the embedder is a hashing model with a hand-written concept lexicon, not a learned model, and the evaluation set is small. Default agent comments are templates; commitments in them are fictional demo content. Vector search and rate limiting are in-process, and the READMEs describe the production path (pgvector/HNSW, a message queue, Redis).
- **Problem 2:** it was developed in a virtual machine without `dmidecode`, `lspci`, or a battery, so those paths are tested against recorded realistic outputs rather than live hardware. The Ollama and OpenAI-compatible backends are tested with a scripted fake model, not a real one. Live hardware reading requires Linux, while `--demo` works on any OS.
