# Asama.ai interview problems: AI Engineering

| | Problem | Run it | Tests |
|---|---|---|---|
| 1 | [Social network for AI agents](problem1_agent_network/README.md) | `cd problem1_agent_network && pip install -r requirements.txt && python demo.py` | 42 |
| 2 | [Linux hardware Q&A assistant](problem2_hw_assistant/README.md) | `cd problem2_hw_assistant && python -m hwassist --demo` | 66 |

Both run offline with no API key. Each README ends with a list of limitations; read those before an interview, because they are the questions you'll most likely be asked.

## 60-second talking points

**Problem 1:** the hard part is relevance, so routing is a funnel (hard filters, then vector retrieval, then a reranker with a threshold). Every notification stores a score and a reason. Spam is controlled structurally (threshold, per-post cap, per-agent budget); missed matches by explicit subscriptions and a bounded feedback multiplier. Trust: hashed keys, admin verification, signed webhooks, SSRF checks, injection quarantine. Measured: precision 0.57 / recall 0.87 on 15 labelled posts, so it over-notifies, and the embedder is a hashing model, not a learned one.

**Problem 2:** the model never touches a shell. It picks from named tools that take no free-text arguments; one module runs fixed, read-only commands with `shell=False`; root is handled by a sudoers rule that pins one exact command. Answers carry sources and limitations so "I can't tell, because X" is a first-class outcome, and a checker verifies every number against tool data. Not tested against a real LLM or real laptop hardware (recorded fixtures instead).
