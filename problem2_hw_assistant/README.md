# Problem 2: Hardware Q&A assistant for a Linux laptop

A chat CLI that answers questions like "How much RAM do I have?" or "Can I add more RAM, and how much?" by calling read-only inspection tools rather than guessing. Standard library only; no install step.

```
python -m hwassist                          # chat about THIS machine (local Ollama if running, else built-in rules)
python -m hwassist --demo                   # chat about a recorded laptop (works on any OS)
python -m hwassist --demo --once "Can I add more RAM, and how much?"
python -m hwassist --backend ollama --model qwen2.5:7b     # fully local LLM, nothing leaves the machine
HWASSIST_API_KEY=... python -m hwassist --backend openai --allow-cloud   # cloud LLM, explicit opt-in
python -m hwassist --print-sudoers          # the one-line rule that enables root-only data
python -m pytest tests -q                   # 66 tests
```

In chat: `/tools`, `/doctor` (what works on this machine and why), `/debug` (show tool calls), `/clear`, `/quit`. Demo machines: `--demo demo_laptop | ultrabook | no_privilege`.

Example (demo laptop): *"You have 16 GB installed in 1 of 2 slots (DDR4 3200 MT/s). The firmware reports a maximum of 64 GB, i.e. 48 GB more than you have. With 1 free slot, the largest standard module you could add per slot is 32 GB."*

## Design

```
question -> LLM (or rules) picks tools -> allow-listed runner -> parsers -> {data, sources, limitations} -> answer
                                                                                              |
                                              grounding check: every number in the answer must be in the tool data
```

- **No arbitrary execution.** The model can only call named tools, and no tool takes a string, path or command (only booleans, validated strictly). `runner.py` is the only code that touches the OS: fixed argv tables, `shell=False`, absolute binary paths from a fixed directory list, scrubbed env (`LC_ALL=C`), stdin closed, timeout, output cap. File reads go through an exact-path allow-list that excludes serial numbers, UUIDs and anything identifying.
- **Root without running as root.** Only `dmidecode -t memory` needs privilege. Tool code just says "privileged"; the runner uses `sudo -n` (never prompts or hangs) and `--print-sudoers` emits a rule pinning that exact command. Without the rule the assistant still answers what it can, and says what it can't and how to fix it.
- **Accuracy.** Every result carries `sources` and `limitations`. Upgrade math (free slots, headroom, largest module) is computed in code. Soldered memory is detected so it never claims you can add RAM to a laptop that can't. Virtual disks aren't labelled SSD/HDD (their rotational flag is meaningless). dmidecode is parsed per "Memory Device" block, not by grepping every `Size:` line. After the model answers, a checker verifies each number against the tool data; on a miss the model is asked to retry, and if it still fails the user gets a warning.
- **Portability.** Tools whose utility is missing (no `lspci`, no `nvidia-smi`) are hidden from the model and `/doctor` says which package to install. lsblk has a fallback for older util-linux. Output parsing is tolerant of missing fields.
- **Privacy.** Local by default (Ollama or the no-LLM rules backend). The cloud backend refuses to start without `--allow-cloud`. Serial numbers, UUIDs and hostname are never collected.
- **Follow-ups.** Conversation text is kept; stale tool JSON is not (it's re-fetched cheaply from a cache). The rules backend carries the last topic so "and what about the GPU?" works without an LLM.
- **Prompt injection.** Hardware strings are untrusted: control characters are stripped, lengths capped, and the system prompt tells the model tool output is data. Tests cover a malicious model name and a model that tries to call `run_command`.

## Honest limitations

- **I could not test against a real LLM here.** The Ollama and OpenAI-compatible backends are exercised only with a scripted fake model, so the tool-call loop and wire formats are tested but real model behaviour (especially small models' tool calling) is not. Try it with `qwen2.5:7b` and expect to tune the prompt.
- **The sandbox this was built in is a VM with no `dmidecode`, `lspci` or battery**, so those paths are tested against recorded fixtures (realistic outputs, including soldered and no-privilege cases), not live hardware. Run `/doctor` on your laptop first.
- The grounding checker only checks numbers, so a wrong word with no digits would pass.
- The firmware's "maximum capacity" is sometimes wrong; answers say to confirm with the manufacturer.
- Linux only for live data (`--demo` works anywhere). NVIDIA VRAM needs the proprietary driver's `nvidia-smi`; AMD GPU VRAM isn't read.
