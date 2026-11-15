"""Offline fallback: no LLM at all.

Keyword intent -> the same tools -> templated answers built only from the tool
JSON. Less flexible than a model, but it cannot hallucinate, runs anywhere,
and keeps the assistant useful when Ollama isn't installed or a cloud call
fails. It is also what the tests use to exercise the full stack deterministically.
"""
from __future__ import annotations

import re

INTENTS = [
    ("memory", r"\b(ram|memory|dimm|sodimm|slots?|ddr\d?)\b"),
    ("cpu", r"\b(cpu|processor|cores?|threads?|clock|ghz|turbo|chip)\b"),
    ("storage", r"\b(ssd|nvme|hdd|disks?|drives?|storage|sata|m\.?2|hard ?drive)\b"),
    ("gpu", r"\b(gpu|graphics|video ?card|vram|nvidia|radeon|geforce|iris)\b"),
    ("battery", r"\b(battery|charge|charging|cycles?|wear|health)\b"),
    ("system", r"\b(os|operating system|distro|distribution|kernel|ubuntu|fedora|laptop|vendor|manufacturer|bios|machine|computer|lenovo|dell|hp|asus)\b"),
    ("network", r"\b(wi-?fi|wireless|ethernet|network|lan)\b"),
]
OVERVIEW = r"\b(everything|all specs|specs|summary|overview|full report|tell me about (my|this))\b"
TOPIC_TOOL = {"memory": "get_memory_info", "cpu": "get_cpu_info", "storage": "get_storage_info", "gpu": "get_gpu_info",
              "battery": "get_battery_info", "system": "get_system_overview", "network": "get_network_adapters"}
INSTALL_HINT = {"get_gpu_info": "needs `lspci` from the 'pciutils' package", "get_network_adapters": "needs `lspci` from the 'pciutils' package",
                "get_cpu_info": "needs `lscpu` from 'util-linux'", "get_storage_info": "needs `lsblk` from 'util-linux'"}
FOLLOWUP = r"\b(it|that|those|them|this|these|more|free|used|available|left|and)\b"
UPGRADE = r"upgrad|\badd\b|\bmore\b|expand|\bmax|increase|extra|bigger|larger|install|upgrade"
CAPABILITIES = ("CPU, RAM and upgrade options, storage (SSD/NVMe/HDD), GPU, battery, laptop model/OS/BIOS, and network adapters")


def detect_topics(q: str, last_topics: list) -> list:
    ql = q.lower()
    if re.search(OVERVIEW, ql):
        return ["system", "cpu", "memory", "storage", "gpu", "battery"]
    found = [t for t, rx in INTENTS if re.search(rx, ql)]
    if not found and last_topics and re.search(FOLLOWUP, ql):
        return list(last_topics)
    return found


def _g(x, unit="", nd=None):
    if x is None:
        return None
    if isinstance(x, float) and x == int(x):
        x = int(x)
    return f"{x}{unit}"


def _lim(res: dict, only=()) -> str:
    items = [l for l in res.get("limitations", []) if not only or any(o in l for o in only)]
    return (" " + " ".join(items)) if items else ""


# ---------------------------------------------------------------- formatters
def fmt_system(res, q):
    if not res["ok"]:
        return "I can't determine the laptop model/OS from system data." + _lim(res)
    d = res["data"]
    name = " ".join(x for x in (d.get("vendor"), d.get("product_version") or d.get("product_name")) if x)
    if d.get("product_name") and d.get("product_version") and d["product_name"] != d["product_version"]:
        name += f" ({d['product_name']})"
    parts = []
    if name:
        parts.append(f"This machine is a {name}.")
    if d.get("os") and d["os"].get("pretty_name"):
        parts.append(f"It runs {d['os']['pretty_name']}" + (f" (kernel {d['kernel']})." if d.get("kernel") else "."))
    elif d.get("kernel"):
        parts.append(f"Kernel {d['kernel']}.")
    if d.get("bios_version"):
        parts.append(f"BIOS version {d['bios_version']}" + (f" dated {d['bios_date']}." if d.get("bios_date") else "."))
    return " ".join(parts) + _lim(res) if parts else "The system doesn't expose model/OS details." + _lim(res)


def fmt_cpu(res, q):
    if not res["ok"]:
        return "I can't read CPU details from system data." + _lim(res)
    d = res["data"]
    out = [d.get("model_name") or "CPU model not reported"]
    bits = []
    if d.get("physical_cores") and d.get("logical_cpus"):
        bits.append(f"{d['physical_cores']} cores / {d['logical_cpus']} threads")
    elif d.get("logical_cpus"):
        bits.append(f"{d['logical_cpus']} logical CPUs")
    if d.get("max_ghz"):
        bits.append(f"up to {_g(d['max_ghz'])} GHz")
    if d.get("l3_cache"):
        bits.append(f"{d['l3_cache']} L3 cache")
    return out[0] + (": " + ", ".join(bits) + "." if bits else ".") + _lim(res, ("hypervisor", "model name"))


def fmt_memory(res, q):
    if not res["ok"]:
        return "I can't read memory information from system data." + _lim(res)
    d = res["data"]
    ql = q.lower()
    asks_upgrade = bool(re.search(UPGRADE, ql))
    usable = f"The OS sees {d.get('usable_gib')} GiB usable ({d.get('available_gib')} GiB available right now)." if d.get("usable_gib") else ""
    if d.get("slots_total") is None:
        return ((usable + " ") if usable else "") + "I can't tell how many slots there are, what's installed, or the maximum supported capacity from system data." + _lim(res, ("root", "dmidecode"))
    mods = [m for m in d.get("modules", []) if m.get("installed")]
    kinds = sorted({" ".join(x for x in (m.get("type"), m.get("speed")) if x) for m in mods if m.get("type") or m.get("speed")})
    up = d.get("upgrade", {})
    if mods and up.get("soldered_modules") == len(mods) and not up.get("removable_modules"):
        installed = f"You have {_g(d['installed_gb'], ' GB')} of memory soldered to the board" + (f" ({', '.join(kinds)})." if kinds else ".")
    else:
        installed = f"You have {_g(d['installed_gb'], ' GB')} installed in {d['slots_used']} of {d['slots_total']} slots" + (f" ({', '.join(kinds)})." if kinds else ".")
    if up.get("mode") == "add_module":
        c = up.get("compatible_with", {})
        kind = " ".join(x for x in (c.get("type"), c.get("form_factor")) if x)
        stick = (f" With {up['free_slots']} free slot{'s' if up['free_slots'] != 1 else ''}, the largest standard module you could add per slot is {up['max_single_module_gb']} GB"
                 if up.get("max_single_module_gb") else "")
        upgrade = (f"The firmware reports a maximum of {_g(d['max_capacity_gb'], ' GB')}, i.e. {_g(up['headroom_gb'], ' GB')} more than you have.{stick}"
                   + (f" ({kind})." if stick and kind else ("." if stick else "")))
    elif up.get("mode") == "replace_modules":
        upgrade = f"All slots are used, but the maximum is {_g(d['max_capacity_gb'], ' GB')}, so you could swap modules for larger ones (up to {_g(up['headroom_gb'], ' GB')} more in total)."
    else:
        upgrade = "Upgrade: " + (up.get("reason") or "I can't determine upgrade options from system data.").replace("and there is no free slot", "so it can't be upgraded")
    caveat = " Check the manufacturer's spec sheet for your exact model, since the firmware's maximum isn't always accurate."
    if asks_upgrade:
        return installed + " " + upgrade + caveat
    return installed + " " + usable + (" " + upgrade if up else "")


def ssd_hdd_lead(drives, hits, one, many, noun) -> str:
    unknown = [d["name"] for d in drives if d["kind"] in ("virtual disk", "unknown")]
    if hits:
        return f"Yes - {', '.join(hits)} {'is ' + one if len(hits) == 1 else 'are ' + many}. " + (f"I can't tell for {', '.join(unknown)}. " if unknown else "")
    if unknown:
        return f"I can't tell whether it's an {noun}/HDD: {', '.join(unknown)} {'is a virtual disk or reports no type' if len(unknown) == 1 else 'are virtual disks or report no type'} (typical inside a VM). "
    return f"No {noun} found. "


def fmt_storage(res, q):
    if not res["ok"]:
        return "I can't read storage details from system data." + _lim(res)
    drives = res["data"]["drives"]
    if not drives:
        return "I didn't find any internal drives in the lsblk output."
    ql = q.lower()
    desc = lambda d: f"{d['name']} - {d.get('model') or 'unknown model'}, {_g(d.get('size_gb'), ' GB')} ({_g(d.get('size_gib'), ' GiB')}), {d['kind']}"  # noqa: E731
    notes = sorted({d["note"] for d in drives if d.get("note")})
    lead = ""
    if "nvme" in ql or "m.2" in ql or "m2" in ql:
        nv = [d["name"] for d in drives if d["interface"] == "NVMe"]
        lead = (f"Yes - {', '.join(nv)} {'is' if len(nv) == 1 else 'are'} NVMe. " if nv else "No NVMe drive found. ")
    elif re.search(r"\bssd\b", ql):
        ss = [d["name"] for d in drives if "SSD" in d["kind"]]
        lead = ssd_hdd_lead(drives, ss, "an SSD", "SSDs", "SSD")
    elif re.search(r"\b(hdd|hard ?drive|spinning)\b", ql):
        hh = [d["name"] for d in drives if "HDD" in d["kind"]]
        lead = ssd_hdd_lead(drives, hh, "a hard disk", "hard disks", "spinning hard disk")
    body = "Drives: " + "; ".join(desc(d) for d in drives) + "."
    return lead + body + (" " + " ".join(notes) if notes else "") + _lim(res)


def fmt_gpu(res, q):
    if not res["ok"]:
        return "I can't read GPU details from system data." + _lim(res)
    gpus, nv = res["data"]["gpus"], res["data"]["nvidia_smi"]
    if not gpus and not nv:
        return "No graphics adapter was reported." + _lim(res)
    vram = {n["name"]: n for n in nv}
    parts = []
    for g in gpus:
        t = {"integrated": "integrated", "discrete": "discrete"}.get(g["likely_type"], "type not determined")
        match = next((v for k, v in vram.items() if k.split()[-1] in g["name"] or g["name"] in k), None)
        extra = f", {match['vram_gib']} GiB VRAM" if match and match.get("vram_gib") else ""
        parts.append(f"{g['vendor']} {g['name']} ({t}{extra})")
    return "Graphics: " + "; ".join(parts) + "." + _lim(res)


def fmt_battery(res, q):
    bats = res["data"].get("batteries", []) if res["ok"] else []
    if not bats:
        return "No battery was found on this machine." if res["ok"] else "I can't read battery details." + _lim(res)
    out = []
    for b in bats:
        s = f"{b['name']}: {b.get('charge_percent')}% ({(b.get('status') or 'status unknown').lower()})"
        if b.get("health_percent") is not None:
            s += f", health {b['health_percent']}% of design capacity"
            if b.get("design_wh"):
                s += f" ({b['full_charge_wh']} of {b['design_wh']} Wh)"
        else:
            s += ", health could not be determined"
        s += f", {b['cycle_count']} charge cycles" if b.get("cycle_count") else ", cycle count not reported by this laptop"
        out.append(s)
    return "; ".join(out) + "."


def fmt_network(res, q):
    if not res["ok"]:
        return "I can't list network adapters from system data." + _lim(res)
    a = res["data"]["adapters"]
    if not a:
        return "No PCI network adapters were reported." + _lim(res)
    return "Network: " + "; ".join(f"{x['vendor']} {x['name']} ({x['kind']})" for x in a) + "." + _lim(res)


FORMATTERS = {"memory": fmt_memory, "cpu": fmt_cpu, "storage": fmt_storage, "gpu": fmt_gpu, "battery": fmt_battery,
              "system": fmt_system, "network": fmt_network}


class RulesBackend:
    kind = "rules"
    name = "built-in rules (no LLM)"

    def answer(self, question: str, registry, last_topics: list):
        topics = detect_topics(question, last_topics)
        if not topics:
            return (f"I can only answer from live system data, and I'm not sure which part you mean. I can look up: {CAPABILITIES}. "
                    "I won't guess at specs.", [], [])
        texts, results = [], []
        for t in topics:
            tool = TOPIC_TOOL[t]
            if tool in registry.unavailable:
                texts.append(f"I can't check {t} on this machine: the needed system tool isn't installed ({INSTALL_HINT.get(tool, 'see /doctor')}).")
                continue
            res = registry.call(tool, {"include_empty_slots": False} if tool == "get_memory_info" else {})
            results.append(res)
            texts.append(FORMATTERS[t](res, question))
        return "\n".join(texts), results, topics
