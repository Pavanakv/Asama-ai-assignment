"""Tool registry: what the LLM may call, and what each call returns.

Design rules
* Every tool is read-only and takes at most a couple of booleans. There is no
  tool that accepts a command, a path or a free-form string.
* Every result uses one envelope so the model (and the grounding checker) can
  rely on it:  {tool, ok, data, sources, limitations}
* `limitations` is where "couldn't read X because Y" goes, so the model has an
  explicit, honest reason to say "I can't tell" instead of guessing.
* Arithmetic the user cares about (upgrade headroom) is computed here, in code.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

from . import parsers as P
from .runner import NotAllowed

SUDO_HINT = "Run `python -m hwassist --print-sudoers` for the one-line sudoers rule that allows exactly this read-only command."
STANDARD_STICKS_GB = (1, 2, 4, 8, 16, 32, 64, 128, 256)


class ToolArgError(ValueError):
    pass


@dataclass
class Tool:
    name: str
    description: str
    properties: dict
    handler: Callable
    volatile: bool = False  # True = values change over time (don't cache for long)
    requires_any: tuple = ()  # command keys; tool is hidden from the LLM if none are available AND no file fallback

    def schema(self) -> dict:
        return {"type": "object", "properties": self.properties, "required": [], "additionalProperties": False}

    def spec(self) -> dict:  # OpenAI-style function spec (Ollama accepts the same shape)
        return {"type": "function", "function": {"name": self.name, "description": self.description, "parameters": self.schema()}}


def _envelope(tool: str, data: dict, sources: list, limitations: list) -> dict:
    return {"tool": tool, "ok": True, "data": data, "sources": sources, "limitations": limitations}


def _failed(tool: str, why: str, limitations=None) -> dict:
    return {"tool": tool, "ok": False, "data": {}, "sources": [], "limitations": (limitations or []) + [why]}


# ---------------------------------------------------------------- handlers
def _cpu(r, args):
    res = r.run("lscpu")
    if not res.ok:
        return _failed("get_cpu_info", f"lscpu unavailable: {res.error}")
    data = P.parse_lscpu(res.stdout)
    lim = []
    if data["model_name"] is None:
        lim.append("lscpu did not report a CPU model name (common on ARM and some VMs).")
    if data["hypervisor"]:
        lim.append(f"Running under a hypervisor ({data['hypervisor']}); values describe the virtual machine, not the physical host.")
    return _envelope("get_cpu_info", data, ["lscpu"], lim)


def _memory(r, args):
    sources, lim = [], []
    mi = r.read_file("/proc/meminfo")
    data: dict = {}
    if mi:
        data.update(P.parse_meminfo(mi))
        sources.append("/proc/meminfo")
    else:
        lim.append("/proc/meminfo could not be read.")

    dmi = r.run("dmidecode_memory")
    if dmi.ok:
        sources.append("dmidecode -t memory")
        d = P.parse_dmidecode_memory(dmi.stdout)
        modules = d["modules"]
        installed = [m for m in modules if m["installed"]]
        slots_total = d["slots_total"]
        data.update({
            "slots_total": slots_total,
            "slots_used": len(installed),
            "slots_free": (slots_total - len(installed)) if slots_total is not None else None,
            "installed_gb": round(sum(m["size_gb"] for m in installed), 1) if installed else None,
            "max_capacity_gb": d["max_capacity_gb"],
            "ecc": d["ecc"],
            "modules": installed if not args.get("include_empty_slots") else modules,
        })
        data["upgrade"] = _upgrade(data, installed)
        lim.append("Maximum capacity comes from the firmware (SMBIOS table). Some manufacturers under-report it, so confirm with the spec sheet for the exact model before buying.")
    else:
        if dmi.error_kind == "permission":
            lim.append(f"Slot layout, module sizes and maximum supported capacity need root (dmidecode): {dmi.error}. {SUDO_HINT}")
        elif dmi.error_kind == "missing":
            lim.append("dmidecode is not installed, so slot layout and maximum capacity are unknown.")
        else:
            lim.append(f"dmidecode failed: {dmi.error}")
        data["slots_total"] = data["slots_used"] = data["slots_free"] = data["max_capacity_gb"] = None
    if not sources:
        return _failed("get_memory_info", "No memory information could be read.", lim)
    return _envelope("get_memory_info", data, sources, lim)


def _upgrade(data: dict, installed: list) -> dict:
    free, mx, inst = data["slots_free"], data["max_capacity_gb"], data["installed_gb"]
    soldered = [m for m in installed if m["soldered"]]
    removable = [m for m in installed if not m["soldered"]]
    base = {"soldered_modules": len(soldered), "removable_modules": len(removable)}
    if installed and not removable and not free:
        return {**base, "possible": False, "mode": "none", "reason": "All installed memory is soldered to the board and there is no free slot."}
    if mx is None or inst is None:
        return {**base, "possible": None, "mode": "unknown", "reason": "Maximum capacity or installed size is not reported by the firmware."}
    headroom = round(mx - inst, 1)
    if free and free > 0:
        if headroom <= 0:
            return {**base, "possible": False, "mode": "none", "headroom_gb": headroom,
                    "reason": "Installed memory already equals the reported maximum."}
        per_slot = headroom / free
        fits = [s for s in STANDARD_STICKS_GB if s <= per_slot]
        mod = (removable or installed)[0]
        return {**base, "possible": True, "mode": "add_module", "free_slots": free, "headroom_gb": headroom,
                "max_single_module_gb": fits[-1] if fits else None,
                "compatible_with": {"type": mod.get("type"), "form_factor": mod.get("form_factor"), "speed": mod.get("speed")}}
    if removable and headroom > 0:
        return {**base, "possible": True, "mode": "replace_modules", "headroom_gb": headroom,
                "reason": "All slots are used; you would need to replace existing modules with larger ones."}
    return {**base, "possible": False, "mode": "none", "headroom_gb": headroom, "reason": "No free slots and no headroom."}


def _storage(r, args):
    res = r.run("lsblk")
    if not res.ok:
        return _failed("get_storage_info", f"lsblk unavailable: {res.error}")
    data = P.parse_lsblk(res.stdout, include_partitions=bool(args.get("include_partitions")))
    lim = []
    if data.get("parse_error"):
        lim.append(data["parse_error"])
    return _envelope("get_storage_info", data, ["lsblk"], lim)


def _gpu(r, args):
    sources, lim, gpus = [], [], []
    pci = r.run("lspci")
    if pci.ok:
        gpus = P.parse_lspci_gpus(pci.stdout)
        sources.append("lspci")
    else:
        lim.append(f"lspci unavailable ({pci.error}); GPUs could not be listed." + (" Install the 'pciutils' package." if pci.error_kind == "missing" else ""))
    smi = r.run("nvidia_smi")
    nv = []
    if smi.ok:
        nv = P.parse_nvidia_smi(smi.stdout)
        sources.append("nvidia-smi")
    elif any("nvidia" in g["vendor"].lower() for g in gpus):
        lim.append("An NVIDIA GPU is present but nvidia-smi isn't available (proprietary driver not installed?), so its VRAM size is unknown.")
    if any(g["likely_type"] == "integrated" for g in gpus):
        lim.append("Integrated GPUs have no dedicated VRAM; they share system RAM.")
    if not sources:
        return _failed("get_gpu_info", "No GPU information could be read.", lim)
    return _envelope("get_gpu_info", {"gpus": gpus, "nvidia_smi": nv}, sources, lim)


def _battery(r, args):
    bats = r.list_batteries()
    if not bats:
        return _envelope("get_battery_info", {"batteries": []}, ["/sys/class/power_supply"],
                         ["No battery was found (desktop, or the kernel doesn't expose one)."])
    out = []
    for b in bats:
        files = {k: r.read_file(f"/sys/class/power_supply/{b}/{k}") for k in
                 ("status", "capacity", "energy_full", "energy_full_design", "charge_full", "charge_full_design", "cycle_count", "technology")}
        out.append({"name": b, **P.parse_battery({k: v for k, v in files.items() if v is not None})})
    return _envelope("get_battery_info", {"batteries": out}, [f"/sys/class/power_supply/{b}" for b in bats], [])


def _system(r, args):
    def f(name):
        v = r.read_file(f"/sys/class/dmi/id/{name}")
        return P.clean(v) if v and v.strip() else None
    data = {
        "vendor": f("sys_vendor"), "product_name": f("product_name"), "product_version": f("product_version"),
        "bios_vendor": f("bios_vendor"), "bios_version": f("bios_version"), "bios_date": f("bios_date"),
    }
    osr = r.read_file("/etc/os-release")
    data["os"] = P.parse_os_release(osr) if osr else None
    kern = r.read_file("/proc/sys/kernel/osrelease")
    data["kernel"] = P.clean(kern, 60) if kern else None
    lim = []
    if not any(data[k] for k in ("vendor", "product_name")):
        lim.append("DMI product info is not exposed here (typical for VMs, containers and some ARM boards).")
    return _envelope("get_system_overview", data, ["/sys/class/dmi/id", "/etc/os-release", "/proc/sys/kernel/osrelease"], lim)


def _network(r, args):
    res = r.run("lspci")
    if not res.ok:
        return _failed("get_network_adapters", f"lspci unavailable: {res.error}")
    return _envelope("get_network_adapters", {"adapters": P.parse_lspci_network(res.stdout)}, ["lspci"],
                     ["Only PCI network adapters are listed (USB dongles and Bluetooth are not covered)."])


_BOOL = lambda desc: {"type": "boolean", "description": desc}  # noqa: E731

TOOLS = [
    Tool("get_system_overview", "Laptop make/model, BIOS version, Linux distribution and kernel version.", {}, _system),
    Tool("get_cpu_info", "CPU model, physical cores, logical threads, max clock speed, cache sizes, virtualisation support.", {}, _cpu, requires_any=("lscpu",)),
    Tool("get_memory_info",
         "RAM: usable/available/used amount now, and (when readable) installed module sizes/type/speed, total and free slots, "
         "maximum supported capacity, and a pre-computed upgrade assessment. Use for any RAM or memory-upgrade question.",
         {"include_empty_slots": _BOOL("Also list empty slots individually.")}, _memory, volatile=True),
    Tool("get_storage_info", "Internal drives: model, size, and whether each is NVMe SSD, SATA SSD or HDD.",
         {"include_partitions": _BOOL("Also list partitions, filesystems and mount points.")}, _storage, requires_any=("lsblk",)),
    Tool("get_gpu_info", "Graphics adapters (integrated and discrete) and, for NVIDIA, dedicated VRAM.", {}, _gpu, requires_any=("lspci", "nvidia_smi")),
    Tool("get_battery_info", "Battery charge, status, design vs. current full capacity (health) and cycle count.", {}, _battery, volatile=True),
    Tool("get_network_adapters", "Wi-Fi and Ethernet adapters (PCI).", {}, _network, requires_any=("lspci",)),
]
BY_NAME = {t.name: t for t in TOOLS}
VOLATILE_TTL_S = 10.0


def validate_args(tool: Tool, args) -> dict:
    """Strict validation. Small local models often send "true" strings or extra keys; accept the former, reject the latter."""
    if args in (None, ""):
        args = {}
    if not isinstance(args, dict):
        raise ToolArgError("arguments must be a JSON object")
    clean_args = {}
    for k, v in args.items():
        spec = tool.properties.get(k)
        if spec is None:
            raise ToolArgError(f"unknown argument {k!r}; allowed: {sorted(tool.properties) or 'none'}")
        if spec["type"] == "boolean":
            if isinstance(v, str) and v.lower() in ("true", "false"):
                v = v.lower() == "true"
            if not isinstance(v, bool):
                raise ToolArgError(f"argument {k!r} must be a boolean")
        clean_args[k] = v
    return clean_args


class Registry:
    """Binds tools to a runner; hides tools that can't work here; caches results for follow-up questions."""

    def __init__(self, runner) -> None:
        self.runner = runner
        self._cache: dict = {}
        self.tools = [t for t in TOOLS if self._usable(t)]
        self.unavailable = [t.name for t in TOOLS if t not in self.tools]

    def _usable(self, t: Tool) -> bool:
        return not t.requires_any or any(self.runner.available(k) for k in t.requires_any)

    def specs(self) -> list:
        return [t.spec() for t in self.tools]

    def call(self, name: str, args) -> dict:
        tool = BY_NAME.get(name)
        if tool is None or tool not in self.tools:
            hint = "" if tool is None else " (not available on this machine)"
            return {"tool": name, "ok": False, "data": {}, "sources": [], "limitations": [f"Unknown or unavailable tool {name!r}{hint}. Available: {[t.name for t in self.tools]}"]}
        try:
            args = validate_args(tool, args)
        except ToolArgError as e:
            return {"tool": name, "ok": False, "data": {}, "sources": [], "limitations": [f"Invalid arguments: {e}"]}
        key = (name, tuple(sorted(args.items())))
        hit = self._cache.get(key)
        ttl = VOLATILE_TTL_S if tool.volatile else float("inf")
        if hit and time.monotonic() - hit[0] < ttl:
            return hit[1]
        try:
            result = tool.handler(self.runner, args)
        except NotAllowed as e:  # a bug, never user-triggerable; fail closed
            result = _failed(name, f"blocked by allow-list: {e}")
        except Exception as e:  # parser/OS surprise: report, don't crash the chat
            result = _failed(name, f"internal error while reading hardware data: {type(e).__name__}")
        self._cache[key] = (time.monotonic(), result)
        return result
