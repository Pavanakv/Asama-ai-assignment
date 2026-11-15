"""Pure parsers: raw command / file text -> small, validated dicts.

Nothing here touches the OS, so every parser is unit-testable with captured
output. Strings are sanitised because hardware-supplied text (model names,
vendor strings) is untrusted data that ends up in an LLM prompt.
"""
from __future__ import annotations

import json
import re
import shlex
from typing import Optional

_CTRL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")


def clean(s, limit: int = 120) -> str:
    """Strip control chars and cap length of any hardware-supplied string."""
    return _CTRL.sub("", str(s)).strip()[:limit]


def _to_float(s) -> Optional[float]:
    try:
        return float(str(s).strip())
    except (TypeError, ValueError):
        return None


def _to_int(s) -> Optional[int]:
    try:
        return int(str(s).strip())
    except (TypeError, ValueError):
        return None


_SIZE_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*(kB|KB|KiB|MB|MiB|GB|GiB|TB|TiB)\s*$")
_TO_GB = {"kb": 1 / 1024 / 1024, "kib": 1 / 1024 / 1024, "mb": 1 / 1024, "mib": 1 / 1024,
          "gb": 1.0, "gib": 1.0, "tb": 1024.0, "tib": 1024.0}


def size_to_gb(text: str) -> Optional[float]:
    """'16 GB' / '8192 MB' / '4 TB' -> GB (dmidecode's GB is really GiB). None if not a size."""
    m = _SIZE_RE.match(text or "")
    if not m:
        return None
    return float(m.group(1)) * _TO_GB[m.group(2).lower()]


# --------------------------------------------------------------------- lscpu
def _cache(v):
    """'8 MiB (1 instance)' -> '8 MiB'"""
    return re.sub(r"\s*\(\d+ instances?\)", "", clean(v)) if v else None


def parse_lscpu(text: str) -> dict:
    kv: dict = {}
    for line in text.splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            kv[k.strip()] = v.strip()
    sockets = _to_int(kv.get("Socket(s)"))
    cores_per_socket = _to_int(kv.get("Core(s) per socket"))
    threads_per_core = _to_int(kv.get("Thread(s) per core"))
    logical = _to_int(kv.get("CPU(s)"))
    physical = sockets * cores_per_socket if sockets and cores_per_socket else None
    max_mhz, min_mhz = _to_float(kv.get("CPU max MHz")), _to_float(kv.get("CPU min MHz"))
    # ARM / some VMs lack "Model name" ("Model:" is then a bare number); report None rather than invent one.
    model = kv.get("Model name")
    hypervisor = kv.get("Hypervisor vendor")
    return {
        "model_name": clean(model) if model else None,
        "vendor": clean(kv["Vendor ID"]) if kv.get("Vendor ID") else None,
        "architecture": clean(kv["Architecture"]) if kv.get("Architecture") else None,
        "physical_cores": physical,
        "logical_cpus": logical,
        "threads_per_core": threads_per_core,
        "sockets": sockets,
        "max_mhz": max_mhz,
        "max_ghz": round(max_mhz / 1000, 2) if max_mhz else None,
        "min_mhz": min_mhz,
        "l2_cache": _cache(kv.get("L2 cache")),
        "l3_cache": _cache(kv.get("L3 cache")),
        "virtualization_support": clean(kv["Virtualization"]) if kv.get("Virtualization") else None,
        "hypervisor": clean(hypervisor) if hypervisor else None,
    }


# ------------------------------------------------------------------ meminfo
def parse_meminfo(text: str) -> dict:
    kib = {}
    for line in text.splitlines():
        m = re.match(r"^(\w[\w()]*):\s+(\d+)\s*kB", line)
        if m:
            kib[m.group(1)] = int(m.group(2))
    g = lambda k: round(kib[k] / 1024 / 1024, 2) if k in kib else None  # noqa: E731
    total, avail = g("MemTotal"), g("MemAvailable")
    return {
        "usable_gib": g("MemTotal"),
        "available_gib": avail,
        "used_gib": round(total - avail, 2) if total is not None and avail is not None else None,
        "swap_total_gib": g("SwapTotal"),
    }


# ---------------------------------------------------------------- dmidecode
SOLDERED_FORM_FACTORS = {"row of chips", "chip", "die", "tsop", "soic"}


def _dmi_blocks(text: str):
    for raw in re.split(r"\n\s*\n", text):
        lines = [l.rstrip() for l in raw.strip().splitlines() if l.strip()]
        if len(lines) >= 2 and lines[0].startswith("Handle "):
            fields = {}
            for l in lines[2:]:
                if ":" in l:
                    k, v = l.split(":", 1)
                    fields[k.strip()] = v.strip()
            yield lines[1].strip(), fields


def parse_dmidecode_memory(text: str) -> dict:
    """Per-block parse (NOT a global regex over 'Size:' lines, which also matches other DMI types)."""
    arrays, modules = [], []
    for title, f in _dmi_blocks(text):
        if title == "Physical Memory Array":
            arrays.append({
                "max_capacity_gb": size_to_gb(f.get("Maximum Capacity", "")),
                "slots": _to_int(f.get("Number Of Devices")),
                "ecc": (f.get("Error Correction Type", "None").lower() not in ("none", "unknown")),
            })
        elif title == "Memory Device":
            size_txt = f.get("Size", "")
            installed = size_to_gb(size_txt) is not None and "no module" not in size_txt.lower()
            form = f.get("Form Factor", "")
            locator = f.get("Locator", "")
            mtype = f.get("Type", "")
            soldered = (form.lower() in SOLDERED_FORM_FACTORS
                        or bool(re.search(r"onboard|soldered|on-board", locator, re.I))
                        or mtype.upper().startswith("LPDDR"))
            speed = f.get("Configured Memory Speed") or f.get("Configured Clock Speed") or f.get("Speed", "")
            modules.append({
                "slot": clean(locator) or None,
                "installed": installed,
                "size_gb": size_to_gb(size_txt) if installed else None,
                "type": clean(mtype) if mtype and mtype.lower() != "unknown" else None,
                "form_factor": clean(form) if form and form.lower() != "unknown" else None,
                "speed": clean(speed) if speed and speed.lower() != "unknown" else None,
                "manufacturer": clean(f.get("Manufacturer", "")) or None if installed else None,
                "part_number": clean(f.get("Part Number", "")) or None if installed else None,
                "soldered": soldered,
                # Serial / asset tag deliberately not collected (identifying, and the LLM doesn't need them).
            })
    max_caps = [a["max_capacity_gb"] for a in arrays if a["max_capacity_gb"]]
    return {
        "arrays": len(arrays),
        "max_capacity_gb": sum(max_caps) if max_caps else None,
        "slots_total": (sum(a["slots"] for a in arrays if a["slots"]) or len(modules)) if (arrays or modules) else None,
        "ecc": any(a["ecc"] for a in arrays) if arrays else None,
        "modules": modules,
    }


# -------------------------------------------------------------------- lsblk
def _truthy(v) -> Optional[bool]:
    if isinstance(v, bool):
        return v
    if v in (1, "1", "true", "True"):
        return True
    if v in (0, "0", "false", "False"):
        return False
    return None


def parse_lsblk(text: str, include_partitions: bool = False) -> dict:
    try:
        tree = json.loads(text)
    except json.JSONDecodeError:
        return {"drives": [], "parse_error": "lsblk returned non-JSON output"}
    drives = []
    for dev in tree.get("blockdevices", []):
        name = str(dev.get("name", ""))
        if dev.get("type") != "disk" or re.match(r"^(zram|ram|loop|sr|fd)\d*", name):
            continue
        tran = (dev.get("tran") or "").lower() or None
        rota = _truthy(dev.get("rota"))
        note = None
        if re.match(r"^(vd|xvd)[a-z]+$", name):
            # Virtual disks (virtio/Xen) report rotational=1 no matter what storage backs them: don't trust the flag.
            kind, iface = "virtual disk", "virtual"
            note = "This is a virtual disk inside a VM; whether the host storage is SSD or HDD can't be determined from here."
        elif name.startswith("nvme") or tran == "nvme":
            kind, iface = "NVMe SSD", "NVMe"
        elif tran == "usb":
            kind, iface = ("external/USB HDD" if rota else "external/USB drive"), "USB"
        elif rota is True:
            kind, iface = "HDD", (tran.upper() if tran else "unknown")
        elif rota is False:
            kind, iface = "SSD", (tran.upper() if tran else "unknown")
        else:
            kind, iface = "unknown", (tran.upper() if tran else "unknown")
        size_b = _to_int(dev.get("size")) if dev.get("size") is not None else None
        if size_b is not None and size_b < 1_000_000:
            continue  # empty card-reader / placeholder device, nothing to report
        d = {
            "name": clean(name),
            "model": clean(dev["model"]) if dev.get("model") else None,
            "size_gb": round(size_b / 1e9, 1) if size_b else None,
            "size_gib": round(size_b / 1024**3, 1) if size_b else None,
            "kind": kind,
            "interface": iface,
            "rotational": rota,
            "removable": _truthy(dev.get("rm")),
        }
        if note:
            d["note"] = note
        if include_partitions:
            parts = []
            for c in dev.get("children", []) or []:
                mps = c.get("mountpoints") or c.get("mountpoint")
                mps = [m for m in (mps if isinstance(mps, list) else [mps]) if m]
                cb = _to_int(c.get("size"))
                parts.append({"name": clean(c.get("name", "")), "size_gib": round(cb / 1024**3, 1) if cb else None,
                              "fstype": clean(c["fstype"]) if c.get("fstype") else None, "mountpoints": [clean(m) for m in mps]})
            d["partitions"] = parts
        drives.append(d)
    return {"drives": drives}


# -------------------------------------------------------------------- lspci
GPU_CLASSES = {"vga compatible controller", "3d controller", "display controller"}
NET_CLASSES = {"network controller", "ethernet controller"}
_AMD_IGPU = re.compile(r"vega|renoir|cezanne|barcelo|rembrandt|phoenix|lucienne|picasso|raven|hawk point|radeon \d{3}m|graphics\b(?!.*rx)", re.I)


def _lspci_entries(text: str):
    for line in text.splitlines():
        try:
            t = shlex.split(line)
        except ValueError:
            continue
        if len(t) >= 4:
            yield {"slot": t[0], "class": t[1], "vendor": t[2], "device": t[3]}


def _gpu_type(vendor: str, device: str, cls: str) -> str:
    v = vendor.lower()
    if "nvidia" in v:
        return "discrete"
    if "intel" in v:
        return "integrated"
    if cls.lower() == "3d controller":
        return "discrete"
    if "advanced micro devices" in v or "amd" in v or "ati" in v:
        return "integrated" if _AMD_IGPU.search(device) else "unknown"
    return "unknown"


def parse_lspci_gpus(text: str) -> list:
    out = []
    for e in _lspci_entries(text):
        if e["class"].lower() in GPU_CLASSES:
            out.append({
                "vendor": clean(e["vendor"]), "name": clean(e["device"]), "pci_slot": clean(e["slot"], 20),
                "likely_type": _gpu_type(e["vendor"], e["device"], e["class"]),
            })
    return out


def parse_lspci_network(text: str) -> list:
    out = []
    for e in _lspci_entries(text):
        if e["class"].lower() in NET_CLASSES:
            out.append({"kind": "wifi/wireless" if e["class"].lower() == "network controller" else "ethernet",
                        "vendor": clean(e["vendor"]), "name": clean(e["device"])})
    return out


def parse_nvidia_smi(text: str) -> list:
    out = []
    for line in text.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 3:
            mib = _to_float(parts[1])
            out.append({"name": clean(parts[0]), "vram_mib": mib, "vram_gib": round(mib / 1024, 1) if mib else None,
                        "driver_version": clean(parts[2], 40)})
    return out


# ------------------------------------------------------------- small files
def parse_os_release(text: str) -> dict:
    kv = {}
    for line in text.splitlines():
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            kv[k.strip()] = v.strip().strip('"').strip("'")
    return {"pretty_name": clean(kv["PRETTY_NAME"]) if kv.get("PRETTY_NAME") else None,
            "id": clean(kv["ID"], 40) if kv.get("ID") else None,
            "version_id": clean(kv["VERSION_ID"], 40) if kv.get("VERSION_ID") else None}


def parse_battery(files: dict) -> dict:
    """files: {'status': '...', 'energy_full': '...', ...} (raw file contents, may be partial)."""
    g = lambda k: (files.get(k) or "").strip() or None  # noqa: E731
    out = {"status": clean(g("status"), 30) if g("status") else None,
           "charge_percent": _to_int(g("capacity")), "cycle_count": _to_int(g("cycle_count")),
           "technology": clean(g("technology"), 30) if g("technology") else None,
           "design_wh": None, "full_charge_wh": None, "health_percent": None}
    # Energy-based (µWh) can be shown in Wh; charge-based (µAh) can't without voltage, but the ratio still gives health.
    for full_k, design_k, to_wh in (("energy_full", "energy_full_design", True), ("charge_full", "charge_full_design", False)):
        full, design = _to_float(g(full_k)), _to_float(g(design_k))
        if full and design:
            out["health_percent"] = round(full / design * 100, 1)
            if to_wh:
                out["design_wh"], out["full_charge_wh"] = round(design / 1e6, 1), round(full / 1e6, 1)
            break
    if out["cycle_count"] == 0:
        out["cycle_count"] = None  # many laptops report 0 = "not supported", not "brand new"
        out["cycle_count_note"] = "firmware reports 0, which usually means 'not tracked'"
    return out
