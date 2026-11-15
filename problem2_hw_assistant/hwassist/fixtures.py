"""Recorded 'machines' for tests and `--demo`.

FixtureRunner has the same interface as runner.SystemRunner, so the entire
stack (tools -> agent -> CLI) runs unchanged against a recorded laptop. That
makes the tests deterministic and lets anyone try the assistant on a
non-Linux box or without the right privileges.
"""
from __future__ import annotations

import copy
from typing import Optional

from .runner import COMMANDS, CommandResult, NotAllowed, file_allowed

LSCPU = """\
Architecture:                    x86_64
CPU op-mode(s):                  32-bit, 64-bit
Address sizes:                   39 bits physical, 48 bits virtual
Byte Order:                      Little Endian
CPU(s):                          8
On-line CPU(s) list:             0-7
Vendor ID:                       GenuineIntel
Model name:                      11th Gen Intel(R) Core(TM) i5-1135G7 @ 2.40GHz
CPU family:                      6
Model:                           140
Thread(s) per core:              2
Core(s) per socket:              4
Socket(s):                       1
Stepping:                        1
CPU max MHz:                     4200.0000
CPU min MHz:                     400.0000
BogoMIPS:                        4838.40
Virtualization:                  VT-x
L1d cache:                       192 KiB (4 instances)
L2 cache:                        5 MiB (4 instances)
L3 cache:                        8 MiB (1 instance)
"""

LSBLK = """{
   "blockdevices": [
      {"name": "nvme0n1", "size": 512110190592, "type": "disk", "mountpoints": [null], "model": "SAMSUNG MZVLQ512HBLU-00B00", "rota": false, "tran": "nvme", "fstype": null, "rm": false,
       "children": [
          {"name": "nvme0n1p1", "size": 536870912, "type": "part", "mountpoints": ["/boot/efi"], "model": null, "rota": false, "tran": null, "fstype": "vfat", "rm": false},
          {"name": "nvme0n1p2", "size": 511571877888, "type": "part", "mountpoints": ["/"], "model": null, "rota": false, "tran": null, "fstype": "ext4", "rm": false}
       ]},
      {"name": "sda", "size": 1000204886016, "type": "disk", "mountpoints": [null], "model": "ST1000LM035-1RK172", "rota": true, "tran": "sata", "fstype": null, "rm": false,
       "children": [
          {"name": "sda1", "size": 1000203091968, "type": "part", "mountpoints": ["/home/data"], "model": null, "rota": true, "tran": null, "fstype": "ext4", "rm": false}
       ]},
      {"name": "zram0", "size": 8589934592, "type": "disk", "mountpoints": ["[SWAP]"], "model": null, "rota": false, "tran": null, "fstype": "swap", "rm": false}
   ]
}"""

LSPCI = """\
00:02.0 "VGA compatible controller" "Intel Corporation" "TigerLake-LP GT2 [Iris Xe Graphics]" -r01 "Lenovo" "Device 22c1"
00:14.3 "Network controller" "Intel Corporation" "Wi-Fi 6 AX201" -r30 "Intel Corporation" "Wi-Fi 6 AX201 160MHz"
01:00.0 "3D controller" "NVIDIA Corporation" "GP108M [GeForce MX350]" -ra1 "Lenovo" "Device 22c1"
02:00.0 "Ethernet controller" "Realtek Semiconductor Co., Ltd." "RTL8111/8168/8411 PCI Express Gigabit Ethernet Controller" -r15 "Lenovo" "Device 22c1"
"""

DMIDECODE_MEMORY = """\
# dmidecode 3.3
Getting SMBIOS data from sysfs.
SMBIOS 3.3.0 present.

Handle 0x0010, DMI type 16, 23 bytes
Physical Memory Array
\tLocation: System Board Or Motherboard
\tUse: System Memory
\tError Correction Type: None
\tMaximum Capacity: 64 GB
\tError Information Handle: Not Provided
\tNumber Of Devices: 2

Handle 0x0011, DMI type 17, 40 bytes
Memory Device
\tArray Handle: 0x0010
\tTotal Width: 64 bits
\tData Width: 64 bits
\tSize: 16 GB
\tForm Factor: SODIMM
\tSet: None
\tLocator: DIMM A
\tBank Locator: Bank 0
\tType: DDR4
\tType Detail: Synchronous
\tSpeed: 3200 MT/s
\tManufacturer: Samsung
\tSerial Number: 0A1B2C3D
\tAsset Tag: Not Specified
\tPart Number: M471A2G43AB2-CWE
\tRank: 2
\tConfigured Memory Speed: 3200 MT/s

Handle 0x0012, DMI type 17, 40 bytes
Memory Device
\tArray Handle: 0x0010
\tTotal Width: Unknown
\tData Width: Unknown
\tSize: No Module Installed
\tForm Factor: SODIMM
\tLocator: DIMM B
\tBank Locator: Bank 2
\tType: Unknown
\tSpeed: Unknown
\tManufacturer: Not Specified
\tPart Number: Not Specified
"""

MEMINFO = """\
MemTotal:       16114424 kB
MemFree:         3126680 kB
MemAvailable:   10873320 kB
Buffers:          412344 kB
Cached:          6718020 kB
SwapTotal:       8388604 kB
SwapFree:        8388604 kB
"""

OS_RELEASE = 'NAME="Ubuntu"\nVERSION="22.04.4 LTS (Jammy Jellyfish)"\nID=ubuntu\nVERSION_ID="22.04"\nPRETTY_NAME="Ubuntu 22.04.4 LTS"\n'

DEMO_LAPTOP = {
    "commands": {"lscpu": LSCPU, "lsblk": LSBLK, "lspci": LSPCI, "dmidecode_memory": DMIDECODE_MEMORY},
    "files": {
        "/proc/meminfo": MEMINFO,
        "/etc/os-release": OS_RELEASE,
        "/proc/sys/kernel/osrelease": "5.15.0-97-generic\n",
        "/sys/class/dmi/id/sys_vendor": "LENOVO\n",
        "/sys/class/dmi/id/product_name": "82A4\n",
        "/sys/class/dmi/id/product_version": "IdeaPad 5 14ITL05\n",
        "/sys/class/dmi/id/bios_vendor": "LENOVO\n",
        "/sys/class/dmi/id/bios_version": "FHCN38WW\n",
        "/sys/class/dmi/id/bios_date": "07/29/2022\n",
        "/sys/class/power_supply/BAT0/status": "Discharging\n",
        "/sys/class/power_supply/BAT0/capacity": "76\n",
        "/sys/class/power_supply/BAT0/energy_full": "49120000\n",
        "/sys/class/power_supply/BAT0/energy_full_design": "56500000\n",
        "/sys/class/power_supply/BAT0/cycle_count": "212\n",
        "/sys/class/power_supply/BAT0/technology": "Li-poly\n",
    },
    "batteries": ["BAT0"],
    # binaries that are NOT installed (e.g. no NVIDIA proprietary driver -> no nvidia-smi)
    "missing": {"nvidia_smi"},
    "privilege": "sudo",
}

# A thin ultrabook: soldered LPDDR4x, no sudo rule, no discrete GPU, no battery cycle counter.
_ULTRABOOK_DMI = """\
Handle 0x0006, DMI type 16, 23 bytes
Physical Memory Array
\tLocation: System Board Or Motherboard
\tUse: System Memory
\tError Correction Type: None
\tMaximum Capacity: 16 GB
\tNumber Of Devices: 2

Handle 0x0007, DMI type 17, 40 bytes
Memory Device
\tSize: 8 GB
\tForm Factor: Row Of Chips
\tLocator: Onboard Channel A
\tType: LPDDR4
\tSpeed: 4267 MT/s
\tManufacturer: Micron
\tPart Number: MT53E1G32D2FW

Handle 0x0008, DMI type 17, 40 bytes
Memory Device
\tSize: 8 GB
\tForm Factor: Row Of Chips
\tLocator: Onboard Channel B
\tType: LPDDR4
\tSpeed: 4267 MT/s
\tManufacturer: Micron
\tPart Number: MT53E1G32D2FW
"""


def _variant(base: dict, **over) -> dict:
    d = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(d.get(k), dict):
            d[k].update(v)
        else:
            d[k] = v
    return d


ULTRABOOK = _variant(
    DEMO_LAPTOP,
    commands={"dmidecode_memory": _ULTRABOOK_DMI,
              "lspci": '00:02.0 "VGA compatible controller" "Intel Corporation" "Iris Xe Graphics" -r01\n'},
    files={"/proc/meminfo": MEMINFO.replace("16114424", "7945216"),
           "/sys/class/power_supply/BAT0/cycle_count": "0\n"},
)

# dmidecode needs root and no rule is configured
NO_PRIVILEGE = _variant(DEMO_LAPTOP, privilege="none")

FIXTURES = {"demo_laptop": DEMO_LAPTOP, "ultrabook": ULTRABOOK, "no_privilege": NO_PRIVILEGE}


class FixtureRunner:
    """Drop-in replacement for SystemRunner that serves recorded data."""

    def __init__(self, fixture: dict) -> None:
        self.fx = fixture

    def available(self, key: str) -> bool:
        if key not in COMMANDS:
            raise NotAllowed(f"command key not in allow-list: {key!r}")
        return key in self.fx["commands"] and key not in self.fx.get("missing", set())

    def privilege_state(self) -> str:
        return self.fx.get("privilege", "none")

    def run(self, key: str) -> CommandResult:
        spec = COMMANDS.get(key)
        if spec is None:
            raise NotAllowed(f"command key not in allow-list: {key!r}")
        label = tuple(spec.variants[0])
        if key in self.fx.get("missing", set()) or key not in self.fx["commands"]:
            return CommandResult(False, error_kind="missing", error=f"{label[0]} is not installed", argv_used=label)
        if spec.privileged and self.privilege_state() == "none":
            return CommandResult(False, error_kind="permission", error="needs root (no passwordless rule configured)", argv_used=label)
        return CommandResult(True, stdout=self.fx["commands"][key], argv_used=label)

    def read_file(self, path: str) -> Optional[str]:
        if not file_allowed(path):
            raise NotAllowed(f"file not in allow-list: {path!r}")
        return self.fx["files"].get(path)

    def list_batteries(self) -> list:
        return list(self.fx.get("batteries", []))
