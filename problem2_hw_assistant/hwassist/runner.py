"""Safe, read-only system access.

This is the ONLY module that touches the OS. The LLM never sees it and never
supplies a command, a flag, or a path:

* Commands come from a fixed table (COMMANDS). Tools refer to them by key.
* No user/LLM-derived text ever reaches an argv list.
* Commands run with shell=False, absolute binary paths resolved from a fixed
  directory list (not the user's PATH), a scrubbed environment, no stdin, a
  timeout and an output cap.
* Files are read through an exact-path / strict-regex allow-list. Serial
  numbers, UUIDs and anything else identifying are not on the list.
* Privilege: tool code only says `privileged=True`. Only this module knows
  about sudo, and it uses `sudo -n` (never prompts, never hangs). The matching
  sudoers rule pins the exact argv, so the grant is "this one read-only
  command", not "root".
"""
from __future__ import annotations

import getpass
import os
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from typing import Optional

SAFE_PATH_DIRS = ("/usr/local/sbin", "/usr/local/bin", "/usr/sbin", "/usr/bin", "/sbin", "/bin")
SAFE_ENV = {"LC_ALL": "C", "LANG": "C", "PATH": ":".join(SAFE_PATH_DIRS)}
MAX_OUTPUT_BYTES = 2_000_000
MAX_FILE_BYTES = 65_536


@dataclass(frozen=True)
class CommandSpec:
    key: str
    # Alternative argv lists, tried in order if the previous one fails.
    # Lets us cope with older/newer tool versions without any dynamic argv.
    variants: tuple
    privileged: bool = False
    timeout: float = 5.0


COMMANDS = {
    spec.key: spec
    for spec in (
        CommandSpec("lscpu", (("lscpu",),)),
        CommandSpec(
            "lsblk",
            (
                ("lsblk", "-J", "-b", "-o", "NAME,SIZE,TYPE,MOUNTPOINTS,MODEL,ROTA,TRAN,FSTYPE,RM"),
                # util-linux < 2.37 has no MOUNTPOINTS column
                ("lsblk", "-J", "-b", "-o", "NAME,SIZE,TYPE,MOUNTPOINT,MODEL,ROTA,TRAN,FSTYPE,RM"),
            ),
        ),
        CommandSpec("lspci", (("lspci", "-mm"),)),
        CommandSpec(
            "nvidia_smi",
            (("nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits"),),
            timeout=8.0,
        ),
        CommandSpec("dmidecode_memory", (("dmidecode", "-t", "memory"),), privileged=True),
    )
}

ALLOWED_FILES = frozenset(
    {
        "/proc/meminfo",
        "/etc/os-release",
        "/proc/sys/kernel/osrelease",
        "/sys/class/dmi/id/sys_vendor",
        "/sys/class/dmi/id/product_name",
        "/sys/class/dmi/id/product_version",
        "/sys/class/dmi/id/bios_vendor",
        "/sys/class/dmi/id/bios_version",
        "/sys/class/dmi/id/bios_date",
        "/sys/class/dmi/id/board_name",
        "/sys/class/dmi/id/board_vendor",
        "/sys/class/dmi/id/chassis_type",
    }
)
ALLOWED_FILE_PATTERNS = (
    re.compile(
        r"^/sys/class/power_supply/BAT\d+/"
        r"(status|capacity|energy_full|energy_full_design|charge_full|charge_full_design|cycle_count|technology)$"
    ),
)
BATTERY_DIR = "/sys/class/power_supply"
BATTERY_NAME = re.compile(r"^BAT\d+$")


class NotAllowed(Exception):
    """Raised when something outside the allow-list is requested."""


@dataclass
class CommandResult:
    ok: bool
    stdout: str = ""
    # None | "missing" | "permission" | "timeout" | "failed"
    error_kind: Optional[str] = None
    error: str = ""
    argv_used: tuple = field(default_factory=tuple)

    def label(self) -> str:
        return " ".join(self.argv_used) if self.argv_used else ""


def file_allowed(path: str) -> bool:
    return path in ALLOWED_FILES or any(p.match(path) for p in ALLOWED_FILE_PATTERNS)


class SystemRunner:
    """Runs allow-listed, read-only inspection commands on the real machine."""

    def __init__(self) -> None:
        self._priv_state: Optional[str] = None

    # -- discovery -------------------------------------------------------
    @staticmethod
    def _resolve(binary: str) -> Optional[str]:
        return shutil.which(binary, path=SAFE_ENV["PATH"])

    def available(self, key: str) -> bool:
        spec = self._spec(key)
        return any(self._resolve(v[0]) for v in spec.variants)

    @staticmethod
    def _spec(key: str) -> CommandSpec:
        if key not in COMMANDS:
            raise NotAllowed(f"command key not in allow-list: {key!r}")
        return COMMANDS[key]

    # -- privilege -------------------------------------------------------
    def privilege_state(self) -> str:
        """'root' | 'sudo' (scoped NOPASSWD rule works) | 'none'."""
        if self._priv_state:
            return self._priv_state
        if os.geteuid() == 0:
            self._priv_state = "root"
            return "root"
        sudo = self._resolve("sudo")
        dmi = self._resolve("dmidecode")
        if sudo and dmi:
            try:
                # `sudo -n -l <cmd>` only *checks* whether the command is permitted.
                r = subprocess.run(
                    [sudo, "-n", "-l", "--", dmi, "-t", "memory"],
                    capture_output=True, text=True, timeout=3, stdin=subprocess.DEVNULL, env=SAFE_ENV,
                )
                self._priv_state = "sudo" if r.returncode == 0 else "none"
            except Exception:
                self._priv_state = "none"
        else:
            self._priv_state = "none"
        return self._priv_state

    # -- execution -------------------------------------------------------
    def run(self, key: str) -> CommandResult:
        spec = self._spec(key)
        last = CommandResult(False, error_kind="missing", error=f"{spec.variants[0][0]} is not installed")
        for variant in spec.variants:
            binary = self._resolve(variant[0])
            if not binary:
                continue
            argv = [binary, *variant[1:]]
            if spec.privileged and os.geteuid() != 0:
                sudo = self._resolve("sudo")
                if not sudo:
                    return CommandResult(False, error_kind="permission", error="needs root and sudo is not installed", argv_used=tuple(variant))
                argv = [sudo, "-n", "--", *argv]
            try:
                p = subprocess.run(
                    argv, shell=False, capture_output=True, text=True, errors="replace",
                    timeout=spec.timeout, stdin=subprocess.DEVNULL, env=SAFE_ENV,
                )
            except subprocess.TimeoutExpired:
                return CommandResult(False, error_kind="timeout", error=f"timed out after {spec.timeout}s", argv_used=tuple(variant))
            except OSError as e:
                last = CommandResult(False, error_kind="failed", error=str(e), argv_used=tuple(variant))
                continue
            if p.returncode == 0:
                return CommandResult(True, stdout=p.stdout[:MAX_OUTPUT_BYTES], argv_used=tuple(variant))
            err = (p.stderr or "").strip()
            low = err.lower()
            if spec.privileged and ("password is required" in low or "not allowed" in low or "permission denied" in low
                                    or "may not run sudo" in low or "/dev/mem" in low or "operation not permitted" in low):
                return CommandResult(False, error_kind="permission", error="needs root (no passwordless rule configured)", argv_used=tuple(variant))
            last = CommandResult(False, error_kind="failed", error=err[:300] or f"exit code {p.returncode}", argv_used=tuple(variant))
        return last

    # -- files -----------------------------------------------------------
    def read_file(self, path: str) -> Optional[str]:
        if not file_allowed(path):
            raise NotAllowed(f"file not in allow-list: {path!r}")
        try:
            with open(path, "rb") as f:
                return f.read(MAX_FILE_BYTES).decode("utf-8", errors="replace")
        except OSError:
            return None

    def list_batteries(self) -> list:
        try:
            return sorted(n for n in os.listdir(BATTERY_DIR) if BATTERY_NAME.match(n))
        except OSError:
            return []


def sudoers_snippet(user: Optional[str] = None) -> str:
    """The exact rule the user can install (via visudo) to enable root-only reads."""
    user = user or getpass.getuser()
    dmi = shutil.which("dmidecode", path=SAFE_ENV["PATH"]) or "/usr/sbin/dmidecode"
    return (
        "# /etc/sudoers.d/hwassist   (install with: sudo visudo -f /etc/sudoers.d/hwassist)\n"
        "# Lets this user run exactly ONE read-only command as root, with exactly these arguments.\n"
        f"{user} ALL=(root) NOPASSWD: {dmi} -t memory\n"
    )
