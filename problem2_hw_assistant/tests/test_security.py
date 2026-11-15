import os
import pytest
from hwassist import runner as R
from hwassist.fixtures import FIXTURES, FixtureRunner
from hwassist.tools import Registry, BY_NAME


def test_every_command_is_fixed_argv_and_read_only():
    for spec in R.COMMANDS.values():
        for v in spec.variants:
            assert all(isinstance(a, str) for a in v)
            assert v[0] in {"lscpu", "lsblk", "lspci", "nvidia-smi", "dmidecode"}
            assert not any(ch in "".join(v) for ch in ";|&$`<>\n")


def test_only_one_privileged_command_and_it_is_exactly_dmidecode_memory():
    priv = [s for s in R.COMMANDS.values() if s.privileged]
    assert len(priv) == 1 and priv[0].variants == (("dmidecode", "-t", "memory"),)


def test_unknown_command_key_rejected():
    r = R.SystemRunner()
    for bad in ("rm -rf /", "lscpu; id", "../bin/sh", "dmidecode"):
        with pytest.raises(R.NotAllowed):
            r.run(bad)


@pytest.mark.parametrize("path", ["/etc/shadow", "/etc/passwd", "/proc/meminfo/../../etc/shadow", "/sys/class/dmi/id/product_serial",
                                  "/sys/class/dmi/id/product_uuid", "/sys/class/power_supply/BAT0/../../../etc/shadow",
                                  "/root/.ssh/id_rsa", "/proc/self/environ", "relative/path", ""])
def test_file_allowlist_blocks_sensitive_and_traversal(path):
    assert not R.file_allowed(path)
    with pytest.raises(R.NotAllowed):
        R.SystemRunner().read_file(path)
    with pytest.raises(R.NotAllowed):
        FixtureRunner(FIXTURES["demo_laptop"]).read_file(path)


def test_allowed_files_work():
    assert R.file_allowed("/proc/meminfo") and R.file_allowed("/sys/class/power_supply/BAT1/status")


def test_subprocess_never_uses_shell(monkeypatch):
    seen = {}
    import subprocess
    real = subprocess.run
    def spy(argv, **kw):
        seen["argv"], seen["kw"] = argv, kw
        return real(argv, **kw)
    monkeypatch.setattr(subprocess, "run", spy)
    R.SystemRunner().run("lscpu")
    assert isinstance(seen["argv"], list) and seen["kw"].get("shell") is False
    assert seen["kw"]["stdin"] == subprocess.DEVNULL and seen["kw"]["env"]["LC_ALL"] == "C"
    assert seen["kw"]["timeout"] > 0


def test_sudo_is_non_interactive_and_pins_argv(monkeypatch):
    import subprocess
    calls = []
    monkeypatch.setattr(os, "geteuid", lambda: 1000)
    monkeypatch.setattr(R.shutil, "which", lambda b, path=None: f"/usr/sbin/{b}")
    monkeypatch.setattr(subprocess, "run", lambda argv, **kw: calls.append(argv) or subprocess.CompletedProcess(argv, 1, "", "sudo: a password is required"))
    res = R.SystemRunner().run("dmidecode_memory")
    assert calls[0] == ["/usr/sbin/sudo", "-n", "--", "/usr/sbin/dmidecode", "-t", "memory"]
    assert not res.ok and res.error_kind == "permission"


def test_missing_binary_reported_not_crashed(monkeypatch):
    monkeypatch.setattr(R.shutil, "which", lambda b, path=None: None)
    res = R.SystemRunner().run("lscpu")
    assert not res.ok and res.error_kind == "missing"


def test_sudoers_snippet_is_scoped():
    s = R.sudoers_snippet("alice")
    assert "alice ALL=(root) NOPASSWD:" in s and s.strip().endswith("dmidecode -t memory") and "ALL\n" not in s.split("NOPASSWD:")[1]


# ---- tool layer ----
def test_tools_have_no_string_parameters_so_nothing_can_be_injected():
    for t in BY_NAME.values():
        for spec in t.properties.values():
            assert spec["type"] == "boolean"


def reg(name="demo_laptop"):
    return Registry(FixtureRunner(FIXTURES[name]))


def test_registry_rejects_unknown_tool_extra_args_and_bad_types():
    r = reg()
    assert not r.call("run_command", {"cmd": "id"})["ok"]
    assert not r.call("get_cpu_info", {"cmd": "id"})["ok"]
    assert not r.call("get_memory_info", {"include_empty_slots": "banana"})["ok"]
    assert not r.call("get_memory_info", ["not", "a", "dict"])["ok"]
    assert r.call("get_memory_info", {"include_empty_slots": "true"})["ok"]  # small models send strings


def test_hidden_tools_cannot_be_called():
    r = reg()
    assert "get_battery_info" in [t.name for t in r.tools]
    class NoPci(FixtureRunner):
        def available(self, key): return key != "lspci" and super().available(key)
    r2 = Registry(NoPci(FIXTURES["demo_laptop"]))
    assert "get_gpu_info" in r2.unavailable and not r2.call("get_gpu_info", {})["ok"]
