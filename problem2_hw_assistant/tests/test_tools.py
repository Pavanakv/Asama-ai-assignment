from hwassist.fixtures import FIXTURES, FixtureRunner
from hwassist.tools import Registry


def call(fx, tool, **a):
    return Registry(FixtureRunner(FIXTURES[fx])).call(tool, a)


def test_memory_upgrade_example_from_the_brief():
    r = call("demo_laptop", "get_memory_info")
    d = r["data"]
    assert (d["installed_gb"], d["slots_used"], d["slots_total"], d["max_capacity_gb"]) == (16, 1, 2, 64)
    assert d["upgrade"]["possible"] and d["upgrade"]["mode"] == "add_module" and d["upgrade"]["max_single_module_gb"] == 32
    assert d["upgrade"]["headroom_gb"] == 48 and d["upgrade"]["compatible_with"]["type"] == "DDR4"


def test_soldered_memory_not_upgradeable():
    u = call("ultrabook", "get_memory_info")["data"]["upgrade"]
    assert u["possible"] is False and u["soldered_modules"] == 2


def test_no_privilege_degrades_honestly():
    r = call("no_privilege", "get_memory_info")
    assert r["ok"] and r["data"]["usable_gib"] == 15.37 and r["data"]["max_capacity_gb"] is None
    assert any("root" in l for l in r["limitations"]) and "upgrade" not in r["data"]


def test_empty_slots_only_when_asked():
    inst = call("demo_laptop", "get_memory_info")["data"]["modules"]
    allm = call("demo_laptop", "get_memory_info", include_empty_slots=True)["data"]["modules"]
    assert len(inst) == 1 and len(allm) == 2


def test_gpu_notes_missing_nvidia_smi():
    r = call("demo_laptop", "get_gpu_info")
    assert len(r["data"]["gpus"]) == 2 and any("nvidia-smi" in l for l in r["limitations"])


def test_battery_health():
    b = call("demo_laptop", "get_battery_info")["data"]["batteries"][0]
    assert b["health_percent"] == 86.9 and b["cycle_count"] == 212


def test_battery_zero_cycles_not_reported_as_zero():
    assert call("ultrabook", "get_battery_info")["data"]["batteries"][0]["cycle_count"] is None


def test_cache_and_internal_errors_contained(monkeypatch):
    reg = Registry(FixtureRunner(FIXTURES["demo_laptop"]))
    a = reg.call("get_cpu_info", {})
    assert reg.call("get_cpu_info", {}) is a
    from hwassist import parsers
    monkeypatch.setattr(parsers, "parse_lsblk", lambda *a, **k: 1 / 0)
    r = Registry(FixtureRunner(FIXTURES["demo_laptop"])).call("get_storage_info", {})
    assert not r["ok"] and "internal error" in r["limitations"][-1]
