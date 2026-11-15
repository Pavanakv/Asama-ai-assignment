import json
from hwassist import parsers as P
from hwassist.fixtures import DMIDECODE_MEMORY, LSCPU, LSBLK, LSPCI, MEMINFO, _ULTRABOOK_DMI


def test_size_units():
    assert P.size_to_gb("16 GB") == 16
    assert P.size_to_gb("8192 MB") == 8
    assert P.size_to_gb("4 TB") == 4096
    assert P.size_to_gb("No Module Installed") is None
    assert P.size_to_gb("Unknown") is None


def test_dmidecode_per_block_not_global_regex():
    d = P.parse_dmidecode_memory(DMIDECODE_MEMORY)
    assert d["max_capacity_gb"] == 64 and d["slots_total"] == 2
    inst = [m for m in d["modules"] if m["installed"]]
    assert len(inst) == 1 and inst[0]["size_gb"] == 16 and inst[0]["type"] == "DDR4" and not inst[0]["soldered"]
    assert d["modules"][1]["installed"] is False


def test_dmidecode_ignores_other_dmi_types_with_size_lines():
    noisy = DMIDECODE_MEMORY + "\nHandle 0x0040, DMI type 22, 26 bytes\nPortable Battery\n\tSize: 99 GB\n"
    d = P.parse_dmidecode_memory(noisy)
    assert sum(m["size_gb"] or 0 for m in d["modules"]) == 16


def test_dmidecode_never_collects_serials():
    blob = json.dumps(P.parse_dmidecode_memory(DMIDECODE_MEMORY))
    assert "0A1B2C3D" not in blob and "serial" not in blob.lower()


def test_dmidecode_soldered_detected():
    d = P.parse_dmidecode_memory(_ULTRABOOK_DMI)
    assert all(m["soldered"] for m in d["modules"]) and d["max_capacity_gb"] == 16


def test_dmidecode_mb_sizes_and_empty_input():
    d = P.parse_dmidecode_memory("Handle 0x1, DMI type 17, 40 bytes\nMemory Device\n\tSize: 8192 MB\n\tLocator: A\n")
    assert d["modules"][0]["size_gb"] == 8
    assert P.parse_dmidecode_memory("")["slots_total"] is None


def test_lscpu():
    c = P.parse_lscpu(LSCPU)
    assert c["physical_cores"] == 4 and c["logical_cpus"] == 8 and c["max_ghz"] == 4.2
    assert c["l3_cache"] == "8 MiB" and "i5-1135G7" in c["model_name"]


def test_lscpu_missing_model_name_is_none_not_invented():
    c = P.parse_lscpu("Architecture: aarch64\nCPU(s): 4\nModel: 1\n")
    assert c["model_name"] is None and c["logical_cpus"] == 4 and c["physical_cores"] is None


def test_meminfo():
    m = P.parse_meminfo(MEMINFO)
    assert m["usable_gib"] == 15.37 and m["used_gib"] == round(15.37 - 10.37, 2)


def test_lsblk_kinds_and_filtering():
    d = P.parse_lsblk(LSBLK)["drives"]
    names = {x["name"]: x for x in d}
    assert set(names) == {"nvme0n1", "sda"}  # zram filtered out
    assert names["nvme0n1"]["kind"] == "NVMe SSD" and names["sda"]["kind"] == "HDD"


def test_lsblk_old_util_linux_string_values():
    old = json.dumps({"blockdevices": [{"name": "sda", "size": "500107862016", "type": "disk", "rota": "0", "tran": "sata", "rm": "0"}]})
    d = P.parse_lsblk(old)["drives"][0]
    assert d["kind"] == "SSD" and d["size_gb"] == 500.1


def test_lsblk_virtual_disk_not_claimed_hdd():
    vm = json.dumps({"blockdevices": [{"name": "vda", "size": 274877906944, "type": "disk", "rota": True, "tran": None}]})
    d = P.parse_lsblk(vm)["drives"][0]
    assert d["kind"] == "virtual disk" and "note" in d


def test_lsblk_garbage_does_not_crash():
    assert P.parse_lsblk("not json")["drives"] == []


def test_lspci_gpu_and_network():
    g = P.parse_lspci_gpus(LSPCI)
    assert [(x["vendor"].split()[0], x["likely_type"]) for x in g] == [("Intel", "integrated"), ("NVIDIA", "discrete")]
    assert len(P.parse_lspci_network(LSPCI)) == 2


def test_nvidia_smi():
    g = P.parse_nvidia_smi("NVIDIA GeForce MX350, 2048, 535.54.03\n")
    assert g[0]["vram_gib"] == 2.0


def test_battery_health_and_zero_cycles():
    b = P.parse_battery({"status": "Charging", "capacity": "50", "energy_full": "40000000", "energy_full_design": "50000000", "cycle_count": "0"})
    assert b["health_percent"] == 80.0 and b["design_wh"] == 50.0 and b["cycle_count"] is None


def test_untrusted_strings_sanitised():
    assert P.clean("Evil\x00\x1b[31m name") == "Evil[31m name"
    assert len(P.clean("x" * 1000)) == 120
