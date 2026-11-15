import subprocess, sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]

def run(*args, inp=None):
    return subprocess.run([sys.executable, "-m", "hwassist", *args], cwd=ROOT, capture_output=True, text=True, input=inp, timeout=30)

def test_once_demo():
    r = run("--demo", "--backend", "rules", "--once", "Can I add more RAM, and how much?")
    assert r.returncode == 0 and "64 GB" in r.stdout

def test_interactive_followup_and_commands():
    r = run("--demo", "--backend", "rules", inp="how much ram do I have\nand the gpu?\n/doctor\n/tools\n/quit\n")
    assert r.returncode == 0 and "Graphics" in r.stdout and "privilege" in r.stdout and "get_memory_info" in r.stdout

def test_cloud_requires_explicit_flag():
    r = run("--demo", "--backend", "openai", "--once", "hi")
    assert r.returncode != 0 and "--allow-cloud" in (r.stderr + r.stdout)

def test_print_sudoers():
    assert "NOPASSWD" in run("--print-sudoers").stdout

def test_auto_falls_back_to_rules_without_ollama():
    r = run("--demo", "--once", "what cpu")
    assert r.returncode == 0 and "i5-1135G7" in r.stdout
