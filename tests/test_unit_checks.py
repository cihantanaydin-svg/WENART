"""pytest wrapper around `python -m app.testplans unit` (the CPU checks setup.sh has always shipped).
The checks run once per session, in order, in a temporary workspace; each named check is one test."""
import re
import subprocess
import sys

import pytest

from conftest import ROOT, find_blender, pipeline_env

CHECKS = ["textnorm", "plan+layout dxf", "plan+layout pdf_vector", "patches", "agent mock + replay",
          "catalog + licences", "llm server + planner", "blender scene/export/validate"]
LINE = re.compile(r"^(PASS|SKIP|FAIL)  (.+?)\s+(\d+\.\d)s  (.*)$")


@pytest.fixture(scope="session")
def unit_run(tmp_path_factory):
    ws = tmp_path_factory.mktemp("ws")
    r = subprocess.run([sys.executable, "-m", "app.testplans", "unit"], cwd=ws, capture_output=True, text=True,
                       env=pipeline_env(ws, find_blender()), timeout=3600)
    out = r.stdout.split("===== UNIT CHECKS (CPU) =====")[-1]
    results = {}
    for line in out.splitlines():
        m = LINE.match(line)
        if m:
            results[m.group(2).strip()] = (m.group(1), m.group(4), line)
    return r, results


def find(results, name):
    return next((v for k, v in results.items() if k == name or k.startswith(name + " ")), None)


@pytest.mark.parametrize("name", CHECKS)
def test_unit_check(unit_run, name):
    r, results = unit_run
    res = find(results, name)
    assert res is not None, f"check '{name}' not reported:\n{r.stdout[-3000:]}\n{r.stderr[-3000:]}"
    status, msg, _ = res
    if status == "SKIP":
        pytest.skip(msg)
    assert status == "PASS", f"{name}: {msg}\n{r.stdout[-3000:]}"


def test_unit_exit_code(unit_run):
    r, results = unit_run
    assert r.returncode == 0, "\n".join(v[2] for v in results.values()) + "\n" + r.stderr[-3000:]
    assert "UNIT CHECKS: PASS" in r.stdout
    assert {k.split(" (")[0] for k in results} >= set(CHECKS), sorted(results)


def test_src_is_the_package():
    """The tests must exercise src/app, the files the bundle embeds (not an installed copy)."""
    r = subprocess.run([sys.executable, "-c", "import app, sys; print(app.__file__)"], capture_output=True,
                       text=True, env=pipeline_env(ROOT / "unused"))
    assert r.returncode == 0 and r.stdout.strip().startswith(str(ROOT / "src/app")), r.stdout + r.stderr
