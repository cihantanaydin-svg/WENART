"""Shared test setup: the pipeline package is src/app, every test run gets its own temporary workspace
(PIPE_WS), and Blender scripts run in a Python with the bpy 5.2.2 module when one is available."""
import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"


def find_blender():
    """PIPE_BLENDER if it exists, else the dev bpy venv (make dev-bpy), else None (Blender checks skip)."""
    for cand in (os.environ.get("PIPE_BLENDER"), ROOT / ".venv-bpy/bin/python", shutil.which("blender")):
        if cand and Path(cand).exists():
            return str(cand)
    return None


def pipeline_env(ws, blender=None):
    """Environment for running `python -m app.<module>` against src/ in the workspace ws."""
    env = dict(os.environ, PIPE_WS=str(ws), PYTHONPATH=os.pathsep.join(
        [str(SRC)] + ([os.environ["PYTHONPATH"]] if os.environ.get("PYTHONPATH") else [])))
    # a path that does not exist makes testplans skip the Blender checks instead of failing them
    env["PIPE_BLENDER"] = blender or str(Path(ws) / "no-blender")
    env["HF_HUB_OFFLINE"] = "1"
    return env
