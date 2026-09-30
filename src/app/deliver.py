"""Final deliverable of a job -> <job>/final/: apartment.blend (primary), apartment.glb, apartment.usdc,
previews/*.png, overlay.png, layout.png, coverage.md/json, validation.json (+ report.md written by the caller).
blender_export.py builds the files, validate_export.py reopens them headless; the job fails if a check fails."""
import shutil
from pathlib import Path
from .common import blender, jload, log
from .furniture import coverage


def collect(job):
    job, fin = Path(job), Path(job) / "final"
    (fin / "previews").mkdir(parents=True, exist_ok=True)
    for old in (fin / "previews").glob("*.png"):
        old.unlink()
    src = job / "06_polish" if list((job / "06_polish").glob("*.png")) else job / "05_render"
    for p in sorted(src.glob("*.png")):
        if not p.name.endswith("_depth.png"):
            shutil.copy2(p, fin / "previews" / p.name)
    for rel, name in (("02_plan/overlay.png", "overlay.png"), ("03_layout/layout.png", "layout.png")):
        if (job / rel).exists():
            shutil.copy2(job / rel, fin / name)


def deliver(job, cfg=None):
    """Export + validate. Returns the validation dict ({'ok': bool, 'checks': [...]})."""
    job = Path(job)
    (job / "final").mkdir(parents=True, exist_ok=True)
    (job / "final/validation.json").unlink(missing_ok=True)
    blender("blender_export.py", job)
    try:
        blender("validate_export.py", job)
    except RuntimeError as e:
        log(f"export validation failed: {e}")
    val = jload(job / "final/validation.json") or {"ok": False, "checks": [{"name": "validate_export.py", "ok": False,
                                                                           "detail": "no validation.json written"}]}
    collect(job)
    coverage(job)
    bad = [c["name"] for c in val.get("checks", []) if not c.get("ok")]
    log(f"deliver: validation {'OK' if val.get('ok') else 'FAILED: ' + ', '.join(bad)}")
    return val
