"""Pipeline runner. Usage: python -m app.main <plan file> [--profile full|quick] [--job-dir DIR]
[--from-stage STAGE] [--to-stage STAGE] [--no-polish] [--redo-vlm]
Stages: parse, vlm, plan, layout, scene, render, polish, export, report. GPU stages run as separate processes.
export writes <job>/final/ (apartment.blend/.glb/.usdc, previews, overlay) and fails the job if validation fails.
For the agentic run (LLM inspects and repairs) use agent.sh / python -m app.agent."""
import argparse, shutil, subprocess, sys, time
from pathlib import Path
from .common import APP, BLENDER, WS, GpuSampler, blender, jload, jsave, load_config, log, run
from .parse import run_parse
from .plan import run_plan
from .layout import run_layout
from .report import write_report

STAGES = ["parse", "vlm", "plan", "layout", "scene", "render", "polish", "export", "report"]
GPU_STAGES = {"vlm", "render", "polish"}


def gpu_present():
    return shutil.which("nvidia-smi") is not None and subprocess.run(["nvidia-smi", "-L"], capture_output=True).returncode == 0


def main():
    ap = argparse.ArgumentParser(description="floor plan -> furnished photoreal renders")
    ap.add_argument("input")
    ap.add_argument("--profile", default="full", choices=["full", "quick"])
    ap.add_argument("--job-dir")
    ap.add_argument("--from-stage", default="parse", choices=STAGES)
    ap.add_argument("--to-stage", default="report", choices=STAGES)
    ap.add_argument("--no-polish", action="store_true")
    ap.add_argument("--redo-vlm", action="store_true")
    ap.add_argument("--wallseg", default="config", choices=["config", "on", "off"],
                    help="trained wall model for scans/photos: config.json setting, on or off")
    a = ap.parse_args()
    inp = Path(a.input).resolve()
    job = Path(a.job_dir) if a.job_dir else WS / "outputs" / f"{inp.stem}_{time.strftime('%Y%m%d-%H%M%S')}"
    (job / "00_input").mkdir(parents=True, exist_ok=True)
    (job / "logs").mkdir(exist_ok=True)
    if not (job / "00_input" / inp.name).exists():
        shutil.copy2(inp, job / "00_input" / inp.name)
    cfg = load_config()
    gpu = gpu_present()
    dev = (jload(APP / "device.json") or {}).get("device", "OPTIX" if gpu else "CPU")
    jsave(job / "run_config.json", {"assets": str(WS / "assets"), "profile": a.profile, "device": dev,
                                    "profile_settings": cfg["profiles"][a.profile], "exposure": cfg["exposure"],
                                    "sun_strength": cfg["sun_strength"], "fill_w_per_m2": cfg["fill_w_per_m2"],
                                    "export": cfg["export"]})
    meta = jload(job / "meta.json", {"stages": {}, "warnings": []})
    meta.update(input=str(inp), profile=a.profile, started=meta.get("started") or time.strftime("%Y-%m-%d %H:%M:%S"))
    sampler = GpuSampler().start() if gpu else None
    failed = None
    for st in STAGES[STAGES.index(a.from_stage):STAGES.index(a.to_stage) + 1]:
        log(f"===== stage {st} =====")
        t0, status = time.time(), "ok"
        stage_gpu = GpuSampler().start() if gpu else None   # VRAM peak of this stage (whole GPU)
        try:
            if st == "parse":
                run_parse(inp, job)
                use = {"on": True, "off": False}.get(a.wallseg, cfg["wallseg"]["enabled"])
                kind = (jload(job / "01_parse/extract.json") or {}).get("kind")
                if use and kind in ("pdf_scan", "photo"):   # optional trained wall model (section 9)
                    try:
                        run([sys.executable, "-m", "app.wallseg", "predict", job], log_file=job / "logs/wallseg.log", cwd=str(WS))
                        used = (jload(job / "01_parse/wallseg.json") or {}).get("used")
                        status = "ok (trained wall model)" if used else "ok (wall model output rejected, classic walls)"
                    except Exception as e:
                        status = f"ok (wall model failed, classic walls used: {e})"
                else:
                    for f in ("wallseg.json", "wallseg_pred.jpg", "wall_mask_classic.png"):
                        (job / "01_parse" / f).unlink(missing_ok=True)
            elif st == "vlm":
                ex = jload(job / "01_parse/extract.json")
                if not ex.get("needs_vlm"):
                    status = "skipped (text read from the file)"
                elif (job / "01_parse/vlm_texts.json").exists() and not a.redo_vlm:
                    status = "reused (01_parse/vlm_texts.json)"
                elif not gpu:
                    status = "skipped (no GPU) - rooms will have no names/areas"
                else:
                    run([sys.executable, "-m", "app.vlm", job], log_file=job / "logs/vlm.log", cwd=str(WS))
            elif st == "plan":
                run_plan(job, cfg)
            elif st == "layout":
                run_layout(job, cfg)
            elif st in ("scene", "render"):
                if st == "scene":
                    try:   # furniture models from the library (03_layout/assets.json); parametric otherwise
                        from .furniture import select_assets
                        sel = select_assets(job, cfg)
                        n = sum(1 for v in sel["items"].values() if v["source"] not in ("parametric", "removed"))
                        status = f"ok ({n} of {len(sel['items'])} furniture items from the model library)"
                    except Exception as e:
                        (job / "03_layout/assets.json").unlink(missing_ok=True)
                        status = f"ok (furniture library not used: {e})"
                blender(f"blender_{st}.py", job)
            elif st == "export":
                from .deliver import deliver
                val = deliver(job, cfg)
                if not val.get("ok"):
                    raise RuntimeError("export validation failed: " + ", ".join(c["name"] for c in val["checks"] if not c["ok"]))
            elif st == "polish":
                if a.no_polish or not cfg["polish"]["enabled"]:
                    status = "skipped (turned off)"
                elif not gpu:
                    status = "skipped (no GPU)"
                else:
                    run([sys.executable, "-m", "app.polish", job], log_file=job / "logs/polish.log", cwd=str(WS))
        except Exception as e:
            if st in ("vlm", "polish"):   # optional AI steps: warn and continue
                status = f"FAILED (continuing without it): {e}"
            else:
                status, failed = f"FAILED: {e}", st
            log(f"stage {st} FAILED: {e}")
        used_gpu = st in GPU_STAGES and status == "ok"
        if st == "render":
            used_gpu = used_gpu and jload(job / "05_render/render.json", {}).get("device") != "CPU"
        meta["stages"][st] = {"seconds": round(time.time() - t0, 1), "gpu": used_gpu, "status": status,
                              "vram_peak_mb": (stage_gpu.stop() if stage_gpu else {}).get("vram_peak_mb")}
        jsave(job / "meta.json", meta)
        if failed:
            break
    meta["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    meta["gpu"] = sampler.stop() if sampler else {}
    jsave(job / "meta.json", meta)
    rep = write_report(job, meta, cfg)
    print(f"\nDONE: {job}\n  rooms: " + "; ".join(f"{r['label'] or r['type']} {r['size_m'][0]:.2f}x{r['size_m'][1]:.2f} m"
                                             for r in rep["rooms"]))
    print(f"  pod time {rep['pod_minutes']} min, GPU stages {rep['gpu_minutes']} min, cost ${rep['cost_usd']}")
    print(f"  report: {job / 'report.md'}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
