"""Stage 5 - render every camera with Cycles on the GPU (OptiX, then CUDA, then CPU).
Run: blender -b --factory-startup --python-exit-code 1 -P blender_render.py -- <job_dir>
Writes 05_render/<camera>.png, <camera>_depth.npy (for the polish step) and render.json."""
import bpy, json, math, sys, time
from pathlib import Path
import numpy as np

ARGS = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
JOB = Path(ARGS[0])
RC = json.loads((JOB / "run_config.json").read_text())
PROF = RC["profile_settings"]
OUT = JOB / "05_render"
OUT.mkdir(parents=True, exist_ok=True)
bpy.ops.wm.open_mainfile(filepath=str(JOB / "04_scene/scene.blend"))
SC = bpy.context.scene


def setup_device(pref):
    order = [] if pref == "CPU" else [pref] + [d for d in ("OPTIX", "CUDA") if d != pref]
    prefs = bpy.context.preferences.addons["cycles"].preferences
    for typ in order:
        try:
            prefs.compute_device_type = typ
            prefs.refresh_devices()
        except Exception:
            continue
        devs = [d for d in prefs.devices if d.type == typ]
        if devs:
            for d in prefs.devices:
                d.use = d.type == typ
            SC.cycles.device = "GPU"
            return typ, [d.name for d in devs]
    SC.cycles.device = "CPU"
    return "CPU", []


def depth_map(cam, W=320, H=180):
    """Distance from the camera for a coarse pixel grid (ray casts) - control image for SDXL."""
    dg = bpy.context.evaluated_depsgraph_get()
    mw = cam.matrix_world
    o = mw.translation
    tr, br, bl, tl = [mw @ v for v in cam.data.view_frame(scene=SC)]
    d = np.full((H, W), 100.0, np.float32)
    for j in range(H):
        a, b = tl.lerp(bl, (j + 0.5) / H), tr.lerp(br, (j + 0.5) / H)
        for i in range(W):
            v = (a.lerp(b, (i + 0.5) / W) - o).normalized()
            hit, loc, *_ = SC.ray_cast(dg, o, v, distance=100.0)
            if hit:
                d[j, i] = (loc - o).length
    return d


device, gpus = setup_device(RC.get("device", "OPTIX"))
c = SC.cycles
c.samples = PROF["max_samples"]
c.use_adaptive_sampling = True
c.adaptive_threshold = PROF["noise_threshold"]
c.time_limit = PROF["time_limit_s"]
c.use_denoising = True
c.denoiser = "OPENIMAGEDENOISE"
if hasattr(c, "denoising_use_gpu"):
    c.denoising_use_gpu = device != "CPU"
c.max_bounces, c.diffuse_bounces, c.glossy_bounces = 8, 4, 4
c.transmission_bounces, c.transparent_max_bounces = 8, 8
c.sample_clamp_indirect, c.blur_glossy = 10.0, 1.0
c.caustics_reflective = c.caustics_refractive = False
SC.render.resolution_x, SC.render.resolution_y, SC.render.resolution_percentage = PROF["width"], PROF["height"], 100
SC.render.image_settings.file_format = "PNG"
SC.render.image_settings.color_depth = "8"
cams = json.loads((JOB / "04_scene/cameras.json").read_text(encoding="utf-8"))["cameras"]
views = []
for cam in cams:
    ob = bpy.data.objects.get(cam["name"])
    if ob is None:
        continue
    SC.camera = ob
    SC.render.filepath = str(OUT / f"{cam['name']}.png")
    t0 = time.time()
    bpy.ops.render.render(write_still=True)
    t1 = time.time()
    np.save(OUT / f"{cam['name']}_depth.npy", depth_map(ob))
    views.append({"name": cam["name"], "room": cam["room"], "room_type": cam["room_type"],
                  "render_s": round(t1 - t0, 1), "depth_s": round(time.time() - t1, 1)})
    print(f"RENDERED {cam['name']} in {t1 - t0:.1f}s on {device}", flush=True)
json.dump({"device": device, "gpus": gpus, "samples": PROF["max_samples"], "resolution": [PROF["width"], PROF["height"]],
           "views": views}, open(OUT / "render.json", "w"), indent=1)
print(f"RENDER_OK views={len(views)} device={device}")
