"""Optional, OFF by default (setup.sh GEN3D=1): generate furniture models that the library lacks with an
open-weights image-to-3D model (TRELLIS.2-4B) on the 48 GB GPU. Results go through the same catalog, Blender
normalisation and QA as every other model. Nothing is downloaded unless setup.sh ran with GEN3D=1.
  python -m app.gen3d plan                         which furniture types have no model (would be generated)
  python -m app.gen3d run [--types sofa,desk] [--max N] [--force]
Runs ALONE on the GPU: stop any LLM server first (python -m app.llm_server stop); agent.sh does not call it.
Input image per type: assets/gen3d_inputs/<type>/*.png|jpg (your own photo/render of the piece), else an SDXL
product image made with the SDXL base that setup.sh downloads for the polish step.
LICENCES (checked for TRELLIS.2 commit 75fbf01, see CHANGELOG): TRELLIS.2 code and weights MIT; its GLB export
imports NVIDIA nvdiffrast (NVIDIA Source Code License: non-commercial, research/evaluation only); the image
encoder is Meta DINOv3 (DINOv3 License, gated download - needs HF_TOKEN at setup); background removal BiRefNet
(MIT). Treat generated models as evaluation-only until the nvdiffrast licence is cleared for your use."""
import argparse, json, subprocess, sys, time
from pathlib import Path
from .common import WS, gpu_used_mb, jload, jsave, load_config, log
from . import furniture

WORDS = {"sofa": "three-seat fabric sofa", "armchair": "upholstered armchair", "coffee_table": "low coffee table",
         "dining_table": "rectangular dining table", "desk": "writing desk", "nightstand": "bedside nightstand",
         "bistro_table": "small round bistro table", "wardrobe": "two-door wardrobe", "bookshelf": "tall bookshelf",
         "shoe_cabinet": "low shoe cabinet", "tv_unit": "low tv media console", "bed": "double bed with headboard",
         "floor_lamp": "floor lamp", "plant": "potted indoor plant", "chair": "dining chair", "fridge": "refrigerator",
         "washer": "front-loading washing machine", "toilet": "toilet", "bathtub": "bathtub",
         "vanity": "bathroom vanity with basin", "basin_small": "small wash basin", "shower": "shower tray with glass screen"}
TARGET_HEIGHT = {"sofa": 0.85, "armchair": 0.80, "coffee_table": 0.42, "dining_table": 0.75, "desk": 0.75,
                 "nightstand": 0.50, "bistro_table": 0.72, "wardrobe": 2.20, "bookshelf": 1.80, "shoe_cabinet": 1.00,
                 "tv_unit": 1.30, "bed": 1.05, "floor_lamp": 1.60, "plant": 1.10, "chair": 0.85, "fridge": 1.90,
                 "washer": 0.85, "toilet": 0.80, "bathtub": 0.58, "vanity": 0.85, "basin_small": 0.85, "shower": 2.00}


def missing_types(cfg):
    cov = furniture.load_catalog(cfg).get("coverage", {})
    return [t for t in furniture.TYPES if not sum(cov.get(t, {}).values())]


def input_image(cfg, typ, out_dir):
    """Your reference image if there is one, else an SDXL product image (GPU, main venv)."""
    user = Path(cfg["gen3d"]["inputs"]) / typ
    for f in sorted(user.glob("*")) if user.exists() else []:
        if f.suffix.lower() in (".png", ".jpg", ".jpeg"):
            return f, "user image"
    out = out_dir / f"{typ}.png"
    if out.exists():
        return out, "sdxl (cached)"
    import torch
    from diffusers import AutoencoderKL, StableDiffusionXLPipeline
    vae = AutoencoderKL.from_pretrained(cfg["models"]["vae"], torch_dtype=torch.float16)
    pipe = StableDiffusionXLPipeline.from_pretrained(cfg["models"]["sdxl"], vae=vae, variant="fp16",
                                                     torch_dtype=torch.float16).to("cuda")
    img = pipe(prompt=f"studio product photo of a single {WORDS.get(typ, typ)}, {cfg['style']}, three-quarter front "
                      "view, the whole object visible, plain white background, soft even light",
               negative_prompt="people, text, watermark, cropped, multiple objects, room, floor pattern",
               num_inference_steps=30, guidance_scale=6.0, width=1024, height=1024,
               generator=torch.Generator("cuda").manual_seed(7)).images[0]
    out_dir.mkdir(parents=True, exist_ok=True)
    img.save(out)
    del pipe
    torch.cuda.empty_cache()
    return out, "sdxl"


def run(types=None, max_items=None, force=False):
    cfg = load_config()
    g = cfg["gen3d"]
    py = Path(g["venv"]) / "bin/python"
    if not g["enabled"] or not py.exists():
        raise SystemExit("GEN3D is off: run setup.sh with GEN3D=1 (48 GB GPU, ~40 GB disk, licence notes in CHANGELOG)")
    used = gpu_used_mb() or 0
    if used > 2500:
        raise SystemExit(f"{used} MB of GPU memory in use - image-to-3D must run alone (stop the LLM server first)")
    todo = types or missing_types(cfg)
    todo = [t for t in todo if t in furniture.TYPES and t not in furniture.PARAMETRIC_ONLY][:max_items or g["max_items"]]
    lib = furniture.lib_dir(cfg) / "generated"
    jobs = []
    for t in todo:
        out = lib / t / f"trellis2_{t}.glb"
        if out.exists() and not force:
            log(f"gen3d: {t} exists ({out.name})")
            continue
        try:
            img, how = input_image(cfg, t, lib / "_inputs")
        except Exception as e:     # e.g. no image of yours and SDXL not downloaded (SKIP_POLISH_MODELS=1)
            log(f"gen3d: {t} skipped - no input image ({type(e).__name__}: {e})")
            continue
        jobs.append({"type": t, "image": str(img), "input": how, "out": str(out)})
    if not jobs:
        log("gen3d: nothing to generate")
        return furniture.build_catalog(cfg)
    spec = lib / "_jobs.json"
    jsave(spec, {"model": g["model"], "decimation_target": g["decimation_target"], "texture_size": g["texture_size"],
                 "jobs": jobs})
    env = {"PYTHONPATH": g["repo"], "HF_HUB_OFFLINE": "1", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
           "OPENCV_IO_ENABLE_OPENEXR": "1"}
    import os
    t0 = time.time()
    r = subprocess.run([str(py), str(Path(__file__).parent / "gen3d_worker.py"), str(spec)],
                       env=dict(os.environ, **env), cwd=g["repo"])
    log(f"gen3d worker finished with code {r.returncode} in {time.time() - t0:.0f}s")
    res = jload(spec.with_suffix(".result.json"), {"results": []})
    for x in res["results"]:
        if not x.get("ok"):
            log(f"gen3d {x['type']}: FAILED {x.get('error')}")
            continue
        try:
            info = furniture.inspect_gltf(x["out"])
        except Exception as e:
            log(f"gen3d {x['type']}: unreadable output {e}")
            continue
        h = info["dims_import"][2]
        jsave(Path(x["out"]).with_suffix(".json"), {
            "type": x["type"], "name": f"generated {x['type']}", "licence": "generated", "author": g["model"],
            "unit_scale": round(TARGET_HEIGHT.get(x["type"], 1.0) / max(h, 1e-6), 5), "front_axis": "-Y",
            "tags": [x["type"], "generated"],
            "notes": f"{time.strftime('%Y-%m-%d')} {g['model']} (MIT) from {x['input']} {Path(x['image']).name}; GLB "
                     "export uses nvdiffrast (NVIDIA non-commercial licence) - evaluation only; front axis assumed -Y "
                     "(UNVERIFIED for generated models, check the renders)",
            "seconds": x.get("seconds"), "vram_peak_mb": x.get("vram_peak_mb")})
        log(f"gen3d {x['type']}: ok ({info['triangles']} triangles, {x.get('seconds')} s)")
    return furniture.build_catalog(cfg)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("plan")
    p = sub.add_parser("run")
    p.add_argument("--types", default="")
    p.add_argument("--max", type=int)
    p.add_argument("--force", action="store_true")
    a = ap.parse_args()
    if a.cmd == "plan":
        print("furniture types without any library model:", ", ".join(missing_types(load_config())) or "none")
        return
    run([t for t in a.types.split(",") if t] or None, a.max, a.force)
    furniture.report()


if __name__ == "__main__":
    main()
