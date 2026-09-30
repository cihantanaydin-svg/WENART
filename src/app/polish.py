"""Stage 6 (GPU, optional) - light photo polish: SDXL img2img guided by the render's depth.
A gate compares edges before/after: if lines moved (hallucination), the raw render is kept.
Run: python -m app.polish <job_dir>"""
import sys, time
from pathlib import Path
import cv2
import numpy as np
from .common import jload, jsave, load_config, log

WORDS = {"living": "living room", "bedroom": "bedroom", "kitchen": "kitchen", "bathroom": "bathroom",
         "wc": "small guest toilet", "hall": "entrance hallway", "study": "home office", "dining": "dining room"}
NEG = ("cartoon, illustration, painting, cgi, 3d render, blurry, distorted lines, deformed furniture, "
       "extra furniture, people, text, watermark, oversaturated")


def edge_score(a, b):
    """F1 overlap of Canny edges (2 px tolerance) between two images, 1.0 = same lines."""
    def edges(x):
        x = cv2.resize(x, (960, int(960 * x.shape[0] / x.shape[1])), interpolation=cv2.INTER_AREA)
        return cv2.Canny(cv2.GaussianBlur(cv2.cvtColor(x, cv2.COLOR_BGR2GRAY), (3, 3), 0), 60, 150) > 0
    ea, eb = edges(a), edges(b)
    if ea.sum() < 50:
        return 1.0
    k = np.ones((5, 5), np.uint8)
    da, db = cv2.dilate(ea.astype(np.uint8), k) > 0, cv2.dilate(eb.astype(np.uint8), k) > 0
    p, r = (eb & da).sum() / max(eb.sum(), 1), (ea & db).sum() / max(ea.sum(), 1)
    return float(2 * p * r / max(p + r, 1e-6))


def depth_image(npy, size):
    d = np.load(npy)
    inv = 1.0 / np.maximum(d, 0.1)
    inv = (inv - inv.min()) / max(float(inv.max() - inv.min()), 1e-6)
    g = cv2.resize((inv * 255).astype(np.uint8), size, interpolation=cv2.INTER_CUBIC)
    return np.dstack([g] * 3)


def main(job):
    import torch
    from PIL import Image
    from diffusers import AutoencoderKL, ControlNetModel, StableDiffusionXLControlNetImg2ImgPipeline
    job = Path(job)
    cfg, rc = load_config(), jload(job / "run_config.json")
    pc, prof = cfg["polish"], rc["profile_settings"]
    t0 = time.time()
    cn = ControlNetModel.from_pretrained(cfg["models"]["controlnet"], variant="fp16", torch_dtype=torch.float16)
    vae = AutoencoderKL.from_pretrained(cfg["models"]["vae"], torch_dtype=torch.float16)
    pipe = StableDiffusionXLControlNetImg2ImgPipeline.from_pretrained(
        cfg["models"]["sdxl"], controlnet=cn, vae=vae, variant="fp16", torch_dtype=torch.float16).to("cuda")
    pipe.vae.enable_tiling()
    pipe.set_progress_bar_config(disable=True)
    log(f"polish: models loaded in {time.time() - t0:.0f}s")
    out = job / "06_polish"
    out.mkdir(exist_ok=True)
    rec = []
    for v in jload(job / "05_render/render.json")["views"][:prof.get("polish_max_images", 999)]:
        raw = cv2.imread(str(job / "05_render" / f"{v['name']}.png"))
        h, w = raw.shape[:2]
        W8, H8 = w // 8 * 8, h // 8 * 8
        src = cv2.resize(raw, (W8, H8)) if (W8, H8) != (w, h) else raw
        ctrl = depth_image(job / "05_render" / f"{v['name']}_depth.npy", (W8, H8))
        prompt = (f"professional interior photograph of a {WORDS.get(v['room_type'], 'room')}, {cfg['style']}, "
                  "natural daylight, realistic materials, fine detail")
        t1 = time.time()
        res = pipe(prompt=prompt, negative_prompt=NEG, image=Image.fromarray(cv2.cvtColor(src, cv2.COLOR_BGR2RGB)),
                   control_image=Image.fromarray(ctrl), strength=pc["strength"], num_inference_steps=pc["steps"],
                   guidance_scale=pc["guidance"], controlnet_conditioning_scale=pc["controlnet_scale"],
                   generator=torch.Generator("cuda").manual_seed(1234), height=H8, width=W8).images[0]
        pol = cv2.resize(cv2.cvtColor(np.asarray(res), cv2.COLOR_RGB2BGR), (w, h))
        sc = edge_score(raw, pol)
        ok = sc >= pc["min_edge_score"]
        cv2.imwrite(str(out / f"{v['name']}.png"), pol if ok else raw)
        cv2.imwrite(str(out / f"{v['name']}_compare.jpg"), np.hstack([raw, pol]), [cv2.IMWRITE_JPEG_QUALITY, 88])
        rec.append({"name": v["name"], "edge_score": round(sc, 3), "accepted": bool(ok), "seconds": round(time.time() - t1, 1)})
        log(f"polish {v['name']}: edge score {sc:.2f} -> {'polished' if ok else 'kept raw render'}")
    jsave(out / "polish.json", {"items": rec, "settings": pc})


if __name__ == "__main__":
    main(sys.argv[1])
