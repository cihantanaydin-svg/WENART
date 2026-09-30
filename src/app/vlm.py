"""Stage 1b (GPU) - read room names, m2 areas and scale notes on scans/photos with Qwen3.5.
Run: python -m app.vlm <job_dir>. Runs in its own process, so the GPU is free again afterwards."""
import json, re, sys, time
from pathlib import Path
from .common import jload, jsave, load_config, log
from .textnorm import fold

PROMPT = ("This image is (part of) an apartment floor plan, usually Turkish. Read every text you can see: "
          "room names (for example SALON, YATAK ODASI, MUTFAK, BANYO, WC, HOL, BALKON), room areas like "
          "12,50 m², dimension numbers and scale notes like ÖLÇEK 1/50. Keep Turkish letters exactly. "
          'Reply ONLY with a JSON list like [{"text": "SALON", "bbox_2d": [x1, y1, x2, y2]}] '
          "with box coordinates from 0 to 1000 relative to this image. No other words.")


def parse_reply(txt, w, h):
    """JSON list of {text, bbox_2d} (0-1000 relative) -> [{text, x, y}] in this image's pixels."""
    txt = re.sub(r"<think>.*?</think>", "", txt, flags=re.S)
    items = []
    m = re.search(r"\[.*\]", txt, re.S)
    if m:
        try:
            items = json.loads(m.group(0))
        except Exception:
            items = []
    if not items:
        for mm in re.finditer(r'"text"\s*:\s*"([^"]+)"\s*,\s*"bbox_2d"\s*:\s*\[([^\]]+)\]', txt):
            try:
                items.append({"text": mm.group(1), "bbox_2d": [float(v) for v in mm.group(2).split(",")]})
            except ValueError:
                pass
    out = []
    for it in items if isinstance(items, list) else []:
        if not isinstance(it, dict) or not str(it.get("text", "")).strip() or len(it.get("bbox_2d") or []) != 4:
            continue
        try:
            x1, y1, x2, y2 = [float(v) for v in it["bbox_2d"]]
        except (TypeError, ValueError):
            continue
        if max(x1, y1, x2, y2) <= 1000:
            x1, x2, y1, y2 = x1 / 1000 * w, x2 / 1000 * w, y1 / 1000 * h, y2 / 1000 * h
        out.append({"text": str(it["text"]).strip(), "x": (x1 + x2) / 2, "y": (y1 + y2) / 2})
    return out


def tiles(W, H):
    """Whole page + 2x2 overlapping tiles (small labels are easier to read in tiles)."""
    boxes = [(0, 0, W, H)]
    ox, oy = int(W * 0.05), int(H * 0.05)
    for i in range(2):
        for j in range(2):
            boxes.append((max(0, i * W // 2 - ox), max(0, j * H // 2 - oy), min(W, (i + 1) * W // 2 + ox), min(H, (j + 1) * H // 2 + oy)))
    return boxes


def merge(found, diag):
    out = []
    for f in found:
        k = fold(f["text"])
        if not k:
            continue
        if any(fold(o["text"]) == k and abs(o["x"] - f["x"]) + abs(o["y"] - f["y"]) < 0.03 * diag for o in out):
            continue
        out.append(dict(f, source="vlm"))
    return out


def main(job):
    import torch
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor
    job = Path(job)
    cfg = load_config()
    mid = cfg["vlm_model"]
    if mid == "auto":
        mid = "Qwen/Qwen3.5-9B" if torch.cuda.get_device_properties(0).total_memory >= 40e9 else "Qwen/Qwen3.5-4B"
    t0 = time.time()
    proc = AutoProcessor.from_pretrained(mid)
    model = AutoModelForImageTextToText.from_pretrained(mid, dtype=torch.bfloat16, device_map="cuda").eval()
    log(f"vlm: loaded {mid} in {time.time() - t0:.0f}s")
    page = Image.open(job / "01_parse/page.png").convert("RGB")
    W, H = page.size
    found, raw = [], []
    for (x0, y0, x1, y1) in tiles(W, H):
        crop = page.crop((x0, y0, x1, y1))
        k = min(1.0, cfg["vlm_max_side"] / max(crop.size))
        img = crop.resize((int(crop.size[0] * k), int(crop.size[1] * k)), Image.LANCZOS) if k < 1 else crop
        msgs = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": PROMPT}]}]
        kw = dict(add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt")
        try:
            inputs = proc.apply_chat_template(msgs, enable_thinking=False, **kw)
        except TypeError:
            inputs = proc.apply_chat_template(msgs, **kw)
        inputs = inputs.to(model.device)
        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=1536, do_sample=False)
        txt = proc.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
        raw.append(txt)
        for f in parse_reply(txt, img.size[0], img.size[1]):
            found.append({"text": f["text"], "x": x0 + f["x"] / k, "y": y0 + f["y"] / k})
    texts = merge(found, (W ** 2 + H ** 2) ** 0.5)
    jsave(job / "01_parse/vlm_texts.json", texts)
    (job / "01_parse/vlm_raw.txt").write_text("\n\n----\n\n".join(raw), encoding="utf-8")
    log(f"vlm: {len(texts)} texts in {time.time() - t0:.0f}s: {[t['text'] for t in texts][:12]}")


if __name__ == "__main__":
    main(sys.argv[1])
