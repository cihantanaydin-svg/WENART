"""Vision for the agent: the served Qwen3.5 model reads images through the same OpenAI-compatible endpoint.
read_texts(): same tiles / prompt / parser as app/vlm.py, writes the same 01_parse/vlm_texts.json.
critique(): asks for a JSON list of problems in overlay.png (vs the source page), layout.png or a render.
When no multimodal server is available the agent reports 'vision unavailable' instead of guessing."""
import base64, io, json, re, time
from pathlib import Path
from .common import jload, jsave, log

CRITIQUE = {
    "overlay": ("Image 1 is the original apartment floor plan. Image 2 is our reading of it: coloured room areas with "
                "name, size (m) and area (m2), red wall bands, green door lines, blue window lines. Compare them. List "
                "every error you can see: missing/extra walls, doors or windows that are missing, extra or in the wrong "
                "place, wrong room types or names, sizes or m2 values that disagree with the printed labels."),
    "layout": ("This is a furniture layout drawn on a floor plan (grey rooms, brown furniture with names, green door "
               "zones, blue window zones, a short red line shows each item's front). List problems: furniture "
               "blocking doors or windows, items facing a wall, missing main furniture, unrealistic arrangements."),
    "render": ("This is a render of one room of a 3D apartment model. List visible problems: furniture floating, "
               "sunk into the floor or intersecting walls, furniture facing the wall, missing walls or holes, "
               "black or broken textures, image far too dark or too bright."),
}
FORMAT = ('Reply ONLY with JSON: {"overall": "ok" | "minor" | "major", "issues": [{"kind": "wall|door|window|room|'
          'label|scale|furniture|render|other", "where": "room name or id", "problem": "short text", '
          '"suggestion": "short text"}]}')


def data_url(path, max_side=1280):
    from PIL import Image
    im = Image.open(path).convert("RGB")
    k = min(1.0, max_side / max(im.size))
    if k < 1:
        im = im.resize((int(im.size[0] * k), int(im.size[1] * k)), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def parse_json(txt):
    txt = re.sub(r"<think>.*?</think>", "", txt or "", flags=re.S)
    m = re.search(r"\{.*\}", txt, re.S)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass
    return {"overall": "unknown", "issues": [], "raw": (txt or "")[:1500]}


class ServerVision:
    def __init__(self, server):
        self.server = server
        self.available = bool(server.cfg["llm"]["vision"])

    def ask(self, images, prompt, max_tokens=1536):
        self.server.start()
        content = [{"type": "image_url", "image_url": {"url": data_url(p)}} for p in images]
        content.append({"type": "text", "text": prompt})
        msg, usage = self.server.chat([{"role": "user", "content": content}], max_tokens=max_tokens, temperature=0.0)
        return msg.get("content") or "", usage

    def critique(self, job, what, image=None):
        job = Path(job)
        if what == "overlay":
            imgs = [job / "01_parse/page.png", job / "02_plan/overlay.png"]
        elif what == "layout":
            imgs = [job / "03_layout/layout.png"]
        else:
            imgs = [image]
        missing = [str(p) for p in imgs if not Path(p).exists()]
        if missing:
            return {"available": True, "error": f"image not found: {missing}"}
        txt, usage = self.ask(imgs, CRITIQUE["render" if what == "render" else what] + " " + FORMAT)
        res = parse_json(txt)
        res.update(available=True, image=[str(p) for p in imgs], tokens=usage.get("total_tokens"))
        return res

    def read_texts(self, job):
        """Room names, m2 labels and scale notes from 01_parse/page.png -> 01_parse/vlm_texts.json."""
        from PIL import Image
        from .vlm import PROMPT, tiles, parse_reply, merge
        job = Path(job)
        cfg = self.server.cfg
        page = Image.open(job / "01_parse/page.png").convert("RGB")
        W, H = page.size
        found, raw, t0 = [], [], time.time()
        tmp = job / "01_parse/_tile.png"
        for (x0, y0, x1, y1) in tiles(W, H):
            crop = page.crop((x0, y0, x1, y1))
            k = min(1.0, cfg["vlm_max_side"] / max(crop.size))
            img = crop.resize((int(crop.size[0] * k), int(crop.size[1] * k)), Image.LANCZOS) if k < 1 else crop
            img.save(tmp)
            txt, _ = self.ask([tmp], PROMPT)
            raw.append(txt)
            for f in parse_reply(txt, img.size[0], img.size[1]):
                found.append({"text": f["text"], "x": x0 + f["x"] / k, "y": y0 + f["y"] / k})
        tmp.unlink(missing_ok=True)
        texts = merge(found, (W ** 2 + H ** 2) ** 0.5)
        jsave(job / "01_parse/vlm_texts.json", texts)
        (job / "01_parse/vlm_raw.txt").write_text("\n\n----\n\n".join(raw), encoding="utf-8")
        log(f"vision: {len(texts)} texts in {time.time() - t0:.0f}s via the LLM server")
        return texts


class NoVision:
    available = False

    def __init__(self, reason="no multimodal LLM server (backend rules/mock or llm.vision=false)"):
        self.reason = reason

    def critique(self, job, what, image=None):
        return {"available": False, "reason": self.reason}

    def read_texts(self, job):
        return None


class ScriptedVision:
    """Tests and replays: returns recorded answers in order."""
    available = True

    def __init__(self, answers):
        self.answers = list(answers)

    def critique(self, job, what, image=None):
        return dict(self.answers.pop(0) if self.answers else {"overall": "ok", "issues": []}, available=True)

    def read_texts(self, job):
        texts = self.answers.pop(0) if self.answers else []
        jsave(Path(job) / "01_parse/vlm_texts.json", texts)
        return texts
