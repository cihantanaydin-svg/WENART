"""Agent layer: a local open-weights LLM (vLLM, OpenAI-compatible) drives the pipeline through tools, inspects
the results, repairs problems with validated patches and delivers <job>/final/ (apartment.blend/.glb/.usdc,
preview renders, overlay.png, report.md).
  python -m app.agent run <plan file> [--job-dir DIR] [--backend llm|rules|mock] [--script mock.json]
  python -m app.agent replay <old job>/agent_log.jsonl [--job-dir DIR]     re-run the same tool calls
  python -m app.agent tools                                                print the tool JSON schemas
Hard limits: agent.max_steps tool calls, agent.max_minutes, agent.max_cost_usd (GPU price x time), a patch
budget and a repair budget per issue. Every tool call (arguments, reasoning, result, seconds, VRAM peak) is
appended to <job>/agent_log.jsonl; replay feeds those calls back in the same order."""
import argparse, json, re, shutil, sys, time, traceback
from pathlib import Path
import numpy as np
from .common import APP, WS, GpuSampler, blender, gpu_used_mb, jload, jsave, load_config, log, pod_tier, run
from .patches import PATCH_SCHEMA, PatchStore, schema_errors
from . import furniture

SYSTEM = """You are the build agent of a pipeline that turns an apartment floor plan into a clean 3D model.
You act ONLY through the tools. You never write geometry, coordinate lists or Blender code: the plan is changed
only with patch_plan operations, furniture only with choose_furniture (or patch_plan swap/remove_furniture).
Goal: final/apartment.blend (+ .glb, .usdc) that matches the source plan, then finish with an honest summary.
Usual order: parse_plan -> read_text_vlm (only if parse_plan says needs_vlm) -> get_plan_summary ->
view_image overlay -> run_checks -> repairs -> relayout -> choose_furniture -> build_scene -> render_preview ->
view_image render -> run_checks -> export_blender -> finish.
Act on every issue that run_checks reports:
- area_label: a room's area disagrees with its printed m2 label. If most labelled rooms are off by the same
  factor the scale is wrong: patch_plan scale_from_label on a large, clearly labelled room. If only one room is
  off, look at the overlay first (label on the wrong room, missed wall) - do not rescale for a single room.
- scale_confidence: read_text_vlm for m2 labels if not done yet, then scale_from_label.
- unlabelled rooms or wrong types (also from view_image critiques): retype_room.
- unfurnished main rooms / layout checks: check room type and door/window positions, patch, then relayout.
- render problems: fix the cause (swap or remove a furniture item, or a plan fix), build_scene again.
- export failures: build_scene and export_blender again.
Every issue has a repair budget. When run_checks marks an issue stop_repairing, stop and list it in
finish.uncertain. Never repeat a patch that was rejected. Use one or two operations per patch with a clear reason.
Ids: rooms r1.., doors d1.., windows win1..; lengths in metres; a wall runs from point a to point b and
move_opening shift_m moves along the wall towards b. Budget: {steps} tool calls and {minutes} minutes."""

STAGE_ORDER = ["plan", "layout", "assets", "scene", "render", "export"]


def obj(props, required=()):
    return {"type": "object", "properties": props, "required": list(required), "additionalProperties": False}


TOOLS = {}


def tool(name, desc, params, gpu=None):
    def deco(fn):
        TOOLS[name] = {"desc": desc, "params": params, "fn": fn, "gpu": gpu}
        return fn
    return deco


def openai_tools():
    return [{"type": "function", "function": {"name": n, "description": t["desc"], "parameters": t["params"]}}
            for n, t in TOOLS.items()]


def gpu_present():
    from .main import gpu_present as g
    return g()


# ---------- context shared by the tools ----------
class Ctx:
    def __init__(self, inp, job, cfg, vision, server=None, planner=None):
        self.inp, self.job, self.cfg = Path(inp), Path(job), cfg
        self.vision, self.server, self.planner = vision, server, planner
        self.store = PatchStore(job)
        self.gpu = gpu_present()
        self.t0 = time.time()
        self.steps, self.done, self.stop_reason = 0, False, None
        self.fresh = set()                          # stages whose output matches the current inputs
        self.plan_version, self.actions = 0, 0
        self.attempts, self.last_keys, self.given_up = {}, set(), set()
        self.critiques, self.renders, self.last_checks = [], [], None
        self.validation, self.finish_warned, self.final = None, False, None
        self.replay_data = None

    def invalidate(self, from_stage):
        for s in STAGE_ORDER[STAGE_ORDER.index(from_stage):]:
            self.fresh.discard(s)

    def minutes(self):
        return (time.time() - self.t0) / 60

    def cost(self):
        return round(self.minutes() / 60 * self.cfg["gpu_price_per_hour"], 3)

    def budget(self):
        a = self.cfg["agent"]
        return {"steps_left": a["max_steps"] - self.steps, "minutes_left": round(a["max_minutes"] - self.minutes(), 1),
                "cost_usd": self.cost(), "cost_limit_usd": a["max_cost_usd"]}

    def over_budget(self):
        a = self.cfg["agent"]
        if self.steps >= a["max_steps"]:
            return f"step limit ({a['max_steps']}) reached"
        if self.minutes() >= a["max_minutes"]:
            return f"time limit ({a['max_minutes']} min) reached"
        if self.cost() >= a["max_cost_usd"]:
            return f"cost limit (${a['max_cost_usd']}) reached"
        return None

    def run_config(self, profile):
        cfg = self.cfg
        dev = (jload(APP / "device.json") or {}).get("device", "OPTIX" if self.gpu else "CPU")
        jsave(self.job / "run_config.json", {"assets": str(WS / "assets"), "profile": profile, "device": dev,
                                             "profile_settings": cfg["profiles"][profile], "exposure": cfg["exposure"],
                                             "sun_strength": cfg["sun_strength"], "fill_w_per_m2": cfg["fill_w_per_m2"],
                                             "export": cfg["export"]})

    def plan(self):
        return jload(self.job / "02_plan/plan.json")


# ---------- summaries ----------
def plan_summary(plan, detail="rooms"):
    sc = plan["scale"]
    rooms = []
    for r in plan["rooms"]:
        dev = round(100 * (r["area_m2"] / r["label_area_m2"] - 1), 1) if r.get("label_area_m2") else None
        rooms.append({"id": r["id"], "type": r["type"], "label": r["label"], "size_m": r["size_m"], "area_m2": r["area_m2"],
                      "label_m2": r.get("label_area_m2"), "diff_pct": dev, "confidence": r.get("confidence")})
    out = {"source": plan["source"], "scale": {"method": sc["method"], "confidence": sc["confidence"],
                                               "agent_factor": sc.get("agent_factor", 1.0)},
           "counts": {k: len(plan[k]) for k in ("walls", "doors", "windows", "rooms")}, "rooms": rooms,
           "warnings": plan.get("warnings", [])[:12], "patches_applied": plan.get("patches_applied", 0)}
    if detail in ("openings", "all"):
        out["doors"] = [{"id": d["id"], "wall": d["wall_id"], "width": d["width"], "kind": d["kind"], "rooms": d.get("rooms"),
                         "swing": d.get("swing")} for d in plan["doors"]]
        out["windows"] = [{"id": w["id"], "wall": w["wall_id"], "width": w["width"], "kind": w["kind"], "sill": w["sill"],
                           "head": w["head"], "rooms": w.get("rooms")} for w in plan["windows"]]
    if detail == "all":
        out["walls"] = [{"id": w["id"], "a": w["a"], "b": w["b"], "t": w["thickness"], "exterior": w["exterior"]}
                        for w in plan["walls"]]
    return out


def render_sanity(path):
    import cv2
    img = cv2.imread(str(path))
    if img is None:
        return {"name": Path(path).stem, "flags": ["unreadable"]}
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    s = {"name": Path(path).stem, "mean": round(float(g.mean()), 1), "std": round(float(g.std()), 1),
         "black": round(float((g < 10).mean()), 3), "blown": round(float((g > 250).mean()), 3)}
    flags = []
    if s["mean"] < 35 or s["black"] > 0.6:
        flags.append("too_dark")
    if s["mean"] > 225 or s["blown"] > 0.25:
        flags.append("too_bright")
    if s["std"] < 12:
        flags.append("flat_image")
    s["flags"] = flags
    return s


# ---------- tools ----------
@tool("parse_plan", "Read the input file and build the 2D plan (walls, doors, windows, rooms, scale). reparse=false "
      "only re-runs the geometry step (e.g. after read_text_vlm). Stored patches are re-applied automatically.",
      obj({"reparse": {"type": "boolean", "default": True},
           "wallseg": {"enum": ["config", "on", "off"], "description": "trained wall model for scans/photos"}}))
def t_parse_plan(ctx, a):
    from .parse import run_parse
    from .plan import run_plan
    if a.get("reparse", True) or not (ctx.job / "01_parse/extract.json").exists():
        run_parse(ctx.inp, ctx.job)
        use = {"on": True, "off": False}.get(a.get("wallseg", "config"), ctx.cfg["wallseg"]["enabled"])
        kind = (jload(ctx.job / "01_parse/extract.json") or {}).get("kind")
        if use and kind in ("pdf_scan", "photo"):
            try:
                run([sys.executable, "-m", "app.wallseg", "predict", ctx.job], log_file=ctx.job / "logs/wallseg.log", cwd=str(WS))
            except Exception as e:
                log(f"wall model failed, classic walls used: {e}")
    run_plan(ctx.job, ctx.cfg)
    plan, skipped = ctx.store.after_plan_stage()
    ctx.invalidate("plan")
    ctx.fresh.add("plan")
    ctx.plan_version += 1
    ex = jload(ctx.job / "01_parse/extract.json")
    out = plan_summary(plan)
    out.update(kind=ex["kind"], needs_vlm=bool(ex.get("needs_vlm")) and not (ctx.job / "01_parse/vlm_texts.json").exists(),
               patches_not_reapplied=skipped)
    return out


def _vlm_component(ctx):
    if ctx.vision.available:
        return None                                 # read through the running server: no extra GPU memory
    return "vlm_transformers_9b" if "9B" in str(ctx.cfg.get("vlm_model")) or pod_tier(ctx.cfg) == "48gb" else "vlm_transformers_4b"


@tool("read_text_vlm", "Read room names, m2 labels and scale notes from the page image with the vision model, then "
      "rebuild the plan with them. Needed for scans and photos (parse_plan says needs_vlm).",
      obj({"force": {"type": "boolean", "description": "read again even if texts exist"}}), gpu=_vlm_component)
def t_read_text_vlm(ctx, a):
    from .plan import run_plan
    vt = ctx.job / "01_parse/vlm_texts.json"
    if vt.exists() and not a.get("force"):
        texts, how = jload(vt), "reused"
    elif ctx.vision.available:
        texts, how = ctx.vision.read_texts(ctx.job), "llm server"
    elif ctx.gpu:
        run([sys.executable, "-m", "app.vlm", ctx.job], log_file=ctx.job / "logs/vlm.log", cwd=str(WS))
        texts, how = jload(vt, []), "transformers"
    else:
        return {"error": "no vision model available (no GPU / no multimodal server)"}
    ctx.replay_data = {"vlm_texts": texts}
    run_plan(ctx.job, ctx.cfg)
    plan, skipped = ctx.store.after_plan_stage()
    ctx.invalidate("plan")
    ctx.fresh.add("plan")
    ctx.plan_version += 1
    ctx.actions += 1
    out = plan_summary(plan)
    out.update(texts_read=len(texts or []), how=how, sample=[t["text"] for t in (texts or [])][:15])
    return out


@tool("get_plan_summary", "Current plan: scale, rooms (size, area, printed label, difference), warnings; "
      "detail=openings adds doors/windows, detail=all adds walls.",
      obj({"detail": {"enum": ["rooms", "openings", "all"], "default": "rooms"}}))
def t_get_plan_summary(ctx, a):
    plan = ctx.plan()
    if plan is None:
        return {"error": "no plan yet: call parse_plan"}
    return plan_summary(plan, a.get("detail", "rooms"))


@tool("view_image", "Ask the vision model to critique an image: 'overlay' (our plan drawn over the source page, "
      "compared with the source), 'layout' (furniture plan) or 'render' (a preview render, name from render_preview).",
      obj({"image": {"enum": ["overlay", "layout", "render"]}, "render_name": {"type": "string", "maxLength": 80}},
          ["image"]))
def t_view_image(ctx, a):
    what = a["image"]
    img = None
    if what == "render":
        names = [r["name"] for r in ctx.renders]
        name = a.get("render_name") or (names[0] if names else None)
        if not name or not (ctx.job / "05_render" / f"{name}.png").exists():
            return {"error": f"no render {name!r}; available: {names}"}
        img = ctx.job / "05_render" / f"{name}.png"
    res = ctx.vision.critique(ctx.job, what, img)
    ctx.replay_data = {"critique": res}
    if res.get("available"):
        ctx.critiques.append({"what": what, "name": a.get("render_name"), "plan_version": ctx.plan_version,
                              "overall": res.get("overall"), "issues": res.get("issues", [])[:12], "step": ctx.steps})
    return res


@tool("patch_plan", "Change the plan or furniture with structured operations (validated against a JSON schema and "
      "plan invariants; rejected patches change nothing). Always give the reason (which check/critique it fixes).",
      PATCH_SCHEMA)
def t_patch_plan(ctx, a):
    furn = [op for op in a["ops"] if op["op"] in ("swap_furniture", "remove_furniture")]
    errs = []
    for op in furn:
        errs += furniture.check_override(ctx.job, ctx.cfg, op["item_id"],
                                         op["asset_id"] if op["op"] == "swap_furniture" else "remove")
    if errs:
        return {"accepted": False, "stage": "furniture", "errors": errs}
    res = ctx.store.propose(a, ctx.cfg["agent"]["max_patches"], meta={"step": ctx.steps})
    if res["accepted"]:
        ctx.actions += 1
        for op in furn:
            furniture.set_override(ctx.job, ctx.cfg, op["item_id"], op["asset_id"] if op["op"] == "swap_furniture" else "remove")
        if res["changes"]:
            ctx.invalidate("layout")
            ctx.plan_version += 1
        if furn:
            ctx.invalidate("assets")
        res["next"] = "relayout, then choose_furniture/build_scene" if res["changes"] else "build_scene"
    return res


@tool("relayout", "Place the furniture again on the current plan (rule based). Returns layout checks, unfurnished "
      "main rooms and what could not be placed per room.", obj({}))
def t_relayout(ctx, a):
    from .layout import run_layout
    if ctx.plan() is None:
        return {"error": "no plan yet: call parse_plan"}
    out = run_layout(ctx.job, ctx.cfg)
    ctx.invalidate("layout")
    ctx.fresh.add("layout")
    ctx.actions += 1
    return {"items": len(out["items"]), "checks": out["checks"], "unfurnished_main_rooms": out["unfurnished_main_rooms"],
            "rooms": out["rooms"], "warnings": out["warnings"][:10]}


@tool("choose_furniture", "Pick a 3D model for every furniture item (type, size fit, style). Optional: force one "
      "item to a catalog model / 'parametric' / 'remove', set a style (e.g. 'scandinavian oak'), or list the "
      "candidate models of one item.",
      obj({"item_id": {"type": "string", "maxLength": 80}, "asset_id": {"type": "string", "maxLength": 120},
           "style": {"type": "string", "maxLength": 200}, "list_candidates_for": {"type": "string", "maxLength": 80}}))
def t_choose_furniture(ctx, a):
    if "layout" not in ctx.fresh and not (ctx.job / "03_layout/layout.json").exists():
        return {"error": "no layout yet: call relayout"}
    if a.get("item_id") and a.get("asset_id"):
        errs = furniture.set_override(ctx.job, ctx.cfg, a["item_id"], a["asset_id"])
        if errs:
            return {"error": "; ".join(errs)}
        ctx.actions += 1
    if a.get("style"):
        ov = jload(furniture.overrides_file(ctx.job), {})
        ov["__style__"] = a["style"]
        jsave(furniture.overrides_file(ctx.job), ov)
        ctx.actions += 1
    res = furniture.select_assets(ctx.job, ctx.cfg)
    ctx.invalidate("assets")
    ctx.fresh.add("assets")
    out = {"by_type": res["summary"], "style_words": res["style_words"],
           "parametric_reasons": {k: v["reason"] for k, v in res["items"].items() if v["source"] == "parametric"}}
    if a.get("list_candidates_for"):
        lay = jload(ctx.job / "03_layout/layout.json")
        it = next((i for i in lay["items"] if i["id"] == a["list_candidates_for"]), None)
        out["candidates"] = (furniture.candidates(it, furniture.load_catalog(ctx.cfg), ctx.cfg, set(res["style_words"]))[:6]
                             if it else f"no item {a['list_candidates_for']}")
    return out


@tool("build_scene", "Build the 3D scene in Blender from the current plan, layout and furniture choice "
      "(walls with real openings, floors, furniture, lights, cameras). CPU only.", obj({}))
def t_build_scene(ctx, a):
    if not (ctx.job / "03_layout/layout.json").exists():
        return {"error": "no layout yet: call relayout"}
    if "assets" not in ctx.fresh:
        furniture.select_assets(ctx.job, ctx.cfg)
        ctx.fresh.add("assets")
    ctx.run_config(ctx.cfg["agent"]["preview_profile"])
    blender("blender_scene.py", ctx.job)
    ctx.invalidate("scene")
    ctx.fresh.add("scene")
    cams = jload(ctx.job / "04_scene/cameras.json", {})
    qa = jload(ctx.job / "04_scene/furniture_qa.json", {})
    fb = [f"{r['item_id']}: {r['reason']}" for r in qa.get("items", []) if r.get("planned") != r.get("used")]
    return {"cameras": [c["name"] for c in cams.get("cameras", [])], "warnings": cams.get("warnings", [])[:10],
            "furniture_used": qa.get("summary", {}), "fallbacks_to_parametric": fb[:15]}


def _render_component(ctx):
    return "cycles_preview"


@tool("render_preview", "Render every camera of the built scene with Cycles at preview quality (GPU) and check "
      "brightness/contrast of each image. Use view_image render to let the vision model look at them.",
      obj({}), gpu=_render_component)
def t_render_preview(ctx, a):
    if "scene" not in ctx.fresh:
        return {"error": "the scene is missing or older than the plan/furniture: call build_scene first"}
    if not ctx.gpu and not ctx.cfg["agent"]["render_on_cpu"]:
        return {"skipped": "no GPU (agent.render_on_cpu is false)"}
    for old in (ctx.job / "05_render").glob("*"):
        old.unlink()
    ctx.run_config(ctx.cfg["agent"]["preview_profile"])
    blender("blender_render.py", ctx.job)
    ctx.fresh.add("render")
    ren = jload(ctx.job / "05_render/render.json", {})
    ctx.renders = [render_sanity(ctx.job / "05_render" / f"{v['name']}.png") for v in ren.get("views", [])]
    return {"device": ren.get("device"), "renders": ctx.renders}


def compute_issues(ctx):
    a = ctx.cfg["agent"]
    plan = ctx.plan()
    if plan is None:
        return [{"key": "no_plan", "severity": "high", "problem": "no plan yet", "suggest": "parse_plan"}]
    raster = plan["source"]["kind"] in ("pdf_scan", "photo")
    tol = a["area_tol_pct_raster"] if raster else a["area_tol_pct_vector"]
    iss = []
    sc = plan["scale"]
    if sc["confidence"] < a["min_scale_confidence"] and sc.get("agent_factor", 1.0) == 1.0:
        iss.append({"key": "scale_confidence", "severity": "high",
                    "problem": f"scale from {sc['method']} with confidence {sc['confidence']}",
                    "suggest": "read_text_vlm for m2 labels, then patch_plan scale_from_label"})
    for r in plan["rooms"]:
        if r.get("label_area_m2") and r["type"] != "balcony":
            dev = 100 * (r["area_m2"] / r["label_area_m2"] - 1)
            if abs(dev) > tol:
                iss.append({"key": f"area_label:{r['id']}", "severity": "high",
                            "problem": f"{r['id']} {r['label'] or r['type']}: {r['area_m2']} m2 measured vs label "
                                       f"{r['label_area_m2']} m2 ({dev:+.1f} %, tolerance {tol} %)",
                            "suggest": "view_image overlay; scale_from_label if all rooms are off by the same factor"})
        if r.get("confidence", 1) <= 0.3 and r["type"] != "balcony":
            iss.append({"key": f"unlabelled:{r['id']}", "severity": "medium",
                        "problem": f"{r['id']} ({r['area_m2']} m2) has no label; typed '{r['type']}' from its shape",
                        "suggest": "view_image overlay, then retype_room if the type is wrong"})
    lay = jload(ctx.job / "03_layout/layout.json")
    if lay and "layout" in ctx.fresh:
        for k, v in lay["checks"].items():
            if v:
                iss.append({"key": f"layout:{k}", "severity": "medium", "problem": f"layout check {k} = {v}",
                            "suggest": "fix doors/windows or room types, relayout; or remove_furniture"})
        for rid in lay["unfurnished_main_rooms"]:
            iss.append({"key": f"unfurnished:{rid}", "severity": "high", "problem": f"main room {rid} has no furniture",
                        "suggest": "check room type / openings (view_image layout), patch, relayout"})
    if "scene" in ctx.fresh and "render" not in ctx.fresh:
        iss.append({"key": "todo:render_preview", "severity": "medium",
                    "problem": "no preview renders of the current scene" + ("" if ctx.gpu else " (no GPU on this machine)"),
                    "suggest": "render_preview, then view_image render"})
    for r in ctx.renders if "render" in ctx.fresh else []:
        for f in r["flags"]:
            iss.append({"key": f"render:{r['name']}:{f}", "severity": "medium", "problem": f"render {r['name']}: {f}",
                        "suggest": "view_image render; check lights/windows or camera placement"})
    for c in ctx.critiques:
        if c["what"] == "overlay" and c["plan_version"] == ctx.plan_version and c["overall"] in ("minor", "major"):
            for i, x in enumerate(c["issues"][:6]):
                iss.append({"key": f"critique:overlay:{x.get('kind', 'other')}:{x.get('where', i)}",
                            "severity": "high" if c["overall"] == "major" else "medium",
                            "problem": f"VLM on overlay: {x.get('where', '')}: {x.get('problem', '')}",
                            "suggest": x.get("suggestion", "")})
        if c["what"] == "render" and "render" in ctx.fresh and c["overall"] in ("minor", "major"):
            for i, x in enumerate(c["issues"][:4]):
                iss.append({"key": f"critique:render:{c.get('name')}:{i}", "severity": "medium",
                            "problem": f"VLM on render {c.get('name')}: {x.get('problem', '')}",
                            "suggest": x.get("suggestion", "")})
    if ctx.vision.available and a["critique_overlay"] and not any(
            c["what"] == "overlay" and c["plan_version"] == ctx.plan_version for c in ctx.critiques):
        iss.append({"key": "todo:overlay_critique", "severity": "low", "problem": "overlay not yet checked by the VLM",
                    "suggest": "view_image overlay"})
    qa = jload(ctx.job / "04_scene/furniture_qa.json", {}) if "scene" in ctx.fresh else {}
    for r in qa.get("items", []):
        if r.get("planned") not in (None, r.get("used")) and r.get("used") == "parametric":
            iss.append({"key": f"furniture_fallback:{r['item_id']}", "severity": "low",
                        "problem": f"{r['item_id']}: library model rejected ({r['reason']}), parametric used",
                        "suggest": "choose_furniture list_candidates_for, or accept"})
    if ctx.validation is not None and "export" in ctx.fresh and not ctx.validation.get("ok"):
        for c in ctx.validation.get("checks", []):
            if not c.get("ok"):
                iss.append({"key": f"export:{c['name']}", "severity": "high", "problem": f"export check {c['name']}: {c.get('detail')}",
                            "suggest": "build_scene then export_blender"})
    return iss


@tool("run_checks", "All checks on the current state: area vs printed m2 labels, scale confidence, room labels, "
      "layout checks, unfurnished main rooms, render sanity, VLM critiques, export validation. Shows repair "
      "attempts per issue and stop_repairing when its budget is used up.", obj({}))
def t_run_checks(ctx, a):
    iss = compute_issues(ctx)
    keys = {i["key"] for i in iss}
    if ctx.actions:                                 # something was tried since the last check
        for k in keys & ctx.last_keys:
            ctx.attempts[k] = ctx.attempts.get(k, 0) + 1
    ctx.actions = 0
    ctx.last_keys = keys
    mx = ctx.cfg["agent"]["max_repairs_per_issue"]
    for i in iss:
        i["attempts"] = ctx.attempts.get(i["key"], 0)
        if i["attempts"] >= mx and i["severity"] != "low":
            i["stop_repairing"] = True
            ctx.given_up.add(i["key"])
    ctx.last_checks = iss
    return {"issues": iss, "high": sum(i["severity"] == "high" for i in iss), "stale_stages":
            [s for s in STAGE_ORDER if s not in ctx.fresh], "budget": ctx.budget()}


@tool("export_blender", "Write final/apartment.blend (+ .glb, .usdc, previews, overlay) from the built scene and "
      "validate it headless (units, collections, bounding box vs plan, textures packed, watertight walls with "
      "real openings). The job fails if validation fails.", obj({}))
def t_export_blender(ctx, a):
    from .deliver import deliver
    if "scene" not in ctx.fresh:
        return {"error": "the scene is missing or older than the plan/furniture: call build_scene first"}
    ctx.validation = deliver(ctx.job, ctx.cfg)
    ctx.fresh.add("export")
    return {"ok": ctx.validation.get("ok"), "failed": [c for c in ctx.validation.get("checks", []) if not c.get("ok")],
            "files": sorted(p.name for p in (ctx.job / "final").iterdir())}


@tool("finish", "End the run. status complete|partial|failed, a short summary, what you fixed and what is still "
      "uncertain (be specific: room ids, checks).",
      obj({"status": {"enum": ["complete", "partial", "failed"]}, "summary": {"type": "string", "maxLength": 2000},
           "fixed": {"type": "array", "items": {"type": "string", "maxLength": 300}, "maxItems": 30},
           "uncertain": {"type": "array", "items": {"type": "string", "maxLength": 300}, "maxItems": 30}},
          ["status", "summary"]))
def t_finish(ctx, a):
    exported = bool(ctx.validation and ctx.validation.get("ok") and "export" in ctx.fresh)
    open_high = [i for i in compute_issues(ctx) if i["severity"] == "high" and i["key"] not in ctx.given_up]
    if not ctx.finish_warned and (not exported or (open_high and not a.get("uncertain"))):
        ctx.finish_warned = True
        msg = []
        if not exported:
            msg.append("final/apartment.blend has not passed export validation yet (build_scene, export_blender)")
        if open_high and not a.get("uncertain"):
            msg.append("open high-severity issues with repair budget left: " + ", ".join(i["key"] for i in open_high[:8])
                       + " - repair them or list them in 'uncertain'")
        return {"accepted": False, "errors": msg, "note": "calling finish again will end the run anyway"}
    ctx.final = dict(a)
    ctx.done = True
    return {"accepted": True}


# ---------- backends ----------
def parse_text_tool_calls(text):
    """Tool calls written as text (tool-call parser not active): <tool_call>{json}</tool_call> or the
    <function=name><parameter=k>v</parameter></function> format of Qwen3-Coder style templates."""
    calls = []
    for m in re.finditer(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", text or "", re.S):
        try:
            d = json.loads(m.group(1))
            calls.append({"name": d["name"], "arguments": d.get("arguments", {})})
        except Exception:
            pass
    for m in re.finditer(r"<function=([\w-]+)>(.*?)</function>", text or "", re.S):
        args = {}
        for p in re.finditer(r"<parameter=([\w-]+)>\s*(.*?)\s*</parameter>", m.group(2), re.S):
            v = p.group(2)
            try:
                args[p.group(1)] = json.loads(v)
            except Exception:
                args[p.group(1)] = v
        calls.append({"name": m.group(1), "arguments": args})
    return [{"id": f"text_{i}", "type": "function", "function": {"name": c["name"], "arguments": json.dumps(c["arguments"])}}
            for i, c in enumerate(calls)]


class LLMBackend:
    name = "llm"

    def __init__(self, server):
        self.server = server
        self.tokens = 0

    def next(self, messages, ctx):
        self.server.start()
        try:
            msg, usage = self.server.chat(messages, openai_tools())
        except RuntimeError as e:
            if self.server.running():
                raise
            log(f"llm: server gone ({e}) - restarting once")
            self.server.start()
            msg, usage = self.server.chat(messages, openai_tools())
        self.tokens += usage.get("total_tokens", 0)
        calls = msg.get("tool_calls") or parse_text_tool_calls(msg.get("content"))
        return {"role": "assistant", "content": msg.get("content") or "", "tool_calls": calls}, usage


class ScriptBackend:
    """mock: scripted turns [{"content": "...", "calls": [{"name": .., "arguments": {..} or "raw text"}]}]."""
    name = "mock"

    def __init__(self, turns):
        self.turns = list(turns)
        self.n = 0

    def next(self, messages, ctx):
        if not self.turns:
            return {"role": "assistant", "content": "(script finished)", "tool_calls": []}, {}
        t = self.turns.pop(0)
        calls = []
        for c in t.get("calls", []):
            self.n += 1
            args = c.get("arguments", {})
            calls.append({"id": f"call_{self.n}", "type": "function",
                          "function": {"name": c["name"], "arguments": args if isinstance(args, str) else json.dumps(args)}})
        return {"role": "assistant", "content": t.get("content", ""), "tool_calls": calls}, {}


def replay_turns(log_file):
    """Tool calls of an earlier run, grouped by LLM turn, plus the recorded vision answers (for ScriptedVision)."""
    turns, vision = {}, []
    for line in Path(log_file).read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if r.get("type") != "tool" or r.get("turn", 0) < 0:     # turn -1 = harness safe-finish actions
            continue
        turns.setdefault(r["turn"], {"content": r.get("reasoning", ""), "calls": []})["calls"].append(
            {"name": r["tool"], "arguments": r["raw_arguments"]})
        rd = r.get("replay_data") or {}
        if "critique" in rd:
            vision.append(rd["critique"])
        if "vlm_texts" in rd:
            vision.append(rd["vlm_texts"])
    return [turns[k] for k in sorted(turns)], vision


class RulesBackend:
    """No LLM: fixed order and simple deterministic repairs. Used with SKIP_AGENT_LLM=1 and as a baseline."""
    name = "rules"

    def __init__(self):
        self.gen, self.n = None, 0

    def policy(self, ctx):
        def call(name, args=None, why=""):
            return name, args or {}, why
        r = yield call("parse_plan", {"reparse": True}, "start: read the input and build the plan")
        if r.get("needs_vlm") and (ctx.vision.available or ctx.gpu):
            yield call("read_text_vlm", {}, "scan/photo: read room names and m2 labels")
        yield call("get_plan_summary", {"detail": "rooms"}, "look at the plan")
        if ctx.vision.available and ctx.cfg["agent"]["critique_overlay"]:
            yield call("view_image", {"image": "overlay"}, "compare the plan reading with the source page")
        chk = yield call("run_checks", {}, "checks before furnishing")
        plan = ctx.plan()
        labelled = [x for x in plan["rooms"] if x.get("label_area_m2") and x["type"] != "balcony"]
        devs = [x["area_m2"] / x["label_area_m2"] for x in labelled]
        if len(devs) >= 2 and any(i["key"].startswith("area_label") for i in chk.get("issues", [])):
            med = float(np.median(devs))
            if all(abs(d / med - 1) < 0.03 for d in devs) and abs(med - 1) > 0.02:
                big = max(labelled, key=lambda x: x["label_area_m2"])
                yield call("patch_plan", {"ops": [{"op": "scale_from_label", "room_id": big["id"], "label_m2": big["label_area_m2"]}],
                                          "reason": f"all {len(devs)} labelled rooms are off by the same factor {med:.3f}: scale error"},
                           "consistent area error = wrong scale")
        yield call("relayout", {}, "furnish the rooms")
        yield call("choose_furniture", {}, "pick 3D models")
        yield call("build_scene", {}, "build the 3D scene")
        if ctx.gpu or ctx.cfg["agent"]["render_on_cpu"]:
            ren = yield call("render_preview", {}, "preview renders")
            if ctx.vision.available:
                for rr in (ren.get("renders") or [])[:ctx.cfg["agent"]["critique_renders"]]:
                    yield call("view_image", {"image": "render", "render_name": rr["name"]}, "look at a render")
        chk = yield call("run_checks", {}, "final checks")
        exp = yield call("export_blender", {}, "write and validate the deliverable")
        unc = [i["problem"] for i in chk.get("issues", []) if i["severity"] != "low"]
        ok = bool(exp.get("ok"))
        yield call("finish", {"status": "complete" if ok and not unc else "partial" if ok else "failed",
                              "summary": "Rules backend (no LLM): fixed stage order with deterministic repairs only.",
                              "uncertain": unc[:30] or []}, "done")
        yield call("finish", {"status": "complete" if ok and not unc else "partial" if ok else "failed",
                              "summary": "Rules backend (no LLM): fixed stage order with deterministic repairs only.",
                              "uncertain": unc[:30] or ["see run_checks"]}, "done (second call after the finish guard)")

    def next(self, messages, ctx):
        if self.gen is None:
            self.gen = self.policy(ctx)
            item = next(self.gen)
        else:
            try:
                item = self.gen.send(ctx.last_result or {})
            except StopIteration:
                return {"role": "assistant", "content": "", "tool_calls": []}, {}
        name, args, why = item
        self.n += 1
        return {"role": "assistant", "content": why, "tool_calls": [
            {"id": f"rules_{self.n}", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}]}, {}


# ---------- the loop ----------
class AgentLog:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, rec):
        rec = dict(rec, ts=time.strftime("%Y-%m-%d %H:%M:%S"))
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")


def execute(ctx, call, turn, reasoning, alog):
    fn = call.get("function", {})
    name, raw = fn.get("name", ""), fn.get("arguments") or "{}"
    ctx.steps += 1
    rec = {"type": "tool", "step": ctx.steps, "turn": turn, "tool": name, "raw_arguments": raw, "reasoning": reasoning}
    try:
        args = json.loads(raw) if isinstance(raw, str) else dict(raw)
        if not isinstance(args, dict):
            raise ValueError("arguments must be a JSON object")
    except Exception as e:
        res = {"error": f"arguments are not valid JSON: {e}"}
        alog.write(dict(rec, ok=False, result=res))
        return res
    if name not in TOOLS:
        res = {"error": f"unknown tool {name!r}; tools: {', '.join(TOOLS)}"}
        alog.write(dict(rec, args=args, ok=False, result=res))
        return res
    errs = schema_errors(TOOLS[name]["params"], args)
    if errs:
        res = {"error": "arguments do not match the schema", "details": errs[:10]}
        alog.write(dict(rec, args=args, ok=False, result=res))
        return res
    g = TOOLS[name]["gpu"]
    comp = g(ctx) if callable(g) else g
    decision = ctx.planner.before(comp) if ctx.planner else None
    sampler = GpuSampler().start() if ctx.gpu else None
    server_up = bool(ctx.server and ctx.server.running())
    t0, ok = time.time(), True
    ctx.replay_data = None
    try:
        res = TOOLS[name]["fn"](ctx, args)
        ok = "error" not in res
    except Exception as e:
        res = {"error": f"{type(e).__name__}: {str(e)[:1500]}", "trace": traceback.format_exc()[-1500:]}
        ok = False
    peak = sampler.stop() if sampler else {}
    alog.write(dict(rec, args=args, ok=ok, seconds=round(time.time() - t0, 1), vram_peak_mb=peak.get("vram_peak_mb"),
                    gpu_util_avg=peak.get("gpu_util_avg"), server_up=server_up, vram_decision=decision,
                    component=comp, result=res, replay_data=ctx.replay_data))
    return res


def compact(messages, max_chars):
    """Keep the conversation inside the context window: shorten old tool results (the full ones are in the log)."""
    total = sum(len(json.dumps(m, ensure_ascii=False)) for m in messages)
    tool_idx = [i for i, m in enumerate(messages) if m["role"] == "tool"]
    for i in tool_idx[:-6]:
        if total <= max_chars:
            break
        c = messages[i]["content"]
        if len(c) > 300:
            messages[i]["content"] = c[:280] + ' ... [shortened - full result in agent_log.jsonl]"'
            total -= len(c) - len(messages[i]["content"])


def run_agent(inp, job, cfg, backend, vision, server=None, planner=None):
    ctx = Ctx(inp, job, cfg, vision, server, planner)
    job = ctx.job
    (job / "00_input").mkdir(parents=True, exist_ok=True)
    (job / "logs").mkdir(exist_ok=True)
    if not (job / "00_input" / ctx.inp.name).exists():
        shutil.copy2(ctx.inp, job / "00_input" / ctx.inp.name)
    alog = AgentLog(job / "agent_log.jsonl")
    model = server.model if server else None
    alog.write({"type": "start", "input": str(ctx.inp), "backend": backend.name, "model": model, "tier": pod_tier(cfg),
                "limits": {k: cfg["agent"][k] for k in ("max_steps", "max_minutes", "max_cost_usd", "max_repairs_per_issue", "max_patches")},
                "vram_budget_mb": server.budget_mb() if server else None})
    ctx.run_config(cfg["agent"]["preview_profile"])
    a = cfg["agent"]
    messages = [{"role": "system", "content": SYSTEM.format(steps=a["max_steps"], minutes=a["max_minutes"])},
                {"role": "user", "content": f"Input file: {ctx.inp.name}. Furnishing style: {cfg['style']}. "
                                            f"GPU available: {ctx.gpu}. Vision model available: {vision.available}. Start with parse_plan."}]
    turn, idle = 0, 0
    ctx.last_result = None
    try:
        while not ctx.done:
            why = ctx.over_budget()
            if why:
                ctx.stop_reason = why
                break
            turn += 1
            t0 = time.time()
            msg, usage = backend.next(messages, ctx)
            calls = msg.get("tool_calls") or []
            alog.write({"type": "llm", "turn": turn, "content": (msg.get("content") or "")[:4000], "tool_calls": len(calls),
                        "seconds": round(time.time() - t0, 1), "tokens": usage.get("total_tokens")})
            messages.append({"role": "assistant", "content": msg.get("content") or "", **({"tool_calls": calls} if calls else {})})
            if not calls:
                idle += 1
                if idle >= 3:
                    ctx.stop_reason = "the model stopped calling tools"
                    break
                messages.append({"role": "user", "content": "Call a tool. When the deliverable is exported and checked, call finish."})
                continue
            idle = 0
            for c in calls:
                res = execute(ctx, c, turn, msg.get("content") or "", alog)
                ctx.last_result = res
                messages.append({"role": "tool", "tool_call_id": c.get("id", ""),
                                 "content": json.dumps(dict(res, budget=ctx.budget()), ensure_ascii=False, default=str)[:a["tool_result_chars"]]})
                if ctx.done or ctx.over_budget():
                    break
            compact(messages, int(cfg["llm"]["max_model_len"] * 2.5))
    except Exception as e:
        ctx.stop_reason = f"agent loop error: {type(e).__name__}: {e}"
        alog.write({"type": "error", "error": ctx.stop_reason, "trace": traceback.format_exc()[-3000:]})
    finally:
        if server:
            server.stop(wait_free=False)
    if not ctx.done:
        safe_finish(ctx, alog)
    rep = write_final_report(ctx)
    alog.write({"type": "end", "status": rep["status"], "steps": ctx.steps, "minutes": round(ctx.minutes(), 1),
                "cost_usd": ctx.cost(), "export_ok": bool(ctx.validation and ctx.validation.get("ok"))})
    return rep


def safe_finish(ctx, alog):
    """Budget or loop ended without finish: build and export deterministically if possible, then stop."""
    log(f"agent: stopping ({ctx.stop_reason}) - delivering what exists")
    for name in ("relayout", "choose_furniture", "build_scene", "export_blender"):
        stage = {"relayout": "layout", "choose_furniture": "assets", "build_scene": "scene", "export_blender": "export"}[name]
        if stage in ctx.fresh or ctx.plan() is None:
            continue
        try:
            res = TOOLS[name]["fn"](ctx, {})
            alog.write({"type": "tool", "step": ctx.steps, "turn": -1, "tool": name, "raw_arguments": "{}", "args": {},
                        "reasoning": "harness: safe finish", "ok": "error" not in res, "result": res})
        except Exception as e:
            alog.write({"type": "error", "error": f"safe finish {name}: {e}"})
            break
    iss = compute_issues(ctx)
    ctx.final = {"status": "partial" if ctx.validation and ctx.validation.get("ok") else "failed",
                 "summary": f"Stopped by the harness: {ctx.stop_reason}.",
                 "uncertain": [i["problem"] for i in iss if i["severity"] != "low"]}


def write_final_report(ctx):
    from .report import write_report
    from .llm_server import table
    job, cfg = ctx.job, ctx.cfg
    recs = [json.loads(l) for l in (job / "agent_log.jsonl").read_text(encoding="utf-8").splitlines()]
    tools = [r for r in recs if r["type"] == "tool"]
    fin = ctx.final or {}
    exported = bool(ctx.validation and ctx.validation.get("ok"))
    status = fin.get("status", "failed")
    if status == "complete" and not exported:
        status = "partial"
    iss = compute_issues(ctx)
    unc = list(fin.get("uncertain") or [])
    unc += [f"{i['problem']} (not resolved{', repair budget used up' if i['key'] in ctx.given_up else ''})"
            for i in iss if i["severity"] == "high" and not any(i["key"] in u or i["problem"] in u for u in unc)]
    stages = {}
    for r in tools:
        s = stages.setdefault(r["tool"], {"seconds": 0.0, "gpu": False, "status": "ok", "calls": 0, "vram_peak_mb": None})
        s["seconds"] = round(s["seconds"] + (r.get("seconds") or 0), 1)
        s["calls"] += 1
        s["gpu"] = s["gpu"] or bool(r.get("component"))
        if r.get("vram_peak_mb") is not None:
            s["vram_peak_mb"] = max(s["vram_peak_mb"] or 0, r["vram_peak_mb"])
        if not r.get("ok"):
            s["status"] = f"errors in {s.get('errors', 0) + 1} call(s)"
            s["errors"] = s.get("errors", 0) + 1
    meta = {"input": str(ctx.inp), "profile": cfg["agent"]["preview_profile"], "stages": stages,
            "warnings": [f"agent: {ctx.stop_reason}"] if ctx.stop_reason else [],
            "started": recs[0].get("ts"), "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
            "gpu": {"vram_peak_mb": max([r.get("vram_peak_mb") or 0 for r in tools] or [0]) or None}}
    rep = write_report(job, meta, cfg)
    patches = ctx.store.patches()
    vt = table(cfg)
    start = recs[0]
    L = [f"# Agent report - {job.name}", "",
         f"**Status: {status.upper()}**  |  backend {start.get('backend')}  |  model {start.get('model') or '-'}  |  "
         f"tier {start.get('tier')}  |  {ctx.steps} tool calls  |  {ctx.minutes():.1f} min  |  ${ctx.cost()}", "",
         fin.get("summary", ""), ""]
    if ctx.stop_reason:
        L += [f"> Stopped by the harness: {ctx.stop_reason}", ""]
    L += ["## Deliverable", "", f"- export validation: **{'PASSED' if exported else 'FAILED / not run'}**"]
    for c in (ctx.validation or {}).get("checks", []):
        L.append(f"  - {'ok  ' if c['ok'] else 'FAIL'} {c['name']}: {c.get('detail', '')}")
    L += [f"- files: {', '.join(sorted(p.name for p in (job / 'final').iterdir())) if (job / 'final').exists() else '-'}", ""]
    L += ["## What the agent did", "", "| # | Tool | Arguments | Result | s | VRAM peak MB | Server up |", "|---|---|---|---|---|---|---|"]
    for r in tools:
        res = r.get("result") or {}
        short = res.get("error") or ("accepted" if res.get("accepted") else "rejected: " + "; ".join(res.get("errors", []))[:120]
                                     if "accepted" in res else "ok")
        L.append(f"| {r['step']} | {r['tool']} | `{json.dumps(r.get('args', r.get('raw_arguments')), ensure_ascii=False)[:90]}` | "
                 f"{str(short)[:140]} | {r.get('seconds', '')} | {r.get('vram_peak_mb') or ''} | {'yes' if r.get('server_up') else ''} |")
    L += ["", "## Fixed", ""]
    L += [f"- patch {i + 1}: {p['reason']} -> `{json.dumps(p['ops'], ensure_ascii=False)[:200]}`" for i, p in enumerate(patches)]
    L += [f"- {x}" for x in fin.get("fixed", [])]
    if not patches and not fin.get("fixed"):
        L.append("- nothing was changed")
    L += ["", "## Still uncertain", ""] + ([f"- {u}" for u in unc] or ["- nothing reported"])
    L += ["", "## Last checks", ""] + ([f"- [{i['severity']}] {i['problem']}" + (" (stop_repairing)" if i.get("stop_repairing") else "")
                                         for i in (ctx.last_checks or iss)] or ["- no issues"])
    L += ["", "## GPU memory", "", f"Tier {vt['tier']}, GPU {vt['gpu_total_mb']} MB. Budget table below = ESTIMATES "
          f"(config vram_estimates_mb); measured peaks per tool are in the table above.", "",
          "| Component | Estimate MB | Note |", "|---|---|---|"] + [f"| {a} | {b} | {c} |" for a, b, c in vt["rows"]]
    if ctx.planner and ctx.planner.decisions:
        L += ["", "Scheduling decisions:"] + [f"- {d}" for d in ctx.planner.decisions]
    if ctx.server:
        L += ["", f"LLM server starts: {ctx.server.starts}, start-up time {ctx.server.start_seconds:.0f} s"]
    cov = job / "final/coverage.md"
    if cov.exists():
        L += ["", cov.read_text(encoding="utf-8").replace("# Furniture coverage", "## Furniture coverage", 1)]
    L += ["", "---", "", (job / "report.md").read_text(encoding="utf-8").replace("# Report", "## Pipeline report", 1)]
    (job / "final").mkdir(exist_ok=True)
    (job / "final/report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    out = {"status": status, "export_ok": exported, "steps": ctx.steps, "uncertain": unc, "report": str(job / "final/report.md"),
           "rooms": rep["rooms"]}
    jsave(job / "final/agent_summary.json", out)
    return out


# ---------- entry points ----------
def make(cfg, backend_name, script=None, replay_log=None):
    from .vision import NoVision, ScriptedVision
    server = planner = None
    if replay_log:
        turns, answers = replay_turns(replay_log)
        return ScriptBackend(turns), ScriptedVision(answers), None, None
    if backend_name == "mock":
        s = jload(script) if script else {"turns": []}
        return ScriptBackend(s["turns"]), ScriptedVision(s.get("vision", [])) if s.get("vision") else NoVision(), None, None
    if backend_name == "rules":
        return RulesBackend(), NoVision("rules backend: no LLM server"), None, None
    from .llm_server import LLMServer, VramPlanner
    from .vision import ServerVision
    server = LLMServer(cfg)
    planner = VramPlanner(cfg, server)
    return LLMBackend(server), ServerVision(server) if cfg["llm"]["vision"] else NoVision(), server, planner


def main():
    ap = argparse.ArgumentParser(description="agentic floor plan -> 3D apartment")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run")
    p.add_argument("input")
    p.add_argument("--job-dir")
    p.add_argument("--backend", choices=["llm", "rules", "mock"])
    p.add_argument("--script", help="mock backend: JSON file with scripted turns")
    p = sub.add_parser("replay")
    p.add_argument("log")
    p.add_argument("--job-dir")
    p.add_argument("--input")
    sub.add_parser("tools")
    a = ap.parse_args()
    cfg = load_config()
    if a.cmd == "tools":
        print(json.dumps(openai_tools(), indent=1))
        return
    if a.cmd == "replay":
        start = json.loads(Path(a.log).read_text(encoding="utf-8").splitlines()[0])
        inp = Path(a.input or start["input"])
        job = Path(a.job_dir) if a.job_dir else WS / "outputs" / f"{inp.stem}_replay_{time.strftime('%Y%m%d-%H%M%S')}"
        backend, vision, server, planner = make(cfg, None, replay_log=a.log)
        cfg["agent"]["max_steps"] = max(cfg["agent"]["max_steps"], sum(1 for _ in open(a.log)))
    else:
        inp = Path(a.input).resolve()
        job = Path(a.job_dir) if a.job_dir else WS / "outputs" / f"{inp.stem}_agent_{time.strftime('%Y%m%d-%H%M%S')}"
        backend, vision, server, planner = make(cfg, a.backend or cfg["agent"]["backend"], a.script)
    rep = run_agent(inp, job, cfg, backend, vision, server, planner)
    print(f"\nAGENT {rep['status'].upper()}: {job}\n  export validation: {'passed' if rep['export_ok'] else 'FAILED'}"
          f"\n  report: {rep['report']}")
    for u in rep["uncertain"][:10]:
        print(f"  uncertain: {u}")
    sys.exit(0 if rep["export_ok"] and rep["status"] != "failed" else 1)


if __name__ == "__main__":
    main()
