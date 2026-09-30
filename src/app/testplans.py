"""Smoke test: build a known 2+1 apartment as DXF/DWG/vector PDF/scanned PDF/phone photo,
run the whole pipeline on each and check room sizes against the truth.
  python -m app.testplans unit     CPU-only checks, no GPU needed: textnorm, plan geometry (DXF + vector PDF),
                                   layout, patch schema + invariants, agent loop with a mock LLM + replay,
                                   furniture catalog + licence manifest, LLM server command + VRAM planner
                                   (+ Blender scene/export/validation when Blender or a bpy Python is available)
  python -m app.testplans quick    unit + full pipeline on every input + one agent run (GPU pod)"""
import base64, copy, json, math, shutil, struct, subprocess, sys, time, traceback
from pathlib import Path
import cv2
import numpy as np
from .common import WS, APP, BLENDER, jload, jsave, load_config, log, font_path

EXT, INT = 0.25, 0.10
# walls: (axis, fixed coordinate of centreline, start, end, thickness)
WALLS = [("h", 0.125, 0, 11.0, EXT), ("h", 8.375, 0, 11.0, EXT), ("v", 0.125, 0.25, 8.25, EXT),
         ("v", 10.875, 0.25, 8.25, EXT), ("v", 4.80, 0.25, 8.25, INT), ("v", 6.40, 0.25, 8.25, INT),
         ("h", 4.25, 0.25, 4.75, INT), ("h", 3.25, 6.45, 10.75, INT), ("h", 5.45, 6.45, 10.75, INT)]
# openings: (wall index, centre along wall, width, kind, hinge sign, swing sign)
OPENINGS = [(0, 5.60, 0.95, "door", -1, +1), (4, 2.20, 0.90, "door", -1, -1), (4, 6.80, 0.80, "door", +1, -1),
            (5, 1.80, 0.80, "door", -1, +1), (5, 4.35, 0.70, "door", -1, +1), (5, 7.20, 0.80, "door", +1, +1),
            (2, 2.25, 0.90, "window", 0, 0), (0, 2.50, 1.80, "window", 0, 0), (1, 2.50, 1.50, "window", 0, 0),
            (0, 8.60, 1.20, "window", 0, 0), (3, 4.35, 0.60, "window", 0, 0), (1, 8.60, 1.20, "window", 0, 0)]
ROOMS = [("SALON", "living", 0.25, 0.25, 4.75, 4.20), ("EBEVEYN YATAK ODASI", "bedroom", 0.25, 4.30, 4.75, 8.25),
         ("HOL", "hall", 4.85, 0.25, 6.35, 8.25), ("MUTFAK", "kitchen", 6.45, 0.25, 10.75, 3.20),
         ("BANYO", "bathroom", 6.45, 3.30, 10.75, 5.40), ("ÇOCUK ODASI", "bedroom", 6.45, 5.50, 10.75, 8.25)]
BALCONY = (-1.40, 1.00, 0.0, 3.50)


def wall_rects():
    """Wall rectangles with the openings cut out (x0, y0, x1, y1) in metres."""
    rects = []
    for i, (ax, c, s, e, t) in enumerate(WALLS):
        cuts = sorted((m - w / 2, m + w / 2) for wi, m, w, *_ in OPENINGS if wi == i)
        pos = s
        for a, b in cuts + [(e, e)]:
            if a - pos > 1e-6:
                rects.append((pos, c - t / 2, a, c + t / 2) if ax == "h" else (c - t / 2, pos, c + t / 2, a))
            pos = b
    return rects


def symbols():
    """Door leaves + swing arcs, window lines and balcony railing as polylines (metres)."""
    lines, arcs, wins = [], [], []
    for wi, m, w, kind, hs, ss in OPENINGS:
        ax, c, s, e, t = WALLS[wi]
        if kind == "window":
            for off in (-t / 2, 0, t / 2):
                a, b = (m - w / 2, m + w / 2)
                wins.append([(a, c + off), (b, c + off)] if ax == "h" else [(c + off, a), (c + off, b)])
            continue
        hinge = m + hs * w / 2
        face = c + ss * t / 2
        if ax == "h":
            hp, tip = (hinge, face), (hinge, face + ss * w)
            a0 = math.atan2(ss, 0)
            a1 = math.atan2(0, -hs)
        else:
            hp, tip = (face, hinge), (face + ss * w, hinge)
            a0 = math.atan2(0, ss)
            a1 = math.atan2(-hs, 0)
        lines.append([hp, tip])
        d = (a1 - a0 + math.pi) % (2 * math.pi) - math.pi
        arcs.append([(hp[0] + w * math.cos(a0 + d * k / 16), hp[1] + w * math.sin(a0 + d * k / 16)) for k in range(17)])
    bx0, by0, bx1, by1 = BALCONY
    rail = [[(bx1, by0), (bx0, by0), (bx0, by1), (bx1, by1)]]
    return lines, arcs, wins, rail


def labels():
    out = []
    for name, typ, x0, y0, x1, y1 in ROOMS:
        a = (x1 - x0) * (y1 - y0)
        out.append((name, f"{a:.2f}".replace(".", ",") + " m²", (x0 + x1) / 2, (y0 + y1) / 2))
    bx0, by0, bx1, by1 = BALCONY
    out.append(("BALKON", f"{(bx1 - bx0) * (by1 - by0):.2f}".replace(".", ",") + " m²", (bx0 + bx1) / 2, (by0 + by1) / 2))
    return out


def make_dxf(p):
    import ezdxf
    doc = ezdxf.new("R2018", setup=True)
    doc.header["$INSUNITS"] = 5   # centimetres
    for name in ("DUVAR", "KAPI", "PENCERE", "YAZI", "KORKULUK"):
        doc.layers.add(name)
    msp = doc.modelspace()
    cm = lambda pts: [(x * 100, y * 100) for x, y in pts]
    for x0, y0, x1, y1 in wall_rects():
        poly = cm([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])
        h = msp.add_hatch(color=7, dxfattribs={"layer": "DUVAR"})
        h.set_solid_fill()
        h.paths.add_polyline_path(poly, is_closed=True)
        msp.add_lwpolyline(poly, close=True, dxfattribs={"layer": "DUVAR"})
    lines, arcs, wins, rail = symbols()
    for group, layer in ((lines + arcs, "KAPI"), (wins, "PENCERE"), (rail, "KORKULUK")):
        for ln in group:
            msp.add_lwpolyline(cm(ln), dxfattribs={"layer": layer})
    for name, area, x, y in labels():
        msp.add_text(name, height=20, dxfattribs={"layer": "YAZI"}).set_placement((x * 100, y * 100 + 15), align=ezdxf.enums.TextEntityAlignment.MIDDLE_CENTER)
        msp.add_text(area, height=18, dxfattribs={"layer": "YAZI"}).set_placement((x * 100, y * 100 - 20), align=ezdxf.enums.TextEntityAlignment.MIDDLE_CENTER)
    doc.saveas(p)


def make_pdf(p, scale=50):
    from reportlab.pdfgen import canvas
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    pdfmetrics.registerFont(TTFont("DejaVu", font_path()))
    W, H = 1190.55, 841.89   # A3 landscape in points
    k = 72 / 0.0254 / scale  # points per metre
    ox, oy = 180, 90
    P = lambda x, y: (ox + x * k, oy + y * k)
    c = canvas.Canvas(str(p), pagesize=(W, H))
    c.setFillGray(0.05)
    for x0, y0, x1, y1 in wall_rects():
        (a, b), (e, f) = P(x0, y0), P(x1, y1)
        c.rect(a, b, e - a, f - b, stroke=0, fill=1)
    c.setLineWidth(0.25)
    lines, arcs, wins, rail = symbols()
    for ln in lines + arcs + wins + rail:
        path = c.beginPath()
        path.moveTo(*P(*ln[0]))
        for pt in ln[1:]:
            path.lineTo(*P(*pt))
        c.drawPath(path, stroke=1, fill=0)
    for name, area, x, y in labels():
        c.setFont("DejaVu", 10)
        c.drawCentredString(*P(x, y + 0.12), name)
        c.drawCentredString(*P(x, y - 0.25), area)
    c.setFont("DejaVu", 12)
    c.drawString(ox, 40, f"KAT PLANI   ÖLÇEK 1/{scale}")
    c.save()


def make_scan(pdf, out_pdf, out_png):
    import pypdfium2 as pdfium
    from PIL import Image
    img = np.asarray(pdfium.PdfDocument(str(pdf))[0].render(scale=200 / 72).to_pil().convert("L")).astype(np.float32)
    h, w = img.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), 0.6, 1.0)
    img = cv2.warpAffine(img, M, (w, h), borderValue=255)
    img = cv2.GaussianBlur(img, (0, 0), 0.9)
    img = img * np.linspace(0.93, 1.0, w)[None, :] + np.random.default_rng(1).normal(0, 6, img.shape)
    img = np.clip(img, 0, 255).astype(np.uint8)
    cv2.imwrite(str(out_png), img)
    ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 70])
    Image.open(__import__("io").BytesIO(enc.tobytes())).save(out_pdf, "PDF", resolution=200.0)


def make_photo(scan_png, out_jpg):
    img = cv2.imread(str(scan_png))
    img = cv2.resize(img, None, fx=0.75, fy=0.75, interpolation=cv2.INTER_AREA)
    h, w = img.shape[:2]
    W, H = int(w * 1.35), int(h * 1.45)
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([[0.12 * W, 0.10 * H], [0.90 * W, 0.14 * H], [0.95 * W, 0.88 * H], [0.06 * W, 0.84 * H]])
    M = cv2.getPerspectiveTransform(src, dst)
    bg = np.full((H, W, 3), (70, 80, 95), np.uint8)
    warped = cv2.warpPerspective(img, M, (W, H), borderValue=(0, 0, 0))
    mask = cv2.warpPerspective(np.full((h, w), 255, np.uint8), M, (W, H))
    out = np.where(mask[..., None] > 0, warped, bg).astype(np.float32)
    out *= np.linspace(0.8, 1.05, W)[None, :, None]
    cv2.imwrite(str(out_jpg), np.clip(out, 0, 255).astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, 88])


def truth():
    return [{"label": n, "type": t, "size": sorted([round(x1 - x0, 3), round(y1 - y0, 3)]),
             "area": round((x1 - x0) * (y1 - y0), 3)} for n, t, x0, y0, x1, y1 in ROOMS]


def make_all(d):
    d = Path(d)
    d.mkdir(parents=True, exist_ok=True)
    make_dxf(d / "test_plan.dxf")
    make_pdf(d / "test_plan_vector.pdf")
    make_scan(d / "test_plan_vector.pdf", d / "test_plan_scan.pdf", d / "_scan.png")
    make_photo(d / "_scan.png", d / "test_plan_photo.jpg")
    files = {"dxf": d / "test_plan.dxf", "pdf_vector": d / "test_plan_vector.pdf",
             "pdf_scan": d / "test_plan_scan.pdf", "photo": d / "test_plan_photo.jpg"}
    tool = WS / "opt/libredwg/bin/dxf2dwg"
    if tool.exists():
        try:
            subprocess.run([str(tool), "-y", "-o", str(d / "test_plan.dwg"), str(d / "test_plan.dxf")],
                           check=True, capture_output=True, timeout=300)
            if (d / "test_plan.dwg").stat().st_size > 1000:
                files["dwg"] = d / "test_plan.dwg"
        except Exception as e:
            log(f"dxf2dwg could not write a test DWG ({e}) - DWG input is skipped in the smoke test")
    jsave(d / "truth.json", truth())
    return files


def check(job, kind):
    """Compare detected rooms with the truth. Tolerance: 5 cm (vector) or 3 % (raster)."""
    plan = jload(Path(job) / "02_plan/plan.json", {})
    rooms = [r for r in plan.get("rooms", []) if r["type"] != "balcony"]
    raster = kind in ("pdf_scan", "photo")
    res, ok = [], True
    for t in truth():
        best, err = None, 1e9
        for r in rooms:
            if r["type"] != t["type"]:
                continue
            s = sorted(r["size_m"])
            e = max(abs(s[0] - t["size"][0]) / (t["size"][0] if raster else 1),
                    abs(s[1] - t["size"][1]) / (t["size"][1] if raster else 1))
            if e < err:
                best, err = r, e
        tol = 0.03 if raster else 0.05
        good = best is not None and err <= tol
        ok &= good
        res.append(f"  {t['label']:<22} truth {t['size'][0]:.2f}x{t['size'][1]:.2f}  found "
                   + (f"{sorted(best['size_m'])[0]:.2f}x{sorted(best['size_m'])[1]:.2f}" if best else "-")
                   + f"  {'OK' if good else 'FAIL'}")
    files = ["02_plan/plan.json", "02_plan/overlay.png", "03_layout/layout.json", "04_scene/scene.blend",
             "04_scene/scene.glb", "report.json", "final/apartment.blend", "final/apartment.glb",
             "final/apartment.usdc", "final/overlay.png", "final/report.md"]
    missing = [f for f in files if not (Path(job) / f).exists()]
    renders = list((Path(job) / "05_render").glob("*.png"))
    if not [r for r in renders if "depth" not in r.name]:
        missing.append("05_render/*.png")
    val = jload(Path(job) / "final/validation.json", {})
    if not val.get("ok"):
        missing.append("final/validation.json ok (failed: " + ", ".join(c["name"] for c in val.get("checks", []) if not c["ok"]) + ")")
    ok &= not missing
    return ok, res, missing


# ---------- CPU-only unit checks ----------
def _gltf_box(path, size, node_scale=None, glb=False, image_uri=None):
    """Minimal glTF 2.0 box (size in glTF axes, Y up) for catalog tests - no Blender needed."""
    sx, sy, sz = size
    v = [(x * sx / 2, y * sy / 2 + sy / 2, z * sz / 2) for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]
    idx = [0, 1, 3, 0, 3, 2, 4, 6, 7, 4, 7, 5, 0, 4, 5, 0, 5, 1, 2, 3, 7, 2, 7, 6, 0, 2, 6, 0, 6, 4, 1, 5, 7, 1, 7, 3]
    pos = b"".join(struct.pack("<3f", *p) for p in v)
    ind = b"".join(struct.pack("<H", i) for i in idx) + b"\0\0"
    data = pos + ind
    node = {"mesh": 0}
    if node_scale:
        node["scale"] = node_scale
    js = {"asset": {"version": "2.0"}, "scene": 0, "scenes": [{"nodes": [0]}], "nodes": [node],
          "meshes": [{"primitives": [{"attributes": {"POSITION": 0}, "indices": 1}]}],
          "accessors": [{"bufferView": 0, "componentType": 5126, "count": 8, "type": "VEC3",
                         "min": [min(p[i] for p in v) for i in range(3)], "max": [max(p[i] for p in v) for i in range(3)]},
                        {"bufferView": 1, "componentType": 5123, "count": 36, "type": "SCALAR"}],
          "bufferViews": [{"buffer": 0, "byteOffset": 0, "byteLength": len(pos)},
                          {"buffer": 0, "byteOffset": len(pos), "byteLength": 72}],
          "buffers": [{"byteLength": len(data)}]}
    if image_uri:
        js["images"] = [{"uri": image_uri}]
    path.parent.mkdir(parents=True, exist_ok=True)
    if glb:
        j = json.dumps(js).encode()
        j += b" " * (-len(j) % 4)
        body = struct.pack("<II", len(j), 0x4E4F534A) + j + struct.pack("<II", len(data), 0x004E4942) + data
        path.write_bytes(b"glTF" + struct.pack("<II", 2, 12 + len(body)) + body)
    else:
        js["buffers"][0]["uri"] = "data:application/octet-stream;base64," + base64.b64encode(data).decode()
        path.write_text(json.dumps(js))


def _unit_catalog(tmp):
    from . import furniture
    cfg = load_config()
    cfg["furniture"].update(library=str(tmp / "lib"), user_dir=str(tmp / "user"), legacy_models=str(tmp / "none"))
    u = tmp / "user"
    _gltf_box(u / "sofa/a.glb", (2.0, 0.8, 0.9), glb=True)                      # 2.0 w x 0.9 d x 0.8 h after import
    (u / "sofa/a.json").write_text(json.dumps({"licence": "CC0", "style_tags": ["modern"]}))
    _gltf_box(u / "sofa/b_cm.gltf", (210, 85, 90))                              # centimetres
    (u / "sofa/b_cm.json").write_text(json.dumps({"licence": "CC BY 4.0", "author": "A. Author", "source_url": "https://x.invalid"}))
    _gltf_box(u / "sofa/c_scaled.gltf", (1, 1, 1), node_scale=[1.8, 0.85, 0.9])  # node transform
    (u / "sofa/licence.json").write_text(json.dumps({"licence": "owned"}))
    _gltf_box(u / "wardrobe/nolic.glb", (2, 2.2, 0.6), glb=True)                 # no licence -> skipped
    _gltf_box(u / "desk/missing_tex.gltf", (1.2, 0.75, 0.6), image_uri="tex/nothere.png")
    (u / "desk/missing_tex.json").write_text(json.dumps({"licence": "CC0-1.0"}))
    _gltf_box(u / "sofa/gpl.glb", (2.0, 0.8, 0.9), glb=True)
    (u / "sofa/gpl.json").write_text(json.dumps({"licence": "GPL-3.0"}))
    cat = furniture.build_catalog(cfg)
    ids = {i["id"]: i for i in cat["items"]}
    reasons = {Path(e["file"]).name: e["reason"] for e in cat["excluded"]}
    assert set(ids) == {"user:sofa_a", "user:sofa_b_cm", "user:sofa_c_scaled"}, sorted(ids)
    assert ids["user:sofa_a"]["dims_m"] == [2.0, 0.9, 0.8], ids["user:sofa_a"]["dims_m"]
    assert ids["user:sofa_b_cm"]["unit_scale"] == 0.01 and ids["user:sofa_b_cm"]["dims_m"] == [2.1, 0.9, 0.85]
    assert ids["user:sofa_c_scaled"]["dims_m"] == [1.8, 0.9, 0.85], ids["user:sofa_c_scaled"]["dims_m"]
    assert ids["user:sofa_b_cm"]["licence"] == "CC-BY-4.0" and ids["user:sofa_c_scaled"]["licence"] == "owned"
    assert "licence" in reasons["nolic.glb"] and "GPL-3.0" in reasons["gpl.glb"] and "missing files" in reasons["missing_tex.gltf"]
    lic = (tmp / "lib/LICENSES.txt").read_text()
    assert "Powered by Poly Haven" in lic and "A. Author" in lic and "Attribution required" in lic and "Skipped files" in lic
    job = tmp / "job"
    (job / "03_layout").mkdir(parents=True)
    jsave(job / "03_layout/layout.json", {"items": [
        {"id": "r1_sofa_0", "type": "sofa", "room": "r1", "size": [2.10, 0.90, 0.85], "center": [0, 0], "rot_deg": 0},
        {"id": "r1_sofa_1", "type": "sofa", "room": "r1", "size": [1.20, 0.80, 0.85], "center": [0, 0], "rot_deg": 0},
        {"id": "r1_rug", "type": "rug", "room": "r1", "size": [2.4, 1.7, 0.01], "center": [0, 0], "rot_deg": 0}]})
    sel = furniture.select_assets(job, cfg)["items"]
    # style match ('modern' is in the config style) ranks before the smaller fit error of the cm sofa
    assert sel["r1_sofa_0"]["asset_id"] == "user:sofa_a" and sel["r1_sofa_0"]["style"] == 1, sel["r1_sofa_0"]
    assert sel["r1_sofa_1"]["source"] == "parametric" and "none fits" in sel["r1_sofa_1"]["reason"], sel["r1_sofa_1"]
    assert sel["r1_rug"]["source"] == "parametric"
    assert furniture.set_override(job, cfg, "r1_sofa_0", "user:sofa_b_cm") == []
    assert furniture.select_assets(job, cfg)["items"]["r1_sofa_0"]["asset_id"] == "user:sofa_b_cm"
    assert furniture.set_override(job, cfg, "r1_sofa_1", "user:sofa_a")          # does not fit -> error list
    return f"{len(ids)} models, {len(cat['excluded'])} skipped with reasons, unit/transform/licence/fit/override OK"


def _unit_patches(job):
    from .patches import PatchStore, validate_patch
    ps = PatchStore(job)
    plan, _ = ps.after_plan_stage()
    assert validate_patch({"ops": [{"op": "explode"}], "reason": "xx x"})
    assert validate_patch({"ops": [{"op": "scale_plan", "factor": 1.1, "vertices": [1]}], "reason": "xxxx"})
    assert validate_patch({"ops": [{"op": "scale_plan", "factor": 1.1}]})
    r = ps.propose({"ops": [{"op": "scale_plan", "factor": 1.9}], "reason": "too big"})
    assert not r["accepted"] and r["stage"] == "invariants", r
    r = ps.propose({"ops": [{"op": "remove_opening", "id": d["id"]} for d in plan["doors"]][:8], "reason": "remove all"})
    assert not r["accepted"], r
    hall = next(x["id"] for x in plan["rooms"] if x["type"] == "hall")
    r = ps.propose({"ops": [{"op": "retype_room", "room_id": hall, "type": "study"}, {"op": "scale_plan", "factor": 1.02}],
                    "reason": "test"})
    assert r["accepted"], r
    now = jload(job / "02_plan/plan.json")
    rb, _ = ps.rebuild()
    assert json.dumps(now, sort_keys=True) == json.dumps(rb, sort_keys=True), "rebuild differs"
    return "schema errors, invariant rejections, accepted patch and deterministic rebuild OK"


MOCK_TURNS = [
    {"content": "start", "calls": [{"name": "parse_plan", "arguments": {}}]},
    {"content": "bad calls", "calls": [{"name": "patch_plan", "arguments": "{not json"}, {"name": "draw_walls", "arguments": {}},
                                       {"name": "get_plan_summary", "arguments": {"detail": "everything"}}]},
    {"content": "too big", "calls": [{"name": "patch_plan", "arguments": {"ops": [{"op": "scale_plan", "factor": 1.9}], "reason": "test"}}]},
    {"content": "retype", "calls": [{"name": "patch_plan", "arguments": {"ops": [{"op": "retype_room", "room_id": "r2", "type": "study"}],
                                                                          "reason": "unit test"}}]},
    {"content": "checks", "calls": [{"name": "run_checks", "arguments": {}}]},
    {"content": "furnish", "calls": [{"name": "relayout", "arguments": {}}, {"name": "choose_furniture", "arguments": {}}]},
    {"content": "drop sofa", "calls": [{"name": "patch_plan", "arguments": {"ops": [{"op": "remove_furniture", "item_id": "r5_sofa_0"}],
                                                                             "reason": "unit test"}}]},
    {"content": "finish", "calls": [{"name": "finish", "arguments": {"status": "complete", "summary": "unit"}}]},
    {"content": "finish again", "calls": [{"name": "finish", "arguments": {"status": "partial", "summary": "unit", "uncertain": ["cpu"]}}]},
]


def _unit_agent(inp, tmp):
    from .agent import ScriptBackend, run_agent, replay_turns
    from .vision import NoVision, ScriptedVision
    cfg = load_config()
    cfg["furniture"].update(library=str(tmp / "lib"), user_dir=str(tmp / "user"))
    run_agent(inp, tmp / "a", cfg, ScriptBackend(copy.deepcopy(MOCK_TURNS)), NoVision())
    recs = [json.loads(l) for l in (tmp / "a/agent_log.jsonl").read_text().splitlines()]
    tools = [r for r in recs if r["type"] == "tool"]
    want = [("parse_plan", True), ("patch_plan", False), ("draw_walls", False), ("get_plan_summary", False),
            ("patch_plan", True), ("patch_plan", True), ("run_checks", True), ("relayout", True), ("choose_furniture", True),
            ("patch_plan", True), ("finish", True), ("finish", True)]
    assert [(t["tool"], t["ok"]) for t in tools] == want, [(t["tool"], t["ok"]) for t in tools]
    assert not tools[4]["result"]["accepted"] and tools[5]["result"]["accepted"] and tools[9]["result"]["accepted"]
    assert not tools[10]["result"]["accepted"] and tools[11]["result"]["accepted"], "finish guard"
    assert jload(tmp / "a/03_layout/asset_overrides.json") == {"r5_sofa_0": "remove"}
    assert (tmp / "a/final/report.md").exists() and "Still uncertain" in (tmp / "a/final/report.md").read_text()
    turns, answers = replay_turns(tmp / "a/agent_log.jsonl")
    run_agent(inp, tmp / "b", cfg, ScriptBackend(turns), ScriptedVision(answers))
    for rel in ("02_plan/plan.json", "02_plan/patches.json", "03_layout/layout.json", "03_layout/assets.json",
                "03_layout/asset_overrides.json"):
        a, b = jload(tmp / "a" / rel), jload(tmp / "b" / rel)
        if isinstance(a, list):
            a, b = [{k: v for k, v in x.items() if k != "ts"} for x in a], [{k: v for k, v in x.items() if k != "ts"} for x in b]
        assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True), f"replay differs: {rel}"
    return f"{len(tools)} tool calls (4 refused as designed), finish guard, replay identical"


def _unit_server():
    from .agent import parse_text_tool_calls, openai_tools
    from .llm_server import LLMServer, VramPlanner, memory_utilization
    from .vision import parse_json
    cfg = load_config()
    s = LLMServer(cfg)
    cmd = " ".join(s.command())
    for flag in ("serve", "--enable-auto-tool-choice", "--tool-call-parser qwen3_coder", "--reasoning-parser qwen3",
                 "--default-chat-template-kwargs", "--max-model-len", "--gpu-memory-utilization", "--limit-mm-per-prompt"):
        assert flag in cmd, flag
    env = s.server_env()
    assert env["VLLM_USE_FLASHINFER_SAMPLER"] == "0" and env["PATH"].split(":")[0].endswith("venv-llm/bin"), env["PATH"][:80]
    assert memory_utilization(cfg, "Qwen/Qwen3.5-9B", 46068) == 0.57, memory_utilization(cfg, "Qwen/Qwen3.5-9B", 46068)
    assert memory_utilization(cfg, "Qwen/Qwen3.5-4B", 24564) == 0.65
    assert memory_utilization(cfg, "Qwen/Qwen3.6-27B-FP8", 46068) == 0.8

    class FakeServer:
        def __init__(self, mb):
            self.mb, self.up = mb, True

        def running(self):
            return self.up

        def budget_mb(self):
            return self.mb

        def stop(self, wait_free=True):
            self.up = False

    for total, srv, comp, keep in ((46068, 26259, "cycles_preview", True), (46068, 26259, "vlm_transformers_9b", False),
                                   (24564, 15967, "cycles_preview", True), (24564, 15967, "sdxl_polish", False)):
        f = FakeServer(srv)
        p = VramPlanner(cfg, f)
        p.total = total
        p.before(comp)
        assert f.up == keep, (total, comp)
    calls = parse_text_tool_calls('<tool_call>\n<function=patch_plan>\n<parameter=reason>\nx\n</parameter>\n'
                                  '<parameter=ops>\n[{"op": "scale_plan", "factor": 1.1}]\n</parameter>\n</function>\n</tool_call>')
    assert calls and json.loads(calls[0]["function"]["arguments"])["ops"][0]["factor"] == 1.1, calls
    assert parse_json('blah {"overall": "minor", "issues": []} end')["overall"] == "minor"
    assert len(openai_tools()) == 12
    return "vLLM command flags, gpu-memory-utilization estimates, VRAM planner, text tool-call + JSON parsing, 12 tools"


def _unit_textnorm():
    from .textnorm import classify
    assert classify("EBEVEYN BANYO")["type"] == "bathroom"
    assert classify("SALON 18,50 m²") == {"kind": "room_label", "type": "living", "area": 18.5}
    assert classify("ÖLÇEK 1/50") == {"kind": "scale", "value": 50}
    assert classify("ÇOCUK ODASI")["child"] is True
    return "room words, areas, scale notes"


def unit():
    """CPU-only checks. Returns True when all pass. Blender checks run only when Blender (or bpy) is available."""
    tmp = WS / "outputs/unit_test"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    d = WS / "inputs/smoke_test"
    files = {"dxf": d / "test_plan.dxf", "pdf_vector": d / "test_plan_vector.pdf"}
    if not all(f.exists() for f in files.values()):
        files = {k: v for k, v in make_all(d).items() if k in files}
    results = []

    def T(name, fn):
        t0 = time.time()
        try:
            results.append((name, True, fn(), time.time() - t0))
        except Exception as e:
            results.append((name, False, f"{type(e).__name__}: {e}\n{traceback.format_exc()[-800:]}", time.time() - t0))

    T("textnorm", _unit_textnorm)
    for kind, f in files.items():
        def geo(kind=kind, f=f):
            job = tmp / f"geo_{kind}"
            r = subprocess.run([sys.executable, "-m", "app.main", str(f), "--job-dir", str(job), "--to-stage", "layout"],
                               cwd=str(WS), capture_output=True, text=True)
            assert r.returncode == 0, r.stdout[-1500:] + r.stderr[-1500:]
            ok, lines, _ = check(job, kind)
            assert all(l.endswith("OK") for l in lines), "\n".join(lines)
            lay = jload(job / "03_layout/layout.json")
            assert all(v == 0 for v in lay["checks"].values()) and not lay["unfurnished_main_rooms"], lay["checks"]
            return f"{len(lines)} rooms within tolerance, {len(lay['items'])} furniture items, layout checks 0"
        T(f"plan+layout {kind}", geo)
    T("patches", lambda: _unit_patches(_copy_job(tmp / "geo_dxf", tmp / "patch_job")))
    T("agent mock + replay", lambda: _unit_agent(files["dxf"], tmp / "agent"))
    T("catalog + licences", lambda: _unit_catalog(tmp / "catalog"))
    T("llm server + planner", _unit_server)
    if Path(BLENDER).exists():
        def bl():
            from .common import blender
            from .deliver import deliver
            job = tmp / "agent/a"
            blender("blender_scene.py", job)
            val = deliver(job)
            bad = [f"{c['name']}: {c['detail']}" for c in val["checks"] if not c["ok"]]
            assert val["ok"], bad
            return f"scene + export + {len(val['checks'])} validation checks passed"
        T("blender scene/export/validate (CPU)", bl)
    else:
        results.append(("blender scene/export/validate", None, f"skipped: no Blender at {BLENDER}", 0.0))
    print("\n===== UNIT CHECKS (CPU) =====")
    for name, ok, msg, sec in results:
        print(f"{'PASS' if ok else 'SKIP' if ok is None else 'FAIL'}  {name:<36} {sec:6.1f}s  {msg}")
    all_ok = all(ok is not False for _, ok, _, _ in results)
    print(f"UNIT CHECKS: {'PASS' if all_ok else 'FAIL'}")
    return all_ok


def _copy_job(src, dst):
    shutil.copytree(src, dst)
    return dst


def agent_smoke(f):
    """One agentic run on the DXF test plan: LLM backend if the agent LLM is installed, else the rules backend."""
    cfg = load_config()
    backend = cfg["agent"]["backend"]
    if backend == "llm" and not (Path(cfg["llm"]["venv"]) / "bin/vllm").exists():
        backend = "rules"
    job = WS / "outputs/smoke_agent"
    if job.exists():
        shutil.rmtree(job)
    t0 = time.time()
    r = subprocess.run([sys.executable, "-m", "app.agent", "run", str(f), "--job-dir", str(job), "--backend", backend],
                       cwd=str(WS))
    s = jload(job / "final/agent_summary.json", {})
    ok, lines, missing = check(job, "dxf")
    missing = [m for m in missing if not m.startswith(("04_scene/scene.glb", "05_render", "report.json"))]
    tools = [json.loads(l) for l in (job / "agent_log.jsonl").read_text().splitlines()] if (job / "agent_log.jsonl").exists() else []
    peak = max([t.get("vram_peak_mb") or 0 for t in tools] or [0])
    good = r.returncode == 0 and s.get("export_ok") and not missing and all(l.endswith("OK") for l in lines)
    out = [f"agent ({backend}) {'PASS' if good else 'FAIL'}  ({time.time() - t0:.0f} s, status {s.get('status')}, "
           f"{s.get('steps')} tool calls, VRAM peak {peak} MB)  job: {job}"] + lines
    out += [f"  missing: {', '.join(missing)}"] if missing else []
    out += [f"  uncertain: {u}" for u in (s.get("uncertain") or [])[:5]]
    return good, out


def smoke(profile="quick", polish=True):
    unit_ok = unit()
    d = WS / "inputs/smoke_test"
    files = make_all(d)
    summary, all_ok = [f"unit checks {'PASS' if unit_ok else 'FAIL'}"], unit_ok
    for kind, f in files.items():
        job = WS / "outputs" / f"smoke_{kind}"
        if job.exists():
            shutil.rmtree(job)
        cmd = [sys.executable, "-m", "app.main", str(f), "--job-dir", str(job), "--profile", profile]
        if not polish or kind not in ("dxf", "pdf_scan"):
            cmd.append("--no-polish")   # polish is tested on two inputs to save time
        t0 = time.time()
        r = subprocess.run(cmd, cwd=str(WS))
        ok, lines, missing = check(job, kind) if r.returncode == 0 else (False, [], ["pipeline crashed"])
        all_ok &= ok
        summary.append(f"{kind:<11} {'PASS' if ok else 'FAIL'}  ({time.time() - t0:.0f} s)  job: {job}")
        summary += lines + ([f"  missing: {', '.join(missing)}"] if missing else [])
    ok, lines = agent_smoke(files["dxf"])
    all_ok &= ok
    summary += lines
    print("\n===== SMOKE TEST =====\n" + "\n".join(summary))
    print(f"SMOKE TEST: {'PASS' if all_ok else 'FAIL'}")
    return all_ok


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "quick"
    ok = unit() if mode == "unit" else smoke(profile=mode)
    sys.exit(0 if ok else 1)
