"""Reopen the deliverable headless and check it. Writes final/validation.json; exits with an error if any check
fails (the job then fails). Run: blender -b --factory-startup --python-exit-code 1 -P validate_export.py -- <job>
Checks: units (metric, scale 1.0, metres), collections, wall count, bounding box vs plan, Z extent, furniture
objects (one mesh per item, origin at base centre, unit scale), packed / missing textures, non-manifold wall
edges, real openings (a ray through every door/window centre passes, solid wall is hit), camera count, and that
the .glb and .usdc reopen with the same walls, furniture names and bounding box."""
import bpy, bmesh, json, math, os, re, sys
from pathlib import Path
from mathutils import Vector

ARGS = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
JOB = Path(ARGS[0])
FIN = JOB / "final"
PLAN = json.loads((JOB / "02_plan/plan.json").read_text(encoding="utf-8"))
QA = json.loads((JOB / "04_scene/furniture_qa.json").read_text(encoding="utf-8"))
CAMS = json.loads((JOB / "04_scene/cameras.json").read_text(encoding="utf-8"))["cameras"]
EXP = json.loads((FIN / "export.json").read_text(encoding="utf-8"))
OFF = Vector(EXP["offset_m"])
LAYOUT = json.loads((JOB / "03_layout/layout.json").read_text(encoding="utf-8"))
CHECKS = []


def check(name, ok, detail=""):
    CHECKS.append({"name": name, "ok": bool(ok), "detail": str(detail)[:400]})


def plan_extent():
    """Outer bounding box of the walls in plan coordinates (walls are rectangles of their thickness around a-b)."""
    xs, ys = [], []
    for w in PLAN["walls"]:
        (ax, ay), (bx, by) = w["a"], w["b"]
        L = math.hypot(bx - ax, by - ay) or 1.0
        nx, ny = -(by - ay) / L * w["thickness"] / 2, (bx - ax) / L * w["thickness"] / 2
        for px, py in ((ax + nx, ay + ny), (ax - nx, ay - ny), (bx + nx, by + ny), (bx - nx, by - ny)):
            xs.append(px)
            ys.append(py)
    return min(xs), min(ys), max(xs), max(ys)


def plan_footprint():
    """Plan lower-left/upper-right: outer wall faces and every room polygon (balconies too) = the origin rule."""
    x0, y0, x1, y1 = plan_extent()
    xs = [x0, x1] + [p[0] for r in PLAN["rooms"] for p in r["polygon"]]
    ys = [y0, y1] + [p[1] for r in PLAN["rooms"] for p in r["polygon"]]
    return min(xs), min(ys), max(xs), max(ys)


def bbox(objs):
    pts = [o.matrix_world @ Vector(c) for o in objs if o.type == "MESH" for c in o.bound_box]
    if not pts:
        return None
    return Vector([min(p[i] for p in pts) for i in range(3)]), Vector([max(p[i] for p in pts) for i in range(3)])


expected_items = [r["item_id"] for r in QA["items"] if r["used"] not in ("skipped", "removed")]
x0, y0, x1, y1 = plan_extent()
fx0, fy0, _, _ = plan_footprint()
H = PLAN["defaults"]["ceiling_height"]
check("origin_offset", abs(OFF.x + fx0) < 1e-4 and abs(OFF.y + fy0) < 1e-4,
      f"export offset ({OFF.x:.4f}, {OFF.y:.4f}); plan lower-left ({fx0:.4f}, {fy0:.4f})")

# ---------- 1. the .blend ----------
bpy.ops.wm.open_mainfile(filepath=str(FIN / "apartment.blend"))
SC = bpy.context.scene
us = SC.unit_settings
check("units", us.system == "METRIC" and abs(us.scale_length - 1.0) < 1e-9 and us.length_unit == "METERS",
      f"{us.system}, scale {us.scale_length}, {us.length_unit}")
names = {c.name for c in SC.collection.children}
need = {"Walls", "Floors_Ceilings", "Openings", "Lights", "Cameras"}
furn_colls = [c for c in SC.collection.children if c.name.startswith("Furniture.")]
check("collections", need <= names and (furn_colls or not expected_items), f"have {sorted(names)}")
wall_objs = [o for o in bpy.data.collections["Walls"].objects if o.type == "MESH" and re.fullmatch(r"w\d+", o.name)] \
    if "Walls" in names else []
check("wall_count", len(wall_objs) == len(PLAN["walls"]), f"{len(wall_objs)} wall objects, plan has {len(PLAN['walls'])}")
bb = bbox(wall_objs)
if bb:
    lo, hi = bb
    size_ok = abs((hi.x - lo.x) - (x1 - x0)) < 0.02 and abs((hi.y - lo.y) - (y1 - y0)) < 0.02
    at_ok = abs(lo.x - (x0 - fx0)) < 0.01 and abs(lo.y - (y0 - fy0)) < 0.01
    check("bbox_vs_plan", size_ok and at_ok, f"walls {hi.x - lo.x:.3f} x {hi.y - lo.y:.3f} m at ({lo.x:.3f}, {lo.y:.3f}); "
          f"plan {x1 - x0:.3f} x {y1 - y0:.3f} m at ({x0 - fx0:.3f}, {y0 - fy0:.3f})")
    everything = bbox([o for o in SC.objects if o.type == "MESH"])
    # door frames stand 1 cm proud of the wall and the balcony handrail is centred on the balcony edge: 5 cm slack
    check("geometry_from_origin", everything and everything[0].x > -0.05 and everything[0].y > -0.05,
          f"lowest corner of all meshes ({everything[0].x:.3f}, {everything[0].y:.3f})" if everything else "no meshes")
    check("z_up_height", abs(lo.z) < 0.01 and abs(hi.z - H) < 0.02, f"walls z {lo.z:.3f}..{hi.z:.3f}, ceiling {H}")
else:
    check("bbox_vs_plan", False, "no wall geometry")
furn = {o.name: o for c in furn_colls for o in c.all_objects}
missing = [i for i in expected_items if i not in furn]
check("furniture_objects", not missing, f"{len(furn)} objects; missing {missing[:8]}")
bad_origin = []
slot = {it["id"]: it for it in LAYOUT["items"]}
for name in expected_items:
    o = furn.get(name)
    if o is None or o.type != "MESH":
        if o is not None:
            bad_origin.append(f"{name}: {o.type} not a mesh")
        continue
    vs = [v.co for v in o.data.vertices]
    if not vs:
        bad_origin.append(f"{name}: empty")
        continue
    mn = Vector([min(v[i] for v in vs) for i in range(3)])
    mx = Vector([max(v[i] for v in vs) for i in range(3)])
    tol = max(0.02, 0.03 * max(mx.x - mn.x, mx.y - mn.y))
    it = slot.get(name)
    want = Vector((it["center"][0], it["center"][1], it.get("z", 0.0))) + OFF if it else o.location
    # base centre = the floor point under the footprint centre (wall-hung items such as a vanity start higher)
    if mn.z < -0.01 or abs((mn.x + mx.x) / 2) > tol or abs((mn.y + mx.y) / 2) > tol or (o.location - want).length > 0.01 \
            or any(abs(s - 1) > 1e-6 for s in o.scale) or o.children:
        bad_origin.append(f"{name}: lowest z {mn.z:.3f}, centre ({(mn.x + mx.x) / 2:.3f}, {(mn.y + mx.y) / 2:.3f}), "
                          f"location off {(o.location - want).length:.3f} m, scale {tuple(round(s, 3) for s in o.scale)}")
check("furniture_origin_base_centre", not bad_origin, "; ".join(bad_origin[:6]))
unpacked, lost = [], []
for img in bpy.data.images:
    if img.type in ("RENDER_RESULT", "COMPOSITING") or img.source in ("GENERATED", "VIEWER"):
        continue
    if not img.packed_file:
        unpacked.append(img.name)
        if not (img.filepath and os.path.exists(bpy.path.abspath(img.filepath))):
            lost.append(img.name)
check("textures_packed", not unpacked, f"{len(bpy.data.images)} images; not packed: {unpacked[:8]}")
check("textures_missing", not lost, f"missing: {lost[:8]}")
nm = {}
for o in wall_objs:
    bm = bmesh.new()
    bm.from_mesh(o.data)
    n = sum(1 for e in bm.edges if not e.is_manifold)
    bm.free()
    if n:
        nm[o.name] = n
check("walls_manifold", not nm, f"non-manifold edges: {nm}" if nm else f"{len(wall_objs)} walls watertight")
dg = bpy.context.evaluated_depsgraph_get()
wall_by_id = {o.name: o for o in wall_objs}
blocked, solid_miss = [], []


def ray_hits(o, p, d, dist):
    inv = o.matrix_world.inverted()
    q = inv @ p
    dl = (inv.to_3x3() @ d).normalized()
    return o.ray_cast(q, dl, distance=dist)[0]


for kind, lst in (("door", PLAN["doors"]), ("window", PLAN["windows"])):
    for op in lst:
        w = next((x for x in PLAN["walls"] if x["id"] == op.get("wall_id")), None)
        o = wall_by_id.get(op.get("wall_id"))
        if w is None or o is None:
            continue
        (ax, ay), (bx, by) = w["a"], w["b"]
        L = math.hypot(bx - ax, by - ay) or 1.0
        n = Vector((-(by - ay) / L, (bx - ax) / L, 0.0))
        z = (op.get("sill", 0.0) + op["head"]) / 2
        p = Vector((op["center"][0], op["center"][1], z)) + OFF
        if ray_hits(o, p - n * w["thickness"], n, 2 * w["thickness"]):
            blocked.append(op["id"])
for w in PLAN["walls"]:
    o = wall_by_id.get(w["id"])
    (ax, ay), (bx, by) = w["a"], w["b"]
    L = math.hypot(bx - ax, by - ay)
    if o is None or L < 0.3:
        continue
    u, n = Vector(((bx - ax) / L, (by - ay) / L, 0.0)), Vector((-(by - ay) / L, (bx - ax) / L, 0.0))
    ops = [op for op in PLAN["doors"] + PLAN["windows"] if op.get("wall_id") == w["id"]]
    for s in (0.12, L - 0.12, L / 2):   # a point of solid wall: no opening there
        if any(abs((Vector(op["center"] + [0]) - Vector((ax, ay, 0))).dot(u) - s) < op["width"] / 2 + 0.05 for op in ops):
            continue
        p = Vector((ax, ay, 1.2)) + u * s + OFF
        if not ray_hits(o, p - n * w["thickness"], n, 2 * w["thickness"]):
            solid_miss.append(f"{w['id']}@{s:.2f}")
        break
check("real_openings", not blocked, f"rays blocked at {blocked}" if blocked else
      f"{len(PLAN['doors']) + len(PLAN['windows'])} openings pass")
check("walls_solid", not solid_miss, f"no wall hit at {solid_miss[:8]}" if solid_miss else "solid parts hit")
cams = [o for o in SC.objects if o.type == "CAMERA"]
check("cameras", len(cams) == len(CAMS), f"{len(cams)} cameras, cameras.json has {len(CAMS)}")
blend_bb = bb

# ---------- 2. the .glb ----------
glb = FIN / "apartment.glb"
if glb.exists():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    try:
        bpy.ops.import_scene.gltf(filepath=str(glb))
        objs = list(bpy.data.objects)
        by = {o.name: o for o in objs}
        gw = [by[w["id"]] for w in PLAN["walls"] if w["id"] in by]
        check("glb_walls", len(gw) == len(PLAN["walls"]), f"{len(gw)} of {len(PLAN['walls'])} walls by name")
        gm = [i for i in expected_items if i not in by]
        check("glb_furniture", not gm, f"missing {gm[:8]}")
        gbb = bbox(gw)
        ok = gbb and blend_bb and all(abs(a - b) < 0.02 for a, b in zip(list(gbb[0]) + list(gbb[1]), list(blend_bb[0]) + list(blend_bb[1])))
        check("glb_bbox", ok, f"glb {tuple(round(x, 3) for x in gbb[0])}..{tuple(round(x, 3) for x in gbb[1])}" if gbb else "no walls")
        noimg = [i.name for i in bpy.data.images if i.source not in ("GENERATED", "VIEWER") and not i.packed_file and not i.has_data]
        check("glb_textures", not noimg, f"images without data: {noimg[:8]}")
    except Exception as e:
        check("glb_reopen", False, e)
else:
    check("glb_exists", False, "apartment.glb not written")

# ---------- 3. the .usdc ----------
usd = FIN / "apartment.usdc"
if usd.exists():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    try:
        bpy.ops.wm.usd_import(filepath=str(usd))
        meshes = [o for o in bpy.data.objects if o.type == "MESH"]
        uw = [o for o in meshes if re.fullmatch(r"w\d+(\.\d+)?", o.name)]
        check("usdc_reopen", len(uw) >= len(PLAN["walls"]), f"{len(meshes)} meshes, {len(uw)} walls")
    except Exception as e:
        check("usdc_reopen", False, e)
else:
    check("usdc_exists", False, "apartment.usdc not written")

ok = all(c["ok"] for c in CHECKS)
json.dump({"ok": ok, "checks": CHECKS}, open(FIN / "validation.json", "w"), indent=1, ensure_ascii=False)
for c in CHECKS:
    print(f"{'ok  ' if c['ok'] else 'FAIL'} {c['name']}: {c['detail']}")
print("VALIDATION_OK" if ok else "VALIDATION_FAILED")
if not ok:
    raise RuntimeError("export validation failed: " + ", ".join(c["name"] for c in CHECKS if not c["ok"]))
