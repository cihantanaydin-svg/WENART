"""Stage 4 - build the 3D apartment in Blender (background mode).
Run: blender -b --factory-startup --python-exit-code 1 -P blender_scene.py -- <job_dir>
Writes 04_scene/scene.blend, scene.glb, cameras.json, furniture_qa.json. Uses only bpy/bmesh/mathutils.
Walls are watertight solids with real openings. Furniture: the catalog model chosen in 03_layout/assets.json
(normalised: real size, Z up, front to +Y, origin at base centre, polygon cap, slot fit with limited non-uniform
scale) or the parametric model; one named object per item. Collections: Walls, Floors_Ceilings, Openings,
Furniture.<room>, Lights, Cameras."""
import bpy, bmesh, json, math, os, re, statistics, sys
from pathlib import Path
from mathutils import Vector, Matrix

ARGS = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
JOB = Path(ARGS[0])
RC = json.loads((JOB / "run_config.json").read_text())
PLAN = json.loads((JOB / "02_plan/plan.json").read_text(encoding="utf-8"))
LAYOUT = json.loads((JOB / "03_layout/layout.json").read_text(encoding="utf-8"))
_AF = JOB / "03_layout/assets.json"
CHOICE = json.loads(_AF.read_text(encoding="utf-8")) if _AF.exists() else None   # None = old behaviour
ASSETS = Path(RC["assets"])
OUT = JOB / "04_scene"
OUT.mkdir(parents=True, exist_ok=True)
WALLS = {w["id"]: w for w in PLAN["walls"]}
ROOMS = PLAN["rooms"]
H = PLAN["defaults"]["ceiling_height"]
WARN = []

bpy.ops.wm.read_factory_settings(use_empty=True)
SC = bpy.context.scene
SC.render.engine = "CYCLES"
COLL = SC.collection


# ---------- small 2D helpers ----------
def pip(pt, poly):
    x, y = pt
    inside = False
    for i in range(len(poly)):
        (x1, y1), (x2, y2) = poly[i], poly[(i + 1) % len(poly)]
        if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1) + x1:
            inside = not inside
    return inside


def room_at(pt):
    return next((r for r in ROOMS if pip(pt, r["polygon"])), None)


def unit2(v):
    L = math.hypot(v[0], v[1]) or 1.0
    return (v[0] / L, v[1] / L)


def centroid(poly):
    a = cx = cy = 0.0
    for i in range(len(poly)):
        (x1, y1), (x2, y2) = poly[i], poly[(i + 1) % len(poly)]
        f = x1 * y2 - x2 * y1
        a += f
        cx += (x1 + x2) * f
        cy += (y1 + y2) * f
    a *= 0.5
    return (cx / (6 * a), cy / (6 * a)) if abs(a) > 1e-9 else poly[0]


# ---------- materials (ambientCG CC0 textures when downloaded, else simple colours) ----------
SPEC = {
    "wall_paint": dict(c=(0.84, 0.80, 0.74), r=0.9, tex=("PaintedPlaster017", 2.5)),
    "ceiling": dict(c=(0.88, 0.87, 0.85), r=0.9), "exterior": dict(c=(0.72, 0.69, 0.63), r=0.9),
    "wood_floor": dict(c=(0.55, 0.40, 0.27), r=0.45, tex=("WoodFloor051", 1.6)),
    "tile_floor": dict(c=(0.76, 0.74, 0.70), r=0.25, tiles=(0.6, 0.6, (0.72, 0.70, 0.66), (0.55, 0.53, 0.50))),
    "tile_wall": dict(c=(0.90, 0.89, 0.87), r=0.12, tiles=(0.3, 0.6, (0.87, 0.86, 0.84), (0.70, 0.70, 0.68))),
    "fabric_sofa": dict(c=(0.60, 0.56, 0.50), r=0.95, tex=("Fabric061", 0.6)),
    "fabric_bed": dict(c=(0.93, 0.92, 0.89), r=0.9), "fabric_accent": dict(c=(0.62, 0.55, 0.46), r=0.92),
    "wood": dict(c=(0.66, 0.50, 0.34), r=0.5, tex=("Wood049", 1.2)), "white_matte": dict(c=(0.88, 0.87, 0.85), r=0.45),
    "black_metal": dict(c=(0.03, 0.03, 0.03), r=0.35, m=1.0), "steel": dict(c=(0.75, 0.75, 0.74), r=0.25, m=1.0),
    "ceramic": dict(c=(0.95, 0.95, 0.94), r=0.06), "stone_counter": dict(c=(0.86, 0.85, 0.83), r=0.2, tex=("Marble012", 1.5)),
    "screen": dict(c=(0.01, 0.01, 0.012), r=0.05), "rug": dict(c=(0.74, 0.69, 0.61), r=1.0, tex=("Carpet016", 1.2)),
    "door": dict(c=(0.90, 0.89, 0.86), r=0.45), "frame": dict(c=(0.94, 0.94, 0.93), r=0.35),
    "sill": dict(c=(0.90, 0.89, 0.87), r=0.2), "mirror": dict(c=(0.95, 0.95, 0.95), r=0.02, m=1.0),
    "glass": dict(c=(1, 1, 1), r=0.0), "pot": dict(c=(0.25, 0.22, 0.2), r=0.6),
}
_M = {}


def find_map(folder, key):
    for f in sorted(folder.glob("*")):
        if key.lower() in f.name.lower() and f.suffix.lower() in (".jpg", ".png"):
            return f
    return None


def mat(name):
    if name in _M:
        return _M[name]
    s = SPEC[name]
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt, N = m.node_tree, m.node_tree.nodes
    b = N.get("Principled BSDF")
    b.inputs["Base Color"].default_value = (*s["c"], 1)
    b.inputs["Roughness"].default_value = s["r"]
    b.inputs["Metallic"].default_value = s.get("m", 0.0)
    m.diffuse_color = (*s["c"], 1)
    uvmap = None
    if "tex" in s or "tiles" in s:
        tc, mp = N.new("ShaderNodeTexCoord"), N.new("ShaderNodeMapping")
        nt.links.new(tc.outputs["UV"], mp.inputs["Vector"])
        uvmap = mp
    if name == "glass":   # clear glass that lets sun and sky light through (shadow rays skip it)
        b.inputs["Transmission Weight"].default_value = 1.0
        b.inputs["IOR"].default_value = 1.45
        out = N.get("Material Output")
        lp, tr, mix = N.new("ShaderNodeLightPath"), N.new("ShaderNodeBsdfTransparent"), N.new("ShaderNodeMixShader")
        nt.links.new(lp.outputs["Is Shadow Ray"], mix.inputs[0])
        nt.links.new(b.outputs[0], mix.inputs[1])
        nt.links.new(tr.outputs[0], mix.inputs[2])
        nt.links.new(mix.outputs[0], out.inputs["Surface"])
    if "tiles" in s:
        tw, th, c2, mortar = s["tiles"]
        br = N.new("ShaderNodeTexBrick")
        br.offset = 0.0
        br.inputs["Scale"].default_value = 1.0
        br.inputs["Brick Width"].default_value = tw
        br.inputs["Row Height"].default_value = th
        br.inputs["Mortar Size"].default_value = 0.003
        br.inputs["Color1"].default_value = (*s["c"], 1)
        br.inputs["Color2"].default_value = (*c2, 1)
        br.inputs["Mortar"].default_value = (*mortar, 1)
        nt.links.new(uvmap.outputs[0], br.inputs["Vector"])
        nt.links.new(br.outputs["Color"], b.inputs["Base Color"])
        bump = N.new("ShaderNodeBump")
        bump.inputs["Strength"].default_value = 0.3
        bump.invert = True
        nt.links.new(br.outputs["Fac"], bump.inputs["Height"])
        nt.links.new(bump.outputs["Normal"], b.inputs["Normal"])
    if "tex" in s:
        tid, size = s["tex"]
        folder = ASSETS / "materials" / tid
        col = find_map(folder, "_Color") if folder.exists() else None
        if col:
            uvmap.inputs["Scale"].default_value = (1 / size, 1 / size, 1 / size)
            for key, inp, colour in (("_Color", "Base Color", True), ("_Roughness", "Roughness", False), ("_NormalGL", None, False)):
                f = find_map(folder, key)
                if not f:
                    continue
                im = N.new("ShaderNodeTexImage")
                im.image = bpy.data.images.load(str(f), check_existing=True)
                if not colour:
                    im.image.colorspace_settings.name = "Non-Color"
                nt.links.new(uvmap.outputs[0], im.inputs["Vector"])
                if inp:
                    nt.links.new(im.outputs["Color"], b.inputs[inp])
                else:
                    nm = N.new("ShaderNodeNormalMap")
                    nm.inputs["Strength"].default_value = 0.6
                    nt.links.new(im.outputs["Color"], nm.inputs["Color"])
                    nt.links.new(nm.outputs["Normal"], b.inputs["Normal"])
    _M[name] = m
    return m


# ---------- mesh helpers ----------
def uv_box(me, offset=Vector((0, 0, 0))):
    uv = me.uv_layers.new(name="UVMap")
    for p in me.polygons:
        n = p.normal
        ax = max(range(3), key=lambda i: abs(n[i]))
        for li in p.loop_indices:
            co = me.vertices[me.loops[li].vertex_index].co + offset
            uv.data[li].uv = (co.y, co.z) if ax == 0 else (co.x, co.z) if ax == 1 else (co.x, co.y)


def new_obj(name, me, mats, parent=None, loc=(0, 0, 0), rot=(0, 0, 0)):
    for m in mats:
        me.materials.append(m)
    ob = bpy.data.objects.new(name, me)
    COLL.objects.link(ob)
    ob.location, ob.rotation_euler = loc, rot
    if parent:
        ob.parent = parent
    return ob


def box(name, size, center, m, parent=None, bevel=0.0, rot_z=0.0):
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=1.0, matrix=Matrix.Diagonal((*size, 1)))
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    uv_box(me)
    ob = new_obj(name, me, [mat(m)], parent, center, (0, 0, rot_z))
    if bevel > 0:
        mod = ob.modifiers.new("bevel", "BEVEL")
        mod.width, mod.segments, mod.limit_method = bevel, 2, "ANGLE"
    return ob


def cyl(name, r, h, center, m, parent=None, axis="z"):
    bm = bmesh.new()
    bmesh.ops.create_cone(bm, cap_ends=True, cap_tris=False, segments=28, radius1=r, radius2=r, depth=h)
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    for p in me.polygons:
        p.use_smooth = len(p.vertices) == 4
    uv_box(me)
    rot = (math.pi / 2, 0, 0) if axis == "y" else (0, 0, 0)
    return new_obj(name, me, [mat(m)], parent, center, rot)


def empty(name, loc, rot_z=0.0):
    e = bpy.data.objects.new(name, None)
    COLL.objects.link(e)
    e.location, e.rotation_euler = loc, (0, 0, rot_z)
    return e


def wall_frame(w):
    (ax, ay), (bx, by) = w["a"], w["b"]
    L = math.hypot(bx - ax, by - ay)
    u = ((bx - ax) / L, (by - ay) / L)
    return (ax, ay), u, (-u[1], u[0]), L


def wall_corners(w):
    (ax, ay), u, n, L = wall_frame(w)
    t = w["thickness"] / 2
    return [(ax + u[0] * s + n[0] * k, ay + u[1] * s + n[1] * k) for s in (0.0, L) for k in (-t, t)]


def side_into(c, nrm, t, room_id):
    """+1 or -1: which side of the wall (along its normal) the given room is."""
    for s in (1, -1):
        r = room_at((c[0] + s * nrm[0] * (t / 2 + 0.15), c[1] + s * nrm[1] * (t / 2 + 0.15)))
        if r and r["id"] == room_id:
            return s
    return 1


# ---------- shell: walls with openings, floors, ceilings, slabs ----------
WET = ("bathroom", "wc")


def build_walls():
    openings = {}
    for o in PLAN["doors"]:
        openings.setdefault(o["wall_id"], []).append((o, 0.0, o["head"]))
    for o in PLAN["windows"]:
        openings.setdefault(o["wall_id"], []).append((o, o["sill"], o["head"]))
    mats = [mat("wall_paint"), mat("tile_wall"), mat("exterior")]
    for w in PLAN["walls"]:
        a, u, n, L = wall_frame(w)
        t, hgt = w["thickness"], w["height"]
        holes = []
        for o, z0, z1 in openings.get(w["id"], []):
            s_ = (o["center"][0] - a[0]) * u[0] + (o["center"][1] - a[1]) * u[1]
            lo, hi = max(0.0, s_ - o["width"] / 2), min(L, s_ + o["width"] / 2)
            if hi - lo > 0.01:
                holes.append((lo, hi, max(0.0, z0), min(hgt, z1)))
        bm = bmesh.new()
        wall_solid(bm, a, u, n, L, t, hgt, holes)
        me = bpy.data.meshes.new(w["id"])
        bm.to_mesh(me)
        bm.free()
        for p in me.polygons:   # paint, tiles or facade depending on what the face looks at
            c, nn = p.center, p.normal
            if abs(nn.z) > 0.5 or abs(nn.x * u[0] + nn.y * u[1]) > 0.5:   # tops, reveals and wall ends: paint
                continue
            r = room_at((c.x + nn.x * 0.06, c.y + nn.y * 0.06))
            p.material_index = 2 if (r is None or r["type"] == "balcony") else 1 if r["type"] in WET else 0
        uv_box(me)
        new_obj(w["id"], me, mats)


def _dedup(vals, eps=1e-3):
    out = []
    for v in sorted(vals):
        if not out or v - out[-1] > eps:
            out.append(v)
    return out


def wall_solid(bm, a, u, n, L, t, hgt, holes):
    """One watertight wall with real openings: the front face is a grid split at every opening edge, cells inside
    an opening are left out, the back is a mirrored copy and every boundary edge of the grid gets a side face
    (wall ends, top, bottom and the reveals of each opening). Every edge ends up with exactly two faces."""
    xs = _dedup([0.0, L] + [v for lo, hi, _, _ in holes for v in (lo, hi) if 0.0 < v < L])
    zs = _dedup([0.0, hgt] + [v for _, _, z0, z1 in holes for v in (z0, z1) if 0.0 < v < hgt])
    grid = {}

    def V(i, j):
        if (i, j) not in grid:
            s, z = xs[i], zs[j]
            grid[(i, j)] = bm.verts.new((a[0] + u[0] * s - n[0] * t / 2, a[1] + u[1] * s - n[1] * t / 2, z))
        return grid[(i, j)]

    front = []
    for i in range(len(xs) - 1):
        for j in range(len(zs) - 1):
            xm, zm = (xs[i] + xs[i + 1]) / 2, (zs[j] + zs[j + 1]) / 2
            if any(lo < xm < hi and z0 < zm < z1 for lo, hi, z0, z1 in holes):
                continue
            front.append(bm.faces.new((V(i, j), V(i + 1, j), V(i + 1, j + 1), V(i, j + 1))))
    if not front:
        return
    bnd = [tuple(e.verts) for e in {e for f in front for e in f.edges} if len(e.link_faces) == 1]
    back = {v: bm.verts.new(v.co + Vector((n[0] * t, n[1] * t, 0.0))) for v in grid.values()}
    for f in front:
        bm.faces.new([back[v] for v in reversed(f.verts)])
    for v1, v2 in bnd:
        bm.faces.new((v2, v1, back[v1], back[v2]))
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])


def poly_mesh(name, pts, z, m, down=False):
    verts = [(x, y, z) for x, y in pts]
    face = list(range(len(pts)))
    me = bpy.data.meshes.new(name)
    me.from_pydata(verts, [], [face[::-1] if down else face])
    me.update()
    uv_box(me)
    return new_obj(name, me, [mat(m)])


def build_floors():
    for r in ROOMS:
        fl = "tile_floor" if r["type"] in ("kitchen", "bathroom", "wc", "balcony", "storage") else "wood_floor"
        poly_mesh(f"floor_{r['id']}", r["polygon"], 0.0, fl)
        if r["type"] != "balcony":
            poly_mesh(f"ceiling_{r['id']}", r["polygon"], r["ceiling_height"], "ceiling", down=True)
    for d in PLAN["doors"] + [w for w in PLAN["windows"] if w["kind"] == "balcony_door"]:
        w = WALLS.get(d["wall_id"])
        if not w:
            continue
        _, u, n, _ = wall_frame(w)
        c, hw, ht = d["center"], d["width"] / 2, w["thickness"] / 2
        pts = [(c[0] + su * u[0] * hw + sn * n[0] * ht, c[1] + su * u[1] * hw + sn * n[1] * ht)
               for su, sn in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
        if (pts[1][0] - pts[0][0]) * (pts[2][1] - pts[0][1]) - (pts[1][1] - pts[0][1]) * (pts[2][0] - pts[0][0]) < 0:
            pts = pts[::-1]
        poly_mesh(f"threshold_{d['id']}", pts, 0.0, "sill")
    corners = [c for w in PLAN["walls"] for c in wall_corners(w)]   # slabs flush with the outer wall faces
    xs, ys = [c[0] for c in corners], [c[1] for c in corners]
    cx, cy, sx, sy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2, max(xs) - min(xs), max(ys) - min(ys)
    box("slab_bottom", (sx, sy, 0.3), (cx, cy, -0.155), "exterior")
    box("slab_top", (sx, sy, 0.3), (cx, cy, H + 0.155), "exterior")


def build_balcony_rails():
    for r in ROOMS:
        if r["type"] != "balcony":
            continue
        p = r["polygon"]
        for i in range(len(p)):
            (x1, y1), (x2, y2) = p[i], p[(i + 1) % len(p)]
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            L = math.hypot(x2 - x1, y2 - y1)
            near_wall = any(abs((mx - w["a"][0]) * -wall_frame(w)[1][1] + (my - w["a"][1]) * wall_frame(w)[1][0]) < w["thickness"] / 2 + 0.2
                            and -0.1 <= (mx - w["a"][0]) * wall_frame(w)[1][0] + (my - w["a"][1]) * wall_frame(w)[1][1] <= wall_frame(w)[3] + 0.1
                            for w in PLAN["walls"])
            if near_wall or L < 0.2:
                continue
            ang = math.atan2(y2 - y1, x2 - x1)
            box(f"rail_glass_{r['id']}_{i}", (L, 0.012, 0.95), (mx, my, 0.55), "glass", rot_z=ang)
            box(f"rail_top_{r['id']}_{i}", (L, 0.05, 0.04), (mx, my, 1.05), "black_metal", rot_z=ang)


# ---------- windows, doors, skirting ----------
def build_windows():
    for o in PLAN["windows"]:
        w = WALLS.get(o["wall_id"])
        if not w:
            continue
        a, u, n, _ = wall_frame(w)
        c, W, z0, z1, t = o["center"], o["width"], o["sill"], o["head"], w["thickness"]
        root = empty(f"win_{o['id']}", (c[0], c[1], 0), math.atan2(u[1], u[0]))
        hz = z1 - z0
        box("frame_l", (0.06, 0.07, hz), (-W / 2 + 0.03, 0, z0 + hz / 2), "frame", root)
        box("frame_r", (0.06, 0.07, hz), (W / 2 - 0.03, 0, z0 + hz / 2), "frame", root)
        box("frame_t", (W, 0.07, 0.06), (0, 0, z1 - 0.03), "frame", root)
        box("frame_b", (W, 0.07, 0.06), (0, 0, z0 + 0.03), "frame", root)
        if W > 1.1:
            box("mullion", (0.06, 0.07, hz), (0, 0, z0 + hz / 2), "frame", root)
        box("glass", (W - 0.1, 0.01, hz - 0.1), (0, 0, z0 + hz / 2), "glass", root)
        s = side_into(c, n, t, o["rooms"][0]) if o.get("rooms") else 1
        if z0 > 0.1:
            box("sill", (W + 0.08, t / 2 + 0.05, 0.03), (0, s * (t / 4 + 0.025), z0 - 0.015), "sill", root)
        ld = bpy.data.lights.new(f"portal_{o['id']}", "AREA")
        ld.shape, ld.size, ld.size_y = "RECTANGLE", W, hz
        ld.cycles.is_portal = True
        lo = bpy.data.objects.new(f"portal_{o['id']}", ld)
        COLL.objects.link(lo)
        lo.location = (c[0] - s * n[0] * t / 2, c[1] - s * n[1] * t / 2, z0 + hz / 2)
        lo.rotation_euler = Vector((s * n[0], s * n[1], 0)).to_track_quat("-Z", "Y").to_euler()


def build_doors():
    for d in PLAN["doors"]:
        w = WALLS.get(d["wall_id"])
        if not w or d["kind"] == "opening":
            continue
        a, u, n, _ = wall_frame(w)
        c, W, hd, t = d["center"], d["width"], d["head"], w["thickness"]
        root = empty(f"door_{d['id']}", (c[0], c[1], 0), math.atan2(u[1], u[0]))
        box("jamb_l", (0.04, t + 0.02, hd), (-W / 2 + 0.02, 0, hd / 2), "frame", root)
        box("jamb_r", (0.04, t + 0.02, hd), (W / 2 - 0.02, 0, hd / 2), "frame", root)
        box("head", (W, t + 0.02, 0.04), (0, 0, hd - 0.02), "frame", root)
        lw = W - 0.08
        into = d["swing"]["into"]
        s = side_into(c, n, t, into)
        hx = -W / 2 + 0.04 if d["swing"]["hinge"] == "a" else W / 2 - 0.04
        v = 1 if d["swing"]["hinge"] == "a" else -1
        entrance = len(d.get("rooms", [])) < 2
        ang = 0.0 if entrance else math.radians(82)
        pivot = empty(f"leaf_{d['id']}", (hx, s * (t / 2 - 0.03) if not entrance else 0, 0))
        pivot.parent = root
        pivot.rotation_euler.z = (math.pi if v < 0 else 0) + (v * s * ang)
        box("leaf", (lw, 0.04, hd - 0.01), (lw / 2, 0, (hd - 0.01) / 2), "door", pivot, bevel=0.003)
        for sy in (1, -1):
            box(f"handle{sy}", (0.12, 0.02, 0.02), (lw - 0.08, sy * 0.035, 1.0), "black_metal", pivot)


def build_skirting():
    for r in ROOMS:
        if r["type"] in ("balcony",) or r["type"] in WET:
            continue
        p = r["polygon"]
        gaps = []
        for o in PLAN["doors"] + [x for x in PLAN["windows"] if x["kind"] == "balcony_door"]:
            if r["id"] in o.get("rooms", []):
                gaps.append(o)
        for i in range(len(p)):
            (x1, y1), (x2, y2) = p[i], p[(i + 1) % len(p)]
            L = math.hypot(x2 - x1, y2 - y1)
            if L < 0.1:
                continue
            ux, uy = (x2 - x1) / L, (y2 - y1) / L
            cut = []
            for o in gaps:
                along = (o["center"][0] - x1) * ux + (o["center"][1] - y1) * uy
                off = abs((o["center"][0] - x1) * -uy + (o["center"][1] - y1) * ux)
                if off < 0.25 and -0.5 < along < L + 0.5:
                    cut.append((along - o["width"] / 2, along + o["width"] / 2))
            pos = 0.0
            for lo, hi in sorted(cut) + [(L, L)]:
                if lo - pos > 0.05:
                    m = (pos + min(lo, L)) / 2
                    box(f"skirt_{r['id']}_{i}", (min(lo, L) - pos, 0.012, 0.08),
                        (x1 + ux * m - uy * 0.006, y1 + uy * m + ux * 0.006, 0.04), "white_matte", rot_z=math.atan2(uy, ux))
                pos = max(pos, hi)


# ---------- furniture (parametric, exact sizes; library models only for plants/lamps) ----------
def B(size, c, m, bev=0.0):
    return ("box", size, c, m, bev)


def C(r, h, c, m, axis="z"):
    return ("cyl", r, h, c, m, axis)


def legs(w, d, h, r=0.02, m="black_metal", inset=0.06):
    return [C(r, h, (sx * (w / 2 - inset), sy * (d / 2 - inset), h / 2), m) for sx in (1, -1) for sy in (1, -1)]


def parts(t, w, d, h, prm):
    P = []
    if t in ("sofa", "armchair"):
        n = 1 if t == "armchair" else (3 if w >= 1.8 else 2)
        P += legs(w, d, 0.08)
        P.append(B((w, d - 0.04, 0.26), (0, 0, 0.21), "fabric_sofa", 0.02))
        seat_w = (w - 0.32) / n
        for k in range(n):
            x = -w / 2 + 0.16 + seat_w * (k + 0.5)
            P.append(B((seat_w - 0.01, d - 0.28, 0.14), (x, 0.1, 0.41), "fabric_sofa", 0.035))
            P.append(B((seat_w - 0.01, 0.2, 0.38), (x, -d / 2 + 0.24, 0.58), "fabric_sofa", 0.05))
        P.append(B((w, 0.16, h - 0.08), (0, -d / 2 + 0.08, 0.08 + (h - 0.08) / 2), "fabric_sofa", 0.03))
        for sx in (1, -1):
            P.append(B((0.16, d - 0.04, 0.62), (sx * (w / 2 - 0.08), 0, 0.39), "fabric_sofa", 0.03))
    elif t == "coffee_table":
        P += [B((w, d, 0.04), (0, 0, h - 0.02), "wood", 0.004)] + legs(w, d, h - 0.04, 0.018)
    elif t == "tv_unit":
        P.append(B((w, d, 0.45), (0, 0, 0.26), "wood", 0.004))
        tw = min(w * 0.8, 1.45)
        P.append(B((0.25, 0.18, 0.04), (0, 0, 0.505), "black_metal"))
        P.append(B((tw, 0.04, tw * 9 / 16), (0, -0.02, 0.55 + tw * 9 / 32), "screen", 0.004))
    elif t == "rug":
        P.append(B((w, d, 0.012), (0, 0, 0.006), "rug"))
    elif t == "floor_lamp":
        P += [C(0.15, 0.02, (0, 0, 0.01), "black_metal"), C(0.012, 1.40, (0, 0, 0.72), "black_metal"),
              C(0.20, 0.28, (0, 0, h - 0.14), "fabric_bed")]
    elif t == "plant":
        P += [C(0.18, 0.36, (0, 0, 0.18), "pot")]
    elif t in ("dining_table", "bistro_table", "desk"):
        if t == "bistro_table":
            P += [C(w / 2, 0.03, (0, 0, h - 0.015), "wood"), C(0.03, h - 0.03, (0, 0, (h - 0.03) / 2), "black_metal"),
                  C(0.22, 0.02, (0, 0, 0.01), "black_metal")]
        else:
            P += [B((w, d, 0.035), (0, 0, h - 0.0175), "wood", 0.003)] + legs(w, d, h - 0.035, 0.022, "black_metal", 0.05)
    elif t == "chair":
        P += [B((w - 0.02, d - 0.08, 0.04), (0, 0.02, 0.45), "wood", 0.004), B((w - 0.02, 0.03, 0.38), (0, -d / 2 + 0.03, 0.66), "wood", 0.004)]
        P += legs(w - 0.02, d - 0.08, 0.43, 0.012)
    elif t in ("bed_double", "bed_single"):
        P.append(B((w + 0.06, 0.08, h), (0, -d / 2 + 0.04, h / 2), "fabric_accent", 0.02))
        P.append(B((w + 0.04, d - 0.08, 0.30), (0, 0.04, 0.20), "fabric_accent", 0.015))
        P.append(B((w, d - 0.12, 0.22), (0, 0.04, 0.46), "fabric_bed", 0.04))
        P.append(B((w + 0.04, (d - 0.1) * 0.62, 0.06), (0, d / 2 - (d - 0.1) * 0.31 - 0.01, 0.59), "fabric_accent", 0.03))
        k = 2 if t == "bed_double" else 1
        for i in range(k):
            x = 0 if k == 1 else (i - 0.5) * w / 2
            P.append(B((w / k - 0.14, 0.38, 0.14), (x, -d / 2 + 0.33, 0.64), "fabric_bed", 0.05))
    elif t == "nightstand":
        P += [B((w, d, 0.48), (0, 0, 0.26), "wood", 0.004), C(0.06, 0.05, (0, 0, 0.525), "ceramic"),
              C(0.11, 0.18, (0, 0, 0.66), "fabric_bed")]
    elif t in ("wardrobe", "shoe_cabinet", "bookshelf"):
        P.append(B((w, d - 0.02, h), (0, -0.01, h / 2), "wood" if t != "wardrobe" else "white_matte", 0.003))
        if t == "bookshelf":
            P += [B((w - 0.04, d - 0.04, 0.02), (0, 0.02, z), "white_matte") for z in (0.4, 0.8, 1.2, 1.6)]
        else:
            k = max(2, round(w / 0.5))
            for i in range(k):
                x = -w / 2 + (i + 0.5) * w / k
                P.append(B((w / k - 0.004, 0.02, h - 0.03), (x, d / 2 - 0.01, h / 2), "white_matte" if t == "wardrobe" else "wood", 0.002))
                P.append(B((0.012, 0.02, 0.3 if t == "wardrobe" else 0.12), (x + (0.5 if i % 2 == 0 else -0.5) * (w / k - 0.08), d / 2 + 0.01, 1.05 if t == "wardrobe" else h - 0.12), "black_metal"))
    elif t == "kitchen_run":
        P.append(B((w, d - 0.08, 0.10), (0, -0.04, 0.05), "black_metal"))
        k = max(1, round(w / 0.6))
        for i in range(k):
            x = -w / 2 + (i + 0.5) * w / k
            P.append(B((w / k - 0.004, d - 0.02, 0.76), (x, -0.01, 0.48), "white_matte", 0.002))
            P.append(B((0.3, 0.02, 0.012), (x, d / 2 - 0.0, 0.8), "black_metal"))
        P.append(B((w + 0.02, d + 0.02, 0.04), (0, 0.0, 0.88), "stone_counter", 0.003))
        if prm.get("sink_x") is not None:
            sx = prm["sink_x"]
            P += [B((0.52, 0.42, 0.006), (sx, 0.02, 0.903), "steel"), C(0.015, 0.28, (sx, -d / 2 + 0.08, 1.04), "steel"),
                  B((0.02, 0.18, 0.02), (sx, -d / 2 + 0.17, 1.17), "steel")]
        if prm.get("hob_x") is not None:
            P.append(B((0.58, 0.50, 0.006), (prm["hob_x"], 0.0, 0.903), "screen"))
        for x0, x1 in prm.get("upper", []):
            P.append(B((x1 - x0 - 0.004, 0.34, 0.70), ((x0 + x1) / 2, -d / 2 + 0.17, 1.80), "white_matte", 0.002))
    elif t == "fridge":
        P += [B((w, d, h), (0, 0, h / 2), "steel", 0.01), B((0.02, 0.03, 0.5), (w / 2 - 0.06, d / 2 + 0.015, 1.2), "black_metal")]
    elif t == "toilet":
        P += [B((0.36, 0.50, 0.40), (0, 0.07, 0.20), "ceramic", 0.06), B((0.38, 0.17, 0.40), (0, -d / 2 + 0.085, 0.60), "ceramic", 0.03),
              B((0.37, 0.45, 0.03), (0, 0.08, 0.415), "ceramic", 0.012)]
    elif t in ("vanity", "basin_small"):
        P += [B((w, d, 0.45), (0, 0, 0.58), "wood", 0.004), B((w, d, 0.05), (0, 0, 0.83), "ceramic", 0.01),
              C(0.015, 0.2, (0, -d / 2 + 0.07, 0.95), "steel"), B((w * 0.9, 0.02, 0.75), (0, -d / 2 + 0.01, 1.55), "mirror")]
    elif t == "shower":
        P += [B((w, d, 0.05), (0, 0, 0.025), "ceramic", 0.005), B((w - 0.02, 0.008, 1.95), (0, d / 2 - 0.004, 1.025), "glass"),
              C(0.10, 0.02, (0, -d / 2 + 0.25, 2.0), "steel"), B((0.02, 0.25, 0.02), (0, -d / 2 + 0.125, 2.0), "steel")]
    elif t == "bathtub":
        P += [B((w, d, 0.05), (0, 0, 0.025), "ceramic")]
        P += [B((w, 0.06, 0.55), (0, sy * (d / 2 - 0.03), 0.3), "ceramic", 0.01) for sy in (1, -1)]
        P += [B((0.06, d - 0.12, 0.55), (sx * (w / 2 - 0.03), 0, 0.3), "ceramic", 0.01) for sx in (1, -1)]
        P.append(C(0.015, 0.25, (-w / 2 + 0.03, -d / 2 + 0.03, 0.7), "steel"))
    elif t == "washer":
        P += [B((w, d, h), (0, 0, h / 2), "white_matte", 0.02), C(0.2, 0.02, (0, d / 2 + 0.01, 0.45), "screen", "y")]
    return P


def library_model(t, w, d, h, loc, rot):
    """Use a downloaded CC0 model for plants and lamps if its proportions fit."""
    folder = ASSETS / "models" / t
    for f in sorted(folder.glob("*/*.gltf")) + sorted(folder.glob("*/*.glb")) if folder.exists() else []:
        before = set(bpy.data.objects)
        try:
            bpy.ops.import_scene.gltf(filepath=str(f))
        except Exception as e:
            WARN.append(f"model import failed {f.name}: {e}")
            continue
        new = [o for o in bpy.data.objects if o not in before]
        meshes = [o for o in new if o.type == "MESH"]
        bpy.context.view_layer.update()
        if meshes:
            pts = [o.matrix_world @ Vector(c) for o in meshes for c in o.bound_box]
            mn = Vector([min(p[i] for p in pts) for i in range(3)])
            mx = Vector([max(p[i] for p in pts) for i in range(3)])
            dim = mx - mn
            s = h / max(dim.z, 1e-6)
            if dim.x * s <= w * 1.8 and dim.y * s <= d * 1.8:
                holder = empty(f"lib_{t}", loc, rot)
                fit = empty(f"fit_{t}", (0, 0, 0))
                fit.parent = holder
                fit.scale = (s, s, s)
                fit.location = (-(mn.x + mx.x) / 2 * s, -(mn.y + mx.y) / 2 * s, -mn.z * s)
                for o in new:
                    if o.parent is None:
                        o.parent = fit
                return holder
        for o in new:
            bpy.data.objects.remove(o, do_unlink=True)
    return None


def merge_meshes(name, objs, inv):
    """One mesh from several objects (modifiers applied, transforms baked relative to inv, materials merged)."""
    dg = bpy.context.evaluated_depsgraph_get()
    bm, mats = bmesh.new(), []
    for o in objs:
        tmp = bpy.data.meshes.new_from_object(o.evaluated_get(dg), preserve_all_data_layers=True, depsgraph=dg)
        M = inv @ o.matrix_world
        tmp.transform(M)
        while len(tmp.uv_layers) > 1:
            tmp.uv_layers.remove(tmp.uv_layers[-1])
        if not tmp.uv_layers:
            tmp.uv_layers.new(name="UVMap")
        tmp.uv_layers[0].name = "UVMap"
        remap = []
        for m in (list(tmp.materials) or [None]):
            if m not in mats:
                mats.append(m)
            remap.append(mats.index(m))
        for p in tmp.polygons:
            p.material_index = remap[min(p.material_index, len(remap) - 1)]
        if M.determinant() < 0:   # mirrored part: turn the faces the right way out
            b2 = bmesh.new()
            b2.from_mesh(tmp)
            bmesh.ops.reverse_faces(b2, faces=b2.faces[:])
            b2.to_mesh(tmp)
            b2.free()
        bm.from_mesh(tmp)
        bpy.data.meshes.remove(tmp)
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    for m in mats:
        me.materials.append(m)
    return me


def mesh_bounds(me):
    if not len(me.vertices):
        return Vector((0, 0, 0)), Vector((0, 0, 0))
    xs, ys, zs = zip(*(v.co[:] for v in me.vertices))
    return Vector((min(xs), min(ys), min(zs))), Vector((max(xs), max(ys), max(zs)))


def textures_ok(me):
    """True when every image used by the mesh's materials is packed or on disk."""
    for m in me.materials:
        if m is None or not m.use_nodes:
            continue
        for nd in m.node_tree.nodes:
            img = getattr(nd, "image", None)
            if img is None or img.source in ("GENERATED", "VIEWER") or img.packed_file:
                continue
            if not (img.filepath and os.path.exists(bpy.path.abspath(img.filepath))):
                return False
    return True


def import_asset(it, a, limits):
    """Catalog model -> one object normalised to the layout slot, or (None, reason) for the parametric fallback."""
    w, d, h = it["size"]
    before = set(bpy.data.objects)
    try:
        bpy.ops.import_scene.gltf(filepath=a["file"])
    except Exception as e:
        for o in [o for o in bpy.data.objects if o not in before]:
            bpy.data.objects.remove(o, do_unlink=True)
        return None, f"import failed: {e}"
    new = [o for o in bpy.data.objects if o not in before]
    try:
        meshes = [o for o in new if o.type == "MESH"]
        if not meshes:
            return None, "no mesh in the file"
        bpy.context.view_layer.update()
        me = merge_meshes(it["id"], meshes, Matrix.Identity(4))
    finally:
        for o in new:
            bpy.data.objects.remove(o, do_unlink=True)
    turn = {"-Y": math.pi, "+Y": 0.0, "+X": math.pi / 2, "-X": -math.pi / 2}.get(a.get("front_axis", "-Y"), math.pi)
    me.transform(Matrix.Rotation(turn, 4, "Z") @ Matrix.Scale(float(a.get("unit_scale", 1.0)), 4))
    lo, hi = mesh_bounds(me)
    me.transform(Matrix.Translation((-(lo.x + hi.x) / 2, -(lo.y + hi.y) / 2, -lo.z)))   # origin at base centre
    dims = hi - lo
    if min(dims) <= 1e-4:
        return _reject(me, f"flat or empty model {tuple(round(x, 3) for x in dims)}")
    if a.get("fit") == "height":   # same rule as furniture.fit(): slot height, smaller if too wide
        s = min(h / dims.z, 1.8 * w / dims.x, 1.8 * d / dims.y)
        if dims.z * s < 0.5 * h or not 0.2 <= s <= 3.0:
            return _reject(me, "too wide for the slot even at half its height")
        sc = (s, s, s)
    else:
        sc = (w / dims.x, d / dims.y, h / dims.z)
        su = statistics.median(sc)
        worst = max(abs(x / su - 1) for x in sc)
        if abs(su - 1) > limits["max_uniform_change"] or worst > limits["max_nonuniform"]:
            return _reject(me, f"slot fit: size x{su:.2f}, non-uniform {100 * worst:.0f} %")
    me.transform(Matrix.Diagonal((*sc, 1.0)))
    ob = bpy.data.objects.new(it["id"], me)
    COLL.objects.link(ob)
    tris0 = sum(len(p.vertices) - 2 for p in me.polygons)
    cap = int(a.get("poly_cap", 60000))
    if tris0 > cap:   # polygon cap: collapse decimation, applied into the mesh
        mod = ob.modifiers.new("decimate", "DECIMATE")
        mod.ratio = max(0.01, cap / tris0)
        dg = bpy.context.evaluated_depsgraph_get()
        dec = bpy.data.meshes.new_from_object(ob.evaluated_get(dg), preserve_all_data_layers=True, depsgraph=dg)
        ob.modifiers.remove(mod)
        old, ob.data = ob.data, dec
        bpy.data.meshes.remove(old)
        dec.name = it["id"]
    me = ob.data
    tris = sum(len(p.vertices) - 2 for p in me.polygons)
    lo, hi = mesh_bounds(me)
    dims = hi - lo
    bad = []
    if tris > cap * 1.05:
        bad.append(f"{tris} triangles > cap {cap}")
    if not all(math.isfinite(c) for v in me.vertices for c in v.co):
        bad.append("non-finite vertex")
    if a.get("fit") != "height" and max(abs(dims.x - w) / w, abs(dims.y - d) / d, abs(dims.z - h) / h) > 0.02:
        bad.append(f"size after fit {tuple(round(x, 3) for x in dims)} != slot {(w, d, h)}")
    if not textures_ok(me):
        bad.append("missing textures")
    if bad:
        bpy.data.objects.remove(ob, do_unlink=True)
        bpy.data.meshes.remove(me)
        return None, "QA: " + "; ".join(bad)
    ob.location = (it["center"][0], it["center"][1], it.get("z", 0.0))
    ob.rotation_euler = (0, 0, math.radians(it["rot_deg"]))
    ob["asset_id"], ob["licence"], ob["source"] = a["asset_id"], a.get("licence", ""), a["source"]
    return ob, {"triangles_in": tris0, "triangles": tris, "scale": [round(x, 4) for x in sc]}


def _reject(me, reason):
    bpy.data.meshes.remove(me)
    return None, reason


def parametric_item(it):
    """The built-in model of an item as ONE mesh object named after the item, origin at its base centre."""
    w, d, h = it["size"]
    loc, rot = (it["center"][0], it["center"][1], it.get("z", 0.0)), math.radians(it["rot_deg"])
    root = empty(it["id"] + "__parts", loc, rot)
    for k, p in enumerate(parts(it["type"], w, d, h, it.get("params") or {})):
        if p[0] == "box":
            box(f"{it['id']}_{k}", p[1], p[2], p[3], root, p[4])
        else:
            cyl(f"{it['id']}_{k}", p[1], p[2], p[3], p[4], root, p[5])
    bpy.context.view_layer.update()
    kids = [o for o in root.children_recursive if o.type == "MESH"]
    me = merge_meshes(it["id"], kids, root.matrix_world.inverted())
    for o in kids + [root]:
        bpy.data.objects.remove(o, do_unlink=True)
    ob = bpy.data.objects.new(it["id"], me)
    COLL.objects.link(ob)
    ob.location, ob.rotation_euler = loc, (0, 0, rot)
    ob["source"] = "parametric"
    return ob


def build_furniture():
    qa = []
    items = CHOICE["items"] if CHOICE else {}
    limits = (CHOICE or {}).get("limits", {"max_uniform_change": 0.35, "max_nonuniform": 0.15})
    for it in LAYOUT["items"]:
        w, d, h = it["size"]
        loc, rot = (it["center"][0], it["center"][1], it.get("z", 0.0)), math.radians(it["rot_deg"])
        a = items.get(it["id"], {})
        rec = {"item_id": it["id"], "type": it["type"], "room": it["room"], "planned": a.get("source", "parametric"),
               "asset_id": a.get("asset_id"), "reason": a.get("reason", "")}
        if a.get("source") == "removed":
            qa.append(dict(rec, used="removed"))
            continue
        if CHOICE is None and it["type"] in ("plant", "floor_lamp"):   # no assets.json: original behaviour
            try:
                holder = library_model(it["type"], w, d, h, loc, rot)
                if holder:
                    holder.name = it["id"]
                    qa.append(dict(rec, used="polyhaven", planned="polyhaven", reason="legacy library model"))
                    continue
            except Exception as e:
                WARN.append(f"library model failed for {it['type']}: {e}")
        elif a.get("file"):
            try:
                ob, info = import_asset(it, a, limits)
            except Exception as e:
                ob, info = None, f"import error: {e}"
            if ob:
                qa.append(dict(rec, used=a["source"], reason="", **info))
                continue
            rec["reason"] = info
            WARN.append(f"{it['id']}: model {a['asset_id']} rejected ({info}) - parametric used")
        if it["type"] == "plant":
            WARN.append("plant skipped: no plant model downloaded" if CHOICE is None
                        else f"plant {it['id']} skipped: {rec['reason'] or 'no fitting model'}")
            qa.append(dict(rec, used="skipped"))
            continue
        parametric_item(it)
        qa.append(dict(rec, used="parametric"))
    summ = {}
    for r in qa:
        summ.setdefault(r["type"], {}).setdefault(r["used"], 0)
        summ[r["type"]][r["used"]] += 1
    json.dump({"items": qa, "summary": summ}, open(OUT / "furniture_qa.json", "w"), indent=1, ensure_ascii=False)


# ---------- light: sky/HDRI + matching sun + window portals + soft ceiling fill ----------
def build_lights():
    wins = [(o["width"] * (o["head"] - o["sill"]), o) for o in PLAN["windows"] if o.get("rooms")]
    az_out = 0.0
    if wins:
        living = [x for x in wins if room_type(x[1]["rooms"][0]) == "living"] or wins
        o = max(living, key=lambda x: x[0])[1]
        w = WALLS[o["wall_id"]]
        _, u, n, _ = wall_frame(w)
        s = side_into(o["center"], n, w["thickness"], o["rooms"][0])
        az_out = math.degrees(math.atan2(-s * n[1], -s * n[0]))
    az = az_out + 25.0
    el = 32.0
    world = bpy.data.worlds.new("World")
    SC.world = world
    world.use_nodes = True
    nt = world.node_tree
    bg = nt.nodes.get("Background")
    hdr = sorted((ASSETS / "hdri").glob("*.hdr")) if (ASSETS / "hdri").exists() else []
    if hdr:
        side = json.loads(hdr[0].with_suffix(".json").read_text()) if hdr[0].with_suffix(".json").exists() else {}
        el = min(55.0, max(18.0, side.get("sun_el_deg", el)))
        env, tc, mp = nt.nodes.new("ShaderNodeTexEnvironment"), nt.nodes.new("ShaderNodeTexCoord"), nt.nodes.new("ShaderNodeMapping")
        env.image = bpy.data.images.load(str(hdr[0]))
        mp.inputs["Rotation"].default_value[2] = math.radians(side.get("sun_az_deg", 0.0) - az)
        nt.links.new(tc.outputs["Generated"], mp.inputs["Vector"])
        nt.links.new(mp.outputs[0], env.inputs["Vector"])
        nt.links.new(env.outputs["Color"], bg.inputs["Color"])
        bg.inputs["Strength"].default_value = 1.0
    else:
        bg.inputs["Color"].default_value = (0.55, 0.66, 0.85, 1)
        bg.inputs["Strength"].default_value = 1.2
    sd = bpy.data.lights.new("sun", "SUN")
    sd.energy, sd.angle, sd.color = RC.get("sun_strength", 3.5), math.radians(1.0), (1.0, 0.96, 0.9)
    so = bpy.data.objects.new("sun", sd)
    COLL.objects.link(so)
    vec = Vector((math.cos(math.radians(el)) * math.cos(math.radians(az)),
                  math.cos(math.radians(el)) * math.sin(math.radians(az)), math.sin(math.radians(el))))
    so.rotation_euler = (-vec).to_track_quat("-Z", "Y").to_euler()
    for r in ROOMS:
        if r["type"] == "balcony":
            continue
        c = centroid(r["polygon"])
        ld = bpy.data.lights.new(f"fill_{r['id']}", "AREA")
        ld.shape, ld.size, ld.energy, ld.color = "DISK", 0.6, RC.get("fill_w_per_m2", 5.0) * r["area_m2"], (1.0, 0.92, 0.82)
        lo = bpy.data.objects.new(f"fill_{r['id']}", ld)
        COLL.objects.link(lo)
        lo.location = (c[0], c[1], r["ceiling_height"] - 0.05)
        lo.visible_camera = False
    return az, el


def room_type(rid):
    return next((r["type"] for r in ROOMS if r["id"] == rid), None)


# ---------- cameras: corners and wall middles, scored by how much furniture they see ----------
def footprints():
    fps = []
    for it in LAYOUT["items"]:
        if it["type"] == "rug":
            continue
        w, d, _ = it["size"]
        a = math.radians(it["rot_deg"])
        u, n = (math.cos(a), math.sin(a)), (-math.sin(a), math.cos(a))
        cx, cy = it["center"]
        fps.append((it["room"], [(cx + su * u[0] * (w / 2 + 0.2) + sn * n[0] * (d / 2 + 0.2),
                                  cy + su * u[1] * (w / 2 + 0.2) + sn * n[1] * (d / 2 + 0.2))
                                 for su, sn in ((-1, -1), (1, -1), (1, 1), (-1, 1))]))
    return fps


def build_cameras(views_main, views_other):
    bpy.context.view_layer.update()
    dg = bpy.context.evaluated_depsgraph_get()
    fps = footprints()
    cams = []
    for r in ROOMS:
        if r["type"] in ("balcony", "storage") or r["area_m2"] < 1.8:
            continue
        nv = views_main if r["type"] in ("living", "bedroom", "kitchen", "bathroom") else views_other
        p = r["polygon"]
        cen = centroid(p)
        zc = 1.30
        cands = []
        for i in range(len(p)):
            a, b = unit2((p[i - 1][0] - p[i][0], p[i - 1][1] - p[i][1])), unit2((p[(i + 1) % len(p)][0] - p[i][0], p[(i + 1) % len(p)][1] - p[i][1]))
            bis = unit2((a[0] + b[0], a[1] + b[1]))
            pos = (p[i][0] + bis[0] * 0.35, p[i][1] + bis[1] * 0.35)
            if pip(pos, p):
                cands.append((pos, math.atan2(cen[1] - pos[1], cen[0] - pos[0])))
        edges = sorted(range(len(p)), key=lambda i: -math.dist(p[i], p[(i + 1) % len(p)]))[:2]
        for i in edges:
            (x1, y1), (x2, y2) = p[i], p[(i + 1) % len(p)]
            u = unit2((x2 - x1, y2 - y1))
            pos = ((x1 + x2) / 2 - u[1] * 0.3, (y1 + y2) / 2 + u[0] * 0.3)
            if pip(pos, p):
                cands.append((pos, math.atan2(u[0], -u[1])))
        small = min(r["size_m"]) < 2.6
        hfov = 2 * math.atan(18 / (16 if small else 18))
        items = [it for it in LAYOUT["items"] if it["room"] == r["id"]]
        scored = []
        for pos, yaw in cands:
            if any(pip(pos, fp) for rid, fp in fps if rid == r["id"]):
                continue
            o = Vector((pos[0], pos[1], zc))
            sc = 0.0
            for it in items:
                tgt = Vector((it["center"][0], it["center"][1], max(0.3, it["size"][2] / 2)))
                v = tgt - o
                da = (math.atan2(v.y, v.x) - yaw + math.pi) % (2 * math.pi) - math.pi
                if abs(da) > hfov / 2 * 0.9:
                    continue
                hit, loc, *_ = SC.ray_cast(dg, o, v.normalized(), distance=v.length)
                if not hit or (loc - o).length > v.length - 0.6:
                    sc += 1 + it["size"][0] * it["size"][1]
            far = max(math.dist(pos, q) for q in p)
            scored.append((sc + 0.3 * far, pos, yaw))
        scored.sort(key=lambda x: -x[0])
        chosen = []
        for sc, pos, yaw in scored:
            if all(abs((yaw - y2 + math.pi) % (2 * math.pi) - math.pi) > math.radians(50) or math.dist(pos, p2) > 1.5 for _, p2, y2 in chosen):
                chosen.append((sc, pos, yaw))
            if len(chosen) >= nv:
                break
        for k, (sc, pos, yaw) in enumerate(chosen):
            cd = bpy.data.cameras.new(f"cam_{r['id']}_{k + 1}")
            cd.lens, cd.sensor_width, cd.clip_start, cd.clip_end = (16 if small else 18), 36, 0.05, 200
            co = bpy.data.objects.new(cd.name, cd)
            COLL.objects.link(co)
            co.location = (pos[0], pos[1], zc)
            co.rotation_euler = (math.pi / 2, 0, yaw - math.pi / 2)
            cams.append({"name": cd.name, "room": r["id"], "room_type": r["type"], "label": r.get("label", ""),
                         "pos": [round(pos[0], 3), round(pos[1], 3), zc], "yaw_deg": round(math.degrees(yaw), 1), "lens": cd.lens})
        if not chosen:
            WARN.append(f"no free camera position in room {r['id']}")
    return cams


# ---------- collections (deliverable structure) ----------
def organize_collections():
    labels = [r.get("label") or r["type"] for r in ROOMS]
    fname = {}
    for r, base in zip(ROOMS, labels):
        fname[r["id"]] = "Furniture." + (base if labels.count(base) == 1 else f"{base} {r['id']}")
    item_room = {it["id"]: it["room"] for it in LAYOUT["items"]}
    colls = {}

    def coll(name):
        if name not in colls:
            c = bpy.data.collections.new(name)
            SC.collection.children.link(c)
            colls[name] = c
        return colls[name]

    for base in ("Walls", "Floors_Ceilings", "Openings"):
        coll(base)
    for rid in sorted(set(item_room.values()), key=lambda x: int(x[1:]) if x[1:].isdigit() else 0):
        coll(fname.get(rid, "Furniture.other"))
    for base in ("Lights", "Cameras"):
        coll(base)
    for ob in list(SC.collection.objects):
        if ob.parent is not None:
            continue
        n = ob.name
        if n in item_room:
            target = fname.get(item_room[n], "Furniture.other")
        elif re.fullmatch(r"w\d+", n) or n.startswith(("skirt_", "rail_")):
            target = "Walls"
        elif n.startswith(("floor_", "ceiling_", "slab_", "threshold_")):
            target = "Floors_Ceilings"
        elif n.startswith(("win_", "door_")):
            target = "Openings"
        elif ob.type == "LIGHT":
            target = "Lights"
        elif ob.type == "CAMERA":
            target = "Cameras"
        else:
            target = "Misc"
        c = coll(target)
        for o in [ob] + list(ob.children_recursive):
            for uc in list(o.users_collection):
                uc.objects.unlink(o)
            c.objects.link(o)


# ---------- main ----------
prof = RC["profile_settings"]
build_walls()
build_floors()
build_balcony_rails()
build_windows()
build_doors()
build_skirting()
build_furniture()
az, el = build_lights()
cams = build_cameras(prof["views_main"], prof["views_other"])
organize_collections()
SC.render.resolution_x, SC.render.resolution_y = prof["width"], prof["height"]
try:
    SC.view_settings.view_transform = "AgX"
except TypeError:
    pass
SC.view_settings.exposure = RC.get("exposure", 0.0)
json.dump({"cameras": cams, "sun": {"azimuth_deg": az, "elevation_deg": el}, "warnings": WARN},
          open(OUT / "cameras.json", "w"), indent=1, ensure_ascii=False)
bpy.ops.wm.save_as_mainfile(filepath=str(OUT / "scene.blend"))
try:
    bpy.ops.export_scene.gltf(filepath=str(OUT / "scene.glb"), export_format="GLB", export_apply=True, export_cameras=True)
except TypeError:
    bpy.ops.export_scene.gltf(filepath=str(OUT / "scene.glb"), export_format="GLB")
print(f"SCENE_OK objects={len(bpy.data.objects)} cameras={len(cams)} warnings={len(WARN)}")
