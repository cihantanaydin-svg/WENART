"""Stage 3 - rule-based furniture layout -> 03_layout/layout.json + layout.png.
Items keep door swings and walkways clear, tall items stay away from windows,
nothing overlaps or leaves its room. Sizes shrink step by step when space is tight."""
import math
from pathlib import Path
import numpy as np
from shapely.geometry import Polygon, Point, LineString
from .common import jload, jsave, log, font_path

# footprint width (along wall) x depth x height in metres, biggest first
SIZES = {
    "sofa": [(2.10, 0.90, 0.85), (1.80, 0.88, 0.85), (1.50, 0.85, 0.85)],
    "armchair": [(0.80, 0.80, 0.80)], "coffee_table": [(1.10, 0.60, 0.42), (0.90, 0.50, 0.42)],
    "tv_unit": [(1.60, 0.40, 1.30), (1.20, 0.40, 1.20)], "rug": [(2.40, 1.70, 0.01), (2.00, 1.40, 0.01), (1.60, 1.10, 0.01)],
    "floor_lamp": [(0.35, 0.35, 1.60)], "plant": [(0.45, 0.45, 1.10)],
    "dining_table": [(1.60, 0.90, 0.75), (1.40, 0.85, 0.75), (1.20, 0.80, 0.75)], "chair": [(0.45, 0.50, 0.85)],
    "bed_double": [(1.60, 2.10, 1.05), (1.40, 2.10, 1.05)], "bed_single": [(0.90, 2.05, 1.00)],
    "nightstand": [(0.45, 0.40, 0.50)], "wardrobe": [(w, 0.60, 2.20) for w in (2.40, 2.00, 1.80, 1.60, 1.20, 1.00)],
    "desk": [(1.20, 0.60, 0.75), (1.00, 0.55, 0.75)], "fridge": [(0.60, 0.65, 1.90)],
    "toilet": [(0.40, 0.65, 0.80)], "vanity": [(0.80, 0.48, 0.85), (0.60, 0.45, 0.85)],
    "shower": [(0.90, 0.90, 2.00), (0.80, 0.80, 2.00)], "bathtub": [(1.70, 0.75, 0.58), (1.60, 0.70, 0.58)],
    "washer": [(0.60, 0.60, 0.85)], "basin_small": [(0.45, 0.35, 0.85)],
    "shoe_cabinet": [(1.00, 0.35, 1.00), (0.80, 0.35, 1.00), (0.60, 0.35, 1.00)],
    "bistro_table": [(0.60, 0.60, 0.72)], "bookshelf": [(0.80, 0.35, 1.80)],
}


def orect(c, u, n, w, d):
    c, u, n = np.asarray(c), np.asarray(u), np.asarray(n)
    return Polygon([tuple(c + su * u * w / 2 + sn * n * d / 2) for su, sn in ((-1, -1), (1, -1), (1, 1), (-1, 1))])


def unit(v):
    v = np.asarray(v, float)
    return v / (np.linalg.norm(v) or 1)


class Room:
    def __init__(self, r, plan, walls):
        self.r, self.id, self.type = r, r["id"], r["type"]
        self.poly = Polygon(r["polygon"])
        self.area = self.poly.area
        self.items, self.clears, self.missing = [], [], []
        pts = [np.asarray(p, float) for p in r["polygon"]]
        self.edges = []
        for i in range(len(pts)):
            p0, p1 = pts[i], pts[(i + 1) % len(pts)]
            L = float(np.linalg.norm(p1 - p0))
            if L >= 0.3:
                d = (p1 - p0) / L
                self.edges.append({"p0": p0, "p1": p1, "L": L, "dir": d, "n": np.array([-d[1], d[0]]),
                                   "door": False, "window": False})
        self.door_zones, self.win_zones, self.doors = [], [], []
        for o in plan["doors"] + plan["windows"]:
            if self.id not in o.get("rooms", []) or o["wall_id"] not in walls:
                continue
            w = walls[o["wall_id"]]
            u = unit(np.subtract(w["b"], w["a"]))
            nrm = np.array([-u[1], u[0]])
            c = np.asarray(o["center"], float)
            s = min((1, -1), key=lambda sg: self.poly.distance(Point(c + sg * nrm * (w["thickness"] / 2 + 0.05))))
            n_in = s * nrm
            face = c + n_in * w["thickness"] / 2
            is_door = o in plan["doors"] or o.get("kind") == "balcony_door"
            if is_door:
                depth = o["width"] + 0.05 if o.get("swing", {}).get("into") == self.id or o.get("kind") == "balcony_door" else 0.6
                if o.get("kind") == "opening":
                    depth = 0.6
                self.door_zones.append(orect(face + n_in * depth / 2, u, n_in, o["width"] + 0.2, depth))
                self.doors.append((face, o))
            else:
                self.win_zones.append((orect(face + n_in * 0.2, u, n_in, o["width"] + 0.1, 0.4), o["sill"], face))
            for e in self.edges:   # mark the room edge that holds this opening
                if LineString([e["p0"], e["p1"]]).distance(Point(face)) < 0.08:
                    e["door" if is_door else "window"] = True

    def ok(self, fp, h, clears=(), allow=()):
        if not self.poly.buffer(0.01).contains(fp):
            return False
        if any(fp.intersection(z).area > 1e-4 for z in self.door_zones):
            return False
        if any(h > sill + 0.02 and fp.intersection(z).area > 1e-4 for z, sill, _ in self.win_zones):
            return False
        for it in self.items:
            if it["type"] == "rug" or it["type"] in allow or it["id"] in allow:
                continue
            if fp.intersection(it["fp"].buffer(0.02)).area > 1e-4:
                return False
        for c, owner in self.clears:
            if owner not in allow and fp.intersection(c).area > 1e-3:
                return False
        for c in clears:
            if not self.poly.buffer(0.01).contains(c):
                return False
            if any(c.intersection(it["fp"]).area > 1e-3 for it in self.items if it["type"] not in ("rug",) and it["type"] not in allow):
                return False
        return True

    def add(self, typ, c, u, n, w, d, h, clears=(), params=None, allow=()):
        fp = orect(c, u, n, w, d)
        if not self.ok(fp, h, clears, allow):
            return None
        it = {"id": f"{self.id}_{typ}_{len(self.items)}", "type": typ, "c": np.asarray(c, float), "u": np.asarray(u, float),
              "n": np.asarray(n, float), "w": w, "d": d, "h": h, "fp": fp, "params": params or {}}
        self.items.append(it)
        self.clears += [(cl, it["id"]) for cl in clears]
        return it


def against_wall(R, typ, score=None, front=0.0, sides=0.0, allow=(), sizes=None, edges=None, params=None):
    for w, d, h in sizes or SIZES[typ]:
        best = None
        for e in edges or R.edges:
            if e["L"] < w + 0.02:
                continue
            for t in list(np.arange(0.01, e["L"] - w - 0.01 + 1e-9, 0.05)) + [(e["L"] - w) / 2]:
                c = e["p0"] + e["dir"] * (t + w / 2) + e["n"] * (d / 2 + 0.005)
                clears = []
                if front:
                    clears.append(orect(c + e["n"] * (d / 2 + front / 2), e["dir"], e["n"], w, front))
                if sides:
                    for sg in (1, -1):
                        clears.append(orect(c + sg * e["dir"] * (w / 2 + sides / 2) + e["n"] * 0.15, e["dir"], e["n"], sides, d - 0.3))
                fp = orect(c, e["dir"], e["n"], w, d)
                if not R.ok(fp, h, clears, allow):
                    continue
                sc = (score(e, t, w, c) if score else 0.0) + 0.3 * abs(t + w / 2 - e["L"] / 2) / e["L"]
                if best is None or sc < best[0]:
                    best = (sc, c, e, clears)
        if best:
            _, c, e, clears = best
            it = R.add(typ, c, e["dir"], e["n"], w, d, h, clears, params, allow)
            if it:
                it["edge"] = e
                return it
    R.missing.append(typ)
    return None


def free_spot(R, typ, sizes, ring=0.0, prefer=None):
    """Put a free-standing item (table) where it fits best, with a free ring for chairs."""
    minx, miny, maxx, maxy = R.poly.bounds
    for w, d, h in sizes:
        best = None
        for x in np.arange(minx + 0.3, maxx - 0.3, 0.1):
            for y in np.arange(miny + 0.3, maxy - 0.3, 0.1):
                for u in (np.array([1.0, 0]), np.array([0, 1.0])):
                    n = np.array([-u[1], u[0]])
                    c = np.array([x, y])
                    clears = [orect(c, u, n, w + 2 * ring, d + 2 * ring)] if ring else []
                    if not R.ok(orect(c, u, n, w, d), h, clears):
                        continue
                    sc = prefer(c) if prefer else 0
                    if best is None or sc < best[0]:
                        best = (sc, c, u, n, clears)
        if best:
            _, c, u, n, clears = best
            return R.add(typ, c, u, n, w, d, h, clears)
    R.missing.append(typ)
    return None


def chairs_around(R, table, per_side=2):
    for sg in (1, -1):
        for k in range(per_side):
            off = (k - (per_side - 1) / 2) * min(0.6, table["w"] / per_side)
            c = table["c"] + table["u"] * off + sg * table["n"] * (table["d"] / 2 + 0.15)
            n = -sg * table["n"]
            R.add("chair", c, np.array([n[1], -n[0]]), n, 0.45, 0.50, 0.85, allow=(table["id"], "chair"))


def dist_to_windows(R, c):
    return min([np.linalg.norm(np.asarray(f) - c) for _, _, f in R.win_zones] or [5.0])


def ray_len(poly, c, n):
    ln = LineString([tuple(c), tuple(c + n * 30)])
    x = ln.intersection(poly.exterior)
    pts = [x] if x.geom_type == "Point" else list(getattr(x, "geoms", []))
    ds = [np.linalg.norm(np.asarray(p.coords[0]) - c) for p in pts if p.geom_type == "Point"]
    ds = [d for d in ds if d > 0.05]
    return min(ds) if ds else 0.0


# ---------- room templates ----------
def kitchen_run(R, edges=None):
    lens = [round(x, 2) for x in np.arange(3.6, 1.19, -0.3)]
    for L in lens:
        it = against_wall(R, "kitchen_run", sizes=[(L, 0.60, 0.90)], front=0.9, edges=edges,
                          score=lambda e, t, w, c: 3 * e["door"] - 1.0 * e["window"] - 0.2 * e["L"])
        if it:
            R.missing = [m for m in R.missing if m != "kitchen_run"]
            break
    else:
        return None
    # sink under a window if there is one on this wall, hob 1 m away, wall cabinets away from windows
    x_win = []
    for z, sill, face in R.win_zones:
        loc = float(np.dot(np.asarray(face) - it["c"], it["u"]))
        if abs(float(np.dot(np.asarray(face) - it["c"], it["n"]))) < it["d"] / 2 + 0.3 and abs(loc) < it["w"] / 2:
            x_win.append(loc)
    sink = x_win[0] if x_win else -it["w"] / 2 + min(0.9, it["w"] / 3)
    sink = float(np.clip(sink, -it["w"] / 2 + 0.35, it["w"] / 2 - 0.35))
    hob = sink + 1.0 if sink + 1.0 < it["w"] / 2 - 0.35 else sink - 1.0
    upper, pos = [], -it["w"] / 2
    for xw in sorted(x_win) + [None]:
        end = (xw - 0.8) if xw is not None else it["w"] / 2
        if end - pos >= 0.4:
            upper.append([round(pos, 3), round(end, 3)])
        pos = (xw + 0.8) if xw is not None else pos
    it["params"] = {"sink_x": round(sink, 3), "hob_x": round(float(np.clip(hob, -it["w"] / 2 + 0.3, it["w"] / 2 - 0.3)), 3),
                    "upper": upper}
    return it


def furnish(R, plan):
    t = R.type
    if t == "living" or t == "dining":
        for z in R.r.get("zones", []):
            if z["type"] == "kitchen":   # open kitchen inside the living room
                zp = np.asarray(z["pos"])
                kitchen_run(R, edges=sorted(R.edges, key=lambda e: LineString([e["p0"], e["p1"]]).distance(Point(zp)))[:2])
        if t == "living":
            sofa = against_wall(R, "sofa", front=0.35, score=lambda e, t_, w, c: -min(e["L"], 4.5) + 3 * e["door"]
                                + (2 if ray_len(R.poly, c, e["n"]) < 2.4 else 0))
            if sofa:
                opp = [e for e in R.edges if np.dot(e["n"], sofa["n"]) < -0.95]
                tv = against_wall(R, "tv_unit", edges=opp, front=0.3,
                                  score=lambda e, t_, w, c: abs(np.dot(c - sofa["c"], sofa["u"]))) if opp else None
                ct_c = sofa["c"] + sofa["n"] * (sofa["d"] / 2 + 0.40 + 0.30)
                ct = None
                for w, d, h in SIZES["coffee_table"]:
                    ct = R.add("coffee_table", sofa["c"] + sofa["n"] * (sofa["d"] / 2 + 0.40 + d / 2), sofa["u"], sofa["n"], w, d, h,
                               allow=(sofa["id"],))
                    if ct:
                        break
                for w, d, h in SIZES["rug"]:
                    c = (ct["c"] if ct else ct_c)
                    fp = orect(c, sofa["u"], sofa["n"], w, d)
                    if R.poly.buffer(-0.05).contains(fp) and not any(fp.intersects(z) for z in R.door_zones):
                        it = {"id": f"{R.id}_rug", "type": "rug", "c": c, "u": sofa["u"], "n": sofa["n"], "w": w, "d": d, "h": 0.01,
                              "fp": fp, "params": {}}
                        R.items.append(it)
                        break
                for sg in (1, -1):   # armchair beside the coffee table, facing it
                    if not ct:
                        break
                    c = ct["c"] + sg * sofa["u"] * (ct["w"] / 2 + 0.35 + 0.40)
                    n = -sg * sofa["u"]
                    if R.add("armchair", c, np.array([n[1], -n[0]]), n, 0.80, 0.80, 0.80, clears=[orect(c + n * 0.55, np.array([n[1], -n[0]]), n, 0.8, 0.3)],
                             allow=(ct["id"],)):
                        break
                for sg in (-1, 1):   # floor lamp at a sofa end
                    c = sofa["c"] + sg * sofa["u"] * (sofa["w"] / 2 + 0.25) - sofa["n"] * (sofa["d"] / 2 - 0.2)
                    if R.add("floor_lamp", c, sofa["u"], sofa["n"], 0.35, 0.35, 1.60):
                        break
            if R.area >= 22 or t == "dining":
                sc = (lambda c: -np.linalg.norm(c - sofa["c"])) if t == "living" and sofa else None
                tab = free_spot(R, "dining_table", SIZES["dining_table"], ring=0.65, prefer=sc)
                if tab:
                    chairs_around(R, tab, 3 if tab["w"] >= 1.6 else 2)
        else:
            tab = free_spot(R, "dining_table", SIZES["dining_table"], ring=0.65)
            if tab:
                chairs_around(R, tab, 3 if tab["w"] >= 1.6 else 2)
    elif t == "bedroom":
        mnx, mny, mxx, mxy = R.poly.bounds
        double = not R.r.get("child") and (R.r.get("master") or (R.area >= 11 and min(mxx - mnx, mxy - mny) >= 2.9))
        bed = against_wall(R, "bed_double" if double else "bed_single", front=0.6, sides=0.45 if double else 0.0,
                           score=lambda e, t_, w, c: 3 * e["door"] + 1.5 * e["window"] - 0.1 * e["L"])
        if bed is None and double:
            R.missing.pop()
            bed = against_wall(R, "bed_single", front=0.6)
        if bed:
            for sg in ((1, -1) if bed["type"] == "bed_double" else (1, -1)):
                c = bed["c"] + sg * bed["u"] * (bed["w"] / 2 + 0.03 + 0.225) - bed["n"] * (bed["d"] / 2 - 0.20)
                ns = R.add("nightstand", c, bed["u"], bed["n"], 0.45, 0.40, 0.50, allow=(bed["id"],))
                if ns and bed["type"] == "bed_single":
                    break
        against_wall(R, "wardrobe", front=0.7, score=lambda e, t_, w, c: 2 * e["window"] - 0.1 * e["L"]
                     + (2 if bed and e is bed.get("edge") else 0))
        if not double or R.area >= 14:
            desk = against_wall(R, "desk", front=0.7, score=lambda e, t_, w, c: dist_to_windows(R, c))
            if desk:
                n = -desk["n"]
                R.add("chair", desk["c"] + desk["n"] * (desk["d"] / 2 + 0.12), np.array([n[1], -n[0]]), n, 0.45, 0.50, 0.85,
                      allow=(desk["id"],))
        plant_in_corner(R)
    elif t == "kitchen":
        run = kitchen_run(R)
        if run and R.area >= 8:   # L-shape on a neighbouring wall
            nb = [e for e in R.edges if abs(np.dot(e["dir"], run["u"])) < 0.1 and e is not run.get("edge")]
            if nb:
                for L in (2.4, 1.8, 1.2):
                    it = against_wall(R, "kitchen_run", sizes=[(L, 0.60, 0.90)], front=0.9, edges=nb,
                                      score=lambda e, t_, w, c: np.linalg.norm(c - run["c"]))
                    if it:
                        it["params"] = {"sink_x": None, "hob_x": None, "upper": [[-L / 2, L / 2]]}
                        break
                R.missing = [m for m in R.missing if m != "kitchen_run"]
        against_wall(R, "fridge", front=0.7, score=lambda e, t_, w, c: min([np.linalg.norm(c - i["c"]) for i in R.items] or [0]))
        if R.area >= 9:
            tab = free_spot(R, "dining_table", [(0.80, 0.80, 0.75)], ring=0.6)
            if tab:
                chairs_around(R, tab, 1)
    elif t in ("bathroom", "wc"):
        if t == "bathroom":
            mnx, mny, mxx, mxy = R.poly.bounds
            corner = lambda e, t_, w, c: min(t_, e["L"] - t_ - w)
            tub = against_wall(R, "bathtub", front=0.5, score=corner) if R.area >= 4.0 and min(mxx - mnx, mxy - mny) >= 1.5 else None
            if not tub:
                R.missing = [m for m in R.missing if m != "bathtub"]
                against_wall(R, "shower", front=0.6, score=corner)
        against_wall(R, "toilet", front=0.55, sides=0.20, score=lambda e, t_, w, c: 2 * e["door"])
        against_wall(R, "vanity" if t == "bathroom" else "basin_small", front=0.6)
        if t == "bathroom" and R.area >= 5:
            against_wall(R, "washer", front=0.6)
            R.missing = [m for m in R.missing if m != "washer"]
    elif t == "hall":
        ent = [f for f, o in R.doors if len(o.get("rooms", [])) == 1]
        against_wall(R, "shoe_cabinet", front=0.5,
                     score=lambda e, t_, w, c: min([np.linalg.norm(c - f) for f in ent] or [0]))
    elif t == "balcony":
        tab = free_spot(R, "bistro_table", SIZES["bistro_table"], ring=0.0)
        if tab:
            for sg in (1, -1):
                n = -sg * tab["u"]
                R.add("chair", tab["c"] + sg * tab["u"] * 0.55, np.array([n[1], -n[0]]), n, 0.45, 0.50, 0.85, allow=(tab["id"],))
    elif t == "study":
        desk = against_wall(R, "desk", front=0.7, score=lambda e, t_, w, c: dist_to_windows(R, c))
        if desk:
            n = -desk["n"]
            R.add("chair", desk["c"] + desk["n"] * (desk["d"] / 2 + 0.12), np.array([n[1], -n[0]]), n, 0.45, 0.50, 0.85, allow=(desk["id"],))
        against_wall(R, "bookshelf", front=0.5)
    if t == "living":
        plant_in_corner(R)


def plant_in_corner(R):
    pts = [np.asarray(p, float) for p in R.r["polygon"]]
    cands = []
    for i, p in enumerate(pts):
        a, b = unit(pts[i - 1] - p), unit(pts[(i + 1) % len(pts)] - p)
        bis = unit(a + b)
        c = p + bis * 0.36
        if R.poly.contains(Point(c)):
            cands.append((dist_to_windows(R, c), c))
    for _, c in sorted(cands, key=lambda x: x[0]):
        if R.add("plant", c, np.array([1.0, 0]), np.array([0, 1.0]), 0.45, 0.45, 1.10):
            return


def verify(rooms):
    ck = {"overlaps": 0, "outside_room": 0, "blocking_doors": 0, "tall_in_front_of_windows": 0}
    for R in rooms:
        its = [i for i in R.items if i["type"] != "rug"]
        for i, a in enumerate(its):
            if not R.poly.buffer(0.02).contains(a["fp"]):
                ck["outside_room"] += 1
            if any(a["fp"].intersection(z).area > 1e-3 for z in R.door_zones):
                ck["blocking_doors"] += 1
            if any(a["h"] > s + 0.02 and a["fp"].intersection(z).area > 1e-3 for z, s, _ in R.win_zones):
                ck["tall_in_front_of_windows"] += 1
            for b in its[i + 1:]:
                pair = {a["type"], b["type"]}
                if pair & {"chair"} and pair & {"dining_table", "desk", "bistro_table"}:
                    continue
                if a["fp"].intersection(b["fp"]).area > 1e-3:
                    ck["overlaps"] += 1
    return ck


def run_layout(job, cfg):
    job = Path(job)
    plan = jload(job / "02_plan/plan.json")
    walls = {w["id"]: w for w in plan["walls"]}
    rooms = [Room(r, plan, walls) for r in plan["rooms"]]
    for R in sorted(rooms, key=lambda R: -R.area):
        try:
            furnish(R, plan)
        except Exception as e:   # one bad room must not stop the run
            R.missing.append(f"error: {e}")
    items = []
    for R in rooms:
        for it in R.items:
            items.append({"id": it["id"], "room": R.id, "room_type": R.type, "type": it["type"],
                          "center": [round(float(it["c"][0]), 4), round(float(it["c"][1]), 4)],
                          "size": [it["w"], it["d"], it["h"]],
                          "rot_deg": round(math.degrees(math.atan2(it["u"][1], it["u"][0])), 3), "z": 0.0,
                          "params": it["params"]})
    checks = verify(rooms)
    empty = [R.id for R in rooms if R.type in ("living", "bedroom", "kitchen", "bathroom") and not R.items]
    warns = [f"{R.id} ({R.type}): could not place {', '.join(R.missing)}" for R in rooms if R.missing]
    warns += [f"main room {r} has no furniture" for r in empty]
    out = {"items": items, "checks": checks, "unfurnished_main_rooms": empty, "warnings": warns,
           "rooms": {R.id: {"type": R.type, "placed": [i["type"] for i in R.items], "missing": R.missing} for R in rooms}}
    (job / "03_layout").mkdir(exist_ok=True)
    jsave(job / "03_layout/layout.json", out)
    draw(job, plan, rooms)
    log(f"layout: {len(items)} items, checks {checks}, warnings {len(warns)}")
    return out


def draw(job, plan, rooms):
    from PIL import Image, ImageDraw, ImageFont
    xs = [p[0] for r in plan["rooms"] for p in r["polygon"]] + [c for w in plan["walls"] for c in (w["a"][0], w["b"][0])]
    ys = [p[1] for r in plan["rooms"] for p in r["polygon"]] + [c for w in plan["walls"] for c in (w["a"][1], w["b"][1])]
    k, pad = 80, 40
    W, H = int((max(xs) - min(xs)) * k) + 2 * pad, int((max(ys) - min(ys)) * k) + 2 * pad
    P = lambda p: (pad + (p[0] - min(xs)) * k, H - pad - (p[1] - min(ys)) * k)
    im = Image.new("RGB", (W, H), "white")
    dr = ImageDraw.Draw(im)
    font = ImageFont.truetype(font_path(), 11)
    for w in plan["walls"]:
        dr.line([P(w["a"]), P(w["b"])], fill=(120, 0, 0), width=max(2, int(w["thickness"] * k)))
    for R in rooms:
        dr.polygon([P(p) for p in R.r["polygon"]], outline=(90, 90, 90))
        for z in R.door_zones:
            dr.polygon([P(p) for p in z.exterior.coords], outline=(0, 160, 0))
        for z, _, _ in R.win_zones:
            dr.polygon([P(p) for p in z.exterior.coords], outline=(40, 80, 255))
        for it in R.items:
            col = (230, 225, 210) if it["type"] == "rug" else (170, 150, 120)
            dr.polygon([P(p) for p in it["fp"].exterior.coords], fill=col, outline=(60, 50, 40))
            front = it["c"] + it["n"] * it["d"] / 2
            dr.line([P(it["c"]), P(front)], fill=(200, 0, 0), width=1)
            dr.text(P(it["c"]), it["type"].replace("_", " "), fill=(0, 0, 0), font=font, anchor="mm")
    im.save(job / "03_layout/layout.png")
