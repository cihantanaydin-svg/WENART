"""Stage 2 - geometry core + scale solver -> 02_plan/plan.json + overlay.png.
Works in mask pixels first (same for every input), converts to metres at the end."""
import math
from pathlib import Path
import cv2
import numpy as np
from scipy import ndimage
from shapely.geometry import Polygon, Point, box
from .common import jload, jsave, log, font_path
from .textnorm import classify

MAIN_TYPES = ("living", "bedroom", "kitchen", "bathroom")


# ---------- 1. walls: split the wall mask into straight rectangles ----------
def wall_runs(mask, axis, min_len, max_th, tol):
    min_len |= 1                                     # odd kernel: no 1-px shift
    k = (min_len, 1) if axis == "h" else (1, min_len)
    m = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, k))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, connectivity=4)
    out = []
    for i in range(1, n):
        x, y, w, h, _ = st[i]
        sub = lab[y:y + h, x:x + w] == i
        if axis == "v":
            sub = sub.T                      # rows = across the wall, columns = along it
        has = sub.any(0)
        top = ndimage.median_filter(np.argmax(sub, 0), 5, mode="nearest")
        bot = ndimage.median_filter(sub.shape[0] - 1 - np.argmax(sub[::-1], 0), 5, mode="nearest")
        segs, start = [], None
        for c in range(sub.shape[1]):
            if not has[c]:
                if start is not None:
                    segs.append((start, c - 1))
                start = None
                continue
            if start is None:
                start = c
            elif abs(top[c] - top[c - 1]) > tol or abs(bot[c] - bot[c - 1]) > tol:
                segs.append((start, c - 1))
                start = c
        if start is not None:
            segs.append((start, sub.shape[1] - 1))
        for s0, s1 in segs:
            t0, t1 = int(np.median(top[s0:s1 + 1])), int(np.median(bot[s0:s1 + 1]))
            if s1 - s0 + 1 < min_len or not (1 <= t1 - t0 + 1 <= max_th):
                continue
            if axis == "h":
                out.append(dict(axis="h", x0=x + s0, x1=x + s1, y0=y + t0, y1=y + t1))
            else:
                out.append(dict(axis="v", x0=x + t0, x1=x + t1, y0=y + s0, y1=y + s1))
    for r in out:
        finish(r)
    return out


def finish(r):
    if r["axis"] == "h":
        r.update(c=(r["y0"] + r["y1"]) / 2, th=r["y1"] - r["y0"] + 1, s=r["x0"], e=r["x1"])
    else:
        r.update(c=(r["x0"] + r["x1"]) / 2, th=r["x1"] - r["x0"] + 1, s=r["y0"], e=r["y1"])
    return r


def trim_vertical(hs, vs):
    """Vertical walls stop at horizontal walls, so wall boxes never overlap in 3D."""
    out = []
    for v in vs:
        cuts = sorted((h["y0"], h["y1"]) for h in hs if h["x0"] <= v["x1"] and h["x1"] >= v["x0"]
                      and h["y0"] <= v["y1"] and h["y1"] >= v["y0"])
        pos = v["y0"]
        for a, b in cuts + [(v["y1"] + 1, v["y1"] + 1)]:
            if a - 1 >= pos + 1:
                out.append(finish(dict(v, y0=pos, y1=a - 1)))
            pos = max(pos, b + 1)
    return out


# ---------- 2. openings: gaps between wall pieces ----------
def find_openings(rects, mask, strokes_d, est):
    H, W = mask.shape
    lo, hi, door_max = int(0.45 / est), int(3.2 / est), 1.3 / est
    found = []

    def at_junction(r, end):
        """True if this wall end sits on a perpendicular wall (T or L junction): no gap to look for."""
        for q in rects:
            if q["axis"] == r["axis"]:
                continue
            if r["axis"] == "h" and q["x0"] - 2 <= end <= q["x1"] + 2 and q["y0"] <= r["y1"] + 3 and q["y1"] >= r["y0"] - 3:
                return True
            if r["axis"] == "v" and q["y0"] - 2 <= end <= q["y1"] + 2 and q["x0"] <= r["x1"] + 3 and q["x1"] >= r["x0"] - 3:
                return True
        return False

    for r in rects:
        for d in (-1, 1):
            end = r["s"] if d < 0 else r["e"]
            if at_junction(r, end):
                continue
            offs = [r["c"] - r["th"] / 4, r["c"], r["c"] + r["th"] / 4]
            hit = None
            for step in range(1, hi + 2):
                p = end + d * step
                if not (0 <= p < (W if r["axis"] == "h" else H)):
                    break
                vals = [mask[int(round(o)), p] if r["axis"] == "h" else mask[p, int(round(o))] for o in offs]
                if sum(v > 0 for v in vals) >= 2:
                    hit = step
                    break
            if hit is None or not (lo <= hit - 1 <= hi):
                continue
            a, b = sorted((end + d, end + d * (hit - 1)))
            hp = end + d * hit
            other = next((q for q in rects if q is not r and q["axis"] == r["axis"] and abs(q["c"] - r["c"]) <= 2
                          and q["s"] - 2 <= hp <= q["e"] + 2), None)
            collinear = other is not None and abs(other["th"] - r["th"]) <= max(2, 0.3 * r["th"])
            if not collinear and (b - a + 1) > door_max and arc_score(strokes_d, dict(axis=r["axis"], c=r["c"], th=r["th"], a=a, b=b))[0] < 0.6:
                continue
            op = dict(axis=r["axis"], c=r["c"], th=r["th"], a=a, b=b, rect=r)
            if not any(o["axis"] == op["axis"] and abs(o["c"] - op["c"]) <= 3 and abs(o["a"] - a) <= 3
                       and abs(o["b"] - b) <= 3 for o in found):
                found.append(op)
    return found


def arc_score(strokes_d, op):
    """Look for a door swing arc next to the gap. Returns (score, hinge_end, side)."""
    best = (0.0, None, None)
    w = op["b"] - op["a"] + 1
    H, W = strokes_d.shape
    for hinge in ("a", "b"):
        hpos = op["a"] - 0.5 if hinge == "a" else op["b"] + 0.5
        toward = 1 if hinge == "a" else -1
        for side in (1, -1):
            face = op["c"] + side * op["th"] / 2
            hits = 0
            angs = np.radians(np.linspace(10, 80, 15))
            for t in angs:
                along, across = hpos + toward * w * math.cos(t), face + side * w * math.sin(t)
                x, y = (along, across) if op["axis"] == "h" else (across, along)
                xi, yi = int(round(x)), int(round(y))
                if 0 <= xi < W and 0 <= yi < H and strokes_d[yi, xi]:
                    hits += 1
            sc = hits / len(angs)
            if sc > best[0]:
                best = (sc, hinge, side)
    return best


# ---------- 3. rooms ----------
def orthogonalize(pts, tol_deg=8):
    n = len(pts)
    kinds = []
    for i in range(n):
        (x0, y0), (x1, y1) = pts[i], pts[(i + 1) % n]
        a = abs(math.degrees(math.atan2(y1 - y0, x1 - x0))) % 180
        kinds.append("h" if min(a, 180 - a) < tol_deg else "v" if abs(a - 90) < tol_deg else "d")
    out = []
    for i in range(n):
        ki, ko = kinds[i - 1], kinds[i]
        (xp, yp), (x, y), (xn, yn) = pts[i - 1], pts[i], pts[(i + 1) % n]
        if ki == ko and ki in "hv":
            continue
        if {ki, ko} == {"h", "v"}:
            hx = (x + xn) / 2 if ko == "v" else (xp + x) / 2
            hy = (y + yn) / 2 if ko == "h" else (yp + y) / 2
            out.append((hx, hy))
        else:
            out.append((x, y))
    return out if len(out) >= 3 else pts


def snap_edges(poly, hl, vl, tol):
    """Move axis-aligned polygon edges onto nearby vector wall-face lines (exact vector geometry)."""
    pts = list(poly.exterior.coords)[:-1]
    n = len(pts)
    ys, xs = {}, {}
    for i in range(n):
        (x0, y0), (x1, y1) = pts[i], pts[(i + 1) % n]
        if abs(y1 - y0) < 1e-6 and hl:
            lo_, hi_ = sorted((x0, x1))
            c = [l[0] for l in hl if abs(l[0] - y0) <= tol and min(hi_, l[2]) - max(lo_, l[1]) > 0.3 * (hi_ - lo_)]
            if c:
                ys[i] = min(c, key=lambda v: abs(v - y0))
        elif abs(x1 - x0) < 1e-6 and vl:
            lo_, hi_ = sorted((y0, y1))
            c = [l[0] for l in vl if abs(l[0] - x0) <= tol and min(hi_, l[2]) - max(lo_, l[1]) > 0.3 * (hi_ - lo_)]
            if c:
                xs[i] = min(c, key=lambda v: abs(v - x0))
    new = []
    for i in range(n):
        x, y = pts[i]
        for e in (i - 1 if i > 0 else n - 1, i):
            if e in ys:
                y = ys[e]
            if e in xs:
                x = xs[e]
        new.append((x, y))
    p = Polygon(new)
    return p if p.is_valid and abs(p.area - poly.area) < 0.2 * poly.area else poly


def comp_polygon(comp_mask, est, hl, vl):
    cnts, _ = cv2.findContours(comp_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(cnts, key=cv2.contourArea)
    ap = cv2.approxPolyDP(c, max(1.0, 0.015 / est), True).reshape(-1, 2).astype(float)
    pts = orthogonalize([tuple(p) for p in ap])
    poly = Polygon(pts).buffer(0)
    if poly.geom_type != "Polygon":
        poly = max(poly.geoms, key=lambda g: g.area)
    poly = poly.buffer(0.5, join_style=2).simplify(0.05)
    if hl or vl:
        poly = snap_edges(poly, hl, vl, 0.03 / est)
    return poly


# ---------- 4. scale ----------
def weighted_median(votes):
    votes = sorted(votes)
    tot = sum(w for _, w in votes)
    acc = 0
    for v, w in votes:
        acc += w
        if acc >= tot / 2:
            return v
    return votes[-1][0]


def solve_scale(ex, rooms, doors_px, th_px, texts):
    checks, warnings = [], []
    lab = []
    for r in rooms:
        if r.get("label_area") and r["type"] != "balcony":
            lab.append((math.sqrt(r["label_area"] / r["poly"].area), 1.0, r))
    if ex.get("m_per_px"):
        s, method, conf = ex["m_per_px"], "dxf_units", 1.0
    else:
        votes = [(v, w) for v, w, _ in lab]
        note = next((h for h in ex.get("hints", []) if h["method"] == "scale_note"), None)
        if not note and ex.get("dpi"):
            n = next((t["value"] for t in texts if t["kind"] == "scale"), None)
            if n:
                note = {"method": "scale_note", "m_per_px": 0.0254 / ex["dpi"] * n, "note": f"1/{n} at {ex['dpi']:.0f} dpi"}
        if votes:
            med = weighted_median(votes)
            good = [(v, w) for v, w in votes if abs(v / med - 1) <= 0.02]
            s = weighted_median(good or votes)
            method, conf = "area_labels", min(0.95, 0.6 + 0.1 * len(good))
            if len(good) < len(votes):
                warnings.append(f"{len(votes) - len(good)} area label(s) disagree with the others by >2 %")
            if note:
                agree = abs(note["m_per_px"] / s - 1)
                checks.append({"method": "scale_note", "m_per_px": note["m_per_px"], "deviation_pct": round(100 * agree, 2)})
                if agree < 0.01 and ex["kind"] == "pdf_vector":
                    s, method, conf = note["m_per_px"], "scale_note+area_labels", 0.99
                elif agree >= 0.02:
                    warnings.append(f"printed scale {note['note']} disagrees with area labels by {100 * agree:.1f} % - using labels")
        elif note:
            s, method, conf = note["m_per_px"], "scale_note", 0.7 if ex["kind"] == "pdf_vector" else 0.4
            warnings.append("scale from the printed scale note only (no area labels found)")
        elif doors_px:
            s, method, conf = 0.90 / float(np.median(doors_px)), "door_width_0.90m", 0.25
            warnings.append("SCALE UNCERTAIN: only door widths available (expect +/-5-10 %)")
        else:
            s, method, conf = 0.18 / max(th_px, 1), "wall_thickness_guess", 0.1
            warnings.append("SCALE UNKNOWN: guessed from wall thickness - sizes are rough")
    for v, _, r in lab:
        checks.append({"method": "area_label", "room": r["id"], "label_m2": r["label_area"],
                       "measured_m2": round(r["poly"].area * s * s, 2),
                       "deviation_pct": round(100 * (r["poly"].area * s * s / r["label_area"] - 1), 2)})
    return s, method, conf, checks, warnings


# ---------- 5. main ----------
def run_plan(job, cfg):
    job = Path(job)
    ex = jload(job / "01_parse/extract.json")
    mask = cv2.imread(str(job / "01_parse/wall_mask.png"), 0)
    strokes = cv2.imread(str(job / "01_parse/strokes.png"), 0)
    warnings = list(ex.get("warnings", []))
    texts = list(ex.get("texts", [])) + jload(job / "01_parse/vlm_texts.json", [])
    for t in texts:
        t.update(classify(t["text"]))
    hl, vl = ex.get("hlines", []), ex.get("vlines", [])
    # typical wall thickness (px) from the medial axis of the wall mask
    from skimage.morphology import skeletonize
    dt = cv2.distanceTransform((mask > 0).astype(np.uint8), cv2.DIST_L2, 3)
    sk = skeletonize(mask > 0)
    th_px = float(np.median(2 * dt[sk])) if sk.any() else 10.0
    est0 = ex.get("m_per_px") or next((h["m_per_px"] for h in ex.get("hints", [])), None) \
        or (0.12 / max(1.0, float(np.percentile(2 * dt[sk], 25))) if sk.any() else 0.01)

    def geometry(est):
        """walls, openings, rooms and labels for one provisional scale (metres per pixel)."""
        tol = max(1, int(round(th_px * 0.15)))
        hs = wall_runs(mask, "h", max(3, int(0.3 / est)), int(0.65 / est), tol)
        vs = trim_vertical(hs, wall_runs(mask, "v", max(3, int(0.3 / est)), int(0.65 / est), tol))
        rects = hs + vs
        strokes_d = cv2.dilate(strokes, np.ones((5, 5), np.uint8)) > 0
        ops = find_openings(rects, mask, strokes_d, est)
        # rooms = enclosed free space once the openings are closed
        closed = mask.copy()
        for o in ops:
            t0, t1 = int(math.floor(o["c"] - o["th"] / 2 + 0.5)), int(math.ceil(o["c"] + o["th"] / 2 - 0.5))
            if o["axis"] == "h":
                closed[t0:t1 + 1, o["a"]:o["b"] + 1] = 255
            else:
                closed[o["a"]:o["b"] + 1, t0:t1 + 1] = 255
        free = np.pad((closed == 0).astype(np.uint8), 1, constant_values=1)
        n, lab = cv2.connectedComponents(free, connectivity=4)
        lab = lab[1:-1, 1:-1]
        ext = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]])).tolist())
        fdt = cv2.distanceTransform((lab > 0).astype(np.uint8) * (closed == 0), cv2.DIST_L2, 3)
        areas = np.bincount(lab.ravel(), minlength=n)
        maxdt = ndimage.maximum(fdt, lab, index=np.arange(n))
        rooms = []
        for i in range(1, n):
            if i in ext or areas[i] * est * est < 1.0 or maxdt[i] * est < 0.3:
                continue
            rooms.append({"lab": i, "poly": comp_polygon(lab == i, est, hl, vl), "labels": [], "areas": []})
        # balconies: labelled areas outside the walls, bounded by thin railing lines
        gk = max(5, int(0.05 / est) | 1)
        grow = np.ones((gk, gk), np.uint8)   # close small gaps where railing lines meet walls
        barrier = np.pad(((cv2.dilate(strokes, grow) > 0) | (closed > 0)).astype(np.uint8), 1, constant_values=0)
        bn, blab = cv2.connectedComponents((1 - barrier).astype(np.uint8), connectivity=4)
        blab = blab[1:-1, 1:-1]
        bext = set(np.unique(np.concatenate([blab[0], blab[-1], blab[:, 0], blab[:, -1]])).tolist())
        for t in texts:
            if t.get("type") != "balcony":
                continue
            x, y = int(round(t["x"])), int(round(t["y"]))
            if not (0 <= y < lab.shape[0] and 0 <= x < lab.shape[1]) or lab[y, x] not in ext:
                continue
            R = int(0.5 / est)     # the label text itself is ink: use the main free region around it
            win = blab[max(0, y - R):y + R + 1, max(0, x - R):x + R + 1].ravel()
            win = win[(win > 0) & ~np.isin(win, list(bext))]
            bid = int(np.bincount(win).argmax()) if win.size else 0
            comp = blab == bid
            if bid and bid not in bext and 1.0 < comp.sum() * est * est < 40:
                comp = (cv2.dilate(comp.astype(np.uint8), grow) > 0) & (closed == 0)
                comp = cv2.morphologyEx(comp.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8)) > 0
                rooms.append({"lab": -1, "poly": comp_polygon(comp, est, [], []), "labels": [t], "areas": [],
                              "balcony": True})
            else:
                warnings.append("balcony label found but its outline is not closed - balcony skipped")
        # attach labels and area texts to rooms
        for t in texts:
            if t["kind"] not in ("room_label", "area") or t.get("type") == "balcony":
                if t.get("type") == "balcony" and t.get("area"):
                    for r in rooms:
                        if r.get("balcony") and r["poly"].buffer(3).contains(Point(t["x"], t["y"])):
                            r["areas"].append(t["area"])
                continue
            p = Point(t["x"], t["y"])
            cand = [r for r in rooms if r["poly"].buffer(0.5 / est).contains(p)]
            if not cand:
                continue
            r = min(cand, key=lambda r: r["poly"].distance(p))
            if t["kind"] == "room_label":
                r["labels"].append(t)
                if t.get("area"):
                    r["areas"].append(t["area"])
            else:
                r["areas"].append(t["value"])
        for k, r in enumerate(rooms):
            r["id"] = f"r{k + 1}"
            lb = r["labels"][0] if r["labels"] else None
            r["type"] = "balcony" if r.get("balcony") else (lb["type"] if lb else None)
            r["label"] = lb["text"] if lb else ""
            r["extra"] = {k2: v for k2, v in (lb or {}).items() if k2 in ("master", "child", "open")}
            r["label_area"] = r["areas"][0] if r["areas"] else None
            if len(r["labels"]) > 1:
                r["zones"] = [{"type": l["type"], "x": l["x"], "y": l["y"]} for l in r["labels"][1:] if l["type"] != r["type"]]

        return est, rects, ops, closed, lab, ext, rooms, strokes_d

    est, rects, ops, closed, lab, ext, rooms, strokes_d = geometry(est0)
    # scale
    doors_px = [o["b"] - o["a"] + 1 for o in ops if o["b"] - o["a"] + 1 < 1.3 / est]
    s, method, conf, checks, sw = solve_scale(ex, rooms, doors_px, th_px, texts)
    if not ex.get("m_per_px") and abs(s / est - 1) > 0.08:   # second pass with the solved scale
        est, rects, ops, closed, lab, ext, rooms, strokes_d = geometry(s)
        doors_px = [o["b"] - o["a"] + 1 for o in ops if o["b"] - o["a"] + 1 < 1.3 / est]
        s, method, conf, checks, sw = solve_scale(ex, rooms, doors_px, th_px, texts)
    warnings += sw
    # unlabelled rooms: simple rules (a VLM pass can refine them later)
    for r in rooms:
        if r["type"]:
            continue
        a = r["poly"].area * s * s
        mnx, mny, mxx, mxy = r["poly"].bounds
        w, d = sorted(((mxx - mnx) * s, (mxy - mny) * s))
        r["type"] = "storage" if a < 2.5 else "hall" if (w < 1.6 and d / w > 2.2) else "other"
        r["confidence"] = 0.3
        warnings.append(f"room {r['id']} ({a:.1f} m2) has no label - typed as '{r['type']}' by shape")
    bed = [r for r in rooms if r["type"] == "bedroom"]
    if bed and not any(r["extra"].get("master") for r in bed):
        max(bed, key=lambda r: r["poly"].area)["extra"]["master"] = True
    # classify openings and pick door swings
    def region(pt):
        """room id, 'outside', or None (wall pixel / tiny gap)."""
        x, y = int(round(pt[0])), int(round(pt[1]))
        if not (0 <= y < lab.shape[0] and 0 <= x < lab.shape[1]):
            return "outside"
        L = lab[y, x]
        if L == 0:
            return None
        if L in ext:
            return next((r["id"] for r in rooms if r["lab"] == -1 and r["poly"].contains(Point(pt))), "outside")
        return next((r["id"] for r in rooms if r["lab"] == L), None)
    rmap = {r["id"]: r for r in rooms}
    doors, windows = [], []
    for o in ops:
        mid = (o["a"] + o["b"]) / 2
        off = o["th"] / 2 + 0.3 / est
        sides = {}
        for sgn in (1, -1):
            pt = (mid, o["c"] + sgn * off) if o["axis"] == "h" else (o["c"] + sgn * off, mid)
            sides[sgn] = region(pt)
        sc, hinge, side = arc_score(strokes_d, o)
        width_px = o["b"] - o["a"] + 1
        ins = [v for v in sides.values() if v and v != "outside" and rmap[v]["type"] != "balcony"]
        balc = [v for v in sides.values() if v and v != "outside" and rmap[v]["type"] == "balcony"]
        if not ins:
            continue
        if len(ins) == 2:
            kind = "hinged" if width_px * s <= 1.3 else "opening"
            if sc >= 0.6 and kind == "hinged":
                into = sides[side]
            else:
                a_, b_ = rmap[ins[0]], rmap[ins[1]]
                into = b_["id"] if a_["type"] == "hall" or (b_["type"] in ("bathroom", "wc") and a_["type"] != "hall") else a_["id"]
                side = next(k for k, v in sides.items() if v == into)
                hinge = None
            doors.append(dict(op=o, kind=kind, into=into, side=side, hinge=hinge, rooms=ins))
        elif sc >= 0.6 and not balc:
            doors.append(dict(op=o, kind="hinged", into=sides[side], side=side, hinge=hinge, rooms=ins))
        else:
            windows.append(dict(op=o, kind="balcony_door" if balc else "window", rooms=ins))
    # walls: merge collinear pieces through their openings
    groups = []
    for r in rects:
        g = next((g for g in groups if g["axis"] == r["axis"] and abs(g["c"] - r["c"]) <= 2
                  and abs(g["th"] - r["th"]) <= max(2, 0.3 * r["th"])), None)
        if g is None:
            g = {"axis": r["axis"], "c": r["c"], "th": r["th"], "iv": []}
            groups.append(g)
        g["iv"].append([r["s"], r["e"]])
        r["g"] = g
    for o in ops:
        o["rect"]["g"]["iv"].append([o["a"], o["b"]])
    walls = []
    for g in groups:
        iv = sorted(g["iv"])
        cur = list(iv[0])
        for a, b in iv[1:] + [[10 ** 9, 10 ** 9]]:
            if a <= cur[1] + 2:
                cur[1] = max(cur[1], b)
            else:
                walls.append({"axis": g["axis"], "c": g["c"], "th": g["th"], "s": cur[0], "e": cur[1]})
                cur = [a, b]
    # ---------- convert pixels -> metres ----------
    allpts = [p for r in rooms for p in r["poly"].exterior.coords]
    for w in walls:
        allpts += [(w["s"] - 0.5, w["c"]), (w["e"] + 0.5, w["c"])] if w["axis"] == "h" else [(w["c"], w["s"] - 0.5), (w["c"], w["e"] + 0.5)]
    xs = [(p[0] + 0.5) * s for p in allpts]
    ys = [-(p[1] + 0.5) * s for p in allpts]
    ox, oy = -min(xs), -min(ys)
    X = lambda u: round((u + 0.5) * s + ox, 4)
    Y = lambda v: round(-(v + 0.5) * s + oy, 4)
    hts = cfg["heights"]
    plan_walls = []
    for k, w in enumerate(walls):
        w["id"] = f"w{k + 1}"
        if w["axis"] == "h":
            a, b = [X(w["s"] - 0.5), Y(w["c"])], [X(w["e"] + 0.5), Y(w["c"])]
            probes = [(u, w["c"] + sg * (w["th"] / 2 + 0.2 / est)) for u in np.linspace(w["s"], w["e"], 7)[1:-1] for sg in (1, -1)]
        else:
            a, b = [X(w["c"]), Y(w["e"] + 0.5)], [X(w["c"]), Y(w["s"] - 0.5)]
            probes = [(w["c"] + sg * (w["th"] / 2 + 0.2 / est), v) for v in np.linspace(w["s"], w["e"], 7)[1:-1] for sg in (1, -1)]
        regs = [region(p) for p in probes]
        exterior = any(r_ == "outside" or (r_ and rmap[r_]["type"] == "balcony") for r_ in regs)
        plan_walls.append({"id": w["id"], "a": a, "b": b, "thickness": round(w["th"] * s, 3),
                           "height": hts["ceiling"], "exterior": exterior, "source": ex["kind"]})

    def wall_of(o):
        for w, pw in zip(walls, plan_walls):
            if w["axis"] == o["axis"] and abs(w["c"] - o["c"]) <= 2 and w["s"] <= (o["a"] + o["b"]) / 2 <= w["e"]:
                return pw["id"]
        return None

    def centre(o):
        m = (o["a"] + o["b"]) / 2
        return [X(m), Y(o["c"])] if o["axis"] == "h" else [X(o["c"]), Y(m)]

    plan_doors = []
    for k, d in enumerate(doors):
        o = d["op"]
        hinge = d["hinge"]
        if hinge is None:   # hinge on the end next to a perpendicular wall
            ends = {"a": o["a"] - 2, "b": o["b"] + 2}
            def near(e):
                off = d["side"] * (o["th"] / 2 + 0.15 / est)
                p = (ends[e], o["c"] + off) if o["axis"] == "h" else (o["c"] + off, ends[e])
                x, y = int(round(p[0])), int(round(p[1]))
                return 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1] and mask[y, x] > 0
            hinge = "a" if near("a") or not near("b") else "b"
        # 'a'/'b' in pixel order -> plan wall direction (vertical walls run bottom->top in metres)
        if o["axis"] == "v":
            hinge = "b" if hinge == "a" else "a"
        plan_doors.append({"id": f"d{k + 1}", "wall_id": wall_of(o), "center": centre(o),
                           "width": round((o["b"] - o["a"] + 1) * s, 3), "head": hts["door_head"],
                           "swing": {"hinge": hinge, "into": d["into"]}, "kind": d["kind"],
                           "rooms": d["rooms"]})
    plan_windows = []
    for k, wdw in enumerate(windows):
        o = wdw["op"]
        width = (o["b"] - o["a"] + 1) * s
        wet = any(rmap[r]["type"] in ("bathroom", "wc") for r in wdw["rooms"]) and width <= 0.9
        sill = 0.0 if wdw["kind"] == "balcony_door" else hts["wet_window_sill"] if wet else hts["window_sill"]
        head = hts["door_head"] if wdw["kind"] == "balcony_door" else hts["wet_window_head"] if wet else hts["window_head"]
        plan_windows.append({"id": f"win{k + 1}", "wall_id": wall_of(o), "center": centre(o), "width": round(width, 3),
                             "sill": sill, "head": head, "kind": wdw["kind"], "rooms": wdw["rooms"]})
    plan_rooms = []
    for r in rooms:
        pts = [(X(u), Y(v)) for u, v in r["poly"].exterior.coords]
        poly = Polygon(pts)
        if not poly.exterior.is_ccw:
            pts = pts[::-1]
        mr = poly.minimum_rotated_rectangle
        e = list(mr.exterior.coords)
        size = sorted([round(math.dist(e[0], e[1]), 3), round(math.dist(e[1], e[2]), 3)])
        bx = poly.bounds
        if abs(poly.area - (bx[2] - bx[0]) * (bx[3] - bx[1])) < 0.02 * poly.area:
            size = [round(bx[2] - bx[0], 3), round(bx[3] - bx[1], 3)]
        pr = {"id": r["id"], "type": r["type"], "label": r["label"], "polygon": [list(p) for p in pts[:-1]],
              "area_m2": round(poly.area, 2), "label_area_m2": r["label_area"], "size_m": size,
              "ceiling_height": hts["ceiling"], "confidence": r.get("confidence", 0.9 if r["labels"] else 0.3)}
        pr.update(r["extra"])
        if r.get("zones"):
            pr["zones"] = [{"type": z["type"], "pos": [X(z["x"]), Y(z["y"])]} for z in r["zones"]]
        plan_rooms.append(pr)
    warnings = list(dict.fromkeys(warnings))
    plan = {"schema_version": "1.0",
            "source": {"file": Path(ex.get("input", "")).name, "kind": ex["kind"], "page": ex.get("page")},
            "units": "m",
            "scale": {"m_per_px": s, "method": method, "confidence": round(conf, 2), "checks": checks},
            "defaults": {"ceiling_height": hts["ceiling"], "door_head": hts["door_head"],
                         "window_sill": hts["window_sill"], "window_head": hts["window_head"]},
            "walls": plan_walls, "doors": plan_doors, "windows": plan_windows, "rooms": plan_rooms,
            "texts": [{"text": t["text"], "pos": [X(t["x"]), Y(t["y"])], "kind": t["kind"],
                       **({"value": t["value"]} if "value" in t else {})} for t in texts],
            "warnings": warnings,
            "debug": {"background": "01_parse/page.png", "m_to_px": [1 / s, 0, -ox / s - 0.5, 0, -1 / s, oy / s - 0.5]}}
    out = job / "02_plan"
    out.mkdir(exist_ok=True)
    jsave(out / "plan.json", plan)
    overlay(job, plan)
    log(f"plan: {len(plan_walls)} walls, {len(plan_doors)} doors, {len(plan_windows)} windows, "
        f"{len(plan_rooms)} rooms, scale {method} ({s:.5f} m/px, conf {conf:.2f})")
    return plan


def overlay(job, plan):
    from PIL import Image, ImageDraw, ImageFont
    job = Path(job)
    bg = cv2.imread(str(job / plan["debug"]["background"]), 0)
    a, _, c, _, e, f = plan["debug"]["m_to_px"]
    k = min(1.0, 2400 / max(bg.shape))
    P = lambda x, y: ((a * x + c + 0.5) * k, (e * y + f + 0.5) * k)
    img = Image.fromarray(cv2.resize(bg, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)).convert("RGBA")
    lay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    dr = ImageDraw.Draw(lay)
    font = ImageFont.truetype(font_path(), max(12, int(img.size[0] / 90)))
    colors = {"living": (255, 190, 80), "bedroom": (120, 170, 255), "kitchen": (120, 220, 120), "bathroom": (80, 220, 220),
              "wc": (80, 200, 200), "hall": (200, 200, 200), "balcony": (220, 150, 255)}
    for r in plan["rooms"]:
        dr.polygon([P(*p) for p in r["polygon"]], fill=colors.get(r["type"], (255, 150, 150)) + (70,),
                   outline=colors.get(r["type"], (255, 150, 150)) + (255,))
    for w in plan["walls"]:
        (x0, y0), (x1, y1) = w["a"], w["b"]
        t = w["thickness"] / 2
        pts = [(x0, y0 - t), (x1, y1 - t), (x1, y1 + t), (x0, y0 + t)] if abs(y1 - y0) < 1e-6 else \
              [(x0 - t, y0), (x1 - t, y1), (x1 + t, y1), (x0 + t, y0)]
        dr.polygon([P(*p) for p in pts], fill=(230, 30, 30, 120))
    for op, col in [(d, (0, 170, 0, 255)) for d in plan["doors"]] + [(wd, (30, 60, 255, 255)) for wd in plan["windows"]]:
        w = next((x for x in plan["walls"] if x["id"] == op["wall_id"]), None)
        if not w:
            continue
        (x0, y0), (x1, y1) = w["a"], w["b"]
        L = math.dist(w["a"], w["b"]) or 1
        ux, uy = (x1 - x0) / L, (y1 - y0) / L
        cx, cy = op["center"]
        hw = op["width"] / 2
        dr.line([P(cx - ux * hw, cy - uy * hw), P(cx + ux * hw, cy + uy * hw)], fill=col, width=max(3, int(6 * k)))
    for r in plan["rooms"]:
        cx, cy = Polygon(r["polygon"]).representative_point().coords[0]
        txt = f"{r['label'] or r['type']}\n{r['size_m'][0]:.2f} x {r['size_m'][1]:.2f} m\n{r['area_m2']:.2f} m²"
        dr.multiline_text(P(cx, cy), txt, fill=(0, 0, 0, 255), font=font, anchor="mm", align="center")
    Image.alpha_composite(img, lay).convert("RGB").save(job / "02_plan/overlay.png")
