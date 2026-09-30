"""Stage 1 - parse any input into one common extraction (01_parse/extract.json):
wall mask + stroke mask + background image (same pixel frame), texts in pixel coords,
known scale (if any), scale hints and vector wall-face lines for snapping."""
import math, os, shutil, subprocess
from pathlib import Path
import cv2
import numpy as np
from scipy import ndimage
from .common import WS, jsave, log, run
from .textnorm import fold, repair, parse_scale

UNITS_M = {1: 0.0254, 2: 0.3048, 4: 0.001, 5: 0.01, 6: 1.0, 14: 0.1}
WALL_KEYS = ("DUVAR", "PERDE", "KOLON", "BETON", "WALL")
DOOR_KEYS = ("KAPI", "DOOR")
WIN_KEYS = ("PENCERE", "GLAZ", "WINDOW", "DOGRAMA")
SKIP_KEYS = ("TEFRIS", "MOBILYA", "FURN", "AKS", "GRID", "OLCU", "DIM", "CIVAR")


# ---------- detection ----------
def detect(path):
    b = Path(path).read_bytes()[:64]
    if b.startswith(b"%PDF"):
        return "pdf"
    if b[:4] == b"AC10":
        return "dwg"
    if b.startswith(b"AutoCAD Binary DXF"):
        return "dxf"
    if b[:3] == b"\xff\xd8\xff" or b[:8] == b"\x89PNG\r\n\x1a\n":
        return "image"
    if b"SECTION" in b or Path(path).suffix.lower() == ".dxf":
        return "dxf"
    raise ValueError("Unsupported input (use PDF, DWG, DXF, JPG or PNG). HEIC: export as JPG first.")


def dxf_entities(p):
    """Number of model-space entities (0 if unreadable) - a converted DXF can be large but empty."""
    try:
        from ezdxf import recover
        return len(recover.readfile(str(p))[0].modelspace())
    except Exception:
        return 0


def dwg_to_dxf(dwg, out_dir):
    """LibreDWG first; ODA File Converter (if you installed it) as fallback.
    LibreDWG 0.14 writes an EMPTY model space when asked for DXF r2004 or newer (checked with r2004-r2018),
    so it writes r2000 (non-ASCII text arrives as \\U+XXXX, which textnorm.repair decodes)."""
    out = Path(out_dir) / (Path(dwg).stem + ".dxf")
    tool = WS / "opt/libredwg/bin/dwg2dxf"
    for ver in ("r2000", None):
        try:
            run([tool, "-y"] + (["--as", ver] if ver else []) + ["-o", out, dwg])
        except Exception as e:
            log(f"LibreDWG failed ({ver or 'default version'}): {e}")
        n = dxf_entities(out) if out.exists() else 0
        if n:
            log(f"LibreDWG: {n} entities in {out.name}")
            return out, "libredwg"
        log(f"LibreDWG ({ver or 'default version'}) wrote no drawing entities")
    oda = Path(os.environ.get("PIPE_ODA", WS / "opt/oda/ODAFileConverter"))
    if oda.exists():
        src = Path(out_dir) / "oda_in"
        src.mkdir(exist_ok=True)
        shutil.copy(dwg, src)
        run(["xvfb-run", "-a", oda, src, out_dir, "ACAD2018", "DXF", "0", "1", "*.DWG"])
        cand = Path(out_dir) / (Path(dwg).stem + ".dxf")
        if cand.exists() and cand.stat().st_size > 2000:   # ODA exit codes are unreliable
            return cand, "oda"
    raise RuntimeError("DWG conversion failed (LibreDWG, and ODA not installed or failed)")


# ---------- helpers ----------
def utf8_mojibake(s):
    """UTF-8 text decoded as Windows-1252 (LibreDWG 0.14 writes UTF-8 into r2000 DXF marked ANSI_1252):
    'Ã‡OCUK' -> 'ÇOCUK'. Only applied when the text re-decodes cleanly."""
    if any(c in s for c in "ÂÃÄÅ"):
        try:
            return s.encode("cp1252").decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            return s
    return s


def fill_polys(img, polys, value=255):
    pts = [np.round(np.asarray(p, dtype=np.float64) * 16).astype(np.int32) for p in polys if len(p) >= 3]
    if pts:
        cv2.fillPoly(img, pts, value, lineType=cv2.LINE_8, shift=4)


def draw_lines(img, polylines, width=1, value=255):
    for p in polylines:
        if len(p) >= 2:
            pts = np.round(np.asarray(p, dtype=np.float64) * 16).astype(np.int32)
            cv2.polylines(img, [pts], False, value, max(1, int(round(width))), cv2.LINE_8, shift=4)


def thin_enclosed(lines_mask, max_th_px, min_len_px):
    """Fill thin closed regions between parallel lines (hollow double-line walls)."""
    free = (lines_mask == 0).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(free, connectivity=4)
    if n <= 1:
        return np.zeros_like(lines_mask)
    dt = cv2.distanceTransform(free, cv2.DIST_L2, 3)
    mx = ndimage.maximum(dt, lab, index=np.arange(1, n))
    h, w = lines_mask.shape
    keep = np.zeros(n, bool)
    for i in range(1, n):
        x, y, bw, bh, a = stats[i]
        border = x == 0 or y == 0 or x + bw >= w or y + bh >= h
        if not border and mx[i - 1] * 2 <= max_th_px and max(bw, bh) >= min_len_px:
            keep[i] = True
    return np.where(keep[lab], 255, 0).astype(np.uint8)


def wall_like(poly, lo, hi, min_len):
    r = cv2.minAreaRect(np.asarray(poly, np.float32))
    a, b = sorted(r[1])
    return lo <= a <= hi and b >= min_len


def axis_lines(polylines, tol_deg=1.0):
    """Split segments into horizontal (y, x0, x1) and vertical (x, y0, y1) lines for snapping."""
    hl, vl = [], []
    for p in polylines:
        for (x0, y0), (x1, y1) in zip(p[:-1], p[1:]):
            dx, dy = x1 - x0, y1 - y0
            L = math.hypot(dx, dy)
            if L < 3:
                continue
            ang = abs(math.degrees(math.atan2(dy, dx))) % 180
            if ang < tol_deg or ang > 180 - tol_deg:
                hl.append([(y0 + y1) / 2, min(x0, x1), max(x0, x1)])
            elif abs(ang - 90) < tol_deg:
                vl.append([(x0 + x1) / 2, min(y0, y1), max(y0, y1)])
    return hl, vl


def layer_kind(name):
    f = fold(name.replace("-", " ").replace("_", " "))
    for keys, k in ((WALL_KEYS, "wall"), (DOOR_KEYS, "door"), (WIN_KEYS, "window"), (SKIP_KEYS, "skip")):
        if any(key in f for key in keys):
            return k
    return "other"


# ---------- DXF ----------
def parse_dxf(path, out):
    import ezdxf
    from ezdxf import recover, path as ezpath
    doc, auditor = recover.readfile(str(path))
    warnings = [f"DXF audit: {len(auditor.errors)} errors fixed"] if auditor.errors else []
    unit = UNITS_M.get(int(doc.header.get("$INSUNITS", 0) or 0))
    items = []   # (kind, layer_kind, geometry)

    def walk(ent, parent_layer=None, depth=0):
        lay = ent.dxf.get("layer", "0")
        if lay == "0" and parent_layer:
            lay = parent_layer
        t = ent.dxftype()
        if t == "INSERT" and depth < 6:
            if ent.dxf.name.startswith("*X") or ent.dxf.name.upper().startswith("XREF"):
                warnings.append(f"external reference {ent.dxf.name} ignored")
            try:
                for v in ent.virtual_entities():
                    walk(v, lay, depth + 1)
            except Exception:
                pass
            return
        lk = layer_kind(lay)
        if t in ("TEXT", "MTEXT", "ATTRIB"):
            txt = ent.plain_text() if t == "MTEXT" else ent.dxf.text
            p = ent.dxf.insert
            hgt = ent.dxf.get("char_height", None) if t == "MTEXT" else ent.dxf.get("height", 0.2)
            for i, line in enumerate(repair(utf8_mojibake(txt)).splitlines()):
                if line.strip():
                    items.append(("text", lk, (line.strip(), p.x, p.y - i * (hgt or 0.2) * 1.4, hgt or 0.2)))
            return
        if t == "DIMENSION":
            try:
                items.append(("dim", lk, (ent.get_measurement(), ent.dxf.get("text", ""))))
            except Exception:
                pass
            return
        if lk == "skip":
            return
        try:
            if t == "HATCH":
                for pth in ezpath.from_hatch(ent):
                    pts = [(v.x, v.y) for v in pth.flattening(0.01 / (unit or 0.01))]
                    items.append(("fill", lk, pts))
                return
            if t in ("SOLID", "TRACE"):
                v = [ent.dxf.vtx0, ent.dxf.vtx1, ent.dxf.vtx3, ent.dxf.vtx2]
                items.append(("fill", lk, [(p.x, p.y) for p in v]))
                return
            pth = ezpath.make_path(ent)
            pts = [(v.x, v.y) for v in pth.flattening(0.01 / (unit or 0.01))]
            closed = bool(getattr(ent, "closed", False)) or (t in ("CIRCLE",))
            width = ent.dxf.get("const_width", 0) if t == "LWPOLYLINE" else 0
            items.append(("line", lk, (pts, closed, width)))
        except Exception:
            pass

    for e in doc.modelspace():
        walk(e)
    if unit is None:   # unitless drawing: guess from dimension texts, else from extents
        dims = [(m, s) for k, _, (m, s) in [i for i in items if i[0] == "dim"] if m]
        unit = 0.01
        for m, s in dims:
            try:
                v = float(str(s).replace(",", "."))
                for u in (0.001, 0.01, 1.0):
                    if abs(m * u - v * 0.01) / max(v * 0.01, 1e-6) < 0.02 or abs(m * u - v) / max(v, 1e-6) < 0.02:
                        unit = u
            except ValueError:
                continue
        warnings.append(f"$INSUNITS missing - assumed {unit} m per drawing unit")
    # plan extents from wall geometry (pick the biggest cluster if the file has several drawings)
    wpts = [p for k, lk, g in items if lk == "wall" for p in (g if k == "fill" else g[0])]
    if not wpts:
        wpts = [p for k, lk, g in items if k in ("fill", "line") for p in (g if k == "fill" else g[0])]
        warnings.append("no wall layer found - using all geometry")
    W = np.asarray(wpts) * unit
    x0, y0 = W.min(0) - 3.0
    x1, y1 = W.max(0) + 3.0
    near = [p for k, lk, g in items if k in ("fill", "line") for p in (g if k == "fill" else g[0])]
    near = np.asarray(near) * unit
    near = near[(near[:, 0] > x0) & (near[:, 0] < x1) & (near[:, 1] > y0) & (near[:, 1] < y1)]
    x0, y0 = np.minimum(W.min(0), near.min(0)) - 0.5
    x1, y1 = np.maximum(W.max(0), near.max(0)) + 0.5
    if max(x1 - x0, y1 - y0) > 45:
        coarse = 0.25
        cw, ch = int((x1 - x0) / coarse) + 1, int((y1 - y0) / coarse) + 1
        cm = np.zeros((ch, cw), np.uint8)
        idx = ((W - [x0, y0]) / coarse).astype(int)
        cm[ch - 1 - idx[:, 1], idx[:, 0]] = 1
        cm = cv2.dilate(cm, np.ones((9, 9), np.uint8))
        n, lab, stats, _ = cv2.connectedComponentsWithStats(cm, 8)
        best = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        bx, by, bw, bh, _ = stats[best]
        if n > 2:
            warnings.append(f"{n - 1} drawing groups found - using the largest (check overlay)")
        nx0, nx1 = x0 + bx * coarse - 1, x0 + (bx + bw) * coarse + 1
        ny1, ny0 = y1 - by * coarse + 1, y1 - (by + bh) * coarse - 1
        x0, x1, y0, y1 = nx0, nx1, ny0, ny1
    res = max(0.01, max(x1 - x0, y1 - y0) / 4000)
    H, Wd = int((y1 - y0) / res) + 1, int((x1 - x0) / res) + 1

    def px(pts):   # drawing units -> continuous pixel coords (pixel i covers [i-0.5, i+0.5])
        a = np.asarray(pts, dtype=np.float64) * unit
        return np.stack([(a[:, 0] - x0) / res - 0.5, (y1 - a[:, 1]) / res - 0.5], 1)

    walls, lines, strokes = (np.zeros((H, Wd), np.uint8) for _ in range(3))
    wall_polys = []
    for k, lk, g in items:
        if k == "fill":
            P = px(g)
            wl = wall_like(P, 0.03 / res, 0.6 / res, 0.2 / res) if len(P) >= 3 else False
            if lk == "wall" or (lk == "other" and wl):
                fill_polys(walls, [P])
            fill_polys(strokes, [P])
        elif k == "line":
            pts, closed, width = g
            P = px(pts + ([pts[0]] if closed and pts else []))
            wpx = max(1, width * unit / res)
            draw_lines(strokes, [P], wpx)
            if lk == "wall":
                draw_lines(lines, [P], 1)
                wall_polys.append(P)
                if width * unit >= 0.05:
                    draw_lines(walls, [P], wpx)
    walls |= thin_enclosed(lines, 0.55 / res, 0.25 / res) | lines
    strokes |= walls
    texts = []
    for k, lk, g in items:
        if k == "text" and lk != "skip":
            s, tx, ty, th = g
            u, v = px([(tx, ty)])[0]
            if 0 <= u < Wd and 0 <= v < H:
                texts.append({"text": s, "x": float(u + 0.3 * len(s) * th * unit / res),
                              "y": float(v - 0.5 * th * unit / res), "source": "dxf"})
    hl, vl = axis_lines(wall_polys)
    bg = (255 - (strokes > 0) * 150 - (walls > 0) * 60).astype(np.uint8)
    return dict(kind="dxf", m_per_px=res, walls=walls, strokes=strokes, bg=bg, texts=texts,
                hlines=hl, vlines=vl, hints=[], warnings=warnings,
                dims=[g for k, _, g in items if k == "dim"])


# ---------- PDF ----------
def lum(c):
    if c is None:
        return 1.0
    if isinstance(c, (int, float)):
        return float(c)
    c = list(c)
    if len(c) == 1:
        return float(c[0])
    if len(c) == 3:
        return 0.3 * c[0] + 0.59 * c[1] + 0.11 * c[2]
    if len(c) == 4:
        return 1 - min(1, c[3] + 0.3 * c[0] + 0.59 * c[1] + 0.11 * c[2])
    return 0.5


def path_points(obj):
    """Flatten pdfplumber path commands (top-based coords) into polylines."""
    out, cur = [], []
    for cmd in obj.get("path") or []:
        op = cmd[0]
        if op == "m":
            if len(cur) > 1:
                out.append(cur)
            cur = [cmd[1]]
        elif op == "l":
            cur.append(cmd[1])
        elif op == "c" and cur:
            p0, (p1, p2, p3) = cur[-1], cmd[1:4]
            for t in np.linspace(0, 1, 13)[1:]:
                a, b, c_, d = (1 - t) ** 3, 3 * (1 - t) ** 2 * t, 3 * (1 - t) * t ** 2, t ** 3
                cur.append((a * p0[0] + b * p1[0] + c_ * p2[0] + d * p3[0],
                            a * p0[1] + b * p1[1] + c_ * p2[1] + d * p3[1]))
        elif op == "h" and cur:
            cur.append(cur[0])
    if len(cur) > 1:
        out.append(cur)
    return out or ([obj["pts"]] if obj.get("pts") else [])


def word_phrases(words):
    """Group PDF words into phrases: same line and small gaps (pdfplumber lines span the page)."""
    words = sorted(words, key=lambda w: (round(w["top"], 0), w["x0"]))
    out = []
    for w in words:
        sz = w.get("size") or (w["bottom"] - w["top"])
        last = out[-1] if out else None
        if last and abs(w["top"] - last["top"]) < 0.3 * sz and 0 <= w["x0"] - last["x1"] < 1.2 * sz:
            last.update(text=last["text"] + " " + w["text"], x1=w["x1"], bottom=max(last["bottom"], w["bottom"]))
        else:
            out.append({"text": w["text"], "x0": w["x0"], "x1": w["x1"], "top": w["top"], "bottom": w["bottom"]})
    return out


def parse_pdf(path, out):
    import pdfplumber
    with pdfplumber.open(str(path)) as pdf:
        scores = []
        for i, pg in enumerate(pdf.pages[:20]):
            nvec = len(pg.lines) + len(pg.rects) + len(pg.curves)
            cover = sum((im["x1"] - im["x0"]) * (im["bottom"] - im["top"]) for im in pg.images)
            scores.append((nvec, cover / (pg.width * pg.height), i))
        nvec, cover, pno = max(scores)
        is_vector = nvec >= 300 or (nvec >= 30 and cover < 0.6)
        if not is_vector:
            best = max(scores, key=lambda s: s[1])
            return parse_raster(path, out, kind="pdf_scan", pdf_page=best[2])
        pg = pdf.pages[pno]
        lines_txt = word_phrases(pg.extract_words(x_tolerance=2, extra_attrs=["size"]))
        note = next((parse_scale(ln["text"]) for ln in lines_txt if parse_scale(ln["text"])), None)
        m_per_pt = 0.0254 / 72 * note if note else None
        ppp = max(4, min(10, math.ceil(m_per_pt / 0.006))) if m_per_pt else 6
        ppp = min(ppp, 7000 / max(pg.width, pg.height))
        H, Wd = int(pg.height * ppp) + 1, int(pg.width * ppp) + 1
        px = lambda pts: np.asarray(pts, np.float64) * ppp - 0.5
        walls, strokes = np.zeros((H, Wd), np.uint8), np.zeros((H, Wd), np.uint8)
        objs = [("line", o) for o in pg.lines] + [("rect", o) for o in pg.rects] + [("curve", o) for o in pg.curves]
        lws = [o.get("linewidth") or 0 for _, o in objs if o.get("stroke", True)]
        lw_med = float(np.median([w for w in lws if w > 0])) if any(w > 0 for w in lws) else 0.5
        seg_polys, thick = [], []
        for typ, o in objs:
            if typ == "rect":
                polys = [[(o["x0"], o["top"]), (o["x1"], o["top"]), (o["x1"], o["bottom"]),
                          (o["x0"], o["bottom"]), (o["x0"], o["top"])]]
            elif typ == "line":
                polys = [[(o["x0"], o["top"]), (o["x1"], o["bottom"])]] if not o.get("path") else path_points(o)
            else:
                polys = path_points(o)
            P = [px(p) for p in polys]
            if o.get("fill") and lum(o.get("non_stroking_color")) < 0.55:
                fill_polys(strokes, P)
                for q in P:
                    if len(q) >= 3 and (wall_like(q, 1.5, 0.6 / ((m_per_pt or 0.03528) / ppp), 3)
                                        or cv2.contourArea(q.astype(np.float32)) < (0.5 / ((m_per_pt or 0.03528) / ppp)) ** 2):
                        fill_polys(walls, [q])
            lw = o.get("linewidth") or 0
            if o.get("stroke", True) and lum(o.get("stroking_color")) < 0.7:
                draw_lines(strokes, P, max(1, lw * ppp))
                seg_polys += P
                if lw >= max(2.5 * lw_med, 1.0):
                    draw_lines(walls, P, lw * ppp)
                    thick += P
        if walls.sum() / 255 < 0.002 * H * Wd:   # no poché: hollow double-line walls
            ms = (m_per_pt or 0.03528) / ppp
            walls |= thin_enclosed(strokes, 0.55 / ms, 0.3 / ms)
        texts = []
        for ln in lines_txt:
            u, v = px([((ln["x0"] + ln["x1"]) / 2, (ln["top"] + ln["bottom"]) / 2)])[0]
            texts.append({"text": ln["text"], "x": float(u), "y": float(v), "source": "pdf"})
        hl, vl = axis_lines(thick or seg_polys)
        bg = np.asarray(pg.to_image(resolution=72 * ppp).original.convert("L"))[:H, :Wd]
        if bg.shape != walls.shape:
            bg = cv2.resize(bg, (Wd, H))
        hints = [{"method": "scale_note", "m_per_px": m_per_pt / ppp, "note": f"1/{note}"}] if note else []
        return dict(kind="pdf_vector", m_per_px=None, walls=walls, strokes=strokes | walls, bg=bg,
                    texts=texts, hlines=hl, vlines=vl, hints=hints, page=pno,
                    warnings=[] if texts else ["PDF has no text layer - VLM will read the page"],
                    needs_vlm=not texts)


# ---------- raster (scanned PDF, photo) ----------
def find_sheet(img):
    """Perspective-correct a phone photo: find the paper as the biggest bright 4-corner shape."""
    g = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (7, 7), 0)
    _, th = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) < 0.2 * img.shape[0] * img.shape[1]:
        return None
    ap = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)
    if len(ap) != 4:
        return None
    p = ap.reshape(4, 2).astype(np.float32)
    s, d = p.sum(1), np.diff(p, axis=1).ravel()
    tl, br, tr, bl = p[np.argmin(s)], p[np.argmax(s)], p[np.argmin(d)], p[np.argmax(d)]
    wpx = max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl))
    hpx = max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr))
    asp = wpx / hpx                     # snap to ISO paper (A4/A3: sqrt 2) - photos distort the ratio
    for iso in (math.sqrt(2), 1 / math.sqrt(2)):
        if abs(asp / iso - 1) < 0.15:
            asp = iso
    k = 4000 / max(wpx, hpx)
    W, H = (4000, int(4000 / asp)) if asp >= 1 else (int(4000 * asp), 4000)
    M = cv2.getPerspectiveTransform(np.float32([tl, tr, br, bl]), np.float32([[0, 0], [W, 0], [W, H], [0, H]]))
    return cv2.warpPerspective(img, M, (W, H), flags=cv2.INTER_CUBIC, borderValue=(255, 255, 255))


def deskew(gray):
    """Find the small rotation (+/-4 deg) that makes wall lines straight (projection profile)."""
    k = 1200 / max(gray.shape)
    small = cv2.resize(gray, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
    ink = (small < cv2.threshold(small, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0]).astype(np.float32)
    h, w = ink.shape
    best, best_a = -1, 0.0
    for a in np.arange(-4, 4.001, 0.05):
        M = cv2.getRotationMatrix2D((w / 2, h / 2), a, 1.0)
        r = cv2.warpAffine(ink, M, (w, h))
        sc = r.sum(1).var() + r.sum(0).var()
        if sc > best:
            best, best_a = sc, a
    if abs(best_a) < 0.05:
        return gray, 0.0
    H, W = gray.shape
    M = cv2.getRotationMatrix2D((W / 2, H / 2), best_a, 1.0)
    return cv2.warpAffine(gray, M, (W, H), flags=cv2.INTER_CUBIC, borderValue=255), float(best_a)


def parse_raster(path, out, kind="photo", pdf_page=0):
    from skimage.filters import threshold_sauvola
    warnings, dpi = [], None
    if kind == "pdf_scan":
        import pypdfium2 as pdfium
        dpi = 300
        pil = pdfium.PdfDocument(str(path))[pdf_page].render(scale=dpi / 72).to_pil()
        img = cv2.cvtColor(np.asarray(pil.convert("RGB")), cv2.COLOR_RGB2BGR)
    else:
        from PIL import Image, ImageOps
        pil = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
        img = cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)
        sheet = find_sheet(img)
        if sheet is None:
            warnings.append("paper edges not found - no perspective correction (keep the whole sheet in the photo)")
        else:
            img = sheet
    k = min(1.0, 5000 / max(img.shape[:2]))
    if k < 1:
        img = cv2.resize(img, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
        dpi = dpi * k if dpi else None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bgb = cv2.GaussianBlur(gray, (0, 0), max(gray.shape) / 40)
    gray = np.clip(gray.astype(np.float32) / np.maximum(bgb, 1) * 235, 0, 255).astype(np.uint8)
    gray, ang = deskew(gray)
    if abs(ang) > 0.05:
        warnings.append(f"deskewed by {ang:.2f} degrees")
    otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0]
    ink = (gray < min(otsu, 170)) | ((gray < threshold_sauvola(gray, window_size=31, k=0.15)) & (gray < 200))
    strokes = ink.astype(np.uint8) * 255
    n, lab, stats, _ = cv2.connectedComponentsWithStats(strokes, 8)
    small = stats[:, cv2.CC_STAT_AREA] < 12
    small[0] = False
    strokes[small[lab]] = 0
    # stroke widths along the skeleton: thin lines vs thick walls
    from skimage.morphology import skeletonize
    dt = cv2.distanceTransform(strokes, cv2.DIST_L2, 3)
    sk = skeletonize(strokes > 0)
    wid = 2 * dt[sk]
    thin = float(np.percentile(wid, 30)) if wid.size else 2.0
    ksz = int(math.ceil(thin * 2.2)) + 2
    walls = cv2.morphologyEx(strokes, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (ksz, ksz)))
    if walls.sum() / 255 < 0.004 * walls.size:
        warnings.append("walls look hollow (double lines) - filling thin enclosed regions")
        walls |= thin_enclosed(strokes, 12 * ksz, 6 * ksz)
    return dict(kind=kind, m_per_px=None, walls=walls, strokes=strokes, bg=gray, texts=[],
                hlines=[], vlines=[], hints=[], dpi=dpi, warnings=warnings, needs_vlm=True, page=pdf_page)


# ---------- stage entry ----------
def run_parse(inp, job):
    out = Path(job) / "01_parse"
    out.mkdir(parents=True, exist_ok=True)
    kind = detect(inp)
    src = Path(inp)
    conv = None
    if kind == "dwg":
        src, conv = dwg_to_dxf(inp, Path(job) / "00_input")
        kind = "dxf"
    if kind == "dxf":
        ex = parse_dxf(src, out)
        if conv:
            ex["kind"], ex["converter"] = "dwg", conv
    elif kind == "pdf":
        ex = parse_pdf(src, out)
    else:
        ex = parse_raster(src, out, kind="photo")
    cv2.imwrite(str(out / "wall_mask.png"), ex.pop("walls"))
    cv2.imwrite(str(out / "strokes.png"), ex.pop("strokes"))
    cv2.imwrite(str(out / "page.png"), ex.pop("bg"))
    ex.setdefault("needs_vlm", False)
    ex["input"] = str(inp)
    if ex.get("dims"):
        ex["dims"] = [[float(m), str(s)] for m, s in ex["dims"]][:200]
    jsave(out / "extract.json", ex)
    log(f"parse: kind={ex['kind']} texts={len(ex['texts'])} needs_vlm={ex['needs_vlm']} warnings={ex['warnings']}")
    return ex
