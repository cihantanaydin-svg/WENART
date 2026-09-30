"""Furniture library: real 3D models instead of parametric boxes where a good one exists.
Sources, in order: Poly Haven CC0 models (downloaded once at setup), your own files in
assets/models_user/<type>/*.glb|*.gltf (licence per file, see below), and - only with GEN3D=1 - generated models.
  python -m app.furniture fetch      download Poly Haven furniture models (setup step, needs internet)
  python -m app.furniture catalog    rebuild catalog.json + LICENSES.txt from the files on disk (offline, ~1 s)
  python -m app.furniture report     which furniture types have models, from which source
Licence per user file: a sidecar <file>.json next to it, or licence.json in its folder, e.g.
  {"licence": "CC0-1.0", "source_url": "https://kenney.nl/assets/furniture-kit", "author": "Kenney",
   "front_axis": "-Y", "style_tags": ["modern"]}
Allowed licences come from config furniture.allowed_licences; files without an allowed licence are skipped (logged).
Pure Python: sizes and triangle counts are read from the glTF files themselves (no Blender needed)."""
import hashlib, json, math, re, statistics, struct, sys, time
from pathlib import Path
from .common import WS, jload, jsave, load_config, log

UA = {"User-Agent": "floorplan-poc-setup/1.0 (Runpod proof of concept; one-time download)"}
# layout type -> how to recognise it. cat = Poly Haven taxonomy paths (the asset's `category` field), kw = words in
# name/tags/legacy categories, no = words that exclude. Order matters: the first matching type wins.
TYPES = {
    "armchair": {"kw": ["armchair", "arm chair", "lounge chair", "club chair"], "no": ["sofa"]},
    "sofa": {"cat": ["Furniture/Seating/Sofas & Couches"], "kw": ["sofa", "couch"],
             "no": ["armchair", "arm chair", "ottoman", "footstool", "cushion", "pillow"]},
    "coffee_table": {"cat": ["Furniture/Tables/Coffee Tables"], "kw": ["coffee table"]},
    "dining_table": {"cat": ["Furniture/Tables/Dining Tables"], "kw": ["dining table"]},
    "desk": {"cat": ["Furniture/Tables/Desks"], "kw": ["desk"], "no": ["lamp", "organizer", "organiser"]},
    "nightstand": {"kw": ["nightstand", "night stand", "bedside"], "no": ["lamp"]},
    "bistro_table": {"cat": ["Furniture/Tables/Side & End Tables", "Furniture/Tables/Outdoor Tables"],
                     "kw": ["bistro", "cafe table", "side table", "round table"], "no": ["lamp"]},
    "wardrobe": {"kw": ["wardrobe", "armoire", "closet"]},
    "bookshelf": {"cat": ["Furniture/Storage Furniture/Shelving & Bookcases"], "kw": ["bookshelf", "bookcase"],
                  "no": ["wall shelf", "book "]},
    "shoe_cabinet": {"cat": ["Furniture/Storage Furniture/Cabinets & Cupboards", "Furniture/Storage Furniture/Sideboards",
                             "Furniture/Storage Furniture/Drawers & Dressers"],
                     "kw": ["shoe cabinet", "sideboard", "dresser", "chest of drawers", "cabinet"],
                     "no": ["wall cabinet", "medicine", "filing", "wardrobe", "nightstand", "bedside"]},
    "tv_unit": {"kw": ["tv stand", "tv unit", "tv cabinet", "media console"]},
    "bed": {"cat": ["Furniture/Beds"], "kw": [" bed", "bed "], "no": ["bedside", "flower bed", "bed frame only", "bunk"]},
    "floor_lamp": {"cat": ["Lighting/Floor & Desk/Floor Lamps"], "kw": ["floor lamp", "standing lamp", "floor_lamp"]},
    "plant": {"cat": ["Nature/Plants/Potted Plants"], "kw": ["potted plant", "houseplant", "potted", "plant"],
              "no": ["planter box", "leaf", "branch", "tree trunk"]},
    "chair": {"cat": ["Furniture/Seating/Chairs"], "kw": ["chair"],
              "no": ["armchair", "arm chair", "lounge", "office", "high chair", "rocking", "deck", "beach", "wheelchair"]},
    "fridge": {"kw": ["fridge", "refrigerator"]}, "washer": {"kw": ["washing machine", "washer"]},
    "toilet": {"kw": ["toilet"]}, "bathtub": {"kw": ["bathtub", "bath tub"]}, "vanity": {"kw": ["vanity"]},
    "basin_small": {"kw": ["wash basin", "washbasin", "hand basin"]}, "shower": {"kw": ["shower"]},
}
ALIASES = {"bed_double": "bed", "bed_single": "bed"}      # layout type -> catalog type
HEIGHT_FIT = ("plant", "floor_lamp")                      # organic shapes: uniform scale to the slot height
PARAMETRIC_ONLY = ("rug", "kitchen_run")                  # sizes change per plan: always built in Blender
LICENCE_NAMES = {"cc0": "CC0-1.0", "cc0-1.0": "CC0-1.0", "cc0 1.0": "CC0-1.0", "public domain": "CC0-1.0",
                 "cc-by-4.0": "CC-BY-4.0", "cc by 4.0": "CC-BY-4.0", "cc-by 4.0": "CC-BY-4.0",
                 "cc-by-3.0": "CC-BY-3.0", "cc by 3.0": "CC-BY-3.0", "mit": "MIT", "apache-2.0": "Apache-2.0",
                 "apache 2.0": "Apache-2.0", "owned": "owned", "own": "owned", "mine": "owned"}
ATTRIBUTION = ("CC-BY-4.0", "CC-BY-3.0")


def lib_dir(cfg):
    return Path(cfg["furniture"]["library"])


def catalog_type(layout_type):
    return ALIASES.get(layout_type, layout_type)


def norm_licence(s):
    s = str(s or "").strip()
    return LICENCE_NAMES.get(s.lower(), s)


def words(s):
    return set(re.findall(r"[a-z]{3,}", str(s).lower()))


# ---------- glTF inspection (sizes, triangles, missing files) without Blender ----------
def read_gltf(path):
    b = Path(path).read_bytes()
    if b[:4] != b"glTF":
        return json.loads(b.decode("utf-8")), False
    _, length = struct.unpack_from("<II", b, 4)
    off, js = 12, None
    while off + 8 <= min(len(b), length):
        clen, ctype = struct.unpack_from("<II", b, off)
        if ctype == 0x4E4F534A:            # 'JSON'
            js = json.loads(b[off + 8:off + 8 + clen].decode("utf-8"))
        off += 8 + clen + (-clen % 4)
    if js is None:
        raise ValueError("GLB without JSON chunk")
    return js, True


def _mat_mul(a, b):
    return [[sum(a[i][k] * b[k][j] for k in range(4)) for j in range(4)] for i in range(4)]


def _node_matrix(n):
    if "matrix" in n:
        m = n["matrix"]                    # column-major
        return [[m[c * 4 + r] for c in range(4)] for r in range(4)]
    tx, ty, tz = n.get("translation", [0, 0, 0])
    x, y, z, w = n.get("rotation", [0, 0, 0, 1])
    sx, sy, sz = n.get("scale", [1, 1, 1])
    R = [[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
         [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
         [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]]
    return [[R[0][0] * sx, R[0][1] * sy, R[0][2] * sz, tx], [R[1][0] * sx, R[1][1] * sy, R[1][2] * sz, ty],
            [R[2][0] * sx, R[2][1] * sy, R[2][2] * sz, tz], [0, 0, 0, 1]]


def inspect_gltf(path):
    """{dims_gltf, dims_import [x, y, z] in Blender axes after import (Z up), triangles, missing_files}."""
    path = Path(path)
    js, is_glb = read_gltf(path)
    acc = js.get("accessors", [])
    lo, hi, tris = [math.inf] * 3, [-math.inf] * 3, 0
    I = [[1.0 if i == j else 0.0 for j in range(4)] for i in range(4)]
    scene = js.get("scenes", [{}])[js.get("scene", 0)] if js.get("scenes") else {"nodes": list(range(len(js.get("nodes", []))))}
    stack = [(n, I) for n in scene.get("nodes", [])]
    seen = 0
    while stack:
        ni, parent = stack.pop()
        seen += 1
        if seen > 100000:
            raise ValueError("node graph too large or cyclic")
        node = js["nodes"][ni]
        M = _mat_mul(parent, _node_matrix(node))
        if "mesh" in node:
            for prim in js["meshes"][node["mesh"]].get("primitives", []):
                pa = acc[prim["attributes"]["POSITION"]]
                mode = prim.get("mode", 4)
                n = acc[prim["indices"]]["count"] if "indices" in prim else pa["count"]
                tris += n // 3 if mode == 4 else max(0, n - 2) if mode in (5, 6) else 0
                if "min" not in pa or "max" not in pa:
                    raise ValueError("POSITION accessor without min/max")
                for cx in (pa["min"][0], pa["max"][0]):
                    for cy in (pa["min"][1], pa["max"][1]):
                        for cz in (pa["min"][2], pa["max"][2]):
                            p = [M[r][0] * cx + M[r][1] * cy + M[r][2] * cz + M[r][3] for r in range(3)]
                            lo, hi = [min(a, b) for a, b in zip(lo, p)], [max(a, b) for a, b in zip(hi, p)]
        stack += [(c, M) for c in node.get("children", [])]
    if lo[0] == math.inf:
        raise ValueError("no mesh geometry")
    d = [h - l for h, l in zip(hi, lo)]
    missing = []
    for item in js.get("buffers", []) + js.get("images", []):
        uri = item.get("uri")
        if uri and not uri.startswith("data:"):
            from urllib.parse import unquote
            if not (path.parent / unquote(uri)).exists():
                missing.append(uri)
    return {"dims_gltf": [round(x, 4) for x in d], "dims_import": [round(d[0], 4), round(d[2], 4), round(d[1], 4)],
            "triangles": int(tris), "missing_files": missing, "glb": is_glb}


def layout_dims(dims_import, front_axis):
    """(width along the wall, depth towards the room, height) once the model's front is turned to layout +Y."""
    x, y, z = dims_import
    return [x, y, z] if front_axis in ("+Y", "-Y") else [y, x, z]


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------- Poly Haven (CC0) ----------
def classify(slug, info):
    """First layout type whose rule matches this Poly Haven model, else None."""
    cat = info.get("category") or ""
    text = " " + " ".join([slug.replace("_", " "), info.get("name", ""), cat, " ".join(info.get("tags", [])),
                           " ".join(info.get("categories", []))]).lower() + " "
    for typ, rule in TYPES.items():
        if any(x in text for x in rule.get("no", [])):
            continue
        if any(cat.startswith(c) for c in rule.get("cat", [])) or any(k in text for k in rule.get("kw", [])):
            return typ
    return None


def fetch_polyhaven(cfg=None):
    """Download the most popular Poly Haven model(s) per furniture type (glTF + textures, md5 checked)."""
    import urllib.request
    cfg = cfg or load_config()
    fc = cfg["furniture"]
    dst = lib_dir(cfg) / "polyhaven"
    dst.mkdir(parents=True, exist_ok=True)
    get = lambda url, t=300: urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=t).read()
    ph = lambda p: json.loads(get("https://api.polyhaven.com" + p, 120))
    models = ph("/assets?type=models")
    by_type = {}
    for slug, info in models.items():
        typ = classify(slug, info)
        if typ:
            by_type.setdefault(typ, []).append((info.get("download_count", 0), slug))
    res_pref = [fc["polyhaven_resolution"], "1k", "2k"]
    for typ in TYPES:
        cands = sorted(by_type.get(typ, []), reverse=True)[:fc["polyhaven_per_type"]]
        if not cands:
            log(f"Poly Haven: no model for '{typ}'")
            continue
        for _, slug in cands:
            d = dst / slug
            info = models[slug]
            old = jload(d / "meta.json")
            if old and old.get("files_hash") == info.get("files_hash"):
                continue
            try:
                files = ph(f"/files/{slug}")["gltf"]
                res = next(r for r in res_pref if r in files)
                g = files[res]["gltf"]
                d.mkdir(parents=True, exist_ok=True)
                items = [(f"{slug}_{res}.gltf", g)] + list((g.get("include") or {}).items())
                for rel, f in items:
                    data = get(f["url"])
                    if f.get("md5") and hashlib.md5(data).hexdigest() != f["md5"]:
                        raise ValueError(f"md5 mismatch for {rel}")
                    q = d / rel
                    q.parent.mkdir(parents=True, exist_ok=True)
                    q.write_bytes(data)
                attrs = info.get("attributes") or {}
                mat = attrs.get("material") or []
                jsave(d / "meta.json", {"slug": slug, "type": typ, "name": info.get("name", slug), "file": f"{slug}_{res}.gltf",
                                        "category": info.get("category"), "tags": info.get("tags", []),
                                        "material": mat if isinstance(mat, list) else [mat],
                                        "authors": list((info.get("authors") or {}).keys()), "licence": "CC0-1.0",
                                        "source_url": f"https://polyhaven.com/a/{slug}", "files_hash": info.get("files_hash"),
                                        "downloaded": time.strftime("%Y-%m-%d")})
                log(f"Poly Haven {typ}: {slug} ({res})")
            except Exception as e:
                log(f"WARNING Poly Haven {slug}: {e}")
    return build_catalog(cfg)


# ---------- catalog ----------
def _entry(cfg, f, typ, source, meta, excluded):
    """Inspect one model file and turn it into a catalog entry (or an exclusion with the reason)."""
    lic = norm_licence(meta.get("licence"))
    rec = {"file": str(f), "type": typ, "source": source}
    if source == "generated":
        if not cfg["furniture"]["allow_generated"]:
            excluded.append(dict(rec, reason="generated models turned off (furniture.allow_generated)"))
            return None
    elif lic not in cfg["furniture"]["allowed_licences"]:
        excluded.append(dict(rec, reason=f"licence {lic or 'missing'} not in furniture.allowed_licences"))
        return None
    try:
        info = inspect_gltf(f)
    except Exception as e:
        excluded.append(dict(rec, reason=f"unreadable glTF: {e}"))
        return None
    if info["missing_files"]:
        excluded.append(dict(rec, reason=f"missing files: {info['missing_files'][:3]}"))
        return None
    dims, unit = info["dims_import"], float(meta.get("unit_scale", 1.0))
    if "unit_scale" not in meta and max(dims) > 30 and 0.05 < max(dims) * 0.01 <= 30:
        unit = 0.01                        # exported in centimetres
    dims = [round(x * unit, 4) for x in dims]
    if max(dims) < 0.05 or max(dims) > 30 or min(dims) <= 0:
        excluded.append(dict(rec, reason=f"implausible size {dims} m"))
        return None
    front = meta.get("front_axis", "-Y")   # glTF front is +Z -> Blender -Y after import (glTF 2.0 convention)
    if front not in ("+X", "-X", "+Y", "-Y"):
        front = "-Y"
    tags = sorted(words(" ".join([str(meta.get("name", "")), " ".join(meta.get("tags", [])),
                                  " ".join(meta.get("material", [])), " ".join(meta.get("style_tags", [])),
                                  str(meta.get("category") or "")])))
    return {"id": f"{source}:{meta.get('slug') or Path(f).stem}", "type": typ, "source": source, "file": str(f),
            "name": meta.get("name") or Path(f).stem, "dims_m": [round(x, 3) for x in layout_dims(dims, front)],
            "import_dims_m": dims, "unit_scale": unit, "front_axis": front,
            "fit": "height" if typ in HEIGHT_FIT else "box", "style_tags": tags, "licence": lic,
            "source_url": meta.get("source_url", ""), "author": ", ".join(meta.get("authors", [])) or meta.get("author", ""),
            "polycount": info["triangles"], "sha256": sha256(f), "notes": meta.get("notes", "")}


def _user_meta(f, type_dir):
    for cand in (f.with_suffix(".json"), f.parent / "licence.json", f.parent / "license.json",
                 type_dir / "licence.json", type_dir / "license.json"):
        if cand.exists() and cand != f:
            m = jload(cand) or {}
            if "licence" not in m and "license" in m:
                m["licence"] = m["license"]
            return m
    return {}


def build_catalog(cfg=None):
    cfg = cfg or load_config()
    fc, lib = cfg["furniture"], lib_dir(cfg)
    lib.mkdir(parents=True, exist_ok=True)
    items, excluded = [], []
    for meta_f in sorted((lib / "polyhaven").glob("*/meta.json")):
        m = jload(meta_f)
        e = _entry(cfg, meta_f.parent / m["file"], m["type"], "polyhaven", m, excluded)
        if e:
            items.append(e)
    legacy = Path(fc["legacy_models"])      # plants/lamps downloaded by assets_fetch.py (Poly Haven, CC0)
    for typ in ("plant", "floor_lamp"):
        for f in sorted((legacy / typ).glob("*/*.gltf")) if (legacy / typ).exists() else []:
            slug = f.parent.name
            if any(i["id"] == f"polyhaven:{slug}" for i in items):
                continue
            m = {"slug": slug, "name": slug.replace("_", " "), "licence": "CC0-1.0", "tags": [typ],
                 "source_url": f"https://polyhaven.com/a/{slug}"}
            e = _entry(cfg, f, typ, "polyhaven", m, excluded)
            if e:
                items.append(e)
    user = Path(fc["user_dir"])
    for type_dir in sorted(p for p in user.iterdir() if p.is_dir()) if user.exists() else []:
        typ = catalog_type(type_dir.name)
        if typ not in TYPES:
            excluded.append({"file": str(type_dir), "type": typ, "source": "user",
                             "reason": f"unknown furniture type folder (use one of: {', '.join(sorted(TYPES))})"})
            continue
        for f in sorted(p for p in type_dir.rglob("*") if p.suffix.lower() in (".glb", ".gltf")):
            m = _user_meta(f, type_dir)
            m.setdefault("slug", f"{typ}/{f.relative_to(type_dir).with_suffix('')}".replace("/", "_"))
            e = _entry(cfg, f, typ, "user", m, excluded)
            if e:
                items.append(e)
    for meta_f in sorted((lib / "generated").glob("*/*.json")):
        m = jload(meta_f)
        f = meta_f.with_suffix(".glb")
        if f.exists():
            e = _entry(cfg, f, m.get("type", meta_f.parent.name), "generated", m, excluded)
            if e:
                items.append(e)
    cov = {t: {s: sum(1 for i in items if i["type"] == t and i["source"] == s) for s in ("polyhaven", "user", "generated")}
           for t in TYPES}
    cat = {"version": 1, "built": time.strftime("%Y-%m-%d %H:%M:%S"), "items": items, "excluded": excluded, "coverage": cov}
    jsave(lib / "catalog.json", cat)
    write_licenses(lib, items, excluded)
    return cat


def write_licenses(lib, items, excluded):
    L = ["Furniture model licences (written by app/furniture.py - regenerate with: python -m app.furniture catalog)", "",
         "Poly Haven models are CC0 (https://polyhaven.com/license). They were downloaded once through the Poly Haven",
         "API (terms: https://github.com/Poly-Haven/Public-API/blob/master/ToS.md). Credit: Powered by Poly Haven.", "",
         "id | type | licence | source | author | file | sha256"]
    L += [f"{i['id']} | {i['type']} | {i['licence']} | {i['source_url'] or '-'} | {i['author'] or '-'} | {i['file']} | {i['sha256']}"
          for i in items]
    att = [i for i in items if i["licence"] in ATTRIBUTION]
    if att:
        L += ["", "Attribution required (CC-BY):"] + [f"  {i['name']} by {i['author'] or 'unknown'} - {i['source_url']} ({i['licence']})" for i in att]
    gen = [i for i in items if i["source"] == "generated"]
    if gen:
        L += ["", "Generated models (GEN3D): see each model's notes. TRELLIS.2 code/weights are MIT, but its GLB export",
              "uses NVIDIA nvdiffrast (NVIDIA Source Code License: non-commercial / research and evaluation only).",
              "Treat generated models as evaluation-only until you have cleared that licence."]
        L += [f"  {i['id']}: {i['notes']}" for i in gen]
    if excluded:
        L += ["", "Skipped files:"] + [f"  {e['file']}: {e['reason']}" for e in excluded]
    (Path(lib) / "LICENSES.txt").write_text("\n".join(L) + "\n", encoding="utf-8")


def load_catalog(cfg):
    cat = jload(lib_dir(cfg) / "catalog.json")
    if cat is None:        # never built (e.g. setup step skipped): scan what is on disk now - offline, ~1 s
        try:
            cat = build_catalog(cfg)
        except Exception as e:
            log(f"furniture catalog could not be built: {e}")
    return cat or {"items": [], "excluded": [], "coverage": {}}


# ---------- choosing a model per layout item ----------
def fit(entry, item, cfg):
    """How the model fits the layout slot. {'ok', 'err', 'scale': [sx, sy, sz], 'why'}."""
    fc = cfg["furniture"]
    tgt, dims = item["size"], entry["dims_m"]
    if entry.get("fit") == "height":   # uniform: slot height, smaller if the footprint would exceed 1.8 x the slot
        s = min(tgt[2] / max(dims[2], 1e-6), 1.8 * tgt[0] / max(dims[0], 1e-6), 1.8 * tgt[1] / max(dims[1], 1e-6))
        ok = dims[2] * s >= 0.5 * tgt[2] and 0.2 <= s <= 3.0
        return {"ok": ok, "err": round(abs(dims[2] * s / tgt[2] - 1) * 0.25 + abs(s - 1) * 0.05, 4),
                "scale": [round(s, 4)] * 3, "why": "" if ok else "too wide for the slot even at half its height"}
    s = [t / max(d, 1e-6) for t, d in zip(tgt, dims)]
    su = statistics.median(s)
    r = [x / su for x in s]
    worst = max(abs(x - 1) for x in r)
    ok_u, ok_n = abs(su - 1) <= fc["max_uniform_change"], worst <= fc["max_nonuniform"]
    why = "" if ok_u and ok_n else (f"non-uniform stretch {100 * worst:.0f} % > {100 * fc['max_nonuniform']:.0f} %"
                                    if not ok_n else f"size off by {100 * abs(su - 1):.0f} %")
    return {"ok": ok_u and ok_n, "err": round(worst + 0.25 * abs(su - 1), 4), "scale": [round(x, 4) for x in s], "why": why}


def style_score(entry, style_words):
    return len(style_words & set(entry.get("style_tags", [])))


def candidates(item, cat, cfg, style_words):
    out = []
    for e in cat["items"]:
        if e["type"] != catalog_type(item["type"]):
            continue
        f = fit(e, item, cfg)
        out.append({"asset_id": e["id"], "ok": f["ok"], "fit_err": f["err"], "why": f["why"], "scale": f["scale"],
                    "style": style_score(e, style_words), "polycount": e["polycount"], "source": e["source"],
                    "dims_m": e["dims_m"], "name": e["name"]})
    out.sort(key=lambda c: (not c["ok"], -c["style"], c["fit_err"], c["polycount"], c["asset_id"]))
    return out


def poly_cap(cfg, typ):
    fc = cfg["furniture"]
    return int(fc["poly_cap_by_type"].get(typ, fc["poly_cap"]))


def overrides_file(job):
    return Path(job) / "03_layout/asset_overrides.json"


def check_override(job, cfg, item_id, asset_id):
    """Is this agent choice valid (asset id, 'parametric' or 'remove')? Returns [] or a list of errors."""
    lay = jload(Path(job) / "03_layout/layout.json") or {}
    item = next((i for i in lay.get("items", []) if i["id"] == item_id), None)
    if item is None:
        return [f"no furniture item {item_id} in layout.json (see relayout / choose_furniture results)"]
    if asset_id not in ("parametric", "remove"):
        e = next((x for x in load_catalog(cfg)["items"] if x["id"] == asset_id), None)
        if e is None:
            return [f"no catalog model {asset_id}"]
        if e["type"] != catalog_type(item["type"]):
            return [f"{asset_id} is a {e['type']}, item {item_id} is a {item['type']}"]
        f = fit(e, item, cfg)
        if not f["ok"]:
            return [f"{asset_id} does not fit the slot {item['size']} m: {f['why']}"]
    return []


def set_override(job, cfg, item_id, asset_id):
    """Store an agent choice for one item. Returns [] or a list of errors (nothing stored then)."""
    errs = check_override(job, cfg, item_id, asset_id)
    if errs:
        return errs
    ov = jload(overrides_file(job), {})
    ov[item_id] = asset_id
    jsave(overrides_file(job), ov)
    return []


def select_assets(job, cfg=None, style=None):
    """Pick a model for every layout item -> 03_layout/assets.json (read by blender_scene.py)."""
    cfg = cfg or load_config()
    job = Path(job)
    lay = jload(job / "03_layout/layout.json") or {"items": []}
    cat = load_catalog(cfg) if cfg["furniture"]["enabled"] else {"items": []}
    ov = jload(overrides_file(job), {})
    style_words = words(style or ov.get("__style__") or cfg.get("style", ""))
    by_id = {e["id"]: e for e in cat["items"]}
    out = {}
    for it in lay["items"]:
        o = ov.get(it["id"])
        rec = {"type": it["type"], "room": it["room"], "size": it["size"]}
        if o == "remove":
            out[it["id"]] = dict(rec, source="removed", reason="removed by the agent")
            continue
        if o == "parametric" or it["type"] in PARAMETRIC_ONLY:
            out[it["id"]] = dict(rec, source="parametric", reason="agent choice" if o else "always parametric")
            continue
        pick = None
        if o in by_id:
            f = fit(by_id[o], it, cfg)
            pick = (by_id[o], f, True) if f["ok"] else None
        if pick is None:
            c = next((c for c in candidates(it, cat, cfg, style_words) if c["ok"]), None)
            if c:
                pick = (by_id[c["asset_id"]], fit(by_id[c["asset_id"]], it, cfg), False)
        if pick is None:
            n = sum(1 for e in cat["items"] if e["type"] == catalog_type(it["type"]))
            out[it["id"]] = dict(rec, source="parametric",
                                 reason=f"{n} catalog model(s) of this type, none fits the slot" if n else "no catalog model of this type")
            continue
        e, f, forced = pick
        out[it["id"]] = dict(rec, source=e["source"], asset_id=e["id"], file=e["file"], licence=e["licence"],
                             front_axis=e["front_axis"], unit_scale=e["unit_scale"], fit=e["fit"], scale=f["scale"],
                             fit_err=f["err"], style=style_score(e, style_words), polycount=e["polycount"],
                             poly_cap=poly_cap(cfg, it["type"]), override=forced)
    summ = {}
    for v in out.values():
        summ.setdefault(v["type"], {}).setdefault(v["source"], 0)
        summ[v["type"]][v["source"]] += 1
    res = {"items": out, "summary": summ, "style_words": sorted(style_words),
           "limits": {k: cfg["furniture"][k] for k in ("max_uniform_change", "max_nonuniform")}}
    jsave(job / "03_layout/assets.json", res)
    return res


def coverage(job):
    """Furniture type vs source actually used (after Blender QA) -> final/coverage.json + coverage.md."""
    job = Path(job)
    qa = jload(job / "04_scene/furniture_qa.json")
    planned = (jload(job / "03_layout/assets.json") or {}).get("items", {})
    rows = qa["items"] if qa else [{"item_id": k, "type": v["type"], "room": v["room"], "planned": v["source"],
                                    "used": v["source"], "reason": v.get("reason", "")} for k, v in planned.items()]
    table = {}
    for r in rows:
        table.setdefault(r["type"], {}).setdefault(r["used"], 0)
        table[r["type"]][r["used"]] += 1
    srcs = ["polyhaven", "user", "generated", "parametric", "skipped", "removed"]
    fallbacks = [r for r in rows if r["used"] in ("parametric", "skipped") and r.get("reason") not in (None, "", "always parametric")]
    out = {"from": "blender QA" if qa else "plan (scene not built yet)", "by_type": table, "fallbacks": fallbacks, "items": rows}
    (job / "final").mkdir(parents=True, exist_ok=True)
    jsave(job / "final/coverage.json", out)
    L = [f"# Furniture coverage - {job.name}", "", f"Source: {out['from']}", "",
         "| Type | " + " | ".join(srcs) + " |", "|---|" + "---|" * len(srcs)]
    L += [f"| {t} | " + " | ".join(str(c.get(s, "")) for s in srcs) + " |" for t, c in sorted(table.items())]
    if fallbacks:
        L += ["", "## Fallbacks (parametric model, or skipped for plants)", ""] + [
            f"- {r['item_id']} ({r['type']}) -> {r['used']}: {r['reason']}" for r in fallbacks]
    (job / "final/coverage.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    return out


def report(cfg=None):
    cat = load_catalog(cfg or load_config())
    cov = cat.get("coverage", {})
    print(f"{'type':<14} polyhaven  user  generated")
    for t in TYPES:
        c = cov.get(t, {})
        print(f"{t:<14} {c.get('polyhaven', 0):>9} {c.get('user', 0):>5} {c.get('generated', 0):>10}")
    none = [t for t in TYPES if not sum(cov.get(t, {}).values())]
    print(f"models: {len(cat['items'])}, skipped files: {len(cat['excluded'])}; types without any model "
          f"(parametric boxes are used): {', '.join(none) or 'none'}")
    for e in cat["excluded"][:20]:
        print(f"  skipped {e['file']}: {e['reason']}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "report"
    if cmd == "fetch":
        fetch_polyhaven()
        report()
    elif cmd == "catalog":
        build_catalog()
        report()
    else:
        report()
