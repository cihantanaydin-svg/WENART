"""Setup step - one-time download of CC0 assets: ambientCG textures, Poly Haven sky HDRIs and a few
plant/lamp models. Every download is optional; the scene falls back to plain colours and a sky colour."""
import io, json, sys, urllib.request, zipfile
from pathlib import Path
import cv2
import numpy as np
from .common import WS, log

UA = {"User-Agent": "floorplan-poc-setup/1.0 (Runpod proof of concept; one-time download)"}
AMBIENT = {"WoodFloor051": "2K", "Fabric061": "1K", "Wood049": "1K", "Marble012": "1K", "Carpet016": "1K",
           "PaintedPlaster017": "1K"}
MODELS = {"plant": ["plant", "potted"], "floor_lamp": ["floor lamp", "floor_lamp", "standing lamp"]}


def get(url, timeout=300):
    return urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout).read()


def ph(path):
    return json.loads(get("https://api.polyhaven.com" + path, 120))


def sun_position(p):
    """Brightest spot of an equirectangular sky = sun (azimuth, elevation in degrees, Blender convention)."""
    img = cv2.imread(str(p), cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR).astype(np.float32)
    lum = cv2.GaussianBlur(img.mean(2), (0, 0), 3)
    h, w = lum.shape
    y, x = np.unravel_index(np.argmax(lum[: h // 2]), lum[: h // 2].shape)
    return {"sun_az_deg": round(180.0 * (1 - 2 * (x + 0.5) / w), 2), "sun_el_deg": round(90.0 * (1 - 2 * (y + 0.5) / h), 2)}


def main(dst=WS / "assets"):
    dst = Path(dst)
    for aid, res in AMBIENT.items():
        d = dst / "materials" / aid
        if d.exists() and any(d.iterdir()):
            continue
        try:
            zipfile.ZipFile(io.BytesIO(get(f"https://ambientcg.com/get?file={aid}_{res}-JPG.zip"))).extractall(d)
            log(f"ambientCG {aid}: ok")
        except Exception as e:
            log(f"WARNING ambientCG {aid}: {e} (plain colour will be used)")
    hd = dst / "hdri"
    hd.mkdir(parents=True, exist_ok=True)
    if not list(hd.glob("*.hdr")):
        try:
            skies = ph("/assets?type=hdris&categories=skies")
            for slug, info in sorted(skies.items(), key=lambda kv: -kv[1].get("download_count", 0))[:2]:
                p = hd / f"{slug}_2k.hdr"
                p.write_bytes(get(ph(f"/files/{slug}")["hdri"]["2k"]["hdr"]["url"]))
                p.with_suffix(".json").write_text(json.dumps(sun_position(p)))
                log(f"Poly Haven HDRI {slug}: ok")
        except Exception as e:
            log(f"WARNING Poly Haven HDRIs: {e} (a plain sky colour will be used)")
    try:
        models = ph("/assets?type=models")
        for typ, keys in MODELS.items():
            d = dst / "models" / typ
            if d.exists() and list(d.rglob("*.gltf")):
                continue
            cand = sorted(((info.get("download_count", 0), slug) for slug, info in models.items()
                           if any(k in " ".join([slug, info.get("name", ""), *info.get("tags", []), *info.get("categories", [])]).lower()
                                  for k in keys)), reverse=True)
            for _, slug in cand[: 3 if typ == "plant" else 1]:
                g = ph(f"/files/{slug}")["gltf"]["1k"]["gltf"]
                sd = d / slug
                sd.mkdir(parents=True, exist_ok=True)
                (sd / f"{slug}.gltf").write_bytes(get(g["url"]))
                for rel, inc in (g.get("include") or {}).items():
                    q = sd / rel
                    q.parent.mkdir(parents=True, exist_ok=True)
                    q.write_bytes(get(inc["url"]))
                log(f"Poly Haven model {typ}: {slug}")
    except Exception as e:
        log(f"WARNING Poly Haven models: {e} (plants are skipped, lamps use a simple model)")
    (dst / "LICENSES.txt").write_text("ambientCG textures: CC0 (https://ambientcg.com)\n"
                                      "Poly Haven HDRIs/models: CC0 (https://polyhaven.com/license). Downloaded once via the "
                                      "Poly Haven API for this proof of concept; check the API terms before commercial use.\n")


if __name__ == "__main__":
    main(*(sys.argv[1:2]))
