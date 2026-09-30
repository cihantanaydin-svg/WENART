"""Optional wall/door/window model for scans and phone photos (section 9, finetune.sh).
Teacher data = your own DWG/DXF/vector-PDF plans: the pipeline reads them exactly, the correct labels are
drawn from plan.json, and training turns the clean pages into fake scans/photos on the fly.
  python -m app.wallseg prepare --sources DIR --data DIR      image + label pairs (CPU)
  python -m app.wallseg train --data DIR --run DIR ...        train, or resume after a stop (GPU)
  python -m app.wallseg install --run DIR --min-iou 0.8       switch the pipeline to the new model
  python -m app.wallseg predict JOB                           used by main.py for scans and photos
  python -m app.wallseg compare --run DIR --inputs DIR        before/after table on raster test plans
Labels: 0 background, 1 wall, 2 door opening, 3 window. Small U-Net trained from zero: no third-party
weights, so no extra licence. The pipeline uses the wall class; doors/windows teach it where the gaps are."""
import argparse, csv, hashlib, json, math, os, re, shutil, subprocess, sys, time
from pathlib import Path
import cv2
import numpy as np
from .common import WS, jload, jsave, load_config, log

RASTER = ("pdf_scan", "photo")
COLORS = np.array([[0, 0, 0], [40, 40, 230], [40, 170, 40], [230, 90, 30]], np.uint8)   # BGR: wall red, door green, window blue


# ---------- labels from plan.json ----------
def draw_mask(plan, shape):
    a, _, c, _, e, f = plan["debug"]["m_to_px"]
    m = np.zeros(shape[:2], np.uint8)
    walls = {w["id"]: w for w in plan["walls"]}

    def rect(cx, cy, ux, uy, hl, ht):
        pts = [(a * (cx + su * ux * hl - sn * uy * ht) + c, e * (cy + su * uy * hl + sn * ux * ht) + f)
               for su, sn in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
        return np.round(np.array(pts)).astype(np.int32)

    def seg_dist(p, w):
        (x0, y0), (x1, y1) = w["a"], w["b"]
        L2 = (x1 - x0) ** 2 + (y1 - y0) ** 2 or 1e-12
        t = max(0.0, min(1.0, ((p[0] - x0) * (x1 - x0) + (p[1] - y0) * (y1 - y0)) / L2))
        return math.hypot(p[0] - x0 - t * (x1 - x0), p[1] - y0 - t * (y1 - y0))

    for w in plan["walls"]:
        (x0, y0), (x1, y1) = w["a"], w["b"]
        L = math.hypot(x1 - x0, y1 - y0) or 1e-9
        ux, uy, ht = (x1 - x0) / L, (y1 - y0) / L, w["thickness"] / 2
        # extend an end by half a thickness only if the extension stays inside another wall (fills L-corners,
        # never sticks out of an outer wall and keeps free ends exact)
        ext = [ht if any(o is not w and seg_dist((p[0] + sg * ux * ht, p[1] + sg * uy * ht), o) <= o["thickness"] / 2 + 0.01
                         for o in plan["walls"]) else 0.0 for p, sg in ((w["a"], -1), (w["b"], 1))]
        s0, s1 = -ext[0], L + ext[1]
        cv2.fillPoly(m, [rect(x0 + ux * (s0 + s1) / 2, y0 + uy * (s0 + s1) / 2, ux, uy, (s1 - s0) / 2, ht)], 1)
    for o in plan["doors"] + plan["windows"]:
        w = walls.get(o["wall_id"])
        if not w:
            continue
        (x0, y0), (x1, y1) = w["a"], w["b"]
        L = math.hypot(x1 - x0, y1 - y0) or 1e-9
        val = 3 if o in plan["windows"] and o.get("kind") != "balcony_door" else 2
        cv2.fillPoly(m, [rect(o["center"][0], o["center"][1], (x1 - x0) / L, (y1 - y0) / L, o["width"] / 2, w["thickness"] / 2 + 0.01)], val)
    return m


def colorize(gray, lab):
    out = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    fg = lab > 0
    out[fg] = (0.45 * out[fg] + 0.55 * COLORS[lab[fg]]).astype(np.uint8)
    return out


def prepare(sources, data, min_sources):
    sources, data = Path(sources), Path(data)
    (data / "pairs").mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in sources.rglob("*") if p.suffix.lower() in (".dwg", ".dxf", ".pdf"))
    log(f"prepare: {len(files)} source files in {sources}")
    for f in files:
        name = re.sub(r"[^A-Za-z0-9_-]+", "_", f.stem)[:50] + "_" + hashlib.md5(str(f.resolve()).encode()).hexdigest()[:6]
        d = data / "pairs" / name
        if (d / "meta.json").exists():
            continue   # done in an earlier run
        d.mkdir(parents=True, exist_ok=True)
        job, t0 = data / "jobs" / name, time.time()
        r = subprocess.run([sys.executable, "-m", "app.main", str(f), "--job-dir", str(job), "--to-stage", "plan",
                            "--wallseg", "off"], cwd=str(WS), capture_output=True, text=True)
        plan = jload(job / "02_plan/plan.json")
        meta = {"name": name, "source": str(f), "ok": False}
        if r.returncode or not plan:
            meta["reason"] = "pipeline failed: " + (r.stdout + r.stderr).strip()[-300:]
        else:
            kind = plan["source"]["kind"]
            lab = [x for x in plan["rooms"] if x.get("label_area_m2")]
            good = [x for x in lab if abs(x["area_m2"] / x["label_area_m2"] - 1) <= 0.03]
            meta.update(kind=kind, walls=len(plan["walls"]), rooms=len(plan["rooms"]), labeled_rooms=len(lab), matching_rooms=len(good))
            if kind not in ("dxf", "pdf_vector"):
                meta["reason"] = f"{kind} is not a vector source, so there are no exact labels"
            elif len(plan["walls"]) < 6 or len(plan["rooms"]) < 2:
                meta["reason"] = "too few walls or rooms found"
            elif kind == "pdf_vector" and (len(lab) < 2 or len(good) < 0.7 * len(lab)):
                meta["reason"] = "room sizes do not match the m2 labels, so the walls are not trusted"
            else:
                bg = cv2.imread(str(job / plan["debug"]["background"]), cv2.IMREAD_GRAYSCALE)
                mask = draw_mask(plan, bg.shape)
                cv2.imwrite(str(d / "image.png"), bg)
                cv2.imwrite(str(d / "mask.png"), mask)
                k = min(1.0, 1600 / max(bg.shape))
                cv2.imwrite(str(d / "preview.jpg"), cv2.resize(colorize(bg, mask), None, fx=k, fy=k, interpolation=cv2.INTER_AREA))
                a = abs(plan["debug"]["m_to_px"][0])
                meta.update(ok=True, wall_px=round(float(np.median([w["thickness"] * a for w in plan["walls"]])), 2),
                            size=list(bg.shape), labels={c: int((mask == i).sum()) for i, c in ((1, "wall"), (2, "door"), (3, "window"))})
        meta["seconds"] = round(time.time() - t0, 1)
        jsave(d / "meta.json", meta)
        log(f"  {f.name}: {'OK' if meta['ok'] else 'skipped - ' + meta['reason']}")
    metas = [jload(p) for p in sorted((data / "pairs").glob("*/meta.json"))]
    ok = [m for m in metas if m.get("ok")]
    for m in ok:   # stable split by plan (never by tile), about 15 % for validation
        m["split"] = "val" if int(hashlib.md5(m["name"].encode()).hexdigest(), 16) % 100 < 15 else "train"
    if ok and not any(m["split"] == "val" for m in ok):
        ok[-1]["split"] = "val"
    if len(ok) > 1 and not any(m["split"] == "train" for m in ok):
        ok[0]["split"] = "train"
    jsave(data / "index.json", {"pairs": metas, "usable": len(ok),
                                "train": sum(m["split"] == "train" for m in ok), "val": sum(m["split"] == "val" for m in ok)})
    print(f"Usable plans: {len(ok)} of {len(metas)} (train {sum(m['split'] == 'train' for m in ok)}, "
          f"val {sum(m['split'] == 'val' for m in ok)}). Check the labels: {data}/pairs/*/preview.jpg")
    if len(ok) < min_sources:
        print(f"Too few usable plans ({len(ok)} < {min_sources}). Add more DWG/DXF/vector PDF files to {sources}.")
        return 2
    return 0


# ---------- fake scans and photos from clean pages ----------
def sample_tile(img, msk, fg, wall_px, norm_px, T, rng, jitter=True):
    s = norm_px / max(wall_px, 1e-3) * (rng.uniform(0.7, 1.4) if jitter else 1.0)
    ang = math.radians(rng.uniform(-3, 3)) if jitter else 0.0
    if len(fg) and rng.random() < 0.85:
        cy, cx = fg[rng.integers(len(fg))] + rng.normal(0, T / (5 * s), 2)
    else:
        cy, cx = rng.uniform(0, img.shape[0]), rng.uniform(0, img.shape[1])
    ca, sa = math.cos(ang) / s, math.sin(ang) / s
    M = np.array([[ca, -sa, cx - ca * T / 2 + sa * T / 2], [sa, ca, cy - sa * T / 2 - ca * T / 2]], np.float32)
    x = cv2.warpAffine(img, M, (T, T), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderValue=255)
    y = cv2.warpAffine(msk, M, (T, T), flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP, borderValue=0)
    if jitter and rng.random() < 0.3:   # slight perspective, like a phone photo after correction
        src = np.float32([[0, 0], [T, 0], [T, T], [0, T]])
        H = cv2.getPerspectiveTransform(src, src + rng.uniform(-0.03, 0.03, (4, 2)).astype(np.float32) * T)
        x = cv2.warpPerspective(x, H, (T, T), flags=cv2.INTER_LINEAR, borderValue=255)
        y = cv2.warpPerspective(y, H, (T, T), flags=cv2.INTER_NEAREST, borderValue=0)
    return x, y


def augment(x, y, rng):
    T = x.shape[0]
    wall = y == 1
    r = rng.random()
    if r < 0.25:     # walls drawn filled
        x[wall] = rng.integers(0, 90)
    elif r < 0.45:   # walls drawn hatched
        yy, xx = np.mgrid[:T, :T]
        per = int(rng.integers(5, 13))
        h = (((xx + yy) if rng.random() < 0.5 else (xx - yy)) % per) < max(1, per // 4)
        x[wall] = 235
        x[wall & h] = rng.integers(0, 80)
        edge = wall & ~cv2.erode(wall.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
        x[edge] = 20
    elif r < 0.6:    # walls drawn as two lines only
        x[cv2.erode(wall.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)] = 240
    x = x.astype(np.float32) * rng.uniform(0.75, 1.1) + rng.uniform(-25, 20)
    if rng.random() < 0.5:   # uneven light
        x *= cv2.resize(rng.uniform(0.55, 1.0, (3, 3)).astype(np.float32), (T, T), interpolation=cv2.INTER_CUBIC)
    if rng.random() < 0.6:
        x = cv2.GaussianBlur(x, (0, 0), rng.uniform(0.3, 1.5))
    if rng.random() < 0.2:   # low-resolution scan
        k = rng.uniform(0.35, 0.7)
        x = cv2.resize(cv2.resize(x, None, fx=k, fy=k, interpolation=cv2.INTER_AREA), (T, T), interpolation=cv2.INTER_LINEAR)
    if rng.random() < 0.6:
        x += rng.normal(0, rng.uniform(2, 10), x.shape).astype(np.float32)
    x = np.clip(x, 0, 255).astype(np.uint8)
    if rng.random() < 0.5:
        x = cv2.imdecode(cv2.imencode(".jpg", x, [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(30, 95))])[1], cv2.IMREAD_GRAYSCALE)
    if rng.random() < 0.08:  # black-and-white scan
        x = cv2.threshold(x, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    return x


def load_pairs(pairs, data):
    out = []
    for m in pairs:
        d = Path(data) / "pairs" / m["name"]
        img, msk = cv2.imread(str(d / "image.png"), cv2.IMREAD_GRAYSCALE), cv2.imread(str(d / "mask.png"), cv2.IMREAD_GRAYSCALE)
        fg = np.argwhere(msk > 0)
        if len(fg) > 20000:
            fg = fg[np.random.default_rng(0).choice(len(fg), 20000, replace=False)]
        out.append((img, msk, fg.astype(np.float32), m["wall_px"]))
    return out


class TileSet:
    """Endless random fake-scan tiles (map-style dataset for torch DataLoader)."""
    def __init__(self, pairs, data, T, n, seed, norm_px):
        self.items, self.T, self.n, self.seed, self.norm_px = load_pairs(pairs, data), T, n, seed, norm_px
        w = np.array([it[0].size for it in self.items], float)
        self.p = w / w.sum()

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        rng = np.random.default_rng([self.seed, i])
        img, msk, fg, wpx = self.items[rng.choice(len(self.items), p=self.p)]
        x, y = sample_tile(img, msk, fg, wpx, self.norm_px, self.T, rng)
        x = augment(x, y, rng)
        return (x.astype(np.float32) / 127.5 - 1.0)[None], y.astype(np.int64)


def val_tiles(pairs, data, T, norm_px, per_plan=24):
    key = hashlib.md5(json.dumps([sorted(m["name"] for m in pairs), T, round(norm_px, 2)]).encode()).hexdigest()[:10]
    f = Path(data) / f"val_{key}.npz"
    if f.exists():
        z = np.load(f)
        return z["x"], z["y"]
    xs, ys = [], []
    for j, (img, msk, fg, wpx) in enumerate(load_pairs(pairs, data)):
        for i in range(per_plan):
            rng = np.random.default_rng([4242, j, i])
            x, y = sample_tile(img, msk, fg, wpx, norm_px, T, rng)
            xs.append(augment(x, y, rng))
            ys.append(y)
    np.savez_compressed(f, x=np.stack(xs), y=np.stack(ys))
    return np.stack(xs), np.stack(ys)


# ---------- model ----------
def build_unet(base=32, n_cls=4):
    import torch
    import torch.nn as nn

    def block(ci, co):
        return nn.Sequential(nn.Conv2d(ci, co, 3, padding=1, bias=False), nn.BatchNorm2d(co), nn.ReLU(inplace=True),
                             nn.Conv2d(co, co, 3, padding=1, bias=False), nn.BatchNorm2d(co), nn.ReLU(inplace=True))

    class UNet(nn.Module):
        def __init__(self):
            super().__init__()
            c = [base * 2 ** i for i in range(5)]
            self.enc = nn.ModuleList([block(1, c[0])] + [block(c[i], c[i + 1]) for i in range(4)])
            self.up = nn.ModuleList([nn.ConvTranspose2d(c[i + 1], c[i], 2, stride=2) for i in reversed(range(4))])
            self.dec = nn.ModuleList([block(2 * c[i], c[i]) for i in reversed(range(4))])
            self.head = nn.Conv2d(c[0], n_cls, 1)

        def forward(self, x):
            skips = []
            for i, e in enumerate(self.enc):
                x = e(x if i == 0 else nn.functional.max_pool2d(x, 2))
                skips.append(x)
            x = skips.pop()
            for up, dec in zip(self.up, self.dec):
                x = dec(torch.cat([up(x), skips.pop()], 1))
            return self.head(x)

    return UNet()


def seg_loss(logits, y, w):
    import torch.nn.functional as F
    ce = F.cross_entropy(logits, y, weight=w)
    p = logits.softmax(1)
    oh = F.one_hot(y, logits.shape[1]).permute(0, 3, 1, 2).float()
    inter, den = (p * oh).sum((0, 2, 3)), p.sum((0, 2, 3)) + oh.sum((0, 2, 3))
    return ce + 1 - ((2 * inter + 1) / (den + 1))[1:].mean()


def predict_full(model, img, T, dev):
    import torch
    H, W = img.shape
    if H < T or W < T:
        img = cv2.copyMakeBorder(img, 0, max(0, T - H), 0, max(0, T - W), cv2.BORDER_CONSTANT, value=255)
    Hp, Wp = img.shape
    st = T - T // 8
    ys = sorted(set(list(range(0, Hp - T + 1, st)) + [Hp - T]))
    xs = sorted(set(list(range(0, Wp - T + 1, st)) + [Wp - T]))
    acc, cnt = np.zeros((4, Hp, Wp), np.float32), np.zeros((Hp, Wp), np.float32)
    x = img.astype(np.float32) / 127.5 - 1.0
    coords = [(yy, xx) for yy in ys for xx in xs]
    model.eval()
    with torch.inference_mode():
        for i in range(0, len(coords), 8):
            chunk = coords[i:i + 8]
            b = torch.from_numpy(np.stack([x[yy:yy + T, xx:xx + T] for yy, xx in chunk])[:, None]).to(dev)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=dev == "cuda"):
                p = model(b).float().softmax(1).cpu().numpy()
            for (yy, xx), pp in zip(chunk, p):
                acc[:, yy:yy + T, xx:xx + T] += pp
                cnt[yy:yy + T, xx:xx + T] += 1
    return (acc / np.maximum(cnt, 1))[:, :H, :W]


def validate(model, vx, vy, dev):
    import torch
    conf = np.zeros((4, 4), np.int64)
    model.eval()
    with torch.inference_mode():
        for i in range(0, len(vx), 16):
            b = torch.from_numpy(vx[i:i + 16].astype(np.float32) / 127.5 - 1.0)[:, None].to(dev)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=dev == "cuda"):
                pred = model(b).argmax(1).cpu().numpy()
            conf += np.bincount(4 * vy[i:i + 16].astype(np.int64).ravel() + pred.ravel(), minlength=16).reshape(4, 4)
    inter = np.diag(conf).astype(float)
    union = conf.sum(0) + conf.sum(1) - inter
    iou = {c: round(float(inter[i] / union[i]) if union[i] else 0.0, 4) for i, c in ((1, "wall"), (2, "door"), (3, "window"))}
    iou["mean"] = round(float(np.mean([iou["wall"], iou["door"], iou["window"]])), 4)
    return iou


def samples(model, vx, vy, dev, out, step, real_dir):
    import torch
    idx = sorted({0, len(vx) // 4, len(vx) // 2, 3 * len(vx) // 4})
    with torch.inference_mode():
        b = torch.from_numpy(vx[idx].astype(np.float32) / 127.5 - 1.0)[:, None].to(dev)
        pred = model(b).argmax(1).cpu().numpy().astype(np.uint8)
    rows = [np.hstack([cv2.cvtColor(vx[i], cv2.COLOR_GRAY2BGR), colorize(vx[i], vy[i]), colorize(vx[i], p)]) for i, p in zip(idx, pred)]
    cv2.imwrite(str(out / f"step_{step:06d}.jpg"), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85])
    for f in sorted(Path(real_dir).glob("*"))[:2] if real_dir and Path(real_dir).exists() else []:
        if f.suffix.lower() not in (".png", ".jpg", ".jpeg"):
            continue
        g = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
        k = min(1.0, 2048 / max(g.shape))
        g = cv2.resize(g, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
        lab = predict_full(model, g, vx.shape[1], dev).argmax(0).astype(np.uint8)
        cv2.imwrite(str(out / f"step_{step:06d}_real_{f.stem}.jpg"), colorize(g, lab), [cv2.IMWRITE_JPEG_QUALITY, 85])


def train(a):
    import torch
    from torch.utils.data import DataLoader
    data, run = Path(a.data), Path(a.run)
    pairs = [m for m in (jload(data / "index.json") or {}).get("pairs", []) if m.get("ok")]
    tr, va = [m for m in pairs if m.get("split") == "train"], [m for m in pairs if m.get("split") == "val"]
    if not tr or not va:
        print("Need at least one training and one validation plan: run the prepare step first.")
        return 2
    if a.fresh and run.exists():
        run.rename(run.with_name(run.name + time.strftime("_old_%Y%m%d_%H%M%S")))
    (run / "samples").mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(a.seed)
    norm_px = float(np.median([m["wall_px"] for m in tr]))
    model = build_unet(a.base_ch).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    warm = min(500, a.steps // 10 + 1)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(s, a.steps) / a.steps)))
    step, best = 0, -1.0
    if (run / "last.pt").exists():
        ck = torch.load(run / "last.pt", map_location=dev, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        step, best = ck["step"], ck["best"]
        log(f"train: resuming {run.name} from step {step} (best wall IoU {best:.3f})")
    meta = {"base": a.base_ch, "tile": a.tile, "norm_px": norm_px, "classes": ["background", "wall", "door", "window"]}
    jsave(run / "config.json", dict(vars(a), **meta, train=[m["name"] for m in tr], val=[m["name"] for m in va], device=dev))
    if step >= a.steps:
        log(f"train: {run.name} already finished ({step} steps)")
        return 0
    vx, vy = val_tiles(va, data, a.tile, norm_px)
    ds = TileSet(tr, data, a.tile, (a.steps - step) * a.batch, a.seed + 7919 * step, norm_px)
    dl = DataLoader(ds, batch_size=a.batch, num_workers=a.workers, pin_memory=dev == "cuda", drop_last=True,
                    persistent_workers=a.workers > 0, prefetch_factor=4 if a.workers > 0 else None)
    wts = torch.tensor([0.5, 1.0, 3.0, 3.0], device=dev)
    log(f"train: {len(tr)} train plans, {len(va)} val plans ({len(vx)} val tiles), device {dev}, steps {step}->{a.steps}")
    t0, run_loss, n_loss = time.time(), 0.0, 0
    model.train()
    with open(run / "log.csv", "a", newline="") as fcsv:
        wr = csv.writer(fcsv)
        for x, y in dl:
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=dev == "cuda"):
                out = model(x)
            loss = seg_loss(out.float(), y, wts)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            run_loss, n_loss = run_loss + loss.item(), n_loss + 1
            if step % 50 == 0 or step == a.steps:
                log(f"step {step}/{a.steps} loss {run_loss / n_loss:.4f} lr {sched.get_last_lr()[0]:.2e} "
                    f"{n_loss / (time.time() - t0):.1f} steps/s")
                wr.writerow([step, round(run_loss / n_loss, 5), "", "", "", ""])
                t0, run_loss, n_loss = time.time(), 0.0, 0
            if step % a.val_every == 0 or step == a.steps:
                iou = validate(model, vx, vy, dev)
                samples(model, vx, vy, dev, run / "samples", step, a.real)
                wr.writerow([step, "", iou["wall"], iou["door"], iou["window"], iou["mean"]])
                fcsv.flush()
                if iou["wall"] > best:
                    best = iou["wall"]
                    torch.save({"model": model.state_dict(), "meta": dict(meta, step=step, val=iou)}, run / "best.pt")
                log(f"validation step {step}: IoU wall {iou['wall']:.3f} door {iou['door']:.3f} window {iou['window']:.3f}"
                    f" (best wall {best:.3f}) -> samples/step_{step:06d}.jpg")
                model.train()
            if step % a.ckpt_every == 0 or step == a.steps:
                torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                            "step": step, "best": best}, run / "last.pt.tmp")
                os.replace(run / "last.pt.tmp", run / "last.pt")
            if step >= a.steps:
                break
    log(f"train: finished {step} steps, best wall IoU {best:.3f}")
    return 0


def install(run, min_iou):
    import torch
    run = Path(run)
    if not (run / "best.pt").exists():
        print(f"No {run}/best.pt yet - train first.")
        return 2
    meta = torch.load(run / "best.pt", map_location="cpu", weights_only=False)["meta"]
    v = meta["val"]
    print(f"Best checkpoint: step {meta['step']}, validation IoU walls {v['wall']:.3f}, doors {v['door']:.3f}, windows {v['window']:.3f}")
    if v["wall"] < min_iou:
        print(f"NOT switched on: wall IoU {v['wall']:.3f} is below {min_iou}. Add more source plans or train longer.")
        return 3
    dst = WS / "models/wallseg"
    dst.mkdir(parents=True, exist_ok=True)
    shutil.copy2(run / "best.pt", dst / f"{run.name}.pt")
    shutil.copy2(run / "best.pt", dst / "model.pt")
    p = WS / "config.json"
    cfg = json.loads(p.read_text()) if p.exists() else {}
    cfg["wallseg"] = {"enabled": True, "model": str(dst / "model.pt"), "run": run.name}
    p.write_text(json.dumps(cfg, indent=1))
    print(f"Switched on: {dst / 'model.pt'} (config.json wallseg.enabled = true)")
    return 0


# ---------- use in the pipeline ----------
def mask_thickness(mask):
    b = (mask > 0).astype(np.uint8)
    if b.sum() < 500:
        return None
    dt = cv2.distanceTransform(b, cv2.DIST_L2, 5)
    ridge = (dt >= 2) & (dt >= cv2.dilate(dt, np.ones((3, 3), np.uint8)) - 1e-6)
    return float(2 * np.median(dt[ridge])) if ridge.sum() > 50 else None


def predict(job):
    import torch
    job = Path(job)
    mp = Path(os.environ.get("PIPE_WALLSEG_MODEL") or load_config()["wallseg"]["model"])
    if not mp.exists():
        raise SystemExit(f"wall model not found: {mp}")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(mp, map_location="cpu", weights_only=False)
    meta = ck["meta"]
    model = build_unet(meta["base"])
    model.load_state_dict(ck["model"])
    model.to(dev)
    page = cv2.imread(str(job / "01_parse/page.png"), cv2.IMREAD_GRAYSCALE)
    cur, backup = job / "01_parse/wall_mask.png", job / "01_parse/wall_mask_classic.png"
    if not backup.exists():
        shutil.copy2(cur, backup)
    classic = cv2.imread(str(backup), cv2.IMREAD_GRAYSCALE)
    t = mask_thickness(classic)
    s = float(np.clip(meta["norm_px"] / t, 0.5, 2.0)) if t else 1.0
    img = page if abs(s - 1) < 0.02 else cv2.resize(page, None, fx=s, fy=s, interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
    lab = predict_full(model, img, meta["tile"], dev).argmax(0).astype(np.uint8)
    if lab.shape != page.shape:
        lab = cv2.resize(lab, (page.shape[1], page.shape[0]), interpolation=cv2.INTER_NEAREST)
    walls = cv2.morphologyEx(((lab == 1) * 255).astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, cc, st, _ = cv2.connectedComponentsWithStats(walls, 8)
    tpx = t or meta["norm_px"] / s
    keep = np.zeros(n, bool)
    keep[1:] = st[1:, cv2.CC_STAT_AREA] >= 4 * tpx * tpx
    walls = (keep[cc] * 255).astype(np.uint8)
    ratio = float(walls.sum()) / max(float(classic.sum()), 1.0)
    info = {"model": str(mp), "scale": round(s, 3), "wall_px_classic": t, "ratio_to_classic": round(ratio, 3)}
    if 0.33 <= ratio <= 3.0:
        cv2.imwrite(str(cur), walls)
        info["used"] = True
    else:
        shutil.copy2(backup, cur)
        info.update(used=False, reason="model mask very different from the classic mask - classic mask kept")
    k = min(1.0, 2400 / max(page.shape))
    cv2.imwrite(str(job / "01_parse/wallseg_pred.jpg"), cv2.resize(colorize(page, lab), None, fx=k, fy=k, interpolation=cv2.INTER_AREA))
    jsave(job / "01_parse/wallseg.json", info)
    log(f"wallseg: {'used' if info['used'] else 'NOT used'} (scale {s:.2f}, wall pixels x{ratio:.2f} vs classic)")
    return 0


def compare(run, inputs, tests):
    from .evaluate import match
    run, out = Path(run), WS / "finetune/compare" / Path(run).name
    truth = {}
    tf = Path(tests) / "truth.csv"
    if tf.exists():
        for row in csv.DictReader(open(tf, encoding="utf-8-sig")):
            if row.get("plan") and row.get("width_m"):
                truth.setdefault(row["plan"].strip(), []).append(row)
    files = list({f.resolve(): f for d in (Path(inputs), WS / "inputs/smoke_test") if d.exists()
                  for f in sorted(d.iterdir()) if f.name in truth}.values())
    if not files:
        print(f"No test plans with truth rows found ({tf}). Skipping before/after.")
        return 0
    lines = ["| Plan | Kind | Base: rooms OK | Base: worst | Fine-tuned: rooms OK | Fine-tuned: worst | Verdict |",
             "|---|---|---|---|---|---|---|"]
    for f in files:
        res = {}
        for variant, flag in (("base", "off"), ("finetuned", "on")):
            job = out / f.stem / variant
            vt = out / f.stem / "base/01_parse/vlm_texts.json"
            if variant == "finetuned" and vt.exists():   # same text reading for both, so only the walls differ
                (job / "01_parse").mkdir(parents=True, exist_ok=True)
                shutil.copy2(vt, job / "01_parse/vlm_texts.json")
            env = dict(os.environ, PIPE_WALLSEG_MODEL=str(run / "best.pt"))
            r = subprocess.run([sys.executable, "-m", "app.main", str(f), "--job-dir", str(job), "--to-stage", "plan",
                                "--wallseg", flag], cwd=str(WS), env=env, capture_output=True, text=True)
            plan = jload(job / "02_plan/plan.json")
            if r.returncode or not plan:
                res[variant] = None
                continue
            kind = plan["source"]["kind"]
            m = match(truth[f.name], plan["rooms"], kind)
            worst = max((x[3] for x in m if x[3]), key=lambda e: e[0], default=None)
            res[variant] = (sum(1 for x in m if x[3] and x[3][1]), len(m), worst, kind)
            if kind not in RASTER:
                break   # the model is only used for scans and photos
        b, ft = res.get("base"), res.get("finetuned")
        cell = lambda v: ("failed", "-") if v is None else (f"{v[0]}/{v[1]}", v[2][2] if v[2] else "-")
        if b and b[3] not in RASTER:
            lines.append(f"| {f.name} | {b[3]} | {cell(b)[0]} | {cell(b)[1]} | (not used for vector input) | | same |")
            continue
        verdict = "better" if ft and (not b or ft[0] > b[0] or (ft[0] == b[0] and ft[2] and b[2] and ft[2][0] < b[2][0] - 0.1)) else \
                  "worse" if b and (not ft or ft[0] < b[0] or (ft[2] and b[2] and ft[2][0] > b[2][0] + 0.1)) else "same"
        lines.append(f"| {f.name} | {(b or ft or (0, 0, 0, '?'))[3]} | {cell(b)[0]} | {cell(b)[1]} | {cell(ft)[0]} | {cell(ft)[1]} | **{verdict}** |")
    out.mkdir(parents=True, exist_ok=True)
    (out / "compare.md").write_text(f"# Before / after - {run.name}\n\n" + "\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines) + f"\n\nWritten: {out / 'compare.md'}")
    return 0


def main():
    ap = argparse.ArgumentParser(description="wall/door/window model for scans and photos")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--sources", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--min-sources", type=int, default=20)
    p = sub.add_parser("train")
    p.add_argument("--data", required=True)
    p.add_argument("--run", required=True)
    for k, v in (("steps", 20000), ("batch", 16), ("tile", 512), ("base-ch", 32), ("val-every", 1000), ("ckpt-every", 500),
                 ("workers", 8), ("seed", 1234), ("fresh", 0)):
        p.add_argument(f"--{k}", type=int, default=v)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--real", default="")
    p = sub.add_parser("install")
    p.add_argument("--run", required=True)
    p.add_argument("--min-iou", type=float, default=0.8)
    p = sub.add_parser("predict")
    p.add_argument("job")
    p = sub.add_parser("compare")
    p.add_argument("--run", required=True)
    p.add_argument("--inputs", default=str(WS / "inputs/test"))
    p.add_argument("--tests", default=str(WS / "tests"))
    a = ap.parse_args()
    if a.cmd == "train" and a.tile % 16:
        raise SystemExit("--tile must be a multiple of 16")
    rc = {"prepare": lambda: prepare(a.sources, a.data, a.min_sources), "train": lambda: train(a),
          "install": lambda: install(a.run, a.min_iou), "predict": lambda: predict(a.job),
          "compare": lambda: compare(a.run, a.inputs, a.tests)}[a.cmd]()
    sys.exit(rc or 0)


if __name__ == "__main__":
    main()
