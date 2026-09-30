"""Stage 8 - report.json + report.md: time and VRAM peak per stage, GPU minutes, cost, rooms and sizes, checks,
deliverable validation, furniture coverage, warnings. Also copied to final/report.md when final/ exists."""
import time
from pathlib import Path
from .common import jload, jsave


def write_report(job, meta, cfg):
    job = Path(job)
    plan = jload(job / "02_plan/plan.json", {})
    lay = jload(job / "03_layout/layout.json", {})
    ren = jload(job / "05_render/render.json", {})
    pol = jload(job / "06_polish/polish.json", {})
    cams = jload(job / "04_scene/cameras.json", {})
    val = jload(job / "final/validation.json", {})
    cov = jload(job / "final/coverage.json", {})
    st = meta.get("stages", {})
    total = sum(s["seconds"] for s in st.values())
    gpu = sum(s["seconds"] for s in st.values() if s.get("gpu"))
    price = cfg["gpu_price_per_hour"]
    rooms = []
    for r in plan.get("rooms", []):
        dev = round(100 * (r["area_m2"] / r["label_area_m2"] - 1), 2) if r.get("label_area_m2") else None
        rooms.append({"id": r["id"], "type": r["type"], "label": r["label"], "size_m": r["size_m"], "area_m2": r["area_m2"],
                      "label_area_m2": r.get("label_area_m2"), "area_vs_label_pct": dev})
    warnings = list(meta.get("warnings", [])) + plan.get("warnings", []) + lay.get("warnings", []) + cams.get("warnings", [])
    warnings += [f"stage {k}: {v['status']}" for k, v in st.items() if v["status"].startswith(("FAILED", "skipped (no GPU)"))]
    rep = {"job": job.name, "input": meta.get("input"), "input_kind": plan.get("source", {}).get("kind"),
           "profile": meta.get("profile"), "started": meta.get("started"), "finished": meta.get("finished"),
           "stages": st, "pod_minutes": round(total / 60, 2), "gpu_minutes": round(gpu / 60, 2),
           "gpu_price_per_hour": price, "cost_usd": round(total / 3600 * price, 3),
           "gpu_util_avg_pct": meta.get("gpu", {}).get("gpu_util_avg"), "vram_peak_mb": meta.get("gpu", {}).get("vram_peak_mb"),
           "render_device": ren.get("device"), "scale": plan.get("scale"), "rooms": rooms,
           "counts": {"walls": len(plan.get("walls", [])), "doors": len(plan.get("doors", [])),
                      "windows": len(plan.get("windows", [])), "furniture": len(lay.get("items", [])),
                      "renders": len(ren.get("views", []))},
           "layout_checks": lay.get("checks"), "unfurnished_main_rooms": lay.get("unfurnished_main_rooms"),
           "polish": pol.get("items", []), "warnings": list(dict.fromkeys(warnings)),
           "export_validation": {"ok": val.get("ok"), "failed": [c["name"] for c in val.get("checks", []) if not c["ok"]]}
           if val else None, "furniture_coverage": cov.get("by_type")}
    jsave(job / "report.json", rep)
    L = [f"# Report - {job.name}", "", f"Input: `{rep['input']}` ({rep['input_kind']}), profile **{rep['profile']}**", "",
         "| Stage | Seconds | GPU | VRAM peak MB | Status |", "|---|---|---|---|---|"]
    L += [f"| {k} | {v['seconds']} | {'yes' if v.get('gpu') else ''} | {v.get('vram_peak_mb') or ''} | {v['status']} |"
          for k, v in st.items()]
    L += ["", f"**Pod time:** {rep['pod_minutes']} min  |  **GPU stages:** {rep['gpu_minutes']} min  |  "
          f"**Cost:** ${rep['cost_usd']} at ${price}/h  |  GPU util avg {rep['gpu_util_avg_pct']} %  |  VRAM peak {rep['vram_peak_mb']} MB", ""]
    sc = rep["scale"] or {}
    L += [f"**Scale:** {sc.get('method')} (confidence {sc.get('confidence')})", "",
          "| Room | Type | Size (m) | Area m² | Label m² | Diff % |", "|---|---|---|---|---|---|"]
    L += [f"| {r['label'] or r['id']} | {r['type']} | {r['size_m'][0]:.2f} x {r['size_m'][1]:.2f} | {r['area_m2']} | "
          f"{r['label_area_m2'] or ''} | {'' if r['area_vs_label_pct'] is None else r['area_vs_label_pct']} |" for r in rooms]
    L += ["", f"**Counts:** {rep['counts']}", f"**Layout checks:** {rep['layout_checks']}", ""]
    if rep["polish"]:
        L += ["**Polish:** " + ", ".join(f"{p['name']} {'polished' if p['accepted'] else 'raw kept'} ({p['edge_score']})" for p in rep["polish"]), ""]
    if val:
        L += [f"**Deliverable (final/):** validation {'PASSED' if val.get('ok') else 'FAILED'}"] + \
             [f"- {'ok' if c['ok'] else 'FAIL'} {c['name']}: {c.get('detail', '')}" for c in val.get("checks", []) if not c["ok"]] + [""]
    if cov.get("by_type"):
        L += ["**Furniture sources:** " + ", ".join(f"{t}: " + "/".join(f"{k} {n}" for k, n in c.items())
                                                   for t, c in sorted(cov["by_type"].items())), ""]
    L += ["## Warnings", ""] + [f"- {w}" for w in rep["warnings"]] + ([] if rep["warnings"] else ["- none"])
    (job / "report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    if (job / "final").is_dir():
        (job / "final/report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    return rep
