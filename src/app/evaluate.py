"""Test campaign helper (section 8). Compares the newest report of every plan with the room sizes you
measured, adds your visual review scores and writes tests/results.md + tests/results.csv.
  python -m app.evaluate --template    creates tests/truth.csv and tests/review.csv (smoke-test examples inside)
  python -m app.evaluate               compares and writes the results
Truth size = clear inside size of a room in metres, in any order (3.95 x 4.50 is the same as 4.50 x 3.95)."""
import argparse, csv, time
from pathlib import Path
from .common import WS, jload
from .textnorm import fold

VECTOR = ("dxf", "pdf_vector")           # DWG/DXF/vector PDF: +-5 cm. pdf_scan and photo: +-3 %
MAX_MINUTES = 120
SMOKE_TRUTH = [("SALON", 3.95, 4.50), ("EBEVEYN YATAK ODASI", 3.95, 4.50), ("ÇOCUK ODASI", 2.75, 4.30),
               ("MUTFAK", 2.95, 4.30), ("BANYO", 2.10, 4.30), ("HOL", 1.50, 8.00)]
SMOKE_FILES = ["test_plan.dxf", "test_plan.dwg", "test_plan_vector.pdf", "test_plan_scan.pdf", "test_plan_photo.jpg"]


def template(d):
    d.mkdir(parents=True, exist_ok=True)
    if not (d / "truth.csv").exists():
        with open(d / "truth.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["plan", "room", "width_m", "depth_m"])
            w.writerows([p, r, a, b] for p in SMOKE_FILES for r, a, b in SMOKE_TRUTH)
    if not (d / "review.csv").exists():
        with open(d / "review.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["plan", "furniture_1to5", "light_1to5", "ai_artifacts_yes_no", "notes"])
            w.writerows([p, "", "", "", ""] for p in SMOKE_FILES)
    print(f"Templates ready: {d / 'truth.csv'} and {d / 'review.csv'}. Replace the example rows with your plans.")


def newest_reports(outputs):
    best = {}
    for rp in Path(outputs).glob("*/report.json"):
        rep = jload(rp)
        name = Path(rep.get("input") or "").name
        if name and (name not in best or rp.stat().st_mtime > best[name][0]):
            best[name] = (rp.stat().st_mtime, rep, rp.parent)
    return {k: (v[1], v[2]) for k, v in best.items()}


def room_error(truth, found, kind):
    a, b = sorted(truth), sorted(found)
    if kind in VECTOR:
        e = max(abs(a[0] - b[0]), abs(a[1] - b[1])) * 100
        return e, e <= 5.0, f"{e:.1f} cm"
    e = max(abs(a[0] - b[0]) / a[0], abs(a[1] - b[1]) / a[1]) * 100
    return e, e <= 3.0, f"{e:.2f} %"


def match(rows, rooms, kind):
    """Same room name first (Turkish letters folded), then the closest size."""
    left, out = list(rooms), []
    for row in rows:
        t = (float(row["width_m"]), float(row["depth_m"]))
        key = fold(row["room"])
        named = [r for r in left if key and fold(r.get("label") or "") and
                 (fold(r["label"]) == key or fold(r["label"]) in key or key in fold(r["label"]))]
        pool = named or left
        if not pool:
            out.append((row, t, None, None, ""))
            continue
        best = min(pool, key=lambda r: room_error(t, r["size_m"], kind)[0])
        left.remove(best)
        out.append((row, t, best, room_error(t, best["size_m"], kind), "name" if named else "size only"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--template", action="store_true")
    ap.add_argument("--tests", default=str(WS / "tests"))
    ap.add_argument("--outputs", default=str(WS / "outputs"))
    a = ap.parse_args()
    d = Path(a.tests)
    if a.template:
        return template(d)
    truth = {}
    for row in csv.DictReader(open(d / "truth.csv", encoding="utf-8-sig")):
        if row.get("plan") and row.get("width_m"):
            truth.setdefault(row["plan"].strip(), []).append(row)
    review = {}
    if (d / "review.csv").exists():
        review = {r["plan"].strip(): r for r in csv.DictReader(open(d / "review.csv", encoding="utf-8-sig")) if r.get("plan")}
    reps = newest_reports(a.outputs)
    plans, rooms_md, notes = [], [], []
    for plan, rows in truth.items():
        if plan not in reps:
            plans.append({"plan": plan, "kind": "", "rooms_ok": "", "worst": "", "layout": "", "minutes": "", "cost": "",
                          "review": "", "result": "NOT RUN"})
            continue
        rep, job = reps[plan]
        kind = rep.get("input_kind") or ""
        m = match(rows, rep.get("rooms", []), kind)
        why = []
        for row, t, r, err, how in m:
            if r is None:
                why.append(f"room {row['room']} not found")
                rooms_md.append(f"| {plan} | {row['room']} | - | {t[0]:.2f} x {t[1]:.2f} | - | - | FAIL |")
                continue
            if not err[1]:
                why.append(f"{row['room']} off by {err[2]}")
            rooms_md.append(f"| {plan} | {row['room']} | {r.get('label') or r['type']} ({how}) | {t[0]:.2f} x {t[1]:.2f} | "
                            f"{r['size_m'][0]:.2f} x {r['size_m'][1]:.2f} | {err[2]} | {'OK' if err[1] else 'FAIL'} |")
        ok_rooms = sum(1 for x in m if x[2] is not None and x[3][1])
        worst = max((x[3] for x in m if x[3]), key=lambda e: e[0], default=None)
        ck = rep.get("layout_checks") or {}
        layout_ok = bool(ck) and all(v == 0 for v in ck.values()) and not rep.get("unfurnished_main_rooms")
        if not layout_ok:
            why.append(f"layout checks {ck}, unfurnished {rep.get('unfurnished_main_rooms')}")
        minutes = rep.get("pod_minutes") or 0
        if minutes > MAX_MINUTES:
            why.append(f"took {minutes} min (limit {MAX_MINUTES})")
        failed_stages = [k for k, v in (rep.get("stages") or {}).items() if str(v.get("status", "")).startswith("FAILED")]
        if failed_stages:
            why.append(f"failed stages: {failed_stages}")
        rv = review.get(plan, {})
        f5, l5, art = (rv.get("furniture_1to5") or "").strip(), (rv.get("light_1to5") or "").strip(), (rv.get("ai_artifacts_yes_no") or "").strip().lower()
        if f5 and l5 and art:
            review_ok = int(f5) >= 4 and int(l5) >= 4 and art.startswith("n")
            review_txt = f"furniture {f5}/5, light {l5}/5, artifacts {art}"
            if not review_ok:
                why.append(f"review: {review_txt}")
        else:
            review_ok, review_txt = None, "pending"
        auto_ok = ok_rooms == len(m) and layout_ok and minutes <= MAX_MINUTES and not failed_stages
        result = "FAIL" if not auto_ok or review_ok is False else "PASS" if review_ok else "REVIEW PENDING"
        plans.append({"plan": plan, "kind": kind, "rooms_ok": f"{ok_rooms}/{len(m)}", "worst": worst[2] if worst else "",
                      "layout": "OK" if layout_ok else "FAIL", "minutes": f"{minutes} / {rep.get('gpu_minutes')}",
                      "cost": rep.get("cost_usd"), "review": review_txt, "result": result, "job": str(job)})
        if why:
            notes.append(f"- **{plan}**: " + "; ".join(why))
    n_pass = sum(p["result"] == "PASS" for p in plans)
    run = [p for p in plans if p["result"] != "NOT RUN"]
    total_min = sum(reps[p["plan"]][0].get("pod_minutes") or 0 for p in run)
    total_cost = sum(reps[p["plan"]][0].get("cost_usd") or 0 for p in run)
    md = [f"# Test results - {time.strftime('%Y-%m-%d %H:%M')}", "",
          "| Plan | Kind | Rooms OK | Worst error | Layout | Minutes (pod / GPU) | Cost $ | Review | Result |",
          "|---|---|---|---|---|---|---|---|---|"]
    md += [f"| {p['plan']} | {p['kind']} | {p['rooms_ok']} | {p['worst']} | {p['layout']} | {p['minutes']} | {p['cost']} | "
           f"{p['review']} | **{p['result']}** |" for p in plans]
    md += ["", f"**Plans passed: {n_pass} of {len(plans)}** (PoC target: at least 8 of 10 real plans). "
               f"Plans run: {len(run)}, total pod time {total_min:.1f} min, total cost ${total_cost:.2f}.", "",
           "## Room details", "", "| Plan | Room (truth) | Matched room | Truth (m) | Found (m) | Error | OK |",
           "|---|---|---|---|---|---|---|"] + rooms_md
    md += ["", "## Why plans did not pass", ""] + (notes or ["- nothing to report"])
    (d / "results.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    with open(d / "results.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["plan", "kind", "rooms_ok", "worst", "layout", "minutes", "cost", "review", "result", "job"])
        w.writeheader()
        w.writerows(plans)
    print("\n".join(md[:len(plans) + 6]))
    print(f"\nWritten: {d / 'results.md'} and {d / 'results.csv'}")


if __name__ == "__main__":
    main()
