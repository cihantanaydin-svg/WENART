"""Structured patches proposed by the agent LLM: JSON-schema check, plan invariants, deterministic apply.
The LLM never writes geometry or Blender code. It names an operation and a few numbers; this module checks the
request against a JSON schema, applies it to a copy of plan.json, and accepts it only if the result does not
introduce a new invariant violation (openings inside their wall, no overlaps, plausible widths/heights/areas,
plausible scale). Accepted patches are stored in 02_plan/patches.json, so plan.json can always be rebuilt as
plan_raw.json (output of the plan stage) + every patch in order - a run can be replayed exactly."""
import copy, math, statistics, time
from pathlib import Path
from .common import jload, jsave

ROOM_TYPES = ["living", "bedroom", "kitchen", "bathroom", "wc", "hall", "storage", "dining", "study", "other"]
ROOM_ID = {"type": "string", "pattern": "^r[0-9]+$"}
OPENING_ID = {"type": "string", "pattern": "^(d|win)[0-9]+$"}
ITEM_ID = {"type": "string", "minLength": 1, "maxLength": 80}
FURNITURE_OPS = ("swap_furniture", "remove_furniture")


def _op(name, desc, props, required, **extra):
    p = {"op": {"const": name}}
    p.update(props)
    return dict({"type": "object", "description": desc, "properties": p, "required": ["op"] + required,
                 "additionalProperties": False}, **extra)


OPS = {
    "scale_plan": _op("scale_plan", "Multiply every plan length by factor (scale override).",
                      {"factor": {"type": "number", "minimum": 0.5, "maximum": 2.0}}, ["factor"]),
    "scale_from_label": _op("scale_from_label", "Rescale the whole plan so that this room's area equals its printed m2 label.",
                            {"room_id": ROOM_ID, "label_m2": {"type": "number", "exclusiveMinimum": 0.5, "maximum": 500}},
                            ["room_id", "label_m2"]),
    "retype_room": _op("retype_room", "Change a room's type (and optionally its label / master / child flag).",
                       {"room_id": ROOM_ID, "type": {"enum": ROOM_TYPES}, "label": {"type": "string", "maxLength": 60},
                        "master": {"type": "boolean"}, "child": {"type": "boolean"}}, ["room_id", "type"]),
    "move_opening": _op("move_opening", "Move a door/window along its wall by shift_m metres (+ = towards the wall end 'b').",
                        {"id": OPENING_ID, "shift_m": {"type": "number", "minimum": -3.0, "maximum": 3.0}}, ["id", "shift_m"]),
    "resize_opening": _op("resize_opening", "Change width (m) of a door/window, and sill/head height (m) of a window.",
                          {"id": OPENING_ID, "width": {"type": "number", "minimum": 0.3, "maximum": 4.0},
                           "sill": {"type": "number", "minimum": 0.0, "maximum": 2.0},
                           "head": {"type": "number", "minimum": 0.5, "maximum": 3.5}}, ["id"],
                          anyOf=[{"required": ["width"]}, {"required": ["sill"]}, {"required": ["head"]}]),
    "set_door_swing": _op("set_door_swing", "Set the hinge side ('a'/'b' end of the wall) and/or the room a door opens into.",
                          {"id": {"type": "string", "pattern": "^d[0-9]+$"}, "hinge": {"enum": ["a", "b"]}, "into": ROOM_ID},
                          ["id"], anyOf=[{"required": ["hinge"]}, {"required": ["into"]}]),
    "remove_opening": _op("remove_opening", "Delete a falsely detected door/window; the wall becomes solid there.",
                          {"id": OPENING_ID}, ["id"]),
    "swap_furniture": _op("swap_furniture", "Use another catalog model for one furniture item ('parametric' = built-in model).",
                          {"item_id": ITEM_ID, "asset_id": {"type": "string", "minLength": 1, "maxLength": 120}},
                          ["item_id", "asset_id"]),
    "remove_furniture": _op("remove_furniture", "Leave one furniture item out of the scene.", {"item_id": ITEM_ID}, ["item_id"]),
}
PATCH_SCHEMA = {
    "type": "object",
    "properties": {"ops": {"type": "array", "minItems": 1, "maxItems": 8, "items": {"oneOf": list(OPS.values())}},
                   "reason": {"type": "string", "minLength": 3, "maxLength": 1000,
                              "description": "why: which check or critique this fixes"}},
    "required": ["ops", "reason"], "additionalProperties": False}


def schema_errors(schema, obj, prefix=""):
    """All JSON-schema errors as short strings ([] = valid). Draft 2020-12."""
    from jsonschema import Draft202012Validator
    errs = sorted(Draft202012Validator(schema).iter_errors(obj), key=lambda e: list(e.absolute_path))
    return [f"{(prefix + '/'.join(str(p) for p in e.absolute_path)).rstrip('/') or '(root)'}: {e.message}" for e in errs]


def validate_patch(patch):
    """Schema check with readable errors: top level first, then each op against its own schema."""
    top = dict(PATCH_SCHEMA, properties=dict(PATCH_SCHEMA["properties"],
                                             ops=dict(PATCH_SCHEMA["properties"]["ops"], items={"type": "object"})))
    errs = schema_errors(top, patch)
    if errs:
        return errs
    for i, op in enumerate(patch["ops"]):
        name = op.get("op")
        if name not in OPS:
            errs.append(f"ops/{i}/op: unknown operation {name!r}; valid: {', '.join(OPS)}")
            continue
        errs += schema_errors(OPS[name], op, prefix=f"ops/{i}/")
    return errs


# ---------- geometry helpers ----------
def _wall_frame(w):
    (ax, ay), (bx, by) = w["a"], w["b"]
    L = math.hypot(bx - ax, by - ay) or 1e-9
    return (ax, ay), ((bx - ax) / L, (by - ay) / L), L


def _along(w, p):
    a, u, _ = _wall_frame(w)
    return (p[0] - a[0]) * u[0] + (p[1] - a[1]) * u[1]


def _area(poly):
    return abs(sum(poly[i][0] * poly[(i + 1) % len(poly)][1] - poly[(i + 1) % len(poly)][0] * poly[i][1]
                   for i in range(len(poly)))) / 2


def violations(plan):
    """Set of invariant violations (stable string keys). A patch may not add new ones."""
    v = set()
    walls = {w["id"]: w for w in plan.get("walls", [])}
    rooms = {r["id"]: r for r in plan.get("rooms", [])}
    per_wall = {}
    for kind, lst in (("door", plan.get("doors", [])), ("window", plan.get("windows", []))):
        for o in lst:
            w = walls.get(o.get("wall_id"))
            if w is None:
                v.add(f"no_wall:{o['id']}")
                continue
            _, _, L = _wall_frame(w)
            s = _along(w, o["center"])
            lo, hi = s - o["width"] / 2, s + o["width"] / 2
            if lo < -0.01 or hi > L + 0.01:
                v.add(f"outside_wall:{o['id']}")
            per_wall.setdefault(w["id"], []).append((lo, hi, o["id"]))
            wmin, wmax = ((0.5, 1.6) if o.get("kind") == "hinged" else (0.5, 4.0)) if kind == "door" else (0.3, 4.0)
            if not wmin <= o["width"] <= wmax:
                v.add(f"width:{o['id']}")
            wh = w.get("height", 2.7)
            if kind == "window":
                if o["sill"] < 0 or o["head"] - o["sill"] < 0.3 or o["head"] > wh - 0.02:
                    v.add(f"height:{o['id']}")
            elif o.get("head", 2.1) > wh - 0.02:
                v.add(f"height:{o['id']}")
            if kind == "door" and o.get("swing", {}).get("into") not in (o.get("rooms") or []) + [None]:
                v.add(f"swing:{o['id']}")
    for wid, lst in per_wall.items():
        lst.sort()
        for (a0, a1, ia), (b0, b1, ib) in zip(lst, lst[1:]):
            if b0 < a1 - 0.01:
                v.add(f"overlap:{ia}:{ib}")
    for r in rooms.values():
        if r.get("type") not in ROOM_TYPES + ["balcony"]:
            v.add(f"type:{r['id']}")
        if len(r.get("polygon", [])) < 3 or not 0.5 <= r.get("area_m2", 0) <= 300:
            v.add(f"area:{r['id']}")
    hinged = [o["width"] for o in plan.get("doors", []) if o.get("kind") == "hinged"]
    if hinged and not 0.6 <= statistics.median(hinged) <= 1.3:
        v.add("scale:door_median")
    th = [w["thickness"] for w in plan.get("walls", [])]
    if th and not 0.05 <= statistics.median(th) <= 0.6:
        v.add("scale:wall_thickness")
    pts = [p for r in plan.get("rooms", []) for p in r["polygon"]]
    if pts and max(max(p[0] for p in pts) - min(p[0] for p in pts), max(p[1] for p in pts) - min(p[1] for p in pts)) > 80:
        v.add("scale:extent")
    inside = sum(r.get("area_m2", 0) for r in plan.get("rooms", []) if r.get("type") != "balcony")
    if plan.get("rooms") and not 10 <= inside <= 1000:
        v.add("scale:total_area")
    if not plan.get("doors"):
        v.add("no_doors")
    return v


# ---------- operations (pure functions on a plan dict) ----------
def _scale(plan, f):
    S = lambda p: [round(p[0] * f, 4), round(p[1] * f, 4)]
    for w in plan["walls"]:
        w["a"], w["b"], w["thickness"] = S(w["a"]), S(w["b"]), round(w["thickness"] * f, 4)
    for o in plan["doors"] + plan["windows"]:
        o["center"], o["width"] = S(o["center"]), round(o["width"] * f, 4)
    for r in plan["rooms"]:
        r["polygon"] = [S(p) for p in r["polygon"]]
        r["area_m2"] = round(r["area_m2"] * f * f, 2)
        r["size_m"] = [round(s * f, 3) for s in r["size_m"]]
        for z in r.get("zones", []):
            z["pos"] = S(z["pos"])
    for t in plan.get("texts", []):
        t["pos"] = S(t["pos"])
    sc = plan["scale"]
    sc["m_per_px"] = sc["m_per_px"] * f
    sc["agent_factor"] = round(sc.get("agent_factor", 1.0) * f, 6)
    for c in sc.get("checks", []):
        if c.get("method") == "area_label":
            c["measured_m2"] = round(c["measured_m2"] * f * f, 2)
            c["deviation_pct"] = round(100 * (c["measured_m2"] / c["label_m2"] - 1), 2)
    a, b, c, d, e, g = plan["debug"]["m_to_px"]      # px = a*x + c, py = e*y + g  (x, y in metres)
    plan["debug"]["m_to_px"] = [a / f, b, c, d, e / f, g]


def _find(plan, oid):
    for lst in (plan["doors"], plan["windows"]):
        for o in lst:
            if o["id"] == oid:
                return lst, o
    raise ValueError(f"no door/window with id {oid}")


def apply_op(plan, op):
    """Apply one plan operation in place. Returns a short description. Raises ValueError on bad references."""
    name = op["op"]
    rooms = {r["id"]: r for r in plan["rooms"]}
    walls = {w["id"]: w for w in plan["walls"]}
    if name == "scale_plan":
        _scale(plan, op["factor"])
        return f"scaled plan x{op['factor']:.4f}"
    if name == "scale_from_label":
        r = rooms.get(op["room_id"])
        if r is None:
            raise ValueError(f"no room {op['room_id']}")
        f = math.sqrt(op["label_m2"] / max(r["area_m2"], 1e-6))
        if not 0.5 <= f <= 2.0:
            raise ValueError(f"label {op['label_m2']} m2 vs measured {r['area_m2']} m2 needs factor {f:.3f} (allowed 0.5-2.0)")
        r["label_area_m2"] = op["label_m2"]
        _scale(plan, f)
        return f"scaled plan x{f:.4f} so {r['id']} = {op['label_m2']} m2"
    if name == "retype_room":
        r = rooms.get(op["room_id"])
        if r is None:
            raise ValueError(f"no room {op['room_id']}")
        if r["type"] == "balcony":
            raise ValueError("balconies cannot be retyped (their outline is outside the walls)")
        old = r["type"]
        r["type"] = op["type"]
        if "label" in op:
            r["label"] = op["label"]
        for flag in ("master", "child"):
            if flag in op:
                r[flag] = op[flag]
            elif op["type"] != "bedroom":
                r.pop(flag, None)
        r["confidence"], r["retyped_by_agent"] = 0.8, True
        return f"{r['id']}: {old} -> {op['type']}"
    if name == "move_opening":
        _, o = _find(plan, op["id"])
        w = walls.get(o["wall_id"])
        if w is None:
            raise ValueError(f"{op['id']} has no wall")
        _, u, _ = _wall_frame(w)
        o["center"] = [round(o["center"][0] + u[0] * op["shift_m"], 4), round(o["center"][1] + u[1] * op["shift_m"], 4)]
        return f"moved {op['id']} by {op['shift_m']:+.2f} m along {w['id']}"
    if name == "resize_opening":
        lst, o = _find(plan, op["id"])
        if "sill" in op and lst is plan["doors"]:
            raise ValueError("doors have no sill")
        for k in ("width", "sill", "head"):
            if k in op:
                o[k] = round(op[k], 4)
        return f"resized {op['id']}: " + ", ".join(f"{k}={op[k]}" for k in ("width", "sill", "head") if k in op)
    if name == "set_door_swing":
        _, o = _find(plan, op["id"])
        sw = o.setdefault("swing", {})
        for k in ("hinge", "into"):
            if k in op:
                sw[k] = op[k]
        return f"door {op['id']} swing {sw}"
    if name == "remove_opening":
        lst, o = _find(plan, op["id"])
        lst.remove(o)
        return f"removed {op['id']}"
    raise ValueError(f"not a plan operation: {name}")


class PatchStore:
    """02_plan/plan_raw.json (plan stage output) + 02_plan/patches.json (accepted patches) -> 02_plan/plan.json."""
    def __init__(self, job):
        self.job = Path(job)
        self.raw, self.file, self.plan = (self.job / "02_plan" / n for n in ("plan_raw.json", "patches.json", "plan.json"))

    def patches(self):
        return jload(self.file, [])

    def after_plan_stage(self):
        """Call right after the plan stage wrote plan.json: keep it as the raw plan and re-apply stored patches."""
        jsave(self.raw, jload(self.plan))
        return self.rebuild()

    def rebuild(self):
        plan = copy.deepcopy(jload(self.raw) or jload(self.plan))
        skipped = []
        for p in self.patches():
            for op in p["ops"]:
                if op["op"] in FURNITURE_OPS:
                    continue
                try:
                    apply_op(plan, op)
                except ValueError as e:      # e.g. an opening id vanished after a new plan stage
                    skipped.append(f"{op['op']}: {e}")
        if skipped:
            plan.setdefault("warnings", []).append("patches not re-applied after re-planning: " + "; ".join(skipped))
        plan["patches_applied"] = sum(any(op["op"] not in FURNITURE_OPS for op in p["ops"]) for p in self.patches())
        self._write(plan)
        return plan, skipped

    def _write(self, plan):
        jsave(self.plan, plan)
        try:
            from .plan import overlay
            overlay(self.job, plan)
        except Exception as e:              # the overlay is a picture for humans/VLM; never block on it
            plan.setdefault("warnings", []).append(f"overlay not redrawn: {e}")

    def propose(self, patch, max_patches=12, meta=None):
        """Validate and apply. Returns {'accepted': bool, 'errors': [...], 'changes': [...], 'furniture_ops': [...]}."""
        errs = validate_patch(patch)
        if errs:
            return {"accepted": False, "stage": "schema", "errors": errs[:12]}
        stored = self.patches()
        if len(stored) >= max_patches:
            return {"accepted": False, "stage": "budget", "errors": [f"patch budget used up ({max_patches} patches)"]}
        cur = jload(self.plan)
        if cur is None:
            return {"accepted": False, "stage": "state", "errors": ["no plan yet: call parse_plan first"]}
        new, changes, furn = copy.deepcopy(cur), [], []
        try:
            for op in patch["ops"]:
                if op["op"] in FURNITURE_OPS:
                    furn.append(op)
                else:
                    changes.append(apply_op(new, op))
        except ValueError as e:
            return {"accepted": False, "stage": "apply", "errors": [str(e)]}
        added = sorted(violations(new) - violations(cur))
        if added:
            return {"accepted": False, "stage": "invariants", "errors": [f"would break: {v}" for v in added]}
        if not 0.5 <= new["scale"].get("agent_factor", 1.0) <= 2.0:
            return {"accepted": False, "stage": "invariants", "errors": ["cumulative scale change outside 0.5-2.0"]}
        rec = {"ts": time.strftime("%Y-%m-%d %H:%M:%S"), "ops": patch["ops"], "reason": patch["reason"]}
        rec.update(meta or {})
        jsave(self.file, stored + [rec])
        if changes:
            new["patches_applied"] = sum(any(op["op"] not in FURNITURE_OPS for op in p["ops"]) for p in stored + [patch])
            self._write(new)
        return {"accepted": True, "changes": changes, "furniture_ops": furn, "patch_index": len(stored)}
