"""Final deliverable (headless Blender): 04_scene/scene.blend -> final/apartment.blend (+ .glb, .usdc).
Run: blender -b --factory-startup --python-exit-code 1 -P blender_export.py -- <job_dir>
Metres, Z up, unit scale 1.0, world origin at the plan's lower-left corner (outer wall faces, balconies),
collections Walls / Floors_Ceilings / Openings / Furniture.<room> / Lights / Cameras (made by blender_scene.py),
textures packed, unused data purged. Ceilings are hidden in the viewport (not in renders) so the rooms are
visible from above when the file is opened. Writes final/export.json (offset, counts)."""
import bpy, json, math, sys
from pathlib import Path
from mathutils import Matrix, Vector

ARGS = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
JOB = Path(ARGS[0])
FIN = JOB / "final"
FIN.mkdir(parents=True, exist_ok=True)
CFG = json.loads((JOB / "run_config.json").read_text()).get("export", {})
PLAN = json.loads((JOB / "02_plan/plan.json").read_text(encoding="utf-8"))


def plan_footprint(plan):
    """Lower-left and upper-right corner of the plan: outer wall faces and every room polygon (balconies too)."""
    xs, ys = [], []
    for w in plan["walls"]:
        (ax, ay), (bx, by) = w["a"], w["b"]
        L = math.hypot(bx - ax, by - ay) or 1.0
        nx, ny = -(by - ay) / L * w["thickness"] / 2, (bx - ax) / L * w["thickness"] / 2
        for px, py in ((ax + nx, ay + ny), (ax - nx, ay - ny), (bx + nx, by + ny), (bx - nx, by - ny)):
            xs.append(px)
            ys.append(py)
    for r in plan["rooms"]:
        xs += [p[0] for p in r["polygon"]]
        ys += [p[1] for p in r["polygon"]]
    return min(xs), min(ys), max(xs), max(ys)


bpy.ops.wm.open_mainfile(filepath=str(JOB / "04_scene/scene.blend"))
SC = bpy.context.scene

us = SC.unit_settings
us.system, us.scale_length, us.length_unit = "METRIC", 1.0, "METERS"

fx0, fy0, _, _ = plan_footprint(PLAN)
off = Vector((-fx0, -fy0, 0.0))            # plan coordinates -> origin at the plan's lower-left corner
for o in list(SC.objects):
    if o.parent is not None:
        continue
    at_origin = o.matrix_world == Matrix.Identity(4)
    if o.type == "MESH" and at_origin and o.data.users == 1:
        o.data.transform(Matrix.Translation(off))   # walls, floors, ceilings: keep their origin at the world origin
    else:
        o.location = o.location + off
SC["plan_offset_m"] = [round(off.x, 5), round(off.y, 5)]
SC["units"] = "metres, Z up, origin = lower-left corner of the plan (outer walls and balconies)"

if CFG.get("hide_ceilings_in_viewport", True):
    fc = bpy.data.collections.get("Floors_Ceilings")
    for o in fc.objects if fc else []:
        if o.name.startswith(("ceiling_", "slab_top")):
            o.hide_set(True)

bpy.ops.file.pack_all()
for _ in range(3):
    bpy.data.orphans_purge(do_local_ids=True, do_linked_ids=True, do_recursive=True)
counts = {c.name: len(c.all_objects) for c in SC.collection.children}
bpy.ops.wm.save_as_mainfile(filepath=str(FIN / "apartment.blend"), compress=True)
files = ["apartment.blend"]
if CFG.get("glb", True):
    bpy.ops.export_scene.gltf(filepath=str(FIN / "apartment.glb"), export_format="GLB", export_apply=True,
                              export_cameras=True, export_lights=True, export_extras=True, export_yup=True)
    files.append("apartment.glb")
if CFG.get("usdc", True):
    bpy.ops.wm.usd_export(filepath=str(FIN / "apartment.usdc"), export_materials=True, generate_preview_surface=True,
                          export_textures_mode="NEW", overwrite_textures=True, relative_paths=True,
                          evaluation_mode="RENDER", convert_scene_units="METERS")
    files.append("apartment.usdc")
json.dump({"offset_m": [off.x, off.y, 0.0], "collections": counts, "files": files,
           "blender": bpy.app.version_string}, open(FIN / "export.json", "w"), indent=1, ensure_ascii=False)
print(f"EXPORT_OK files={files} offset={tuple(round(x, 4) for x in off)} collections={counts}")
