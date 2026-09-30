"""Setup check: can Cycles render on this GPU? Tries OptiX, then CUDA. Writes device.json.
Run: blender -b --factory-startup --python-exit-code 1 -P device_check.py -- <out.json>"""
import bpy, json, sys
out = sys.argv[sys.argv.index("--") + 1]
result = {"device": "CPU", "gpus": [], "tried": {}}
for typ in ("OPTIX", "CUDA"):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = bpy.context.scene
    sc.render.engine = "CYCLES"
    bpy.ops.mesh.primitive_plane_add(size=4)
    bpy.ops.mesh.primitive_cube_add(size=1, location=(0, 0, 0.5))
    bpy.ops.object.light_add(type="SUN", location=(0, 0, 5))
    bpy.ops.object.camera_add(location=(3, -3, 2.5), rotation=(1.05, 0, 0.785))
    sc.camera = bpy.context.object
    prefs = bpy.context.preferences.addons["cycles"].preferences
    try:
        prefs.compute_device_type = typ
        prefs.refresh_devices()
        devs = [d for d in prefs.devices if d.type == typ]
        if not devs:
            result["tried"][typ] = "no device"
            continue
        for d in prefs.devices:
            d.use = d.type == typ
        sc.cycles.device = "GPU"
        sc.cycles.samples = 8
        sc.render.resolution_x, sc.render.resolution_y = 128, 72
        sc.render.filepath = f"/tmp/device_check_{typ}.png"
        bpy.ops.render.render(write_still=True)
        img = bpy.data.images.load(sc.render.filepath)
        px = list(img.pixels)
        mean = sum(px[0::4]) / (len(px) / 4)
        result["tried"][typ] = f"ok mean={mean:.3f}"
        if mean > 0.02:
            result.update(device=typ, gpus=[d.name for d in devs])
            break
    except Exception as e:
        result["tried"][typ] = f"error {e}"
json.dump(result, open(out, "w"), indent=1)
print("DEVICE_CHECK", json.dumps(result))
