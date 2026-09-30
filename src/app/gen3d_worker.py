"""GEN3D worker - runs inside /workspace/venv-gen3d (NOT the main venv) with the TRELLIS.2 repo on PYTHONPATH.
Input: a jobs JSON written by app/gen3d.py. Output: one GLB per job + <jobs>.result.json.
API as in the TRELLIS.2 README/example.py at commit 75fbf0183001ed9876c8dbb35de6b68552ee08bd (UNVERIFIED here:
no GPU in the environment where this was written)."""
import json, os, subprocess, sys, time, traceback
from pathlib import Path

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def vram():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout.split()
        return int(float(out[0]))
    except Exception:
        return None


def main(spec_file):
    spec = json.loads(Path(spec_file).read_text())
    from PIL import Image
    from trellis2.pipelines import Trellis2ImageTo3DPipeline
    import o_voxel
    t0 = time.time()
    pipe = Trellis2ImageTo3DPipeline.from_pretrained(spec["model"])
    pipe.cuda()
    print(f"GEN3D model loaded in {time.time() - t0:.0f}s, VRAM {vram()} MB", flush=True)
    results = []
    for job in spec["jobs"]:
        t1, peak = time.time(), 0
        try:
            mesh = pipe.run(Image.open(job["image"]))[0]
            peak = max(peak, vram() or 0)
            mesh.simplify(16777216)            # nvdiffrast limit (upstream example)
            glb = o_voxel.postprocess.to_glb(
                vertices=mesh.vertices, faces=mesh.faces, attr_volume=mesh.attrs, coords=mesh.coords,
                attr_layout=mesh.layout, voxel_size=mesh.voxel_size, aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
                decimation_target=spec["decimation_target"], texture_size=spec["texture_size"],
                remesh=True, remesh_band=1, remesh_project=0, verbose=False)
            Path(job["out"]).parent.mkdir(parents=True, exist_ok=True)
            glb.export(job["out"], extension_webp=False)   # PNG textures: Blender imports them without extra options
            results.append(dict(job, ok=True, seconds=round(time.time() - t1, 1), vram_peak_mb=peak))
        except Exception as e:
            results.append(dict(job, ok=False, error=f"{type(e).__name__}: {e}", trace=traceback.format_exc()[-1500:]))
        print(f"GEN3D {job['type']}: {'ok' if results[-1]['ok'] else results[-1]['error']}", flush=True)
    Path(spec_file).with_suffix(".result.json").write_text(json.dumps({"results": results}, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
