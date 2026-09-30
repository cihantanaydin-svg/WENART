import json, os
from huggingface_hub import snapshot_download
path = snapshot_download("microsoft/TRELLIS.2-4B")
cfg = json.load(open(os.path.join(path, "pipeline.json")))
repos = set()                                     # sub-models named in pipeline.json (image encoder, rembg ...)
def walk(x):
    if isinstance(x, dict):
        for k, v in x.items():
            if k == "model_name" and isinstance(v, str) and v.count("/") == 1:
                repos.add(v)
            walk(v)
    elif isinstance(x, list):
        for v in x:
            walk(v)
walk(cfg)
for r in sorted(repos):
    print("downloading", r, "@", snapshot_download(r).rstrip("/").split("/")[-1], flush=True)
print("GEN3D weights OK:", ["microsoft/TRELLIS.2-4B"] + sorted(repos))
