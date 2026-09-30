import json, os, sys
from huggingface_hub import snapshot_download
repo, lock_file = sys.argv[1], sys.argv[2]
path = snapshot_download(repo)
lock = json.load(open(lock_file)) if os.path.exists(lock_file) else {}
lock[repo] = path.rstrip("/").split("/")[-1]      # commit hash = the exact model version used
json.dump(lock, open(lock_file, "w"), indent=1)
print(f"downloaded {repo} @ {lock[repo]}", flush=True)
