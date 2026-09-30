import json, sys
from huggingface_hub import snapshot_download
vlm, skip_polish, lock_file = sys.argv[1], sys.argv[2] == "1", sys.argv[3]
jobs = [(vlm, None)]
if not skip_polish:
    jobs += [("stabilityai/stable-diffusion-xl-base-1.0",
              ["model_index.json", "scheduler/*", "tokenizer/*", "tokenizer_2/*", "vae/config.json",
               "text_encoder/config.json", "text_encoder/model.fp16.safetensors",
               "text_encoder_2/config.json", "text_encoder_2/model.fp16.safetensors",
               "unet/config.json", "unet/diffusion_pytorch_model.fp16.safetensors"]),
             ("diffusers/controlnet-depth-sdxl-1.0", ["config.json", "diffusion_pytorch_model.fp16.safetensors"]),
             ("madebyollin/sdxl-vae-fp16-fix", ["config.json", "diffusion_pytorch_model.safetensors"])]
lock = {}
for repo, patterns in jobs:
    path = snapshot_download(repo, allow_patterns=patterns)
    lock[repo] = path.rstrip("/").split("/")[-1]      # commit hash = the exact model version used
    print(f"downloaded {repo} @ {lock[repo]}", flush=True)
json.dump(lock, open(lock_file, "w"), indent=1)
