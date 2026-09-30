#!/usr/bin/env bash
# =============================================================================
# setup.sh - floor plan -> furnished 3D apartment (Blender) + photoreal renders (PoC on ONE Runpod GPU pod)
SETUP_VERSION="2.0.3"   # see CHANGELOG.md
# SOURCE FORM (development): needs src/ next to it. The file for the pod is the single-file dist/setup.sh,  # bundle:source-only
# built by: python3 tools/bundle.py  (see docs/DEVELOPMENT.md)  # bundle:source-only
#
# What it does : checks GPU/driver/disk, installs pinned tools into /workspace (uv, Python venv,
#                PyTorch, Blender 5.2.2, LibreDWG), downloads the AI models and CC0 assets, writes the
#                pipeline code to /workspace/app, creates run.sh + start.sh + agent.sh, installs the local
#                agent LLM server (vLLM, own venv), the furniture model library (Poly Haven CC0 + your own
#                files) and - only with GEN3D=1 - an image-to-3D model, then runs a smoke test.
# How to run   : bash /workspace/setup.sh            (inside tmux; safe to run again)
#                bash /workspace/setup.sh --code-only (only rewrite /workspace/app + scripts, ~1 s)
# Time         : about 60-100 min the first time, 10-30 min when run again (smoke test incl. one agent run)
#                (+30-90 min with GEN3D=1: CUDA extensions are compiled). These are estimates.
# Disk         : about 75 GB on /workspace (models ~33 GB, agent LLM venv ~15 GB incl. download cache,
#                furniture ~1 GB); +30 GB with AGENT_MODEL=Qwen/Qwen3.6-27B-FP8; +40 GB with GEN3D=1.
#                About 2 GB on the container disk (+5 GB during a GEN3D=1 install). Estimates.
# =============================================================================
set -Eeuo pipefail

# ---------- settings you can change (or set them as environment variables) ----------
WS="${WS:-/workspace}"
GPU_PRICE_PER_HOUR="${GPU_PRICE_PER_HOUR:-0.84}"   # USD per hour of your pod (used in the cost report)
VLM_MODEL="${VLM_MODEL:-auto}"                     # auto = Qwen3.5-9B on GPUs with >=40 GB, else Qwen3.5-4B
SKIP_SMOKE_TEST="${SKIP_SMOKE_TEST:-0}"            # 1 = skip the smoke test at the end
SKIP_POLISH_MODELS="${SKIP_POLISH_MODELS:-0}"      # 1 = no SDXL download (saves ~10 GB, polish step off)
POD_TIER="${POD_TIER:-auto}"                       # auto = 48gb when the GPU has >= 44000 MB, else 24gb
AGENT_MODEL="${AGENT_MODEL:-auto}"                 # agent LLM (HF id). auto = reuse the VLM model above
                                                   # (Qwen3.5 reads images and calls tools). Evaluated 48 GB
                                                   # option: Qwen/Qwen3.6-27B-FP8 (Apache-2.0, ~30 GB weights)
SKIP_AGENT_LLM="${SKIP_AGENT_LLM:-0}"              # 1 = no vLLM venv / agent model; agent.sh uses --backend rules
GEN3D="${GEN3D:-0}"                                # 1 = image-to-3D furniture (TRELLIS.2-4B, 48 GB tier only,
                                                   # ~40 GB disk, licence limits: see CHANGELOG.md). Default off:
                                                   # nothing of it is downloaded unless GEN3D=1
MIN_FREE_GB="${MIN_FREE_GB:-auto}"                 # free space needed on /workspace for the first install
                                                   # (auto = 85 GB, +30 with a separate agent model, +40 GEN3D)
# HF_TOKEN (optional): set it as a Runpod environment variable - never type it into this file.
# GEN3D=1 needs HF_TOKEN with the Meta DINOv3 licence accepted on Hugging Face (gated image encoder).

# ---------- pinned versions (change only together with a new test run) ----------
UV_VERSION="0.12.20"
PY_VERSION="3.12.12"
BLENDER_VERSION="5.2.2"
BLENDER_SERIES="5.2"
LIBREDWG_VERSION="0.14"
TORCH_VERSION="2.14.0"
TORCHVISION_VERSION="0.29.0"
VLLM_VERSION="0.30.0"                  # agent LLM server, own venv (pins torch 2.13.0) - app/requirements-llm.lock
MICROMAMBA_VERSION="2.9.0-0"           # GEN3D only: CUDA 12.4 toolkit for compiling when the pod has no nvcc
TRELLIS2_COMMIT="75fbf0183001ed9876c8dbb35de6b68552ee08bd"   # GEN3D only: microsoft/TRELLIS.2 (2026-06-05)
NVDIFFRAST_TAG="v0.4.0"                                      # GEN3D only, as pinned by TRELLIS.2's setup.sh
NVDIFFREC_COMMIT="b296927cc7fd01c2ac1087c8065c4d7248f72da4"  # GEN3D only: JeffreyXiang/nvdiffrec renderutils
CUMESH_COMMIT="12289e1062f0603f2f0d0771b02e1395d247f26f"     # GEN3D only: JeffreyXiang/CuMesh
FLEXGEMM_COMMIT="6dd94a859c26ee8246888502eada3dd8ad85532e"   # GEN3D only: JeffreyXiang/FlexGEMM
UTILS3D_COMMIT="9a4eb15e4021b67b12c460c7057d642626897ec8"    # GEN3D only, as pinned by TRELLIS.2's setup.sh
FLASH_ATTN_VERSION="2.7.3"                                   # GEN3D only, as pinned by TRELLIS.2's setup.sh

# ---------- logging, error trap, folders ----------
mkdir -p "$WS/logs"
LOG="$WS/logs/setup_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1
# shellcheck disable=SC2154  # rc is assigned inside the trap string
trap 'rc=$?; echo; echo "ERROR: setup.sh stopped at line $LINENO (exit code $rc): $BASH_COMMAND"; echo "Full log: $LOG"; exit $rc' ERR
STATE="$WS/.setup_state"
mkdir -p "$STATE" "$WS"/{opt,cache,hf,assets,inputs,outputs,app}
APP="$WS/app"
PY="$WS/venv/bin/python"
UV="$WS/opt/uv-$UV_VERSION/uv"
BL_DIR="$WS/opt/blender-$BLENDER_VERSION"
LDWG="$WS/opt/libredwg"
LLM_PY="$WS/venv-llm/bin/python"      # agent LLM server (vLLM) - separate venv, the main torch pins stay untouched
GEN_PY="$WS/venv-gen3d/bin/python"    # GEN3D=1 only - separate venv (TRELLIS.2 needs torch 2.6 / CUDA 12.4)
export UV_CACHE_DIR="$WS/cache/uv" UV_PYTHON_INSTALL_DIR="$WS/opt/python" UV_LINK_MODE=copy
export HF_HOME="$WS/hf" PIP_CACHE_DIR="$WS/cache/pip" XDG_CACHE_HOME="$WS/cache"
export PIPE_WS="$WS" PIPE_BLENDER="$BL_DIR/blender" PYTHONPATH="$WS"
APT_PKGS="tmux nano curl ca-certificates xz-utils build-essential pkg-config xvfb libx11-6 libxi6 libxxf86vm1 libxfixes3 libxrender1 libxkbcommon0 libsm6 libice6 libgl1 libegl1 libglu1-mesa"
step() { echo; echo "=== [$(date +%H:%M:%S)] $* ==="; }
# source form: the embedded files live in src/ next to this script; tools/bundle.py builds the single-file dist/setup.sh  # bundle:source-only
SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/src"  # bundle:source-only
[[ -f "$SRC/app/common.py" ]] || { echo "src/ not found next to $0 - on a pod use the single-file dist/setup.sh"; exit 1; }  # bundle:source-only

# ---------- the pipeline code (one Python module per stage) + helper scripts ----------
write_code() {
  mkdir -p "$APP"
  cat > "$APP/__init__.py" < "$SRC/app/__init__.py"  # bundle:heredoc __END_OF___INIT___PY__
  cat > "$APP/common.py" < "$SRC/app/common.py"  # bundle:heredoc __END_OF_COMMON_PY__
  cat > "$APP/textnorm.py" < "$SRC/app/textnorm.py"  # bundle:heredoc __END_OF_TEXTNORM_PY__
  cat > "$APP/parse.py" < "$SRC/app/parse.py"  # bundle:heredoc __END_OF_PARSE_PY__
  cat > "$APP/vlm.py" < "$SRC/app/vlm.py"  # bundle:heredoc __END_OF_VLM_PY__
  cat > "$APP/plan.py" < "$SRC/app/plan.py"  # bundle:heredoc __END_OF_PLAN_PY__
  cat > "$APP/layout.py" < "$SRC/app/layout.py"  # bundle:heredoc __END_OF_LAYOUT_PY__
  cat > "$APP/blender_scene.py" < "$SRC/app/blender_scene.py"  # bundle:heredoc __END_OF_BLENDER_SCENE_PY__
  cat > "$APP/blender_render.py" < "$SRC/app/blender_render.py"  # bundle:heredoc __END_OF_BLENDER_RENDER_PY__
  cat > "$APP/device_check.py" < "$SRC/app/device_check.py"  # bundle:heredoc __END_OF_DEVICE_CHECK_PY__
  cat > "$APP/polish.py" < "$SRC/app/polish.py"  # bundle:heredoc __END_OF_POLISH_PY__
  cat > "$APP/report.py" < "$SRC/app/report.py"  # bundle:heredoc __END_OF_REPORT_PY__
  cat > "$APP/main.py" < "$SRC/app/main.py"  # bundle:heredoc __END_OF_MAIN_PY__
  cat > "$APP/assets_fetch.py" < "$SRC/app/assets_fetch.py"  # bundle:heredoc __END_OF_ASSETS_FETCH_PY__
  cat > "$APP/testplans.py" < "$SRC/app/testplans.py"  # bundle:heredoc __END_OF_TESTPLANS_PY__
  cat > "$APP/evaluate.py" < "$SRC/app/evaluate.py"  # bundle:heredoc __END_OF_EVALUATE_PY__
  cat > "$APP/wallseg.py" < "$SRC/app/wallseg.py"  # bundle:heredoc __END_OF_WALLSEG_PY__
  cat > "$APP/patches.py" < "$SRC/app/patches.py"  # bundle:heredoc __END_OF_PATCHES_PY__
  cat > "$APP/furniture.py" < "$SRC/app/furniture.py"  # bundle:heredoc __END_OF_FURNITURE_PY__
  cat > "$APP/llm_server.py" < "$SRC/app/llm_server.py"  # bundle:heredoc __END_OF_LLM_SERVER_PY__
  cat > "$APP/vision.py" < "$SRC/app/vision.py"  # bundle:heredoc __END_OF_VISION_PY__
  cat > "$APP/agent.py" < "$SRC/app/agent.py"  # bundle:heredoc __END_OF_AGENT_PY__
  cat > "$APP/deliver.py" < "$SRC/app/deliver.py"  # bundle:heredoc __END_OF_DELIVER_PY__
  cat > "$APP/blender_export.py" < "$SRC/app/blender_export.py"  # bundle:heredoc __END_OF_BLENDER_EXPORT_PY__
  cat > "$APP/validate_export.py" < "$SRC/app/validate_export.py"  # bundle:heredoc __END_OF_VALIDATE_EXPORT_PY__
  cat > "$APP/gen3d.py" < "$SRC/app/gen3d.py"  # bundle:heredoc __END_OF_GEN3D_PY__
  cat > "$APP/gen3d_worker.py" < "$SRC/app/gen3d_worker.py"  # bundle:heredoc __END_OF_GEN3D_WORKER_PY__
  cat > "$APP/requirements.lock" < "$SRC/app/requirements.lock"  # bundle:heredoc __END_OF_LOCK__
  cat > "$APP/requirements-llm.lock" < "$SRC/app/requirements-llm.lock"  # bundle:heredoc __END_OF_LLM_LOCK__
  cat > "$APP/requirements-gen3d.lock" < "$SRC/app/requirements-gen3d.lock"  # bundle:heredoc __END_OF_GEN3D_LOCK__
  cat > "$APP/env.sh" < "$SRC/app/env.sh"  # bundle:heredoc __END_OF_ENV__
  cat > "$WS/run.sh" < "$SRC/scripts/run.sh"  # bundle:heredoc __END_OF_RUN__
  cat > "$WS/start.sh" < "$SRC/scripts/start.sh"  # bundle:heredoc __END_OF_START__
  cat > "$WS/agent.sh" < "$SRC/scripts/agent.sh"  # bundle:heredoc __END_OF_AGENT_SH__
  sed -i "s#__WS__#$WS#g; s#__BLV__#$BLENDER_VERSION#g; s#__APT__#$APT_PKGS#g" "$APP/env.sh" "$WS/run.sh" "$WS/start.sh" "$WS/agent.sh"
  chmod +x "$WS/run.sh" "$WS/start.sh" "$WS/agent.sh"
  echo "Pipeline code written to $APP ($(find "$APP" -maxdepth 1 -name '*.py' | wc -l) Python files), plus $WS/run.sh, $WS/start.sh and $WS/agent.sh"
}

if [[ "${1:-}" == "--code-only" ]]; then
  write_code
  exit 0
fi

# ---------- 1. checks: GPU, driver, CUDA, disk ----------
step "1/13 Checking GPU, driver and disk (~5 s)"
command -v nvidia-smi >/dev/null || { echo "nvidia-smi not found - this is not a GPU pod."; exit 1; }
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
VRAM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1 | tr -dc 0-9)
DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
CUDA_MAX=$(nvidia-smi | grep -oE 'CUDA Version: [0-9.]+' | awk '{print $3}' || true)
echo "GPU: $GPU_NAME | VRAM: $VRAM_MB MB | driver: $DRIVER | driver supports CUDA up to: ${CUDA_MAX:-unknown}"
(( VRAM_MB >= 22000 )) || { echo "This PoC needs a GPU with at least 24 GB of VRAM."; exit 1; }
TORCH_CUDA="cu130"
if (( ${DRIVER%%.*} < 580 )); then
  TORCH_CUDA="cu128"
  echo "NOTE: driver < 580 cannot run CUDA 13 wheels - using the PyTorch CUDA 12.8 build instead."
  echo "      (Better: create the pod with the 'CUDA Versions: 13.0' filter.)"
fi
[[ "${NVIDIA_DRIVER_CAPABILITIES:-}" == *all* || "${NVIDIA_DRIVER_CAPABILITIES:-}" == *graphics* ]] || \
  echo "NOTE: NVIDIA_DRIVER_CAPABILITIES is '${NVIDIA_DRIVER_CAPABILITIES:-unset}'. If OptiX fails, set it to 'all' in the template."
if [[ "$VLM_MODEL" == "auto" ]]; then
  if (( VRAM_MB >= 40000 )); then VLM_MODEL="Qwen/Qwen3.5-9B"; else VLM_MODEL="Qwen/Qwen3.5-4B"; fi
fi
echo "Text-reading model: $VLM_MODEL"
if [[ "$POD_TIER" == "auto" ]]; then
  if (( VRAM_MB >= 44000 )); then POD_TIER="48gb"; else POD_TIER="24gb"; fi
fi
[[ "$POD_TIER" == "24gb" || "$POD_TIER" == "48gb" ]] || { echo "POD_TIER must be auto, 24gb or 48gb (got '$POD_TIER')."; exit 1; }
[[ "$AGENT_MODEL" == "auto" ]] && AGENT_MODEL="$VLM_MODEL"
if [[ "$POD_TIER" == "24gb" && "$AGENT_MODEL" =~ (27B|35B|122B|397B) ]]; then
  echo "AGENT_MODEL=$AGENT_MODEL does not fit a 24 GB GPU. Use AGENT_MODEL=auto (reuses $VLM_MODEL)."; exit 1
fi
if [[ "$GEN3D" == "1" && "$POD_TIER" != "48gb" ]]; then
  echo "NOTE: GEN3D=1 needs the 48 GB tier (TRELLIS.2 needs >= 24 GB on its own) - GEN3D turned off on this pod."
  GEN3D=0
fi
if [[ "$SKIP_AGENT_LLM" == "1" ]]; then
  echo "Pod tier: $POD_TIER | agent LLM: skipped (SKIP_AGENT_LLM=1, agent.sh uses --backend rules) | GEN3D: $GEN3D"
else
  echo "Pod tier: $POD_TIER | agent LLM: $AGENT_MODEL (vLLM $VLLM_VERSION) | GEN3D: $GEN3D"
fi
if [[ "$MIN_FREE_GB" == "auto" ]]; then        # estimates, see the Disk line at the top
  MIN_FREE_GB=85
  [[ "$SKIP_AGENT_LLM" != "1" && "$AGENT_MODEL" != "$VLM_MODEL" ]] && MIN_FREE_GB=$(( MIN_FREE_GB + 30 ))
  [[ "$GEN3D" == "1" ]] && MIN_FREE_GB=$(( MIN_FREE_GB + 40 ))
fi
FREE_GB=$(df -BG --output=avail "$WS" | tail -1 | tr -dc 0-9)
ROOT_FREE=$(df -BG --output=avail / | tail -1 | tr -dc 0-9)
echo "Free space: $WS ${FREE_GB} GB, container disk ${ROOT_FREE} GB"
if [[ ! -f "$STATE/models.done" ]] && (( FREE_GB < MIN_FREE_GB )); then
  echo "Need at least ${MIN_FREE_GB} GB free on $WS for the first install (models ~33 GB, tools ~12 GB,"
  echo "agent LLM venv ~15 GB, furniture ~1 GB; more with a separate AGENT_MODEL or GEN3D=1)."; exit 1
fi
(( ROOT_FREE >= 3 )) || { echo "Container disk is almost full (${ROOT_FREE} GB free)."; exit 1; }

# ---------- 2. system packages (container disk; start.sh repeats this after a restart) ----------
step "2/13 System packages (~1-2 min)"
MISSING=""
for p in $APT_PKGS; do dpkg -s "$p" >/dev/null 2>&1 || MISSING="$MISSING $p"; done
if [[ -n "$MISSING" ]]; then
  apt-get update -qq
  # shellcheck disable=SC2086  # MISSING is a space-separated package list
  DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends $MISSING
fi

# ---------- 3. uv, Python and the venv (all under /workspace) ----------
step "3/13 uv $UV_VERSION, Python $PY_VERSION, venv $WS/venv (~1 min)"
if [[ ! -x "$UV" ]]; then
  mkdir -p "$(dirname "$UV")"
  curl -fsSL --retry 3 "https://github.com/astral-sh/uv/releases/download/$UV_VERSION/uv-x86_64-unknown-linux-gnu.tar.gz" \
    | tar -xz --no-same-owner -C "$(dirname "$UV")" --strip-components=1
fi
"$UV" --version
"$UV" python install "$PY_VERSION"
[[ -x "$PY" ]] || "$UV" venv "$WS/venv" --python "$PY_VERSION"

# ---------- 4. pipeline code, env.sh, run.sh, start.sh ----------
step "4/13 Writing the pipeline code (~1 s)"
write_code

# ---------- 5. Python packages (exact versions from the lock file, prebuilt wheels only) ----------
step "5/13 Python packages (~3-6 min, ~9 GB)"
LOCK_SUM=$(sha256sum "$APP/requirements.lock" | cut -c1-16)
if [[ ! -f "$STATE/pip_${LOCK_SUM}_${TORCH_CUDA}.done" ]]; then
  if [[ "$TORCH_CUDA" == "cu130" ]]; then
    "$UV" pip install --python "$PY" --only-binary :all: -r "$APP/requirements.lock"
  else
    grep -vE '^(torch|torchvision|triton|nvidia-|cuda-)' "$APP/requirements.lock" > "$WS/cache/req_no_cuda.txt"
    "$UV" pip install --python "$PY" --only-binary :all: -r "$WS/cache/req_no_cuda.txt"
    "$UV" pip install --python "$PY" --only-binary :all: "torch==$TORCH_VERSION" "torchvision==$TORCHVISION_VERSION" \
      --index-url https://download.pytorch.org/whl/cu128
  fi
  touch "$STATE/pip_${LOCK_SUM}_${TORCH_CUDA}.done"
fi
"$PY" -c "import torch; assert torch.cuda.is_available(), 'PyTorch cannot see the GPU'; print('PyTorch', torch.__version__, '| CUDA', torch.version.cuda, '|', torch.cuda.get_device_name(0))"

# ---------- 6. Blender (official Linux build, checksum checked) ----------
step "6/13 Blender $BLENDER_VERSION (~1-2 min, ~1 GB)"
if [[ ! -x "$BL_DIR/blender" ]]; then
  T=$(mktemp -d)
  F="blender-$BLENDER_VERSION-linux-x64.tar.xz"
  curl -fL --retry 3 -o "$T/$F" "https://download.blender.org/release/Blender$BLENDER_SERIES/$F"
  curl -fsSL --retry 3 -o "$T/sums.txt" "https://download.blender.org/release/Blender$BLENDER_SERIES/blender-$BLENDER_VERSION.sha256"
  (cd "$T" && grep "$F\$" sums.txt | sha256sum -c -)
  mkdir -p "$BL_DIR"
  tar -xJf "$T/$F" --no-same-owner -C "$BL_DIR" --strip-components=1
  rm -rf "$T"
fi
"$BL_DIR/blender" -b --factory-startup --python-expr "import bpy; print('BLENDER_OK', bpy.app.version_string)" 2>&1 | grep BLENDER_OK \
  || { echo "Blender does not start. Missing libraries:"; ldd "$BL_DIR/blender" | grep "not found" || true; exit 1; }

# ---------- 7. LibreDWG (DWG -> DXF), built from the release tarball ----------
step "7/13 LibreDWG $LIBREDWG_VERSION for DWG files (~4-8 min the first time)"
if [[ ! -x "$LDWG/bin/dwg2dxf" ]]; then
  T=$(mktemp -d)
  if curl -fL --retry 3 -o "$T/l.tar.xz" "https://github.com/LibreDWG/libredwg/releases/download/$LIBREDWG_VERSION/libredwg-$LIBREDWG_VERSION.tar.xz" \
     && tar -xJf "$T/l.tar.xz" --no-same-owner -C "$T" \
     && (cd "$T/libredwg-$LIBREDWG_VERSION" && ./configure --prefix="$LDWG" --disable-bindings > "$WS/logs/libredwg_build.log" 2>&1 \
         && make -j"$(nproc)" >> "$WS/logs/libredwg_build.log" 2>&1 && make install >> "$WS/logs/libredwg_build.log" 2>&1); then
    echo "LibreDWG installed to $LDWG"
  else
    echo "WARNING: LibreDWG build failed (see $WS/logs/libredwg_build.log). DWG files will not work;"
    echo "         DXF, PDF and photos still work. Optional fallback: ODA File Converter in $WS/opt/oda/"
  fi
  rm -rf "$T"
fi
if [[ -x "$LDWG/bin/dwg2dxf" ]]; then LD_LIBRARY_PATH="$LDWG/lib" "$LDWG/bin/dwg2dxf" --version | head -1 || true; fi

# ---------- 8. AI models (Hugging Face cache in /workspace/hf) ----------
step "8/13 AI models (~15-35 min the first time, ~33 GB)"
MODEL_MARK="$STATE/models_${VLM_MODEL//\//_}_${SKIP_POLISH_MODELS}.done"
if [[ ! -f "$MODEL_MARK" ]]; then
  HF_HUB_OFFLINE=0 "$PY" - "$VLM_MODEL" "$SKIP_POLISH_MODELS" "$APP/models.lock.json" < "$SRC/installer/dl_models.py"  # bundle:heredoc __END_OF_DL__
  touch "$MODEL_MARK" "$STATE/models.done"
fi
"$PY" - "$WS/config.json" "$GPU_PRICE_PER_HOUR" "$VLM_MODEL" "$SKIP_POLISH_MODELS" "$POD_TIER" "$AGENT_MODEL" \
  "$SKIP_AGENT_LLM" "$GEN3D" < "$SRC/installer/write_config.py"  # bundle:heredoc __END_OF_CFG__

# ---------- 9. CC0 textures, sky HDRIs, plant models ----------
step "9/13 CC0 textures, sky HDRIs and plant models (~1-3 min, ~0.5 GB)"
if [[ ! -f "$STATE/assets.done" ]]; then
  (cd "$WS" && "$PY" -m app.assets_fetch "$WS/assets")
  touch "$STATE/assets.done"
fi

# ---------- 10. agent LLM server: vLLM in its own venv (+ the agent model if it is not the VLM) ----------
step "10/13 Agent LLM server: vLLM $VLLM_VERSION in $WS/venv-llm (~3-8 min, ~15 GB incl. download cache)"
if [[ "$SKIP_AGENT_LLM" == "1" ]]; then
  echo "Skipped (SKIP_AGENT_LLM=1): agent.sh runs with --backend rules (no LLM)."
else
  [[ -x "$LLM_PY" ]] || "$UV" venv "$WS/venv-llm" --python "$PY_VERSION"
  LLM_SUM=$(sha256sum "$APP/requirements-llm.lock" | cut -c1-16)
  if [[ ! -f "$STATE/llm_${LLM_SUM}_${TORCH_CUDA}.done" ]]; then
    if [[ "$TORCH_CUDA" == "cu130" ]]; then
      "$UV" pip install --python "$LLM_PY" --only-binary :all: -r "$APP/requirements-llm.lock"
    else
      # driver < 580: vLLM's CUDA 12.9 build (asset of its GitHub release) + PyTorch cu129 wheels; every other
      # package stays pinned by the lock. UNVERIFIED path: no pod with an older driver was available for testing.
      grep -vE '^(vllm|torch|torchvision|torchaudio|triton|nvidia-|cuda-)' "$APP/requirements-llm.lock" > "$WS/cache/req_llm_no_cuda.txt"
      "$UV" pip install --python "$LLM_PY" --only-binary :all: -c "$WS/cache/req_llm_no_cuda.txt" \
        "vllm @ https://github.com/vllm-project/vllm/releases/download/v$VLLM_VERSION/vllm-$VLLM_VERSION+cu129-cp38-abi3-manylinux_2_28_x86_64.whl" \
        --extra-index-url https://download.pytorch.org/whl/cu129
    fi
    touch "$STATE/llm_${LLM_SUM}_${TORCH_CUDA}.done"
  fi
  "$LLM_PY" -c "import vllm, torch; assert torch.cuda.is_available(), 'the vLLM venv cannot see the GPU'; print('vLLM', vllm.__version__, '| torch', torch.__version__, '| CUDA', torch.version.cuda)"
  AGENT_MARK="$STATE/agent_model_${AGENT_MODEL//\//_}.done"
  if [[ "$AGENT_MODEL" != "$VLM_MODEL" && ! -f "$AGENT_MARK" ]]; then
    FREE_GB=$(df -BG --output=avail "$WS" | tail -1 | tr -dc 0-9)
    (( FREE_GB >= 40 )) || { echo "Need ~40 GB free on $WS to download $AGENT_MODEL (${FREE_GB} GB free)."; exit 1; }
    HF_HUB_OFFLINE=0 "$PY" - "$AGENT_MODEL" "$APP/models.lock.json" < "$SRC/installer/dl_agent_model.py"  # bundle:heredoc __END_OF_AGENT_DL__
    touch "$AGENT_MARK"
  fi
fi

# ---------- 11. furniture model library: Poly Haven CC0 models + your own files -> catalog.json ----------
step "11/13 Furniture models: Poly Haven CC0 download + catalog incl. $WS/assets/models_user (~2-5 min, ~1 GB)"
mkdir -p "$WS/assets/models_user" "$WS/assets/furniture"
if [[ ! -f "$STATE/furniture_v1.done" ]]; then
  if (cd "$WS" && "$PY" -m app.furniture fetch); then
    touch "$STATE/furniture_v1.done"
  else
    echo "WARNING: Poly Haven furniture download failed - parametric furniture is used where no model exists."
  fi
fi
(cd "$WS" && "$PY" -m app.furniture catalog)   # offline, ~1 s: also picks up files you add to assets/models_user

# ---------- 12. optional image-to-3D furniture (GEN3D=1 only; nothing is downloaded otherwise) ----------
step "12/13 GEN3D image-to-3D furniture (only with GEN3D=1; ~30-90 min, ~40 GB)"
if [[ "$GEN3D" != "1" ]]; then
  echo "Skipped (GEN3D=$GEN3D): nothing downloaded or installed for image-to-3D."
else
  GEN_MARK="$STATE/gen3d_${TRELLIS2_COMMIT:0:12}_$(sha256sum "$APP/requirements-gen3d.lock" | cut -c1-12).done"
  if [[ ! -f "$GEN_MARK" ]]; then
    FREE_GB=$(df -BG --output=avail "$WS" | tail -1 | tr -dc 0-9)
    (( FREE_GB >= 45 )) || { echo "GEN3D needs ~45 GB free on $WS (${FREE_GB} GB free)."; exit 1; }
    [[ -n "${HF_TOKEN:-}" ]] || { echo "GEN3D=1 needs HF_TOKEN (Runpod env var) with the Meta DINOv3 licence accepted on Hugging Face."; exit 1; }
    command -v git >/dev/null || { apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends git; }
    [[ -x "$GEN_PY" ]] || "$UV" venv "$WS/venv-gen3d" --python "$PY_VERSION" --seed
    "$UV" pip install --python "$GEN_PY" --only-binary :all: -r "$APP/requirements-gen3d.lock"
    # The CUDA extensions are compiled here (TRELLIS.2 publishes no wheels): the pod's nvcc if it is CUDA 12.x,
    # else NVIDIA's CUDA 12.4.1 conda packages through a pinned, checksum-verified micromamba (all under $WS).
    cat > "$WS/cache/gen3d_build.sh" < "$SRC/installer/gen3d_build.sh"  # bundle:heredoc __END_OF_GEN3D_BUILD__
    # the build runs in its own bash process, so its set -e is active even inside this if-condition
    if ( export WS UV GEN_PY MICROMAMBA_VERSION TRELLIS2_COMMIT NVDIFFRAST_TAG NVDIFFREC_COMMIT CUMESH_COMMIT \
                FLEXGEMM_COMMIT UTILS3D_COMMIT FLASH_ATTN_VERSION
         bash "$WS/cache/gen3d_build.sh" ) > "$WS/logs/gen3d_build.log" 2>&1 \
       && HF_HUB_OFFLINE=0 "$PY" - < "$SRC/installer/dl_gen3d.py"  # bundle:heredoc __END_OF_GEN3D_DL__
    then
      touch "$GEN_MARK"
    else
      echo "WARNING: GEN3D install failed (build log: $WS/logs/gen3d_build.log). GEN3D stays off - parametric/library furniture only."
      GEN3D=0
      "$PY" -c "import json; p='$WS/config.json'; c=json.load(open(p)); c.setdefault('gen3d', {})['enabled']=False; json.dump(c, open(p, 'w'), indent=1)"
    fi
  fi
  if [[ "$GEN3D" == "1" ]]; then
    (cd "$WS" && "$PY" -m app.gen3d run) || echo "WARNING: GEN3D generation failed - see the messages above."
  fi
fi

# ---------- 13. GPU render check, agent LLM check, smoke test ----------
step "13/13 GPU render check (OptiX, then CUDA), agent LLM check and smoke test (~15-35 min)"
"$BL_DIR/blender" -b --factory-startup --python-exit-code 1 -P "$APP/device_check.py" -- "$APP/device.json" 2>&1 | grep DEVICE_CHECK || true
DEV=$("$PY" -c "import json; print(json.load(open('$APP/device.json'))['device'])" 2>/dev/null || echo CPU)
echo "Blender render device: $DEV"
[[ "$DEV" != "CPU" ]] || echo "WARNING: Blender cannot use the GPU - renders will be very slow. Set NVIDIA_DRIVER_CAPABILITIES=all and restart the pod."
# shellcheck source=/dev/null
source "$APP/env.sh"
cd "$WS"
if [[ "$SKIP_AGENT_LLM" != "1" ]]; then
  echo "Agent LLM check: start vLLM, one tool call, stop (~2-5 min)"
  if "$PY" -m app.llm_server test; then
    echo "AGENT LLM OK"
  else
    echo "WARNING: the agent LLM check failed (log: $WS/logs/llm_server.log) - agent.sh falls back to --backend rules."
    "$PY" -c "import json; p='$WS/config.json'; c=json.load(open(p)); c.setdefault('agent', {})['backend']='rules'; json.dump(c, open(p, 'w'), indent=1)"
  fi
fi
if [[ "$SKIP_SMOKE_TEST" == "1" ]]; then
  echo "Smoke test skipped (SKIP_SMOKE_TEST=1). CPU-only checks: python -m app.testplans unit"
else
  if "$PY" -m app.testplans quick; then
    echo "SMOKE TEST PASSED"
  else
    echo "SMOKE TEST FAILED - open $WS/outputs/smoke_*/report.md, $WS/outputs/smoke_*/final/validation.json and"
    echo "                    $WS/outputs/smoke_agent/final/report.md"
    exit 1
  fi
fi
echo
echo "Setup $SETUP_VERSION finished. Log: $LOG"
echo "Next: put a plan in $WS/inputs and run:  cd $WS && ./run.sh $WS/inputs/<plan file>"
echo "Agentic run (LLM checks + repairs):      cd $WS && ./agent.sh $WS/inputs/<plan file>"
echo "Deliverable per job: $WS/outputs/<job>/final/apartment.blend (+ .glb, .usdc, previews, report.md)"
echo "After a pod restart run:  bash $WS/start.sh"
