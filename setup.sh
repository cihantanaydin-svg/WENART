#!/usr/bin/env bash
# =============================================================================
# setup.sh - floor plan -> furnished 3D apartment (Blender) + photoreal renders (PoC on ONE Runpod GPU pod)
SETUP_VERSION="2.0.1"   # see CHANGELOG.md
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

# ---------- the pipeline code (one Python module per stage) + helper scripts ----------
write_code() {
  mkdir -p "$APP"
  cat > "$APP/__init__.py" <<'__END_OF___INIT___PY__'

__END_OF___INIT___PY__
  cat > "$APP/common.py" <<'__END_OF_COMMON_PY__'
"""Shared helpers: paths, config, logging, JSON, subprocess, GPU sampling."""
import json, os, subprocess, sys, time, threading
from pathlib import Path

WS = Path(os.environ.get("PIPE_WS", "/workspace"))
APP = Path(__file__).resolve().parent
BLENDER = os.environ.get("PIPE_BLENDER", str(WS / "opt/blender-5.2.2/blender"))

# ---- default settings (override any key in /workspace/config.json) ----
DEFAULTS = {
    "gpu_price_per_hour": 0.84,
    "heights": {"ceiling": 2.70, "door_head": 2.10, "window_sill": 0.90, "window_head": 2.30,
                "wet_window_sill": 1.40, "wet_window_head": 2.10},
    "profiles": {
        "full":  {"width": 1920, "height": 1080, "max_samples": 384, "noise_threshold": 0.02,
                  "time_limit_s": 240, "views_main": 3, "views_other": 1, "polish_max_images": 999},
        "quick": {"width": 640, "height": 360, "max_samples": 48, "noise_threshold": 0.05,
                  "time_limit_s": 60, "views_main": 1, "views_other": 1, "polish_max_images": 2},
        "preview": {"width": 480, "height": 270, "max_samples": 24, "noise_threshold": 0.08,   # agent previews
                    "time_limit_s": 30, "views_main": 1, "views_other": 1, "polish_max_images": 0},
    },
    "exposure": 0.0, "sun_strength": 3.5, "fill_w_per_m2": 5.0,   # render brightness knobs
    "vlm_model": "auto",            # auto | Qwen/Qwen3.5-9B | Qwen/Qwen3.5-4B
    "vlm_max_side": 1600,
    "wallseg": {"enabled": False, "model": str(WS / "models/wallseg/model.pt")},   # section 9, off until trained
    "polish": {"enabled": True, "strength": 0.25, "steps": 30, "guidance": 5.0,
               "controlnet_scale": 0.8, "min_edge_score": 0.45},
    "style": "modern warm neutral interior, light oak wood floor, warm white walls, "
             "beige and greige fabrics, black metal accents",
    "models": {"sdxl": "stabilityai/stable-diffusion-xl-base-1.0",
               "controlnet": "diffusers/controlnet-depth-sdxl-1.0",
               "vae": "madebyollin/sdxl-vae-fp16-fix"},
    # ---- agent layer (agent.sh / app.agent) ----
    "pod_tier": "auto",             # auto (48gb when the GPU has >= 44000 MB) | 24gb | 48gb
    # GPU memory per component in MB. ESTIMATES used to decide whether the LLM server must be stopped
    # before a GPU stage; the agent logs the real peak of every step in agent_log.jsonl and report.md.
    "vram_estimates_mb": {"vlm_transformers_4b": 12000, "vlm_transformers_9b": 24000, "cycles_preview": 3000,
                          "cycles_full": 6000, "sdxl_polish": 13000, "gen3d": 40000, "margin": 2000},
    "llm": {"model": "auto",        # auto = the vlm_model (Qwen3.5 reads images AND calls tools); any HF id
            "manage_server": True,  # False = you run your own OpenAI-compatible server at host:port
            "host": "127.0.0.1", "port": 8011, "served_name": "agent", "venv": str(WS / "venv-llm"),
            "max_model_len": 32768, "max_num_seqs": 2, "gpu_memory_utilization": "auto",
            "tool_call_parser": "qwen3_coder", "reasoning_parser": "qwen3", "vision": True,
            "startup_timeout_s": 1200, "request_timeout_s": 600, "temperature": 0.2, "max_tokens": 2048,
            "extra_args": [],
            # weights in GB (ESTIMATES from parameter count x bytes per weight) for gpu_memory_utilization=auto
            "weights_gb": {"Qwen/Qwen3.5-4B": 9.5, "Qwen/Qwen3.5-9B": 19.5, "Qwen/Qwen3.6-27B-FP8": 30.0,
                           "Qwen/Qwen3.6-35B-A3B-FP8": 37.0, "Qwen/Qwen3.8-27B-FP8": 30.0},
            "kv_overhead_gb": 6.0},   # KV cache + activations + CUDA graphs (ESTIMATE)
    "agent": {"backend": "llm",     # llm | rules (no LLM) | mock (tests)
              "max_steps": 40, "max_minutes": 45, "max_cost_usd": 1.5, "max_repairs_per_issue": 2,
              "max_patches": 12, "preview_profile": "preview", "area_tol_pct_vector": 3.0,
              "area_tol_pct_raster": 6.0, "min_scale_confidence": 0.6, "render_on_cpu": False,
              "critique_overlay": True, "critique_renders": 2, "tool_result_chars": 6000},
    "furniture": {"enabled": True, "library": str(WS / "assets/furniture"),
                  "user_dir": str(WS / "assets/models_user"), "legacy_models": str(WS / "assets/models"),
                  "allowed_licences": ["CC0-1.0", "CC-BY-4.0", "CC-BY-3.0", "MIT", "Apache-2.0", "owned"],
                  "allow_generated": True, "max_nonuniform": 0.15, "max_uniform_change": 0.35,
                  "poly_cap": 60000, "poly_cap_by_type": {"sofa": 120000, "bed_double": 120000,
                                                          "bed_single": 100000, "plant": 80000},
                  "polyhaven_per_type": 3, "polyhaven_resolution": "1k"},
    "gen3d": {"enabled": False, "backend": "trellis2", "model": "microsoft/TRELLIS.2-4B",
              "venv": str(WS / "venv-gen3d"), "repo": str(WS / "opt/TRELLIS.2"),
              "inputs": str(WS / "assets/gen3d_inputs"), "max_items": 6, "decimation_target": 60000,
              "texture_size": 1024},
    "export": {"hide_ceilings_in_viewport": True, "glb": True, "usdc": True},
}


def deep_update(base, extra):
    for k, v in extra.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_update(base[k], v)
        else:
            base[k] = v
    return base


def load_config():
    cfg = json.loads(json.dumps(DEFAULTS))
    user = WS / "config.json"
    if user.exists():
        deep_update(cfg, json.loads(user.read_text()))
    return cfg


def jload(p, default=None):
    p = Path(p)
    if not p.exists():
        return default
    return json.loads(p.read_text(encoding="utf-8"))


def jsave(p, obj):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=1, ensure_ascii=False), encoding="utf-8")


def log(msg):
    print(time.strftime("[%H:%M:%S] ") + str(msg), flush=True)


def run(cmd, log_file=None, env=None, cwd=None):
    """Run a command, stream its output to the console and a log file, raise on failure."""
    log("RUN " + " ".join(str(c) for c in cmd))
    e = dict(os.environ)
    if env:
        e.update(env)
    with subprocess.Popen([str(c) for c in cmd], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True, env=e, cwd=cwd, bufsize=1) as p:
        fh = open(log_file, "a", encoding="utf-8") if log_file else None
        for line in p.stdout:
            sys.stdout.write(line)
            if fh:
                fh.write(line)
        p.wait()
        if fh:
            fh.close()
    if p.returncode != 0:
        raise RuntimeError(f"command failed ({p.returncode}): {cmd[:3]}")


def blender(script, job):
    """Run one of the Blender scripts headless on a job (or with a Python that has the bpy module: test mode)."""
    b = Path(BLENDER)
    if "python" in b.name:   # test mode: a Python with the bpy module
        cmd = [b, APP / script, "--", job]
    else:
        cmd = [b, "-b", "--factory-startup", "--python-exit-code", "1", "-P", APP / script, "--", job]
    run(cmd, log_file=Path(job) / "logs" / f"{script[:-3]}.log")


def font_path():
    """A TTF font with Turkish glyphs (DejaVu Sans ships inside matplotlib)."""
    import matplotlib
    return str(Path(matplotlib.get_data_path()) / "fonts/ttf/DejaVuSans.ttf")


class GpuSampler:
    """Samples GPU utilisation and memory once per second with nvidia-smi."""
    def __init__(self):
        self.samples, self._stop, self._t = [], threading.Event(), None

    def _loop(self):
        while not self._stop.is_set():
            try:
                out = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
                                      "--format=csv,noheader,nounits"], capture_output=True,
                                     text=True, timeout=10).stdout.strip().splitlines()
                if out:
                    u, m = [float(x) for x in out[0].split(",")]
                    self.samples.append((u, m))
            except Exception:
                pass
            self._stop.wait(1.0)

    def start(self):
        self._t = threading.Thread(target=self._loop, daemon=True)
        self._t.start()
        return self

    def stop(self):
        self._stop.set()
        if self._t:
            self._t.join(timeout=5)
        if not self.samples:
            return {"gpu_util_avg": None, "vram_peak_mb": None}
        return {"gpu_util_avg": round(sum(s[0] for s in self.samples) / len(self.samples), 1),
                "vram_peak_mb": max(s[1] for s in self.samples)}


def has_cuda():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def _smi(field):
    try:
        out = subprocess.run(["nvidia-smi", f"--query-gpu={field}", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout.split()
        return int(float(out[0])) if out else None
    except Exception:
        return None


def gpu_total_mb():
    return _smi("memory.total")


def gpu_used_mb():
    return _smi("memory.used")


def pod_tier(cfg=None):
    """'48gb' when the GPU has >= 44000 MB (A40, A6000, L40S, 6000 Ada), else '24gb'. config pod_tier overrides."""
    t = (cfg or load_config()).get("pod_tier", "auto")
    if t in ("24gb", "48gb"):
        return t
    total = gpu_total_mb()
    return "48gb" if total and total >= 44000 else "24gb"
__END_OF_COMMON_PY__
  cat > "$APP/textnorm.py" <<'__END_OF_TEXTNORM_PY__'
"""Turkish plan text: case folding, old-codepage repair, room labels, areas, scale notes."""
import re, difflib

MOJIBAKE = {"þ": "ş", "ý": "ı", "ð": "ğ", "Þ": "Ş", "Ý": "İ", "Ð": "Ğ"}
FOLD = str.maketrans({"Ç": "C", "Ğ": "G", "İ": "I", "Ö": "O", "Ş": "S", "Ü": "U", "Â": "A", "Î": "I", "Û": "U"})

# Order matters: first match wins ("EBEVEYN BANYO" must be a bathroom, not a bedroom).
ROOM_WORDS = [
    ("BANYO", "bathroom", {}), ("DUS", "bathroom", {}),
    ("WC", "wc", {}), ("W C", "wc", {}), ("TUVALET", "wc", {}), ("LAVABO", "wc", {}),
    ("AMERIKAN MUTFAK", "kitchen", {"open": True}), ("ACIK MUTFAK", "kitchen", {"open": True}),
    ("MUTFAK", "kitchen", {}),
    ("BALKON", "balcony", {}), ("TERAS", "balcony", {}),
    ("SALON", "living", {}), ("OTURMA", "living", {}), ("MISAFIR", "living", {}),
    ("YEMEK", "dining", {}), ("CALISMA", "study", {}),
    ("EBEVEYN", "bedroom", {"master": True}), ("COCUK", "bedroom", {"child": True}),
    ("YATAK ODASI", "bedroom", {}), ("Y ODASI", "bedroom", {}), ("YATAK", "bedroom", {}),
    ("HOL", "hall", {}), ("ANTRE", "hall", {}), ("GIRIS", "hall", {}), ("KORIDOR", "hall", {}),
    ("GECIS", "hall", {}),
    ("KILER", "storage", {}), ("DEPO", "storage", {}), ("CAMASIR", "storage", {}),
    ("GIYINME", "storage", {}), ("VESTIYER", "storage", {}),
    ("ODA", "bedroom", {}),
]
SCALES = {20, 25, 50, 75, 100, 200, 250, 500}


def repair(s):
    """Fix Windows-1254 text read as 1252 and \\U+XXXX escapes from old DWGs."""
    s = re.sub(r"\\U\+([0-9A-Fa-f]{4})", lambda m: chr(int(m.group(1), 16)), s)
    return "".join(MOJIBAKE.get(c, c) for c in s)


def tr_upper(s):
    return s.replace("i", "İ").replace("ı", "I").upper()


def fold(s):
    """Upper-case the Turkish way, strip accents, keep letters/digits/,.²/ and spaces."""
    s = tr_upper(repair(s)).translate(FOLD)
    s = re.sub(r"[^A-Z0-9,./²:\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _fuzzy_in(pattern, text):
    if re.search(r"(^|\s)" + re.escape(pattern) + r"(\s|$)", text):
        return True
    ptoks, ttoks = pattern.split(), text.split()
    if len(pattern) < 4:
        return False
    for i in range(len(ttoks) - len(ptoks) + 1):
        cand = " ".join(ttoks[i:i + len(ptoks)])
        if difflib.SequenceMatcher(None, pattern, cand).ratio() >= 0.8:
            return True
    return False


def room_type(text):
    t = re.sub(r"[0-9,.²/:]", " ", fold(text))
    t = re.sub(r"\s+", " ", t.replace("0", "O")).strip()
    for pat, typ, extra in ROOM_WORDS:
        if _fuzzy_in(pat, t):
            return typ, dict(extra)
    return None, {}


def parse_area(text):
    t = fold(text).replace(" ", "")
    m = re.search(r"(\d{1,3}(?:[.,]\d{1,2})?)M(?:2|²|\^2)", t)
    if not m:
        return None
    v = float(m.group(1).replace(",", "."))
    return v if 0.5 <= v <= 500 else None


def parse_scale(text):
    t = fold(text)
    m = re.search(r"(?:OLCEK|OLC|SCALE|M)?\s*[:.]?\s*1\s*[/:]\s*(\d{2,3})\b", t)
    if m and int(m.group(1)) in SCALES:
        return int(m.group(1))
    return None


def parse_number(text):
    t = fold(text).replace(" ", "")
    if re.fullmatch(r"\d{1,4}([.,]\d{1,3})?", t):
        return float(t.replace(",", "."))
    return None


def classify(text):
    """Return {'kind': room_label|area|scale|dimension|other, ...} for one text item."""
    out = {"kind": "other"}
    area = parse_area(text)
    sc = parse_scale(text)
    typ, extra = room_type(text)
    if typ:
        out.update(kind="room_label", type=typ, **extra)
        if area:
            out["area"] = area
    elif area:
        out.update(kind="area", value=area)
    elif sc:
        out.update(kind="scale", value=sc)
    elif parse_number(text) is not None:
        out.update(kind="dimension", value=parse_number(text))
    return out
__END_OF_TEXTNORM_PY__
  cat > "$APP/parse.py" <<'__END_OF_PARSE_PY__'
"""Stage 1 - parse any input into one common extraction (01_parse/extract.json):
wall mask + stroke mask + background image (same pixel frame), texts in pixel coords,
known scale (if any), scale hints and vector wall-face lines for snapping."""
import math, os, shutil, subprocess
from pathlib import Path
import cv2
import numpy as np
from scipy import ndimage
from .common import WS, jsave, log, run
from .textnorm import fold, repair, parse_scale

UNITS_M = {1: 0.0254, 2: 0.3048, 4: 0.001, 5: 0.01, 6: 1.0, 14: 0.1}
WALL_KEYS = ("DUVAR", "PERDE", "KOLON", "BETON", "WALL")
DOOR_KEYS = ("KAPI", "DOOR")
WIN_KEYS = ("PENCERE", "GLAZ", "WINDOW", "DOGRAMA")
SKIP_KEYS = ("TEFRIS", "MOBILYA", "FURN", "AKS", "GRID", "OLCU", "DIM", "CIVAR")


# ---------- detection ----------
def detect(path):
    b = Path(path).read_bytes()[:64]
    if b.startswith(b"%PDF"):
        return "pdf"
    if b[:4] == b"AC10":
        return "dwg"
    if b.startswith(b"AutoCAD Binary DXF"):
        return "dxf"
    if b[:3] == b"\xff\xd8\xff" or b[:8] == b"\x89PNG\r\n\x1a\n":
        return "image"
    if b"SECTION" in b or Path(path).suffix.lower() == ".dxf":
        return "dxf"
    raise ValueError("Unsupported input (use PDF, DWG, DXF, JPG or PNG). HEIC: export as JPG first.")


def dwg_to_dxf(dwg, out_dir):
    """LibreDWG first; ODA File Converter (if you installed it) as fallback."""
    out = Path(out_dir) / (Path(dwg).stem + ".dxf")
    tool = WS / "opt/libredwg/bin/dwg2dxf"
    try:
        run([tool, "-y", "--as", "r2018", "-o", out, dwg])
    except Exception as e:
        log(f"LibreDWG failed: {e}")
    if out.exists() and out.stat().st_size > 2000:
        return out, "libredwg"
    oda = Path(os.environ.get("PIPE_ODA", WS / "opt/oda/ODAFileConverter"))
    if oda.exists():
        src = Path(out_dir) / "oda_in"
        src.mkdir(exist_ok=True)
        shutil.copy(dwg, src)
        run(["xvfb-run", "-a", oda, src, out_dir, "ACAD2018", "DXF", "0", "1", "*.DWG"])
        cand = Path(out_dir) / (Path(dwg).stem + ".dxf")
        if cand.exists() and cand.stat().st_size > 2000:   # ODA exit codes are unreliable
            return cand, "oda"
    raise RuntimeError("DWG conversion failed (LibreDWG, and ODA not installed or failed)")


# ---------- helpers ----------
def fill_polys(img, polys, value=255):
    pts = [np.round(np.asarray(p, dtype=np.float64) * 16).astype(np.int32) for p in polys if len(p) >= 3]
    if pts:
        cv2.fillPoly(img, pts, value, lineType=cv2.LINE_8, shift=4)


def draw_lines(img, polylines, width=1, value=255):
    for p in polylines:
        if len(p) >= 2:
            pts = np.round(np.asarray(p, dtype=np.float64) * 16).astype(np.int32)
            cv2.polylines(img, [pts], False, value, max(1, int(round(width))), cv2.LINE_8, shift=4)


def thin_enclosed(lines_mask, max_th_px, min_len_px):
    """Fill thin closed regions between parallel lines (hollow double-line walls)."""
    free = (lines_mask == 0).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(free, connectivity=4)
    if n <= 1:
        return np.zeros_like(lines_mask)
    dt = cv2.distanceTransform(free, cv2.DIST_L2, 3)
    mx = ndimage.maximum(dt, lab, index=np.arange(1, n))
    h, w = lines_mask.shape
    keep = np.zeros(n, bool)
    for i in range(1, n):
        x, y, bw, bh, a = stats[i]
        border = x == 0 or y == 0 or x + bw >= w or y + bh >= h
        if not border and mx[i - 1] * 2 <= max_th_px and max(bw, bh) >= min_len_px:
            keep[i] = True
    return np.where(keep[lab], 255, 0).astype(np.uint8)


def wall_like(poly, lo, hi, min_len):
    r = cv2.minAreaRect(np.asarray(poly, np.float32))
    a, b = sorted(r[1])
    return lo <= a <= hi and b >= min_len


def axis_lines(polylines, tol_deg=1.0):
    """Split segments into horizontal (y, x0, x1) and vertical (x, y0, y1) lines for snapping."""
    hl, vl = [], []
    for p in polylines:
        for (x0, y0), (x1, y1) in zip(p[:-1], p[1:]):
            dx, dy = x1 - x0, y1 - y0
            L = math.hypot(dx, dy)
            if L < 3:
                continue
            ang = abs(math.degrees(math.atan2(dy, dx))) % 180
            if ang < tol_deg or ang > 180 - tol_deg:
                hl.append([(y0 + y1) / 2, min(x0, x1), max(x0, x1)])
            elif abs(ang - 90) < tol_deg:
                vl.append([(x0 + x1) / 2, min(y0, y1), max(y0, y1)])
    return hl, vl


def layer_kind(name):
    f = fold(name.replace("-", " ").replace("_", " "))
    for keys, k in ((WALL_KEYS, "wall"), (DOOR_KEYS, "door"), (WIN_KEYS, "window"), (SKIP_KEYS, "skip")):
        if any(key in f for key in keys):
            return k
    return "other"


# ---------- DXF ----------
def parse_dxf(path, out):
    import ezdxf
    from ezdxf import recover, path as ezpath
    doc, auditor = recover.readfile(str(path))
    warnings = [f"DXF audit: {len(auditor.errors)} errors fixed"] if auditor.errors else []
    unit = UNITS_M.get(int(doc.header.get("$INSUNITS", 0) or 0))
    items = []   # (kind, layer_kind, geometry)

    def walk(ent, parent_layer=None, depth=0):
        lay = ent.dxf.get("layer", "0")
        if lay == "0" and parent_layer:
            lay = parent_layer
        t = ent.dxftype()
        if t == "INSERT" and depth < 6:
            if ent.dxf.name.startswith("*X") or ent.dxf.name.upper().startswith("XREF"):
                warnings.append(f"external reference {ent.dxf.name} ignored")
            try:
                for v in ent.virtual_entities():
                    walk(v, lay, depth + 1)
            except Exception:
                pass
            return
        lk = layer_kind(lay)
        if t in ("TEXT", "MTEXT", "ATTRIB"):
            txt = ent.plain_text() if t == "MTEXT" else ent.dxf.text
            p = ent.dxf.insert
            hgt = ent.dxf.get("char_height", None) if t == "MTEXT" else ent.dxf.get("height", 0.2)
            for i, line in enumerate(repair(txt).splitlines()):
                if line.strip():
                    items.append(("text", lk, (line.strip(), p.x, p.y - i * (hgt or 0.2) * 1.4, hgt or 0.2)))
            return
        if t == "DIMENSION":
            try:
                items.append(("dim", lk, (ent.get_measurement(), ent.dxf.get("text", ""))))
            except Exception:
                pass
            return
        if lk == "skip":
            return
        try:
            if t == "HATCH":
                for pth in ezpath.from_hatch(ent):
                    pts = [(v.x, v.y) for v in pth.flattening(0.01 / (unit or 0.01))]
                    items.append(("fill", lk, pts))
                return
            if t in ("SOLID", "TRACE"):
                v = [ent.dxf.vtx0, ent.dxf.vtx1, ent.dxf.vtx3, ent.dxf.vtx2]
                items.append(("fill", lk, [(p.x, p.y) for p in v]))
                return
            pth = ezpath.make_path(ent)
            pts = [(v.x, v.y) for v in pth.flattening(0.01 / (unit or 0.01))]
            closed = bool(getattr(ent, "closed", False)) or (t in ("CIRCLE",))
            width = ent.dxf.get("const_width", 0) if t == "LWPOLYLINE" else 0
            items.append(("line", lk, (pts, closed, width)))
        except Exception:
            pass

    for e in doc.modelspace():
        walk(e)
    if unit is None:   # unitless drawing: guess from dimension texts, else from extents
        dims = [(m, s) for k, _, (m, s) in [i for i in items if i[0] == "dim"] if m]
        unit = 0.01
        for m, s in dims:
            try:
                v = float(str(s).replace(",", "."))
                for u in (0.001, 0.01, 1.0):
                    if abs(m * u - v * 0.01) / max(v * 0.01, 1e-6) < 0.02 or abs(m * u - v) / max(v, 1e-6) < 0.02:
                        unit = u
            except ValueError:
                continue
        warnings.append(f"$INSUNITS missing - assumed {unit} m per drawing unit")
    # plan extents from wall geometry (pick the biggest cluster if the file has several drawings)
    wpts = [p for k, lk, g in items if lk == "wall" for p in (g if k == "fill" else g[0])]
    if not wpts:
        wpts = [p for k, lk, g in items if k in ("fill", "line") for p in (g if k == "fill" else g[0])]
        warnings.append("no wall layer found - using all geometry")
    W = np.asarray(wpts) * unit
    x0, y0 = W.min(0) - 3.0
    x1, y1 = W.max(0) + 3.0
    near = [p for k, lk, g in items if k in ("fill", "line") for p in (g if k == "fill" else g[0])]
    near = np.asarray(near) * unit
    near = near[(near[:, 0] > x0) & (near[:, 0] < x1) & (near[:, 1] > y0) & (near[:, 1] < y1)]
    x0, y0 = np.minimum(W.min(0), near.min(0)) - 0.5
    x1, y1 = np.maximum(W.max(0), near.max(0)) + 0.5
    if max(x1 - x0, y1 - y0) > 45:
        coarse = 0.25
        cw, ch = int((x1 - x0) / coarse) + 1, int((y1 - y0) / coarse) + 1
        cm = np.zeros((ch, cw), np.uint8)
        idx = ((W - [x0, y0]) / coarse).astype(int)
        cm[ch - 1 - idx[:, 1], idx[:, 0]] = 1
        cm = cv2.dilate(cm, np.ones((9, 9), np.uint8))
        n, lab, stats, _ = cv2.connectedComponentsWithStats(cm, 8)
        best = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        bx, by, bw, bh, _ = stats[best]
        if n > 2:
            warnings.append(f"{n - 1} drawing groups found - using the largest (check overlay)")
        nx0, nx1 = x0 + bx * coarse - 1, x0 + (bx + bw) * coarse + 1
        ny1, ny0 = y1 - by * coarse + 1, y1 - (by + bh) * coarse - 1
        x0, x1, y0, y1 = nx0, nx1, ny0, ny1
    res = max(0.01, max(x1 - x0, y1 - y0) / 4000)
    H, Wd = int((y1 - y0) / res) + 1, int((x1 - x0) / res) + 1

    def px(pts):   # drawing units -> continuous pixel coords (pixel i covers [i-0.5, i+0.5])
        a = np.asarray(pts, dtype=np.float64) * unit
        return np.stack([(a[:, 0] - x0) / res - 0.5, (y1 - a[:, 1]) / res - 0.5], 1)

    walls, lines, strokes = (np.zeros((H, Wd), np.uint8) for _ in range(3))
    wall_polys = []
    for k, lk, g in items:
        if k == "fill":
            P = px(g)
            wl = wall_like(P, 0.03 / res, 0.6 / res, 0.2 / res) if len(P) >= 3 else False
            if lk == "wall" or (lk == "other" and wl):
                fill_polys(walls, [P])
            fill_polys(strokes, [P])
        elif k == "line":
            pts, closed, width = g
            P = px(pts + ([pts[0]] if closed and pts else []))
            wpx = max(1, width * unit / res)
            draw_lines(strokes, [P], wpx)
            if lk == "wall":
                draw_lines(lines, [P], 1)
                wall_polys.append(P)
                if width * unit >= 0.05:
                    draw_lines(walls, [P], wpx)
    walls |= thin_enclosed(lines, 0.55 / res, 0.25 / res) | lines
    strokes |= walls
    texts = []
    for k, lk, g in items:
        if k == "text" and lk != "skip":
            s, tx, ty, th = g
            u, v = px([(tx, ty)])[0]
            if 0 <= u < Wd and 0 <= v < H:
                texts.append({"text": s, "x": float(u + 0.3 * len(s) * th * unit / res),
                              "y": float(v - 0.5 * th * unit / res), "source": "dxf"})
    hl, vl = axis_lines(wall_polys)
    bg = (255 - (strokes > 0) * 150 - (walls > 0) * 60).astype(np.uint8)
    return dict(kind="dxf", m_per_px=res, walls=walls, strokes=strokes, bg=bg, texts=texts,
                hlines=hl, vlines=vl, hints=[], warnings=warnings,
                dims=[g for k, _, g in items if k == "dim"])


# ---------- PDF ----------
def lum(c):
    if c is None:
        return 1.0
    if isinstance(c, (int, float)):
        return float(c)
    c = list(c)
    if len(c) == 1:
        return float(c[0])
    if len(c) == 3:
        return 0.3 * c[0] + 0.59 * c[1] + 0.11 * c[2]
    if len(c) == 4:
        return 1 - min(1, c[3] + 0.3 * c[0] + 0.59 * c[1] + 0.11 * c[2])
    return 0.5


def path_points(obj):
    """Flatten pdfplumber path commands (top-based coords) into polylines."""
    out, cur = [], []
    for cmd in obj.get("path") or []:
        op = cmd[0]
        if op == "m":
            if len(cur) > 1:
                out.append(cur)
            cur = [cmd[1]]
        elif op == "l":
            cur.append(cmd[1])
        elif op == "c" and cur:
            p0, (p1, p2, p3) = cur[-1], cmd[1:4]
            for t in np.linspace(0, 1, 13)[1:]:
                a, b, c_, d = (1 - t) ** 3, 3 * (1 - t) ** 2 * t, 3 * (1 - t) * t ** 2, t ** 3
                cur.append((a * p0[0] + b * p1[0] + c_ * p2[0] + d * p3[0],
                            a * p0[1] + b * p1[1] + c_ * p2[1] + d * p3[1]))
        elif op == "h" and cur:
            cur.append(cur[0])
    if len(cur) > 1:
        out.append(cur)
    return out or ([obj["pts"]] if obj.get("pts") else [])


def word_phrases(words):
    """Group PDF words into phrases: same line and small gaps (pdfplumber lines span the page)."""
    words = sorted(words, key=lambda w: (round(w["top"], 0), w["x0"]))
    out = []
    for w in words:
        sz = w.get("size") or (w["bottom"] - w["top"])
        last = out[-1] if out else None
        if last and abs(w["top"] - last["top"]) < 0.3 * sz and 0 <= w["x0"] - last["x1"] < 1.2 * sz:
            last.update(text=last["text"] + " " + w["text"], x1=w["x1"], bottom=max(last["bottom"], w["bottom"]))
        else:
            out.append({"text": w["text"], "x0": w["x0"], "x1": w["x1"], "top": w["top"], "bottom": w["bottom"]})
    return out


def parse_pdf(path, out):
    import pdfplumber
    with pdfplumber.open(str(path)) as pdf:
        scores = []
        for i, pg in enumerate(pdf.pages[:20]):
            nvec = len(pg.lines) + len(pg.rects) + len(pg.curves)
            cover = sum((im["x1"] - im["x0"]) * (im["bottom"] - im["top"]) for im in pg.images)
            scores.append((nvec, cover / (pg.width * pg.height), i))
        nvec, cover, pno = max(scores)
        is_vector = nvec >= 300 or (nvec >= 30 and cover < 0.6)
        if not is_vector:
            best = max(scores, key=lambda s: s[1])
            return parse_raster(path, out, kind="pdf_scan", pdf_page=best[2])
        pg = pdf.pages[pno]
        lines_txt = word_phrases(pg.extract_words(x_tolerance=2, extra_attrs=["size"]))
        note = next((parse_scale(ln["text"]) for ln in lines_txt if parse_scale(ln["text"])), None)
        m_per_pt = 0.0254 / 72 * note if note else None
        ppp = max(4, min(10, math.ceil(m_per_pt / 0.006))) if m_per_pt else 6
        ppp = min(ppp, 7000 / max(pg.width, pg.height))
        H, Wd = int(pg.height * ppp) + 1, int(pg.width * ppp) + 1
        px = lambda pts: np.asarray(pts, np.float64) * ppp - 0.5
        walls, strokes = np.zeros((H, Wd), np.uint8), np.zeros((H, Wd), np.uint8)
        objs = [("line", o) for o in pg.lines] + [("rect", o) for o in pg.rects] + [("curve", o) for o in pg.curves]
        lws = [o.get("linewidth") or 0 for _, o in objs if o.get("stroke", True)]
        lw_med = float(np.median([w for w in lws if w > 0])) if any(w > 0 for w in lws) else 0.5
        seg_polys, thick = [], []
        for typ, o in objs:
            if typ == "rect":
                polys = [[(o["x0"], o["top"]), (o["x1"], o["top"]), (o["x1"], o["bottom"]),
                          (o["x0"], o["bottom"]), (o["x0"], o["top"])]]
            elif typ == "line":
                polys = [[(o["x0"], o["top"]), (o["x1"], o["bottom"])]] if not o.get("path") else path_points(o)
            else:
                polys = path_points(o)
            P = [px(p) for p in polys]
            if o.get("fill") and lum(o.get("non_stroking_color")) < 0.55:
                fill_polys(strokes, P)
                for q in P:
                    if len(q) >= 3 and (wall_like(q, 1.5, 0.6 / ((m_per_pt or 0.03528) / ppp), 3)
                                        or cv2.contourArea(q.astype(np.float32)) < (0.5 / ((m_per_pt or 0.03528) / ppp)) ** 2):
                        fill_polys(walls, [q])
            lw = o.get("linewidth") or 0
            if o.get("stroke", True) and lum(o.get("stroking_color")) < 0.7:
                draw_lines(strokes, P, max(1, lw * ppp))
                seg_polys += P
                if lw >= max(2.5 * lw_med, 1.0):
                    draw_lines(walls, P, lw * ppp)
                    thick += P
        if walls.sum() / 255 < 0.002 * H * Wd:   # no poché: hollow double-line walls
            ms = (m_per_pt or 0.03528) / ppp
            walls |= thin_enclosed(strokes, 0.55 / ms, 0.3 / ms)
        texts = []
        for ln in lines_txt:
            u, v = px([((ln["x0"] + ln["x1"]) / 2, (ln["top"] + ln["bottom"]) / 2)])[0]
            texts.append({"text": ln["text"], "x": float(u), "y": float(v), "source": "pdf"})
        hl, vl = axis_lines(thick or seg_polys)
        bg = np.asarray(pg.to_image(resolution=72 * ppp).original.convert("L"))[:H, :Wd]
        if bg.shape != walls.shape:
            bg = cv2.resize(bg, (Wd, H))
        hints = [{"method": "scale_note", "m_per_px": m_per_pt / ppp, "note": f"1/{note}"}] if note else []
        return dict(kind="pdf_vector", m_per_px=None, walls=walls, strokes=strokes | walls, bg=bg,
                    texts=texts, hlines=hl, vlines=vl, hints=hints, page=pno,
                    warnings=[] if texts else ["PDF has no text layer - VLM will read the page"],
                    needs_vlm=not texts)


# ---------- raster (scanned PDF, photo) ----------
def find_sheet(img):
    """Perspective-correct a phone photo: find the paper as the biggest bright 4-corner shape."""
    g = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (7, 7), 0)
    _, th = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) < 0.2 * img.shape[0] * img.shape[1]:
        return None
    ap = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)
    if len(ap) != 4:
        return None
    p = ap.reshape(4, 2).astype(np.float32)
    s, d = p.sum(1), np.diff(p, axis=1).ravel()
    tl, br, tr, bl = p[np.argmin(s)], p[np.argmax(s)], p[np.argmin(d)], p[np.argmax(d)]
    wpx = max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl))
    hpx = max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr))
    asp = wpx / hpx                     # snap to ISO paper (A4/A3: sqrt 2) - photos distort the ratio
    for iso in (math.sqrt(2), 1 / math.sqrt(2)):
        if abs(asp / iso - 1) < 0.15:
            asp = iso
    k = 4000 / max(wpx, hpx)
    W, H = (4000, int(4000 / asp)) if asp >= 1 else (int(4000 * asp), 4000)
    M = cv2.getPerspectiveTransform(np.float32([tl, tr, br, bl]), np.float32([[0, 0], [W, 0], [W, H], [0, H]]))
    return cv2.warpPerspective(img, M, (W, H), flags=cv2.INTER_CUBIC, borderValue=(255, 255, 255))


def deskew(gray):
    """Find the small rotation (+/-4 deg) that makes wall lines straight (projection profile)."""
    k = 1200 / max(gray.shape)
    small = cv2.resize(gray, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
    ink = (small < cv2.threshold(small, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0]).astype(np.float32)
    h, w = ink.shape
    best, best_a = -1, 0.0
    for a in np.arange(-4, 4.001, 0.05):
        M = cv2.getRotationMatrix2D((w / 2, h / 2), a, 1.0)
        r = cv2.warpAffine(ink, M, (w, h))
        sc = r.sum(1).var() + r.sum(0).var()
        if sc > best:
            best, best_a = sc, a
    if abs(best_a) < 0.05:
        return gray, 0.0
    H, W = gray.shape
    M = cv2.getRotationMatrix2D((W / 2, H / 2), best_a, 1.0)
    return cv2.warpAffine(gray, M, (W, H), flags=cv2.INTER_CUBIC, borderValue=255), float(best_a)


def parse_raster(path, out, kind="photo", pdf_page=0):
    from skimage.filters import threshold_sauvola
    warnings, dpi = [], None
    if kind == "pdf_scan":
        import pypdfium2 as pdfium
        dpi = 300
        pil = pdfium.PdfDocument(str(path))[pdf_page].render(scale=dpi / 72).to_pil()
        img = cv2.cvtColor(np.asarray(pil.convert("RGB")), cv2.COLOR_RGB2BGR)
    else:
        from PIL import Image, ImageOps
        pil = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
        img = cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)
        sheet = find_sheet(img)
        if sheet is None:
            warnings.append("paper edges not found - no perspective correction (keep the whole sheet in the photo)")
        else:
            img = sheet
    k = min(1.0, 5000 / max(img.shape[:2]))
    if k < 1:
        img = cv2.resize(img, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
        dpi = dpi * k if dpi else None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bgb = cv2.GaussianBlur(gray, (0, 0), max(gray.shape) / 40)
    gray = np.clip(gray.astype(np.float32) / np.maximum(bgb, 1) * 235, 0, 255).astype(np.uint8)
    gray, ang = deskew(gray)
    if abs(ang) > 0.05:
        warnings.append(f"deskewed by {ang:.2f} degrees")
    otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0]
    ink = (gray < min(otsu, 170)) | ((gray < threshold_sauvola(gray, window_size=31, k=0.15)) & (gray < 200))
    strokes = ink.astype(np.uint8) * 255
    n, lab, stats, _ = cv2.connectedComponentsWithStats(strokes, 8)
    small = stats[:, cv2.CC_STAT_AREA] < 12
    small[0] = False
    strokes[small[lab]] = 0
    # stroke widths along the skeleton: thin lines vs thick walls
    from skimage.morphology import skeletonize
    dt = cv2.distanceTransform(strokes, cv2.DIST_L2, 3)
    sk = skeletonize(strokes > 0)
    wid = 2 * dt[sk]
    thin = float(np.percentile(wid, 30)) if wid.size else 2.0
    ksz = int(math.ceil(thin * 2.2)) + 2
    walls = cv2.morphologyEx(strokes, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (ksz, ksz)))
    if walls.sum() / 255 < 0.004 * walls.size:
        warnings.append("walls look hollow (double lines) - filling thin enclosed regions")
        walls |= thin_enclosed(strokes, 12 * ksz, 6 * ksz)
    return dict(kind=kind, m_per_px=None, walls=walls, strokes=strokes, bg=gray, texts=[],
                hlines=[], vlines=[], hints=[], dpi=dpi, warnings=warnings, needs_vlm=True, page=pdf_page)


# ---------- stage entry ----------
def run_parse(inp, job):
    out = Path(job) / "01_parse"
    out.mkdir(parents=True, exist_ok=True)
    kind = detect(inp)
    src = Path(inp)
    conv = None
    if kind == "dwg":
        src, conv = dwg_to_dxf(inp, Path(job) / "00_input")
        kind = "dxf"
    if kind == "dxf":
        ex = parse_dxf(src, out)
        if conv:
            ex["kind"], ex["converter"] = "dwg", conv
    elif kind == "pdf":
        ex = parse_pdf(src, out)
    else:
        ex = parse_raster(src, out, kind="photo")
    cv2.imwrite(str(out / "wall_mask.png"), ex.pop("walls"))
    cv2.imwrite(str(out / "strokes.png"), ex.pop("strokes"))
    cv2.imwrite(str(out / "page.png"), ex.pop("bg"))
    ex.setdefault("needs_vlm", False)
    ex["input"] = str(inp)
    if ex.get("dims"):
        ex["dims"] = [[float(m), str(s)] for m, s in ex["dims"]][:200]
    jsave(out / "extract.json", ex)
    log(f"parse: kind={ex['kind']} texts={len(ex['texts'])} needs_vlm={ex['needs_vlm']} warnings={ex['warnings']}")
    return ex
__END_OF_PARSE_PY__
  cat > "$APP/vlm.py" <<'__END_OF_VLM_PY__'
"""Stage 1b (GPU) - read room names, m2 areas and scale notes on scans/photos with Qwen3.5.
Run: python -m app.vlm <job_dir>. Runs in its own process, so the GPU is free again afterwards."""
import json, re, sys, time
from pathlib import Path
from .common import jload, jsave, load_config, log
from .textnorm import fold

PROMPT = ("This image is (part of) an apartment floor plan, usually Turkish. Read every text you can see: "
          "room names (for example SALON, YATAK ODASI, MUTFAK, BANYO, WC, HOL, BALKON), room areas like "
          "12,50 m², dimension numbers and scale notes like ÖLÇEK 1/50. Keep Turkish letters exactly. "
          'Reply ONLY with a JSON list like [{"text": "SALON", "bbox_2d": [x1, y1, x2, y2]}] '
          "with box coordinates from 0 to 1000 relative to this image. No other words.")


def parse_reply(txt, w, h):
    """JSON list of {text, bbox_2d} (0-1000 relative) -> [{text, x, y}] in this image's pixels."""
    txt = re.sub(r"<think>.*?</think>", "", txt, flags=re.S)
    items = []
    m = re.search(r"\[.*\]", txt, re.S)
    if m:
        try:
            items = json.loads(m.group(0))
        except Exception:
            items = []
    if not items:
        for mm in re.finditer(r'"text"\s*:\s*"([^"]+)"\s*,\s*"bbox_2d"\s*:\s*\[([^\]]+)\]', txt):
            try:
                items.append({"text": mm.group(1), "bbox_2d": [float(v) for v in mm.group(2).split(",")]})
            except ValueError:
                pass
    out = []
    for it in items if isinstance(items, list) else []:
        if not isinstance(it, dict) or not str(it.get("text", "")).strip() or len(it.get("bbox_2d") or []) != 4:
            continue
        try:
            x1, y1, x2, y2 = [float(v) for v in it["bbox_2d"]]
        except (TypeError, ValueError):
            continue
        if max(x1, y1, x2, y2) <= 1000:
            x1, x2, y1, y2 = x1 / 1000 * w, x2 / 1000 * w, y1 / 1000 * h, y2 / 1000 * h
        out.append({"text": str(it["text"]).strip(), "x": (x1 + x2) / 2, "y": (y1 + y2) / 2})
    return out


def tiles(W, H):
    """Whole page + 2x2 overlapping tiles (small labels are easier to read in tiles)."""
    boxes = [(0, 0, W, H)]
    ox, oy = int(W * 0.05), int(H * 0.05)
    for i in range(2):
        for j in range(2):
            boxes.append((max(0, i * W // 2 - ox), max(0, j * H // 2 - oy), min(W, (i + 1) * W // 2 + ox), min(H, (j + 1) * H // 2 + oy)))
    return boxes


def merge(found, diag):
    out = []
    for f in found:
        k = fold(f["text"])
        if not k:
            continue
        if any(fold(o["text"]) == k and abs(o["x"] - f["x"]) + abs(o["y"] - f["y"]) < 0.03 * diag for o in out):
            continue
        out.append(dict(f, source="vlm"))
    return out


def main(job):
    import torch
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor
    job = Path(job)
    cfg = load_config()
    mid = cfg["vlm_model"]
    if mid == "auto":
        mid = "Qwen/Qwen3.5-9B" if torch.cuda.get_device_properties(0).total_memory >= 40e9 else "Qwen/Qwen3.5-4B"
    t0 = time.time()
    proc = AutoProcessor.from_pretrained(mid)
    model = AutoModelForImageTextToText.from_pretrained(mid, dtype=torch.bfloat16, device_map="cuda").eval()
    log(f"vlm: loaded {mid} in {time.time() - t0:.0f}s")
    page = Image.open(job / "01_parse/page.png").convert("RGB")
    W, H = page.size
    found, raw = [], []
    for (x0, y0, x1, y1) in tiles(W, H):
        crop = page.crop((x0, y0, x1, y1))
        k = min(1.0, cfg["vlm_max_side"] / max(crop.size))
        img = crop.resize((int(crop.size[0] * k), int(crop.size[1] * k)), Image.LANCZOS) if k < 1 else crop
        msgs = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": PROMPT}]}]
        kw = dict(add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt")
        try:
            inputs = proc.apply_chat_template(msgs, enable_thinking=False, **kw)
        except TypeError:
            inputs = proc.apply_chat_template(msgs, **kw)
        inputs = inputs.to(model.device)
        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=1536, do_sample=False)
        txt = proc.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
        raw.append(txt)
        for f in parse_reply(txt, img.size[0], img.size[1]):
            found.append({"text": f["text"], "x": x0 + f["x"] / k, "y": y0 + f["y"] / k})
    texts = merge(found, (W ** 2 + H ** 2) ** 0.5)
    jsave(job / "01_parse/vlm_texts.json", texts)
    (job / "01_parse/vlm_raw.txt").write_text("\n\n----\n\n".join(raw), encoding="utf-8")
    log(f"vlm: {len(texts)} texts in {time.time() - t0:.0f}s: {[t['text'] for t in texts][:12]}")


if __name__ == "__main__":
    main(sys.argv[1])
__END_OF_VLM_PY__
  cat > "$APP/plan.py" <<'__END_OF_PLAN_PY__'
"""Stage 2 - geometry core + scale solver -> 02_plan/plan.json + overlay.png.
Works in mask pixels first (same for every input), converts to metres at the end."""
import math
from pathlib import Path
import cv2
import numpy as np
from scipy import ndimage
from shapely.geometry import Polygon, Point, box
from .common import jload, jsave, log, font_path
from .textnorm import classify

MAIN_TYPES = ("living", "bedroom", "kitchen", "bathroom")


# ---------- 1. walls: split the wall mask into straight rectangles ----------
def wall_runs(mask, axis, min_len, max_th, tol):
    min_len |= 1                                     # odd kernel: no 1-px shift
    k = (min_len, 1) if axis == "h" else (1, min_len)
    m = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, k))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, connectivity=4)
    out = []
    for i in range(1, n):
        x, y, w, h, _ = st[i]
        sub = lab[y:y + h, x:x + w] == i
        if axis == "v":
            sub = sub.T                      # rows = across the wall, columns = along it
        has = sub.any(0)
        top = ndimage.median_filter(np.argmax(sub, 0), 5, mode="nearest")
        bot = ndimage.median_filter(sub.shape[0] - 1 - np.argmax(sub[::-1], 0), 5, mode="nearest")
        segs, start = [], None
        for c in range(sub.shape[1]):
            if not has[c]:
                if start is not None:
                    segs.append((start, c - 1))
                start = None
                continue
            if start is None:
                start = c
            elif abs(top[c] - top[c - 1]) > tol or abs(bot[c] - bot[c - 1]) > tol:
                segs.append((start, c - 1))
                start = c
        if start is not None:
            segs.append((start, sub.shape[1] - 1))
        for s0, s1 in segs:
            t0, t1 = int(np.median(top[s0:s1 + 1])), int(np.median(bot[s0:s1 + 1]))
            if s1 - s0 + 1 < min_len or not (1 <= t1 - t0 + 1 <= max_th):
                continue
            if axis == "h":
                out.append(dict(axis="h", x0=x + s0, x1=x + s1, y0=y + t0, y1=y + t1))
            else:
                out.append(dict(axis="v", x0=x + t0, x1=x + t1, y0=y + s0, y1=y + s1))
    for r in out:
        finish(r)
    return out


def finish(r):
    if r["axis"] == "h":
        r.update(c=(r["y0"] + r["y1"]) / 2, th=r["y1"] - r["y0"] + 1, s=r["x0"], e=r["x1"])
    else:
        r.update(c=(r["x0"] + r["x1"]) / 2, th=r["x1"] - r["x0"] + 1, s=r["y0"], e=r["y1"])
    return r


def trim_vertical(hs, vs):
    """Vertical walls stop at horizontal walls, so wall boxes never overlap in 3D."""
    out = []
    for v in vs:
        cuts = sorted((h["y0"], h["y1"]) for h in hs if h["x0"] <= v["x1"] and h["x1"] >= v["x0"]
                      and h["y0"] <= v["y1"] and h["y1"] >= v["y0"])
        pos = v["y0"]
        for a, b in cuts + [(v["y1"] + 1, v["y1"] + 1)]:
            if a - 1 >= pos + 1:
                out.append(finish(dict(v, y0=pos, y1=a - 1)))
            pos = max(pos, b + 1)
    return out


# ---------- 2. openings: gaps between wall pieces ----------
def find_openings(rects, mask, strokes_d, est):
    H, W = mask.shape
    lo, hi, door_max = int(0.45 / est), int(3.2 / est), 1.3 / est
    found = []

    def at_junction(r, end):
        """True if this wall end sits on a perpendicular wall (T or L junction): no gap to look for."""
        for q in rects:
            if q["axis"] == r["axis"]:
                continue
            if r["axis"] == "h" and q["x0"] - 2 <= end <= q["x1"] + 2 and q["y0"] <= r["y1"] + 3 and q["y1"] >= r["y0"] - 3:
                return True
            if r["axis"] == "v" and q["y0"] - 2 <= end <= q["y1"] + 2 and q["x0"] <= r["x1"] + 3 and q["x1"] >= r["x0"] - 3:
                return True
        return False

    for r in rects:
        for d in (-1, 1):
            end = r["s"] if d < 0 else r["e"]
            if at_junction(r, end):
                continue
            offs = [r["c"] - r["th"] / 4, r["c"], r["c"] + r["th"] / 4]
            hit = None
            for step in range(1, hi + 2):
                p = end + d * step
                if not (0 <= p < (W if r["axis"] == "h" else H)):
                    break
                vals = [mask[int(round(o)), p] if r["axis"] == "h" else mask[p, int(round(o))] for o in offs]
                if sum(v > 0 for v in vals) >= 2:
                    hit = step
                    break
            if hit is None or not (lo <= hit - 1 <= hi):
                continue
            a, b = sorted((end + d, end + d * (hit - 1)))
            hp = end + d * hit
            other = next((q for q in rects if q is not r and q["axis"] == r["axis"] and abs(q["c"] - r["c"]) <= 2
                          and q["s"] - 2 <= hp <= q["e"] + 2), None)
            collinear = other is not None and abs(other["th"] - r["th"]) <= max(2, 0.3 * r["th"])
            if not collinear and (b - a + 1) > door_max and arc_score(strokes_d, dict(axis=r["axis"], c=r["c"], th=r["th"], a=a, b=b))[0] < 0.6:
                continue
            op = dict(axis=r["axis"], c=r["c"], th=r["th"], a=a, b=b, rect=r)
            if not any(o["axis"] == op["axis"] and abs(o["c"] - op["c"]) <= 3 and abs(o["a"] - a) <= 3
                       and abs(o["b"] - b) <= 3 for o in found):
                found.append(op)
    return found


def arc_score(strokes_d, op):
    """Look for a door swing arc next to the gap. Returns (score, hinge_end, side)."""
    best = (0.0, None, None)
    w = op["b"] - op["a"] + 1
    H, W = strokes_d.shape
    for hinge in ("a", "b"):
        hpos = op["a"] - 0.5 if hinge == "a" else op["b"] + 0.5
        toward = 1 if hinge == "a" else -1
        for side in (1, -1):
            face = op["c"] + side * op["th"] / 2
            hits = 0
            angs = np.radians(np.linspace(10, 80, 15))
            for t in angs:
                along, across = hpos + toward * w * math.cos(t), face + side * w * math.sin(t)
                x, y = (along, across) if op["axis"] == "h" else (across, along)
                xi, yi = int(round(x)), int(round(y))
                if 0 <= xi < W and 0 <= yi < H and strokes_d[yi, xi]:
                    hits += 1
            sc = hits / len(angs)
            if sc > best[0]:
                best = (sc, hinge, side)
    return best


# ---------- 3. rooms ----------
def orthogonalize(pts, tol_deg=8):
    n = len(pts)
    kinds = []
    for i in range(n):
        (x0, y0), (x1, y1) = pts[i], pts[(i + 1) % n]
        a = abs(math.degrees(math.atan2(y1 - y0, x1 - x0))) % 180
        kinds.append("h" if min(a, 180 - a) < tol_deg else "v" if abs(a - 90) < tol_deg else "d")
    out = []
    for i in range(n):
        ki, ko = kinds[i - 1], kinds[i]
        (xp, yp), (x, y), (xn, yn) = pts[i - 1], pts[i], pts[(i + 1) % n]
        if ki == ko and ki in "hv":
            continue
        if {ki, ko} == {"h", "v"}:
            hx = (x + xn) / 2 if ko == "v" else (xp + x) / 2
            hy = (y + yn) / 2 if ko == "h" else (yp + y) / 2
            out.append((hx, hy))
        else:
            out.append((x, y))
    return out if len(out) >= 3 else pts


def snap_edges(poly, hl, vl, tol):
    """Move axis-aligned polygon edges onto nearby vector wall-face lines (exact vector geometry)."""
    pts = list(poly.exterior.coords)[:-1]
    n = len(pts)
    ys, xs = {}, {}
    for i in range(n):
        (x0, y0), (x1, y1) = pts[i], pts[(i + 1) % n]
        if abs(y1 - y0) < 1e-6 and hl:
            lo_, hi_ = sorted((x0, x1))
            c = [l[0] for l in hl if abs(l[0] - y0) <= tol and min(hi_, l[2]) - max(lo_, l[1]) > 0.3 * (hi_ - lo_)]
            if c:
                ys[i] = min(c, key=lambda v: abs(v - y0))
        elif abs(x1 - x0) < 1e-6 and vl:
            lo_, hi_ = sorted((y0, y1))
            c = [l[0] for l in vl if abs(l[0] - x0) <= tol and min(hi_, l[2]) - max(lo_, l[1]) > 0.3 * (hi_ - lo_)]
            if c:
                xs[i] = min(c, key=lambda v: abs(v - x0))
    new = []
    for i in range(n):
        x, y = pts[i]
        for e in (i - 1 if i > 0 else n - 1, i):
            if e in ys:
                y = ys[e]
            if e in xs:
                x = xs[e]
        new.append((x, y))
    p = Polygon(new)
    return p if p.is_valid and abs(p.area - poly.area) < 0.2 * poly.area else poly


def comp_polygon(comp_mask, est, hl, vl):
    cnts, _ = cv2.findContours(comp_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(cnts, key=cv2.contourArea)
    ap = cv2.approxPolyDP(c, max(1.0, 0.015 / est), True).reshape(-1, 2).astype(float)
    pts = orthogonalize([tuple(p) for p in ap])
    poly = Polygon(pts).buffer(0)
    if poly.geom_type != "Polygon":
        poly = max(poly.geoms, key=lambda g: g.area)
    poly = poly.buffer(0.5, join_style=2).simplify(0.05)
    if hl or vl:
        poly = snap_edges(poly, hl, vl, 0.03 / est)
    return poly


# ---------- 4. scale ----------
def weighted_median(votes):
    votes = sorted(votes)
    tot = sum(w for _, w in votes)
    acc = 0
    for v, w in votes:
        acc += w
        if acc >= tot / 2:
            return v
    return votes[-1][0]


def solve_scale(ex, rooms, doors_px, th_px, texts):
    checks, warnings = [], []
    lab = []
    for r in rooms:
        if r.get("label_area") and r["type"] != "balcony":
            lab.append((math.sqrt(r["label_area"] / r["poly"].area), 1.0, r))
    if ex.get("m_per_px"):
        s, method, conf = ex["m_per_px"], "dxf_units", 1.0
    else:
        votes = [(v, w) for v, w, _ in lab]
        note = next((h for h in ex.get("hints", []) if h["method"] == "scale_note"), None)
        if not note and ex.get("dpi"):
            n = next((t["value"] for t in texts if t["kind"] == "scale"), None)
            if n:
                note = {"method": "scale_note", "m_per_px": 0.0254 / ex["dpi"] * n, "note": f"1/{n} at {ex['dpi']:.0f} dpi"}
        if votes:
            med = weighted_median(votes)
            good = [(v, w) for v, w in votes if abs(v / med - 1) <= 0.02]
            s = weighted_median(good or votes)
            method, conf = "area_labels", min(0.95, 0.6 + 0.1 * len(good))
            if len(good) < len(votes):
                warnings.append(f"{len(votes) - len(good)} area label(s) disagree with the others by >2 %")
            if note:
                agree = abs(note["m_per_px"] / s - 1)
                checks.append({"method": "scale_note", "m_per_px": note["m_per_px"], "deviation_pct": round(100 * agree, 2)})
                if agree < 0.01 and ex["kind"] == "pdf_vector":
                    s, method, conf = note["m_per_px"], "scale_note+area_labels", 0.99
                elif agree >= 0.02:
                    warnings.append(f"printed scale {note['note']} disagrees with area labels by {100 * agree:.1f} % - using labels")
        elif note:
            s, method, conf = note["m_per_px"], "scale_note", 0.7 if ex["kind"] == "pdf_vector" else 0.4
            warnings.append("scale from the printed scale note only (no area labels found)")
        elif doors_px:
            s, method, conf = 0.90 / float(np.median(doors_px)), "door_width_0.90m", 0.25
            warnings.append("SCALE UNCERTAIN: only door widths available (expect +/-5-10 %)")
        else:
            s, method, conf = 0.18 / max(th_px, 1), "wall_thickness_guess", 0.1
            warnings.append("SCALE UNKNOWN: guessed from wall thickness - sizes are rough")
    for v, _, r in lab:
        checks.append({"method": "area_label", "room": r["id"], "label_m2": r["label_area"],
                       "measured_m2": round(r["poly"].area * s * s, 2),
                       "deviation_pct": round(100 * (r["poly"].area * s * s / r["label_area"] - 1), 2)})
    return s, method, conf, checks, warnings


# ---------- 5. main ----------
def run_plan(job, cfg):
    job = Path(job)
    ex = jload(job / "01_parse/extract.json")
    mask = cv2.imread(str(job / "01_parse/wall_mask.png"), 0)
    strokes = cv2.imread(str(job / "01_parse/strokes.png"), 0)
    warnings = list(ex.get("warnings", []))
    texts = list(ex.get("texts", [])) + jload(job / "01_parse/vlm_texts.json", [])
    for t in texts:
        t.update(classify(t["text"]))
    hl, vl = ex.get("hlines", []), ex.get("vlines", [])
    # typical wall thickness (px) from the medial axis of the wall mask
    from skimage.morphology import skeletonize
    dt = cv2.distanceTransform((mask > 0).astype(np.uint8), cv2.DIST_L2, 3)
    sk = skeletonize(mask > 0)
    th_px = float(np.median(2 * dt[sk])) if sk.any() else 10.0
    est0 = ex.get("m_per_px") or next((h["m_per_px"] for h in ex.get("hints", [])), None) \
        or (0.12 / max(1.0, float(np.percentile(2 * dt[sk], 25))) if sk.any() else 0.01)

    def geometry(est):
        """walls, openings, rooms and labels for one provisional scale (metres per pixel)."""
        tol = max(1, int(round(th_px * 0.15)))
        hs = wall_runs(mask, "h", max(3, int(0.3 / est)), int(0.65 / est), tol)
        vs = trim_vertical(hs, wall_runs(mask, "v", max(3, int(0.3 / est)), int(0.65 / est), tol))
        rects = hs + vs
        strokes_d = cv2.dilate(strokes, np.ones((5, 5), np.uint8)) > 0
        ops = find_openings(rects, mask, strokes_d, est)
        # rooms = enclosed free space once the openings are closed
        closed = mask.copy()
        for o in ops:
            t0, t1 = int(math.floor(o["c"] - o["th"] / 2 + 0.5)), int(math.ceil(o["c"] + o["th"] / 2 - 0.5))
            if o["axis"] == "h":
                closed[t0:t1 + 1, o["a"]:o["b"] + 1] = 255
            else:
                closed[o["a"]:o["b"] + 1, t0:t1 + 1] = 255
        free = np.pad((closed == 0).astype(np.uint8), 1, constant_values=1)
        n, lab = cv2.connectedComponents(free, connectivity=4)
        lab = lab[1:-1, 1:-1]
        ext = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]])).tolist())
        fdt = cv2.distanceTransform((lab > 0).astype(np.uint8) * (closed == 0), cv2.DIST_L2, 3)
        areas = np.bincount(lab.ravel(), minlength=n)
        maxdt = ndimage.maximum(fdt, lab, index=np.arange(n))
        rooms = []
        for i in range(1, n):
            if i in ext or areas[i] * est * est < 1.0 or maxdt[i] * est < 0.3:
                continue
            rooms.append({"lab": i, "poly": comp_polygon(lab == i, est, hl, vl), "labels": [], "areas": []})
        # balconies: labelled areas outside the walls, bounded by thin railing lines
        gk = max(5, int(0.05 / est) | 1)
        grow = np.ones((gk, gk), np.uint8)   # close small gaps where railing lines meet walls
        barrier = np.pad(((cv2.dilate(strokes, grow) > 0) | (closed > 0)).astype(np.uint8), 1, constant_values=0)
        bn, blab = cv2.connectedComponents((1 - barrier).astype(np.uint8), connectivity=4)
        blab = blab[1:-1, 1:-1]
        bext = set(np.unique(np.concatenate([blab[0], blab[-1], blab[:, 0], blab[:, -1]])).tolist())
        for t in texts:
            if t.get("type") != "balcony":
                continue
            x, y = int(round(t["x"])), int(round(t["y"]))
            if not (0 <= y < lab.shape[0] and 0 <= x < lab.shape[1]) or lab[y, x] not in ext:
                continue
            R = int(0.5 / est)     # the label text itself is ink: use the main free region around it
            win = blab[max(0, y - R):y + R + 1, max(0, x - R):x + R + 1].ravel()
            win = win[(win > 0) & ~np.isin(win, list(bext))]
            bid = int(np.bincount(win).argmax()) if win.size else 0
            comp = blab == bid
            if bid and bid not in bext and 1.0 < comp.sum() * est * est < 40:
                comp = (cv2.dilate(comp.astype(np.uint8), grow) > 0) & (closed == 0)
                comp = cv2.morphologyEx(comp.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8)) > 0
                rooms.append({"lab": -1, "poly": comp_polygon(comp, est, [], []), "labels": [t], "areas": [],
                              "balcony": True})
            else:
                warnings.append("balcony label found but its outline is not closed - balcony skipped")
        # attach labels and area texts to rooms
        for t in texts:
            if t["kind"] not in ("room_label", "area") or t.get("type") == "balcony":
                if t.get("type") == "balcony" and t.get("area"):
                    for r in rooms:
                        if r.get("balcony") and r["poly"].buffer(3).contains(Point(t["x"], t["y"])):
                            r["areas"].append(t["area"])
                continue
            p = Point(t["x"], t["y"])
            cand = [r for r in rooms if r["poly"].buffer(0.5 / est).contains(p)]
            if not cand:
                continue
            r = min(cand, key=lambda r: r["poly"].distance(p))
            if t["kind"] == "room_label":
                r["labels"].append(t)
                if t.get("area"):
                    r["areas"].append(t["area"])
            else:
                r["areas"].append(t["value"])
        for k, r in enumerate(rooms):
            r["id"] = f"r{k + 1}"
            lb = r["labels"][0] if r["labels"] else None
            r["type"] = "balcony" if r.get("balcony") else (lb["type"] if lb else None)
            r["label"] = lb["text"] if lb else ""
            r["extra"] = {k2: v for k2, v in (lb or {}).items() if k2 in ("master", "child", "open")}
            r["label_area"] = r["areas"][0] if r["areas"] else None
            if len(r["labels"]) > 1:
                r["zones"] = [{"type": l["type"], "x": l["x"], "y": l["y"]} for l in r["labels"][1:] if l["type"] != r["type"]]

        return est, rects, ops, closed, lab, ext, rooms, strokes_d

    est, rects, ops, closed, lab, ext, rooms, strokes_d = geometry(est0)
    # scale
    doors_px = [o["b"] - o["a"] + 1 for o in ops if o["b"] - o["a"] + 1 < 1.3 / est]
    s, method, conf, checks, sw = solve_scale(ex, rooms, doors_px, th_px, texts)
    if not ex.get("m_per_px") and abs(s / est - 1) > 0.08:   # second pass with the solved scale
        est, rects, ops, closed, lab, ext, rooms, strokes_d = geometry(s)
        doors_px = [o["b"] - o["a"] + 1 for o in ops if o["b"] - o["a"] + 1 < 1.3 / est]
        s, method, conf, checks, sw = solve_scale(ex, rooms, doors_px, th_px, texts)
    warnings += sw
    # unlabelled rooms: simple rules (a VLM pass can refine them later)
    for r in rooms:
        if r["type"]:
            continue
        a = r["poly"].area * s * s
        mnx, mny, mxx, mxy = r["poly"].bounds
        w, d = sorted(((mxx - mnx) * s, (mxy - mny) * s))
        r["type"] = "storage" if a < 2.5 else "hall" if (w < 1.6 and d / w > 2.2) else "other"
        r["confidence"] = 0.3
        warnings.append(f"room {r['id']} ({a:.1f} m2) has no label - typed as '{r['type']}' by shape")
    bed = [r for r in rooms if r["type"] == "bedroom"]
    if bed and not any(r["extra"].get("master") for r in bed):
        max(bed, key=lambda r: r["poly"].area)["extra"]["master"] = True
    # classify openings and pick door swings
    def region(pt):
        """room id, 'outside', or None (wall pixel / tiny gap)."""
        x, y = int(round(pt[0])), int(round(pt[1]))
        if not (0 <= y < lab.shape[0] and 0 <= x < lab.shape[1]):
            return "outside"
        L = lab[y, x]
        if L == 0:
            return None
        if L in ext:
            return next((r["id"] for r in rooms if r["lab"] == -1 and r["poly"].contains(Point(pt))), "outside")
        return next((r["id"] for r in rooms if r["lab"] == L), None)
    rmap = {r["id"]: r for r in rooms}
    doors, windows = [], []
    for o in ops:
        mid = (o["a"] + o["b"]) / 2
        off = o["th"] / 2 + 0.3 / est
        sides = {}
        for sgn in (1, -1):
            pt = (mid, o["c"] + sgn * off) if o["axis"] == "h" else (o["c"] + sgn * off, mid)
            sides[sgn] = region(pt)
        sc, hinge, side = arc_score(strokes_d, o)
        width_px = o["b"] - o["a"] + 1
        ins = [v for v in sides.values() if v and v != "outside" and rmap[v]["type"] != "balcony"]
        balc = [v for v in sides.values() if v and v != "outside" and rmap[v]["type"] == "balcony"]
        if not ins:
            continue
        if len(ins) == 2:
            kind = "hinged" if width_px * s <= 1.3 else "opening"
            if sc >= 0.6 and kind == "hinged":
                into = sides[side]
            else:
                a_, b_ = rmap[ins[0]], rmap[ins[1]]
                into = b_["id"] if a_["type"] == "hall" or (b_["type"] in ("bathroom", "wc") and a_["type"] != "hall") else a_["id"]
                side = next(k for k, v in sides.items() if v == into)
                hinge = None
            doors.append(dict(op=o, kind=kind, into=into, side=side, hinge=hinge, rooms=ins))
        elif sc >= 0.6 and not balc:
            doors.append(dict(op=o, kind="hinged", into=sides[side], side=side, hinge=hinge, rooms=ins))
        else:
            windows.append(dict(op=o, kind="balcony_door" if balc else "window", rooms=ins))
    # walls: merge collinear pieces through their openings
    groups = []
    for r in rects:
        g = next((g for g in groups if g["axis"] == r["axis"] and abs(g["c"] - r["c"]) <= 2
                  and abs(g["th"] - r["th"]) <= max(2, 0.3 * r["th"])), None)
        if g is None:
            g = {"axis": r["axis"], "c": r["c"], "th": r["th"], "iv": []}
            groups.append(g)
        g["iv"].append([r["s"], r["e"]])
        r["g"] = g
    for o in ops:
        o["rect"]["g"]["iv"].append([o["a"], o["b"]])
    walls = []
    for g in groups:
        iv = sorted(g["iv"])
        cur = list(iv[0])
        for a, b in iv[1:] + [[10 ** 9, 10 ** 9]]:
            if a <= cur[1] + 2:
                cur[1] = max(cur[1], b)
            else:
                walls.append({"axis": g["axis"], "c": g["c"], "th": g["th"], "s": cur[0], "e": cur[1]})
                cur = [a, b]
    # ---------- convert pixels -> metres ----------
    allpts = [p for r in rooms for p in r["poly"].exterior.coords]
    for w in walls:
        allpts += [(w["s"] - 0.5, w["c"]), (w["e"] + 0.5, w["c"])] if w["axis"] == "h" else [(w["c"], w["s"] - 0.5), (w["c"], w["e"] + 0.5)]
    xs = [(p[0] + 0.5) * s for p in allpts]
    ys = [-(p[1] + 0.5) * s for p in allpts]
    ox, oy = -min(xs), -min(ys)
    X = lambda u: round((u + 0.5) * s + ox, 4)
    Y = lambda v: round(-(v + 0.5) * s + oy, 4)
    hts = cfg["heights"]
    plan_walls = []
    for k, w in enumerate(walls):
        w["id"] = f"w{k + 1}"
        if w["axis"] == "h":
            a, b = [X(w["s"] - 0.5), Y(w["c"])], [X(w["e"] + 0.5), Y(w["c"])]
            probes = [(u, w["c"] + sg * (w["th"] / 2 + 0.2 / est)) for u in np.linspace(w["s"], w["e"], 7)[1:-1] for sg in (1, -1)]
        else:
            a, b = [X(w["c"]), Y(w["e"] + 0.5)], [X(w["c"]), Y(w["s"] - 0.5)]
            probes = [(w["c"] + sg * (w["th"] / 2 + 0.2 / est), v) for v in np.linspace(w["s"], w["e"], 7)[1:-1] for sg in (1, -1)]
        regs = [region(p) for p in probes]
        exterior = any(r_ == "outside" or (r_ and rmap[r_]["type"] == "balcony") for r_ in regs)
        plan_walls.append({"id": w["id"], "a": a, "b": b, "thickness": round(w["th"] * s, 3),
                           "height": hts["ceiling"], "exterior": exterior, "source": ex["kind"]})

    def wall_of(o):
        for w, pw in zip(walls, plan_walls):
            if w["axis"] == o["axis"] and abs(w["c"] - o["c"]) <= 2 and w["s"] <= (o["a"] + o["b"]) / 2 <= w["e"]:
                return pw["id"]
        return None

    def centre(o):
        m = (o["a"] + o["b"]) / 2
        return [X(m), Y(o["c"])] if o["axis"] == "h" else [X(o["c"]), Y(m)]

    plan_doors = []
    for k, d in enumerate(doors):
        o = d["op"]
        hinge = d["hinge"]
        if hinge is None:   # hinge on the end next to a perpendicular wall
            ends = {"a": o["a"] - 2, "b": o["b"] + 2}
            def near(e):
                off = d["side"] * (o["th"] / 2 + 0.15 / est)
                p = (ends[e], o["c"] + off) if o["axis"] == "h" else (o["c"] + off, ends[e])
                x, y = int(round(p[0])), int(round(p[1]))
                return 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1] and mask[y, x] > 0
            hinge = "a" if near("a") or not near("b") else "b"
        # 'a'/'b' in pixel order -> plan wall direction (vertical walls run bottom->top in metres)
        if o["axis"] == "v":
            hinge = "b" if hinge == "a" else "a"
        plan_doors.append({"id": f"d{k + 1}", "wall_id": wall_of(o), "center": centre(o),
                           "width": round((o["b"] - o["a"] + 1) * s, 3), "head": hts["door_head"],
                           "swing": {"hinge": hinge, "into": d["into"]}, "kind": d["kind"],
                           "rooms": d["rooms"]})
    plan_windows = []
    for k, wdw in enumerate(windows):
        o = wdw["op"]
        width = (o["b"] - o["a"] + 1) * s
        wet = any(rmap[r]["type"] in ("bathroom", "wc") for r in wdw["rooms"]) and width <= 0.9
        sill = 0.0 if wdw["kind"] == "balcony_door" else hts["wet_window_sill"] if wet else hts["window_sill"]
        head = hts["door_head"] if wdw["kind"] == "balcony_door" else hts["wet_window_head"] if wet else hts["window_head"]
        plan_windows.append({"id": f"win{k + 1}", "wall_id": wall_of(o), "center": centre(o), "width": round(width, 3),
                             "sill": sill, "head": head, "kind": wdw["kind"], "rooms": wdw["rooms"]})
    plan_rooms = []
    for r in rooms:
        pts = [(X(u), Y(v)) for u, v in r["poly"].exterior.coords]
        poly = Polygon(pts)
        if not poly.exterior.is_ccw:
            pts = pts[::-1]
        mr = poly.minimum_rotated_rectangle
        e = list(mr.exterior.coords)
        size = sorted([round(math.dist(e[0], e[1]), 3), round(math.dist(e[1], e[2]), 3)])
        bx = poly.bounds
        if abs(poly.area - (bx[2] - bx[0]) * (bx[3] - bx[1])) < 0.02 * poly.area:
            size = [round(bx[2] - bx[0], 3), round(bx[3] - bx[1], 3)]
        pr = {"id": r["id"], "type": r["type"], "label": r["label"], "polygon": [list(p) for p in pts[:-1]],
              "area_m2": round(poly.area, 2), "label_area_m2": r["label_area"], "size_m": size,
              "ceiling_height": hts["ceiling"], "confidence": r.get("confidence", 0.9 if r["labels"] else 0.3)}
        pr.update(r["extra"])
        if r.get("zones"):
            pr["zones"] = [{"type": z["type"], "pos": [X(z["x"]), Y(z["y"])]} for z in r["zones"]]
        plan_rooms.append(pr)
    warnings = list(dict.fromkeys(warnings))
    plan = {"schema_version": "1.0",
            "source": {"file": Path(ex.get("input", "")).name, "kind": ex["kind"], "page": ex.get("page")},
            "units": "m",
            "scale": {"m_per_px": s, "method": method, "confidence": round(conf, 2), "checks": checks},
            "defaults": {"ceiling_height": hts["ceiling"], "door_head": hts["door_head"],
                         "window_sill": hts["window_sill"], "window_head": hts["window_head"]},
            "walls": plan_walls, "doors": plan_doors, "windows": plan_windows, "rooms": plan_rooms,
            "texts": [{"text": t["text"], "pos": [X(t["x"]), Y(t["y"])], "kind": t["kind"],
                       **({"value": t["value"]} if "value" in t else {})} for t in texts],
            "warnings": warnings,
            "debug": {"background": "01_parse/page.png", "m_to_px": [1 / s, 0, -ox / s - 0.5, 0, -1 / s, oy / s - 0.5]}}
    out = job / "02_plan"
    out.mkdir(exist_ok=True)
    jsave(out / "plan.json", plan)
    overlay(job, plan)
    log(f"plan: {len(plan_walls)} walls, {len(plan_doors)} doors, {len(plan_windows)} windows, "
        f"{len(plan_rooms)} rooms, scale {method} ({s:.5f} m/px, conf {conf:.2f})")
    return plan


def overlay(job, plan):
    from PIL import Image, ImageDraw, ImageFont
    job = Path(job)
    bg = cv2.imread(str(job / plan["debug"]["background"]), 0)
    a, _, c, _, e, f = plan["debug"]["m_to_px"]
    k = min(1.0, 2400 / max(bg.shape))
    P = lambda x, y: ((a * x + c + 0.5) * k, (e * y + f + 0.5) * k)
    img = Image.fromarray(cv2.resize(bg, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)).convert("RGBA")
    lay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    dr = ImageDraw.Draw(lay)
    font = ImageFont.truetype(font_path(), max(12, int(img.size[0] / 90)))
    colors = {"living": (255, 190, 80), "bedroom": (120, 170, 255), "kitchen": (120, 220, 120), "bathroom": (80, 220, 220),
              "wc": (80, 200, 200), "hall": (200, 200, 200), "balcony": (220, 150, 255)}
    for r in plan["rooms"]:
        dr.polygon([P(*p) for p in r["polygon"]], fill=colors.get(r["type"], (255, 150, 150)) + (70,),
                   outline=colors.get(r["type"], (255, 150, 150)) + (255,))
    for w in plan["walls"]:
        (x0, y0), (x1, y1) = w["a"], w["b"]
        t = w["thickness"] / 2
        pts = [(x0, y0 - t), (x1, y1 - t), (x1, y1 + t), (x0, y0 + t)] if abs(y1 - y0) < 1e-6 else \
              [(x0 - t, y0), (x1 - t, y1), (x1 + t, y1), (x0 + t, y0)]
        dr.polygon([P(*p) for p in pts], fill=(230, 30, 30, 120))
    for op, col in [(d, (0, 170, 0, 255)) for d in plan["doors"]] + [(wd, (30, 60, 255, 255)) for wd in plan["windows"]]:
        w = next((x for x in plan["walls"] if x["id"] == op["wall_id"]), None)
        if not w:
            continue
        (x0, y0), (x1, y1) = w["a"], w["b"]
        L = math.dist(w["a"], w["b"]) or 1
        ux, uy = (x1 - x0) / L, (y1 - y0) / L
        cx, cy = op["center"]
        hw = op["width"] / 2
        dr.line([P(cx - ux * hw, cy - uy * hw), P(cx + ux * hw, cy + uy * hw)], fill=col, width=max(3, int(6 * k)))
    for r in plan["rooms"]:
        cx, cy = Polygon(r["polygon"]).representative_point().coords[0]
        txt = f"{r['label'] or r['type']}\n{r['size_m'][0]:.2f} x {r['size_m'][1]:.2f} m\n{r['area_m2']:.2f} m²"
        dr.multiline_text(P(cx, cy), txt, fill=(0, 0, 0, 255), font=font, anchor="mm", align="center")
    Image.alpha_composite(img, lay).convert("RGB").save(job / "02_plan/overlay.png")
__END_OF_PLAN_PY__
  cat > "$APP/layout.py" <<'__END_OF_LAYOUT_PY__'
"""Stage 3 - rule-based furniture layout -> 03_layout/layout.json + layout.png.
Items keep door swings and walkways clear, tall items stay away from windows,
nothing overlaps or leaves its room. Sizes shrink step by step when space is tight."""
import math
from pathlib import Path
import numpy as np
from shapely.geometry import Polygon, Point, LineString
from .common import jload, jsave, log, font_path

# footprint width (along wall) x depth x height in metres, biggest first
SIZES = {
    "sofa": [(2.10, 0.90, 0.85), (1.80, 0.88, 0.85), (1.50, 0.85, 0.85)],
    "armchair": [(0.80, 0.80, 0.80)], "coffee_table": [(1.10, 0.60, 0.42), (0.90, 0.50, 0.42)],
    "tv_unit": [(1.60, 0.40, 1.30), (1.20, 0.40, 1.20)], "rug": [(2.40, 1.70, 0.01), (2.00, 1.40, 0.01), (1.60, 1.10, 0.01)],
    "floor_lamp": [(0.35, 0.35, 1.60)], "plant": [(0.45, 0.45, 1.10)],
    "dining_table": [(1.60, 0.90, 0.75), (1.40, 0.85, 0.75), (1.20, 0.80, 0.75)], "chair": [(0.45, 0.50, 0.85)],
    "bed_double": [(1.60, 2.10, 1.05), (1.40, 2.10, 1.05)], "bed_single": [(0.90, 2.05, 1.00)],
    "nightstand": [(0.45, 0.40, 0.50)], "wardrobe": [(w, 0.60, 2.20) for w in (2.40, 2.00, 1.80, 1.60, 1.20, 1.00)],
    "desk": [(1.20, 0.60, 0.75), (1.00, 0.55, 0.75)], "fridge": [(0.60, 0.65, 1.90)],
    "toilet": [(0.40, 0.65, 0.80)], "vanity": [(0.80, 0.48, 0.85), (0.60, 0.45, 0.85)],
    "shower": [(0.90, 0.90, 2.00), (0.80, 0.80, 2.00)], "bathtub": [(1.70, 0.75, 0.58), (1.60, 0.70, 0.58)],
    "washer": [(0.60, 0.60, 0.85)], "basin_small": [(0.45, 0.35, 0.85)],
    "shoe_cabinet": [(1.00, 0.35, 1.00), (0.80, 0.35, 1.00), (0.60, 0.35, 1.00)],
    "bistro_table": [(0.60, 0.60, 0.72)], "bookshelf": [(0.80, 0.35, 1.80)],
}


def orect(c, u, n, w, d):
    c, u, n = np.asarray(c), np.asarray(u), np.asarray(n)
    return Polygon([tuple(c + su * u * w / 2 + sn * n * d / 2) for su, sn in ((-1, -1), (1, -1), (1, 1), (-1, 1))])


def unit(v):
    v = np.asarray(v, float)
    return v / (np.linalg.norm(v) or 1)


class Room:
    def __init__(self, r, plan, walls):
        self.r, self.id, self.type = r, r["id"], r["type"]
        self.poly = Polygon(r["polygon"])
        self.area = self.poly.area
        self.items, self.clears, self.missing = [], [], []
        pts = [np.asarray(p, float) for p in r["polygon"]]
        self.edges = []
        for i in range(len(pts)):
            p0, p1 = pts[i], pts[(i + 1) % len(pts)]
            L = float(np.linalg.norm(p1 - p0))
            if L >= 0.3:
                d = (p1 - p0) / L
                self.edges.append({"p0": p0, "p1": p1, "L": L, "dir": d, "n": np.array([-d[1], d[0]]),
                                   "door": False, "window": False})
        self.door_zones, self.win_zones, self.doors = [], [], []
        for o in plan["doors"] + plan["windows"]:
            if self.id not in o.get("rooms", []) or o["wall_id"] not in walls:
                continue
            w = walls[o["wall_id"]]
            u = unit(np.subtract(w["b"], w["a"]))
            nrm = np.array([-u[1], u[0]])
            c = np.asarray(o["center"], float)
            s = min((1, -1), key=lambda sg: self.poly.distance(Point(c + sg * nrm * (w["thickness"] / 2 + 0.05))))
            n_in = s * nrm
            face = c + n_in * w["thickness"] / 2
            is_door = o in plan["doors"] or o.get("kind") == "balcony_door"
            if is_door:
                depth = o["width"] + 0.05 if o.get("swing", {}).get("into") == self.id or o.get("kind") == "balcony_door" else 0.6
                if o.get("kind") == "opening":
                    depth = 0.6
                self.door_zones.append(orect(face + n_in * depth / 2, u, n_in, o["width"] + 0.2, depth))
                self.doors.append((face, o))
            else:
                self.win_zones.append((orect(face + n_in * 0.2, u, n_in, o["width"] + 0.1, 0.4), o["sill"], face))
            for e in self.edges:   # mark the room edge that holds this opening
                if LineString([e["p0"], e["p1"]]).distance(Point(face)) < 0.08:
                    e["door" if is_door else "window"] = True

    def ok(self, fp, h, clears=(), allow=()):
        if not self.poly.buffer(0.01).contains(fp):
            return False
        if any(fp.intersection(z).area > 1e-4 for z in self.door_zones):
            return False
        if any(h > sill + 0.02 and fp.intersection(z).area > 1e-4 for z, sill, _ in self.win_zones):
            return False
        for it in self.items:
            if it["type"] == "rug" or it["type"] in allow or it["id"] in allow:
                continue
            if fp.intersection(it["fp"].buffer(0.02)).area > 1e-4:
                return False
        for c, owner in self.clears:
            if owner not in allow and fp.intersection(c).area > 1e-3:
                return False
        for c in clears:
            if not self.poly.buffer(0.01).contains(c):
                return False
            if any(c.intersection(it["fp"]).area > 1e-3 for it in self.items if it["type"] not in ("rug",) and it["type"] not in allow):
                return False
        return True

    def add(self, typ, c, u, n, w, d, h, clears=(), params=None, allow=()):
        fp = orect(c, u, n, w, d)
        if not self.ok(fp, h, clears, allow):
            return None
        it = {"id": f"{self.id}_{typ}_{len(self.items)}", "type": typ, "c": np.asarray(c, float), "u": np.asarray(u, float),
              "n": np.asarray(n, float), "w": w, "d": d, "h": h, "fp": fp, "params": params or {}}
        self.items.append(it)
        self.clears += [(cl, it["id"]) for cl in clears]
        return it


def against_wall(R, typ, score=None, front=0.0, sides=0.0, allow=(), sizes=None, edges=None, params=None):
    for w, d, h in sizes or SIZES[typ]:
        best = None
        for e in edges or R.edges:
            if e["L"] < w + 0.02:
                continue
            for t in list(np.arange(0.01, e["L"] - w - 0.01 + 1e-9, 0.05)) + [(e["L"] - w) / 2]:
                c = e["p0"] + e["dir"] * (t + w / 2) + e["n"] * (d / 2 + 0.005)
                clears = []
                if front:
                    clears.append(orect(c + e["n"] * (d / 2 + front / 2), e["dir"], e["n"], w, front))
                if sides:
                    for sg in (1, -1):
                        clears.append(orect(c + sg * e["dir"] * (w / 2 + sides / 2) + e["n"] * 0.15, e["dir"], e["n"], sides, d - 0.3))
                fp = orect(c, e["dir"], e["n"], w, d)
                if not R.ok(fp, h, clears, allow):
                    continue
                sc = (score(e, t, w, c) if score else 0.0) + 0.3 * abs(t + w / 2 - e["L"] / 2) / e["L"]
                if best is None or sc < best[0]:
                    best = (sc, c, e, clears)
        if best:
            _, c, e, clears = best
            it = R.add(typ, c, e["dir"], e["n"], w, d, h, clears, params, allow)
            if it:
                it["edge"] = e
                return it
    R.missing.append(typ)
    return None


def free_spot(R, typ, sizes, ring=0.0, prefer=None):
    """Put a free-standing item (table) where it fits best, with a free ring for chairs."""
    minx, miny, maxx, maxy = R.poly.bounds
    for w, d, h in sizes:
        best = None
        for x in np.arange(minx + 0.3, maxx - 0.3, 0.1):
            for y in np.arange(miny + 0.3, maxy - 0.3, 0.1):
                for u in (np.array([1.0, 0]), np.array([0, 1.0])):
                    n = np.array([-u[1], u[0]])
                    c = np.array([x, y])
                    clears = [orect(c, u, n, w + 2 * ring, d + 2 * ring)] if ring else []
                    if not R.ok(orect(c, u, n, w, d), h, clears):
                        continue
                    sc = prefer(c) if prefer else 0
                    if best is None or sc < best[0]:
                        best = (sc, c, u, n, clears)
        if best:
            _, c, u, n, clears = best
            return R.add(typ, c, u, n, w, d, h, clears)
    R.missing.append(typ)
    return None


def chairs_around(R, table, per_side=2):
    for sg in (1, -1):
        for k in range(per_side):
            off = (k - (per_side - 1) / 2) * min(0.6, table["w"] / per_side)
            c = table["c"] + table["u"] * off + sg * table["n"] * (table["d"] / 2 + 0.15)
            n = -sg * table["n"]
            R.add("chair", c, np.array([n[1], -n[0]]), n, 0.45, 0.50, 0.85, allow=(table["id"], "chair"))


def dist_to_windows(R, c):
    return min([np.linalg.norm(np.asarray(f) - c) for _, _, f in R.win_zones] or [5.0])


def ray_len(poly, c, n):
    ln = LineString([tuple(c), tuple(c + n * 30)])
    x = ln.intersection(poly.exterior)
    pts = [x] if x.geom_type == "Point" else list(getattr(x, "geoms", []))
    ds = [np.linalg.norm(np.asarray(p.coords[0]) - c) for p in pts if p.geom_type == "Point"]
    ds = [d for d in ds if d > 0.05]
    return min(ds) if ds else 0.0


# ---------- room templates ----------
def kitchen_run(R, edges=None):
    lens = [round(x, 2) for x in np.arange(3.6, 1.19, -0.3)]
    for L in lens:
        it = against_wall(R, "kitchen_run", sizes=[(L, 0.60, 0.90)], front=0.9, edges=edges,
                          score=lambda e, t, w, c: 3 * e["door"] - 1.0 * e["window"] - 0.2 * e["L"])
        if it:
            R.missing = [m for m in R.missing if m != "kitchen_run"]
            break
    else:
        return None
    # sink under a window if there is one on this wall, hob 1 m away, wall cabinets away from windows
    x_win = []
    for z, sill, face in R.win_zones:
        loc = float(np.dot(np.asarray(face) - it["c"], it["u"]))
        if abs(float(np.dot(np.asarray(face) - it["c"], it["n"]))) < it["d"] / 2 + 0.3 and abs(loc) < it["w"] / 2:
            x_win.append(loc)
    sink = x_win[0] if x_win else -it["w"] / 2 + min(0.9, it["w"] / 3)
    sink = float(np.clip(sink, -it["w"] / 2 + 0.35, it["w"] / 2 - 0.35))
    hob = sink + 1.0 if sink + 1.0 < it["w"] / 2 - 0.35 else sink - 1.0
    upper, pos = [], -it["w"] / 2
    for xw in sorted(x_win) + [None]:
        end = (xw - 0.8) if xw is not None else it["w"] / 2
        if end - pos >= 0.4:
            upper.append([round(pos, 3), round(end, 3)])
        pos = (xw + 0.8) if xw is not None else pos
    it["params"] = {"sink_x": round(sink, 3), "hob_x": round(float(np.clip(hob, -it["w"] / 2 + 0.3, it["w"] / 2 - 0.3)), 3),
                    "upper": upper}
    return it


def furnish(R, plan):
    t = R.type
    if t == "living" or t == "dining":
        for z in R.r.get("zones", []):
            if z["type"] == "kitchen":   # open kitchen inside the living room
                zp = np.asarray(z["pos"])
                kitchen_run(R, edges=sorted(R.edges, key=lambda e: LineString([e["p0"], e["p1"]]).distance(Point(zp)))[:2])
        if t == "living":
            sofa = against_wall(R, "sofa", front=0.35, score=lambda e, t_, w, c: -min(e["L"], 4.5) + 3 * e["door"]
                                + (2 if ray_len(R.poly, c, e["n"]) < 2.4 else 0))
            if sofa:
                opp = [e for e in R.edges if np.dot(e["n"], sofa["n"]) < -0.95]
                tv = against_wall(R, "tv_unit", edges=opp, front=0.3,
                                  score=lambda e, t_, w, c: abs(np.dot(c - sofa["c"], sofa["u"]))) if opp else None
                ct_c = sofa["c"] + sofa["n"] * (sofa["d"] / 2 + 0.40 + 0.30)
                ct = None
                for w, d, h in SIZES["coffee_table"]:
                    ct = R.add("coffee_table", sofa["c"] + sofa["n"] * (sofa["d"] / 2 + 0.40 + d / 2), sofa["u"], sofa["n"], w, d, h,
                               allow=(sofa["id"],))
                    if ct:
                        break
                for w, d, h in SIZES["rug"]:
                    c = (ct["c"] if ct else ct_c)
                    fp = orect(c, sofa["u"], sofa["n"], w, d)
                    if R.poly.buffer(-0.05).contains(fp) and not any(fp.intersects(z) for z in R.door_zones):
                        it = {"id": f"{R.id}_rug", "type": "rug", "c": c, "u": sofa["u"], "n": sofa["n"], "w": w, "d": d, "h": 0.01,
                              "fp": fp, "params": {}}
                        R.items.append(it)
                        break
                for sg in (1, -1):   # armchair beside the coffee table, facing it
                    if not ct:
                        break
                    c = ct["c"] + sg * sofa["u"] * (ct["w"] / 2 + 0.35 + 0.40)
                    n = -sg * sofa["u"]
                    if R.add("armchair", c, np.array([n[1], -n[0]]), n, 0.80, 0.80, 0.80, clears=[orect(c + n * 0.55, np.array([n[1], -n[0]]), n, 0.8, 0.3)],
                             allow=(ct["id"],)):
                        break
                for sg in (-1, 1):   # floor lamp at a sofa end
                    c = sofa["c"] + sg * sofa["u"] * (sofa["w"] / 2 + 0.25) - sofa["n"] * (sofa["d"] / 2 - 0.2)
                    if R.add("floor_lamp", c, sofa["u"], sofa["n"], 0.35, 0.35, 1.60):
                        break
            if R.area >= 22 or t == "dining":
                sc = (lambda c: -np.linalg.norm(c - sofa["c"])) if t == "living" and sofa else None
                tab = free_spot(R, "dining_table", SIZES["dining_table"], ring=0.65, prefer=sc)
                if tab:
                    chairs_around(R, tab, 3 if tab["w"] >= 1.6 else 2)
        else:
            tab = free_spot(R, "dining_table", SIZES["dining_table"], ring=0.65)
            if tab:
                chairs_around(R, tab, 3 if tab["w"] >= 1.6 else 2)
    elif t == "bedroom":
        mnx, mny, mxx, mxy = R.poly.bounds
        double = not R.r.get("child") and (R.r.get("master") or (R.area >= 11 and min(mxx - mnx, mxy - mny) >= 2.9))
        bed = against_wall(R, "bed_double" if double else "bed_single", front=0.6, sides=0.45 if double else 0.0,
                           score=lambda e, t_, w, c: 3 * e["door"] + 1.5 * e["window"] - 0.1 * e["L"])
        if bed is None and double:
            R.missing.pop()
            bed = against_wall(R, "bed_single", front=0.6)
        if bed:
            for sg in ((1, -1) if bed["type"] == "bed_double" else (1, -1)):
                c = bed["c"] + sg * bed["u"] * (bed["w"] / 2 + 0.03 + 0.225) - bed["n"] * (bed["d"] / 2 - 0.20)
                ns = R.add("nightstand", c, bed["u"], bed["n"], 0.45, 0.40, 0.50, allow=(bed["id"],))
                if ns and bed["type"] == "bed_single":
                    break
        against_wall(R, "wardrobe", front=0.7, score=lambda e, t_, w, c: 2 * e["window"] - 0.1 * e["L"]
                     + (2 if bed and e is bed.get("edge") else 0))
        if not double or R.area >= 14:
            desk = against_wall(R, "desk", front=0.7, score=lambda e, t_, w, c: dist_to_windows(R, c))
            if desk:
                n = -desk["n"]
                R.add("chair", desk["c"] + desk["n"] * (desk["d"] / 2 + 0.12), np.array([n[1], -n[0]]), n, 0.45, 0.50, 0.85,
                      allow=(desk["id"],))
        plant_in_corner(R)
    elif t == "kitchen":
        run = kitchen_run(R)
        if run and R.area >= 8:   # L-shape on a neighbouring wall
            nb = [e for e in R.edges if abs(np.dot(e["dir"], run["u"])) < 0.1 and e is not run.get("edge")]
            if nb:
                for L in (2.4, 1.8, 1.2):
                    it = against_wall(R, "kitchen_run", sizes=[(L, 0.60, 0.90)], front=0.9, edges=nb,
                                      score=lambda e, t_, w, c: np.linalg.norm(c - run["c"]))
                    if it:
                        it["params"] = {"sink_x": None, "hob_x": None, "upper": [[-L / 2, L / 2]]}
                        break
                R.missing = [m for m in R.missing if m != "kitchen_run"]
        against_wall(R, "fridge", front=0.7, score=lambda e, t_, w, c: min([np.linalg.norm(c - i["c"]) for i in R.items] or [0]))
        if R.area >= 9:
            tab = free_spot(R, "dining_table", [(0.80, 0.80, 0.75)], ring=0.6)
            if tab:
                chairs_around(R, tab, 1)
    elif t in ("bathroom", "wc"):
        if t == "bathroom":
            mnx, mny, mxx, mxy = R.poly.bounds
            corner = lambda e, t_, w, c: min(t_, e["L"] - t_ - w)
            tub = against_wall(R, "bathtub", front=0.5, score=corner) if R.area >= 4.0 and min(mxx - mnx, mxy - mny) >= 1.5 else None
            if not tub:
                R.missing = [m for m in R.missing if m != "bathtub"]
                against_wall(R, "shower", front=0.6, score=corner)
        against_wall(R, "toilet", front=0.55, sides=0.20, score=lambda e, t_, w, c: 2 * e["door"])
        against_wall(R, "vanity" if t == "bathroom" else "basin_small", front=0.6)
        if t == "bathroom" and R.area >= 5:
            against_wall(R, "washer", front=0.6)
            R.missing = [m for m in R.missing if m != "washer"]
    elif t == "hall":
        ent = [f for f, o in R.doors if len(o.get("rooms", [])) == 1]
        against_wall(R, "shoe_cabinet", front=0.5,
                     score=lambda e, t_, w, c: min([np.linalg.norm(c - f) for f in ent] or [0]))
    elif t == "balcony":
        tab = free_spot(R, "bistro_table", SIZES["bistro_table"], ring=0.0)
        if tab:
            for sg in (1, -1):
                n = -sg * tab["u"]
                R.add("chair", tab["c"] + sg * tab["u"] * 0.55, np.array([n[1], -n[0]]), n, 0.45, 0.50, 0.85, allow=(tab["id"],))
    elif t == "study":
        desk = against_wall(R, "desk", front=0.7, score=lambda e, t_, w, c: dist_to_windows(R, c))
        if desk:
            n = -desk["n"]
            R.add("chair", desk["c"] + desk["n"] * (desk["d"] / 2 + 0.12), np.array([n[1], -n[0]]), n, 0.45, 0.50, 0.85, allow=(desk["id"],))
        against_wall(R, "bookshelf", front=0.5)
    if t == "living":
        plant_in_corner(R)


def plant_in_corner(R):
    pts = [np.asarray(p, float) for p in R.r["polygon"]]
    cands = []
    for i, p in enumerate(pts):
        a, b = unit(pts[i - 1] - p), unit(pts[(i + 1) % len(pts)] - p)
        bis = unit(a + b)
        c = p + bis * 0.36
        if R.poly.contains(Point(c)):
            cands.append((dist_to_windows(R, c), c))
    for _, c in sorted(cands, key=lambda x: x[0]):
        if R.add("plant", c, np.array([1.0, 0]), np.array([0, 1.0]), 0.45, 0.45, 1.10):
            return


def verify(rooms):
    ck = {"overlaps": 0, "outside_room": 0, "blocking_doors": 0, "tall_in_front_of_windows": 0}
    for R in rooms:
        its = [i for i in R.items if i["type"] != "rug"]
        for i, a in enumerate(its):
            if not R.poly.buffer(0.02).contains(a["fp"]):
                ck["outside_room"] += 1
            if any(a["fp"].intersection(z).area > 1e-3 for z in R.door_zones):
                ck["blocking_doors"] += 1
            if any(a["h"] > s + 0.02 and a["fp"].intersection(z).area > 1e-3 for z, s, _ in R.win_zones):
                ck["tall_in_front_of_windows"] += 1
            for b in its[i + 1:]:
                pair = {a["type"], b["type"]}
                if pair & {"chair"} and pair & {"dining_table", "desk", "bistro_table"}:
                    continue
                if a["fp"].intersection(b["fp"]).area > 1e-3:
                    ck["overlaps"] += 1
    return ck


def run_layout(job, cfg):
    job = Path(job)
    plan = jload(job / "02_plan/plan.json")
    walls = {w["id"]: w for w in plan["walls"]}
    rooms = [Room(r, plan, walls) for r in plan["rooms"]]
    for R in sorted(rooms, key=lambda R: -R.area):
        try:
            furnish(R, plan)
        except Exception as e:   # one bad room must not stop the run
            R.missing.append(f"error: {e}")
    items = []
    for R in rooms:
        for it in R.items:
            items.append({"id": it["id"], "room": R.id, "room_type": R.type, "type": it["type"],
                          "center": [round(float(it["c"][0]), 4), round(float(it["c"][1]), 4)],
                          "size": [it["w"], it["d"], it["h"]],
                          "rot_deg": round(math.degrees(math.atan2(it["u"][1], it["u"][0])), 3), "z": 0.0,
                          "params": it["params"]})
    checks = verify(rooms)
    empty = [R.id for R in rooms if R.type in ("living", "bedroom", "kitchen", "bathroom") and not R.items]
    warns = [f"{R.id} ({R.type}): could not place {', '.join(R.missing)}" for R in rooms if R.missing]
    warns += [f"main room {r} has no furniture" for r in empty]
    out = {"items": items, "checks": checks, "unfurnished_main_rooms": empty, "warnings": warns,
           "rooms": {R.id: {"type": R.type, "placed": [i["type"] for i in R.items], "missing": R.missing} for R in rooms}}
    (job / "03_layout").mkdir(exist_ok=True)
    jsave(job / "03_layout/layout.json", out)
    draw(job, plan, rooms)
    log(f"layout: {len(items)} items, checks {checks}, warnings {len(warns)}")
    return out


def draw(job, plan, rooms):
    from PIL import Image, ImageDraw, ImageFont
    xs = [p[0] for r in plan["rooms"] for p in r["polygon"]] + [c for w in plan["walls"] for c in (w["a"][0], w["b"][0])]
    ys = [p[1] for r in plan["rooms"] for p in r["polygon"]] + [c for w in plan["walls"] for c in (w["a"][1], w["b"][1])]
    k, pad = 80, 40
    W, H = int((max(xs) - min(xs)) * k) + 2 * pad, int((max(ys) - min(ys)) * k) + 2 * pad
    P = lambda p: (pad + (p[0] - min(xs)) * k, H - pad - (p[1] - min(ys)) * k)
    im = Image.new("RGB", (W, H), "white")
    dr = ImageDraw.Draw(im)
    font = ImageFont.truetype(font_path(), 11)
    for w in plan["walls"]:
        dr.line([P(w["a"]), P(w["b"])], fill=(120, 0, 0), width=max(2, int(w["thickness"] * k)))
    for R in rooms:
        dr.polygon([P(p) for p in R.r["polygon"]], outline=(90, 90, 90))
        for z in R.door_zones:
            dr.polygon([P(p) for p in z.exterior.coords], outline=(0, 160, 0))
        for z, _, _ in R.win_zones:
            dr.polygon([P(p) for p in z.exterior.coords], outline=(40, 80, 255))
        for it in R.items:
            col = (230, 225, 210) if it["type"] == "rug" else (170, 150, 120)
            dr.polygon([P(p) for p in it["fp"].exterior.coords], fill=col, outline=(60, 50, 40))
            front = it["c"] + it["n"] * it["d"] / 2
            dr.line([P(it["c"]), P(front)], fill=(200, 0, 0), width=1)
            dr.text(P(it["c"]), it["type"].replace("_", " "), fill=(0, 0, 0), font=font, anchor="mm")
    im.save(job / "03_layout/layout.png")
__END_OF_LAYOUT_PY__
  cat > "$APP/blender_scene.py" <<'__END_OF_BLENDER_SCENE_PY__'
"""Stage 4 - build the 3D apartment in Blender (background mode).
Run: blender -b --factory-startup --python-exit-code 1 -P blender_scene.py -- <job_dir>
Writes 04_scene/scene.blend, scene.glb, cameras.json, furniture_qa.json. Uses only bpy/bmesh/mathutils.
Walls are watertight solids with real openings. Furniture: the catalog model chosen in 03_layout/assets.json
(normalised: real size, Z up, front to +Y, origin at base centre, polygon cap, slot fit with limited non-uniform
scale) or the parametric model; one named object per item. Collections: Walls, Floors_Ceilings, Openings,
Furniture.<room>, Lights, Cameras."""
import bpy, bmesh, json, math, os, re, statistics, sys
from pathlib import Path
from mathutils import Vector, Matrix

ARGS = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
JOB = Path(ARGS[0])
RC = json.loads((JOB / "run_config.json").read_text())
PLAN = json.loads((JOB / "02_plan/plan.json").read_text(encoding="utf-8"))
LAYOUT = json.loads((JOB / "03_layout/layout.json").read_text(encoding="utf-8"))
_AF = JOB / "03_layout/assets.json"
CHOICE = json.loads(_AF.read_text(encoding="utf-8")) if _AF.exists() else None   # None = old behaviour
ASSETS = Path(RC["assets"])
OUT = JOB / "04_scene"
OUT.mkdir(parents=True, exist_ok=True)
WALLS = {w["id"]: w for w in PLAN["walls"]}
ROOMS = PLAN["rooms"]
H = PLAN["defaults"]["ceiling_height"]
WARN = []

bpy.ops.wm.read_factory_settings(use_empty=True)
SC = bpy.context.scene
SC.render.engine = "CYCLES"
COLL = SC.collection


# ---------- small 2D helpers ----------
def pip(pt, poly):
    x, y = pt
    inside = False
    for i in range(len(poly)):
        (x1, y1), (x2, y2) = poly[i], poly[(i + 1) % len(poly)]
        if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1) + x1:
            inside = not inside
    return inside


def room_at(pt):
    return next((r for r in ROOMS if pip(pt, r["polygon"])), None)


def unit2(v):
    L = math.hypot(v[0], v[1]) or 1.0
    return (v[0] / L, v[1] / L)


def centroid(poly):
    a = cx = cy = 0.0
    for i in range(len(poly)):
        (x1, y1), (x2, y2) = poly[i], poly[(i + 1) % len(poly)]
        f = x1 * y2 - x2 * y1
        a += f
        cx += (x1 + x2) * f
        cy += (y1 + y2) * f
    a *= 0.5
    return (cx / (6 * a), cy / (6 * a)) if abs(a) > 1e-9 else poly[0]


# ---------- materials (ambientCG CC0 textures when downloaded, else simple colours) ----------
SPEC = {
    "wall_paint": dict(c=(0.84, 0.80, 0.74), r=0.9, tex=("PaintedPlaster017", 2.5)),
    "ceiling": dict(c=(0.88, 0.87, 0.85), r=0.9), "exterior": dict(c=(0.72, 0.69, 0.63), r=0.9),
    "wood_floor": dict(c=(0.55, 0.40, 0.27), r=0.45, tex=("WoodFloor051", 1.6)),
    "tile_floor": dict(c=(0.76, 0.74, 0.70), r=0.25, tiles=(0.6, 0.6, (0.72, 0.70, 0.66), (0.55, 0.53, 0.50))),
    "tile_wall": dict(c=(0.90, 0.89, 0.87), r=0.12, tiles=(0.3, 0.6, (0.87, 0.86, 0.84), (0.70, 0.70, 0.68))),
    "fabric_sofa": dict(c=(0.60, 0.56, 0.50), r=0.95, tex=("Fabric061", 0.6)),
    "fabric_bed": dict(c=(0.93, 0.92, 0.89), r=0.9), "fabric_accent": dict(c=(0.62, 0.55, 0.46), r=0.92),
    "wood": dict(c=(0.66, 0.50, 0.34), r=0.5, tex=("Wood049", 1.2)), "white_matte": dict(c=(0.88, 0.87, 0.85), r=0.45),
    "black_metal": dict(c=(0.03, 0.03, 0.03), r=0.35, m=1.0), "steel": dict(c=(0.75, 0.75, 0.74), r=0.25, m=1.0),
    "ceramic": dict(c=(0.95, 0.95, 0.94), r=0.06), "stone_counter": dict(c=(0.86, 0.85, 0.83), r=0.2, tex=("Marble012", 1.5)),
    "screen": dict(c=(0.01, 0.01, 0.012), r=0.05), "rug": dict(c=(0.74, 0.69, 0.61), r=1.0, tex=("Carpet016", 1.2)),
    "door": dict(c=(0.90, 0.89, 0.86), r=0.45), "frame": dict(c=(0.94, 0.94, 0.93), r=0.35),
    "sill": dict(c=(0.90, 0.89, 0.87), r=0.2), "mirror": dict(c=(0.95, 0.95, 0.95), r=0.02, m=1.0),
    "glass": dict(c=(1, 1, 1), r=0.0), "pot": dict(c=(0.25, 0.22, 0.2), r=0.6),
}
_M = {}


def find_map(folder, key):
    for f in sorted(folder.glob("*")):
        if key.lower() in f.name.lower() and f.suffix.lower() in (".jpg", ".png"):
            return f
    return None


def mat(name):
    if name in _M:
        return _M[name]
    s = SPEC[name]
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt, N = m.node_tree, m.node_tree.nodes
    b = N.get("Principled BSDF")
    b.inputs["Base Color"].default_value = (*s["c"], 1)
    b.inputs["Roughness"].default_value = s["r"]
    b.inputs["Metallic"].default_value = s.get("m", 0.0)
    m.diffuse_color = (*s["c"], 1)
    uvmap = None
    if "tex" in s or "tiles" in s:
        tc, mp = N.new("ShaderNodeTexCoord"), N.new("ShaderNodeMapping")
        nt.links.new(tc.outputs["UV"], mp.inputs["Vector"])
        uvmap = mp
    if name == "glass":   # clear glass that lets sun and sky light through (shadow rays skip it)
        b.inputs["Transmission Weight"].default_value = 1.0
        b.inputs["IOR"].default_value = 1.45
        out = N.get("Material Output")
        lp, tr, mix = N.new("ShaderNodeLightPath"), N.new("ShaderNodeBsdfTransparent"), N.new("ShaderNodeMixShader")
        nt.links.new(lp.outputs["Is Shadow Ray"], mix.inputs[0])
        nt.links.new(b.outputs[0], mix.inputs[1])
        nt.links.new(tr.outputs[0], mix.inputs[2])
        nt.links.new(mix.outputs[0], out.inputs["Surface"])
    if "tiles" in s:
        tw, th, c2, mortar = s["tiles"]
        br = N.new("ShaderNodeTexBrick")
        br.offset = 0.0
        br.inputs["Scale"].default_value = 1.0
        br.inputs["Brick Width"].default_value = tw
        br.inputs["Row Height"].default_value = th
        br.inputs["Mortar Size"].default_value = 0.003
        br.inputs["Color1"].default_value = (*s["c"], 1)
        br.inputs["Color2"].default_value = (*c2, 1)
        br.inputs["Mortar"].default_value = (*mortar, 1)
        nt.links.new(uvmap.outputs[0], br.inputs["Vector"])
        nt.links.new(br.outputs["Color"], b.inputs["Base Color"])
        bump = N.new("ShaderNodeBump")
        bump.inputs["Strength"].default_value = 0.3
        bump.invert = True
        nt.links.new(br.outputs["Fac"], bump.inputs["Height"])
        nt.links.new(bump.outputs["Normal"], b.inputs["Normal"])
    if "tex" in s:
        tid, size = s["tex"]
        folder = ASSETS / "materials" / tid
        col = find_map(folder, "_Color") if folder.exists() else None
        if col:
            uvmap.inputs["Scale"].default_value = (1 / size, 1 / size, 1 / size)
            for key, inp, colour in (("_Color", "Base Color", True), ("_Roughness", "Roughness", False), ("_NormalGL", None, False)):
                f = find_map(folder, key)
                if not f:
                    continue
                im = N.new("ShaderNodeTexImage")
                im.image = bpy.data.images.load(str(f), check_existing=True)
                if not colour:
                    im.image.colorspace_settings.name = "Non-Color"
                nt.links.new(uvmap.outputs[0], im.inputs["Vector"])
                if inp:
                    nt.links.new(im.outputs["Color"], b.inputs[inp])
                else:
                    nm = N.new("ShaderNodeNormalMap")
                    nm.inputs["Strength"].default_value = 0.6
                    nt.links.new(im.outputs["Color"], nm.inputs["Color"])
                    nt.links.new(nm.outputs["Normal"], b.inputs["Normal"])
    _M[name] = m
    return m


# ---------- mesh helpers ----------
def uv_box(me, offset=Vector((0, 0, 0))):
    uv = me.uv_layers.new(name="UVMap")
    for p in me.polygons:
        n = p.normal
        ax = max(range(3), key=lambda i: abs(n[i]))
        for li in p.loop_indices:
            co = me.vertices[me.loops[li].vertex_index].co + offset
            uv.data[li].uv = (co.y, co.z) if ax == 0 else (co.x, co.z) if ax == 1 else (co.x, co.y)


def new_obj(name, me, mats, parent=None, loc=(0, 0, 0), rot=(0, 0, 0)):
    for m in mats:
        me.materials.append(m)
    ob = bpy.data.objects.new(name, me)
    COLL.objects.link(ob)
    ob.location, ob.rotation_euler = loc, rot
    if parent:
        ob.parent = parent
    return ob


def box(name, size, center, m, parent=None, bevel=0.0, rot_z=0.0):
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=1.0, matrix=Matrix.Diagonal((*size, 1)))
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    uv_box(me)
    ob = new_obj(name, me, [mat(m)], parent, center, (0, 0, rot_z))
    if bevel > 0:
        mod = ob.modifiers.new("bevel", "BEVEL")
        mod.width, mod.segments, mod.limit_method = bevel, 2, "ANGLE"
    return ob


def cyl(name, r, h, center, m, parent=None, axis="z"):
    bm = bmesh.new()
    bmesh.ops.create_cone(bm, cap_ends=True, cap_tris=False, segments=28, radius1=r, radius2=r, depth=h)
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    for p in me.polygons:
        p.use_smooth = len(p.vertices) == 4
    uv_box(me)
    rot = (math.pi / 2, 0, 0) if axis == "y" else (0, 0, 0)
    return new_obj(name, me, [mat(m)], parent, center, rot)


def empty(name, loc, rot_z=0.0):
    e = bpy.data.objects.new(name, None)
    COLL.objects.link(e)
    e.location, e.rotation_euler = loc, (0, 0, rot_z)
    return e


def wall_frame(w):
    (ax, ay), (bx, by) = w["a"], w["b"]
    L = math.hypot(bx - ax, by - ay)
    u = ((bx - ax) / L, (by - ay) / L)
    return (ax, ay), u, (-u[1], u[0]), L


def wall_corners(w):
    (ax, ay), u, n, L = wall_frame(w)
    t = w["thickness"] / 2
    return [(ax + u[0] * s + n[0] * k, ay + u[1] * s + n[1] * k) for s in (0.0, L) for k in (-t, t)]


def side_into(c, nrm, t, room_id):
    """+1 or -1: which side of the wall (along its normal) the given room is."""
    for s in (1, -1):
        r = room_at((c[0] + s * nrm[0] * (t / 2 + 0.15), c[1] + s * nrm[1] * (t / 2 + 0.15)))
        if r and r["id"] == room_id:
            return s
    return 1


# ---------- shell: walls with openings, floors, ceilings, slabs ----------
WET = ("bathroom", "wc")


def build_walls():
    openings = {}
    for o in PLAN["doors"]:
        openings.setdefault(o["wall_id"], []).append((o, 0.0, o["head"]))
    for o in PLAN["windows"]:
        openings.setdefault(o["wall_id"], []).append((o, o["sill"], o["head"]))
    mats = [mat("wall_paint"), mat("tile_wall"), mat("exterior")]
    for w in PLAN["walls"]:
        a, u, n, L = wall_frame(w)
        t, hgt = w["thickness"], w["height"]
        holes = []
        for o, z0, z1 in openings.get(w["id"], []):
            s_ = (o["center"][0] - a[0]) * u[0] + (o["center"][1] - a[1]) * u[1]
            lo, hi = max(0.0, s_ - o["width"] / 2), min(L, s_ + o["width"] / 2)
            if hi - lo > 0.01:
                holes.append((lo, hi, max(0.0, z0), min(hgt, z1)))
        bm = bmesh.new()
        wall_solid(bm, a, u, n, L, t, hgt, holes)
        me = bpy.data.meshes.new(w["id"])
        bm.to_mesh(me)
        bm.free()
        for p in me.polygons:   # paint, tiles or facade depending on what the face looks at
            c, nn = p.center, p.normal
            if abs(nn.z) > 0.5 or abs(nn.x * u[0] + nn.y * u[1]) > 0.5:   # tops, reveals and wall ends: paint
                continue
            r = room_at((c.x + nn.x * 0.06, c.y + nn.y * 0.06))
            p.material_index = 2 if (r is None or r["type"] == "balcony") else 1 if r["type"] in WET else 0
        uv_box(me)
        new_obj(w["id"], me, mats)


def _dedup(vals, eps=1e-3):
    out = []
    for v in sorted(vals):
        if not out or v - out[-1] > eps:
            out.append(v)
    return out


def wall_solid(bm, a, u, n, L, t, hgt, holes):
    """One watertight wall with real openings: the front face is a grid split at every opening edge, cells inside
    an opening are left out, the back is a mirrored copy and every boundary edge of the grid gets a side face
    (wall ends, top, bottom and the reveals of each opening). Every edge ends up with exactly two faces."""
    xs = _dedup([0.0, L] + [v for lo, hi, _, _ in holes for v in (lo, hi) if 0.0 < v < L])
    zs = _dedup([0.0, hgt] + [v for _, _, z0, z1 in holes for v in (z0, z1) if 0.0 < v < hgt])
    grid = {}

    def V(i, j):
        if (i, j) not in grid:
            s, z = xs[i], zs[j]
            grid[(i, j)] = bm.verts.new((a[0] + u[0] * s - n[0] * t / 2, a[1] + u[1] * s - n[1] * t / 2, z))
        return grid[(i, j)]

    front = []
    for i in range(len(xs) - 1):
        for j in range(len(zs) - 1):
            xm, zm = (xs[i] + xs[i + 1]) / 2, (zs[j] + zs[j + 1]) / 2
            if any(lo < xm < hi and z0 < zm < z1 for lo, hi, z0, z1 in holes):
                continue
            front.append(bm.faces.new((V(i, j), V(i + 1, j), V(i + 1, j + 1), V(i, j + 1))))
    if not front:
        return
    bnd = [tuple(e.verts) for e in {e for f in front for e in f.edges} if len(e.link_faces) == 1]
    back = {v: bm.verts.new(v.co + Vector((n[0] * t, n[1] * t, 0.0))) for v in grid.values()}
    for f in front:
        bm.faces.new([back[v] for v in reversed(f.verts)])
    for v1, v2 in bnd:
        bm.faces.new((v2, v1, back[v1], back[v2]))
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])


def poly_mesh(name, pts, z, m, down=False):
    verts = [(x, y, z) for x, y in pts]
    face = list(range(len(pts)))
    me = bpy.data.meshes.new(name)
    me.from_pydata(verts, [], [face[::-1] if down else face])
    me.update()
    uv_box(me)
    return new_obj(name, me, [mat(m)])


def build_floors():
    for r in ROOMS:
        fl = "tile_floor" if r["type"] in ("kitchen", "bathroom", "wc", "balcony", "storage") else "wood_floor"
        poly_mesh(f"floor_{r['id']}", r["polygon"], 0.0, fl)
        if r["type"] != "balcony":
            poly_mesh(f"ceiling_{r['id']}", r["polygon"], r["ceiling_height"], "ceiling", down=True)
    for d in PLAN["doors"] + [w for w in PLAN["windows"] if w["kind"] == "balcony_door"]:
        w = WALLS.get(d["wall_id"])
        if not w:
            continue
        _, u, n, _ = wall_frame(w)
        c, hw, ht = d["center"], d["width"] / 2, w["thickness"] / 2
        pts = [(c[0] + su * u[0] * hw + sn * n[0] * ht, c[1] + su * u[1] * hw + sn * n[1] * ht)
               for su, sn in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
        if (pts[1][0] - pts[0][0]) * (pts[2][1] - pts[0][1]) - (pts[1][1] - pts[0][1]) * (pts[2][0] - pts[0][0]) < 0:
            pts = pts[::-1]
        poly_mesh(f"threshold_{d['id']}", pts, 0.0, "sill")
    corners = [c for w in PLAN["walls"] for c in wall_corners(w)]   # slabs flush with the outer wall faces
    xs, ys = [c[0] for c in corners], [c[1] for c in corners]
    cx, cy, sx, sy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2, max(xs) - min(xs), max(ys) - min(ys)
    box("slab_bottom", (sx, sy, 0.3), (cx, cy, -0.155), "exterior")
    box("slab_top", (sx, sy, 0.3), (cx, cy, H + 0.155), "exterior")


def build_balcony_rails():
    for r in ROOMS:
        if r["type"] != "balcony":
            continue
        p = r["polygon"]
        for i in range(len(p)):
            (x1, y1), (x2, y2) = p[i], p[(i + 1) % len(p)]
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            L = math.hypot(x2 - x1, y2 - y1)
            near_wall = any(abs((mx - w["a"][0]) * -wall_frame(w)[1][1] + (my - w["a"][1]) * wall_frame(w)[1][0]) < w["thickness"] / 2 + 0.2
                            and -0.1 <= (mx - w["a"][0]) * wall_frame(w)[1][0] + (my - w["a"][1]) * wall_frame(w)[1][1] <= wall_frame(w)[3] + 0.1
                            for w in PLAN["walls"])
            if near_wall or L < 0.2:
                continue
            ang = math.atan2(y2 - y1, x2 - x1)
            box(f"rail_glass_{r['id']}_{i}", (L, 0.012, 0.95), (mx, my, 0.55), "glass", rot_z=ang)
            box(f"rail_top_{r['id']}_{i}", (L, 0.05, 0.04), (mx, my, 1.05), "black_metal", rot_z=ang)


# ---------- windows, doors, skirting ----------
def build_windows():
    for o in PLAN["windows"]:
        w = WALLS.get(o["wall_id"])
        if not w:
            continue
        a, u, n, _ = wall_frame(w)
        c, W, z0, z1, t = o["center"], o["width"], o["sill"], o["head"], w["thickness"]
        root = empty(f"win_{o['id']}", (c[0], c[1], 0), math.atan2(u[1], u[0]))
        hz = z1 - z0
        box("frame_l", (0.06, 0.07, hz), (-W / 2 + 0.03, 0, z0 + hz / 2), "frame", root)
        box("frame_r", (0.06, 0.07, hz), (W / 2 - 0.03, 0, z0 + hz / 2), "frame", root)
        box("frame_t", (W, 0.07, 0.06), (0, 0, z1 - 0.03), "frame", root)
        box("frame_b", (W, 0.07, 0.06), (0, 0, z0 + 0.03), "frame", root)
        if W > 1.1:
            box("mullion", (0.06, 0.07, hz), (0, 0, z0 + hz / 2), "frame", root)
        box("glass", (W - 0.1, 0.01, hz - 0.1), (0, 0, z0 + hz / 2), "glass", root)
        s = side_into(c, n, t, o["rooms"][0]) if o.get("rooms") else 1
        if z0 > 0.1:
            box("sill", (W + 0.08, t / 2 + 0.05, 0.03), (0, s * (t / 4 + 0.025), z0 - 0.015), "sill", root)
        ld = bpy.data.lights.new(f"portal_{o['id']}", "AREA")
        ld.shape, ld.size, ld.size_y = "RECTANGLE", W, hz
        ld.cycles.is_portal = True
        lo = bpy.data.objects.new(f"portal_{o['id']}", ld)
        COLL.objects.link(lo)
        lo.location = (c[0] - s * n[0] * t / 2, c[1] - s * n[1] * t / 2, z0 + hz / 2)
        lo.rotation_euler = Vector((s * n[0], s * n[1], 0)).to_track_quat("-Z", "Y").to_euler()


def build_doors():
    for d in PLAN["doors"]:
        w = WALLS.get(d["wall_id"])
        if not w or d["kind"] == "opening":
            continue
        a, u, n, _ = wall_frame(w)
        c, W, hd, t = d["center"], d["width"], d["head"], w["thickness"]
        root = empty(f"door_{d['id']}", (c[0], c[1], 0), math.atan2(u[1], u[0]))
        box("jamb_l", (0.04, t + 0.02, hd), (-W / 2 + 0.02, 0, hd / 2), "frame", root)
        box("jamb_r", (0.04, t + 0.02, hd), (W / 2 - 0.02, 0, hd / 2), "frame", root)
        box("head", (W, t + 0.02, 0.04), (0, 0, hd - 0.02), "frame", root)
        lw = W - 0.08
        into = d["swing"]["into"]
        s = side_into(c, n, t, into)
        hx = -W / 2 + 0.04 if d["swing"]["hinge"] == "a" else W / 2 - 0.04
        v = 1 if d["swing"]["hinge"] == "a" else -1
        entrance = len(d.get("rooms", [])) < 2
        ang = 0.0 if entrance else math.radians(82)
        pivot = empty(f"leaf_{d['id']}", (hx, s * (t / 2 - 0.03) if not entrance else 0, 0))
        pivot.parent = root
        pivot.rotation_euler.z = (math.pi if v < 0 else 0) + (v * s * ang)
        box("leaf", (lw, 0.04, hd - 0.01), (lw / 2, 0, (hd - 0.01) / 2), "door", pivot, bevel=0.003)
        for sy in (1, -1):
            box(f"handle{sy}", (0.12, 0.02, 0.02), (lw - 0.08, sy * 0.035, 1.0), "black_metal", pivot)


def build_skirting():
    for r in ROOMS:
        if r["type"] in ("balcony",) or r["type"] in WET:
            continue
        p = r["polygon"]
        gaps = []
        for o in PLAN["doors"] + [x for x in PLAN["windows"] if x["kind"] == "balcony_door"]:
            if r["id"] in o.get("rooms", []):
                gaps.append(o)
        for i in range(len(p)):
            (x1, y1), (x2, y2) = p[i], p[(i + 1) % len(p)]
            L = math.hypot(x2 - x1, y2 - y1)
            if L < 0.1:
                continue
            ux, uy = (x2 - x1) / L, (y2 - y1) / L
            cut = []
            for o in gaps:
                along = (o["center"][0] - x1) * ux + (o["center"][1] - y1) * uy
                off = abs((o["center"][0] - x1) * -uy + (o["center"][1] - y1) * ux)
                if off < 0.25 and -0.5 < along < L + 0.5:
                    cut.append((along - o["width"] / 2, along + o["width"] / 2))
            pos = 0.0
            for lo, hi in sorted(cut) + [(L, L)]:
                if lo - pos > 0.05:
                    m = (pos + min(lo, L)) / 2
                    box(f"skirt_{r['id']}_{i}", (min(lo, L) - pos, 0.012, 0.08),
                        (x1 + ux * m - uy * 0.006, y1 + uy * m + ux * 0.006, 0.04), "white_matte", rot_z=math.atan2(uy, ux))
                pos = max(pos, hi)


# ---------- furniture (parametric, exact sizes; library models only for plants/lamps) ----------
def B(size, c, m, bev=0.0):
    return ("box", size, c, m, bev)


def C(r, h, c, m, axis="z"):
    return ("cyl", r, h, c, m, axis)


def legs(w, d, h, r=0.02, m="black_metal", inset=0.06):
    return [C(r, h, (sx * (w / 2 - inset), sy * (d / 2 - inset), h / 2), m) for sx in (1, -1) for sy in (1, -1)]


def parts(t, w, d, h, prm):
    P = []
    if t in ("sofa", "armchair"):
        n = 1 if t == "armchair" else (3 if w >= 1.8 else 2)
        P += legs(w, d, 0.08)
        P.append(B((w, d - 0.04, 0.26), (0, 0, 0.21), "fabric_sofa", 0.02))
        seat_w = (w - 0.32) / n
        for k in range(n):
            x = -w / 2 + 0.16 + seat_w * (k + 0.5)
            P.append(B((seat_w - 0.01, d - 0.28, 0.14), (x, 0.1, 0.41), "fabric_sofa", 0.035))
            P.append(B((seat_w - 0.01, 0.2, 0.38), (x, -d / 2 + 0.24, 0.58), "fabric_sofa", 0.05))
        P.append(B((w, 0.16, h - 0.08), (0, -d / 2 + 0.08, 0.08 + (h - 0.08) / 2), "fabric_sofa", 0.03))
        for sx in (1, -1):
            P.append(B((0.16, d - 0.04, 0.62), (sx * (w / 2 - 0.08), 0, 0.39), "fabric_sofa", 0.03))
    elif t == "coffee_table":
        P += [B((w, d, 0.04), (0, 0, h - 0.02), "wood", 0.004)] + legs(w, d, h - 0.04, 0.018)
    elif t == "tv_unit":
        P.append(B((w, d, 0.45), (0, 0, 0.26), "wood", 0.004))
        tw = min(w * 0.8, 1.45)
        P.append(B((0.25, 0.18, 0.04), (0, 0, 0.505), "black_metal"))
        P.append(B((tw, 0.04, tw * 9 / 16), (0, -0.02, 0.55 + tw * 9 / 32), "screen", 0.004))
    elif t == "rug":
        P.append(B((w, d, 0.012), (0, 0, 0.006), "rug"))
    elif t == "floor_lamp":
        P += [C(0.15, 0.02, (0, 0, 0.01), "black_metal"), C(0.012, 1.40, (0, 0, 0.72), "black_metal"),
              C(0.20, 0.28, (0, 0, h - 0.14), "fabric_bed")]
    elif t == "plant":
        P += [C(0.18, 0.36, (0, 0, 0.18), "pot")]
    elif t in ("dining_table", "bistro_table", "desk"):
        if t == "bistro_table":
            P += [C(w / 2, 0.03, (0, 0, h - 0.015), "wood"), C(0.03, h - 0.03, (0, 0, (h - 0.03) / 2), "black_metal"),
                  C(0.22, 0.02, (0, 0, 0.01), "black_metal")]
        else:
            P += [B((w, d, 0.035), (0, 0, h - 0.0175), "wood", 0.003)] + legs(w, d, h - 0.035, 0.022, "black_metal", 0.05)
    elif t == "chair":
        P += [B((w - 0.02, d - 0.08, 0.04), (0, 0.02, 0.45), "wood", 0.004), B((w - 0.02, 0.03, 0.38), (0, -d / 2 + 0.03, 0.66), "wood", 0.004)]
        P += legs(w - 0.02, d - 0.08, 0.43, 0.012)
    elif t in ("bed_double", "bed_single"):
        P.append(B((w + 0.06, 0.08, h), (0, -d / 2 + 0.04, h / 2), "fabric_accent", 0.02))
        P.append(B((w + 0.04, d - 0.08, 0.30), (0, 0.04, 0.20), "fabric_accent", 0.015))
        P.append(B((w, d - 0.12, 0.22), (0, 0.04, 0.46), "fabric_bed", 0.04))
        P.append(B((w + 0.04, (d - 0.1) * 0.62, 0.06), (0, d / 2 - (d - 0.1) * 0.31 - 0.01, 0.59), "fabric_accent", 0.03))
        k = 2 if t == "bed_double" else 1
        for i in range(k):
            x = 0 if k == 1 else (i - 0.5) * w / 2
            P.append(B((w / k - 0.14, 0.38, 0.14), (x, -d / 2 + 0.33, 0.64), "fabric_bed", 0.05))
    elif t == "nightstand":
        P += [B((w, d, 0.48), (0, 0, 0.26), "wood", 0.004), C(0.06, 0.05, (0, 0, 0.525), "ceramic"),
              C(0.11, 0.18, (0, 0, 0.66), "fabric_bed")]
    elif t in ("wardrobe", "shoe_cabinet", "bookshelf"):
        P.append(B((w, d - 0.02, h), (0, -0.01, h / 2), "wood" if t != "wardrobe" else "white_matte", 0.003))
        if t == "bookshelf":
            P += [B((w - 0.04, d - 0.04, 0.02), (0, 0.02, z), "white_matte") for z in (0.4, 0.8, 1.2, 1.6)]
        else:
            k = max(2, round(w / 0.5))
            for i in range(k):
                x = -w / 2 + (i + 0.5) * w / k
                P.append(B((w / k - 0.004, 0.02, h - 0.03), (x, d / 2 - 0.01, h / 2), "white_matte" if t == "wardrobe" else "wood", 0.002))
                P.append(B((0.012, 0.02, 0.3 if t == "wardrobe" else 0.12), (x + (0.5 if i % 2 == 0 else -0.5) * (w / k - 0.08), d / 2 + 0.01, 1.05 if t == "wardrobe" else h - 0.12), "black_metal"))
    elif t == "kitchen_run":
        P.append(B((w, d - 0.08, 0.10), (0, -0.04, 0.05), "black_metal"))
        k = max(1, round(w / 0.6))
        for i in range(k):
            x = -w / 2 + (i + 0.5) * w / k
            P.append(B((w / k - 0.004, d - 0.02, 0.76), (x, -0.01, 0.48), "white_matte", 0.002))
            P.append(B((0.3, 0.02, 0.012), (x, d / 2 - 0.0, 0.8), "black_metal"))
        P.append(B((w + 0.02, d + 0.02, 0.04), (0, 0.0, 0.88), "stone_counter", 0.003))
        if prm.get("sink_x") is not None:
            sx = prm["sink_x"]
            P += [B((0.52, 0.42, 0.006), (sx, 0.02, 0.903), "steel"), C(0.015, 0.28, (sx, -d / 2 + 0.08, 1.04), "steel"),
                  B((0.02, 0.18, 0.02), (sx, -d / 2 + 0.17, 1.17), "steel")]
        if prm.get("hob_x") is not None:
            P.append(B((0.58, 0.50, 0.006), (prm["hob_x"], 0.0, 0.903), "screen"))
        for x0, x1 in prm.get("upper", []):
            P.append(B((x1 - x0 - 0.004, 0.34, 0.70), ((x0 + x1) / 2, -d / 2 + 0.17, 1.80), "white_matte", 0.002))
    elif t == "fridge":
        P += [B((w, d, h), (0, 0, h / 2), "steel", 0.01), B((0.02, 0.03, 0.5), (w / 2 - 0.06, d / 2 + 0.015, 1.2), "black_metal")]
    elif t == "toilet":
        P += [B((0.36, 0.50, 0.40), (0, 0.07, 0.20), "ceramic", 0.06), B((0.38, 0.17, 0.40), (0, -d / 2 + 0.085, 0.60), "ceramic", 0.03),
              B((0.37, 0.45, 0.03), (0, 0.08, 0.415), "ceramic", 0.012)]
    elif t in ("vanity", "basin_small"):
        P += [B((w, d, 0.45), (0, 0, 0.58), "wood", 0.004), B((w, d, 0.05), (0, 0, 0.83), "ceramic", 0.01),
              C(0.015, 0.2, (0, -d / 2 + 0.07, 0.95), "steel"), B((w * 0.9, 0.02, 0.75), (0, -d / 2 + 0.01, 1.55), "mirror")]
    elif t == "shower":
        P += [B((w, d, 0.05), (0, 0, 0.025), "ceramic", 0.005), B((w - 0.02, 0.008, 1.95), (0, d / 2 - 0.004, 1.025), "glass"),
              C(0.10, 0.02, (0, -d / 2 + 0.25, 2.0), "steel"), B((0.02, 0.25, 0.02), (0, -d / 2 + 0.125, 2.0), "steel")]
    elif t == "bathtub":
        P += [B((w, d, 0.05), (0, 0, 0.025), "ceramic")]
        P += [B((w, 0.06, 0.55), (0, sy * (d / 2 - 0.03), 0.3), "ceramic", 0.01) for sy in (1, -1)]
        P += [B((0.06, d - 0.12, 0.55), (sx * (w / 2 - 0.03), 0, 0.3), "ceramic", 0.01) for sx in (1, -1)]
        P.append(C(0.015, 0.25, (-w / 2 + 0.03, -d / 2 + 0.03, 0.7), "steel"))
    elif t == "washer":
        P += [B((w, d, h), (0, 0, h / 2), "white_matte", 0.02), C(0.2, 0.02, (0, d / 2 + 0.01, 0.45), "screen", "y")]
    return P


def library_model(t, w, d, h, loc, rot):
    """Use a downloaded CC0 model for plants and lamps if its proportions fit."""
    folder = ASSETS / "models" / t
    for f in sorted(folder.glob("*/*.gltf")) + sorted(folder.glob("*/*.glb")) if folder.exists() else []:
        before = set(bpy.data.objects)
        try:
            bpy.ops.import_scene.gltf(filepath=str(f))
        except Exception as e:
            WARN.append(f"model import failed {f.name}: {e}")
            continue
        new = [o for o in bpy.data.objects if o not in before]
        meshes = [o for o in new if o.type == "MESH"]
        bpy.context.view_layer.update()
        if meshes:
            pts = [o.matrix_world @ Vector(c) for o in meshes for c in o.bound_box]
            mn = Vector([min(p[i] for p in pts) for i in range(3)])
            mx = Vector([max(p[i] for p in pts) for i in range(3)])
            dim = mx - mn
            s = h / max(dim.z, 1e-6)
            if dim.x * s <= w * 1.8 and dim.y * s <= d * 1.8:
                holder = empty(f"lib_{t}", loc, rot)
                fit = empty(f"fit_{t}", (0, 0, 0))
                fit.parent = holder
                fit.scale = (s, s, s)
                fit.location = (-(mn.x + mx.x) / 2 * s, -(mn.y + mx.y) / 2 * s, -mn.z * s)
                for o in new:
                    if o.parent is None:
                        o.parent = fit
                return holder
        for o in new:
            bpy.data.objects.remove(o, do_unlink=True)
    return None


def merge_meshes(name, objs, inv):
    """One mesh from several objects (modifiers applied, transforms baked relative to inv, materials merged)."""
    dg = bpy.context.evaluated_depsgraph_get()
    bm, mats = bmesh.new(), []
    for o in objs:
        tmp = bpy.data.meshes.new_from_object(o.evaluated_get(dg), preserve_all_data_layers=True, depsgraph=dg)
        M = inv @ o.matrix_world
        tmp.transform(M)
        while len(tmp.uv_layers) > 1:
            tmp.uv_layers.remove(tmp.uv_layers[-1])
        if not tmp.uv_layers:
            tmp.uv_layers.new(name="UVMap")
        tmp.uv_layers[0].name = "UVMap"
        remap = []
        for m in (list(tmp.materials) or [None]):
            if m not in mats:
                mats.append(m)
            remap.append(mats.index(m))
        for p in tmp.polygons:
            p.material_index = remap[min(p.material_index, len(remap) - 1)]
        if M.determinant() < 0:   # mirrored part: turn the faces the right way out
            b2 = bmesh.new()
            b2.from_mesh(tmp)
            bmesh.ops.reverse_faces(b2, faces=b2.faces[:])
            b2.to_mesh(tmp)
            b2.free()
        bm.from_mesh(tmp)
        bpy.data.meshes.remove(tmp)
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    for m in mats:
        me.materials.append(m)
    return me


def mesh_bounds(me):
    if not len(me.vertices):
        return Vector((0, 0, 0)), Vector((0, 0, 0))
    xs, ys, zs = zip(*(v.co[:] for v in me.vertices))
    return Vector((min(xs), min(ys), min(zs))), Vector((max(xs), max(ys), max(zs)))


def textures_ok(me):
    """True when every image used by the mesh's materials is packed or on disk."""
    for m in me.materials:
        if m is None or not m.use_nodes:
            continue
        for nd in m.node_tree.nodes:
            img = getattr(nd, "image", None)
            if img is None or img.source in ("GENERATED", "VIEWER") or img.packed_file:
                continue
            if not (img.filepath and os.path.exists(bpy.path.abspath(img.filepath))):
                return False
    return True


def import_asset(it, a, limits):
    """Catalog model -> one object normalised to the layout slot, or (None, reason) for the parametric fallback."""
    w, d, h = it["size"]
    before = set(bpy.data.objects)
    try:
        bpy.ops.import_scene.gltf(filepath=a["file"])
    except Exception as e:
        for o in [o for o in bpy.data.objects if o not in before]:
            bpy.data.objects.remove(o, do_unlink=True)
        return None, f"import failed: {e}"
    new = [o for o in bpy.data.objects if o not in before]
    try:
        meshes = [o for o in new if o.type == "MESH"]
        if not meshes:
            return None, "no mesh in the file"
        bpy.context.view_layer.update()
        me = merge_meshes(it["id"], meshes, Matrix.Identity(4))
    finally:
        for o in new:
            bpy.data.objects.remove(o, do_unlink=True)
    turn = {"-Y": math.pi, "+Y": 0.0, "+X": math.pi / 2, "-X": -math.pi / 2}.get(a.get("front_axis", "-Y"), math.pi)
    me.transform(Matrix.Rotation(turn, 4, "Z") @ Matrix.Scale(float(a.get("unit_scale", 1.0)), 4))
    lo, hi = mesh_bounds(me)
    me.transform(Matrix.Translation((-(lo.x + hi.x) / 2, -(lo.y + hi.y) / 2, -lo.z)))   # origin at base centre
    dims = hi - lo
    if min(dims) <= 1e-4:
        return _reject(me, f"flat or empty model {tuple(round(x, 3) for x in dims)}")
    if a.get("fit") == "height":
        s = h / dims.z
        if dims.x * s > w * 1.8 or dims.y * s > d * 1.8:
            return _reject(me, "footprint too big for the slot")
        sc = (s, s, s)
    else:
        sc = (w / dims.x, d / dims.y, h / dims.z)
        su = statistics.median(sc)
        worst = max(abs(x / su - 1) for x in sc)
        if abs(su - 1) > limits["max_uniform_change"] or worst > limits["max_nonuniform"]:
            return _reject(me, f"slot fit: size x{su:.2f}, non-uniform {100 * worst:.0f} %")
    me.transform(Matrix.Diagonal((*sc, 1.0)))
    ob = bpy.data.objects.new(it["id"], me)
    COLL.objects.link(ob)
    tris0 = sum(len(p.vertices) - 2 for p in me.polygons)
    cap = int(a.get("poly_cap", 60000))
    if tris0 > cap:   # polygon cap: collapse decimation, applied into the mesh
        mod = ob.modifiers.new("decimate", "DECIMATE")
        mod.ratio = max(0.01, cap / tris0)
        dg = bpy.context.evaluated_depsgraph_get()
        dec = bpy.data.meshes.new_from_object(ob.evaluated_get(dg), preserve_all_data_layers=True, depsgraph=dg)
        ob.modifiers.remove(mod)
        old, ob.data = ob.data, dec
        bpy.data.meshes.remove(old)
        dec.name = it["id"]
    me = ob.data
    tris = sum(len(p.vertices) - 2 for p in me.polygons)
    lo, hi = mesh_bounds(me)
    dims = hi - lo
    bad = []
    if tris > cap * 1.05:
        bad.append(f"{tris} triangles > cap {cap}")
    if not all(math.isfinite(c) for v in me.vertices for c in v.co):
        bad.append("non-finite vertex")
    if a.get("fit") != "height" and max(abs(dims.x - w) / w, abs(dims.y - d) / d, abs(dims.z - h) / h) > 0.02:
        bad.append(f"size after fit {tuple(round(x, 3) for x in dims)} != slot {(w, d, h)}")
    if not textures_ok(me):
        bad.append("missing textures")
    if bad:
        bpy.data.objects.remove(ob, do_unlink=True)
        bpy.data.meshes.remove(me)
        return None, "QA: " + "; ".join(bad)
    ob.location = (it["center"][0], it["center"][1], it.get("z", 0.0))
    ob.rotation_euler = (0, 0, math.radians(it["rot_deg"]))
    ob["asset_id"], ob["licence"], ob["source"] = a["asset_id"], a.get("licence", ""), a["source"]
    return ob, {"triangles_in": tris0, "triangles": tris, "scale": [round(x, 4) for x in sc]}


def _reject(me, reason):
    bpy.data.meshes.remove(me)
    return None, reason


def parametric_item(it):
    """The built-in model of an item as ONE mesh object named after the item, origin at its base centre."""
    w, d, h = it["size"]
    loc, rot = (it["center"][0], it["center"][1], it.get("z", 0.0)), math.radians(it["rot_deg"])
    root = empty(it["id"] + "__parts", loc, rot)
    for k, p in enumerate(parts(it["type"], w, d, h, it.get("params") or {})):
        if p[0] == "box":
            box(f"{it['id']}_{k}", p[1], p[2], p[3], root, p[4])
        else:
            cyl(f"{it['id']}_{k}", p[1], p[2], p[3], p[4], root, p[5])
    bpy.context.view_layer.update()
    kids = [o for o in root.children_recursive if o.type == "MESH"]
    me = merge_meshes(it["id"], kids, root.matrix_world.inverted())
    for o in kids + [root]:
        bpy.data.objects.remove(o, do_unlink=True)
    ob = bpy.data.objects.new(it["id"], me)
    COLL.objects.link(ob)
    ob.location, ob.rotation_euler = loc, (0, 0, rot)
    ob["source"] = "parametric"
    return ob


def build_furniture():
    qa = []
    items = CHOICE["items"] if CHOICE else {}
    limits = (CHOICE or {}).get("limits", {"max_uniform_change": 0.35, "max_nonuniform": 0.15})
    for it in LAYOUT["items"]:
        w, d, h = it["size"]
        loc, rot = (it["center"][0], it["center"][1], it.get("z", 0.0)), math.radians(it["rot_deg"])
        a = items.get(it["id"], {})
        rec = {"item_id": it["id"], "type": it["type"], "room": it["room"], "planned": a.get("source", "parametric"),
               "asset_id": a.get("asset_id"), "reason": a.get("reason", "")}
        if a.get("source") == "removed":
            qa.append(dict(rec, used="removed"))
            continue
        if CHOICE is None and it["type"] in ("plant", "floor_lamp"):   # no assets.json: original behaviour
            try:
                holder = library_model(it["type"], w, d, h, loc, rot)
                if holder:
                    holder.name = it["id"]
                    qa.append(dict(rec, used="polyhaven", planned="polyhaven", reason="legacy library model"))
                    continue
            except Exception as e:
                WARN.append(f"library model failed for {it['type']}: {e}")
        elif a.get("file"):
            try:
                ob, info = import_asset(it, a, limits)
            except Exception as e:
                ob, info = None, f"import error: {e}"
            if ob:
                qa.append(dict(rec, used=a["source"], reason="", **info))
                continue
            rec["reason"] = info
            WARN.append(f"{it['id']}: model {a['asset_id']} rejected ({info}) - parametric used")
        if it["type"] == "plant":
            WARN.append("plant skipped: no plant model downloaded" if not a.get("file") else f"plant {it['id']} skipped")
            qa.append(dict(rec, used="skipped"))
            continue
        parametric_item(it)
        qa.append(dict(rec, used="parametric"))
    summ = {}
    for r in qa:
        summ.setdefault(r["type"], {}).setdefault(r["used"], 0)
        summ[r["type"]][r["used"]] += 1
    json.dump({"items": qa, "summary": summ}, open(OUT / "furniture_qa.json", "w"), indent=1, ensure_ascii=False)


# ---------- light: sky/HDRI + matching sun + window portals + soft ceiling fill ----------
def build_lights():
    wins = [(o["width"] * (o["head"] - o["sill"]), o) for o in PLAN["windows"] if o.get("rooms")]
    az_out = 0.0
    if wins:
        living = [x for x in wins if room_type(x[1]["rooms"][0]) == "living"] or wins
        o = max(living, key=lambda x: x[0])[1]
        w = WALLS[o["wall_id"]]
        _, u, n, _ = wall_frame(w)
        s = side_into(o["center"], n, w["thickness"], o["rooms"][0])
        az_out = math.degrees(math.atan2(-s * n[1], -s * n[0]))
    az = az_out + 25.0
    el = 32.0
    world = bpy.data.worlds.new("World")
    SC.world = world
    world.use_nodes = True
    nt = world.node_tree
    bg = nt.nodes.get("Background")
    hdr = sorted((ASSETS / "hdri").glob("*.hdr")) if (ASSETS / "hdri").exists() else []
    if hdr:
        side = json.loads(hdr[0].with_suffix(".json").read_text()) if hdr[0].with_suffix(".json").exists() else {}
        el = min(55.0, max(18.0, side.get("sun_el_deg", el)))
        env, tc, mp = nt.nodes.new("ShaderNodeTexEnvironment"), nt.nodes.new("ShaderNodeTexCoord"), nt.nodes.new("ShaderNodeMapping")
        env.image = bpy.data.images.load(str(hdr[0]))
        mp.inputs["Rotation"].default_value[2] = math.radians(side.get("sun_az_deg", 0.0) - az)
        nt.links.new(tc.outputs["Generated"], mp.inputs["Vector"])
        nt.links.new(mp.outputs[0], env.inputs["Vector"])
        nt.links.new(env.outputs["Color"], bg.inputs["Color"])
        bg.inputs["Strength"].default_value = 1.0
    else:
        bg.inputs["Color"].default_value = (0.55, 0.66, 0.85, 1)
        bg.inputs["Strength"].default_value = 1.2
    sd = bpy.data.lights.new("sun", "SUN")
    sd.energy, sd.angle, sd.color = RC.get("sun_strength", 3.5), math.radians(1.0), (1.0, 0.96, 0.9)
    so = bpy.data.objects.new("sun", sd)
    COLL.objects.link(so)
    vec = Vector((math.cos(math.radians(el)) * math.cos(math.radians(az)),
                  math.cos(math.radians(el)) * math.sin(math.radians(az)), math.sin(math.radians(el))))
    so.rotation_euler = (-vec).to_track_quat("-Z", "Y").to_euler()
    for r in ROOMS:
        if r["type"] == "balcony":
            continue
        c = centroid(r["polygon"])
        ld = bpy.data.lights.new(f"fill_{r['id']}", "AREA")
        ld.shape, ld.size, ld.energy, ld.color = "DISK", 0.6, RC.get("fill_w_per_m2", 5.0) * r["area_m2"], (1.0, 0.92, 0.82)
        lo = bpy.data.objects.new(f"fill_{r['id']}", ld)
        COLL.objects.link(lo)
        lo.location = (c[0], c[1], r["ceiling_height"] - 0.05)
        lo.visible_camera = False
    return az, el


def room_type(rid):
    return next((r["type"] for r in ROOMS if r["id"] == rid), None)


# ---------- cameras: corners and wall middles, scored by how much furniture they see ----------
def footprints():
    fps = []
    for it in LAYOUT["items"]:
        if it["type"] == "rug":
            continue
        w, d, _ = it["size"]
        a = math.radians(it["rot_deg"])
        u, n = (math.cos(a), math.sin(a)), (-math.sin(a), math.cos(a))
        cx, cy = it["center"]
        fps.append((it["room"], [(cx + su * u[0] * (w / 2 + 0.2) + sn * n[0] * (d / 2 + 0.2),
                                  cy + su * u[1] * (w / 2 + 0.2) + sn * n[1] * (d / 2 + 0.2))
                                 for su, sn in ((-1, -1), (1, -1), (1, 1), (-1, 1))]))
    return fps


def build_cameras(views_main, views_other):
    bpy.context.view_layer.update()
    dg = bpy.context.evaluated_depsgraph_get()
    fps = footprints()
    cams = []
    for r in ROOMS:
        if r["type"] in ("balcony", "storage") or r["area_m2"] < 1.8:
            continue
        nv = views_main if r["type"] in ("living", "bedroom", "kitchen", "bathroom") else views_other
        p = r["polygon"]
        cen = centroid(p)
        zc = 1.30
        cands = []
        for i in range(len(p)):
            a, b = unit2((p[i - 1][0] - p[i][0], p[i - 1][1] - p[i][1])), unit2((p[(i + 1) % len(p)][0] - p[i][0], p[(i + 1) % len(p)][1] - p[i][1]))
            bis = unit2((a[0] + b[0], a[1] + b[1]))
            pos = (p[i][0] + bis[0] * 0.35, p[i][1] + bis[1] * 0.35)
            if pip(pos, p):
                cands.append((pos, math.atan2(cen[1] - pos[1], cen[0] - pos[0])))
        edges = sorted(range(len(p)), key=lambda i: -math.dist(p[i], p[(i + 1) % len(p)]))[:2]
        for i in edges:
            (x1, y1), (x2, y2) = p[i], p[(i + 1) % len(p)]
            u = unit2((x2 - x1, y2 - y1))
            pos = ((x1 + x2) / 2 - u[1] * 0.3, (y1 + y2) / 2 + u[0] * 0.3)
            if pip(pos, p):
                cands.append((pos, math.atan2(u[0], -u[1])))
        small = min(r["size_m"]) < 2.6
        hfov = 2 * math.atan(18 / (16 if small else 18))
        items = [it for it in LAYOUT["items"] if it["room"] == r["id"]]
        scored = []
        for pos, yaw in cands:
            if any(pip(pos, fp) for rid, fp in fps if rid == r["id"]):
                continue
            o = Vector((pos[0], pos[1], zc))
            sc = 0.0
            for it in items:
                tgt = Vector((it["center"][0], it["center"][1], max(0.3, it["size"][2] / 2)))
                v = tgt - o
                da = (math.atan2(v.y, v.x) - yaw + math.pi) % (2 * math.pi) - math.pi
                if abs(da) > hfov / 2 * 0.9:
                    continue
                hit, loc, *_ = SC.ray_cast(dg, o, v.normalized(), distance=v.length)
                if not hit or (loc - o).length > v.length - 0.6:
                    sc += 1 + it["size"][0] * it["size"][1]
            far = max(math.dist(pos, q) for q in p)
            scored.append((sc + 0.3 * far, pos, yaw))
        scored.sort(key=lambda x: -x[0])
        chosen = []
        for sc, pos, yaw in scored:
            if all(abs((yaw - y2 + math.pi) % (2 * math.pi) - math.pi) > math.radians(50) or math.dist(pos, p2) > 1.5 for _, p2, y2 in chosen):
                chosen.append((sc, pos, yaw))
            if len(chosen) >= nv:
                break
        for k, (sc, pos, yaw) in enumerate(chosen):
            cd = bpy.data.cameras.new(f"cam_{r['id']}_{k + 1}")
            cd.lens, cd.sensor_width, cd.clip_start, cd.clip_end = (16 if small else 18), 36, 0.05, 200
            co = bpy.data.objects.new(cd.name, cd)
            COLL.objects.link(co)
            co.location = (pos[0], pos[1], zc)
            co.rotation_euler = (math.pi / 2, 0, yaw - math.pi / 2)
            cams.append({"name": cd.name, "room": r["id"], "room_type": r["type"], "label": r.get("label", ""),
                         "pos": [round(pos[0], 3), round(pos[1], 3), zc], "yaw_deg": round(math.degrees(yaw), 1), "lens": cd.lens})
        if not chosen:
            WARN.append(f"no free camera position in room {r['id']}")
    return cams


# ---------- collections (deliverable structure) ----------
def organize_collections():
    labels = [r.get("label") or r["type"] for r in ROOMS]
    fname = {}
    for r, base in zip(ROOMS, labels):
        fname[r["id"]] = "Furniture." + (base if labels.count(base) == 1 else f"{base} {r['id']}")
    item_room = {it["id"]: it["room"] for it in LAYOUT["items"]}
    colls = {}

    def coll(name):
        if name not in colls:
            c = bpy.data.collections.new(name)
            SC.collection.children.link(c)
            colls[name] = c
        return colls[name]

    for base in ("Walls", "Floors_Ceilings", "Openings"):
        coll(base)
    for rid in sorted(set(item_room.values()), key=lambda x: int(x[1:]) if x[1:].isdigit() else 0):
        coll(fname.get(rid, "Furniture.other"))
    for base in ("Lights", "Cameras"):
        coll(base)
    for ob in list(SC.collection.objects):
        if ob.parent is not None:
            continue
        n = ob.name
        if n in item_room:
            target = fname.get(item_room[n], "Furniture.other")
        elif re.fullmatch(r"w\d+", n) or n.startswith(("skirt_", "rail_")):
            target = "Walls"
        elif n.startswith(("floor_", "ceiling_", "slab_", "threshold_")):
            target = "Floors_Ceilings"
        elif n.startswith(("win_", "door_")):
            target = "Openings"
        elif ob.type == "LIGHT":
            target = "Lights"
        elif ob.type == "CAMERA":
            target = "Cameras"
        else:
            target = "Misc"
        c = coll(target)
        for o in [ob] + list(ob.children_recursive):
            for uc in list(o.users_collection):
                uc.objects.unlink(o)
            c.objects.link(o)


# ---------- main ----------
prof = RC["profile_settings"]
build_walls()
build_floors()
build_balcony_rails()
build_windows()
build_doors()
build_skirting()
build_furniture()
az, el = build_lights()
cams = build_cameras(prof["views_main"], prof["views_other"])
organize_collections()
SC.render.resolution_x, SC.render.resolution_y = prof["width"], prof["height"]
try:
    SC.view_settings.view_transform = "AgX"
except TypeError:
    pass
SC.view_settings.exposure = RC.get("exposure", 0.0)
json.dump({"cameras": cams, "sun": {"azimuth_deg": az, "elevation_deg": el}, "warnings": WARN},
          open(OUT / "cameras.json", "w"), indent=1, ensure_ascii=False)
bpy.ops.wm.save_as_mainfile(filepath=str(OUT / "scene.blend"))
try:
    bpy.ops.export_scene.gltf(filepath=str(OUT / "scene.glb"), export_format="GLB", export_apply=True, export_cameras=True)
except TypeError:
    bpy.ops.export_scene.gltf(filepath=str(OUT / "scene.glb"), export_format="GLB")
print(f"SCENE_OK objects={len(bpy.data.objects)} cameras={len(cams)} warnings={len(WARN)}")
__END_OF_BLENDER_SCENE_PY__
  cat > "$APP/blender_render.py" <<'__END_OF_BLENDER_RENDER_PY__'
"""Stage 5 - render every camera with Cycles on the GPU (OptiX, then CUDA, then CPU).
Run: blender -b --factory-startup --python-exit-code 1 -P blender_render.py -- <job_dir>
Writes 05_render/<camera>.png, <camera>_depth.npy (for the polish step) and render.json."""
import bpy, json, math, sys, time
from pathlib import Path
import numpy as np

ARGS = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
JOB = Path(ARGS[0])
RC = json.loads((JOB / "run_config.json").read_text())
PROF = RC["profile_settings"]
OUT = JOB / "05_render"
OUT.mkdir(parents=True, exist_ok=True)
bpy.ops.wm.open_mainfile(filepath=str(JOB / "04_scene/scene.blend"))
SC = bpy.context.scene


def setup_device(pref):
    order = [] if pref == "CPU" else [pref] + [d for d in ("OPTIX", "CUDA") if d != pref]
    prefs = bpy.context.preferences.addons["cycles"].preferences
    for typ in order:
        try:
            prefs.compute_device_type = typ
            prefs.refresh_devices()
        except Exception:
            continue
        devs = [d for d in prefs.devices if d.type == typ]
        if devs:
            for d in prefs.devices:
                d.use = d.type == typ
            SC.cycles.device = "GPU"
            return typ, [d.name for d in devs]
    SC.cycles.device = "CPU"
    return "CPU", []


def depth_map(cam, W=320, H=180):
    """Distance from the camera for a coarse pixel grid (ray casts) - control image for SDXL."""
    dg = bpy.context.evaluated_depsgraph_get()
    mw = cam.matrix_world
    o = mw.translation
    tr, br, bl, tl = [mw @ v for v in cam.data.view_frame(scene=SC)]
    d = np.full((H, W), 100.0, np.float32)
    for j in range(H):
        a, b = tl.lerp(bl, (j + 0.5) / H), tr.lerp(br, (j + 0.5) / H)
        for i in range(W):
            v = (a.lerp(b, (i + 0.5) / W) - o).normalized()
            hit, loc, *_ = SC.ray_cast(dg, o, v, distance=100.0)
            if hit:
                d[j, i] = (loc - o).length
    return d


device, gpus = setup_device(RC.get("device", "OPTIX"))
c = SC.cycles
c.samples = PROF["max_samples"]
c.use_adaptive_sampling = True
c.adaptive_threshold = PROF["noise_threshold"]
c.time_limit = PROF["time_limit_s"]
c.use_denoising = True
c.denoiser = "OPENIMAGEDENOISE"
if hasattr(c, "denoising_use_gpu"):
    c.denoising_use_gpu = device != "CPU"
c.max_bounces, c.diffuse_bounces, c.glossy_bounces = 8, 4, 4
c.transmission_bounces, c.transparent_max_bounces = 8, 8
c.sample_clamp_indirect, c.blur_glossy = 10.0, 1.0
c.caustics_reflective = c.caustics_refractive = False
SC.render.resolution_x, SC.render.resolution_y, SC.render.resolution_percentage = PROF["width"], PROF["height"], 100
SC.render.image_settings.file_format = "PNG"
SC.render.image_settings.color_depth = "8"
cams = json.loads((JOB / "04_scene/cameras.json").read_text(encoding="utf-8"))["cameras"]
views = []
for cam in cams:
    ob = bpy.data.objects.get(cam["name"])
    if ob is None:
        continue
    SC.camera = ob
    SC.render.filepath = str(OUT / f"{cam['name']}.png")
    t0 = time.time()
    bpy.ops.render.render(write_still=True)
    t1 = time.time()
    np.save(OUT / f"{cam['name']}_depth.npy", depth_map(ob))
    views.append({"name": cam["name"], "room": cam["room"], "room_type": cam["room_type"],
                  "render_s": round(t1 - t0, 1), "depth_s": round(time.time() - t1, 1)})
    print(f"RENDERED {cam['name']} in {t1 - t0:.1f}s on {device}", flush=True)
json.dump({"device": device, "gpus": gpus, "samples": PROF["max_samples"], "resolution": [PROF["width"], PROF["height"]],
           "views": views}, open(OUT / "render.json", "w"), indent=1)
print(f"RENDER_OK views={len(views)} device={device}")
__END_OF_BLENDER_RENDER_PY__
  cat > "$APP/device_check.py" <<'__END_OF_DEVICE_CHECK_PY__'
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
__END_OF_DEVICE_CHECK_PY__
  cat > "$APP/polish.py" <<'__END_OF_POLISH_PY__'
"""Stage 6 (GPU, optional) - light photo polish: SDXL img2img guided by the render's depth.
A gate compares edges before/after: if lines moved (hallucination), the raw render is kept.
Run: python -m app.polish <job_dir>"""
import sys, time
from pathlib import Path
import cv2
import numpy as np
from .common import jload, jsave, load_config, log

WORDS = {"living": "living room", "bedroom": "bedroom", "kitchen": "kitchen", "bathroom": "bathroom",
         "wc": "small guest toilet", "hall": "entrance hallway", "study": "home office", "dining": "dining room"}
NEG = ("cartoon, illustration, painting, cgi, 3d render, blurry, distorted lines, deformed furniture, "
       "extra furniture, people, text, watermark, oversaturated")


def edge_score(a, b):
    """F1 overlap of Canny edges (2 px tolerance) between two images, 1.0 = same lines."""
    def edges(x):
        x = cv2.resize(x, (960, int(960 * x.shape[0] / x.shape[1])), interpolation=cv2.INTER_AREA)
        return cv2.Canny(cv2.GaussianBlur(cv2.cvtColor(x, cv2.COLOR_BGR2GRAY), (3, 3), 0), 60, 150) > 0
    ea, eb = edges(a), edges(b)
    if ea.sum() < 50:
        return 1.0
    k = np.ones((5, 5), np.uint8)
    da, db = cv2.dilate(ea.astype(np.uint8), k) > 0, cv2.dilate(eb.astype(np.uint8), k) > 0
    p, r = (eb & da).sum() / max(eb.sum(), 1), (ea & db).sum() / max(ea.sum(), 1)
    return float(2 * p * r / max(p + r, 1e-6))


def depth_image(npy, size):
    d = np.load(npy)
    inv = 1.0 / np.maximum(d, 0.1)
    inv = (inv - inv.min()) / max(float(inv.max() - inv.min()), 1e-6)
    g = cv2.resize((inv * 255).astype(np.uint8), size, interpolation=cv2.INTER_CUBIC)
    return np.dstack([g] * 3)


def main(job):
    import torch
    from PIL import Image
    from diffusers import AutoencoderKL, ControlNetModel, StableDiffusionXLControlNetImg2ImgPipeline
    job = Path(job)
    cfg, rc = load_config(), jload(job / "run_config.json")
    pc, prof = cfg["polish"], rc["profile_settings"]
    t0 = time.time()
    cn = ControlNetModel.from_pretrained(cfg["models"]["controlnet"], variant="fp16", torch_dtype=torch.float16)
    vae = AutoencoderKL.from_pretrained(cfg["models"]["vae"], torch_dtype=torch.float16)
    pipe = StableDiffusionXLControlNetImg2ImgPipeline.from_pretrained(
        cfg["models"]["sdxl"], controlnet=cn, vae=vae, variant="fp16", torch_dtype=torch.float16).to("cuda")
    pipe.vae.enable_tiling()
    pipe.set_progress_bar_config(disable=True)
    log(f"polish: models loaded in {time.time() - t0:.0f}s")
    out = job / "06_polish"
    out.mkdir(exist_ok=True)
    rec = []
    for v in jload(job / "05_render/render.json")["views"][:prof.get("polish_max_images", 999)]:
        raw = cv2.imread(str(job / "05_render" / f"{v['name']}.png"))
        h, w = raw.shape[:2]
        W8, H8 = w // 8 * 8, h // 8 * 8
        src = cv2.resize(raw, (W8, H8)) if (W8, H8) != (w, h) else raw
        ctrl = depth_image(job / "05_render" / f"{v['name']}_depth.npy", (W8, H8))
        prompt = (f"professional interior photograph of a {WORDS.get(v['room_type'], 'room')}, {cfg['style']}, "
                  "natural daylight, realistic materials, fine detail")
        t1 = time.time()
        res = pipe(prompt=prompt, negative_prompt=NEG, image=Image.fromarray(cv2.cvtColor(src, cv2.COLOR_BGR2RGB)),
                   control_image=Image.fromarray(ctrl), strength=pc["strength"], num_inference_steps=pc["steps"],
                   guidance_scale=pc["guidance"], controlnet_conditioning_scale=pc["controlnet_scale"],
                   generator=torch.Generator("cuda").manual_seed(1234), height=H8, width=W8).images[0]
        pol = cv2.resize(cv2.cvtColor(np.asarray(res), cv2.COLOR_RGB2BGR), (w, h))
        sc = edge_score(raw, pol)
        ok = sc >= pc["min_edge_score"]
        cv2.imwrite(str(out / f"{v['name']}.png"), pol if ok else raw)
        cv2.imwrite(str(out / f"{v['name']}_compare.jpg"), np.hstack([raw, pol]), [cv2.IMWRITE_JPEG_QUALITY, 88])
        rec.append({"name": v["name"], "edge_score": round(sc, 3), "accepted": bool(ok), "seconds": round(time.time() - t1, 1)})
        log(f"polish {v['name']}: edge score {sc:.2f} -> {'polished' if ok else 'kept raw render'}")
    jsave(out / "polish.json", {"items": rec, "settings": pc})


if __name__ == "__main__":
    main(sys.argv[1])
__END_OF_POLISH_PY__
  cat > "$APP/report.py" <<'__END_OF_REPORT_PY__'
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
__END_OF_REPORT_PY__
  cat > "$APP/main.py" <<'__END_OF_MAIN_PY__'
"""Pipeline runner. Usage: python -m app.main <plan file> [--profile full|quick] [--job-dir DIR]
[--from-stage STAGE] [--to-stage STAGE] [--no-polish] [--redo-vlm]
Stages: parse, vlm, plan, layout, scene, render, polish, export, report. GPU stages run as separate processes.
export writes <job>/final/ (apartment.blend/.glb/.usdc, previews, overlay) and fails the job if validation fails.
For the agentic run (LLM inspects and repairs) use agent.sh / python -m app.agent."""
import argparse, shutil, subprocess, sys, time
from pathlib import Path
from .common import APP, BLENDER, WS, GpuSampler, blender, jload, jsave, load_config, log, run
from .parse import run_parse
from .plan import run_plan
from .layout import run_layout
from .report import write_report

STAGES = ["parse", "vlm", "plan", "layout", "scene", "render", "polish", "export", "report"]
GPU_STAGES = {"vlm", "render", "polish"}


def gpu_present():
    return shutil.which("nvidia-smi") is not None and subprocess.run(["nvidia-smi", "-L"], capture_output=True).returncode == 0


def main():
    ap = argparse.ArgumentParser(description="floor plan -> furnished photoreal renders")
    ap.add_argument("input")
    ap.add_argument("--profile", default="full", choices=["full", "quick"])
    ap.add_argument("--job-dir")
    ap.add_argument("--from-stage", default="parse", choices=STAGES)
    ap.add_argument("--to-stage", default="report", choices=STAGES)
    ap.add_argument("--no-polish", action="store_true")
    ap.add_argument("--redo-vlm", action="store_true")
    ap.add_argument("--wallseg", default="config", choices=["config", "on", "off"],
                    help="trained wall model for scans/photos: config.json setting, on or off")
    a = ap.parse_args()
    inp = Path(a.input).resolve()
    job = Path(a.job_dir) if a.job_dir else WS / "outputs" / f"{inp.stem}_{time.strftime('%Y%m%d-%H%M%S')}"
    (job / "00_input").mkdir(parents=True, exist_ok=True)
    (job / "logs").mkdir(exist_ok=True)
    if not (job / "00_input" / inp.name).exists():
        shutil.copy2(inp, job / "00_input" / inp.name)
    cfg = load_config()
    gpu = gpu_present()
    dev = (jload(APP / "device.json") or {}).get("device", "OPTIX" if gpu else "CPU")
    jsave(job / "run_config.json", {"assets": str(WS / "assets"), "profile": a.profile, "device": dev,
                                    "profile_settings": cfg["profiles"][a.profile], "exposure": cfg["exposure"],
                                    "sun_strength": cfg["sun_strength"], "fill_w_per_m2": cfg["fill_w_per_m2"],
                                    "export": cfg["export"]})
    meta = jload(job / "meta.json", {"stages": {}, "warnings": []})
    meta.update(input=str(inp), profile=a.profile, started=meta.get("started") or time.strftime("%Y-%m-%d %H:%M:%S"))
    sampler = GpuSampler().start() if gpu else None
    failed = None
    for st in STAGES[STAGES.index(a.from_stage):STAGES.index(a.to_stage) + 1]:
        log(f"===== stage {st} =====")
        t0, status = time.time(), "ok"
        stage_gpu = GpuSampler().start() if gpu else None   # VRAM peak of this stage (whole GPU)
        try:
            if st == "parse":
                run_parse(inp, job)
                use = {"on": True, "off": False}.get(a.wallseg, cfg["wallseg"]["enabled"])
                kind = (jload(job / "01_parse/extract.json") or {}).get("kind")
                if use and kind in ("pdf_scan", "photo"):   # optional trained wall model (section 9)
                    try:
                        run([sys.executable, "-m", "app.wallseg", "predict", job], log_file=job / "logs/wallseg.log", cwd=str(WS))
                        used = (jload(job / "01_parse/wallseg.json") or {}).get("used")
                        status = "ok (trained wall model)" if used else "ok (wall model output rejected, classic walls)"
                    except Exception as e:
                        status = f"ok (wall model failed, classic walls used: {e})"
                else:
                    for f in ("wallseg.json", "wallseg_pred.jpg", "wall_mask_classic.png"):
                        (job / "01_parse" / f).unlink(missing_ok=True)
            elif st == "vlm":
                ex = jload(job / "01_parse/extract.json")
                if not ex.get("needs_vlm"):
                    status = "skipped (text read from the file)"
                elif (job / "01_parse/vlm_texts.json").exists() and not a.redo_vlm:
                    status = "reused (01_parse/vlm_texts.json)"
                elif not gpu:
                    status = "skipped (no GPU) - rooms will have no names/areas"
                else:
                    run([sys.executable, "-m", "app.vlm", job], log_file=job / "logs/vlm.log", cwd=str(WS))
            elif st == "plan":
                run_plan(job, cfg)
            elif st == "layout":
                run_layout(job, cfg)
            elif st in ("scene", "render"):
                if st == "scene":
                    try:   # furniture models from the library (03_layout/assets.json); parametric otherwise
                        from .furniture import select_assets
                        sel = select_assets(job, cfg)
                        n = sum(1 for v in sel["items"].values() if v["source"] not in ("parametric", "removed"))
                        status = f"ok ({n} of {len(sel['items'])} furniture items from the model library)"
                    except Exception as e:
                        (job / "03_layout/assets.json").unlink(missing_ok=True)
                        status = f"ok (furniture library not used: {e})"
                blender(f"blender_{st}.py", job)
            elif st == "export":
                from .deliver import deliver
                val = deliver(job, cfg)
                if not val.get("ok"):
                    raise RuntimeError("export validation failed: " + ", ".join(c["name"] for c in val["checks"] if not c["ok"]))
            elif st == "polish":
                if a.no_polish or not cfg["polish"]["enabled"]:
                    status = "skipped (turned off)"
                elif not gpu:
                    status = "skipped (no GPU)"
                else:
                    run([sys.executable, "-m", "app.polish", job], log_file=job / "logs/polish.log", cwd=str(WS))
        except Exception as e:
            if st in ("vlm", "polish"):   # optional AI steps: warn and continue
                status = f"FAILED (continuing without it): {e}"
            else:
                status, failed = f"FAILED: {e}", st
            log(f"stage {st} FAILED: {e}")
        used_gpu = st in GPU_STAGES and status == "ok"
        if st == "render":
            used_gpu = used_gpu and jload(job / "05_render/render.json", {}).get("device") != "CPU"
        meta["stages"][st] = {"seconds": round(time.time() - t0, 1), "gpu": used_gpu, "status": status,
                              "vram_peak_mb": (stage_gpu.stop() if stage_gpu else {}).get("vram_peak_mb")}
        jsave(job / "meta.json", meta)
        if failed:
            break
    meta["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    meta["gpu"] = sampler.stop() if sampler else {}
    jsave(job / "meta.json", meta)
    rep = write_report(job, meta, cfg)
    print(f"\nDONE: {job}\n  rooms: " + "; ".join(f"{r['label'] or r['type']} {r['size_m'][0]:.2f}x{r['size_m'][1]:.2f} m"
                                             for r in rep["rooms"]))
    print(f"  pod time {rep['pod_minutes']} min, GPU stages {rep['gpu_minutes']} min, cost ${rep['cost_usd']}")
    print(f"  report: {job / 'report.md'}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
__END_OF_MAIN_PY__
  cat > "$APP/assets_fetch.py" <<'__END_OF_ASSETS_FETCH_PY__'
"""Setup step - one-time download of CC0 assets: ambientCG textures, Poly Haven sky HDRIs and a few
plant/lamp models. Every download is optional; the scene falls back to plain colours and a sky colour."""
import io, json, sys, urllib.request, zipfile
from pathlib import Path
import cv2
import numpy as np
from .common import WS, log

UA = {"User-Agent": "floorplan-poc-setup/1.0 (Runpod proof of concept; one-time download)"}
AMBIENT = {"WoodFloor051": "2K", "Fabric061": "1K", "Wood049": "1K", "Marble012": "1K", "Carpet016": "1K",
           "PaintedPlaster017": "1K"}
MODELS = {"plant": ["plant", "potted"], "floor_lamp": ["floor lamp", "floor_lamp", "standing lamp"]}


def get(url, timeout=300):
    return urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout).read()


def ph(path):
    return json.loads(get("https://api.polyhaven.com" + path, 120))


def sun_position(p):
    """Brightest spot of an equirectangular sky = sun (azimuth, elevation in degrees, Blender convention)."""
    img = cv2.imread(str(p), cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR).astype(np.float32)
    lum = cv2.GaussianBlur(img.mean(2), (0, 0), 3)
    h, w = lum.shape
    y, x = np.unravel_index(np.argmax(lum[: h // 2]), lum[: h // 2].shape)
    return {"sun_az_deg": round(180.0 * (1 - 2 * (x + 0.5) / w), 2), "sun_el_deg": round(90.0 * (1 - 2 * (y + 0.5) / h), 2)}


def main(dst=WS / "assets"):
    dst = Path(dst)
    for aid, res in AMBIENT.items():
        d = dst / "materials" / aid
        if d.exists() and any(d.iterdir()):
            continue
        try:
            zipfile.ZipFile(io.BytesIO(get(f"https://ambientcg.com/get?file={aid}_{res}-JPG.zip"))).extractall(d)
            log(f"ambientCG {aid}: ok")
        except Exception as e:
            log(f"WARNING ambientCG {aid}: {e} (plain colour will be used)")
    hd = dst / "hdri"
    hd.mkdir(parents=True, exist_ok=True)
    if not list(hd.glob("*.hdr")):
        try:
            skies = ph("/assets?type=hdris&categories=skies")
            for slug, info in sorted(skies.items(), key=lambda kv: -kv[1].get("download_count", 0))[:2]:
                p = hd / f"{slug}_2k.hdr"
                p.write_bytes(get(ph(f"/files/{slug}")["hdri"]["2k"]["hdr"]["url"]))
                p.with_suffix(".json").write_text(json.dumps(sun_position(p)))
                log(f"Poly Haven HDRI {slug}: ok")
        except Exception as e:
            log(f"WARNING Poly Haven HDRIs: {e} (a plain sky colour will be used)")
    try:
        models = ph("/assets?type=models")
        for typ, keys in MODELS.items():
            d = dst / "models" / typ
            if d.exists() and list(d.rglob("*.gltf")):
                continue
            cand = sorted(((info.get("download_count", 0), slug) for slug, info in models.items()
                           if any(k in " ".join([slug, info.get("name", ""), *info.get("tags", []), *info.get("categories", [])]).lower()
                                  for k in keys)), reverse=True)
            for _, slug in cand[: 3 if typ == "plant" else 1]:
                g = ph(f"/files/{slug}")["gltf"]["1k"]["gltf"]
                sd = d / slug
                sd.mkdir(parents=True, exist_ok=True)
                (sd / f"{slug}.gltf").write_bytes(get(g["url"]))
                for rel, inc in (g.get("include") or {}).items():
                    q = sd / rel
                    q.parent.mkdir(parents=True, exist_ok=True)
                    q.write_bytes(get(inc["url"]))
                log(f"Poly Haven model {typ}: {slug}")
    except Exception as e:
        log(f"WARNING Poly Haven models: {e} (plants are skipped, lamps use a simple model)")
    (dst / "LICENSES.txt").write_text("ambientCG textures: CC0 (https://ambientcg.com)\n"
                                      "Poly Haven HDRIs/models: CC0 (https://polyhaven.com/license). Downloaded once via the "
                                      "Poly Haven API for this proof of concept; check the API terms before commercial use.\n")


if __name__ == "__main__":
    main(*(sys.argv[1:2]))
__END_OF_ASSETS_FETCH_PY__
  cat > "$APP/testplans.py" <<'__END_OF_TESTPLANS_PY__'
"""Smoke test: build a known 2+1 apartment as DXF/DWG/vector PDF/scanned PDF/phone photo,
run the whole pipeline on each and check room sizes against the truth.
  python -m app.testplans unit     CPU-only checks, no GPU needed: textnorm, plan geometry (DXF + vector PDF),
                                   layout, patch schema + invariants, agent loop with a mock LLM + replay,
                                   furniture catalog + licence manifest, LLM server command + VRAM planner
                                   (+ Blender scene/export/validation when Blender or a bpy Python is available)
  python -m app.testplans quick    unit + full pipeline on every input + one agent run (GPU pod)"""
import base64, copy, json, math, shutil, struct, subprocess, sys, time, traceback
from pathlib import Path
import cv2
import numpy as np
from .common import WS, APP, BLENDER, jload, jsave, load_config, log, font_path

EXT, INT = 0.25, 0.10
# walls: (axis, fixed coordinate of centreline, start, end, thickness)
WALLS = [("h", 0.125, 0, 11.0, EXT), ("h", 8.375, 0, 11.0, EXT), ("v", 0.125, 0.25, 8.25, EXT),
         ("v", 10.875, 0.25, 8.25, EXT), ("v", 4.80, 0.25, 8.25, INT), ("v", 6.40, 0.25, 8.25, INT),
         ("h", 4.25, 0.25, 4.75, INT), ("h", 3.25, 6.45, 10.75, INT), ("h", 5.45, 6.45, 10.75, INT)]
# openings: (wall index, centre along wall, width, kind, hinge sign, swing sign)
OPENINGS = [(0, 5.60, 0.95, "door", -1, +1), (4, 2.20, 0.90, "door", -1, -1), (4, 6.80, 0.80, "door", +1, -1),
            (5, 1.80, 0.80, "door", -1, +1), (5, 4.35, 0.70, "door", -1, +1), (5, 7.20, 0.80, "door", +1, +1),
            (2, 2.25, 0.90, "window", 0, 0), (0, 2.50, 1.80, "window", 0, 0), (1, 2.50, 1.50, "window", 0, 0),
            (0, 8.60, 1.20, "window", 0, 0), (3, 4.35, 0.60, "window", 0, 0), (1, 8.60, 1.20, "window", 0, 0)]
ROOMS = [("SALON", "living", 0.25, 0.25, 4.75, 4.20), ("EBEVEYN YATAK ODASI", "bedroom", 0.25, 4.30, 4.75, 8.25),
         ("HOL", "hall", 4.85, 0.25, 6.35, 8.25), ("MUTFAK", "kitchen", 6.45, 0.25, 10.75, 3.20),
         ("BANYO", "bathroom", 6.45, 3.30, 10.75, 5.40), ("ÇOCUK ODASI", "bedroom", 6.45, 5.50, 10.75, 8.25)]
BALCONY = (-1.40, 1.00, 0.0, 3.50)


def wall_rects():
    """Wall rectangles with the openings cut out (x0, y0, x1, y1) in metres."""
    rects = []
    for i, (ax, c, s, e, t) in enumerate(WALLS):
        cuts = sorted((m - w / 2, m + w / 2) for wi, m, w, *_ in OPENINGS if wi == i)
        pos = s
        for a, b in cuts + [(e, e)]:
            if a - pos > 1e-6:
                rects.append((pos, c - t / 2, a, c + t / 2) if ax == "h" else (c - t / 2, pos, c + t / 2, a))
            pos = b
    return rects


def symbols():
    """Door leaves + swing arcs, window lines and balcony railing as polylines (metres)."""
    lines, arcs, wins = [], [], []
    for wi, m, w, kind, hs, ss in OPENINGS:
        ax, c, s, e, t = WALLS[wi]
        if kind == "window":
            for off in (-t / 2, 0, t / 2):
                a, b = (m - w / 2, m + w / 2)
                wins.append([(a, c + off), (b, c + off)] if ax == "h" else [(c + off, a), (c + off, b)])
            continue
        hinge = m + hs * w / 2
        face = c + ss * t / 2
        if ax == "h":
            hp, tip = (hinge, face), (hinge, face + ss * w)
            a0 = math.atan2(ss, 0)
            a1 = math.atan2(0, -hs)
        else:
            hp, tip = (face, hinge), (face + ss * w, hinge)
            a0 = math.atan2(0, ss)
            a1 = math.atan2(-hs, 0)
        lines.append([hp, tip])
        d = (a1 - a0 + math.pi) % (2 * math.pi) - math.pi
        arcs.append([(hp[0] + w * math.cos(a0 + d * k / 16), hp[1] + w * math.sin(a0 + d * k / 16)) for k in range(17)])
    bx0, by0, bx1, by1 = BALCONY
    rail = [[(bx1, by0), (bx0, by0), (bx0, by1), (bx1, by1)]]
    return lines, arcs, wins, rail


def labels():
    out = []
    for name, typ, x0, y0, x1, y1 in ROOMS:
        a = (x1 - x0) * (y1 - y0)
        out.append((name, f"{a:.2f}".replace(".", ",") + " m²", (x0 + x1) / 2, (y0 + y1) / 2))
    bx0, by0, bx1, by1 = BALCONY
    out.append(("BALKON", f"{(bx1 - bx0) * (by1 - by0):.2f}".replace(".", ",") + " m²", (bx0 + bx1) / 2, (by0 + by1) / 2))
    return out


def make_dxf(p):
    import ezdxf
    doc = ezdxf.new("R2018", setup=True)
    doc.header["$INSUNITS"] = 5   # centimetres
    for name in ("DUVAR", "KAPI", "PENCERE", "YAZI", "KORKULUK"):
        doc.layers.add(name)
    msp = doc.modelspace()
    cm = lambda pts: [(x * 100, y * 100) for x, y in pts]
    for x0, y0, x1, y1 in wall_rects():
        poly = cm([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])
        h = msp.add_hatch(color=7, dxfattribs={"layer": "DUVAR"})
        h.set_solid_fill()
        h.paths.add_polyline_path(poly, is_closed=True)
        msp.add_lwpolyline(poly, close=True, dxfattribs={"layer": "DUVAR"})
    lines, arcs, wins, rail = symbols()
    for group, layer in ((lines + arcs, "KAPI"), (wins, "PENCERE"), (rail, "KORKULUK")):
        for ln in group:
            msp.add_lwpolyline(cm(ln), dxfattribs={"layer": layer})
    for name, area, x, y in labels():
        msp.add_text(name, height=20, dxfattribs={"layer": "YAZI"}).set_placement((x * 100, y * 100 + 15), align=ezdxf.enums.TextEntityAlignment.MIDDLE_CENTER)
        msp.add_text(area, height=18, dxfattribs={"layer": "YAZI"}).set_placement((x * 100, y * 100 - 20), align=ezdxf.enums.TextEntityAlignment.MIDDLE_CENTER)
    doc.saveas(p)


def make_pdf(p, scale=50):
    from reportlab.pdfgen import canvas
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    pdfmetrics.registerFont(TTFont("DejaVu", font_path()))
    W, H = 1190.55, 841.89   # A3 landscape in points
    k = 72 / 0.0254 / scale  # points per metre
    ox, oy = 180, 90
    P = lambda x, y: (ox + x * k, oy + y * k)
    c = canvas.Canvas(str(p), pagesize=(W, H))
    c.setFillGray(0.05)
    for x0, y0, x1, y1 in wall_rects():
        (a, b), (e, f) = P(x0, y0), P(x1, y1)
        c.rect(a, b, e - a, f - b, stroke=0, fill=1)
    c.setLineWidth(0.25)
    lines, arcs, wins, rail = symbols()
    for ln in lines + arcs + wins + rail:
        path = c.beginPath()
        path.moveTo(*P(*ln[0]))
        for pt in ln[1:]:
            path.lineTo(*P(*pt))
        c.drawPath(path, stroke=1, fill=0)
    for name, area, x, y in labels():
        c.setFont("DejaVu", 10)
        c.drawCentredString(*P(x, y + 0.12), name)
        c.drawCentredString(*P(x, y - 0.25), area)
    c.setFont("DejaVu", 12)
    c.drawString(ox, 40, f"KAT PLANI   ÖLÇEK 1/{scale}")
    c.save()


def make_scan(pdf, out_pdf, out_png):
    import pypdfium2 as pdfium
    from PIL import Image
    img = np.asarray(pdfium.PdfDocument(str(pdf))[0].render(scale=200 / 72).to_pil().convert("L")).astype(np.float32)
    h, w = img.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), 0.6, 1.0)
    img = cv2.warpAffine(img, M, (w, h), borderValue=255)
    img = cv2.GaussianBlur(img, (0, 0), 0.9)
    img = img * np.linspace(0.93, 1.0, w)[None, :] + np.random.default_rng(1).normal(0, 6, img.shape)
    img = np.clip(img, 0, 255).astype(np.uint8)
    cv2.imwrite(str(out_png), img)
    ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 70])
    Image.open(__import__("io").BytesIO(enc.tobytes())).save(out_pdf, "PDF", resolution=200.0)


def make_photo(scan_png, out_jpg):
    img = cv2.imread(str(scan_png))
    img = cv2.resize(img, None, fx=0.75, fy=0.75, interpolation=cv2.INTER_AREA)
    h, w = img.shape[:2]
    W, H = int(w * 1.35), int(h * 1.45)
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([[0.12 * W, 0.10 * H], [0.90 * W, 0.14 * H], [0.95 * W, 0.88 * H], [0.06 * W, 0.84 * H]])
    M = cv2.getPerspectiveTransform(src, dst)
    bg = np.full((H, W, 3), (70, 80, 95), np.uint8)
    warped = cv2.warpPerspective(img, M, (W, H), borderValue=(0, 0, 0))
    mask = cv2.warpPerspective(np.full((h, w), 255, np.uint8), M, (W, H))
    out = np.where(mask[..., None] > 0, warped, bg).astype(np.float32)
    out *= np.linspace(0.8, 1.05, W)[None, :, None]
    cv2.imwrite(str(out_jpg), np.clip(out, 0, 255).astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, 88])


def truth():
    return [{"label": n, "type": t, "size": sorted([round(x1 - x0, 3), round(y1 - y0, 3)]),
             "area": round((x1 - x0) * (y1 - y0), 3)} for n, t, x0, y0, x1, y1 in ROOMS]


def make_all(d):
    d = Path(d)
    d.mkdir(parents=True, exist_ok=True)
    make_dxf(d / "test_plan.dxf")
    make_pdf(d / "test_plan_vector.pdf")
    make_scan(d / "test_plan_vector.pdf", d / "test_plan_scan.pdf", d / "_scan.png")
    make_photo(d / "_scan.png", d / "test_plan_photo.jpg")
    files = {"dxf": d / "test_plan.dxf", "pdf_vector": d / "test_plan_vector.pdf",
             "pdf_scan": d / "test_plan_scan.pdf", "photo": d / "test_plan_photo.jpg"}
    tool = WS / "opt/libredwg/bin/dxf2dwg"
    if tool.exists():
        try:
            subprocess.run([str(tool), "-y", "-o", str(d / "test_plan.dwg"), str(d / "test_plan.dxf")],
                           check=True, capture_output=True, timeout=300)
            if (d / "test_plan.dwg").stat().st_size > 1000:
                files["dwg"] = d / "test_plan.dwg"
        except Exception as e:
            log(f"dxf2dwg could not write a test DWG ({e}) - DWG input is skipped in the smoke test")
    jsave(d / "truth.json", truth())
    return files


def check(job, kind):
    """Compare detected rooms with the truth. Tolerance: 5 cm (vector) or 3 % (raster)."""
    plan = jload(Path(job) / "02_plan/plan.json", {})
    rooms = [r for r in plan.get("rooms", []) if r["type"] != "balcony"]
    raster = kind in ("pdf_scan", "photo")
    res, ok = [], True
    for t in truth():
        best, err = None, 1e9
        for r in rooms:
            if r["type"] != t["type"]:
                continue
            s = sorted(r["size_m"])
            e = max(abs(s[0] - t["size"][0]) / (t["size"][0] if raster else 1),
                    abs(s[1] - t["size"][1]) / (t["size"][1] if raster else 1))
            if e < err:
                best, err = r, e
        tol = 0.03 if raster else 0.05
        good = best is not None and err <= tol
        ok &= good
        res.append(f"  {t['label']:<22} truth {t['size'][0]:.2f}x{t['size'][1]:.2f}  found "
                   + (f"{sorted(best['size_m'])[0]:.2f}x{sorted(best['size_m'])[1]:.2f}" if best else "-")
                   + f"  {'OK' if good else 'FAIL'}")
    files = ["02_plan/plan.json", "02_plan/overlay.png", "03_layout/layout.json", "04_scene/scene.blend",
             "04_scene/scene.glb", "report.json", "final/apartment.blend", "final/apartment.glb",
             "final/apartment.usdc", "final/overlay.png", "final/report.md"]
    missing = [f for f in files if not (Path(job) / f).exists()]
    renders = list((Path(job) / "05_render").glob("*.png"))
    if not [r for r in renders if "depth" not in r.name]:
        missing.append("05_render/*.png")
    val = jload(Path(job) / "final/validation.json", {})
    if not val.get("ok"):
        missing.append("final/validation.json ok (failed: " + ", ".join(c["name"] for c in val.get("checks", []) if not c["ok"]) + ")")
    ok &= not missing
    return ok, res, missing


# ---------- CPU-only unit checks ----------
def _gltf_box(path, size, node_scale=None, glb=False, image_uri=None):
    """Minimal glTF 2.0 box (size in glTF axes, Y up) for catalog tests - no Blender needed."""
    sx, sy, sz = size
    v = [(x * sx / 2, y * sy / 2 + sy / 2, z * sz / 2) for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]
    idx = [0, 1, 3, 0, 3, 2, 4, 6, 7, 4, 7, 5, 0, 4, 5, 0, 5, 1, 2, 3, 7, 2, 7, 6, 0, 2, 6, 0, 6, 4, 1, 5, 7, 1, 7, 3]
    pos = b"".join(struct.pack("<3f", *p) for p in v)
    ind = b"".join(struct.pack("<H", i) for i in idx) + b"\0\0"
    data = pos + ind
    node = {"mesh": 0}
    if node_scale:
        node["scale"] = node_scale
    js = {"asset": {"version": "2.0"}, "scene": 0, "scenes": [{"nodes": [0]}], "nodes": [node],
          "meshes": [{"primitives": [{"attributes": {"POSITION": 0}, "indices": 1}]}],
          "accessors": [{"bufferView": 0, "componentType": 5126, "count": 8, "type": "VEC3",
                         "min": [min(p[i] for p in v) for i in range(3)], "max": [max(p[i] for p in v) for i in range(3)]},
                        {"bufferView": 1, "componentType": 5123, "count": 36, "type": "SCALAR"}],
          "bufferViews": [{"buffer": 0, "byteOffset": 0, "byteLength": len(pos)},
                          {"buffer": 0, "byteOffset": len(pos), "byteLength": 72}],
          "buffers": [{"byteLength": len(data)}]}
    if image_uri:
        js["images"] = [{"uri": image_uri}]
    path.parent.mkdir(parents=True, exist_ok=True)
    if glb:
        j = json.dumps(js).encode()
        j += b" " * (-len(j) % 4)
        body = struct.pack("<II", len(j), 0x4E4F534A) + j + struct.pack("<II", len(data), 0x004E4942) + data
        path.write_bytes(b"glTF" + struct.pack("<II", 2, 12 + len(body)) + body)
    else:
        js["buffers"][0]["uri"] = "data:application/octet-stream;base64," + base64.b64encode(data).decode()
        path.write_text(json.dumps(js))


def _unit_catalog(tmp):
    from . import furniture
    cfg = load_config()
    cfg["furniture"].update(library=str(tmp / "lib"), user_dir=str(tmp / "user"), legacy_models=str(tmp / "none"))
    u = tmp / "user"
    _gltf_box(u / "sofa/a.glb", (2.0, 0.8, 0.9), glb=True)                      # 2.0 w x 0.9 d x 0.8 h after import
    (u / "sofa/a.json").write_text(json.dumps({"licence": "CC0", "style_tags": ["modern"]}))
    _gltf_box(u / "sofa/b_cm.gltf", (210, 85, 90))                              # centimetres
    (u / "sofa/b_cm.json").write_text(json.dumps({"licence": "CC BY 4.0", "author": "A. Author", "source_url": "https://x.invalid"}))
    _gltf_box(u / "sofa/c_scaled.gltf", (1, 1, 1), node_scale=[1.8, 0.85, 0.9])  # node transform
    (u / "sofa/licence.json").write_text(json.dumps({"licence": "owned"}))
    _gltf_box(u / "wardrobe/nolic.glb", (2, 2.2, 0.6), glb=True)                 # no licence -> skipped
    _gltf_box(u / "desk/missing_tex.gltf", (1.2, 0.75, 0.6), image_uri="tex/nothere.png")
    (u / "desk/missing_tex.json").write_text(json.dumps({"licence": "CC0-1.0"}))
    _gltf_box(u / "sofa/gpl.glb", (2.0, 0.8, 0.9), glb=True)
    (u / "sofa/gpl.json").write_text(json.dumps({"licence": "GPL-3.0"}))
    cat = furniture.build_catalog(cfg)
    ids = {i["id"]: i for i in cat["items"]}
    reasons = {Path(e["file"]).name: e["reason"] for e in cat["excluded"]}
    assert set(ids) == {"user:sofa_a", "user:sofa_b_cm", "user:sofa_c_scaled"}, sorted(ids)
    assert ids["user:sofa_a"]["dims_m"] == [2.0, 0.9, 0.8], ids["user:sofa_a"]["dims_m"]
    assert ids["user:sofa_b_cm"]["unit_scale"] == 0.01 and ids["user:sofa_b_cm"]["dims_m"] == [2.1, 0.9, 0.85]
    assert ids["user:sofa_c_scaled"]["dims_m"] == [1.8, 0.9, 0.85], ids["user:sofa_c_scaled"]["dims_m"]
    assert ids["user:sofa_b_cm"]["licence"] == "CC-BY-4.0" and ids["user:sofa_c_scaled"]["licence"] == "owned"
    assert "licence" in reasons["nolic.glb"] and "GPL-3.0" in reasons["gpl.glb"] and "missing files" in reasons["missing_tex.gltf"]
    lic = (tmp / "lib/LICENSES.txt").read_text()
    assert "Powered by Poly Haven" in lic and "A. Author" in lic and "Attribution required" in lic and "Skipped files" in lic
    job = tmp / "job"
    (job / "03_layout").mkdir(parents=True)
    jsave(job / "03_layout/layout.json", {"items": [
        {"id": "r1_sofa_0", "type": "sofa", "room": "r1", "size": [2.10, 0.90, 0.85], "center": [0, 0], "rot_deg": 0},
        {"id": "r1_sofa_1", "type": "sofa", "room": "r1", "size": [1.20, 0.80, 0.85], "center": [0, 0], "rot_deg": 0},
        {"id": "r1_rug", "type": "rug", "room": "r1", "size": [2.4, 1.7, 0.01], "center": [0, 0], "rot_deg": 0}]})
    sel = furniture.select_assets(job, cfg)["items"]
    # style match ('modern' is in the config style) ranks before the smaller fit error of the cm sofa
    assert sel["r1_sofa_0"]["asset_id"] == "user:sofa_a" and sel["r1_sofa_0"]["style"] == 1, sel["r1_sofa_0"]
    assert sel["r1_sofa_1"]["source"] == "parametric" and "none fits" in sel["r1_sofa_1"]["reason"], sel["r1_sofa_1"]
    assert sel["r1_rug"]["source"] == "parametric"
    assert furniture.set_override(job, cfg, "r1_sofa_0", "user:sofa_b_cm") == []
    assert furniture.select_assets(job, cfg)["items"]["r1_sofa_0"]["asset_id"] == "user:sofa_b_cm"
    assert furniture.set_override(job, cfg, "r1_sofa_1", "user:sofa_a")          # does not fit -> error list
    return f"{len(ids)} models, {len(cat['excluded'])} skipped with reasons, unit/transform/licence/fit/override OK"


def _unit_patches(job):
    from .patches import PatchStore, validate_patch
    ps = PatchStore(job)
    plan, _ = ps.after_plan_stage()
    assert validate_patch({"ops": [{"op": "explode"}], "reason": "xx x"})
    assert validate_patch({"ops": [{"op": "scale_plan", "factor": 1.1, "vertices": [1]}], "reason": "xxxx"})
    assert validate_patch({"ops": [{"op": "scale_plan", "factor": 1.1}]})
    r = ps.propose({"ops": [{"op": "scale_plan", "factor": 1.9}], "reason": "too big"})
    assert not r["accepted"] and r["stage"] == "invariants", r
    r = ps.propose({"ops": [{"op": "remove_opening", "id": d["id"]} for d in plan["doors"]][:8], "reason": "remove all"})
    assert not r["accepted"], r
    hall = next(x["id"] for x in plan["rooms"] if x["type"] == "hall")
    r = ps.propose({"ops": [{"op": "retype_room", "room_id": hall, "type": "study"}, {"op": "scale_plan", "factor": 1.02}],
                    "reason": "test"})
    assert r["accepted"], r
    now = jload(job / "02_plan/plan.json")
    rb, _ = ps.rebuild()
    assert json.dumps(now, sort_keys=True) == json.dumps(rb, sort_keys=True), "rebuild differs"
    return "schema errors, invariant rejections, accepted patch and deterministic rebuild OK"


MOCK_TURNS = [
    {"content": "start", "calls": [{"name": "parse_plan", "arguments": {}}]},
    {"content": "bad calls", "calls": [{"name": "patch_plan", "arguments": "{not json"}, {"name": "draw_walls", "arguments": {}},
                                       {"name": "get_plan_summary", "arguments": {"detail": "everything"}}]},
    {"content": "too big", "calls": [{"name": "patch_plan", "arguments": {"ops": [{"op": "scale_plan", "factor": 1.9}], "reason": "test"}}]},
    {"content": "retype", "calls": [{"name": "patch_plan", "arguments": {"ops": [{"op": "retype_room", "room_id": "r2", "type": "study"}],
                                                                          "reason": "unit test"}}]},
    {"content": "checks", "calls": [{"name": "run_checks", "arguments": {}}]},
    {"content": "furnish", "calls": [{"name": "relayout", "arguments": {}}, {"name": "choose_furniture", "arguments": {}}]},
    {"content": "drop sofa", "calls": [{"name": "patch_plan", "arguments": {"ops": [{"op": "remove_furniture", "item_id": "r5_sofa_0"}],
                                                                             "reason": "unit test"}}]},
    {"content": "finish", "calls": [{"name": "finish", "arguments": {"status": "complete", "summary": "unit"}}]},
    {"content": "finish again", "calls": [{"name": "finish", "arguments": {"status": "partial", "summary": "unit", "uncertain": ["cpu"]}}]},
]


def _unit_agent(inp, tmp):
    from .agent import ScriptBackend, run_agent, replay_turns
    from .vision import NoVision, ScriptedVision
    cfg = load_config()
    cfg["furniture"].update(library=str(tmp / "lib"), user_dir=str(tmp / "user"))
    run_agent(inp, tmp / "a", cfg, ScriptBackend(copy.deepcopy(MOCK_TURNS)), NoVision())
    recs = [json.loads(l) for l in (tmp / "a/agent_log.jsonl").read_text().splitlines()]
    tools = [r for r in recs if r["type"] == "tool"]
    want = [("parse_plan", True), ("patch_plan", False), ("draw_walls", False), ("get_plan_summary", False),
            ("patch_plan", True), ("patch_plan", True), ("run_checks", True), ("relayout", True), ("choose_furniture", True),
            ("patch_plan", True), ("finish", True), ("finish", True)]
    assert [(t["tool"], t["ok"]) for t in tools] == want, [(t["tool"], t["ok"]) for t in tools]
    assert not tools[4]["result"]["accepted"] and tools[5]["result"]["accepted"] and tools[9]["result"]["accepted"]
    assert not tools[10]["result"]["accepted"] and tools[11]["result"]["accepted"], "finish guard"
    assert jload(tmp / "a/03_layout/asset_overrides.json") == {"r5_sofa_0": "remove"}
    assert (tmp / "a/final/report.md").exists() and "Still uncertain" in (tmp / "a/final/report.md").read_text()
    turns, answers = replay_turns(tmp / "a/agent_log.jsonl")
    run_agent(inp, tmp / "b", cfg, ScriptBackend(turns), ScriptedVision(answers))
    for rel in ("02_plan/plan.json", "02_plan/patches.json", "03_layout/layout.json", "03_layout/assets.json",
                "03_layout/asset_overrides.json"):
        a, b = jload(tmp / "a" / rel), jload(tmp / "b" / rel)
        if isinstance(a, list):
            a, b = [{k: v for k, v in x.items() if k != "ts"} for x in a], [{k: v for k, v in x.items() if k != "ts"} for x in b]
        assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True), f"replay differs: {rel}"
    return f"{len(tools)} tool calls (4 refused as designed), finish guard, replay identical"


def _unit_server():
    from .agent import parse_text_tool_calls, openai_tools
    from .llm_server import LLMServer, VramPlanner, memory_utilization
    from .vision import parse_json
    cfg = load_config()
    s = LLMServer(cfg)
    cmd = " ".join(s.command())
    for flag in ("serve", "--enable-auto-tool-choice", "--tool-call-parser qwen3_coder", "--reasoning-parser qwen3",
                 "--default-chat-template-kwargs", "--max-model-len", "--gpu-memory-utilization", "--limit-mm-per-prompt"):
        assert flag in cmd, flag
    assert memory_utilization(cfg, "Qwen/Qwen3.5-9B", 46068) == 0.57, memory_utilization(cfg, "Qwen/Qwen3.5-9B", 46068)
    assert memory_utilization(cfg, "Qwen/Qwen3.5-4B", 24564) == 0.65
    assert memory_utilization(cfg, "Qwen/Qwen3.6-27B-FP8", 46068) == 0.8

    class FakeServer:
        def __init__(self, mb):
            self.mb, self.up = mb, True

        def running(self):
            return self.up

        def budget_mb(self):
            return self.mb

        def stop(self, wait_free=True):
            self.up = False

    for total, srv, comp, keep in ((46068, 26259, "cycles_preview", True), (46068, 26259, "vlm_transformers_9b", False),
                                   (24564, 15967, "cycles_preview", True), (24564, 15967, "sdxl_polish", False)):
        f = FakeServer(srv)
        p = VramPlanner(cfg, f)
        p.total = total
        p.before(comp)
        assert f.up == keep, (total, comp)
    calls = parse_text_tool_calls('<tool_call>\n<function=patch_plan>\n<parameter=reason>\nx\n</parameter>\n'
                                  '<parameter=ops>\n[{"op": "scale_plan", "factor": 1.1}]\n</parameter>\n</function>\n</tool_call>')
    assert calls and json.loads(calls[0]["function"]["arguments"])["ops"][0]["factor"] == 1.1, calls
    assert parse_json('blah {"overall": "minor", "issues": []} end')["overall"] == "minor"
    assert len(openai_tools()) == 12
    return "vLLM command flags, gpu-memory-utilization estimates, VRAM planner, text tool-call + JSON parsing, 12 tools"


def _unit_textnorm():
    from .textnorm import classify
    assert classify("EBEVEYN BANYO")["type"] == "bathroom"
    assert classify("SALON 18,50 m²") == {"kind": "room_label", "type": "living", "area": 18.5}
    assert classify("ÖLÇEK 1/50") == {"kind": "scale", "value": 50}
    assert classify("ÇOCUK ODASI")["child"] is True
    return "room words, areas, scale notes"


def unit():
    """CPU-only checks. Returns True when all pass. Blender checks run only when Blender (or bpy) is available."""
    tmp = WS / "outputs/unit_test"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    d = WS / "inputs/smoke_test"
    files = {"dxf": d / "test_plan.dxf", "pdf_vector": d / "test_plan_vector.pdf"}
    if not all(f.exists() for f in files.values()):
        files = {k: v for k, v in make_all(d).items() if k in files}
    results = []

    def T(name, fn):
        t0 = time.time()
        try:
            results.append((name, True, fn(), time.time() - t0))
        except Exception as e:
            results.append((name, False, f"{type(e).__name__}: {e}\n{traceback.format_exc()[-800:]}", time.time() - t0))

    T("textnorm", _unit_textnorm)
    for kind, f in files.items():
        def geo(kind=kind, f=f):
            job = tmp / f"geo_{kind}"
            r = subprocess.run([sys.executable, "-m", "app.main", str(f), "--job-dir", str(job), "--to-stage", "layout"],
                               cwd=str(WS), capture_output=True, text=True)
            assert r.returncode == 0, r.stdout[-1500:] + r.stderr[-1500:]
            ok, lines, _ = check(job, kind)
            assert all(l.endswith("OK") for l in lines), "\n".join(lines)
            lay = jload(job / "03_layout/layout.json")
            assert all(v == 0 for v in lay["checks"].values()) and not lay["unfurnished_main_rooms"], lay["checks"]
            return f"{len(lines)} rooms within tolerance, {len(lay['items'])} furniture items, layout checks 0"
        T(f"plan+layout {kind}", geo)
    T("patches", lambda: _unit_patches(_copy_job(tmp / "geo_dxf", tmp / "patch_job")))
    T("agent mock + replay", lambda: _unit_agent(files["dxf"], tmp / "agent"))
    T("catalog + licences", lambda: _unit_catalog(tmp / "catalog"))
    T("llm server + planner", _unit_server)
    if Path(BLENDER).exists():
        def bl():
            from .common import blender
            from .deliver import deliver
            job = tmp / "agent/a"
            blender("blender_scene.py", job)
            val = deliver(job)
            bad = [f"{c['name']}: {c['detail']}" for c in val["checks"] if not c["ok"]]
            assert val["ok"], bad
            return f"scene + export + {len(val['checks'])} validation checks passed"
        T("blender scene/export/validate (CPU)", bl)
    else:
        results.append(("blender scene/export/validate", None, f"skipped: no Blender at {BLENDER}", 0.0))
    print("\n===== UNIT CHECKS (CPU) =====")
    for name, ok, msg, sec in results:
        print(f"{'PASS' if ok else 'SKIP' if ok is None else 'FAIL'}  {name:<36} {sec:6.1f}s  {msg}")
    all_ok = all(ok is not False for _, ok, _, _ in results)
    print(f"UNIT CHECKS: {'PASS' if all_ok else 'FAIL'}")
    return all_ok


def _copy_job(src, dst):
    shutil.copytree(src, dst)
    return dst


def agent_smoke(f):
    """One agentic run on the DXF test plan: LLM backend if the agent LLM is installed, else the rules backend."""
    cfg = load_config()
    backend = cfg["agent"]["backend"]
    if backend == "llm" and not (Path(cfg["llm"]["venv"]) / "bin/vllm").exists():
        backend = "rules"
    job = WS / "outputs/smoke_agent"
    if job.exists():
        shutil.rmtree(job)
    t0 = time.time()
    r = subprocess.run([sys.executable, "-m", "app.agent", "run", str(f), "--job-dir", str(job), "--backend", backend],
                       cwd=str(WS))
    s = jload(job / "final/agent_summary.json", {})
    ok, lines, missing = check(job, "dxf")
    missing = [m for m in missing if not m.startswith(("04_scene/scene.glb", "05_render", "report.json"))]
    tools = [json.loads(l) for l in (job / "agent_log.jsonl").read_text().splitlines()] if (job / "agent_log.jsonl").exists() else []
    peak = max([t.get("vram_peak_mb") or 0 for t in tools] or [0])
    good = r.returncode == 0 and s.get("export_ok") and not missing and all(l.endswith("OK") for l in lines)
    out = [f"agent ({backend}) {'PASS' if good else 'FAIL'}  ({time.time() - t0:.0f} s, status {s.get('status')}, "
           f"{s.get('steps')} tool calls, VRAM peak {peak} MB)  job: {job}"] + lines
    out += [f"  missing: {', '.join(missing)}"] if missing else []
    out += [f"  uncertain: {u}" for u in (s.get("uncertain") or [])[:5]]
    return good, out


def smoke(profile="quick", polish=True):
    unit_ok = unit()
    d = WS / "inputs/smoke_test"
    files = make_all(d)
    summary, all_ok = [f"unit checks {'PASS' if unit_ok else 'FAIL'}"], unit_ok
    for kind, f in files.items():
        job = WS / "outputs" / f"smoke_{kind}"
        if job.exists():
            shutil.rmtree(job)
        cmd = [sys.executable, "-m", "app.main", str(f), "--job-dir", str(job), "--profile", profile]
        if not polish or kind not in ("dxf", "pdf_scan"):
            cmd.append("--no-polish")   # polish is tested on two inputs to save time
        t0 = time.time()
        r = subprocess.run(cmd, cwd=str(WS))
        ok, lines, missing = check(job, kind) if r.returncode == 0 else (False, [], ["pipeline crashed"])
        all_ok &= ok
        summary.append(f"{kind:<11} {'PASS' if ok else 'FAIL'}  ({time.time() - t0:.0f} s)  job: {job}")
        summary += lines + ([f"  missing: {', '.join(missing)}"] if missing else [])
    ok, lines = agent_smoke(files["dxf"])
    all_ok &= ok
    summary += lines
    print("\n===== SMOKE TEST =====\n" + "\n".join(summary))
    print(f"SMOKE TEST: {'PASS' if all_ok else 'FAIL'}")
    return all_ok


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "quick"
    ok = unit() if mode == "unit" else smoke(profile=mode)
    sys.exit(0 if ok else 1)
__END_OF_TESTPLANS_PY__
  cat > "$APP/evaluate.py" <<'__END_OF_EVALUATE_PY__'
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
__END_OF_EVALUATE_PY__
  cat > "$APP/wallseg.py" <<'__END_OF_WALLSEG_PY__'
"""Optional wall/door/window model for scans and phone photos (section 9, finetune.sh).
Teacher data = your own DWG/DXF/vector-PDF plans: the pipeline reads them exactly, the correct labels are
drawn from plan.json, and training turns the clean pages into fake scans/photos on the fly.
  python -m app.wallseg prepare --sources DIR --data DIR      image + label pairs (CPU)
  python -m app.wallseg train --data DIR --run DIR ...        train, or resume after a stop (GPU)
  python -m app.wallseg install --run DIR --min-iou 0.8       switch the pipeline to the new model
  python -m app.wallseg predict JOB                           used by main.py for scans and photos
  python -m app.wallseg compare --run DIR --inputs DIR        before/after table on raster test plans
Labels: 0 background, 1 wall, 2 door opening, 3 window. Small U-Net trained from zero: no third-party
weights, so no extra licence. The pipeline uses the wall class; doors/windows teach it where the gaps are."""
import argparse, csv, hashlib, json, math, os, re, shutil, subprocess, sys, time
from pathlib import Path
import cv2
import numpy as np
from .common import WS, jload, jsave, load_config, log

RASTER = ("pdf_scan", "photo")
COLORS = np.array([[0, 0, 0], [40, 40, 230], [40, 170, 40], [230, 90, 30]], np.uint8)   # BGR: wall red, door green, window blue


# ---------- labels from plan.json ----------
def draw_mask(plan, shape):
    a, _, c, _, e, f = plan["debug"]["m_to_px"]
    m = np.zeros(shape[:2], np.uint8)
    walls = {w["id"]: w for w in plan["walls"]}

    def rect(cx, cy, ux, uy, hl, ht):
        pts = [(a * (cx + su * ux * hl - sn * uy * ht) + c, e * (cy + su * uy * hl + sn * ux * ht) + f)
               for su, sn in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
        return np.round(np.array(pts)).astype(np.int32)

    def seg_dist(p, w):
        (x0, y0), (x1, y1) = w["a"], w["b"]
        L2 = (x1 - x0) ** 2 + (y1 - y0) ** 2 or 1e-12
        t = max(0.0, min(1.0, ((p[0] - x0) * (x1 - x0) + (p[1] - y0) * (y1 - y0)) / L2))
        return math.hypot(p[0] - x0 - t * (x1 - x0), p[1] - y0 - t * (y1 - y0))

    for w in plan["walls"]:
        (x0, y0), (x1, y1) = w["a"], w["b"]
        L = math.hypot(x1 - x0, y1 - y0) or 1e-9
        ux, uy, ht = (x1 - x0) / L, (y1 - y0) / L, w["thickness"] / 2
        # extend an end by half a thickness only if the extension stays inside another wall (fills L-corners,
        # never sticks out of an outer wall and keeps free ends exact)
        ext = [ht if any(o is not w and seg_dist((p[0] + sg * ux * ht, p[1] + sg * uy * ht), o) <= o["thickness"] / 2 + 0.01
                         for o in plan["walls"]) else 0.0 for p, sg in ((w["a"], -1), (w["b"], 1))]
        s0, s1 = -ext[0], L + ext[1]
        cv2.fillPoly(m, [rect(x0 + ux * (s0 + s1) / 2, y0 + uy * (s0 + s1) / 2, ux, uy, (s1 - s0) / 2, ht)], 1)
    for o in plan["doors"] + plan["windows"]:
        w = walls.get(o["wall_id"])
        if not w:
            continue
        (x0, y0), (x1, y1) = w["a"], w["b"]
        L = math.hypot(x1 - x0, y1 - y0) or 1e-9
        val = 3 if o in plan["windows"] and o.get("kind") != "balcony_door" else 2
        cv2.fillPoly(m, [rect(o["center"][0], o["center"][1], (x1 - x0) / L, (y1 - y0) / L, o["width"] / 2, w["thickness"] / 2 + 0.01)], val)
    return m


def colorize(gray, lab):
    out = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    fg = lab > 0
    out[fg] = (0.45 * out[fg] + 0.55 * COLORS[lab[fg]]).astype(np.uint8)
    return out


def prepare(sources, data, min_sources):
    sources, data = Path(sources), Path(data)
    (data / "pairs").mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in sources.rglob("*") if p.suffix.lower() in (".dwg", ".dxf", ".pdf"))
    log(f"prepare: {len(files)} source files in {sources}")
    for f in files:
        name = re.sub(r"[^A-Za-z0-9_-]+", "_", f.stem)[:50] + "_" + hashlib.md5(str(f.resolve()).encode()).hexdigest()[:6]
        d = data / "pairs" / name
        if (d / "meta.json").exists():
            continue   # done in an earlier run
        d.mkdir(parents=True, exist_ok=True)
        job, t0 = data / "jobs" / name, time.time()
        r = subprocess.run([sys.executable, "-m", "app.main", str(f), "--job-dir", str(job), "--to-stage", "plan",
                            "--wallseg", "off"], cwd=str(WS), capture_output=True, text=True)
        plan = jload(job / "02_plan/plan.json")
        meta = {"name": name, "source": str(f), "ok": False}
        if r.returncode or not plan:
            meta["reason"] = "pipeline failed: " + (r.stdout + r.stderr).strip()[-300:]
        else:
            kind = plan["source"]["kind"]
            lab = [x for x in plan["rooms"] if x.get("label_area_m2")]
            good = [x for x in lab if abs(x["area_m2"] / x["label_area_m2"] - 1) <= 0.03]
            meta.update(kind=kind, walls=len(plan["walls"]), rooms=len(plan["rooms"]), labeled_rooms=len(lab), matching_rooms=len(good))
            if kind not in ("dxf", "pdf_vector"):
                meta["reason"] = f"{kind} is not a vector source, so there are no exact labels"
            elif len(plan["walls"]) < 6 or len(plan["rooms"]) < 2:
                meta["reason"] = "too few walls or rooms found"
            elif kind == "pdf_vector" and (len(lab) < 2 or len(good) < 0.7 * len(lab)):
                meta["reason"] = "room sizes do not match the m2 labels, so the walls are not trusted"
            else:
                bg = cv2.imread(str(job / plan["debug"]["background"]), cv2.IMREAD_GRAYSCALE)
                mask = draw_mask(plan, bg.shape)
                cv2.imwrite(str(d / "image.png"), bg)
                cv2.imwrite(str(d / "mask.png"), mask)
                k = min(1.0, 1600 / max(bg.shape))
                cv2.imwrite(str(d / "preview.jpg"), cv2.resize(colorize(bg, mask), None, fx=k, fy=k, interpolation=cv2.INTER_AREA))
                a = abs(plan["debug"]["m_to_px"][0])
                meta.update(ok=True, wall_px=round(float(np.median([w["thickness"] * a for w in plan["walls"]])), 2),
                            size=list(bg.shape), labels={c: int((mask == i).sum()) for i, c in ((1, "wall"), (2, "door"), (3, "window"))})
        meta["seconds"] = round(time.time() - t0, 1)
        jsave(d / "meta.json", meta)
        log(f"  {f.name}: {'OK' if meta['ok'] else 'skipped - ' + meta['reason']}")
    metas = [jload(p) for p in sorted((data / "pairs").glob("*/meta.json"))]
    ok = [m for m in metas if m.get("ok")]
    for m in ok:   # stable split by plan (never by tile), about 15 % for validation
        m["split"] = "val" if int(hashlib.md5(m["name"].encode()).hexdigest(), 16) % 100 < 15 else "train"
    if ok and not any(m["split"] == "val" for m in ok):
        ok[-1]["split"] = "val"
    if len(ok) > 1 and not any(m["split"] == "train" for m in ok):
        ok[0]["split"] = "train"
    jsave(data / "index.json", {"pairs": metas, "usable": len(ok),
                                "train": sum(m["split"] == "train" for m in ok), "val": sum(m["split"] == "val" for m in ok)})
    print(f"Usable plans: {len(ok)} of {len(metas)} (train {sum(m['split'] == 'train' for m in ok)}, "
          f"val {sum(m['split'] == 'val' for m in ok)}). Check the labels: {data}/pairs/*/preview.jpg")
    if len(ok) < min_sources:
        print(f"Too few usable plans ({len(ok)} < {min_sources}). Add more DWG/DXF/vector PDF files to {sources}.")
        return 2
    return 0


# ---------- fake scans and photos from clean pages ----------
def sample_tile(img, msk, fg, wall_px, norm_px, T, rng, jitter=True):
    s = norm_px / max(wall_px, 1e-3) * (rng.uniform(0.7, 1.4) if jitter else 1.0)
    ang = math.radians(rng.uniform(-3, 3)) if jitter else 0.0
    if len(fg) and rng.random() < 0.85:
        cy, cx = fg[rng.integers(len(fg))] + rng.normal(0, T / (5 * s), 2)
    else:
        cy, cx = rng.uniform(0, img.shape[0]), rng.uniform(0, img.shape[1])
    ca, sa = math.cos(ang) / s, math.sin(ang) / s
    M = np.array([[ca, -sa, cx - ca * T / 2 + sa * T / 2], [sa, ca, cy - sa * T / 2 - ca * T / 2]], np.float32)
    x = cv2.warpAffine(img, M, (T, T), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderValue=255)
    y = cv2.warpAffine(msk, M, (T, T), flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP, borderValue=0)
    if jitter and rng.random() < 0.3:   # slight perspective, like a phone photo after correction
        src = np.float32([[0, 0], [T, 0], [T, T], [0, T]])
        H = cv2.getPerspectiveTransform(src, src + rng.uniform(-0.03, 0.03, (4, 2)).astype(np.float32) * T)
        x = cv2.warpPerspective(x, H, (T, T), flags=cv2.INTER_LINEAR, borderValue=255)
        y = cv2.warpPerspective(y, H, (T, T), flags=cv2.INTER_NEAREST, borderValue=0)
    return x, y


def augment(x, y, rng):
    T = x.shape[0]
    wall = y == 1
    r = rng.random()
    if r < 0.25:     # walls drawn filled
        x[wall] = rng.integers(0, 90)
    elif r < 0.45:   # walls drawn hatched
        yy, xx = np.mgrid[:T, :T]
        per = int(rng.integers(5, 13))
        h = (((xx + yy) if rng.random() < 0.5 else (xx - yy)) % per) < max(1, per // 4)
        x[wall] = 235
        x[wall & h] = rng.integers(0, 80)
        edge = wall & ~cv2.erode(wall.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
        x[edge] = 20
    elif r < 0.6:    # walls drawn as two lines only
        x[cv2.erode(wall.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)] = 240
    x = x.astype(np.float32) * rng.uniform(0.75, 1.1) + rng.uniform(-25, 20)
    if rng.random() < 0.5:   # uneven light
        x *= cv2.resize(rng.uniform(0.55, 1.0, (3, 3)).astype(np.float32), (T, T), interpolation=cv2.INTER_CUBIC)
    if rng.random() < 0.6:
        x = cv2.GaussianBlur(x, (0, 0), rng.uniform(0.3, 1.5))
    if rng.random() < 0.2:   # low-resolution scan
        k = rng.uniform(0.35, 0.7)
        x = cv2.resize(cv2.resize(x, None, fx=k, fy=k, interpolation=cv2.INTER_AREA), (T, T), interpolation=cv2.INTER_LINEAR)
    if rng.random() < 0.6:
        x += rng.normal(0, rng.uniform(2, 10), x.shape).astype(np.float32)
    x = np.clip(x, 0, 255).astype(np.uint8)
    if rng.random() < 0.5:
        x = cv2.imdecode(cv2.imencode(".jpg", x, [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(30, 95))])[1], cv2.IMREAD_GRAYSCALE)
    if rng.random() < 0.08:  # black-and-white scan
        x = cv2.threshold(x, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    return x


def load_pairs(pairs, data):
    out = []
    for m in pairs:
        d = Path(data) / "pairs" / m["name"]
        img, msk = cv2.imread(str(d / "image.png"), cv2.IMREAD_GRAYSCALE), cv2.imread(str(d / "mask.png"), cv2.IMREAD_GRAYSCALE)
        fg = np.argwhere(msk > 0)
        if len(fg) > 20000:
            fg = fg[np.random.default_rng(0).choice(len(fg), 20000, replace=False)]
        out.append((img, msk, fg.astype(np.float32), m["wall_px"]))
    return out


class TileSet:
    """Endless random fake-scan tiles (map-style dataset for torch DataLoader)."""
    def __init__(self, pairs, data, T, n, seed, norm_px):
        self.items, self.T, self.n, self.seed, self.norm_px = load_pairs(pairs, data), T, n, seed, norm_px
        w = np.array([it[0].size for it in self.items], float)
        self.p = w / w.sum()

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        rng = np.random.default_rng([self.seed, i])
        img, msk, fg, wpx = self.items[rng.choice(len(self.items), p=self.p)]
        x, y = sample_tile(img, msk, fg, wpx, self.norm_px, self.T, rng)
        x = augment(x, y, rng)
        return (x.astype(np.float32) / 127.5 - 1.0)[None], y.astype(np.int64)


def val_tiles(pairs, data, T, norm_px, per_plan=24):
    key = hashlib.md5(json.dumps([sorted(m["name"] for m in pairs), T, round(norm_px, 2)]).encode()).hexdigest()[:10]
    f = Path(data) / f"val_{key}.npz"
    if f.exists():
        z = np.load(f)
        return z["x"], z["y"]
    xs, ys = [], []
    for j, (img, msk, fg, wpx) in enumerate(load_pairs(pairs, data)):
        for i in range(per_plan):
            rng = np.random.default_rng([4242, j, i])
            x, y = sample_tile(img, msk, fg, wpx, norm_px, T, rng)
            xs.append(augment(x, y, rng))
            ys.append(y)
    np.savez_compressed(f, x=np.stack(xs), y=np.stack(ys))
    return np.stack(xs), np.stack(ys)


# ---------- model ----------
def build_unet(base=32, n_cls=4):
    import torch
    import torch.nn as nn

    def block(ci, co):
        return nn.Sequential(nn.Conv2d(ci, co, 3, padding=1, bias=False), nn.BatchNorm2d(co), nn.ReLU(inplace=True),
                             nn.Conv2d(co, co, 3, padding=1, bias=False), nn.BatchNorm2d(co), nn.ReLU(inplace=True))

    class UNet(nn.Module):
        def __init__(self):
            super().__init__()
            c = [base * 2 ** i for i in range(5)]
            self.enc = nn.ModuleList([block(1, c[0])] + [block(c[i], c[i + 1]) for i in range(4)])
            self.up = nn.ModuleList([nn.ConvTranspose2d(c[i + 1], c[i], 2, stride=2) for i in reversed(range(4))])
            self.dec = nn.ModuleList([block(2 * c[i], c[i]) for i in reversed(range(4))])
            self.head = nn.Conv2d(c[0], n_cls, 1)

        def forward(self, x):
            skips = []
            for i, e in enumerate(self.enc):
                x = e(x if i == 0 else nn.functional.max_pool2d(x, 2))
                skips.append(x)
            x = skips.pop()
            for up, dec in zip(self.up, self.dec):
                x = dec(torch.cat([up(x), skips.pop()], 1))
            return self.head(x)

    return UNet()


def seg_loss(logits, y, w):
    import torch.nn.functional as F
    ce = F.cross_entropy(logits, y, weight=w)
    p = logits.softmax(1)
    oh = F.one_hot(y, logits.shape[1]).permute(0, 3, 1, 2).float()
    inter, den = (p * oh).sum((0, 2, 3)), p.sum((0, 2, 3)) + oh.sum((0, 2, 3))
    return ce + 1 - ((2 * inter + 1) / (den + 1))[1:].mean()


def predict_full(model, img, T, dev):
    import torch
    H, W = img.shape
    if H < T or W < T:
        img = cv2.copyMakeBorder(img, 0, max(0, T - H), 0, max(0, T - W), cv2.BORDER_CONSTANT, value=255)
    Hp, Wp = img.shape
    st = T - T // 8
    ys = sorted(set(list(range(0, Hp - T + 1, st)) + [Hp - T]))
    xs = sorted(set(list(range(0, Wp - T + 1, st)) + [Wp - T]))
    acc, cnt = np.zeros((4, Hp, Wp), np.float32), np.zeros((Hp, Wp), np.float32)
    x = img.astype(np.float32) / 127.5 - 1.0
    coords = [(yy, xx) for yy in ys for xx in xs]
    model.eval()
    with torch.inference_mode():
        for i in range(0, len(coords), 8):
            chunk = coords[i:i + 8]
            b = torch.from_numpy(np.stack([x[yy:yy + T, xx:xx + T] for yy, xx in chunk])[:, None]).to(dev)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=dev == "cuda"):
                p = model(b).float().softmax(1).cpu().numpy()
            for (yy, xx), pp in zip(chunk, p):
                acc[:, yy:yy + T, xx:xx + T] += pp
                cnt[yy:yy + T, xx:xx + T] += 1
    return (acc / np.maximum(cnt, 1))[:, :H, :W]


def validate(model, vx, vy, dev):
    import torch
    conf = np.zeros((4, 4), np.int64)
    model.eval()
    with torch.inference_mode():
        for i in range(0, len(vx), 16):
            b = torch.from_numpy(vx[i:i + 16].astype(np.float32) / 127.5 - 1.0)[:, None].to(dev)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=dev == "cuda"):
                pred = model(b).argmax(1).cpu().numpy()
            conf += np.bincount(4 * vy[i:i + 16].astype(np.int64).ravel() + pred.ravel(), minlength=16).reshape(4, 4)
    inter = np.diag(conf).astype(float)
    union = conf.sum(0) + conf.sum(1) - inter
    iou = {c: round(float(inter[i] / union[i]) if union[i] else 0.0, 4) for i, c in ((1, "wall"), (2, "door"), (3, "window"))}
    iou["mean"] = round(float(np.mean([iou["wall"], iou["door"], iou["window"]])), 4)
    return iou


def samples(model, vx, vy, dev, out, step, real_dir):
    import torch
    idx = sorted({0, len(vx) // 4, len(vx) // 2, 3 * len(vx) // 4})
    with torch.inference_mode():
        b = torch.from_numpy(vx[idx].astype(np.float32) / 127.5 - 1.0)[:, None].to(dev)
        pred = model(b).argmax(1).cpu().numpy().astype(np.uint8)
    rows = [np.hstack([cv2.cvtColor(vx[i], cv2.COLOR_GRAY2BGR), colorize(vx[i], vy[i]), colorize(vx[i], p)]) for i, p in zip(idx, pred)]
    cv2.imwrite(str(out / f"step_{step:06d}.jpg"), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85])
    for f in sorted(Path(real_dir).glob("*"))[:2] if real_dir and Path(real_dir).exists() else []:
        if f.suffix.lower() not in (".png", ".jpg", ".jpeg"):
            continue
        g = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
        k = min(1.0, 2048 / max(g.shape))
        g = cv2.resize(g, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
        lab = predict_full(model, g, vx.shape[1], dev).argmax(0).astype(np.uint8)
        cv2.imwrite(str(out / f"step_{step:06d}_real_{f.stem}.jpg"), colorize(g, lab), [cv2.IMWRITE_JPEG_QUALITY, 85])


def train(a):
    import torch
    from torch.utils.data import DataLoader
    data, run = Path(a.data), Path(a.run)
    pairs = [m for m in (jload(data / "index.json") or {}).get("pairs", []) if m.get("ok")]
    tr, va = [m for m in pairs if m.get("split") == "train"], [m for m in pairs if m.get("split") == "val"]
    if not tr or not va:
        print("Need at least one training and one validation plan: run the prepare step first.")
        return 2
    if a.fresh and run.exists():
        run.rename(run.with_name(run.name + time.strftime("_old_%Y%m%d_%H%M%S")))
    (run / "samples").mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(a.seed)
    norm_px = float(np.median([m["wall_px"] for m in tr]))
    model = build_unet(a.base_ch).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    warm = min(500, a.steps // 10 + 1)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(s, a.steps) / a.steps)))
    step, best = 0, -1.0
    if (run / "last.pt").exists():
        ck = torch.load(run / "last.pt", map_location=dev, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        step, best = ck["step"], ck["best"]
        log(f"train: resuming {run.name} from step {step} (best wall IoU {best:.3f})")
    meta = {"base": a.base_ch, "tile": a.tile, "norm_px": norm_px, "classes": ["background", "wall", "door", "window"]}
    jsave(run / "config.json", dict(vars(a), **meta, train=[m["name"] for m in tr], val=[m["name"] for m in va], device=dev))
    if step >= a.steps:
        log(f"train: {run.name} already finished ({step} steps)")
        return 0
    vx, vy = val_tiles(va, data, a.tile, norm_px)
    ds = TileSet(tr, data, a.tile, (a.steps - step) * a.batch, a.seed + 7919 * step, norm_px)
    dl = DataLoader(ds, batch_size=a.batch, num_workers=a.workers, pin_memory=dev == "cuda", drop_last=True,
                    persistent_workers=a.workers > 0, prefetch_factor=4 if a.workers > 0 else None)
    wts = torch.tensor([0.5, 1.0, 3.0, 3.0], device=dev)
    log(f"train: {len(tr)} train plans, {len(va)} val plans ({len(vx)} val tiles), device {dev}, steps {step}->{a.steps}")
    t0, run_loss, n_loss = time.time(), 0.0, 0
    model.train()
    with open(run / "log.csv", "a", newline="") as fcsv:
        wr = csv.writer(fcsv)
        for x, y in dl:
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=dev == "cuda"):
                out = model(x)
            loss = seg_loss(out.float(), y, wts)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            run_loss, n_loss = run_loss + loss.item(), n_loss + 1
            if step % 50 == 0 or step == a.steps:
                log(f"step {step}/{a.steps} loss {run_loss / n_loss:.4f} lr {sched.get_last_lr()[0]:.2e} "
                    f"{n_loss / (time.time() - t0):.1f} steps/s")
                wr.writerow([step, round(run_loss / n_loss, 5), "", "", "", ""])
                t0, run_loss, n_loss = time.time(), 0.0, 0
            if step % a.val_every == 0 or step == a.steps:
                iou = validate(model, vx, vy, dev)
                samples(model, vx, vy, dev, run / "samples", step, a.real)
                wr.writerow([step, "", iou["wall"], iou["door"], iou["window"], iou["mean"]])
                fcsv.flush()
                if iou["wall"] > best:
                    best = iou["wall"]
                    torch.save({"model": model.state_dict(), "meta": dict(meta, step=step, val=iou)}, run / "best.pt")
                log(f"validation step {step}: IoU wall {iou['wall']:.3f} door {iou['door']:.3f} window {iou['window']:.3f}"
                    f" (best wall {best:.3f}) -> samples/step_{step:06d}.jpg")
                model.train()
            if step % a.ckpt_every == 0 or step == a.steps:
                torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                            "step": step, "best": best}, run / "last.pt.tmp")
                os.replace(run / "last.pt.tmp", run / "last.pt")
            if step >= a.steps:
                break
    log(f"train: finished {step} steps, best wall IoU {best:.3f}")
    return 0


def install(run, min_iou):
    import torch
    run = Path(run)
    if not (run / "best.pt").exists():
        print(f"No {run}/best.pt yet - train first.")
        return 2
    meta = torch.load(run / "best.pt", map_location="cpu", weights_only=False)["meta"]
    v = meta["val"]
    print(f"Best checkpoint: step {meta['step']}, validation IoU walls {v['wall']:.3f}, doors {v['door']:.3f}, windows {v['window']:.3f}")
    if v["wall"] < min_iou:
        print(f"NOT switched on: wall IoU {v['wall']:.3f} is below {min_iou}. Add more source plans or train longer.")
        return 3
    dst = WS / "models/wallseg"
    dst.mkdir(parents=True, exist_ok=True)
    shutil.copy2(run / "best.pt", dst / f"{run.name}.pt")
    shutil.copy2(run / "best.pt", dst / "model.pt")
    p = WS / "config.json"
    cfg = json.loads(p.read_text()) if p.exists() else {}
    cfg["wallseg"] = {"enabled": True, "model": str(dst / "model.pt"), "run": run.name}
    p.write_text(json.dumps(cfg, indent=1))
    print(f"Switched on: {dst / 'model.pt'} (config.json wallseg.enabled = true)")
    return 0


# ---------- use in the pipeline ----------
def mask_thickness(mask):
    b = (mask > 0).astype(np.uint8)
    if b.sum() < 500:
        return None
    dt = cv2.distanceTransform(b, cv2.DIST_L2, 5)
    ridge = (dt >= 2) & (dt >= cv2.dilate(dt, np.ones((3, 3), np.uint8)) - 1e-6)
    return float(2 * np.median(dt[ridge])) if ridge.sum() > 50 else None


def predict(job):
    import torch
    job = Path(job)
    mp = Path(os.environ.get("PIPE_WALLSEG_MODEL") or load_config()["wallseg"]["model"])
    if not mp.exists():
        raise SystemExit(f"wall model not found: {mp}")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(mp, map_location="cpu", weights_only=False)
    meta = ck["meta"]
    model = build_unet(meta["base"])
    model.load_state_dict(ck["model"])
    model.to(dev)
    page = cv2.imread(str(job / "01_parse/page.png"), cv2.IMREAD_GRAYSCALE)
    cur, backup = job / "01_parse/wall_mask.png", job / "01_parse/wall_mask_classic.png"
    if not backup.exists():
        shutil.copy2(cur, backup)
    classic = cv2.imread(str(backup), cv2.IMREAD_GRAYSCALE)
    t = mask_thickness(classic)
    s = float(np.clip(meta["norm_px"] / t, 0.5, 2.0)) if t else 1.0
    img = page if abs(s - 1) < 0.02 else cv2.resize(page, None, fx=s, fy=s, interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
    lab = predict_full(model, img, meta["tile"], dev).argmax(0).astype(np.uint8)
    if lab.shape != page.shape:
        lab = cv2.resize(lab, (page.shape[1], page.shape[0]), interpolation=cv2.INTER_NEAREST)
    walls = cv2.morphologyEx(((lab == 1) * 255).astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, cc, st, _ = cv2.connectedComponentsWithStats(walls, 8)
    tpx = t or meta["norm_px"] / s
    keep = np.zeros(n, bool)
    keep[1:] = st[1:, cv2.CC_STAT_AREA] >= 4 * tpx * tpx
    walls = (keep[cc] * 255).astype(np.uint8)
    ratio = float(walls.sum()) / max(float(classic.sum()), 1.0)
    info = {"model": str(mp), "scale": round(s, 3), "wall_px_classic": t, "ratio_to_classic": round(ratio, 3)}
    if 0.33 <= ratio <= 3.0:
        cv2.imwrite(str(cur), walls)
        info["used"] = True
    else:
        shutil.copy2(backup, cur)
        info.update(used=False, reason="model mask very different from the classic mask - classic mask kept")
    k = min(1.0, 2400 / max(page.shape))
    cv2.imwrite(str(job / "01_parse/wallseg_pred.jpg"), cv2.resize(colorize(page, lab), None, fx=k, fy=k, interpolation=cv2.INTER_AREA))
    jsave(job / "01_parse/wallseg.json", info)
    log(f"wallseg: {'used' if info['used'] else 'NOT used'} (scale {s:.2f}, wall pixels x{ratio:.2f} vs classic)")
    return 0


def compare(run, inputs, tests):
    from .evaluate import match
    run, out = Path(run), WS / "finetune/compare" / Path(run).name
    truth = {}
    tf = Path(tests) / "truth.csv"
    if tf.exists():
        for row in csv.DictReader(open(tf, encoding="utf-8-sig")):
            if row.get("plan") and row.get("width_m"):
                truth.setdefault(row["plan"].strip(), []).append(row)
    files = list({f.resolve(): f for d in (Path(inputs), WS / "inputs/smoke_test") if d.exists()
                  for f in sorted(d.iterdir()) if f.name in truth}.values())
    if not files:
        print(f"No test plans with truth rows found ({tf}). Skipping before/after.")
        return 0
    lines = ["| Plan | Kind | Base: rooms OK | Base: worst | Fine-tuned: rooms OK | Fine-tuned: worst | Verdict |",
             "|---|---|---|---|---|---|---|"]
    for f in files:
        res = {}
        for variant, flag in (("base", "off"), ("finetuned", "on")):
            job = out / f.stem / variant
            vt = out / f.stem / "base/01_parse/vlm_texts.json"
            if variant == "finetuned" and vt.exists():   # same text reading for both, so only the walls differ
                (job / "01_parse").mkdir(parents=True, exist_ok=True)
                shutil.copy2(vt, job / "01_parse/vlm_texts.json")
            env = dict(os.environ, PIPE_WALLSEG_MODEL=str(run / "best.pt"))
            r = subprocess.run([sys.executable, "-m", "app.main", str(f), "--job-dir", str(job), "--to-stage", "plan",
                                "--wallseg", flag], cwd=str(WS), env=env, capture_output=True, text=True)
            plan = jload(job / "02_plan/plan.json")
            if r.returncode or not plan:
                res[variant] = None
                continue
            kind = plan["source"]["kind"]
            m = match(truth[f.name], plan["rooms"], kind)
            worst = max((x[3] for x in m if x[3]), key=lambda e: e[0], default=None)
            res[variant] = (sum(1 for x in m if x[3] and x[3][1]), len(m), worst, kind)
            if kind not in RASTER:
                break   # the model is only used for scans and photos
        b, ft = res.get("base"), res.get("finetuned")
        cell = lambda v: ("failed", "-") if v is None else (f"{v[0]}/{v[1]}", v[2][2] if v[2] else "-")
        if b and b[3] not in RASTER:
            lines.append(f"| {f.name} | {b[3]} | {cell(b)[0]} | {cell(b)[1]} | (not used for vector input) | | same |")
            continue
        verdict = "better" if ft and (not b or ft[0] > b[0] or (ft[0] == b[0] and ft[2] and b[2] and ft[2][0] < b[2][0] - 0.1)) else \
                  "worse" if b and (not ft or ft[0] < b[0] or (ft[2] and b[2] and ft[2][0] > b[2][0] + 0.1)) else "same"
        lines.append(f"| {f.name} | {(b or ft or (0, 0, 0, '?'))[3]} | {cell(b)[0]} | {cell(b)[1]} | {cell(ft)[0]} | {cell(ft)[1]} | **{verdict}** |")
    out.mkdir(parents=True, exist_ok=True)
    (out / "compare.md").write_text(f"# Before / after - {run.name}\n\n" + "\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines) + f"\n\nWritten: {out / 'compare.md'}")
    return 0


def main():
    ap = argparse.ArgumentParser(description="wall/door/window model for scans and photos")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--sources", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--min-sources", type=int, default=20)
    p = sub.add_parser("train")
    p.add_argument("--data", required=True)
    p.add_argument("--run", required=True)
    for k, v in (("steps", 20000), ("batch", 16), ("tile", 512), ("base-ch", 32), ("val-every", 1000), ("ckpt-every", 500),
                 ("workers", 8), ("seed", 1234), ("fresh", 0)):
        p.add_argument(f"--{k}", type=int, default=v)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--real", default="")
    p = sub.add_parser("install")
    p.add_argument("--run", required=True)
    p.add_argument("--min-iou", type=float, default=0.8)
    p = sub.add_parser("predict")
    p.add_argument("job")
    p = sub.add_parser("compare")
    p.add_argument("--run", required=True)
    p.add_argument("--inputs", default=str(WS / "inputs/test"))
    p.add_argument("--tests", default=str(WS / "tests"))
    a = ap.parse_args()
    if a.cmd == "train" and a.tile % 16:
        raise SystemExit("--tile must be a multiple of 16")
    rc = {"prepare": lambda: prepare(a.sources, a.data, a.min_sources), "train": lambda: train(a),
          "install": lambda: install(a.run, a.min_iou), "predict": lambda: predict(a.job),
          "compare": lambda: compare(a.run, a.inputs, a.tests)}[a.cmd]()
    sys.exit(rc or 0)


if __name__ == "__main__":
    main()
__END_OF_WALLSEG_PY__
  cat > "$APP/patches.py" <<'__END_OF_PATCHES_PY__'
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
__END_OF_PATCHES_PY__
  cat > "$APP/furniture.py" <<'__END_OF_FURNITURE_PY__'
"""Furniture library: real 3D models instead of parametric boxes where a good one exists.
Sources, in order: Poly Haven CC0 models (downloaded once at setup), your own files in
assets/models_user/<type>/*.glb|*.gltf (licence per file, see below), and - only with GEN3D=1 - generated models.
  python -m app.furniture fetch      download Poly Haven furniture models (setup step, needs internet)
  python -m app.furniture catalog    rebuild catalog.json + LICENSES.txt from the files on disk (offline, ~1 s)
  python -m app.furniture report     which furniture types have models, from which source
Licence per user file: a sidecar <file>.json next to it, or licence.json in its folder, e.g.
  {"licence": "CC0-1.0", "source_url": "https://kenney.nl/assets/furniture-kit", "author": "Kenney",
   "front_axis": "-Y", "style_tags": ["modern"]}
Allowed licences come from config furniture.allowed_licences; files without an allowed licence are skipped (logged).
Pure Python: sizes and triangle counts are read from the glTF files themselves (no Blender needed)."""
import hashlib, json, math, re, statistics, struct, sys, time
from pathlib import Path
from .common import WS, jload, jsave, load_config, log

UA = {"User-Agent": "floorplan-poc-setup/1.0 (Runpod proof of concept; one-time download)"}
# layout type -> how to recognise it. cat = Poly Haven taxonomy paths (the asset's `category` field), kw = words in
# name/tags/legacy categories, no = words that exclude. Order matters: the first matching type wins.
TYPES = {
    "armchair": {"kw": ["armchair", "arm chair", "lounge chair", "club chair"], "no": ["sofa"]},
    "sofa": {"cat": ["Furniture/Seating/Sofas & Couches"], "kw": ["sofa", "couch"],
             "no": ["armchair", "arm chair", "ottoman", "footstool", "cushion", "pillow"]},
    "coffee_table": {"cat": ["Furniture/Tables/Coffee Tables"], "kw": ["coffee table"]},
    "dining_table": {"cat": ["Furniture/Tables/Dining Tables"], "kw": ["dining table"]},
    "desk": {"cat": ["Furniture/Tables/Desks"], "kw": ["desk"], "no": ["lamp", "organizer", "organiser"]},
    "nightstand": {"kw": ["nightstand", "night stand", "bedside"], "no": ["lamp"]},
    "bistro_table": {"cat": ["Furniture/Tables/Side & End Tables", "Furniture/Tables/Outdoor Tables"],
                     "kw": ["bistro", "cafe table", "side table", "round table"], "no": ["lamp"]},
    "wardrobe": {"kw": ["wardrobe", "armoire", "closet"]},
    "bookshelf": {"cat": ["Furniture/Storage Furniture/Shelving & Bookcases"], "kw": ["bookshelf", "bookcase"],
                  "no": ["wall shelf", "book "]},
    "shoe_cabinet": {"cat": ["Furniture/Storage Furniture/Cabinets & Cupboards", "Furniture/Storage Furniture/Sideboards",
                             "Furniture/Storage Furniture/Drawers & Dressers"],
                     "kw": ["shoe cabinet", "sideboard", "dresser", "chest of drawers", "cabinet"],
                     "no": ["wall cabinet", "medicine", "filing", "wardrobe", "nightstand", "bedside"]},
    "tv_unit": {"kw": ["tv stand", "tv unit", "tv cabinet", "media console"]},
    "bed": {"cat": ["Furniture/Beds"], "kw": [" bed", "bed "], "no": ["bedside", "flower bed", "bed frame only", "bunk"]},
    "floor_lamp": {"cat": ["Lighting/Floor & Desk/Floor Lamps"], "kw": ["floor lamp", "standing lamp", "floor_lamp"]},
    "plant": {"cat": ["Nature/Plants/Potted Plants"], "kw": ["potted plant", "houseplant", "potted", "plant"],
              "no": ["planter box", "leaf", "branch", "tree trunk"]},
    "chair": {"cat": ["Furniture/Seating/Chairs"], "kw": ["chair"],
              "no": ["armchair", "arm chair", "lounge", "office", "high chair", "rocking", "deck", "beach", "wheelchair"]},
    "fridge": {"kw": ["fridge", "refrigerator"]}, "washer": {"kw": ["washing machine", "washer"]},
    "toilet": {"kw": ["toilet"]}, "bathtub": {"kw": ["bathtub", "bath tub"]}, "vanity": {"kw": ["vanity"]},
    "basin_small": {"kw": ["wash basin", "washbasin", "hand basin"]}, "shower": {"kw": ["shower"]},
}
ALIASES = {"bed_double": "bed", "bed_single": "bed"}      # layout type -> catalog type
HEIGHT_FIT = ("plant", "floor_lamp")                      # organic shapes: uniform scale to the slot height
PARAMETRIC_ONLY = ("rug", "kitchen_run")                  # sizes change per plan: always built in Blender
LICENCE_NAMES = {"cc0": "CC0-1.0", "cc0-1.0": "CC0-1.0", "cc0 1.0": "CC0-1.0", "public domain": "CC0-1.0",
                 "cc-by-4.0": "CC-BY-4.0", "cc by 4.0": "CC-BY-4.0", "cc-by 4.0": "CC-BY-4.0",
                 "cc-by-3.0": "CC-BY-3.0", "cc by 3.0": "CC-BY-3.0", "mit": "MIT", "apache-2.0": "Apache-2.0",
                 "apache 2.0": "Apache-2.0", "owned": "owned", "own": "owned", "mine": "owned"}
ATTRIBUTION = ("CC-BY-4.0", "CC-BY-3.0")


def lib_dir(cfg):
    return Path(cfg["furniture"]["library"])


def catalog_type(layout_type):
    return ALIASES.get(layout_type, layout_type)


def norm_licence(s):
    s = str(s or "").strip()
    return LICENCE_NAMES.get(s.lower(), s)


def words(s):
    return set(re.findall(r"[a-z]{3,}", str(s).lower()))


# ---------- glTF inspection (sizes, triangles, missing files) without Blender ----------
def read_gltf(path):
    b = Path(path).read_bytes()
    if b[:4] != b"glTF":
        return json.loads(b.decode("utf-8")), False
    _, length = struct.unpack_from("<II", b, 4)
    off, js = 12, None
    while off + 8 <= min(len(b), length):
        clen, ctype = struct.unpack_from("<II", b, off)
        if ctype == 0x4E4F534A:            # 'JSON'
            js = json.loads(b[off + 8:off + 8 + clen].decode("utf-8"))
        off += 8 + clen + (-clen % 4)
    if js is None:
        raise ValueError("GLB without JSON chunk")
    return js, True


def _mat_mul(a, b):
    return [[sum(a[i][k] * b[k][j] for k in range(4)) for j in range(4)] for i in range(4)]


def _node_matrix(n):
    if "matrix" in n:
        m = n["matrix"]                    # column-major
        return [[m[c * 4 + r] for c in range(4)] for r in range(4)]
    tx, ty, tz = n.get("translation", [0, 0, 0])
    x, y, z, w = n.get("rotation", [0, 0, 0, 1])
    sx, sy, sz = n.get("scale", [1, 1, 1])
    R = [[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
         [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
         [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]]
    return [[R[0][0] * sx, R[0][1] * sy, R[0][2] * sz, tx], [R[1][0] * sx, R[1][1] * sy, R[1][2] * sz, ty],
            [R[2][0] * sx, R[2][1] * sy, R[2][2] * sz, tz], [0, 0, 0, 1]]


def inspect_gltf(path):
    """{dims_gltf, dims_import [x, y, z] in Blender axes after import (Z up), triangles, missing_files}."""
    path = Path(path)
    js, is_glb = read_gltf(path)
    acc = js.get("accessors", [])
    lo, hi, tris = [math.inf] * 3, [-math.inf] * 3, 0
    I = [[1.0 if i == j else 0.0 for j in range(4)] for i in range(4)]
    scene = js.get("scenes", [{}])[js.get("scene", 0)] if js.get("scenes") else {"nodes": list(range(len(js.get("nodes", []))))}
    stack = [(n, I) for n in scene.get("nodes", [])]
    seen = 0
    while stack:
        ni, parent = stack.pop()
        seen += 1
        if seen > 100000:
            raise ValueError("node graph too large or cyclic")
        node = js["nodes"][ni]
        M = _mat_mul(parent, _node_matrix(node))
        if "mesh" in node:
            for prim in js["meshes"][node["mesh"]].get("primitives", []):
                pa = acc[prim["attributes"]["POSITION"]]
                mode = prim.get("mode", 4)
                n = acc[prim["indices"]]["count"] if "indices" in prim else pa["count"]
                tris += n // 3 if mode == 4 else max(0, n - 2) if mode in (5, 6) else 0
                if "min" not in pa or "max" not in pa:
                    raise ValueError("POSITION accessor without min/max")
                for cx in (pa["min"][0], pa["max"][0]):
                    for cy in (pa["min"][1], pa["max"][1]):
                        for cz in (pa["min"][2], pa["max"][2]):
                            p = [M[r][0] * cx + M[r][1] * cy + M[r][2] * cz + M[r][3] for r in range(3)]
                            lo, hi = [min(a, b) for a, b in zip(lo, p)], [max(a, b) for a, b in zip(hi, p)]
        stack += [(c, M) for c in node.get("children", [])]
    if lo[0] == math.inf:
        raise ValueError("no mesh geometry")
    d = [h - l for h, l in zip(hi, lo)]
    missing = []
    for item in js.get("buffers", []) + js.get("images", []):
        uri = item.get("uri")
        if uri and not uri.startswith("data:"):
            from urllib.parse import unquote
            if not (path.parent / unquote(uri)).exists():
                missing.append(uri)
    return {"dims_gltf": [round(x, 4) for x in d], "dims_import": [round(d[0], 4), round(d[2], 4), round(d[1], 4)],
            "triangles": int(tris), "missing_files": missing, "glb": is_glb}


def layout_dims(dims_import, front_axis):
    """(width along the wall, depth towards the room, height) once the model's front is turned to layout +Y."""
    x, y, z = dims_import
    return [x, y, z] if front_axis in ("+Y", "-Y") else [y, x, z]


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------- Poly Haven (CC0) ----------
def classify(slug, info):
    """First layout type whose rule matches this Poly Haven model, else None."""
    cat = info.get("category") or ""
    text = " " + " ".join([slug.replace("_", " "), info.get("name", ""), cat, " ".join(info.get("tags", [])),
                           " ".join(info.get("categories", []))]).lower() + " "
    for typ, rule in TYPES.items():
        if any(x in text for x in rule.get("no", [])):
            continue
        if any(cat.startswith(c) for c in rule.get("cat", [])) or any(k in text for k in rule.get("kw", [])):
            return typ
    return None


def fetch_polyhaven(cfg=None):
    """Download the most popular Poly Haven model(s) per furniture type (glTF + textures, md5 checked)."""
    import urllib.request
    cfg = cfg or load_config()
    fc = cfg["furniture"]
    dst = lib_dir(cfg) / "polyhaven"
    dst.mkdir(parents=True, exist_ok=True)
    get = lambda url, t=300: urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=t).read()
    ph = lambda p: json.loads(get("https://api.polyhaven.com" + p, 120))
    models = ph("/assets?type=models")
    by_type = {}
    for slug, info in models.items():
        typ = classify(slug, info)
        if typ:
            by_type.setdefault(typ, []).append((info.get("download_count", 0), slug))
    res_pref = [fc["polyhaven_resolution"], "1k", "2k"]
    for typ in TYPES:
        cands = sorted(by_type.get(typ, []), reverse=True)[:fc["polyhaven_per_type"]]
        if not cands:
            log(f"Poly Haven: no model for '{typ}'")
            continue
        for _, slug in cands:
            d = dst / slug
            info = models[slug]
            old = jload(d / "meta.json")
            if old and old.get("files_hash") == info.get("files_hash"):
                continue
            try:
                files = ph(f"/files/{slug}")["gltf"]
                res = next(r for r in res_pref if r in files)
                g = files[res]["gltf"]
                d.mkdir(parents=True, exist_ok=True)
                items = [(f"{slug}_{res}.gltf", g)] + list((g.get("include") or {}).items())
                for rel, f in items:
                    data = get(f["url"])
                    if f.get("md5") and hashlib.md5(data).hexdigest() != f["md5"]:
                        raise ValueError(f"md5 mismatch for {rel}")
                    q = d / rel
                    q.parent.mkdir(parents=True, exist_ok=True)
                    q.write_bytes(data)
                attrs = info.get("attributes") or {}
                mat = attrs.get("material") or []
                jsave(d / "meta.json", {"slug": slug, "type": typ, "name": info.get("name", slug), "file": f"{slug}_{res}.gltf",
                                        "category": info.get("category"), "tags": info.get("tags", []),
                                        "material": mat if isinstance(mat, list) else [mat],
                                        "authors": list((info.get("authors") or {}).keys()), "licence": "CC0-1.0",
                                        "source_url": f"https://polyhaven.com/a/{slug}", "files_hash": info.get("files_hash"),
                                        "downloaded": time.strftime("%Y-%m-%d")})
                log(f"Poly Haven {typ}: {slug} ({res})")
            except Exception as e:
                log(f"WARNING Poly Haven {slug}: {e}")
    return build_catalog(cfg)


# ---------- catalog ----------
def _entry(cfg, f, typ, source, meta, excluded):
    """Inspect one model file and turn it into a catalog entry (or an exclusion with the reason)."""
    lic = norm_licence(meta.get("licence"))
    rec = {"file": str(f), "type": typ, "source": source}
    if source == "generated":
        if not cfg["furniture"]["allow_generated"]:
            excluded.append(dict(rec, reason="generated models turned off (furniture.allow_generated)"))
            return None
    elif lic not in cfg["furniture"]["allowed_licences"]:
        excluded.append(dict(rec, reason=f"licence {lic or 'missing'} not in furniture.allowed_licences"))
        return None
    try:
        info = inspect_gltf(f)
    except Exception as e:
        excluded.append(dict(rec, reason=f"unreadable glTF: {e}"))
        return None
    if info["missing_files"]:
        excluded.append(dict(rec, reason=f"missing files: {info['missing_files'][:3]}"))
        return None
    dims, unit = info["dims_import"], float(meta.get("unit_scale", 1.0))
    if "unit_scale" not in meta and max(dims) > 30 and 0.05 < max(dims) * 0.01 <= 30:
        unit = 0.01                        # exported in centimetres
    dims = [round(x * unit, 4) for x in dims]
    if max(dims) < 0.05 or max(dims) > 30 or min(dims) <= 0:
        excluded.append(dict(rec, reason=f"implausible size {dims} m"))
        return None
    front = meta.get("front_axis", "-Y")   # glTF front is +Z -> Blender -Y after import (glTF 2.0 convention)
    if front not in ("+X", "-X", "+Y", "-Y"):
        front = "-Y"
    tags = sorted(words(" ".join([str(meta.get("name", "")), " ".join(meta.get("tags", [])),
                                  " ".join(meta.get("material", [])), " ".join(meta.get("style_tags", [])),
                                  str(meta.get("category") or "")])))
    return {"id": f"{source}:{meta.get('slug') or Path(f).stem}", "type": typ, "source": source, "file": str(f),
            "name": meta.get("name") or Path(f).stem, "dims_m": [round(x, 3) for x in layout_dims(dims, front)],
            "import_dims_m": dims, "unit_scale": unit, "front_axis": front,
            "fit": "height" if typ in HEIGHT_FIT else "box", "style_tags": tags, "licence": lic,
            "source_url": meta.get("source_url", ""), "author": ", ".join(meta.get("authors", [])) or meta.get("author", ""),
            "polycount": info["triangles"], "sha256": sha256(f), "notes": meta.get("notes", "")}


def _user_meta(f, type_dir):
    for cand in (f.with_suffix(".json"), f.parent / "licence.json", f.parent / "license.json",
                 type_dir / "licence.json", type_dir / "license.json"):
        if cand.exists() and cand != f:
            m = jload(cand) or {}
            if "licence" not in m and "license" in m:
                m["licence"] = m["license"]
            return m
    return {}


def build_catalog(cfg=None):
    cfg = cfg or load_config()
    fc, lib = cfg["furniture"], lib_dir(cfg)
    lib.mkdir(parents=True, exist_ok=True)
    items, excluded = [], []
    for meta_f in sorted((lib / "polyhaven").glob("*/meta.json")):
        m = jload(meta_f)
        e = _entry(cfg, meta_f.parent / m["file"], m["type"], "polyhaven", m, excluded)
        if e:
            items.append(e)
    legacy = Path(fc["legacy_models"])      # plants/lamps downloaded by assets_fetch.py (Poly Haven, CC0)
    for typ in ("plant", "floor_lamp"):
        for f in sorted((legacy / typ).glob("*/*.gltf")) if (legacy / typ).exists() else []:
            slug = f.parent.name
            if any(i["id"] == f"polyhaven:{slug}" for i in items):
                continue
            m = {"slug": slug, "name": slug.replace("_", " "), "licence": "CC0-1.0", "tags": [typ],
                 "source_url": f"https://polyhaven.com/a/{slug}"}
            e = _entry(cfg, f, typ, "polyhaven", m, excluded)
            if e:
                items.append(e)
    user = Path(fc["user_dir"])
    for type_dir in sorted(p for p in user.iterdir() if p.is_dir()) if user.exists() else []:
        typ = catalog_type(type_dir.name)
        if typ not in TYPES:
            excluded.append({"file": str(type_dir), "type": typ, "source": "user",
                             "reason": f"unknown furniture type folder (use one of: {', '.join(sorted(TYPES))})"})
            continue
        for f in sorted(p for p in type_dir.rglob("*") if p.suffix.lower() in (".glb", ".gltf")):
            m = _user_meta(f, type_dir)
            m.setdefault("slug", f"{typ}/{f.relative_to(type_dir).with_suffix('')}".replace("/", "_"))
            e = _entry(cfg, f, typ, "user", m, excluded)
            if e:
                items.append(e)
    for meta_f in sorted((lib / "generated").glob("*/*.json")):
        m = jload(meta_f)
        f = meta_f.with_suffix(".glb")
        if f.exists():
            e = _entry(cfg, f, m.get("type", meta_f.parent.name), "generated", m, excluded)
            if e:
                items.append(e)
    cov = {t: {s: sum(1 for i in items if i["type"] == t and i["source"] == s) for s in ("polyhaven", "user", "generated")}
           for t in TYPES}
    cat = {"version": 1, "built": time.strftime("%Y-%m-%d %H:%M:%S"), "items": items, "excluded": excluded, "coverage": cov}
    jsave(lib / "catalog.json", cat)
    write_licenses(lib, items, excluded)
    return cat


def write_licenses(lib, items, excluded):
    L = ["Furniture model licences (written by app/furniture.py - regenerate with: python -m app.furniture catalog)", "",
         "Poly Haven models are CC0 (https://polyhaven.com/license). They were downloaded once through the Poly Haven",
         "API (terms: https://github.com/Poly-Haven/Public-API/blob/master/ToS.md). Credit: Powered by Poly Haven.", "",
         "id | type | licence | source | author | file | sha256"]
    L += [f"{i['id']} | {i['type']} | {i['licence']} | {i['source_url'] or '-'} | {i['author'] or '-'} | {i['file']} | {i['sha256']}"
          for i in items]
    att = [i for i in items if i["licence"] in ATTRIBUTION]
    if att:
        L += ["", "Attribution required (CC-BY):"] + [f"  {i['name']} by {i['author'] or 'unknown'} - {i['source_url']} ({i['licence']})" for i in att]
    gen = [i for i in items if i["source"] == "generated"]
    if gen:
        L += ["", "Generated models (GEN3D): see each model's notes. TRELLIS.2 code/weights are MIT, but its GLB export",
              "uses NVIDIA nvdiffrast (NVIDIA Source Code License: non-commercial / research and evaluation only).",
              "Treat generated models as evaluation-only until you have cleared that licence."]
        L += [f"  {i['id']}: {i['notes']}" for i in gen]
    if excluded:
        L += ["", "Skipped files:"] + [f"  {e['file']}: {e['reason']}" for e in excluded]
    (Path(lib) / "LICENSES.txt").write_text("\n".join(L) + "\n", encoding="utf-8")


def load_catalog(cfg):
    cat = jload(lib_dir(cfg) / "catalog.json")
    if cat is None:        # never built (e.g. setup step skipped): scan what is on disk now - offline, ~1 s
        try:
            cat = build_catalog(cfg)
        except Exception as e:
            log(f"furniture catalog could not be built: {e}")
    return cat or {"items": [], "excluded": [], "coverage": {}}


# ---------- choosing a model per layout item ----------
def fit(entry, item, cfg):
    """How the model fits the layout slot. {'ok', 'err', 'scale': [sx, sy, sz], 'why'}."""
    fc = cfg["furniture"]
    tgt, dims = item["size"], entry["dims_m"]
    if entry.get("fit") == "height":
        s = tgt[2] / max(dims[2], 1e-6)
        ok = dims[0] * s <= tgt[0] * 1.8 and dims[1] * s <= tgt[1] * 1.8 and 0.3 <= s <= 3.0
        return {"ok": ok, "err": round(abs(s - 1) * 0.25, 4), "scale": [round(s, 4)] * 3,
                "why": "" if ok else "footprint too big for the slot at this height"}
    s = [t / max(d, 1e-6) for t, d in zip(tgt, dims)]
    su = statistics.median(s)
    r = [x / su for x in s]
    worst = max(abs(x - 1) for x in r)
    ok_u, ok_n = abs(su - 1) <= fc["max_uniform_change"], worst <= fc["max_nonuniform"]
    why = "" if ok_u and ok_n else (f"non-uniform stretch {100 * worst:.0f} % > {100 * fc['max_nonuniform']:.0f} %"
                                    if not ok_n else f"size off by {100 * abs(su - 1):.0f} %")
    return {"ok": ok_u and ok_n, "err": round(worst + 0.25 * abs(su - 1), 4), "scale": [round(x, 4) for x in s], "why": why}


def style_score(entry, style_words):
    return len(style_words & set(entry.get("style_tags", [])))


def candidates(item, cat, cfg, style_words):
    out = []
    for e in cat["items"]:
        if e["type"] != catalog_type(item["type"]):
            continue
        f = fit(e, item, cfg)
        out.append({"asset_id": e["id"], "ok": f["ok"], "fit_err": f["err"], "why": f["why"], "scale": f["scale"],
                    "style": style_score(e, style_words), "polycount": e["polycount"], "source": e["source"],
                    "dims_m": e["dims_m"], "name": e["name"]})
    out.sort(key=lambda c: (not c["ok"], -c["style"], c["fit_err"], c["polycount"], c["asset_id"]))
    return out


def poly_cap(cfg, typ):
    fc = cfg["furniture"]
    return int(fc["poly_cap_by_type"].get(typ, fc["poly_cap"]))


def overrides_file(job):
    return Path(job) / "03_layout/asset_overrides.json"


def check_override(job, cfg, item_id, asset_id):
    """Is this agent choice valid (asset id, 'parametric' or 'remove')? Returns [] or a list of errors."""
    lay = jload(Path(job) / "03_layout/layout.json") or {}
    item = next((i for i in lay.get("items", []) if i["id"] == item_id), None)
    if item is None:
        return [f"no furniture item {item_id} in layout.json (see relayout / choose_furniture results)"]
    if asset_id not in ("parametric", "remove"):
        e = next((x for x in load_catalog(cfg)["items"] if x["id"] == asset_id), None)
        if e is None:
            return [f"no catalog model {asset_id}"]
        if e["type"] != catalog_type(item["type"]):
            return [f"{asset_id} is a {e['type']}, item {item_id} is a {item['type']}"]
        f = fit(e, item, cfg)
        if not f["ok"]:
            return [f"{asset_id} does not fit the slot {item['size']} m: {f['why']}"]
    return []


def set_override(job, cfg, item_id, asset_id):
    """Store an agent choice for one item. Returns [] or a list of errors (nothing stored then)."""
    errs = check_override(job, cfg, item_id, asset_id)
    if errs:
        return errs
    ov = jload(overrides_file(job), {})
    ov[item_id] = asset_id
    jsave(overrides_file(job), ov)
    return []


def select_assets(job, cfg=None, style=None):
    """Pick a model for every layout item -> 03_layout/assets.json (read by blender_scene.py)."""
    cfg = cfg or load_config()
    job = Path(job)
    lay = jload(job / "03_layout/layout.json") or {"items": []}
    cat = load_catalog(cfg) if cfg["furniture"]["enabled"] else {"items": []}
    ov = jload(overrides_file(job), {})
    style_words = words(style or ov.get("__style__") or cfg.get("style", ""))
    by_id = {e["id"]: e for e in cat["items"]}
    out = {}
    for it in lay["items"]:
        o = ov.get(it["id"])
        rec = {"type": it["type"], "room": it["room"], "size": it["size"]}
        if o == "remove":
            out[it["id"]] = dict(rec, source="removed", reason="removed by the agent")
            continue
        if o == "parametric" or it["type"] in PARAMETRIC_ONLY:
            out[it["id"]] = dict(rec, source="parametric", reason="agent choice" if o else "always parametric")
            continue
        pick = None
        if o in by_id:
            f = fit(by_id[o], it, cfg)
            pick = (by_id[o], f, True) if f["ok"] else None
        if pick is None:
            c = next((c for c in candidates(it, cat, cfg, style_words) if c["ok"]), None)
            if c:
                pick = (by_id[c["asset_id"]], fit(by_id[c["asset_id"]], it, cfg), False)
        if pick is None:
            n = sum(1 for e in cat["items"] if e["type"] == catalog_type(it["type"]))
            out[it["id"]] = dict(rec, source="parametric",
                                 reason=f"{n} catalog model(s) of this type, none fits the slot" if n else "no catalog model of this type")
            continue
        e, f, forced = pick
        out[it["id"]] = dict(rec, source=e["source"], asset_id=e["id"], file=e["file"], licence=e["licence"],
                             front_axis=e["front_axis"], unit_scale=e["unit_scale"], fit=e["fit"], scale=f["scale"],
                             fit_err=f["err"], style=style_score(e, style_words), polycount=e["polycount"],
                             poly_cap=poly_cap(cfg, it["type"]), override=forced)
    summ = {}
    for v in out.values():
        summ.setdefault(v["type"], {}).setdefault(v["source"], 0)
        summ[v["type"]][v["source"]] += 1
    res = {"items": out, "summary": summ, "style_words": sorted(style_words),
           "limits": {k: cfg["furniture"][k] for k in ("max_uniform_change", "max_nonuniform")}}
    jsave(job / "03_layout/assets.json", res)
    return res


def coverage(job):
    """Furniture type vs source actually used (after Blender QA) -> final/coverage.json + coverage.md."""
    job = Path(job)
    qa = jload(job / "04_scene/furniture_qa.json")
    planned = (jload(job / "03_layout/assets.json") or {}).get("items", {})
    rows = qa["items"] if qa else [{"item_id": k, "type": v["type"], "room": v["room"], "planned": v["source"],
                                    "used": v["source"], "reason": v.get("reason", "")} for k, v in planned.items()]
    table = {}
    for r in rows:
        table.setdefault(r["type"], {}).setdefault(r["used"], 0)
        table[r["type"]][r["used"]] += 1
    srcs = ["polyhaven", "user", "generated", "parametric", "skipped", "removed"]
    fallbacks = [r for r in rows if r["used"] in ("parametric", "skipped") and r.get("reason") not in (None, "", "always parametric")]
    out = {"from": "blender QA" if qa else "plan (scene not built yet)", "by_type": table, "fallbacks": fallbacks, "items": rows}
    (job / "final").mkdir(parents=True, exist_ok=True)
    jsave(job / "final/coverage.json", out)
    L = [f"# Furniture coverage - {job.name}", "", f"Source: {out['from']}", "",
         "| Type | " + " | ".join(srcs) + " |", "|---|" + "---|" * len(srcs)]
    L += [f"| {t} | " + " | ".join(str(c.get(s, "")) for s in srcs) + " |" for t, c in sorted(table.items())]
    if fallbacks:
        L += ["", "## Fallbacks (parametric model, or skipped for plants)", ""] + [
            f"- {r['item_id']} ({r['type']}) -> {r['used']}: {r['reason']}" for r in fallbacks]
    (job / "final/coverage.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    return out


def report(cfg=None):
    cat = load_catalog(cfg or load_config())
    cov = cat.get("coverage", {})
    print(f"{'type':<14} polyhaven  user  generated")
    for t in TYPES:
        c = cov.get(t, {})
        print(f"{t:<14} {c.get('polyhaven', 0):>9} {c.get('user', 0):>5} {c.get('generated', 0):>10}")
    none = [t for t in TYPES if not sum(cov.get(t, {}).values())]
    print(f"models: {len(cat['items'])}, skipped files: {len(cat['excluded'])}; types without any model "
          f"(parametric boxes are used): {', '.join(none) or 'none'}")
    for e in cat["excluded"][:20]:
        print(f"  skipped {e['file']}: {e['reason']}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "report"
    if cmd == "fetch":
        fetch_polyhaven()
        report()
    elif cmd == "catalog":
        build_catalog()
        report()
    else:
        report()
__END_OF_FURNITURE_PY__
  cat > "$APP/llm_server.py" <<'__END_OF_LLM_SERVER_PY__'
"""Local OpenAI-compatible LLM server (vLLM, own venv at /workspace/venv-llm) + GPU memory scheduling.
The agent starts the server, stops it before a GPU stage that would not fit next to it, and starts it again
lazily before the next LLM call. Budgets are ESTIMATES (config vram_estimates_mb, llm.weights_gb); real peaks
are measured with nvidia-smi and logged per agent step.
  python -m app.llm_server start|stop|status|test     manual control (test = one tool call round trip)"""
import json, os, signal, subprocess, sys, time, urllib.error, urllib.request
from pathlib import Path
from .common import WS, gpu_total_mb, gpu_used_mb, jload, jsave, load_config, log, pod_tier

PID_FILE = WS / "cache/llm_server.pid"


def resolve_model(cfg):
    m = cfg["llm"]["model"]
    if m in ("auto", "", None):
        m = cfg.get("vlm_model", "auto")
    if m == "auto":
        m = "Qwen/Qwen3.5-9B" if pod_tier(cfg) == "48gb" else "Qwen/Qwen3.5-4B"
    return m


def memory_utilization(cfg, model, total_mb):
    """vLLM --gpu-memory-utilization: weights + KV/activations (estimates) as a share of the whole GPU."""
    g = cfg["llm"]["gpu_memory_utilization"]
    if g not in ("auto", None):
        return float(g)
    w = cfg["llm"]["weights_gb"].get(model)
    if w is None or not total_mb:
        return 0.85
    return round(min(0.92, max(0.30, (w + cfg["llm"]["kv_overhead_gb"]) * 1024 / total_mb)), 2)


class LLMServer:
    def __init__(self, cfg=None):
        self.cfg = cfg or load_config()
        lc = self.cfg["llm"]
        self.model = resolve_model(self.cfg)
        self.base = f"http://{lc['host']}:{lc['port']}"
        self.proc, self.log_file = None, WS / "logs/llm_server.log"
        self.total_mb = gpu_total_mb()
        self.util = memory_utilization(self.cfg, self.model, self.total_mb)
        self.starts, self.start_seconds = 0, 0.0

    # ---- budget ----
    def budget_mb(self):
        return int(self.util * self.total_mb) if self.total_mb else None

    def command(self):
        lc = self.cfg["llm"]
        exe = Path(lc["venv"]) / "bin/vllm"
        cmd = [str(exe), "serve", self.model, "--host", lc["host"], "--port", str(lc["port"]),
               "--served-model-name", lc["served_name"], "--max-model-len", str(lc["max_model_len"]),
               "--max-num-seqs", str(lc["max_num_seqs"]), "--gpu-memory-utilization", str(self.util),
               "--enable-auto-tool-choice", "--tool-call-parser", lc["tool_call_parser"],
               "--reasoning-parser", lc["reasoning_parser"],
               "--default-chat-template-kwargs", json.dumps({"enable_thinking": False})]
        cmd += ["--limit-mm-per-prompt", json.dumps({"image": 2, "video": 0})] if lc["vision"] else ["--language-model-only"]
        return cmd + [str(a) for a in lc["extra_args"]]

    # ---- process ----
    def healthy(self):
        try:
            with urllib.request.urlopen(self.base + "/v1/models", timeout=5) as r:
                return r.status == 200
        except Exception:
            return False

    def running(self):
        return self.proc is not None and self.proc.poll() is None

    def start(self):
        if not self.cfg["llm"]["manage_server"]:
            if not self.healthy():
                raise RuntimeError(f"llm.manage_server is false but nothing answers at {self.base}")
            return
        if self.running() and self.healthy():
            return
        exe = Path(self.cfg["llm"]["venv"]) / "bin/vllm"
        if not exe.exists():
            raise RuntimeError(f"{exe} not found - run setup.sh without SKIP_AGENT_LLM=1, or use --backend rules")
        self._kill_stale()
        env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", VLLM_CACHE_ROOT=str(WS / "cache/vllm"),
                   VLLM_NO_USAGE_STATS="1", DO_NOT_TRACK="1")
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = open(self.log_file, "a", encoding="utf-8")
        fh.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} {' '.join(self.command())}\n")
        fh.flush()
        t0 = time.time()
        self.proc = subprocess.Popen(self.command(), stdout=fh, stderr=subprocess.STDOUT, env=env, start_new_session=True)
        PID_FILE.parent.mkdir(parents=True, exist_ok=True)
        PID_FILE.write_text(str(self.proc.pid))
        log(f"llm: starting {self.model} (gpu-memory-utilization {self.util}, ~{self.budget_mb()} MB), log {self.log_file}")
        while time.time() - t0 < self.cfg["llm"]["startup_timeout_s"]:
            if self.proc.poll() is not None:
                tail = self.log_file.read_text(encoding="utf-8", errors="replace")[-3000:]
                raise RuntimeError(f"vLLM exited with code {self.proc.returncode} during start-up:\n{tail}")
            if self.healthy():
                self.starts += 1
                self.start_seconds += time.time() - t0
                log(f"llm: ready in {time.time() - t0:.0f}s")
                return
            time.sleep(3)
        self.stop()
        raise RuntimeError(f"vLLM not ready after {self.cfg['llm']['startup_timeout_s']} s (see {self.log_file})")

    def stop(self, wait_free=True):
        if self.proc is None or not self.cfg["llm"]["manage_server"]:
            return
        if self.proc.poll() is None:
            try:
                os.killpg(self.proc.pid, signal.SIGTERM)
                self.proc.wait(timeout=60)
            except Exception:
                try:
                    os.killpg(self.proc.pid, signal.SIGKILL)
                    self.proc.wait(timeout=30)
                except Exception:
                    pass
        self.proc = None
        PID_FILE.unlink(missing_ok=True)
        if wait_free and self.total_mb:          # CUDA memory is released a moment after the process ends
            t0 = time.time()
            while time.time() - t0 < 60 and (gpu_used_mb() or 0) > 2500:
                time.sleep(2)
        log("llm: server stopped")

    def _kill_stale(self):
        """A server left over from a crashed run would hold the GPU memory: stop it first. The PID file lives on
        /workspace and survives pod restarts, so the PID is only signalled if it still belongs to a vLLM process."""
        try:
            pid = int(PID_FILE.read_text())
            cmd = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
            if "vllm" in cmd and "serve" in cmd:
                os.killpg(pid, signal.SIGTERM)
                time.sleep(5)
                log(f"llm: stopped a stale server (pid {pid})")
        except Exception:
            pass
        PID_FILE.unlink(missing_ok=True)

    # ---- API ----
    def chat(self, messages, tools=None, max_tokens=None, temperature=None):
        lc = self.cfg["llm"]
        body = {"model": lc["served_name"], "messages": messages, "max_tokens": max_tokens or lc["max_tokens"],
                "temperature": lc["temperature"] if temperature is None else temperature,
                "chat_template_kwargs": {"enable_thinking": False}}
        if tools:
            body.update(tools=tools, tool_choice="auto")
        req = urllib.request.Request(self.base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json", "Authorization": "Bearer EMPTY"})
        try:
            with urllib.request.urlopen(req, timeout=lc["request_timeout_s"]) as r:
                out = json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"LLM HTTP {e.code}: {e.read().decode(errors='replace')[:800]}")
        return out["choices"][0]["message"], out.get("usage", {})


class VramPlanner:
    """Decides, from the stated budget, whether a GPU component can run while the LLM server is up."""
    def __init__(self, cfg, server=None):
        self.cfg, self.server = cfg, server
        self.total = gpu_total_mb()
        self.est = cfg["vram_estimates_mb"]
        self.decisions = []

    def fits(self, component):
        if not self.total:
            return True
        need = self.est.get(component, 0) + self.est["margin"]
        srv = self.server.budget_mb() if self.server and self.server.running() else 0
        return srv + need <= self.total

    def before(self, component):
        """Stop the LLM server if the component does not fit next to it. Returns the decision text."""
        if component is None or self.server is None or not self.server.running():
            return None
        if self.fits(component):
            d = (f"{component}: server kept ({self.server.budget_mb()} MB) + ~{self.est.get(component, 0)} MB "
                 f"+ {self.est['margin']} MB margin <= {self.total} MB")
        else:
            self.server.stop()
            d = (f"{component}: server stopped (~{self.est.get(component, 0)} MB + server would exceed {self.total} MB)")
        self.decisions.append(d)
        return d


def table(cfg):
    """VRAM plan (estimates) as rows for the report."""
    tier, total = pod_tier(cfg), gpu_total_mb()
    model = resolve_model(cfg)
    util = memory_utilization(cfg, model, total or (46068 if tier == "48gb" else 24564))
    est = cfg["vram_estimates_mb"]
    rows = [("LLM server " + model, int(util * (total or 0)) or f"{util} x GPU", "reserved while up")]
    rows += [(k, v, "") for k, v in est.items() if k != "margin"]
    return {"tier": tier, "gpu_total_mb": total, "rows": rows, "margin_mb": est["margin"]}


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
    s = LLMServer()
    if cmd == "status":
        print(json.dumps({"model": s.model, "util": s.util, "budget_mb": s.budget_mb(), "healthy": s.healthy(),
                          "command": " ".join(s.command())}, indent=1))
    elif cmd == "stop":
        s._kill_stale()
    elif cmd in ("start", "test"):
        s.start()
        if cmd == "test":
            tools = [{"type": "function", "function": {"name": "get_room_count", "description": "Number of rooms",
                      "parameters": {"type": "object", "properties": {"plan": {"type": "string"}}, "required": ["plan"]}}}]
            t0 = time.time()
            msg, usage = s.chat([{"role": "user", "content": "How many rooms does plan 'demo' have? Use the tool."}], tools)
            ok = bool(msg.get("tool_calls"))
            print(json.dumps({"tool_call_ok": ok, "message": msg, "usage": usage, "seconds": round(time.time() - t0, 1),
                              "vram_used_mb": gpu_used_mb()}, indent=1, ensure_ascii=False))
            s.stop()
            sys.exit(0 if ok else 1)
        print(f"server up at {s.base} (pid {s.proc.pid}); stop with: python -m app.llm_server stop")


if __name__ == "__main__":
    main()
__END_OF_LLM_SERVER_PY__
  cat > "$APP/vision.py" <<'__END_OF_VISION_PY__'
"""Vision for the agent: the served Qwen3.5 model reads images through the same OpenAI-compatible endpoint.
read_texts(): same tiles / prompt / parser as app/vlm.py, writes the same 01_parse/vlm_texts.json.
critique(): asks for a JSON list of problems in overlay.png (vs the source page), layout.png or a render.
When no multimodal server is available the agent reports 'vision unavailable' instead of guessing."""
import base64, io, json, re, time
from pathlib import Path
from .common import jload, jsave, log

CRITIQUE = {
    "overlay": ("Image 1 is the original apartment floor plan. Image 2 is our reading of it: coloured room areas with "
                "name, size (m) and area (m2), red wall bands, green door lines, blue window lines. Compare them. List "
                "every error you can see: missing/extra walls, doors or windows that are missing, extra or in the wrong "
                "place, wrong room types or names, sizes or m2 values that disagree with the printed labels."),
    "layout": ("This is a furniture layout drawn on a floor plan (grey rooms, brown furniture with names, green door "
               "zones, blue window zones, a short red line shows each item's front). List problems: furniture "
               "blocking doors or windows, items facing a wall, missing main furniture, unrealistic arrangements."),
    "render": ("This is a render of one room of a 3D apartment model. List visible problems: furniture floating, "
               "sunk into the floor or intersecting walls, furniture facing the wall, missing walls or holes, "
               "black or broken textures, image far too dark or too bright."),
}
FORMAT = ('Reply ONLY with JSON: {"overall": "ok" | "minor" | "major", "issues": [{"kind": "wall|door|window|room|'
          'label|scale|furniture|render|other", "where": "room name or id", "problem": "short text", '
          '"suggestion": "short text"}]}')


def data_url(path, max_side=1280):
    from PIL import Image
    im = Image.open(path).convert("RGB")
    k = min(1.0, max_side / max(im.size))
    if k < 1:
        im = im.resize((int(im.size[0] * k), int(im.size[1] * k)), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def parse_json(txt):
    txt = re.sub(r"<think>.*?</think>", "", txt or "", flags=re.S)
    m = re.search(r"\{.*\}", txt, re.S)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass
    return {"overall": "unknown", "issues": [], "raw": (txt or "")[:1500]}


class ServerVision:
    def __init__(self, server):
        self.server = server
        self.available = bool(server.cfg["llm"]["vision"])

    def ask(self, images, prompt, max_tokens=1536):
        self.server.start()
        content = [{"type": "image_url", "image_url": {"url": data_url(p)}} for p in images]
        content.append({"type": "text", "text": prompt})
        msg, usage = self.server.chat([{"role": "user", "content": content}], max_tokens=max_tokens, temperature=0.0)
        return msg.get("content") or "", usage

    def critique(self, job, what, image=None):
        job = Path(job)
        if what == "overlay":
            imgs = [job / "01_parse/page.png", job / "02_plan/overlay.png"]
        elif what == "layout":
            imgs = [job / "03_layout/layout.png"]
        else:
            imgs = [image]
        missing = [str(p) for p in imgs if not Path(p).exists()]
        if missing:
            return {"available": True, "error": f"image not found: {missing}"}
        txt, usage = self.ask(imgs, CRITIQUE["render" if what == "render" else what] + " " + FORMAT)
        res = parse_json(txt)
        res.update(available=True, image=[str(p) for p in imgs], tokens=usage.get("total_tokens"))
        return res

    def read_texts(self, job):
        """Room names, m2 labels and scale notes from 01_parse/page.png -> 01_parse/vlm_texts.json."""
        from PIL import Image
        from .vlm import PROMPT, tiles, parse_reply, merge
        job = Path(job)
        cfg = self.server.cfg
        page = Image.open(job / "01_parse/page.png").convert("RGB")
        W, H = page.size
        found, raw, t0 = [], [], time.time()
        tmp = job / "01_parse/_tile.png"
        for (x0, y0, x1, y1) in tiles(W, H):
            crop = page.crop((x0, y0, x1, y1))
            k = min(1.0, cfg["vlm_max_side"] / max(crop.size))
            img = crop.resize((int(crop.size[0] * k), int(crop.size[1] * k)), Image.LANCZOS) if k < 1 else crop
            img.save(tmp)
            txt, _ = self.ask([tmp], PROMPT)
            raw.append(txt)
            for f in parse_reply(txt, img.size[0], img.size[1]):
                found.append({"text": f["text"], "x": x0 + f["x"] / k, "y": y0 + f["y"] / k})
        tmp.unlink(missing_ok=True)
        texts = merge(found, (W ** 2 + H ** 2) ** 0.5)
        jsave(job / "01_parse/vlm_texts.json", texts)
        (job / "01_parse/vlm_raw.txt").write_text("\n\n----\n\n".join(raw), encoding="utf-8")
        log(f"vision: {len(texts)} texts in {time.time() - t0:.0f}s via the LLM server")
        return texts


class NoVision:
    available = False

    def __init__(self, reason="no multimodal LLM server (backend rules/mock or llm.vision=false)"):
        self.reason = reason

    def critique(self, job, what, image=None):
        return {"available": False, "reason": self.reason}

    def read_texts(self, job):
        return None


class ScriptedVision:
    """Tests and replays: returns recorded answers in order."""
    available = True

    def __init__(self, answers):
        self.answers = list(answers)

    def critique(self, job, what, image=None):
        return dict(self.answers.pop(0) if self.answers else {"overall": "ok", "issues": []}, available=True)

    def read_texts(self, job):
        texts = self.answers.pop(0) if self.answers else []
        jsave(Path(job) / "01_parse/vlm_texts.json", texts)
        return texts
__END_OF_VISION_PY__
  cat > "$APP/agent.py" <<'__END_OF_AGENT_PY__'
"""Agent layer: a local open-weights LLM (vLLM, OpenAI-compatible) drives the pipeline through tools, inspects
the results, repairs problems with validated patches and delivers <job>/final/ (apartment.blend/.glb/.usdc,
preview renders, overlay.png, report.md).
  python -m app.agent run <plan file> [--job-dir DIR] [--backend llm|rules|mock] [--script mock.json]
  python -m app.agent replay <old job>/agent_log.jsonl [--job-dir DIR]     re-run the same tool calls
  python -m app.agent tools                                                print the tool JSON schemas
Hard limits: agent.max_steps tool calls, agent.max_minutes, agent.max_cost_usd (GPU price x time), a patch
budget and a repair budget per issue. Every tool call (arguments, reasoning, result, seconds, VRAM peak) is
appended to <job>/agent_log.jsonl; replay feeds those calls back in the same order."""
import argparse, json, re, shutil, sys, time, traceback
from pathlib import Path
import numpy as np
from .common import APP, WS, GpuSampler, blender, gpu_used_mb, jload, jsave, load_config, log, pod_tier, run
from .patches import PATCH_SCHEMA, PatchStore, schema_errors
from . import furniture

SYSTEM = """You are the build agent of a pipeline that turns an apartment floor plan into a clean 3D model.
You act ONLY through the tools. You never write geometry, coordinate lists or Blender code: the plan is changed
only with patch_plan operations, furniture only with choose_furniture (or patch_plan swap/remove_furniture).
Goal: final/apartment.blend (+ .glb, .usdc) that matches the source plan, then finish with an honest summary.
Usual order: parse_plan -> read_text_vlm (only if parse_plan says needs_vlm) -> get_plan_summary ->
view_image overlay -> run_checks -> repairs -> relayout -> choose_furniture -> build_scene -> render_preview ->
view_image render -> run_checks -> export_blender -> finish.
Act on every issue that run_checks reports:
- area_label: a room's area disagrees with its printed m2 label. If most labelled rooms are off by the same
  factor the scale is wrong: patch_plan scale_from_label on a large, clearly labelled room. If only one room is
  off, look at the overlay first (label on the wrong room, missed wall) - do not rescale for a single room.
- scale_confidence: read_text_vlm for m2 labels if not done yet, then scale_from_label.
- unlabelled rooms or wrong types (also from view_image critiques): retype_room.
- unfurnished main rooms / layout checks: check room type and door/window positions, patch, then relayout.
- render problems: fix the cause (swap or remove a furniture item, or a plan fix), build_scene again.
- export failures: build_scene and export_blender again.
Every issue has a repair budget. When run_checks marks an issue stop_repairing, stop and list it in
finish.uncertain. Never repeat a patch that was rejected. Use one or two operations per patch with a clear reason.
Ids: rooms r1.., doors d1.., windows win1..; lengths in metres; a wall runs from point a to point b and
move_opening shift_m moves along the wall towards b. Budget: {steps} tool calls and {minutes} minutes."""

STAGE_ORDER = ["plan", "layout", "assets", "scene", "render", "export"]


def obj(props, required=()):
    return {"type": "object", "properties": props, "required": list(required), "additionalProperties": False}


TOOLS = {}


def tool(name, desc, params, gpu=None):
    def deco(fn):
        TOOLS[name] = {"desc": desc, "params": params, "fn": fn, "gpu": gpu}
        return fn
    return deco


def openai_tools():
    return [{"type": "function", "function": {"name": n, "description": t["desc"], "parameters": t["params"]}}
            for n, t in TOOLS.items()]


def gpu_present():
    from .main import gpu_present as g
    return g()


# ---------- context shared by the tools ----------
class Ctx:
    def __init__(self, inp, job, cfg, vision, server=None, planner=None):
        self.inp, self.job, self.cfg = Path(inp), Path(job), cfg
        self.vision, self.server, self.planner = vision, server, planner
        self.store = PatchStore(job)
        self.gpu = gpu_present()
        self.t0 = time.time()
        self.steps, self.done, self.stop_reason = 0, False, None
        self.fresh = set()                          # stages whose output matches the current inputs
        self.plan_version, self.actions = 0, 0
        self.attempts, self.last_keys, self.given_up = {}, set(), set()
        self.critiques, self.renders, self.last_checks = [], [], None
        self.validation, self.finish_warned, self.final = None, False, None
        self.replay_data = None

    def invalidate(self, from_stage):
        for s in STAGE_ORDER[STAGE_ORDER.index(from_stage):]:
            self.fresh.discard(s)

    def minutes(self):
        return (time.time() - self.t0) / 60

    def cost(self):
        return round(self.minutes() / 60 * self.cfg["gpu_price_per_hour"], 3)

    def budget(self):
        a = self.cfg["agent"]
        return {"steps_left": a["max_steps"] - self.steps, "minutes_left": round(a["max_minutes"] - self.minutes(), 1),
                "cost_usd": self.cost(), "cost_limit_usd": a["max_cost_usd"]}

    def over_budget(self):
        a = self.cfg["agent"]
        if self.steps >= a["max_steps"]:
            return f"step limit ({a['max_steps']}) reached"
        if self.minutes() >= a["max_minutes"]:
            return f"time limit ({a['max_minutes']} min) reached"
        if self.cost() >= a["max_cost_usd"]:
            return f"cost limit (${a['max_cost_usd']}) reached"
        return None

    def run_config(self, profile):
        cfg = self.cfg
        dev = (jload(APP / "device.json") or {}).get("device", "OPTIX" if self.gpu else "CPU")
        jsave(self.job / "run_config.json", {"assets": str(WS / "assets"), "profile": profile, "device": dev,
                                             "profile_settings": cfg["profiles"][profile], "exposure": cfg["exposure"],
                                             "sun_strength": cfg["sun_strength"], "fill_w_per_m2": cfg["fill_w_per_m2"],
                                             "export": cfg["export"]})

    def plan(self):
        return jload(self.job / "02_plan/plan.json")


# ---------- summaries ----------
def plan_summary(plan, detail="rooms"):
    sc = plan["scale"]
    rooms = []
    for r in plan["rooms"]:
        dev = round(100 * (r["area_m2"] / r["label_area_m2"] - 1), 1) if r.get("label_area_m2") else None
        rooms.append({"id": r["id"], "type": r["type"], "label": r["label"], "size_m": r["size_m"], "area_m2": r["area_m2"],
                      "label_m2": r.get("label_area_m2"), "diff_pct": dev, "confidence": r.get("confidence")})
    out = {"source": plan["source"], "scale": {"method": sc["method"], "confidence": sc["confidence"],
                                               "agent_factor": sc.get("agent_factor", 1.0)},
           "counts": {k: len(plan[k]) for k in ("walls", "doors", "windows", "rooms")}, "rooms": rooms,
           "warnings": plan.get("warnings", [])[:12], "patches_applied": plan.get("patches_applied", 0)}
    if detail in ("openings", "all"):
        out["doors"] = [{"id": d["id"], "wall": d["wall_id"], "width": d["width"], "kind": d["kind"], "rooms": d.get("rooms"),
                         "swing": d.get("swing")} for d in plan["doors"]]
        out["windows"] = [{"id": w["id"], "wall": w["wall_id"], "width": w["width"], "kind": w["kind"], "sill": w["sill"],
                           "head": w["head"], "rooms": w.get("rooms")} for w in plan["windows"]]
    if detail == "all":
        out["walls"] = [{"id": w["id"], "a": w["a"], "b": w["b"], "t": w["thickness"], "exterior": w["exterior"]}
                        for w in plan["walls"]]
    return out


def render_sanity(path):
    import cv2
    img = cv2.imread(str(path))
    if img is None:
        return {"name": Path(path).stem, "flags": ["unreadable"]}
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    s = {"name": Path(path).stem, "mean": round(float(g.mean()), 1), "std": round(float(g.std()), 1),
         "black": round(float((g < 10).mean()), 3), "blown": round(float((g > 250).mean()), 3)}
    flags = []
    if s["mean"] < 35 or s["black"] > 0.6:
        flags.append("too_dark")
    if s["mean"] > 225 or s["blown"] > 0.25:
        flags.append("too_bright")
    if s["std"] < 12:
        flags.append("flat_image")
    s["flags"] = flags
    return s


# ---------- tools ----------
@tool("parse_plan", "Read the input file and build the 2D plan (walls, doors, windows, rooms, scale). reparse=false "
      "only re-runs the geometry step (e.g. after read_text_vlm). Stored patches are re-applied automatically.",
      obj({"reparse": {"type": "boolean", "default": True},
           "wallseg": {"enum": ["config", "on", "off"], "description": "trained wall model for scans/photos"}}))
def t_parse_plan(ctx, a):
    from .parse import run_parse
    from .plan import run_plan
    if a.get("reparse", True) or not (ctx.job / "01_parse/extract.json").exists():
        run_parse(ctx.inp, ctx.job)
        use = {"on": True, "off": False}.get(a.get("wallseg", "config"), ctx.cfg["wallseg"]["enabled"])
        kind = (jload(ctx.job / "01_parse/extract.json") or {}).get("kind")
        if use and kind in ("pdf_scan", "photo"):
            try:
                run([sys.executable, "-m", "app.wallseg", "predict", ctx.job], log_file=ctx.job / "logs/wallseg.log", cwd=str(WS))
            except Exception as e:
                log(f"wall model failed, classic walls used: {e}")
    run_plan(ctx.job, ctx.cfg)
    plan, skipped = ctx.store.after_plan_stage()
    ctx.invalidate("plan")
    ctx.fresh.add("plan")
    ctx.plan_version += 1
    ex = jload(ctx.job / "01_parse/extract.json")
    out = plan_summary(plan)
    out.update(kind=ex["kind"], needs_vlm=bool(ex.get("needs_vlm")) and not (ctx.job / "01_parse/vlm_texts.json").exists(),
               patches_not_reapplied=skipped)
    return out


def _vlm_component(ctx):
    if ctx.vision.available:
        return None                                 # read through the running server: no extra GPU memory
    return "vlm_transformers_9b" if "9B" in str(ctx.cfg.get("vlm_model")) or pod_tier(ctx.cfg) == "48gb" else "vlm_transformers_4b"


@tool("read_text_vlm", "Read room names, m2 labels and scale notes from the page image with the vision model, then "
      "rebuild the plan with them. Needed for scans and photos (parse_plan says needs_vlm).",
      obj({"force": {"type": "boolean", "description": "read again even if texts exist"}}), gpu=_vlm_component)
def t_read_text_vlm(ctx, a):
    from .plan import run_plan
    vt = ctx.job / "01_parse/vlm_texts.json"
    if vt.exists() and not a.get("force"):
        texts, how = jload(vt), "reused"
    elif ctx.vision.available:
        texts, how = ctx.vision.read_texts(ctx.job), "llm server"
    elif ctx.gpu:
        run([sys.executable, "-m", "app.vlm", ctx.job], log_file=ctx.job / "logs/vlm.log", cwd=str(WS))
        texts, how = jload(vt, []), "transformers"
    else:
        return {"error": "no vision model available (no GPU / no multimodal server)"}
    ctx.replay_data = {"vlm_texts": texts}
    run_plan(ctx.job, ctx.cfg)
    plan, skipped = ctx.store.after_plan_stage()
    ctx.invalidate("plan")
    ctx.fresh.add("plan")
    ctx.plan_version += 1
    ctx.actions += 1
    out = plan_summary(plan)
    out.update(texts_read=len(texts or []), how=how, sample=[t["text"] for t in (texts or [])][:15])
    return out


@tool("get_plan_summary", "Current plan: scale, rooms (size, area, printed label, difference), warnings; "
      "detail=openings adds doors/windows, detail=all adds walls.",
      obj({"detail": {"enum": ["rooms", "openings", "all"], "default": "rooms"}}))
def t_get_plan_summary(ctx, a):
    plan = ctx.plan()
    if plan is None:
        return {"error": "no plan yet: call parse_plan"}
    return plan_summary(plan, a.get("detail", "rooms"))


@tool("view_image", "Ask the vision model to critique an image: 'overlay' (our plan drawn over the source page, "
      "compared with the source), 'layout' (furniture plan) or 'render' (a preview render, name from render_preview).",
      obj({"image": {"enum": ["overlay", "layout", "render"]}, "render_name": {"type": "string", "maxLength": 80}},
          ["image"]))
def t_view_image(ctx, a):
    what = a["image"]
    img = None
    if what == "render":
        names = [r["name"] for r in ctx.renders]
        name = a.get("render_name") or (names[0] if names else None)
        if not name or not (ctx.job / "05_render" / f"{name}.png").exists():
            return {"error": f"no render {name!r}; available: {names}"}
        img = ctx.job / "05_render" / f"{name}.png"
    res = ctx.vision.critique(ctx.job, what, img)
    ctx.replay_data = {"critique": res}
    if res.get("available"):
        ctx.critiques.append({"what": what, "name": a.get("render_name"), "plan_version": ctx.plan_version,
                              "overall": res.get("overall"), "issues": res.get("issues", [])[:12], "step": ctx.steps})
    return res


@tool("patch_plan", "Change the plan or furniture with structured operations (validated against a JSON schema and "
      "plan invariants; rejected patches change nothing). Always give the reason (which check/critique it fixes).",
      PATCH_SCHEMA)
def t_patch_plan(ctx, a):
    furn = [op for op in a["ops"] if op["op"] in ("swap_furniture", "remove_furniture")]
    errs = []
    for op in furn:
        errs += furniture.check_override(ctx.job, ctx.cfg, op["item_id"],
                                         op["asset_id"] if op["op"] == "swap_furniture" else "remove")
    if errs:
        return {"accepted": False, "stage": "furniture", "errors": errs}
    res = ctx.store.propose(a, ctx.cfg["agent"]["max_patches"], meta={"step": ctx.steps})
    if res["accepted"]:
        ctx.actions += 1
        for op in furn:
            furniture.set_override(ctx.job, ctx.cfg, op["item_id"], op["asset_id"] if op["op"] == "swap_furniture" else "remove")
        if res["changes"]:
            ctx.invalidate("layout")
            ctx.plan_version += 1
        if furn:
            ctx.invalidate("assets")
        res["next"] = "relayout, then choose_furniture/build_scene" if res["changes"] else "build_scene"
    return res


@tool("relayout", "Place the furniture again on the current plan (rule based). Returns layout checks, unfurnished "
      "main rooms and what could not be placed per room.", obj({}))
def t_relayout(ctx, a):
    from .layout import run_layout
    if ctx.plan() is None:
        return {"error": "no plan yet: call parse_plan"}
    out = run_layout(ctx.job, ctx.cfg)
    ctx.invalidate("layout")
    ctx.fresh.add("layout")
    ctx.actions += 1
    return {"items": len(out["items"]), "checks": out["checks"], "unfurnished_main_rooms": out["unfurnished_main_rooms"],
            "rooms": out["rooms"], "warnings": out["warnings"][:10]}


@tool("choose_furniture", "Pick a 3D model for every furniture item (type, size fit, style). Optional: force one "
      "item to a catalog model / 'parametric' / 'remove', set a style (e.g. 'scandinavian oak'), or list the "
      "candidate models of one item.",
      obj({"item_id": {"type": "string", "maxLength": 80}, "asset_id": {"type": "string", "maxLength": 120},
           "style": {"type": "string", "maxLength": 200}, "list_candidates_for": {"type": "string", "maxLength": 80}}))
def t_choose_furniture(ctx, a):
    if "layout" not in ctx.fresh and not (ctx.job / "03_layout/layout.json").exists():
        return {"error": "no layout yet: call relayout"}
    if a.get("item_id") and a.get("asset_id"):
        errs = furniture.set_override(ctx.job, ctx.cfg, a["item_id"], a["asset_id"])
        if errs:
            return {"error": "; ".join(errs)}
        ctx.actions += 1
    if a.get("style"):
        ov = jload(furniture.overrides_file(ctx.job), {})
        ov["__style__"] = a["style"]
        jsave(furniture.overrides_file(ctx.job), ov)
        ctx.actions += 1
    res = furniture.select_assets(ctx.job, ctx.cfg)
    ctx.invalidate("assets")
    ctx.fresh.add("assets")
    out = {"by_type": res["summary"], "style_words": res["style_words"],
           "parametric_reasons": {k: v["reason"] for k, v in res["items"].items() if v["source"] == "parametric"}}
    if a.get("list_candidates_for"):
        lay = jload(ctx.job / "03_layout/layout.json")
        it = next((i for i in lay["items"] if i["id"] == a["list_candidates_for"]), None)
        out["candidates"] = (furniture.candidates(it, furniture.load_catalog(ctx.cfg), ctx.cfg, set(res["style_words"]))[:6]
                             if it else f"no item {a['list_candidates_for']}")
    return out


@tool("build_scene", "Build the 3D scene in Blender from the current plan, layout and furniture choice "
      "(walls with real openings, floors, furniture, lights, cameras). CPU only.", obj({}))
def t_build_scene(ctx, a):
    if not (ctx.job / "03_layout/layout.json").exists():
        return {"error": "no layout yet: call relayout"}
    if "assets" not in ctx.fresh:
        furniture.select_assets(ctx.job, ctx.cfg)
        ctx.fresh.add("assets")
    ctx.run_config(ctx.cfg["agent"]["preview_profile"])
    blender("blender_scene.py", ctx.job)
    ctx.invalidate("scene")
    ctx.fresh.add("scene")
    cams = jload(ctx.job / "04_scene/cameras.json", {})
    qa = jload(ctx.job / "04_scene/furniture_qa.json", {})
    fb = [f"{r['item_id']}: {r['reason']}" for r in qa.get("items", []) if r.get("planned") != r.get("used")]
    return {"cameras": [c["name"] for c in cams.get("cameras", [])], "warnings": cams.get("warnings", [])[:10],
            "furniture_used": qa.get("summary", {}), "fallbacks_to_parametric": fb[:15]}


def _render_component(ctx):
    return "cycles_preview"


@tool("render_preview", "Render every camera of the built scene with Cycles at preview quality (GPU) and check "
      "brightness/contrast of each image. Use view_image render to let the vision model look at them.",
      obj({}), gpu=_render_component)
def t_render_preview(ctx, a):
    if "scene" not in ctx.fresh:
        return {"error": "the scene is missing or older than the plan/furniture: call build_scene first"}
    if not ctx.gpu and not ctx.cfg["agent"]["render_on_cpu"]:
        return {"skipped": "no GPU (agent.render_on_cpu is false)"}
    for old in (ctx.job / "05_render").glob("*"):
        old.unlink()
    ctx.run_config(ctx.cfg["agent"]["preview_profile"])
    blender("blender_render.py", ctx.job)
    ctx.fresh.add("render")
    ren = jload(ctx.job / "05_render/render.json", {})
    ctx.renders = [render_sanity(ctx.job / "05_render" / f"{v['name']}.png") for v in ren.get("views", [])]
    return {"device": ren.get("device"), "renders": ctx.renders}


def compute_issues(ctx):
    a = ctx.cfg["agent"]
    plan = ctx.plan()
    if plan is None:
        return [{"key": "no_plan", "severity": "high", "problem": "no plan yet", "suggest": "parse_plan"}]
    raster = plan["source"]["kind"] in ("pdf_scan", "photo")
    tol = a["area_tol_pct_raster"] if raster else a["area_tol_pct_vector"]
    iss = []
    sc = plan["scale"]
    if sc["confidence"] < a["min_scale_confidence"] and sc.get("agent_factor", 1.0) == 1.0:
        iss.append({"key": "scale_confidence", "severity": "high",
                    "problem": f"scale from {sc['method']} with confidence {sc['confidence']}",
                    "suggest": "read_text_vlm for m2 labels, then patch_plan scale_from_label"})
    for r in plan["rooms"]:
        if r.get("label_area_m2") and r["type"] != "balcony":
            dev = 100 * (r["area_m2"] / r["label_area_m2"] - 1)
            if abs(dev) > tol:
                iss.append({"key": f"area_label:{r['id']}", "severity": "high",
                            "problem": f"{r['id']} {r['label'] or r['type']}: {r['area_m2']} m2 measured vs label "
                                       f"{r['label_area_m2']} m2 ({dev:+.1f} %, tolerance {tol} %)",
                            "suggest": "view_image overlay; scale_from_label if all rooms are off by the same factor"})
        if r.get("confidence", 1) <= 0.3 and r["type"] != "balcony":
            iss.append({"key": f"unlabelled:{r['id']}", "severity": "medium",
                        "problem": f"{r['id']} ({r['area_m2']} m2) has no label; typed '{r['type']}' from its shape",
                        "suggest": "view_image overlay, then retype_room if the type is wrong"})
    lay = jload(ctx.job / "03_layout/layout.json")
    if lay and "layout" in ctx.fresh:
        for k, v in lay["checks"].items():
            if v:
                iss.append({"key": f"layout:{k}", "severity": "medium", "problem": f"layout check {k} = {v}",
                            "suggest": "fix doors/windows or room types, relayout; or remove_furniture"})
        for rid in lay["unfurnished_main_rooms"]:
            iss.append({"key": f"unfurnished:{rid}", "severity": "high", "problem": f"main room {rid} has no furniture",
                        "suggest": "check room type / openings (view_image layout), patch, relayout"})
    if "scene" in ctx.fresh and "render" not in ctx.fresh:
        iss.append({"key": "todo:render_preview", "severity": "medium",
                    "problem": "no preview renders of the current scene" + ("" if ctx.gpu else " (no GPU on this machine)"),
                    "suggest": "render_preview, then view_image render"})
    for r in ctx.renders if "render" in ctx.fresh else []:
        for f in r["flags"]:
            iss.append({"key": f"render:{r['name']}:{f}", "severity": "medium", "problem": f"render {r['name']}: {f}",
                        "suggest": "view_image render; check lights/windows or camera placement"})
    for c in ctx.critiques:
        if c["what"] == "overlay" and c["plan_version"] == ctx.plan_version and c["overall"] in ("minor", "major"):
            for i, x in enumerate(c["issues"][:6]):
                iss.append({"key": f"critique:overlay:{x.get('kind', 'other')}:{x.get('where', i)}",
                            "severity": "high" if c["overall"] == "major" else "medium",
                            "problem": f"VLM on overlay: {x.get('where', '')}: {x.get('problem', '')}",
                            "suggest": x.get("suggestion", "")})
        if c["what"] == "render" and "render" in ctx.fresh and c["overall"] in ("minor", "major"):
            for i, x in enumerate(c["issues"][:4]):
                iss.append({"key": f"critique:render:{c.get('name')}:{i}", "severity": "medium",
                            "problem": f"VLM on render {c.get('name')}: {x.get('problem', '')}",
                            "suggest": x.get("suggestion", "")})
    if ctx.vision.available and a["critique_overlay"] and not any(
            c["what"] == "overlay" and c["plan_version"] == ctx.plan_version for c in ctx.critiques):
        iss.append({"key": "todo:overlay_critique", "severity": "low", "problem": "overlay not yet checked by the VLM",
                    "suggest": "view_image overlay"})
    qa = jload(ctx.job / "04_scene/furniture_qa.json", {}) if "scene" in ctx.fresh else {}
    for r in qa.get("items", []):
        if r.get("planned") not in (None, r.get("used")) and r.get("used") == "parametric":
            iss.append({"key": f"furniture_fallback:{r['item_id']}", "severity": "low",
                        "problem": f"{r['item_id']}: library model rejected ({r['reason']}), parametric used",
                        "suggest": "choose_furniture list_candidates_for, or accept"})
    if ctx.validation is not None and "export" in ctx.fresh and not ctx.validation.get("ok"):
        for c in ctx.validation.get("checks", []):
            if not c.get("ok"):
                iss.append({"key": f"export:{c['name']}", "severity": "high", "problem": f"export check {c['name']}: {c.get('detail')}",
                            "suggest": "build_scene then export_blender"})
    return iss


@tool("run_checks", "All checks on the current state: area vs printed m2 labels, scale confidence, room labels, "
      "layout checks, unfurnished main rooms, render sanity, VLM critiques, export validation. Shows repair "
      "attempts per issue and stop_repairing when its budget is used up.", obj({}))
def t_run_checks(ctx, a):
    iss = compute_issues(ctx)
    keys = {i["key"] for i in iss}
    if ctx.actions:                                 # something was tried since the last check
        for k in keys & ctx.last_keys:
            ctx.attempts[k] = ctx.attempts.get(k, 0) + 1
    ctx.actions = 0
    ctx.last_keys = keys
    mx = ctx.cfg["agent"]["max_repairs_per_issue"]
    for i in iss:
        i["attempts"] = ctx.attempts.get(i["key"], 0)
        if i["attempts"] >= mx and i["severity"] != "low":
            i["stop_repairing"] = True
            ctx.given_up.add(i["key"])
    ctx.last_checks = iss
    return {"issues": iss, "high": sum(i["severity"] == "high" for i in iss), "stale_stages":
            [s for s in STAGE_ORDER if s not in ctx.fresh], "budget": ctx.budget()}


@tool("export_blender", "Write final/apartment.blend (+ .glb, .usdc, previews, overlay) from the built scene and "
      "validate it headless (units, collections, bounding box vs plan, textures packed, watertight walls with "
      "real openings). The job fails if validation fails.", obj({}))
def t_export_blender(ctx, a):
    from .deliver import deliver
    if "scene" not in ctx.fresh:
        return {"error": "the scene is missing or older than the plan/furniture: call build_scene first"}
    ctx.validation = deliver(ctx.job, ctx.cfg)
    ctx.fresh.add("export")
    return {"ok": ctx.validation.get("ok"), "failed": [c for c in ctx.validation.get("checks", []) if not c.get("ok")],
            "files": sorted(p.name for p in (ctx.job / "final").iterdir())}


@tool("finish", "End the run. status complete|partial|failed, a short summary, what you fixed and what is still "
      "uncertain (be specific: room ids, checks).",
      obj({"status": {"enum": ["complete", "partial", "failed"]}, "summary": {"type": "string", "maxLength": 2000},
           "fixed": {"type": "array", "items": {"type": "string", "maxLength": 300}, "maxItems": 30},
           "uncertain": {"type": "array", "items": {"type": "string", "maxLength": 300}, "maxItems": 30}},
          ["status", "summary"]))
def t_finish(ctx, a):
    exported = bool(ctx.validation and ctx.validation.get("ok") and "export" in ctx.fresh)
    open_high = [i for i in compute_issues(ctx) if i["severity"] == "high" and i["key"] not in ctx.given_up]
    if not ctx.finish_warned and (not exported or (open_high and not a.get("uncertain"))):
        ctx.finish_warned = True
        msg = []
        if not exported:
            msg.append("final/apartment.blend has not passed export validation yet (build_scene, export_blender)")
        if open_high and not a.get("uncertain"):
            msg.append("open high-severity issues with repair budget left: " + ", ".join(i["key"] for i in open_high[:8])
                       + " - repair them or list them in 'uncertain'")
        return {"accepted": False, "errors": msg, "note": "calling finish again will end the run anyway"}
    ctx.final = dict(a)
    ctx.done = True
    return {"accepted": True}


# ---------- backends ----------
def parse_text_tool_calls(text):
    """Tool calls written as text (tool-call parser not active): <tool_call>{json}</tool_call> or the
    <function=name><parameter=k>v</parameter></function> format of Qwen3-Coder style templates."""
    calls = []
    for m in re.finditer(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", text or "", re.S):
        try:
            d = json.loads(m.group(1))
            calls.append({"name": d["name"], "arguments": d.get("arguments", {})})
        except Exception:
            pass
    for m in re.finditer(r"<function=([\w-]+)>(.*?)</function>", text or "", re.S):
        args = {}
        for p in re.finditer(r"<parameter=([\w-]+)>\s*(.*?)\s*</parameter>", m.group(2), re.S):
            v = p.group(2)
            try:
                args[p.group(1)] = json.loads(v)
            except Exception:
                args[p.group(1)] = v
        calls.append({"name": m.group(1), "arguments": args})
    return [{"id": f"text_{i}", "type": "function", "function": {"name": c["name"], "arguments": json.dumps(c["arguments"])}}
            for i, c in enumerate(calls)]


class LLMBackend:
    name = "llm"

    def __init__(self, server):
        self.server = server
        self.tokens = 0

    def next(self, messages, ctx):
        self.server.start()
        try:
            msg, usage = self.server.chat(messages, openai_tools())
        except RuntimeError as e:
            if self.server.running():
                raise
            log(f"llm: server gone ({e}) - restarting once")
            self.server.start()
            msg, usage = self.server.chat(messages, openai_tools())
        self.tokens += usage.get("total_tokens", 0)
        calls = msg.get("tool_calls") or parse_text_tool_calls(msg.get("content"))
        return {"role": "assistant", "content": msg.get("content") or "", "tool_calls": calls}, usage


class ScriptBackend:
    """mock: scripted turns [{"content": "...", "calls": [{"name": .., "arguments": {..} or "raw text"}]}]."""
    name = "mock"

    def __init__(self, turns):
        self.turns = list(turns)
        self.n = 0

    def next(self, messages, ctx):
        if not self.turns:
            return {"role": "assistant", "content": "(script finished)", "tool_calls": []}, {}
        t = self.turns.pop(0)
        calls = []
        for c in t.get("calls", []):
            self.n += 1
            args = c.get("arguments", {})
            calls.append({"id": f"call_{self.n}", "type": "function",
                          "function": {"name": c["name"], "arguments": args if isinstance(args, str) else json.dumps(args)}})
        return {"role": "assistant", "content": t.get("content", ""), "tool_calls": calls}, {}


def replay_turns(log_file):
    """Tool calls of an earlier run, grouped by LLM turn, plus the recorded vision answers (for ScriptedVision)."""
    turns, vision = {}, []
    for line in Path(log_file).read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if r.get("type") != "tool" or r.get("turn", 0) < 0:     # turn -1 = harness safe-finish actions
            continue
        turns.setdefault(r["turn"], {"content": r.get("reasoning", ""), "calls": []})["calls"].append(
            {"name": r["tool"], "arguments": r["raw_arguments"]})
        rd = r.get("replay_data") or {}
        if "critique" in rd:
            vision.append(rd["critique"])
        if "vlm_texts" in rd:
            vision.append(rd["vlm_texts"])
    return [turns[k] for k in sorted(turns)], vision


class RulesBackend:
    """No LLM: fixed order and simple deterministic repairs. Used with SKIP_AGENT_LLM=1 and as a baseline."""
    name = "rules"

    def __init__(self):
        self.gen, self.n = None, 0

    def policy(self, ctx):
        def call(name, args=None, why=""):
            return name, args or {}, why
        r = yield call("parse_plan", {"reparse": True}, "start: read the input and build the plan")
        if r.get("needs_vlm") and (ctx.vision.available or ctx.gpu):
            yield call("read_text_vlm", {}, "scan/photo: read room names and m2 labels")
        yield call("get_plan_summary", {"detail": "rooms"}, "look at the plan")
        if ctx.vision.available and ctx.cfg["agent"]["critique_overlay"]:
            yield call("view_image", {"image": "overlay"}, "compare the plan reading with the source page")
        chk = yield call("run_checks", {}, "checks before furnishing")
        plan = ctx.plan()
        labelled = [x for x in plan["rooms"] if x.get("label_area_m2") and x["type"] != "balcony"]
        devs = [x["area_m2"] / x["label_area_m2"] for x in labelled]
        if len(devs) >= 2 and any(i["key"].startswith("area_label") for i in chk.get("issues", [])):
            med = float(np.median(devs))
            if all(abs(d / med - 1) < 0.03 for d in devs) and abs(med - 1) > 0.02:
                big = max(labelled, key=lambda x: x["label_area_m2"])
                yield call("patch_plan", {"ops": [{"op": "scale_from_label", "room_id": big["id"], "label_m2": big["label_area_m2"]}],
                                          "reason": f"all {len(devs)} labelled rooms are off by the same factor {med:.3f}: scale error"},
                           "consistent area error = wrong scale")
        yield call("relayout", {}, "furnish the rooms")
        yield call("choose_furniture", {}, "pick 3D models")
        yield call("build_scene", {}, "build the 3D scene")
        if ctx.gpu or ctx.cfg["agent"]["render_on_cpu"]:
            ren = yield call("render_preview", {}, "preview renders")
            if ctx.vision.available:
                for rr in (ren.get("renders") or [])[:ctx.cfg["agent"]["critique_renders"]]:
                    yield call("view_image", {"image": "render", "render_name": rr["name"]}, "look at a render")
        chk = yield call("run_checks", {}, "final checks")
        exp = yield call("export_blender", {}, "write and validate the deliverable")
        unc = [i["problem"] for i in chk.get("issues", []) if i["severity"] != "low"]
        ok = bool(exp.get("ok"))
        yield call("finish", {"status": "complete" if ok and not unc else "partial" if ok else "failed",
                              "summary": "Rules backend (no LLM): fixed stage order with deterministic repairs only.",
                              "uncertain": unc[:30] or []}, "done")
        yield call("finish", {"status": "complete" if ok and not unc else "partial" if ok else "failed",
                              "summary": "Rules backend (no LLM): fixed stage order with deterministic repairs only.",
                              "uncertain": unc[:30] or ["see run_checks"]}, "done (second call after the finish guard)")

    def next(self, messages, ctx):
        if self.gen is None:
            self.gen = self.policy(ctx)
            item = next(self.gen)
        else:
            try:
                item = self.gen.send(ctx.last_result or {})
            except StopIteration:
                return {"role": "assistant", "content": "", "tool_calls": []}, {}
        name, args, why = item
        self.n += 1
        return {"role": "assistant", "content": why, "tool_calls": [
            {"id": f"rules_{self.n}", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}]}, {}


# ---------- the loop ----------
class AgentLog:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, rec):
        rec = dict(rec, ts=time.strftime("%Y-%m-%d %H:%M:%S"))
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")


def execute(ctx, call, turn, reasoning, alog):
    fn = call.get("function", {})
    name, raw = fn.get("name", ""), fn.get("arguments") or "{}"
    ctx.steps += 1
    rec = {"type": "tool", "step": ctx.steps, "turn": turn, "tool": name, "raw_arguments": raw, "reasoning": reasoning}
    try:
        args = json.loads(raw) if isinstance(raw, str) else dict(raw)
        if not isinstance(args, dict):
            raise ValueError("arguments must be a JSON object")
    except Exception as e:
        res = {"error": f"arguments are not valid JSON: {e}"}
        alog.write(dict(rec, ok=False, result=res))
        return res
    if name not in TOOLS:
        res = {"error": f"unknown tool {name!r}; tools: {', '.join(TOOLS)}"}
        alog.write(dict(rec, args=args, ok=False, result=res))
        return res
    errs = schema_errors(TOOLS[name]["params"], args)
    if errs:
        res = {"error": "arguments do not match the schema", "details": errs[:10]}
        alog.write(dict(rec, args=args, ok=False, result=res))
        return res
    g = TOOLS[name]["gpu"]
    comp = g(ctx) if callable(g) else g
    decision = ctx.planner.before(comp) if ctx.planner else None
    sampler = GpuSampler().start() if ctx.gpu else None
    server_up = bool(ctx.server and ctx.server.running())
    t0, ok = time.time(), True
    ctx.replay_data = None
    try:
        res = TOOLS[name]["fn"](ctx, args)
        ok = "error" not in res
    except Exception as e:
        res = {"error": f"{type(e).__name__}: {str(e)[:1500]}", "trace": traceback.format_exc()[-1500:]}
        ok = False
    peak = sampler.stop() if sampler else {}
    alog.write(dict(rec, args=args, ok=ok, seconds=round(time.time() - t0, 1), vram_peak_mb=peak.get("vram_peak_mb"),
                    gpu_util_avg=peak.get("gpu_util_avg"), server_up=server_up, vram_decision=decision,
                    component=comp, result=res, replay_data=ctx.replay_data))
    return res


def compact(messages, max_chars):
    """Keep the conversation inside the context window: shorten old tool results (the full ones are in the log)."""
    total = sum(len(json.dumps(m, ensure_ascii=False)) for m in messages)
    tool_idx = [i for i, m in enumerate(messages) if m["role"] == "tool"]
    for i in tool_idx[:-6]:
        if total <= max_chars:
            break
        c = messages[i]["content"]
        if len(c) > 300:
            messages[i]["content"] = c[:280] + ' ... [shortened - full result in agent_log.jsonl]"'
            total -= len(c) - len(messages[i]["content"])


def run_agent(inp, job, cfg, backend, vision, server=None, planner=None):
    ctx = Ctx(inp, job, cfg, vision, server, planner)
    job = ctx.job
    (job / "00_input").mkdir(parents=True, exist_ok=True)
    (job / "logs").mkdir(exist_ok=True)
    if not (job / "00_input" / ctx.inp.name).exists():
        shutil.copy2(ctx.inp, job / "00_input" / ctx.inp.name)
    alog = AgentLog(job / "agent_log.jsonl")
    model = server.model if server else None
    alog.write({"type": "start", "input": str(ctx.inp), "backend": backend.name, "model": model, "tier": pod_tier(cfg),
                "limits": {k: cfg["agent"][k] for k in ("max_steps", "max_minutes", "max_cost_usd", "max_repairs_per_issue", "max_patches")},
                "vram_budget_mb": server.budget_mb() if server else None})
    ctx.run_config(cfg["agent"]["preview_profile"])
    a = cfg["agent"]
    messages = [{"role": "system", "content": SYSTEM.format(steps=a["max_steps"], minutes=a["max_minutes"])},
                {"role": "user", "content": f"Input file: {ctx.inp.name}. Furnishing style: {cfg['style']}. "
                                            f"GPU available: {ctx.gpu}. Vision model available: {vision.available}. Start with parse_plan."}]
    turn, idle = 0, 0
    ctx.last_result = None
    try:
        while not ctx.done:
            why = ctx.over_budget()
            if why:
                ctx.stop_reason = why
                break
            turn += 1
            t0 = time.time()
            msg, usage = backend.next(messages, ctx)
            calls = msg.get("tool_calls") or []
            alog.write({"type": "llm", "turn": turn, "content": (msg.get("content") or "")[:4000], "tool_calls": len(calls),
                        "seconds": round(time.time() - t0, 1), "tokens": usage.get("total_tokens")})
            messages.append({"role": "assistant", "content": msg.get("content") or "", **({"tool_calls": calls} if calls else {})})
            if not calls:
                idle += 1
                if idle >= 3:
                    ctx.stop_reason = "the model stopped calling tools"
                    break
                messages.append({"role": "user", "content": "Call a tool. When the deliverable is exported and checked, call finish."})
                continue
            idle = 0
            for c in calls:
                res = execute(ctx, c, turn, msg.get("content") or "", alog)
                ctx.last_result = res
                messages.append({"role": "tool", "tool_call_id": c.get("id", ""),
                                 "content": json.dumps(dict(res, budget=ctx.budget()), ensure_ascii=False, default=str)[:a["tool_result_chars"]]})
                if ctx.done or ctx.over_budget():
                    break
            compact(messages, int(cfg["llm"]["max_model_len"] * 2.5))
    except Exception as e:
        ctx.stop_reason = f"agent loop error: {type(e).__name__}: {e}"
        alog.write({"type": "error", "error": ctx.stop_reason, "trace": traceback.format_exc()[-3000:]})
    finally:
        if server:
            server.stop(wait_free=False)
    if not ctx.done:
        safe_finish(ctx, alog)
    rep = write_final_report(ctx)
    alog.write({"type": "end", "status": rep["status"], "steps": ctx.steps, "minutes": round(ctx.minutes(), 1),
                "cost_usd": ctx.cost(), "export_ok": bool(ctx.validation and ctx.validation.get("ok"))})
    return rep


def safe_finish(ctx, alog):
    """Budget or loop ended without finish: build and export deterministically if possible, then stop."""
    log(f"agent: stopping ({ctx.stop_reason}) - delivering what exists")
    for name in ("relayout", "choose_furniture", "build_scene", "export_blender"):
        stage = {"relayout": "layout", "choose_furniture": "assets", "build_scene": "scene", "export_blender": "export"}[name]
        if stage in ctx.fresh or ctx.plan() is None:
            continue
        try:
            res = TOOLS[name]["fn"](ctx, {})
            alog.write({"type": "tool", "step": ctx.steps, "turn": -1, "tool": name, "raw_arguments": "{}", "args": {},
                        "reasoning": "harness: safe finish", "ok": "error" not in res, "result": res})
        except Exception as e:
            alog.write({"type": "error", "error": f"safe finish {name}: {e}"})
            break
    iss = compute_issues(ctx)
    ctx.final = {"status": "partial" if ctx.validation and ctx.validation.get("ok") else "failed",
                 "summary": f"Stopped by the harness: {ctx.stop_reason}.",
                 "uncertain": [i["problem"] for i in iss if i["severity"] != "low"]}


def write_final_report(ctx):
    from .report import write_report
    from .llm_server import table
    job, cfg = ctx.job, ctx.cfg
    recs = [json.loads(l) for l in (job / "agent_log.jsonl").read_text(encoding="utf-8").splitlines()]
    tools = [r for r in recs if r["type"] == "tool"]
    fin = ctx.final or {}
    exported = bool(ctx.validation and ctx.validation.get("ok"))
    status = fin.get("status", "failed")
    if status == "complete" and not exported:
        status = "partial"
    iss = compute_issues(ctx)
    unc = list(fin.get("uncertain") or [])
    unc += [f"{i['problem']} (not resolved{', repair budget used up' if i['key'] in ctx.given_up else ''})"
            for i in iss if i["severity"] == "high" and not any(i["key"] in u or i["problem"] in u for u in unc)]
    stages = {}
    for r in tools:
        s = stages.setdefault(r["tool"], {"seconds": 0.0, "gpu": False, "status": "ok", "calls": 0, "vram_peak_mb": None})
        s["seconds"] = round(s["seconds"] + (r.get("seconds") or 0), 1)
        s["calls"] += 1
        s["gpu"] = s["gpu"] or bool(r.get("component"))
        if r.get("vram_peak_mb") is not None:
            s["vram_peak_mb"] = max(s["vram_peak_mb"] or 0, r["vram_peak_mb"])
        if not r.get("ok"):
            s["status"] = f"errors in {s.get('errors', 0) + 1} call(s)"
            s["errors"] = s.get("errors", 0) + 1
    meta = {"input": str(ctx.inp), "profile": cfg["agent"]["preview_profile"], "stages": stages,
            "warnings": [f"agent: {ctx.stop_reason}"] if ctx.stop_reason else [],
            "started": recs[0].get("ts"), "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
            "gpu": {"vram_peak_mb": max([r.get("vram_peak_mb") or 0 for r in tools] or [0]) or None}}
    rep = write_report(job, meta, cfg)
    patches = ctx.store.patches()
    vt = table(cfg)
    start = recs[0]
    L = [f"# Agent report - {job.name}", "",
         f"**Status: {status.upper()}**  |  backend {start.get('backend')}  |  model {start.get('model') or '-'}  |  "
         f"tier {start.get('tier')}  |  {ctx.steps} tool calls  |  {ctx.minutes():.1f} min  |  ${ctx.cost()}", "",
         fin.get("summary", ""), ""]
    if ctx.stop_reason:
        L += [f"> Stopped by the harness: {ctx.stop_reason}", ""]
    L += ["## Deliverable", "", f"- export validation: **{'PASSED' if exported else 'FAILED / not run'}**"]
    for c in (ctx.validation or {}).get("checks", []):
        L.append(f"  - {'ok  ' if c['ok'] else 'FAIL'} {c['name']}: {c.get('detail', '')}")
    L += [f"- files: {', '.join(sorted(p.name for p in (job / 'final').iterdir())) if (job / 'final').exists() else '-'}", ""]
    L += ["## What the agent did", "", "| # | Tool | Arguments | Result | s | VRAM peak MB | Server up |", "|---|---|---|---|---|---|---|"]
    for r in tools:
        res = r.get("result") or {}
        short = res.get("error") or ("accepted" if res.get("accepted") else "rejected: " + "; ".join(res.get("errors", []))[:120]
                                     if "accepted" in res else "ok")
        L.append(f"| {r['step']} | {r['tool']} | `{json.dumps(r.get('args', r.get('raw_arguments')), ensure_ascii=False)[:90]}` | "
                 f"{str(short)[:140]} | {r.get('seconds', '')} | {r.get('vram_peak_mb') or ''} | {'yes' if r.get('server_up') else ''} |")
    L += ["", "## Fixed", ""]
    L += [f"- patch {i + 1}: {p['reason']} -> `{json.dumps(p['ops'], ensure_ascii=False)[:200]}`" for i, p in enumerate(patches)]
    L += [f"- {x}" for x in fin.get("fixed", [])]
    if not patches and not fin.get("fixed"):
        L.append("- nothing was changed")
    L += ["", "## Still uncertain", ""] + ([f"- {u}" for u in unc] or ["- nothing reported"])
    L += ["", "## Last checks", ""] + ([f"- [{i['severity']}] {i['problem']}" + (" (stop_repairing)" if i.get("stop_repairing") else "")
                                         for i in (ctx.last_checks or iss)] or ["- no issues"])
    L += ["", "## GPU memory", "", f"Tier {vt['tier']}, GPU {vt['gpu_total_mb']} MB. Budget table below = ESTIMATES "
          f"(config vram_estimates_mb); measured peaks per tool are in the table above.", "",
          "| Component | Estimate MB | Note |", "|---|---|---|"] + [f"| {a} | {b} | {c} |" for a, b, c in vt["rows"]]
    if ctx.planner and ctx.planner.decisions:
        L += ["", "Scheduling decisions:"] + [f"- {d}" for d in ctx.planner.decisions]
    if ctx.server:
        L += ["", f"LLM server starts: {ctx.server.starts}, start-up time {ctx.server.start_seconds:.0f} s"]
    cov = job / "final/coverage.md"
    if cov.exists():
        L += ["", cov.read_text(encoding="utf-8").replace("# Furniture coverage", "## Furniture coverage", 1)]
    L += ["", "---", "", (job / "report.md").read_text(encoding="utf-8").replace("# Report", "## Pipeline report", 1)]
    (job / "final").mkdir(exist_ok=True)
    (job / "final/report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    out = {"status": status, "export_ok": exported, "steps": ctx.steps, "uncertain": unc, "report": str(job / "final/report.md"),
           "rooms": rep["rooms"]}
    jsave(job / "final/agent_summary.json", out)
    return out


# ---------- entry points ----------
def make(cfg, backend_name, script=None, replay_log=None):
    from .vision import NoVision, ScriptedVision
    server = planner = None
    if replay_log:
        turns, answers = replay_turns(replay_log)
        return ScriptBackend(turns), ScriptedVision(answers), None, None
    if backend_name == "mock":
        s = jload(script) if script else {"turns": []}
        return ScriptBackend(s["turns"]), ScriptedVision(s.get("vision", [])) if s.get("vision") else NoVision(), None, None
    if backend_name == "rules":
        return RulesBackend(), NoVision("rules backend: no LLM server"), None, None
    from .llm_server import LLMServer, VramPlanner
    from .vision import ServerVision
    server = LLMServer(cfg)
    planner = VramPlanner(cfg, server)
    return LLMBackend(server), ServerVision(server) if cfg["llm"]["vision"] else NoVision(), server, planner


def main():
    ap = argparse.ArgumentParser(description="agentic floor plan -> 3D apartment")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run")
    p.add_argument("input")
    p.add_argument("--job-dir")
    p.add_argument("--backend", choices=["llm", "rules", "mock"])
    p.add_argument("--script", help="mock backend: JSON file with scripted turns")
    p = sub.add_parser("replay")
    p.add_argument("log")
    p.add_argument("--job-dir")
    p.add_argument("--input")
    sub.add_parser("tools")
    a = ap.parse_args()
    cfg = load_config()
    if a.cmd == "tools":
        print(json.dumps(openai_tools(), indent=1))
        return
    if a.cmd == "replay":
        start = json.loads(Path(a.log).read_text(encoding="utf-8").splitlines()[0])
        inp = Path(a.input or start["input"])
        job = Path(a.job_dir) if a.job_dir else WS / "outputs" / f"{inp.stem}_replay_{time.strftime('%Y%m%d-%H%M%S')}"
        backend, vision, server, planner = make(cfg, None, replay_log=a.log)
        cfg["agent"]["max_steps"] = max(cfg["agent"]["max_steps"], sum(1 for _ in open(a.log)))
    else:
        inp = Path(a.input).resolve()
        job = Path(a.job_dir) if a.job_dir else WS / "outputs" / f"{inp.stem}_agent_{time.strftime('%Y%m%d-%H%M%S')}"
        backend, vision, server, planner = make(cfg, a.backend or cfg["agent"]["backend"], a.script)
    rep = run_agent(inp, job, cfg, backend, vision, server, planner)
    print(f"\nAGENT {rep['status'].upper()}: {job}\n  export validation: {'passed' if rep['export_ok'] else 'FAILED'}"
          f"\n  report: {rep['report']}")
    for u in rep["uncertain"][:10]:
        print(f"  uncertain: {u}")
    sys.exit(0 if rep["export_ok"] and rep["status"] != "failed" else 1)


if __name__ == "__main__":
    main()
__END_OF_AGENT_PY__
  cat > "$APP/deliver.py" <<'__END_OF_DELIVER_PY__'
"""Final deliverable of a job -> <job>/final/: apartment.blend (primary), apartment.glb, apartment.usdc,
previews/*.png, overlay.png, layout.png, coverage.md/json, validation.json (+ report.md written by the caller).
blender_export.py builds the files, validate_export.py reopens them headless; the job fails if a check fails."""
import shutil
from pathlib import Path
from .common import blender, jload, log
from .furniture import coverage


def collect(job):
    job, fin = Path(job), Path(job) / "final"
    (fin / "previews").mkdir(parents=True, exist_ok=True)
    for old in (fin / "previews").glob("*.png"):
        old.unlink()
    src = job / "06_polish" if list((job / "06_polish").glob("*.png")) else job / "05_render"
    for p in sorted(src.glob("*.png")):
        if not p.name.endswith("_depth.png"):
            shutil.copy2(p, fin / "previews" / p.name)
    for rel, name in (("02_plan/overlay.png", "overlay.png"), ("03_layout/layout.png", "layout.png")):
        if (job / rel).exists():
            shutil.copy2(job / rel, fin / name)


def deliver(job, cfg=None):
    """Export + validate. Returns the validation dict ({'ok': bool, 'checks': [...]})."""
    job = Path(job)
    (job / "final").mkdir(parents=True, exist_ok=True)
    (job / "final/validation.json").unlink(missing_ok=True)
    blender("blender_export.py", job)
    try:
        blender("validate_export.py", job)
    except RuntimeError as e:
        log(f"export validation failed: {e}")
    val = jload(job / "final/validation.json") or {"ok": False, "checks": [{"name": "validate_export.py", "ok": False,
                                                                           "detail": "no validation.json written"}]}
    collect(job)
    coverage(job)
    bad = [c["name"] for c in val.get("checks", []) if not c.get("ok")]
    log(f"deliver: validation {'OK' if val.get('ok') else 'FAILED: ' + ', '.join(bad)}")
    return val
__END_OF_DELIVER_PY__
  cat > "$APP/blender_export.py" <<'__END_OF_BLENDER_EXPORT_PY__'
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
__END_OF_BLENDER_EXPORT_PY__
  cat > "$APP/validate_export.py" <<'__END_OF_VALIDATE_EXPORT_PY__'
"""Reopen the deliverable headless and check it. Writes final/validation.json; exits with an error if any check
fails (the job then fails). Run: blender -b --factory-startup --python-exit-code 1 -P validate_export.py -- <job>
Checks: units (metric, scale 1.0, metres), collections, wall count, bounding box vs plan, Z extent, furniture
objects (one mesh per item, origin at base centre, unit scale), packed / missing textures, non-manifold wall
edges, real openings (a ray through every door/window centre passes, solid wall is hit), camera count, and that
the .glb and .usdc reopen with the same walls, furniture names and bounding box."""
import bpy, bmesh, json, math, os, re, sys
from pathlib import Path
from mathutils import Vector

ARGS = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
JOB = Path(ARGS[0])
FIN = JOB / "final"
PLAN = json.loads((JOB / "02_plan/plan.json").read_text(encoding="utf-8"))
QA = json.loads((JOB / "04_scene/furniture_qa.json").read_text(encoding="utf-8"))
CAMS = json.loads((JOB / "04_scene/cameras.json").read_text(encoding="utf-8"))["cameras"]
EXP = json.loads((FIN / "export.json").read_text(encoding="utf-8"))
OFF = Vector(EXP["offset_m"])
LAYOUT = json.loads((JOB / "03_layout/layout.json").read_text(encoding="utf-8"))
CHECKS = []


def check(name, ok, detail=""):
    CHECKS.append({"name": name, "ok": bool(ok), "detail": str(detail)[:400]})


def plan_extent():
    """Outer bounding box of the walls in plan coordinates (walls are rectangles of their thickness around a-b)."""
    xs, ys = [], []
    for w in PLAN["walls"]:
        (ax, ay), (bx, by) = w["a"], w["b"]
        L = math.hypot(bx - ax, by - ay) or 1.0
        nx, ny = -(by - ay) / L * w["thickness"] / 2, (bx - ax) / L * w["thickness"] / 2
        for px, py in ((ax + nx, ay + ny), (ax - nx, ay - ny), (bx + nx, by + ny), (bx - nx, by - ny)):
            xs.append(px)
            ys.append(py)
    return min(xs), min(ys), max(xs), max(ys)


def plan_footprint():
    """Plan lower-left/upper-right: outer wall faces and every room polygon (balconies too) = the origin rule."""
    x0, y0, x1, y1 = plan_extent()
    xs = [x0, x1] + [p[0] for r in PLAN["rooms"] for p in r["polygon"]]
    ys = [y0, y1] + [p[1] for r in PLAN["rooms"] for p in r["polygon"]]
    return min(xs), min(ys), max(xs), max(ys)


def bbox(objs):
    pts = [o.matrix_world @ Vector(c) for o in objs if o.type == "MESH" for c in o.bound_box]
    if not pts:
        return None
    return Vector([min(p[i] for p in pts) for i in range(3)]), Vector([max(p[i] for p in pts) for i in range(3)])


expected_items = [r["item_id"] for r in QA["items"] if r["used"] not in ("skipped", "removed")]
x0, y0, x1, y1 = plan_extent()
fx0, fy0, _, _ = plan_footprint()
H = PLAN["defaults"]["ceiling_height"]
check("origin_offset", abs(OFF.x + fx0) < 1e-4 and abs(OFF.y + fy0) < 1e-4,
      f"export offset ({OFF.x:.4f}, {OFF.y:.4f}); plan lower-left ({fx0:.4f}, {fy0:.4f})")

# ---------- 1. the .blend ----------
bpy.ops.wm.open_mainfile(filepath=str(FIN / "apartment.blend"))
SC = bpy.context.scene
us = SC.unit_settings
check("units", us.system == "METRIC" and abs(us.scale_length - 1.0) < 1e-9 and us.length_unit == "METERS",
      f"{us.system}, scale {us.scale_length}, {us.length_unit}")
names = {c.name for c in SC.collection.children}
need = {"Walls", "Floors_Ceilings", "Openings", "Lights", "Cameras"}
furn_colls = [c for c in SC.collection.children if c.name.startswith("Furniture.")]
check("collections", need <= names and (furn_colls or not expected_items), f"have {sorted(names)}")
wall_objs = [o for o in bpy.data.collections["Walls"].objects if o.type == "MESH" and re.fullmatch(r"w\d+", o.name)] \
    if "Walls" in names else []
check("wall_count", len(wall_objs) == len(PLAN["walls"]), f"{len(wall_objs)} wall objects, plan has {len(PLAN['walls'])}")
bb = bbox(wall_objs)
if bb:
    lo, hi = bb
    size_ok = abs((hi.x - lo.x) - (x1 - x0)) < 0.02 and abs((hi.y - lo.y) - (y1 - y0)) < 0.02
    at_ok = abs(lo.x - (x0 - fx0)) < 0.01 and abs(lo.y - (y0 - fy0)) < 0.01
    check("bbox_vs_plan", size_ok and at_ok, f"walls {hi.x - lo.x:.3f} x {hi.y - lo.y:.3f} m at ({lo.x:.3f}, {lo.y:.3f}); "
          f"plan {x1 - x0:.3f} x {y1 - y0:.3f} m at ({x0 - fx0:.3f}, {y0 - fy0:.3f})")
    everything = bbox([o for o in SC.objects if o.type == "MESH"])
    # door frames stand 1 cm proud of the wall and the balcony handrail is centred on the balcony edge: 5 cm slack
    check("geometry_from_origin", everything and everything[0].x > -0.05 and everything[0].y > -0.05,
          f"lowest corner of all meshes ({everything[0].x:.3f}, {everything[0].y:.3f})" if everything else "no meshes")
    check("z_up_height", abs(lo.z) < 0.01 and abs(hi.z - H) < 0.02, f"walls z {lo.z:.3f}..{hi.z:.3f}, ceiling {H}")
else:
    check("bbox_vs_plan", False, "no wall geometry")
furn = {o.name: o for c in furn_colls for o in c.all_objects}
missing = [i for i in expected_items if i not in furn]
check("furniture_objects", not missing, f"{len(furn)} objects; missing {missing[:8]}")
bad_origin = []
slot = {it["id"]: it for it in LAYOUT["items"]}
for name in expected_items:
    o = furn.get(name)
    if o is None or o.type != "MESH":
        if o is not None:
            bad_origin.append(f"{name}: {o.type} not a mesh")
        continue
    vs = [v.co for v in o.data.vertices]
    if not vs:
        bad_origin.append(f"{name}: empty")
        continue
    mn = Vector([min(v[i] for v in vs) for i in range(3)])
    mx = Vector([max(v[i] for v in vs) for i in range(3)])
    tol = max(0.02, 0.03 * max(mx.x - mn.x, mx.y - mn.y))
    it = slot.get(name)
    want = Vector((it["center"][0], it["center"][1], it.get("z", 0.0))) + OFF if it else o.location
    # base centre = the floor point under the footprint centre (wall-hung items such as a vanity start higher)
    if mn.z < -0.01 or abs((mn.x + mx.x) / 2) > tol or abs((mn.y + mx.y) / 2) > tol or (o.location - want).length > 0.01 \
            or any(abs(s - 1) > 1e-6 for s in o.scale) or o.children:
        bad_origin.append(f"{name}: lowest z {mn.z:.3f}, centre ({(mn.x + mx.x) / 2:.3f}, {(mn.y + mx.y) / 2:.3f}), "
                          f"location off {(o.location - want).length:.3f} m, scale {tuple(round(s, 3) for s in o.scale)}")
check("furniture_origin_base_centre", not bad_origin, "; ".join(bad_origin[:6]))
unpacked, lost = [], []
for img in bpy.data.images:
    if img.type in ("RENDER_RESULT", "COMPOSITING") or img.source in ("GENERATED", "VIEWER"):
        continue
    if not img.packed_file:
        unpacked.append(img.name)
        if not (img.filepath and os.path.exists(bpy.path.abspath(img.filepath))):
            lost.append(img.name)
check("textures_packed", not unpacked, f"{len(bpy.data.images)} images; not packed: {unpacked[:8]}")
check("textures_missing", not lost, f"missing: {lost[:8]}")
nm = {}
for o in wall_objs:
    bm = bmesh.new()
    bm.from_mesh(o.data)
    n = sum(1 for e in bm.edges if not e.is_manifold)
    bm.free()
    if n:
        nm[o.name] = n
check("walls_manifold", not nm, f"non-manifold edges: {nm}" if nm else f"{len(wall_objs)} walls watertight")
dg = bpy.context.evaluated_depsgraph_get()
wall_by_id = {o.name: o for o in wall_objs}
blocked, solid_miss = [], []


def ray_hits(o, p, d, dist):
    inv = o.matrix_world.inverted()
    q = inv @ p
    dl = (inv.to_3x3() @ d).normalized()
    return o.ray_cast(q, dl, distance=dist)[0]


for kind, lst in (("door", PLAN["doors"]), ("window", PLAN["windows"])):
    for op in lst:
        w = next((x for x in PLAN["walls"] if x["id"] == op.get("wall_id")), None)
        o = wall_by_id.get(op.get("wall_id"))
        if w is None or o is None:
            continue
        (ax, ay), (bx, by) = w["a"], w["b"]
        L = math.hypot(bx - ax, by - ay) or 1.0
        n = Vector((-(by - ay) / L, (bx - ax) / L, 0.0))
        z = (op.get("sill", 0.0) + op["head"]) / 2
        p = Vector((op["center"][0], op["center"][1], z)) + OFF
        if ray_hits(o, p - n * w["thickness"], n, 2 * w["thickness"]):
            blocked.append(op["id"])
for w in PLAN["walls"]:
    o = wall_by_id.get(w["id"])
    (ax, ay), (bx, by) = w["a"], w["b"]
    L = math.hypot(bx - ax, by - ay)
    if o is None or L < 0.3:
        continue
    u, n = Vector(((bx - ax) / L, (by - ay) / L, 0.0)), Vector((-(by - ay) / L, (bx - ax) / L, 0.0))
    ops = [op for op in PLAN["doors"] + PLAN["windows"] if op.get("wall_id") == w["id"]]
    for s in (0.12, L - 0.12, L / 2):   # a point of solid wall: no opening there
        if any(abs((Vector(op["center"] + [0]) - Vector((ax, ay, 0))).dot(u) - s) < op["width"] / 2 + 0.05 for op in ops):
            continue
        p = Vector((ax, ay, 1.2)) + u * s + OFF
        if not ray_hits(o, p - n * w["thickness"], n, 2 * w["thickness"]):
            solid_miss.append(f"{w['id']}@{s:.2f}")
        break
check("real_openings", not blocked, f"rays blocked at {blocked}" if blocked else
      f"{len(PLAN['doors']) + len(PLAN['windows'])} openings pass")
check("walls_solid", not solid_miss, f"no wall hit at {solid_miss[:8]}" if solid_miss else "solid parts hit")
cams = [o for o in SC.objects if o.type == "CAMERA"]
check("cameras", len(cams) == len(CAMS), f"{len(cams)} cameras, cameras.json has {len(CAMS)}")
blend_bb = bb

# ---------- 2. the .glb ----------
glb = FIN / "apartment.glb"
if glb.exists():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    try:
        bpy.ops.import_scene.gltf(filepath=str(glb))
        objs = list(bpy.data.objects)
        by = {o.name: o for o in objs}
        gw = [by[w["id"]] for w in PLAN["walls"] if w["id"] in by]
        check("glb_walls", len(gw) == len(PLAN["walls"]), f"{len(gw)} of {len(PLAN['walls'])} walls by name")
        gm = [i for i in expected_items if i not in by]
        check("glb_furniture", not gm, f"missing {gm[:8]}")
        gbb = bbox(gw)
        ok = gbb and blend_bb and all(abs(a - b) < 0.02 for a, b in zip(list(gbb[0]) + list(gbb[1]), list(blend_bb[0]) + list(blend_bb[1])))
        check("glb_bbox", ok, f"glb {tuple(round(x, 3) for x in gbb[0])}..{tuple(round(x, 3) for x in gbb[1])}" if gbb else "no walls")
        noimg = [i.name for i in bpy.data.images if i.source not in ("GENERATED", "VIEWER") and not i.packed_file and not i.has_data]
        check("glb_textures", not noimg, f"images without data: {noimg[:8]}")
    except Exception as e:
        check("glb_reopen", False, e)
else:
    check("glb_exists", False, "apartment.glb not written")

# ---------- 3. the .usdc ----------
usd = FIN / "apartment.usdc"
if usd.exists():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    try:
        bpy.ops.wm.usd_import(filepath=str(usd))
        meshes = [o for o in bpy.data.objects if o.type == "MESH"]
        uw = [o for o in meshes if re.fullmatch(r"w\d+(\.\d+)?", o.name)]
        check("usdc_reopen", len(uw) >= len(PLAN["walls"]), f"{len(meshes)} meshes, {len(uw)} walls")
    except Exception as e:
        check("usdc_reopen", False, e)
else:
    check("usdc_exists", False, "apartment.usdc not written")

ok = all(c["ok"] for c in CHECKS)
json.dump({"ok": ok, "checks": CHECKS}, open(FIN / "validation.json", "w"), indent=1, ensure_ascii=False)
for c in CHECKS:
    print(f"{'ok  ' if c['ok'] else 'FAIL'} {c['name']}: {c['detail']}")
print("VALIDATION_OK" if ok else "VALIDATION_FAILED")
if not ok:
    raise RuntimeError("export validation failed: " + ", ".join(c["name"] for c in CHECKS if not c["ok"]))
__END_OF_VALIDATE_EXPORT_PY__
  cat > "$APP/gen3d.py" <<'__END_OF_GEN3D_PY__'
"""Optional, OFF by default (setup.sh GEN3D=1): generate furniture models that the library lacks with an
open-weights image-to-3D model (TRELLIS.2-4B) on the 48 GB GPU. Results go through the same catalog, Blender
normalisation and QA as every other model. Nothing is downloaded unless setup.sh ran with GEN3D=1.
  python -m app.gen3d plan                         which furniture types have no model (would be generated)
  python -m app.gen3d run [--types sofa,desk] [--max N] [--force]
Runs ALONE on the GPU: stop any LLM server first (python -m app.llm_server stop); agent.sh does not call it.
Input image per type: assets/gen3d_inputs/<type>/*.png|jpg (your own photo/render of the piece), else an SDXL
product image made with the SDXL base that setup.sh downloads for the polish step.
LICENCES (checked for TRELLIS.2 commit 75fbf01, see CHANGELOG): TRELLIS.2 code and weights MIT; its GLB export
imports NVIDIA nvdiffrast (NVIDIA Source Code License: non-commercial, research/evaluation only); the image
encoder is Meta DINOv3 (DINOv3 License, gated download - needs HF_TOKEN at setup); background removal BiRefNet
(MIT). Treat generated models as evaluation-only until the nvdiffrast licence is cleared for your use."""
import argparse, json, subprocess, sys, time
from pathlib import Path
from .common import WS, gpu_used_mb, jload, jsave, load_config, log
from . import furniture

WORDS = {"sofa": "three-seat fabric sofa", "armchair": "upholstered armchair", "coffee_table": "low coffee table",
         "dining_table": "rectangular dining table", "desk": "writing desk", "nightstand": "bedside nightstand",
         "bistro_table": "small round bistro table", "wardrobe": "two-door wardrobe", "bookshelf": "tall bookshelf",
         "shoe_cabinet": "low shoe cabinet", "tv_unit": "low tv media console", "bed": "double bed with headboard",
         "floor_lamp": "floor lamp", "plant": "potted indoor plant", "chair": "dining chair", "fridge": "refrigerator",
         "washer": "front-loading washing machine", "toilet": "toilet", "bathtub": "bathtub",
         "vanity": "bathroom vanity with basin", "basin_small": "small wash basin", "shower": "shower tray with glass screen"}
TARGET_HEIGHT = {"sofa": 0.85, "armchair": 0.80, "coffee_table": 0.42, "dining_table": 0.75, "desk": 0.75,
                 "nightstand": 0.50, "bistro_table": 0.72, "wardrobe": 2.20, "bookshelf": 1.80, "shoe_cabinet": 1.00,
                 "tv_unit": 1.30, "bed": 1.05, "floor_lamp": 1.60, "plant": 1.10, "chair": 0.85, "fridge": 1.90,
                 "washer": 0.85, "toilet": 0.80, "bathtub": 0.58, "vanity": 0.85, "basin_small": 0.85, "shower": 2.00}


def missing_types(cfg):
    cov = furniture.load_catalog(cfg).get("coverage", {})
    return [t for t in furniture.TYPES if not sum(cov.get(t, {}).values())]


def input_image(cfg, typ, out_dir):
    """Your reference image if there is one, else an SDXL product image (GPU, main venv)."""
    user = Path(cfg["gen3d"]["inputs"]) / typ
    for f in sorted(user.glob("*")) if user.exists() else []:
        if f.suffix.lower() in (".png", ".jpg", ".jpeg"):
            return f, "user image"
    out = out_dir / f"{typ}.png"
    if out.exists():
        return out, "sdxl (cached)"
    import torch
    from diffusers import AutoencoderKL, StableDiffusionXLPipeline
    vae = AutoencoderKL.from_pretrained(cfg["models"]["vae"], torch_dtype=torch.float16)
    pipe = StableDiffusionXLPipeline.from_pretrained(cfg["models"]["sdxl"], vae=vae, variant="fp16",
                                                     torch_dtype=torch.float16).to("cuda")
    img = pipe(prompt=f"studio product photo of a single {WORDS.get(typ, typ)}, {cfg['style']}, three-quarter front "
                      "view, the whole object visible, plain white background, soft even light",
               negative_prompt="people, text, watermark, cropped, multiple objects, room, floor pattern",
               num_inference_steps=30, guidance_scale=6.0, width=1024, height=1024,
               generator=torch.Generator("cuda").manual_seed(7)).images[0]
    out_dir.mkdir(parents=True, exist_ok=True)
    img.save(out)
    del pipe
    torch.cuda.empty_cache()
    return out, "sdxl"


def run(types=None, max_items=None, force=False):
    cfg = load_config()
    g = cfg["gen3d"]
    py = Path(g["venv"]) / "bin/python"
    if not g["enabled"] or not py.exists():
        raise SystemExit("GEN3D is off: run setup.sh with GEN3D=1 (48 GB GPU, ~40 GB disk, licence notes in CHANGELOG)")
    used = gpu_used_mb() or 0
    if used > 2500:
        raise SystemExit(f"{used} MB of GPU memory in use - image-to-3D must run alone (stop the LLM server first)")
    todo = types or missing_types(cfg)
    todo = [t for t in todo if t in furniture.TYPES and t not in furniture.PARAMETRIC_ONLY][:max_items or g["max_items"]]
    lib = furniture.lib_dir(cfg) / "generated"
    jobs = []
    for t in todo:
        out = lib / t / f"trellis2_{t}.glb"
        if out.exists() and not force:
            log(f"gen3d: {t} exists ({out.name})")
            continue
        try:
            img, how = input_image(cfg, t, lib / "_inputs")
        except Exception as e:     # e.g. no image of yours and SDXL not downloaded (SKIP_POLISH_MODELS=1)
            log(f"gen3d: {t} skipped - no input image ({type(e).__name__}: {e})")
            continue
        jobs.append({"type": t, "image": str(img), "input": how, "out": str(out)})
    if not jobs:
        log("gen3d: nothing to generate")
        return furniture.build_catalog(cfg)
    spec = lib / "_jobs.json"
    jsave(spec, {"model": g["model"], "decimation_target": g["decimation_target"], "texture_size": g["texture_size"],
                 "jobs": jobs})
    env = {"PYTHONPATH": g["repo"], "HF_HUB_OFFLINE": "1", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
           "OPENCV_IO_ENABLE_OPENEXR": "1"}
    import os
    t0 = time.time()
    r = subprocess.run([str(py), str(Path(__file__).parent / "gen3d_worker.py"), str(spec)],
                       env=dict(os.environ, **env), cwd=g["repo"])
    log(f"gen3d worker finished with code {r.returncode} in {time.time() - t0:.0f}s")
    res = jload(spec.with_suffix(".result.json"), {"results": []})
    for x in res["results"]:
        if not x.get("ok"):
            log(f"gen3d {x['type']}: FAILED {x.get('error')}")
            continue
        try:
            info = furniture.inspect_gltf(x["out"])
        except Exception as e:
            log(f"gen3d {x['type']}: unreadable output {e}")
            continue
        h = info["dims_import"][2]
        jsave(Path(x["out"]).with_suffix(".json"), {
            "type": x["type"], "name": f"generated {x['type']}", "licence": "generated", "author": g["model"],
            "unit_scale": round(TARGET_HEIGHT.get(x["type"], 1.0) / max(h, 1e-6), 5), "front_axis": "-Y",
            "tags": [x["type"], "generated"],
            "notes": f"{time.strftime('%Y-%m-%d')} {g['model']} (MIT) from {x['input']} {Path(x['image']).name}; GLB "
                     "export uses nvdiffrast (NVIDIA non-commercial licence) - evaluation only; front axis assumed -Y "
                     "(UNVERIFIED for generated models, check the renders)",
            "seconds": x.get("seconds"), "vram_peak_mb": x.get("vram_peak_mb")})
        log(f"gen3d {x['type']}: ok ({info['triangles']} triangles, {x.get('seconds')} s)")
    return furniture.build_catalog(cfg)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("plan")
    p = sub.add_parser("run")
    p.add_argument("--types", default="")
    p.add_argument("--max", type=int)
    p.add_argument("--force", action="store_true")
    a = ap.parse_args()
    if a.cmd == "plan":
        print("furniture types without any library model:", ", ".join(missing_types(load_config())) or "none")
        return
    run([t for t in a.types.split(",") if t] or None, a.max, a.force)
    furniture.report()


if __name__ == "__main__":
    main()
__END_OF_GEN3D_PY__
  cat > "$APP/gen3d_worker.py" <<'__END_OF_GEN3D_WORKER_PY__'
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
__END_OF_GEN3D_WORKER_PY__
  cat > "$APP/requirements.lock" <<'__END_OF_LOCK__'
accelerate==1.15.0
annotated-doc==0.0.5
anyio==4.15.1
attrs==26.1.0
certifi==2026.7.22
cffi==2.1.1
charset-normalizer==3.5.1
click==8.5.0
contourpy==1.4.0
cryptography==50.0.1
cuda-bindings==13.4.3
cuda-pathfinder==1.8.2
cuda-toolkit==13.0.3.0
cycler==0.12.1
diffusers==0.40.0
ezdxf==1.4.4
filelock==4.0.7
fonttools==4.66.1
fsspec==2026.9.0
h11==0.16.0
hf-xet==1.6.0
httpcore==1.0.9
httpx==0.28.1
huggingface-hub==1.33.0
idna==3.20
imageio==2.38.0
importlib-metadata==9.0.1
jinja2==3.1.6
jsonschema==4.26.0
jsonschema-specifications==2025.9.1
kiwisolver==1.5.1
lazy-loader==0.6
markdown-it-py==4.2.0
markupsafe==3.0.3
matplotlib==3.11.2
mdurl==0.1.2
mpmath==1.3.0
networkx==3.7
numpy==2.5.3
nvidia-cublas==13.1.1.3
nvidia-cuda-cupti==13.0.85
nvidia-cuda-nvrtc==13.0.88
nvidia-cuda-runtime==13.0.96
nvidia-cudnn-cu13==9.24.0.43
nvidia-cufft==12.0.0.61
nvidia-cufile==1.15.1.6
nvidia-curand==10.4.0.35
nvidia-cusolver==12.0.4.66
nvidia-cusparse==12.6.3.3
nvidia-cusparselt-cu13==0.8.1
nvidia-nccl-cu13==2.30.7
nvidia-nvjitlink==13.4.92
nvidia-nvshmem-cu13==3.4.5
nvidia-nvtx==13.0.85
opencv-python-headless==4.14.0.94
packaging==26.3
pdfminer-six==20260107
pdfplumber==0.11.10
peft==0.21.1
pillow==12.3.0
psutil==7.2.2
pycparser==3.0
pygments==2.21.0
pyparsing==3.3.3
pypdfium2==5.13.0
python-dateutil==2.9.0.post0
pyyaml==6.0.3
referencing==0.37.0
regex==2026.9.29
reportlab==5.0.1
requests==2.34.2
rich==15.0.0
rpds-py==2026.6.3
safetensors==0.8.0
scikit-image==0.26.0
scipy==1.18.1
setuptools==84.0.0
shapely==2.1.2
shellingham==1.5.4
six==1.17.0
sympy==1.14.0
tifffile==2026.9.20
tokenizers==0.23.2
torch==2.14.0
torchvision==0.29.0
tqdm==4.70.1
transformers==5.17.0
triton==3.8.0
typer==0.27.2
typing-extensions==4.16.0
urllib3==2.8.0
zipp==4.1.0
__END_OF_LOCK__
  cat > "$APP/requirements-llm.lock" <<'__END_OF_LLM_LOCK__'
agent-detector==2.0.0
aiohappyeyeballs==2.7.1
aiohttp==3.14.3
aiosignal==1.4.0
annotated-doc==0.0.5
annotated-types==0.8.0
anthropic==1.9.0
anyio==4.15.1
apache-tvm-ffi==0.1.11
astor==0.8.1
attrs==26.1.0
blake3==1.0.10
cachetools==7.2.0
cbor2==6.1.4
certifi==2026.7.22
cffi==2.1.1
charset-normalizer==3.5.2
click==8.5.0
cloudpickle==3.1.2
compressed-tensors==0.17.0
cryptography==50.0.1
cuda-bindings==13.4.3
cuda-core==1.2.1
cuda-pathfinder==1.8.2
cuda-python==13.4.1
cuda-tile==1.6.0
cuda-toolkit==13.0.3.0
depyf==0.20.0
detect-installer==0.2.1
dill==0.4.1
dnspython==2.8.0
docstring-parser==0.18.0
einops==0.8.2
email-validator==2.3.0
fastapi==0.136.3
fastapi-cli==0.0.32
fastapi-cloud-cli==0.26.0
fastar==0.12.0
fastsafetensors==0.4.0
filelock==4.0.7
flashinfer-python==0.6.18.post1
frozenlist==1.8.0
fsspec==2026.9.0
googleapis-common-protos==1.75.5
grpcio==1.84.0
h11==0.16.0
hf-xet==1.6.0
httpcore==1.0.9
httpcore2==2.13.1
httptools==0.8.0
httpx==0.28.1
httpx2==2.13.1
huggingface-hub==1.33.0
humming-kernels==0.1.12
idna==3.20
ijson==3.5.1
instanttensor==0.2.0
interegular==0.3.3
jinja2==3.1.6
jiter==0.17.0
jmespath==1.1.0
jsonschema==4.26.0
jsonschema-specifications==2025.9.1
lark==1.2.2
llguidance==1.7.6
llvmlite==0.47.0
lm-format-enforcer==0.11.3
loguru==0.7.3
markdown-it-py==4.2.0
markupsafe==3.0.3
mcp==2.2.0
mcp-types==2.2.0
mdurl==0.1.2
mistral-common==1.12.0
ml-dtypes==0.6.0
model-hosting-container-standards==0.1.16
mpmath==1.3.0
msgspec==0.22.0
multidict==6.9.1
nccl4py==0.6.0
networkx==3.7
ninja==1.13.2
numba==0.65.0
numpy==2.3.5
nvidia-cublas==13.1.1.3
nvidia-cuda-cccl==13.3.4.3.1
nvidia-cuda-crt==13.4.92
nvidia-cuda-cupti==13.0.85
nvidia-cuda-nvcc==13.4.92
nvidia-cuda-nvdisasm==13.4.92
nvidia-cuda-nvrtc==13.0.88
nvidia-cuda-runtime==13.0.96
nvidia-cudnn-cu13==9.20.0.48
nvidia-cudnn-frontend==1.30.0
nvidia-cufft==12.0.0.61
nvidia-cufile==1.15.1.6
nvidia-curand==10.4.0.35
nvidia-cusolver==12.0.4.66
nvidia-cusparse==12.6.3.3
nvidia-cusparselt-cu13==0.8.1
nvidia-cutlass-dsl==4.7.1
nvidia-cutlass-dsl-libs-base==4.7.1
nvidia-cutlass-dsl-libs-core==4.7.1
nvidia-cutlass-dsl-libs-cu12==4.7.1
nvidia-cutlass-dsl-libs-cu13==4.7.1
nvidia-ml-py==13.615.71
nvidia-nccl-cu13==2.29.7
nvidia-nvjitlink==13.4.92
nvidia-nvshmem-cu13==3.4.5
nvidia-nvtx==13.0.85
nvidia-nvvm==13.4.92
nvtx==0.2.15
openai==3.22.1
openai-harmony==0.0.8
opencv-python-headless==5.0.0.93
opentelemetry-api==1.45.0
opentelemetry-exporter-http-transport==0.66b0
opentelemetry-exporter-otlp==1.45.0
opentelemetry-exporter-otlp-common==0.66b0
opentelemetry-exporter-otlp-proto-common==1.45.0
opentelemetry-exporter-otlp-proto-grpc==1.45.0
opentelemetry-exporter-otlp-proto-http==1.45.0
opentelemetry-proto==1.45.0
opentelemetry-sdk==1.45.0
opentelemetry-semantic-conventions==0.66b0
opentelemetry-semantic-conventions-ai==0.5.1
outlines-core==0.2.14
packaging==26.3
partial-json-parser==0.2.1.1.post7
pillow==12.3.0
prometheus-client==0.26.0
prometheus-fastapi-instrumentator==8.1.0
propcache==0.5.4
protobuf==7.36.2
psutil==7.2.2
py-cpuinfo==9.0.0
pybase64==1.5.0
pycountry==26.2.16
pycparser==3.0
pydantic==2.13.5
pydantic-core==2.46.5
pydantic-extra-types==2.11.1
pydantic-settings==2.15.0
pygments==2.21.0
pyjwt==2.15.1
pynvvideocodec==2.0.4
python-dotenv==1.2.3
python-json-logger==4.2.0
python-multipart==0.0.32
pyyaml==6.0.3
pyzmq==27.2.0
quack-kernels==0.6.5
referencing==0.37.0
regex==2026.9.29
requests==2.34.2
rich==15.0.0
rich-toolkit==0.20.5
rignore==0.8.1
rpds-py==2026.6.3
safetensors==0.8.0
sentencepiece==0.2.2
sentry-sdk==2.71.0
setproctitle==1.3.7
setuptools==80.10.2
shellingham==1.5.4
six==1.17.0
sniffio==1.3.1
sse-starlette==3.5.0
starlette==1.7.0
supervisor==4.3.0
sympy==1.14.0
tabulate==0.10.0
tiktoken==0.14.0
tilelang==0.1.12
tokenizers==0.23.2
tokenspeed-mla==0.1.8
tokenspeed-triton==3.8.10.post20260920
torch==2.13.0
torch-c-dlpack-ext==0.1.5
torchaudio==2.11.0
torchcodec==0.16.0
torchvision==0.28.0
tqdm==4.70.1
transformers==5.17.0
triton==3.7.1
truststore==0.10.4
typer==0.27.2
typing-extensions==4.16.0
typing-inspection==0.4.4
urllib3==2.8.0
uvicorn==0.54.0
uvloop==0.22.1
vllm==0.30.0
watchfiles==1.3.0
websockets==17.1
xgrammar==0.2.8
yarl==1.25.1
z3-solver==4.15.4.0
__END_OF_LLM_LOCK__
  cat > "$APP/requirements-gen3d.lock" <<'__END_OF_GEN3D_LOCK__'
certifi==2026.7.22
charset-normalizer==3.5.2
easydict==1.13
einops==0.8.2
filelock==4.0.7
fsspec==2026.9.0
hf-xet==1.6.0
huggingface-hub==0.36.2
idna==3.20
imageio==2.38.0
imageio-ffmpeg==0.6.0
jinja2==3.1.6
kornia==0.8.3
kornia-rs==0.2.0
markupsafe==3.0.3
mpmath==1.3.0
networkx==3.7
ninja==1.13.2
numpy==2.2.6
nvidia-cublas-cu12==12.4.5.8
nvidia-cuda-cupti-cu12==12.4.127
nvidia-cuda-nvrtc-cu12==12.4.127
nvidia-cuda-runtime-cu12==12.4.127
nvidia-cudnn-cu12==9.1.0.70
nvidia-cufft-cu12==11.2.1.3
nvidia-curand-cu12==10.3.5.147
nvidia-cusolver-cu12==11.6.1.9
nvidia-cusparse-cu12==12.3.1.170
nvidia-cusparselt-cu12==0.6.2
nvidia-nccl-cu12==2.21.5
nvidia-nvjitlink-cu12==12.4.127
nvidia-nvtx-cu12==12.4.127
opencv-python-headless==5.0.0.93
packaging==26.3
pandas==3.0.6
pillow==12.3.0
python-dateutil==2.9.0.post0
pyyaml==6.0.3
regex==2026.9.29
requests==2.34.2
safetensors==0.8.0
scipy==1.18.1
setuptools==84.0.0
six==1.17.0
sympy==1.13.1
timm==1.0.30
tokenizers==0.22.2
torch==2.6.0
torchvision==0.21.0
tqdm==4.70.1
transformers==4.57.6
trimesh==5.1.0
triton==3.2.0
typing-extensions==4.16.0
urllib3==2.8.0
wheel==0.48.0
zstandard==0.25.0
__END_OF_GEN3D_LOCK__
  cat > "$APP/env.sh" <<'__END_OF_ENV__'
# Environment for the pipeline. run.sh and start.sh load it: source __WS__/app/env.sh
export WS=__WS__ PIPE_WS=__WS__
export HF_HOME=__WS__/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export UV_CACHE_DIR=__WS__/cache/uv UV_PYTHON_INSTALL_DIR=__WS__/opt/python PIP_CACHE_DIR=__WS__/cache/pip XDG_CACHE_HOME=__WS__/cache
export PIPE_BLENDER=__WS__/opt/blender-__BLV__/blender
export LD_LIBRARY_PATH=__WS__/opt/libredwg/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export PATH=__WS__/venv/bin:__WS__/opt/libredwg/bin:$PATH
export PYTHONPATH=__WS__${PYTHONPATH:+:$PYTHONPATH}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1 VLLM_CACHE_ROOT=__WS__/cache/vllm
__END_OF_ENV__
  cat > "$WS/run.sh" <<'__END_OF_RUN__'
#!/usr/bin/env bash
# Run the whole pipeline on one plan (PDF, DWG, DXF, JPG/PNG photo). Outputs: __WS__/outputs/<job>/
#   ./run.sh __WS__/inputs/plan.pdf                  full quality (1920x1080, 3 views per main room)
#   ./run.sh __WS__/inputs/plan.pdf --profile quick  fast check (640x360, 1 view per room)
#   ./run.sh __WS__/inputs/plan.pdf --job-dir __WS__/outputs/<job> --from-stage render   resume a job
#   Stages: parse vlm plan layout scene render polish export report.  --no-polish skips the AI polish.
#   Deliverable: __WS__/outputs/<job>/final/ (apartment.blend/.glb/.usdc). Agentic run with repairs: ./agent.sh
set -Eeuo pipefail
trap 'echo "ERROR: run.sh stopped at line $LINENO: $BASH_COMMAND"' ERR
source __WS__/app/env.sh
if [[ $# -lt 1 ]]; then sed -n '2,7p' "$0"; exit 1; fi
mkdir -p __WS__/logs
cd __WS__
python -m app.main "$@" 2>&1 | tee -a "__WS__/logs/run_$(date +%Y%m%d_%H%M%S).log"
__END_OF_RUN__
  cat > "$WS/start.sh" <<'__END_OF_START__'
#!/usr/bin/env bash
# After a pod stop/restart the container disk is new (apt packages gone) but __WS__ is kept.
# This puts the small system packages back and checks GPU, PyTorch and Blender. Time: ~1-2 min.
set -Eeuo pipefail
trap 'echo "ERROR: start.sh stopped at line $LINENO: $BASH_COMMAND"' ERR
MISSING=""
for p in __APT__; do dpkg -s "$p" >/dev/null 2>&1 || MISSING="$MISSING $p"; done
# shellcheck disable=SC2086  # MISSING is a space-separated package list
if [[ -n "$MISSING" ]]; then apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends $MISSING; fi
source __WS__/app/env.sh
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
python -c "import torch; assert torch.cuda.is_available(), 'no GPU for PyTorch'; print('PyTorch OK', torch.__version__)"
"$PIPE_BLENDER" -b --factory-startup --python-expr "import bpy; print('Blender OK', bpy.app.version_string)" 2>/dev/null | grep "Blender OK"
rm -f __WS__/cache/llm_server.pid   # an LLM server from before the restart is gone; its PID may be reused now
if [[ -x __WS__/venv-llm/bin/vllm ]]; then
  __WS__/venv-llm/bin/python -c "import vllm, torch; print('vLLM OK', vllm.__version__, '| torch', torch.__version__, '| GPU', torch.cuda.is_available())"
else
  echo "vLLM not installed (SKIP_AGENT_LLM=1): agent.sh runs with --backend rules"
fi
echo "Ready. Put plans in __WS__/inputs and run:  cd __WS__ && ./run.sh __WS__/inputs/<plan file>"
echo "Agentic run (LLM checks and repairs):         cd __WS__ && ./agent.sh __WS__/inputs/<plan file>"
__END_OF_START__
  cat > "$WS/agent.sh" <<'__END_OF_AGENT_SH__'
#!/usr/bin/env bash
# Agentic run: a local open-weights LLM (vLLM, started and stopped automatically) drives the pipeline, checks the
# results, repairs problems with validated patches and delivers __WS__/outputs/<job>/final/
# (apartment.blend primary, apartment.glb, apartment.usdc, previews/, overlay.png, report.md).
#   ./agent.sh __WS__/inputs/plan.pdf                        LLM agent (settings: config.json llm.* and agent.*)
#   ./agent.sh __WS__/inputs/plan.pdf --backend rules        no LLM: fixed order + deterministic repairs
#   ./agent.sh --replay __WS__/outputs/<job>/agent_log.jsonl  replay the same tool calls into a new job folder
set -Eeuo pipefail
trap 'echo "ERROR: agent.sh stopped at line $LINENO: $BASH_COMMAND"' ERR
source __WS__/app/env.sh
if [[ $# -lt 1 ]]; then sed -n '2,7p' "$0"; exit 1; fi
mkdir -p __WS__/logs
cd __WS__
cleanup() { python -m app.llm_server stop >/dev/null 2>&1 || true; }   # never leave a server holding the GPU
trap cleanup EXIT
LOGF="__WS__/logs/agent_$(date +%Y%m%d_%H%M%S).log"
if [[ "$1" == "--replay" ]]; then
  shift
  python -m app.agent replay "$@" 2>&1 | tee -a "$LOGF"
else
  python -m app.agent run "$@" 2>&1 | tee -a "$LOGF"
fi
__END_OF_AGENT_SH__
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
  HF_HUB_OFFLINE=0 "$PY" - "$VLM_MODEL" "$SKIP_POLISH_MODELS" "$APP/models.lock.json" <<'__END_OF_DL__'
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
__END_OF_DL__
  touch "$MODEL_MARK" "$STATE/models.done"
fi
"$PY" - "$WS/config.json" "$GPU_PRICE_PER_HOUR" "$VLM_MODEL" "$SKIP_POLISH_MODELS" "$POD_TIER" "$AGENT_MODEL" \
  "$SKIP_AGENT_LLM" "$GEN3D" <<'__END_OF_CFG__'
import json, os, sys
p, price, vlm, skip = sys.argv[1], float(sys.argv[2]), sys.argv[3], sys.argv[4] == "1"
tier, agent_model, skip_llm, gen3d = sys.argv[5], sys.argv[6], sys.argv[7] == "1", sys.argv[8] == "1"
cfg = json.load(open(p)) if os.path.exists(p) else {}
cfg.update(gpu_price_per_hour=price, vlm_model=vlm, pod_tier=tier)
cfg.setdefault("polish", {})["enabled"] = not skip and cfg.get("polish", {}).get("enabled", True)
cfg.setdefault("llm", {})["model"] = agent_model
cfg.setdefault("agent", {})["backend"] = "rules" if skip_llm else "llm"
cfg.setdefault("gen3d", {})["enabled"] = gen3d
json.dump(cfg, open(p, "w"), indent=1)
print("settings file:", p, cfg)
__END_OF_CFG__

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
    HF_HUB_OFFLINE=0 "$PY" - "$AGENT_MODEL" "$APP/models.lock.json" <<'__END_OF_AGENT_DL__'
import json, os, sys
from huggingface_hub import snapshot_download
repo, lock_file = sys.argv[1], sys.argv[2]
path = snapshot_download(repo)
lock = json.load(open(lock_file)) if os.path.exists(lock_file) else {}
lock[repo] = path.rstrip("/").split("/")[-1]      # commit hash = the exact model version used
json.dump(lock, open(lock_file, "w"), indent=1)
print(f"downloaded {repo} @ {lock[repo]}", flush=True)
__END_OF_AGENT_DL__
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
    cat > "$WS/cache/gen3d_build.sh" <<'__END_OF_GEN3D_BUILD__'
set -Eeuo pipefail
if command -v nvcc >/dev/null && nvcc --version | grep -q "release 12\."; then
  CUDA_HOME=$(dirname "$(dirname "$(command -v nvcc)")")
else
  MM="$WS/opt/micromamba-$MICROMAMBA_VERSION/micromamba"
  if [[ ! -x "$MM" ]]; then
    mkdir -p "$(dirname "$MM")"
    U="https://github.com/mamba-org/micromamba-releases/releases/download/$MICROMAMBA_VERSION/micromamba-linux-64"
    curl -fL --retry 3 -o "$MM" "$U"
    curl -fsSL --retry 3 -o "$MM.sha256" "$U.sha256"
    echo "$(tr -dc 0-9a-f < "$MM.sha256" | head -c 64)  $MM" | sha256sum -c -
    chmod +x "$MM"
  fi
  CUDA_HOME="$WS/opt/cuda-12.4"
  [[ -x "$CUDA_HOME/bin/nvcc" ]] || "$MM" create -y -p "$CUDA_HOME" -r "$WS/opt/mamba-root" -c nvidia/label/cuda-12.4.1 cuda-toolkit
fi
MAX_JOBS=$(nproc)
export CUDA_HOME PATH="$CUDA_HOME/bin:$PATH" MAX_JOBS
TORCH_CUDA_ARCH_LIST="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)"
export TORCH_CUDA_ARCH_LIST
R="$WS/opt/TRELLIS.2"
X="$WS/opt/gen3d_ext"
[[ -d "$R/.git" ]] || git clone https://github.com/microsoft/TRELLIS.2.git "$R"
git -C "$R" checkout -q "$TRELLIS2_COMMIT"
git -C "$R" submodule update --init --recursive
mkdir -p "$X"
clone() { [[ -d "$X/$1/.git" ]] || git clone "$2" "$X/$1"; git -C "$X/$1" checkout -q "$3"; git -C "$X/$1" submodule update --init --recursive; }
clone nvdiffrast https://github.com/NVlabs/nvdiffrast.git "$NVDIFFRAST_TAG"
clone nvdiffrec https://github.com/JeffreyXiang/nvdiffrec.git "$NVDIFFREC_COMMIT"
clone CuMesh https://github.com/JeffreyXiang/CuMesh.git "$CUMESH_COMMIT"
clone FlexGEMM https://github.com/JeffreyXiang/FlexGEMM.git "$FLEXGEMM_COMMIT"
"$UV" pip install --python "$GEN_PY" "utils3d @ git+https://github.com/EasternJournalist/utils3d.git@$UTILS3D_COMMIT"
"$UV" pip install --python "$GEN_PY" --no-build-isolation "flash-attn==$FLASH_ATTN_VERSION"
for e in nvdiffrast nvdiffrec CuMesh FlexGEMM; do "$UV" pip install --python "$GEN_PY" --no-build-isolation "$X/$e"; done
"$UV" pip install --python "$GEN_PY" --no-build-isolation "$R/o-voxel"
PYTHONPATH="$R" "$GEN_PY" -c "import trellis2, o_voxel, nvdiffrast.torch, cumesh, flex_gemm; print('GEN3D imports OK')"
__END_OF_GEN3D_BUILD__
    # the build runs in its own bash process, so its set -e is active even inside this if-condition
    if ( export WS UV GEN_PY MICROMAMBA_VERSION TRELLIS2_COMMIT NVDIFFRAST_TAG NVDIFFREC_COMMIT CUMESH_COMMIT \
                FLEXGEMM_COMMIT UTILS3D_COMMIT FLASH_ATTN_VERSION
         bash "$WS/cache/gen3d_build.sh" ) > "$WS/logs/gen3d_build.log" 2>&1 \
       && HF_HUB_OFFLINE=0 "$PY" - <<'__END_OF_GEN3D_DL__'
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
__END_OF_GEN3D_DL__
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
