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
