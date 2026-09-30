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

    def server_env(self):
        """Environment of the vLLM process. The vLLM venv's bin/ goes first on PATH (its ninja and nvcc for any
        just-in-time kernel build), and FlashInfer's top-k/top-p sampler is off: it compiles a CUDA kernel at the
        first warm-up, which failed on the pod (no ninja on PATH, no CUDA compiler in the image). vLLM then uses its
        PyTorch sampler - the agent samples at temperature 0-0.2, so nothing is lost."""
        venv_bin = str(Path(self.cfg["llm"]["venv"]) / "bin")
        env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", VLLM_CACHE_ROOT=str(WS / "cache/vllm"),
                   VLLM_NO_USAGE_STATS="1", DO_NOT_TRACK="1", VLLM_USE_FLASHINFER_SAMPLER="0",
                   FLASHINFER_WORKSPACE_BASE=str(WS / "cache"),
                   PATH=venv_bin + os.pathsep + os.environ.get("PATH", ""))
        env.pop("VIRTUAL_ENV", None)
        return env

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
        env = self.server_env()
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
