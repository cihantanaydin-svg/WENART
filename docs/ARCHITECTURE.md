# Architecture - current state (setup.sh 2.0.3)

This describes the code as it is at commit `3881587` (`setup.sh` 2.0.3, sha256 `75431821…c043051`,
8 354 lines). It was written in Phase 0 by reading every module. Nothing was changed.

How the facts were checked:

* **Code:** the 34 files written by `write_code()` were extracted from `setup.sh` and compared
  byte for byte with the development copies that were read.
* **Pinned packages:** checked against the pinned versions, not from memory:
  * ezdxf 1.4.4: installed and probed.
  * vLLM 0.30.0: the PyPI sdist, sha256 checked against PyPI, source read.
  * Blender 5.2.2: the `bpy` 5.2.2 module, probed.
  * diffusers 0.40.0: the PyPI wheel, `__init__` read.
* **No GPU here:** this machine has no GPU and no `nvidia-smi`. Nothing in this document was run on
  a GPU.

## 1. Delivery model

One self-contained bash file, `setup.sh`, is uploaded to a Runpod pod and run (`bash /workspace/setup.sh`).

* It installs pinned tools into `/workspace`.
* It writes the whole pipeline as quoted heredocs to `/workspace/app`.
* It downloads models and CC0 assets, then runs a smoke test.
* `--code-only` runs only `write_code()`, about 1 s.
* Re-runs are cheap: every expensive step has a marker in `/workspace/.setup_state/`.

### 1.1 Layout of `setup.sh`

| Lines | Section |
|---|---|
| 1-17 | header, `SETUP_VERSION="2.0.3"`, usage, time/disk estimates |
| 19-37 | `set -Eeuo pipefail`, user switches: `WS`, `GPU_PRICE_PER_HOUR`, `VLM_MODEL`, `SKIP_SMOKE_TEST`, `SKIP_POLISH_MODELS`, `POD_TIER`, `AGENT_MODEL`, `SKIP_AGENT_LLM`, `GEN3D`, `MIN_FREE_GB`. `HF_TOKEN` comes only from the environment |
| 39-56 | pinned versions: uv 0.12.20, Python 3.12.12, Blender 5.2.2, LibreDWG 0.14, torch 2.14.0 / torchvision 0.29.0, vLLM 0.30.0, micromamba 2.9.0-0; TRELLIS.2 and its extension commits, flash-attn 2.7.3 (GEN3D only) |
| 58-77 | log file + `tee`, ERR trap, folders, paths (`APP`, `PY`, `UV`, `BL_DIR`, `LDWG`, `LLM_PY`, `GEN_PY`), cache env vars, `APT_PKGS` |
| 80-7999 | `write_code()`: 34 heredocs (table 1.2), then `sed` substitution of `__WS__`, `__BLV__`, `__APT__` in env.sh/run.sh/start.sh/agent.sh, then `chmod +x` |
| 8001-8004 | `--code-only` exit |
| 8006-8354 | install steps 1/13 … 13/13 (table 1.3) |

### 1.2 Heredocs (39 in total)

34 are inside `write_code()`: 27 Python modules + 3 lock files + 4 shell files. The other 5 are
inline installer scripts. Every delimiter is quoted (`<<'__END_OF_…__'`), so no shell expansion
happens inside the code.

| Line | Target | Delimiter | Lines of code |
|---|---|---|---|
| 82 | app/`__init__.py` | `__END_OF___INIT___PY__` | 1 |
| 85 | app/common.py | `__END_OF_COMMON_PY__` | 204 |
| 291 | app/textnorm.py | `__END_OF_TEXTNORM_PY__` | 106 |
| 399 | app/parse.py | `__END_OF_PARSE_PY__` | 535 |
| 936 | app/vlm.py | `__END_OF_VLM_PY__` | 107 |
| 1045 | app/plan.py | `__END_OF_PLAN_PY__` | 601 |
| 1648 | app/layout.py | `__END_OF_LAYOUT_PY__` | 428 |
| 2078 | app/blender_scene.py | `__END_OF_BLENDER_SCENE_PY__` | 1012 |
| 3092 | app/blender_render.py | `__END_OF_BLENDER_RENDER_PY__` | 88 |
| 3182 | app/device_check.py | `__END_OF_DEVICE_CHECK_PY__` | 40 |
| 3224 | app/polish.py | `__END_OF_POLISH_PY__` | 80 |
| 3306 | app/report.py | `__END_OF_REPORT_PY__` | 66 |
| 3374 | app/main.py | `__END_OF_MAIN_PY__` | 133 |
| 3509 | app/assets_fetch.py | `__END_OF_ASSETS_FETCH_PY__` | 82 |
| 3593 | app/testplans.py | `__END_OF_TESTPLANS_PY__` | 540 |
| 4135 | app/evaluate.py | `__END_OF_EVALUATE_PY__` | 158 |
| 4295 | app/wallseg.py | `__END_OF_WALLSEG_PY__` | 571 |
| 4868 | app/patches.py | `__END_OF_PATCHES_PY__` | 324 |
| 5194 | app/furniture.py | `__END_OF_FURNITURE_PY__` | 539 |
| 5735 | app/llm_server.py | `__END_OF_LLM_SERVER_PY__` | 235 |
| 5972 | app/vision.py | `__END_OF_VISION_PY__` | 129 |
| 6103 | app/agent.py | `__END_OF_AGENT_PY__` | 922 |
| 7027 | app/deliver.py | `__END_OF_DELIVER_PY__` | 40 |
| 7069 | app/blender_export.py | `__END_OF_BLENDER_EXPORT_PY__` | 77 |
| 7148 | app/validate_export.py | `__END_OF_VALIDATE_EXPORT_PY__` | 223 |
| 7373 | app/gen3d.py | `__END_OF_GEN3D_PY__` | 139 |
| 7514 | app/gen3d_worker.py | `__END_OF_GEN3D_WORKER_PY__` | 52 |
| 7568 | app/requirements.lock | `__END_OF_LOCK__` | main venv, 92 pins |
| 7662 | app/requirements-llm.lock | `__END_OF_LLM_LOCK__` | vLLM venv, 198 pins |
| 7862 | app/requirements-gen3d.lock | `__END_OF_GEN3D_LOCK__` | GEN3D venv, 57 pins |
| 7921 | app/env.sh | `__END_OF_ENV__` | template (`__WS__`, `__BLV__`) |
| 7933 | `$WS`/run.sh | `__END_OF_RUN__` | template |
| 7949 | `$WS`/start.sh | `__END_OF_START__` | template (`__APT__`) |
| 7972 | `$WS`/agent.sh | `__END_OF_AGENT_SH__` | template |
| 8135 | stdin of `"$PY" -` (step 8) | `__END_OF_DL__` | HF model download, writes `app/models.lock.json` |
| 8158 | stdin of `"$PY" -` (step 8) | `__END_OF_CFG__` | writes/updates `/workspace/config.json` |
| 8204 | stdin of `"$PY" -` (step 10) | `__END_OF_AGENT_DL__` | separate agent model download, appends to models.lock.json |
| 8245 | `$WS/cache/gen3d_build.sh` (step 12) | `__END_OF_GEN3D_BUILD__` | GEN3D CUDA extension build (4-space indent) |
| 8287 | stdin of `"$PY" -` (step 12) | `__END_OF_GEN3D_DL__` | TRELLIS.2 weight download |

`models.lock.json` is **written** at install time. It records whatever HF commit was downloaded; it is
not shipped. Nothing passes a `revision=` today.

### 1.3 Install steps

| Step | What | Marker / guard |
|---|---|---|
| 1 | GPU/driver/disk checks. The driver version picks the torch build: `TORCH_CUDA` is cu130, or cu128 when the driver is below 580. Also resolves the VLM model, pod tier, agent model and GEN3D gating | `MIN_FREE_GB` unless `models.done` |
| 2 | apt packages (container disk; `start.sh` repeats this after a restart) | dpkg check |
| 3 | uv (`tar --no-same-owner`), Python, `/workspace/venv` | exists |
| 4 | `write_code` | always |
| 5 | main lock, `--only-binary :all:` (cu128: torch from the PyTorch index) | `pip_<locksum>_<cuda>.done` |
| 6 | Blender 5.2.2 tarball, sha256 from download.blender.org | binary exists |
| 7 | LibreDWG 0.14 built from source (optional; failure is a warning) | binary exists |
| 8 | HF models (VLM, and SDXL + ControlNet depth + VAE unless skipped) + config writer | `models_<vlm>_<skip>.done` |
| 9 | CC0 assets: ambientCG textures, Poly Haven HDRIs + plant/lamp models | `assets.done` |
| 10 | `/workspace/venv-llm` (vLLM lock; cu129 wheel on old drivers) + separate agent model | `llm_<locksum>_<cuda>.done`, `agent_model_<id>.done` |
| 11 | Poly Haven furniture fetch + `furniture catalog` | `furniture_v1.done` |
| 12 | GEN3D (only `GEN3D=1`): venv, CUDA build script, weights; on failure GEN3D is switched off | `gen3d_<commit>_<locksum>.done` |
| 13 | Blender device check → `app/device.json`; `llm_server test` (on failure `agent.backend` becomes `rules`); smoke test `testplans quick` | - |

### 1.4 Runtime layout on the pod

```
/workspace/
  setup.sh  run.sh  start.sh  agent.sh  config.json      (config.json: user overrides of common.DEFAULTS)
  app/                 pipeline package (written by setup.sh), device.json, models.lock.json
  venv/                main Python 3.12 venv (torch 2.14 cu130/cu128, opencv, ezdxf 1.4.4, diffusers 0.40.0, ...)
  venv-llm/            vLLM 0.30.0 (torch 2.13.0)          venv-gen3d/  TRELLIS.2 (GEN3D=1 only)
  opt/                 uv, python, blender-5.2.2, libredwg, (TRELLIS.2, cuda-12.4, micromamba)
  hf/                  Hugging Face cache (HF_HUB_OFFLINE=1 at runtime, set by env.sh)
  assets/              materials/ (ambientCG), hdri/, models/ (legacy plant/lamp), furniture/ (Poly Haven +
                       catalog.json + LICENSES.txt), models_user/ (drop-in), gen3d_inputs/, LICENSES.txt
  inputs/  outputs/<job>/  logs/  cache/ (uv, pip, vllm, llm_server.pid)  .setup_state/
```

## 2. The deterministic pipeline (`run.sh` → `app.main`)

`python -m app.main <file> [--profile full|quick] [--job-dir] [--from-stage] [--to-stage] [--no-polish] [--redo-vlm] [--wallseg config|on|off]`

The stages are `parse, vlm, plan, layout, scene, render, polish, export, report`.

* The GPU stages (vlm, render, polish) run as separate processes, so their GPU memory is freed when
  they end.
* `vlm` and `polish` are optional: if they fail, the run warns and carries on.
* Any other failure stops the run.
* Each stage writes seconds, GPU use, status and VRAM peak into `meta.json`.

### 2.1 Job folder (the contract the brief says must be kept)

```
outputs/<job>/
  00_input/        copy of the input (+ DWG→DXF conversion output)
  01_parse/        extract.json, wall_mask.png, strokes.png, page.png, [vlm_texts.json, vlm_raw.txt,
                   wallseg.json, wallseg_pred.jpg, wall_mask_classic.png]
  02_plan/         plan.json, overlay.png, [plan_raw.json, patches.json - agent/patches only]
  03_layout/       layout.json, layout.png, assets.json, [asset_overrides.json]
  04_scene/        scene.blend, scene.glb, cameras.json, furniture_qa.json
  05_render/       <camera>.png, <camera>_depth.npy (320x180 ray-cast), render.json
  06_polish/       <camera>.png, <camera>_compare.jpg, polish.json
  final/           apartment.blend (primary), apartment.glb, apartment.usdc, export.json, validation.json,
                   previews/*.png, overlay.png, layout.png, coverage.md/.json, report.md, [agent_summary.json]
  logs/  meta.json  run_config.json  report.md  report.json  [agent_log.jsonl]
```

### 2.2 Stage by stage

| Stage | Module (key functions) | Reads | Writes | Notes |
|---|---|---|---|---|
| parse | `parse.py`: `detect`, `dwg_to_dxf`, `parse_dxf`, `parse_pdf`, `parse_raster`, `run_parse` | input | `01_parse/*` | Everything becomes one pixel frame: wall mask, stroke mask, background, texts in pixels, scale hints, h/v wall lines. DXF sets `m_per_px` from `$INSUNITS` or a guess; DWG goes through LibreDWG `--as r2000` (entity-count check, ODA fallback) |
| (wallseg) | `wallseg.py predict` | page.png, classic mask | replaces wall_mask.png if the ratio to the classic mask is within 0.33–3 | trained U-Net, off by default; only class 1 (wall) is used; classes 2/3 (door/window) are ignored |
| vlm | `vlm.py` (transformers, own process) | page.png | vlm_texts.json | only when `needs_vlm` (scans/photos, PDFs without text); page + 2×2 tiles, JSON bbox replies |
| plan | `plan.py`: `wall_runs` (h/v only), `trim_vertical`, `find_openings`, `arc_score`, `comp_polygon`, `solve_scale`, `run_plan`, `overlay` | 01_parse | plan.json, overlay.png | geometry at a provisional scale, then a second pass with the solved scale; scale methods: dxf_units, area_labels (weighted median), scale_note, door_width_0.90m, wall_thickness_guess; opening classification from room adjacency + swing arcs |
| layout | `layout.py`: `Room`, `against_wall`, `free_spot`, `kitchen_run`, `furnish` (per-type templates), `verify` | plan.json | layout.json, layout.png | `SIZES` table hard-coded; checks: overlaps, outside_room, blocking_doors, tall_in_front_of_windows |
| (assets) | `furniture.select_assets` (runs inside the scene stage) | layout, catalog, overrides | assets.json | style words from `cfg.style` or an override |
| scene | `blender_scene.py` (Blender, CPU) | plan, layout, assets, run_config | 04_scene/* | watertight walls with real openings, `SPEC` materials hard-coded (ambientCG when present), library model import + QA with parametric fallback, lights, cameras scored by visible furniture, collections |
| render | `blender_render.py` (Blender, GPU: OptiX → CUDA → CPU) | scene.blend, profile | PNGs, depth .npy, render.json | 8-bit PNG only; depth is a 320×180 Python ray-cast per camera |
| polish | `polish.py` (SDXL + ControlNet depth img2img) | renders, depth | 06_polish | fixed strength (0.25); Canny edge-F1 gate (0.45) keeps the raw render when lines moved |
| export | `deliver.py` → `blender_export.py` + `validate_export.py` | scene | final/* | origin at the plan footprint's lower-left; 20+ headless checks; the job fails if validation fails |
| report | `report.py` | meta, plan, layout, … | report.md/json, final/report.md | costs from `gpu_price_per_hour` |

### 2.3 Data contracts (JSON)

* **`extract.json`** (parse → plan): `kind` (dxf/dwg/pdf_vector/pdf_scan/photo), `m_per_px` (DXF only),
  `texts[{text,x,y,source}]`, `hints[{method:scale_note,m_per_px}]`, `hlines`/`vlines`, `dims`, `dpi`,
  `page`, `needs_vlm`, `warnings`.
* **`plan.json`** (`schema_version` "1.0", metres, y up):
  * `source`, `scale{m_per_px,method,confidence,checks[],agent_factor?}`, `defaults{…heights}`
  * `walls[{id,a,b,thickness,height,exterior,source}]`
  * `doors[{id,wall_id,center,width,head,swing{hinge,into},kind: hinged|opening,rooms}]`
  * `windows[{id,wall_id,center,width,sill,head,kind: window|balcony_door,rooms}]`
  * `rooms[{id,type,label,polygon,area_m2,label_area_m2,size_m,ceiling_height,confidence,master?,child?,open?,zones?}]`
  * `texts`, `warnings`, `debug.m_to_px`, `patches_applied?`
* **`layout.json`**:
  * `items[{id,room,room_type,type,center,size[w,d,h],rot_deg,z,params}]`
  * `checks{overlaps,outside_room,blocking_doors,tall_in_front_of_windows}`
  * `unfurnished_main_rooms`, `warnings`, `rooms{id:{type,placed,missing}}`
* **`assets.json`**:
  * `items{item_id:{source: polyhaven|user|generated|parametric|removed, asset_id?, file?, licence?, front_axis?, unit_scale?, fit?, scale?, fit_err?, poly_cap?, reason?}}`
  * `summary`, `style_words`, `limits`
* **`run_config.json`**: assets path, profile + profile settings, device, exposure/sun/fill knobs,
  export options. This is how Blender scripts get their settings.
* **`agent_log.jsonl`** records:
  * `start`
  * `llm`: turn, content, number of calls, seconds, tokens.
  * `tool`: step, turn, tool, `raw_arguments`, args, reasoning, ok, seconds, `vram_peak_mb`,
    `server_up`, `vram_decision`, component, result, `replay_data`.
  * `error`
  * `end`
  * Turn −1 marks the harness's safe-finish actions.

## 3. Agent layer (`agent.sh` → `app.agent`)

`python -m app.agent run <file> [--backend llm|rules|mock] [--script]`, `replay <agent_log.jsonl>`, `tools`.

### 3.1 Loop

`run_agent()` is a single-agent tool-calling loop:

* **Messages:** one `SYSTEM` prompt (302 words) plus one user message. All 12 tools are offered
  on every turn with `tool_choice="auto"`.
* **Hard budgets:**
  * `max_steps` 40, `max_minutes` 45, `max_cost_usd` 1.5 (GPU price × wall time)
  * `max_patches` 12, `max_repairs_per_issue` 2 (issue keys that stay open after an action)
* **Context control:** `compact()` shortens old tool results so the total stays under
  2.5 × `max_model_len` characters.
* **Tool calls written as text** are parsed as a fallback: the `<tool_call>` JSON form and the
  `<function=…>` Qwen3-Coder form.
* **Finish guard:** the first `finish` call is refused if the export has not been validated, or if
  open high-severity issues are not listed.
* **`safe_finish()`:** when a budget runs out or the loop breaks, the harness deterministically runs
  relayout → choose_furniture → build_scene → export_blender for whatever is stale.
* **`write_final_report()`** writes `final/report.md` and `final/agent_summary.json`.

### 3.2 Tools (registry = `TOOLS` dict filled by `@tool`)

| Tool | Does | Invalidates / marks fresh | GPU component |
|---|---|---|---|
| parse_plan | run_parse (+wallseg) + run_plan + re-apply stored patches | invalidates plan… ; fresh plan | - |
| read_text_vlm | VLM texts (server, else transformers, else error) + run_plan | same as above | `vlm_transformers_*` only without a server |
| get_plan_summary | compact plan summary (rooms / openings / all) | - | - |
| view_image | VLM critique of overlay (with the source page) / layout / render | - | (server) |
| patch_plan | validated patch (9 ops) | plan change: from layout; furniture op: from assets | - |
| relayout | run_layout | from layout; fresh layout | - |
| choose_furniture | overrides, style, `select_assets`, candidate listing | from assets; fresh assets | - |
| build_scene | Blender scene at the preview profile | from scene; fresh scene | - (CPU) |
| render_preview | Blender render at the preview profile + brightness checks | fresh render | `cycles_preview` |
| run_checks | `compute_issues` + repair counters + `stop_repairing` | - | - |
| export_blender | `deliver()` | fresh export | - |
| finish | guarded end | - | - |

**Staleness.** `Ctx.fresh` is an in-memory set. `STAGE_ORDER = plan, layout, assets, scene, render,
export`, and `invalidate(s)` drops s and every stage after it. Nothing is persisted: a new process
starts with an empty set, and freshness is not based on content.

### 3.3 Backends and replay

* **`LLMBackend`:** the managed vLLM server. The request always carries
  `chat_template_kwargs.enable_thinking=false`. If the server dies, it is restarted once.
* **`RulesBackend`:** a generator policy that follows a fixed order. Its only repair is
  `scale_from_label`, used when every labelled room is off by the same factor (within 3 %, and more
  than 2 % in total). It also critiques the overlay and renders when vision is available. It calls
  `finish` twice so it passes the finish guard.
* **`ScriptBackend`:** the `mock` backend, with scripted turns.
* **Replay:** `replay_turns(log)` regroups the logged tool calls by turn, keeping the raw argument
  strings. `ScriptedVision` returns the recorded critiques and VLM texts in their original order.
  Replay recomputes every tool result; only vision answers are replayed from the log.
* **Vision implementations:**
  * `ServerVision`: the same served model reads images through the OpenAI-compatible API, images
    sent as data URLs of at most 1280 px.
  * `NoVision`
  * `ScriptedVision`

### 3.4 Patches (`patches.py`) - "AI proposes, deterministic code disposes" already holds here

* **Checks:** `PATCH_SCHEMA` is JSON Schema Draft 2020-12, with `oneOf` over 9 ops and at most 8 ops
  per patch. Its errors are returned as short path strings.
* **Plan ops:** `scale_plan` (factor 0.5–2), `scale_from_label`, `retype_room`, `move_opening`,
  `resize_opening`, `set_door_swing`, `remove_opening`.
* **Furniture ops:** `swap_furniture`, `remove_furniture`. These are checked against the catalog and
  the slot fit.
* **Invariants (`violations()`):**
  * opening inside its wall; widths and heights plausible; swing `into` ∈ rooms; no overlap on one wall
  * room type and area 0.5–300 m²
  * scale sanity: median hinged door 0.6–1.3 m, median wall 0.05–0.6 m, extent ≤ 80 m,
    inside area 10–1000 m²
  * at least one door
* **Acceptance rule:** a patch is accepted only if it adds no new violation and the cumulative
  `agent_factor` stays within 0.5–2.
* **Storage:** `PatchStore` keeps `plan_raw.json` + `patches.json`, from which `plan.json` is
  rebuilt. Patches whose ids vanished after re-planning are skipped with a warning. Patch records
  carry a wall-clock `ts`.

### 3.5 LLM server and VRAM planner (`llm_server.py`)

* **Server:** vLLM 0.30.0 in its own venv and its own process group, with a PID file and a
  stale-process guard.
* **Flags:**
  * `--enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3`
  * `--default-chat-template-kwargs {"enable_thinking": false}`
  * `--limit-mm-per-prompt {"image":2,"video":0}`, or `--language-model-only` when vision is off
  * `--max-model-len 32768 --max-num-seqs 2`
* **Server environment:** offline HF, FlashInfer sampler off, venv `bin` first on PATH.
* **Memory:** `gpu_memory_utilization = (weights_gb + 6.0) × 1024 / total`, clamped to 0.30–0.92.
* **Model:** `resolve_model`: `llm.model` → `vlm_model` → Qwen3.5-9B on the 48 GB tier, Qwen3.5-4B on
  24 GB.
* **`VramPlanner.before(component)`:** stops the server when the component's estimate + margin does
  not fit next to it. The next LLM call starts the server again, lazily.
* **Measured on the pod (L40S), 2.0.3:** first start 300 s, 24 429 MB, tool call OK. The second
  start-up time is still unknown.

### 3.6 Furniture library (`furniture.py`) and licences

* **Sources:**
  * Poly Haven: the top `polyhaven_per_type` (3) models **by download count**, taken at install time
    (so the pick is not reproducible); md5 is checked per file.
  * User drop-ins in `models_user/<type>/`, with a licence sidecar.
  * GEN3D output (evaluation licence).
* **Inspection:** pure-Python glTF/GLB reading (node transforms, triangles, missing files, dims).
* **Licences:** `catalog.json`, plus `LICENSES.txt` listing the files that were skipped and why.
* **Fit:** a median uniform scale (±35 %) and a non-uniform stretch limit (±15 %). Plants and lamps
  use height-fit.
* **Choice:** ranked by (fits, style words, fit error, polycount, id). Agent overrides live in
  `asset_overrides.json`.
* **`coverage()`:** writes `final/coverage.md/json`, from the Blender QA results.
* **`assets_fetch.py` (step 9):** ambientCG zips (no checksum) and the 2 most-downloaded Poly Haven
  sky HDRIs, also not reproducible.

### 3.7 Tests and evaluation

* **`python -m app.testplans unit`** (CPU): these checks run in one process, in order, and write into
  `$WS/outputs/unit_test` and `$WS/inputs/smoke_test`:
  * textnorm; plan+layout on the synthetic DXF and vector PDF (all 6 rooms within tolerance, layout
    checks 0)
  * patches; agent mock + replay (`MOCK_TURNS`)
  * catalog + licences on synthetic glTF; LLM server command + planner
  * Blender scene/export/validate when Blender or a bpy-Python is at `PIPE_BLENDER`
* **`quick`:** `unit`, then the full pipeline on DXF, DWG, vector PDF, scan PDF and photo, then one
  agent run.
* **Synthetic plan:** one fixed 2+1 apartment (`WALLS`, `OPENINGS`, `ROOMS`, `BALCONY`), with labels
  written as TEXT `MIDDLE_CENTER` in centimetre units. It is not randomised.
* **`python -m app.evaluate`:** compares room sizes against `tests/truth.csv` (±5 cm for vector,
  ±3 % for raster) plus human review scores; writes `tests/results.md` and `.csv`.

### 3.8 Configuration

* `common.DEFAULTS` is deep-merged with `/workspace/config.json`. There is no schema check, and unknown
  keys are merged silently.
* Setup step 8 writes `gpu_price_per_hour`, `vlm_model`, `pod_tier`, `polish.enabled`, `llm.model`,
  `agent.backend` and `gen3d.enabled`.
* Other helpers in `common.py`:
  * `jsave`: a plain `write_text`, not atomic.
  * `run()`: streams output and has no timeout.
  * `blender()`: uses a bpy-Python when `PIPE_BLENDER` names a python.

## 4. Guarantees in place today (must survive every phase)

1. `run.sh`, `start.sh`, `agent.sh` (backends llm/rules/mock, `--replay`) and their usage text.
2. `python -m app.testplans unit|quick`, `python -m app.evaluate`.
3. Job folder layout (section 2.1) and the file names that `testplans.check()` requires.
4. `agent_log.jsonl` replay: identical tool-call sequence; the artifacts are identical too (this
   was verified in 2.0.0 for plan, patches, layout, furniture choice and Blender QA; timestamps
   differ).
5. Export validation: a failed check makes the job and the agent status fail.
6. Licence tracking: `catalog.json` + `LICENSES.txt`, licence filter, attribution, GEN3D flagged as
   evaluation-only.
7. Turkish text: `textnorm.repair/fold/room_type/parse_area/parse_scale`, cp1252/cp1254 and `\U+`
   repair, UTF-8 mojibake repair for DWG.
8. Offline runtime: `env.sh` sets `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`; the vLLM env also
   sets both; telemetry is off.
9. Hard budgets, the VRAM planner, and the rules fallback when the LLM check fails.
10. Pinned versions and checksums: Blender sha256, Poly Haven md5, `--only-binary`, and the three
    lock files.

## 5. Verified defects and limitations (evidence for Phase 2+)

| # | Where | Finding | Evidence |
|---|---|---|---|
| D1 | `parse_dxf` unit | When `$INSUNITS` is set, it is trusted blindly: `solve_scale` then uses `dxf_units` with confidence 1.0 and never cross-checks area labels. When it is missing, the dimension vote keeps the **last** match (including the `u=1.0` branch), not a majority; the default is 0.01 | code, plan.py `solve_scale` first branch |
| D2 | `parse_dxf.walk` | Block attributes are lost: `INSERT.virtual_entities()` does not yield the attached ATTRIBs (the `"ATTRIB"` branch is dead) | ezdxf 1.4.4 probe: block with ATTDEF + LINE → virtual_entities = `[LINE]`, `insert.attribs` = `[ATTRIB 'SALON']` |
| D3 | `parse_dxf` labels | Label x is `insert.x + 0.3·len·height` (assumes left-aligned TEXT). It ignores `halign/valign/align_point` and MTEXT `attachment_point`: "EBEVEYN YATAK ODASI" at 20 cm lands 1.14 m right of its anchor | ezdxf 1.4.4: `Text.get_placement()` returns `(align, p1, p2)`; MTEXT `attachment_point` 5 = middle centre with `insert` as anchor |
| D4 | `textnorm.parse_scale` | No `(?<!\d)`: "21/50" or "11:100" match as 1/50, 1/100. `parse_pdf` takes the **first** phrase with any scale match, not one containing ÖLÇEK/SCALE | regex |
| D5 | `textnorm.room_type` | `0→O` is replaced **after** digits are stripped, so it never fires ("0DA" → " DA" → no type) | code |
| D6 | `common.jsave/run` | non-atomic writes (a crash mid-write corrupts JSON); no subprocess timeout | code |
| D7 | config | no schema check and no warning for unknown or mistyped keys | code |
| D8 | reproducibility | no HF `revision=`; Poly Haven furniture and HDRIs chosen by live download counts; ambientCG zips without checksums | code |
| L1 | plan | h/v walls only (`wall_runs`, `axis_lines`); `orthogonalize` snaps within 8° | code |
| L2 | wallseg | door/window classes predicted but unused | `predict()` uses `lab == 1` |
| L3 | agent | no way to add an opening, merge rooms or change an opening's kind; no reachability invariant | patches.py ops |
| L4 | layout/design | furniture templates are code (`furnish`), `SIZES` hard-coded; style is one prompt string; materials hard-coded (`SPEC`) | code |
| L5 | render | 8-bit PNG only; depth is a 320×180 ray-cast in Python; no auto-exposure; `render_preview` renders every camera | code |
| L6 | polish | one strength, then raw; the agent path never runs a full render or polish | code |
| L7 | staleness | `Ctx.fresh` lives in memory only, is not content-based, and has no versions or revert | code |
| L8 | agent | one generalist prompt with all 12 tools on every turn; critique answers are free JSON parsed with a regex (no structured output) | code |

## 6. Pinned-package facts checked in Phase 0 (used by the plan)

* **ezdxf 1.4.4:**
  * see D2/D3 for attribs and placement.
  * `Text.set_placement(..., MIDDLE_CENTER)` writes the same point to `insert` and `align_point`
    (this is how ezdxf does it; AutoCAD files put the left baseline in `insert`), so the
    reliable API is `get_placement()`.
* **vLLM 0.30.0 (sdist, sha256 `5f8f4e89…bef62b` = PyPI):**
  * **Structured outputs.** A chat request accepts
    `response_format={"type":"json_schema","json_schema":{…}}` or
    `structured_outputs={"json":…|"regex"|"choice"|"grammar"|…}`. There is no `guided_json` field on
    the chat request. The backend is `auto` (xgrammar 0.2.8 / llguidance 1.7.6 / outlines-core in the
    lock), and structured output inside reasoning is off by default (`enable_in_reasoning=False`).
  * **Named and required tool choice.** Named `tool_choice` and `"required"` derive a JSON schema from
    the tool parameters and force it through structured outputs. The `qwen3_coder` parser maps to
    `Qwen3EngineToolParser` (structural tag `qwen_3_coder`). It keeps `supports_required_and_named =
    True` unless `VLLM_ENFORCE_STRICT_TOOL_CALLING` is set.
  * **Sleep mode.**
    * The `--enable-sleep-mode` flag exists and turns on the cumem allocator.
    * The HTTP `/sleep?level=1|2&mode=abort`, `/wake_up` and `/is_sleeping` routes are registered
      only when `VLLM_SERVER_DEV_MODE=1`.
  * All of this is read from source. None of it has run on a GPU yet.
* **Blender 5.2.2 (bpy module):**
  * **Multilayer EXR:** needs `image_settings.media_type = "MULTI_LAYER_IMAGE"` before
    `file_format = "OPEN_EXR_MULTILAYER"`. Setting the format directly raises a TypeError.
  * **Passes and options present:** view-layer passes `use_pass_z/normal/mist/diffuse_color/object_index`,
    `render.use_persistent_data`, `view_settings.exposure`, and the Cycles options adaptive sampling,
    `time_limit`, `use_guiding` and `texture_limit_render`.
  * **Compositor:** `scene.compositing_node_group`; `scene.node_tree` no longer exists.
  * **Unverified:** in the bpy module, `view_transform` lists only `NONE` (no OCIO config), so the
    AgX/Filmic behaviour must be checked on the official binary on the pod.
* **diffusers 0.40.0:** `StableDiffusionXLControlNetImg2ImgPipeline` (used today),
  `StableDiffusionXLControlNetUnionImg2ImgPipeline`, `StableDiffusionXLImg2ImgPipeline` and
  `FluxControlImg2ImgPipeline` are all exported.
* **Network from this sandbox:** PyPI is reachable. huggingface.co, api.polyhaven.com and
  ambientcg.com are **not**, so HF revisions and asset checksums have to come from the pod.
