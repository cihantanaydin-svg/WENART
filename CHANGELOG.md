# Changelog - setup.sh

Format: newest first. `setup.sh.orig` is the untouched 1.0.0 file (sha256 `f32e3a30…af4ed7d`).
Anything marked **UNVERIFIED** could not be checked from the environment this was written in (no GPU, and
huggingface.co, api.polyhaven.com, download.pytorch.org and download.blender.org were blocked by its network
policy). All estimates are marked **ESTIMATE**.

---

## 2.0.3 - 2026-09-30 - fix: vLLM start-up (FlashInfer just-in-time build)

* Pod report (L40S): the agent LLM check failed, setup fell back to the rules backend. `logs/llm_server.log`:
  vLLM's warm-up sampled with FlashInfer's top-k/top-p kernel, which FlashInfer compiles on first use, and the
  build failed with `FileNotFoundError: 'ninja'`.
* `llm_server.py` now starts vLLM with `VLLM_USE_FLASHINFER_SAMPLER=0` (vLLM 0.30.0 then uses its PyTorch
  top-k/top-p sampler; verified in `vllm/v1/sample/ops/topk_topp_sampler.py`), with the vLLM venv's `bin/` first on
  PATH (the lock already installs `ninja` and `nvcc` there, for any other just-in-time build), and with
  `FLASHINFER_WORKSPACE_BASE=/workspace/cache` (its default is the home folder on the container disk).
* UNVERIFIED until the pod rerun: that no other start-up step needs a compiler.

---

## 2.0.2 - 2026-09-30 - fix: DWG input, Turkish text from DWG, plant fit

From the first pod smoke test (all inputs passed except `dwg`: "pipeline crashed" after 3 s). Reproduced here with a
LibreDWG 0.14 build made exactly like setup step 7:
* **DWG files lost all geometry.** `parse.py` converted with `dwg2dxf --as r2018`; LibreDWG 0.14 writes an *empty*
  model space for every DXF version from r2004 up (tested r2004, r2010, r2013, r2018: 0 of 87 entities), while r2000
  keeps all 87. This bug was already in 1.0.0 and hits real DWG files too, not only the smoke test.
  `dwg_to_dxf` now converts to r2000 (default version as a second try) and accepts a result only if it contains
  drawing entities (was: file size > 2 kB), then falls back to ODA as before.
* **Turkish text from DWG.** LibreDWG writes UTF-8 text into that r2000 DXF but marks it Windows-1252, so
  `ÇOCUK ODASI` arrived as `Ã‡OCUK ODASI` (child-room flag lost) and `m²` as `mÂ²` (area labels lost).
  `parse.py` repairs such text only when it re-decodes cleanly as UTF-8 (all Turkish letters and m² tested).
* **Plants were always skipped** when the Poly Haven plant models are wider than the slot at 1.10 m. Plants and
  floor lamps are now scaled to the slot height, or smaller when needed to keep the footprint within 1.8 x the slot,
  as long as they keep at least half the slot height (same rule in `furniture.py` and `blender_scene.py`). The
  "plant skipped" warning now gives the real reason.
* Verified here: the generated test DWG converts with 87 entities, all 6 rooms within tolerance with the correct
  labels, child/master flags and m2 labels; `testplans unit` 8/8 PASS.

---

## 2.0.1 - 2026-09-30 - fix: tar on volumes without chown

* Pod report: step 3 stopped with `tar: uv: Cannot change ownership to uid 1001, gid 117: Operation not permitted`.
  The /workspace volume does not allow chown, and tar running as root tries to keep the archive's file owner.
* All three extractions into /workspace (uv, Blender, LibreDWG) now use `tar --no-same-owner`. Nothing else changed.

---

## 2.0.0 - 2026-09-30 - agent layer, furniture library, Blender deliverable

### What changed at a glance

| Area | Change |
|---|---|
| Version | `SETUP_VERSION="2.0.0"` at the top (1.0.0 had no version string). |
| Switches | New: `AGENT_MODEL`, `SKIP_AGENT_LLM`, `GEN3D` (default 0), `POD_TIER` (auto). `MIN_FREE_GB` is now `auto`. |
| Steps | 10 -> 13 numbered steps. New: 10 agent LLM server, 11 furniture library, 12 GEN3D (optional). Step 13 = old step 10 + an LLM tool-call check + the extended smoke test. |
| New app files | `agent.py`, `patches.py`, `llm_server.py`, `vision.py`, `deliver.py`, `furniture.py`, `blender_export.py`, `validate_export.py`, `gen3d.py`, `gen3d_worker.py`, `requirements-llm.lock`, `requirements-gen3d.lock` |
| New scripts | `/workspace/agent.sh` (agentic run, replay) |
| Changed app files | `common.py`, `main.py`, `report.py`, `blender_scene.py`, `testplans.py`, `requirements.lock`, `env.sh`, `run.sh`, `start.sh` |
| Unchanged (byte-identical output of `--code-only`) | `textnorm.py`, `parse.py`, `vlm.py`, `plan.py`, `layout.py`, `blender_render.py`, `device_check.py`, `polish.py`, `assets_fetch.py`, `evaluate.py`, `wallseg.py`, `__init__.py` |
| Deliverable | `outputs/<job>/final/`: `apartment.blend` (primary), `apartment.glb`, `apartment.usdc`, `previews/`, `overlay.png`, `layout.png`, `report.md`, `coverage.md/json`, `validation.json`, `export.json` (+ `agent_summary.json` for agent runs) |

### New switches (documented at the top of setup.sh)

| Switch | Default | Meaning |
|---|---|---|
| `POD_TIER` | `auto` | `48gb` when `nvidia-smi` reports >= 44000 MB, else `24gb`. Stored in `config.json` (`pod_tier`). |
| `AGENT_MODEL` | `auto` | Hugging Face id of the agent LLM. `auto` = the VLM that setup already downloads (Qwen3.5-9B on >= 40 GB, Qwen3.5-4B otherwise). Stored as `llm.model`. A 27B/35B model on the 24 GB tier stops setup with a message. |
| `SKIP_AGENT_LLM` | `0` | `1` = no vLLM venv and no agent model; `agent.backend` becomes `rules` (deterministic sequence, no LLM). |
| `GEN3D` | `0` | `1` = install TRELLIS.2 image-to-3D (48 GB tier only; ignored with a note on 24 GB). Nothing of it is downloaded unless it is `1`. Needs `HF_TOKEN` (gated DINOv3 encoder). |
| `MIN_FREE_GB` | `auto` | 85 GB, +30 when `AGENT_MODEL` differs from the VLM, +40 with `GEN3D=1` (ESTIMATES). A number overrides it. |

### Part A - setup.sh

* Still one file; every new file is written by `write_code()` through quoted heredocs, so `bash setup.sh --code-only`
  regenerates everything (now 27 Python files + 3 lock files + env.sh + run.sh/start.sh/agent.sh).
* `set -Eeuo pipefail` and the ERR trap are unchanged. Every new step is idempotent with `.setup_state` markers:
  `llm_<locksum>_<cu130|cu128>.done`, `agent_model_<id>.done`, `furniture_v1.done`, `gen3d_<commit>_<locksum>.done`.
  The furniture catalog rebuild (offline, ~1 s) runs on every setup so files you add are picked up.
* The GEN3D build runs as its own `bash` process (`$WS/cache/gen3d_build.sh`) so its `set -e` stays active inside
  the `if` that turns a failed optional build into a warning (bash ignores `-e` inside an `if` condition).
* Header: version, what it does, time and disk updated (ESTIMATES, see below).
* shellcheck 0.11.0: the original had 4 findings (SC2154, SC2012, SC2086, SC1091); all fixed (directives or a
  `find` count instead of `ls | wc -l`, no behaviour change). setup.sh and every generated script now pass clean.
* `start.sh` also checks the vLLM venv and deletes a stale LLM PID file after a restart.
* `env.sh` adds `VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1 VLLM_CACHE_ROOT=/workspace/cache/vllm` (offline, persistent compile cache).

#### Pinned versions added (all verified on PyPI / GitHub on 2026-09-30 unless marked)

| Pin | Value | Evidence |
|---|---|---|
| `VLLM_VERSION` | 0.30.0 | PyPI (released 2026-09-22). Wheel metadata: `torch==2.13.0`, PyPI build = CUDA 13.0 (release notes: "PyPI (CUDA 13.0)"). |
| `requirements-llm.lock` | 198 packages | `uv pip compile` of `vllm==0.30.0`, Python 3.12, `x86_64-manylinux_2_35`, `--only-binary :all:` - resolves with wheels only. Needs glibc >= 2.31 (llguidance wheels); Ubuntu 22.04 = 2.35. |
| `requirements.lock` | +5 packages | `jsonschema==4.26.0`, `jsonschema-specifications==2025.9.1`, `referencing==0.37.0`, `rpds-py==2026.6.3`, `attrs==26.1.0`. Re-resolved together with all existing pins: no existing pin moved, wheels only. |
| `requirements-gen3d.lock` | 57 packages | `torch==2.6.0` (PyPI build = CUDA 12.4: depends on `nvidia-cuda-runtime-cu12==12.4.127`), `transformers==4.57.6` (has `DINOv3ViTModel`, added in 4.56.0), Python 3.12, wheels only. |
| `TRELLIS2_COMMIT` | 75fbf0183001… | microsoft/TRELLIS.2 main, 2026-06-05 (raw files fetched at this commit). |
| `NVDIFFRAST_TAG`, `UTILS3D_COMMIT`, `FLASH_ATTN_VERSION` | v0.4.0, 9a4eb15e…, 2.7.3 | exactly as pinned in TRELLIS.2's own `setup.sh` at that commit. |
| `NVDIFFREC_COMMIT`, `CUMESH_COMMIT`, `FLEXGEMM_COMMIT` | b296927c…, 12289e10…, 6dd94a85… | upstream had them unpinned; pinned to the current heads (2025-12-16, 2026-05-09, 2026-04-22). |
| `MICROMAMBA_VERSION` | 2.9.0-0 | mamba-org/micromamba-releases (2026-08-07), static `micromamba-linux-64` with `.sha256` (checked at install). |

The existing pins (uv 0.12.20, Python 3.12.12, Blender 5.2.2 with checksum, LibreDWG 0.14, torch 2.14.0 cu130 with the
cu128 fallback, all model downloads) are untouched. The main venv is never modified by the new steps.

### Part B - agent layer

**Server choice: vLLM in its own venv** (`/workspace/venv-llm`), not llama.cpp.
* It installs from prebuilt wheels on PyPI; llama.cpp would have to be compiled with `nvcc` on the pod (its Linux
  release binaries are CPU/Vulkan builds) and `llama-cpp-python` CUDA wheels are not on PyPI.
* vLLM 0.30.0 pins `torch==2.13.0`, the pipeline pins `torch==2.14.0`: one venv cannot hold both, so vLLM gets its
  own venv and the pinned torch/CUDA setup of the pipeline stays exactly as it was.
* OpenAI-compatible server with tool calling for Qwen: `--enable-auto-tool-choice --tool-call-parser qwen3_coder
  --reasoning-parser qwen3` (the flags in Qwen's README and the vLLM Qwen3.5 recipe; parser names verified in
  `vllm/tool_parsers/__init__.py` and `vllm/reasoning/__init__.py` at tag v0.30.0; `Qwen3_5ForConditionalGeneration`
  is registered in `vllm/model_executor/models/registry.py`).
* Serves Qwen3.5 with image input, so the same server does tool calling, `read_text_vlm` and the image critiques;
  no second vision model has to be loaded while the agent runs.
* `--default-chat-template-kwargs '{"enable_thinking": false}'` and `chat_template_kwargs` per request (both exist
  in 0.30.0) keep Qwen in non-thinking mode for short, fast tool turns.
* Driver < 580 (the pipeline's cu128 fallback): vLLM publishes `vllm-0.30.0+cu129-...manylinux_2_28_x86_64.whl`
  (seen on the release page); setup installs it with PyTorch's cu129 index and the lock as constraints.
  **UNVERIFIED**: download.pytorch.org was blocked here, so `torch==2.13.0+cu129` could not be checked.

**Models**
* Default: `AGENT_MODEL=auto` reuses the VLM that setup already downloads (Qwen/Qwen3.5-9B on >= 40 GB, -4B below).
  Licence Apache-2.0 for Qwen3.5-9B (HF card, read through a search result - direct HF access was blocked);
  Qwen3.5-4B assumed to carry the same series licence - **UNVERIFIED**, check its LICENSE file on the pod.
* Evaluated for the 48 GB tier: **Qwen/Qwen3.6-27B-FP8** - dense 27B with a vision encoder
  (`Qwen3_5ForConditionalGeneration`), official fine-grained FP8 checkpoint (block 128), Apache-2.0, same
  `qwen3_coder` tool parser (Qwen README: "Agentic Coding" focus). ~30 GB weights (ESTIMATE) -> with its KV cache (~0.80 of
  the GPU) it does not fit next to Cycles/SDXL; the VRAM planner stops it for those stages. Set `AGENT_MODEL=Qwen/Qwen3.6-27B-FP8`.
  * **UNVERIFIED on Ampere (A40 / RTX A6000)**: these GPUs have no FP8 units; vLLM 0.30.0's `fp8.py` falls back to
    Marlin weight-only FP8 ("For GPUs that lack FP8 hardware support, we can leverage the Marlin kernel",
    min capability 7.5). Native FP8 on L40S / RTX 6000 Ada.
  * Not chosen: `Qwen/Qwen3.8-27B-FP8` (Aug 2026, Apache-2.0 per search results) - an open HF discussion reports
    vLLM running it text-only ("no registered multimodal processor"); it is a one-line swap once that is fixed.
    `Qwen/Qwen3.6-35B-A3B-FP8` (MoE, 3B active, fast) is ~37 GB (ESTIMATE) - too tight on 46 GB with KV cache.

**VRAM plan - ESTIMATES** (config `vram_estimates_mb`, `llm.weights_gb`; the agent logs the measured peak of every
tool call in `agent_log.jsonl` and `final/report.md`)

| Component | 24 GB tier (A5000/4090 class, ~24564 MB) | 48 GB tier (A40/A6000/L40S, ~46068 MB) | How it is scheduled |
|---|---|---|---|
| vLLM + Qwen3.5-4B (bf16 9.5 GB + 6 GB KV/activations/graphs) | 0.65 x GPU ~ 15 970 MB | - | reserved while up |
| vLLM + Qwen3.5-9B (bf16 19.5 GB + 6 GB) | does not fit with others | 0.57 x GPU ~ 26 260 MB | reserved while up |
| vLLM + Qwen3.6-27B-FP8 (30 GB + 6 GB) | not allowed | 0.80 x GPU ~ 36 850 MB | reserved while up |
| transformers VLM 4B / 9B (`vlm.py`, run.sh only) | ~12 000 | ~24 000 | agent reads text through the server instead |
| Cycles preview (480x270) / full (1920x1080) | ~3 000 / ~6 000 | ~3 000 / ~6 000 | kept next to the server if it fits |
| SDXL + ControlNet polish | ~13 000 | ~13 000 | run.sh only (not in the agent loop) |
| TRELLIS.2 (GEN3D) | not installed | 24 000+ (upstream minimum) - ~40 000 | runs alone: refuses to start if > 2.5 GB is in use |
| Safety margin | 2 000 | 2 000 | |

Rule (`VramPlanner`): a GPU stage may run next to the server only if `server budget + stage estimate + margin <=
GPU total`; otherwise the server is stopped (SIGTERM on its process group, then wait until VRAM is released) and
started again lazily before the next LLM call. Examples: 48 GB + 9B keeps the server during preview renders
(26 260 + 3 000 + 2 000 < 46 068); 24 GB + 4B keeps it for previews (20 970 < 24 564) but not for SDXL.

**Tool loop** (`app/agent.py`): 12 tools with JSON schemas - `parse_plan`, `read_text_vlm`, `get_plan_summary`,
`view_image`, `patch_plan`, `relayout`, `choose_furniture`, `build_scene`, `render_preview`, `run_checks`,
`export_blender`, `finish`. Hard limits: `agent.max_steps` (40 tool calls), `agent.max_minutes` (45),
`agent.max_cost_usd` (1.5 at the configured GPU price), `agent.max_patches` (12), `agent.max_repairs_per_issue` (2).
Every call is validated (JSON parse, known tool, JSON schema) before it runs; errors go back to the model. When a
limit is hit the harness finishes deterministically (relayout/furniture/scene/export if stale) and writes the report.
`finish` is refused once if the deliverable has not passed validation or if high-severity issues with repair budget
left are neither repaired nor listed as uncertain. Old tool results are shortened to stay inside the context window
(full results stay in the log). Text-format tool calls (`<tool_call>` JSON or the qwen3_coder XML form) are parsed
if the server's tool parser did not catch them.

**Patches** (`app/patches.py`): the LLM never writes geometry or Blender code. Operations: `scale_plan`,
`scale_from_label`, `retype_room`, `move_opening`, `resize_opening`, `set_door_swing`, `remove_opening`,
`swap_furniture`, `remove_furniture`. Each is checked against its own JSON schema (Draft 2020-12, `jsonschema`),
applied to a copy, and accepted only if it introduces **no new** invariant violation: opening inside its wall, no
overlapping openings, door/window widths and heights, swing into one of the door's rooms, room areas, median door
width 0.6-1.3 m and wall thickness 0.05-0.6 m (scale sanity), plan extent, total area, at least one door,
cumulative scale 0.5-2.0. Balconies cannot be retyped. Accepted patches are stored in `02_plan/patches.json`;
`plan.json` = `plan_raw.json` + all patches in order, re-applied automatically after re-planning.

**Acting on checks** (`run_checks`): area vs printed m2 label (3 % vector / 6 % raster), scale confidence (< 0.6),
unlabelled rooms, layout checks, unfurnished main rooms, missing previews, render sanity (dark / blown / flat),
VLM critiques of `overlay.png` against the source page and of renders, library-model fallbacks, export validation.
Attempts per issue are counted; at the budget the issue is marked `stop_repairing` and must appear in
`finish.uncertain`; the report lists everything unresolved.

**Replay**: `agent_log.jsonl` stores every tool call with raw arguments, reasoning, result, seconds, VRAM peak,
server state and the VLM answers; `./agent.sh --replay <log>` feeds the same calls (and recorded VLM answers) into a
new job. Tested here: the replayed job's plan, patches, layout, furniture choice and overrides are identical.

**Backends**: `llm` (vLLM), `rules` (no LLM: fixed order + one deterministic repair: rescale when all labelled rooms
are off by the same factor), `mock` (scripted turns for CPU tests).

### Part C - furniture models

* Library `/workspace/assets/furniture/`: `catalog.json` (type, size in metres, style tags, licence, source URL,
  author, triangle count, sha256, front axis, unit scale), `LICENSES.txt` (every file + attribution list + skipped
  files with reasons + "Powered by Poly Haven" credit).
* Sources: (a) Poly Haven CC0 through the API `assets_fetch.py` already used - models classified by the asset's
  `category` field (taxonomy e.g. `Furniture/Seating/Sofas & Couches`, verified in Poly Haven's `taxonomy.json`)
  plus name/tag keywords, top-3 per type by downloads, glTF 1k with md5 check. Which types it covers is printed at
  setup (`python -m app.furniture report`) and not assumed. (b) `/workspace/assets/models_user/<type>/*.glb|gltf`
  with a licence per file (sidecar `<file>.json` or `licence.json` in the folder); files without an allowed licence
  (`furniture.allowed_licences`: CC0-1.0, CC-BY-4.0, CC-BY-3.0, MIT, Apache-2.0, owned) are skipped and listed.
  (c) The plant/lamp models `assets_fetch.py` already downloads are included as Poly Haven CC0.
* Poly Haven API terms (ToS.md): free for any use incl. commercial; requires a unique User-Agent (kept:
  `floorplan-poc-setup/1.0 ...`); the live-API credit ("Powered by Poly Haven") is written to LICENSES.txt and the report.
* Sizes and triangle counts are read from the glTF itself (accessor min/max through the node transforms) - Poly
  Haven's API has no model dimensions. Files authored in centimetres are detected (x0.01).
* Choice per layout item: same type, fit to the layout slot with uniform size change <= 35 % and non-uniform stretch
  <= 15 % per axis, then style words from config `style` (or `choose_furniture(style=...)`), then fit error.
* Blender import (`blender_scene.py`): all meshes merged into one object named after the item, turned so the front
  faces layout +Y (glTF front +Z = Blender -Y after import; per-file `front_axis` override), origin at the base
  centre, scale baked into the mesh (object scale 1), collapse decimation above the polygon cap (60k, 100-120k for
  sofas/beds, 80k plants), QA (finite coords, size after fit within 2 %, triangle cap, textures present). Any failure
  -> parametric model + reason in `04_scene/furniture_qa.json`.
* Coverage report: `final/coverage.md/json` - furniture type x source used (polyhaven / user / generated /
  parametric / skipped / removed) and every fallback with its reason.
* Parametric furniture is now one merged mesh object per item too (bevels applied), origin at the base centre.
* GEN3D (off by default, see licences below): `python -m app.gen3d plan|run`; input image = your photo in
  `assets/gen3d_inputs/<type>/` or an SDXL product image; runs alone on the GPU in its own venv; results get a size
  from the type's usual height and go through the same catalog, normalisation and QA.

### Part D - Blender deliverable

* Walls: each wall is now one watertight solid with real openings (front grid split at every opening edge, holes
  left out, mirrored back, side faces on every boundary incl. reveals) - 0 non-manifold edges, verified here.
  Before, walls were stacks of touching boxes.
* Slabs are flush with the outer wall faces (were 0.3 m larger).
* Collections: `Walls`, `Floors_Ceilings`, `Openings`, `Furniture.<room name>` (room id added if names repeat),
  `Lights`, `Cameras`; one named mesh object per furniture item.
* `blender_export.py`: metric, unit scale 1.0, metres, Z up; world origin moved to the plan's lower-left corner
  (outer wall faces and balconies); ceilings hidden in the viewport (still rendered); `file.pack_all`, orphan purge;
  writes `apartment.blend` (compressed), `apartment.glb`, `apartment.usdc` (+ `textures/`).
* `validate_export.py` (headless, job fails on any failed check): origin offset, units, collections, wall count,
  wall bounding box vs plan (size and position, 1-2 cm), Z extent, geometry starts at the origin (5 cm slack for door
  frames/handrail), furniture objects (one mesh each, base centre, placed at the layout slot, unit scale), textures
  packed / missing, non-manifold wall edges, rays through every door/window centre pass and solid wall is hit,
  camera count, `.glb` reopens with the same walls, furniture names, bounding box and texture data, `.usdc` reopens.
* `run.sh` gets the new `export` stage (`parse vlm plan layout scene render polish export report`), so run.sh jobs
  deliver `final/` too; the report gains a VRAM-peak column, the validation result and furniture sources.

### testplans.py

* `python -m app.testplans unit` - CPU only, no GPU: textnorm, plan geometry + layout on the generated DXF and vector
  PDF (room sizes vs truth), patch schema + invariants + deterministic rebuild, agent loop with a mock LLM (bad JSON,
  unknown tool, schema error, invariant rejection, finish guard) + replay identity, catalog + licence manifest on
  synthetic glTF (unit fix, node transform, licence normalisation, missing texture, fit, override), vLLM command and
  VRAM planner, text tool-call parsing. Blender scene + export + validation run when Blender (or a bpy Python) exists,
  else SKIP.
* `python -m app.testplans quick` (step 13) = unit + the full pipeline on every test input (now also requires
  `final/` and a passing `validation.json`) + one agent run on the DXF (LLM backend, or rules if not installed).

### Time and disk (ESTIMATES)

| Item | Estimate |
|---|---|
| First run | ~60-100 min (was 45-70): + vLLM venv 3-8 min, furniture 2-5 min, LLM check 2-5 min, agent smoke 5-15 min |
| Re-run | ~10-30 min (smoke test incl. one agent run) |
| GEN3D=1 | +30-90 min (CUDA extensions compile; flash-attn may build from source) |
| Disk, default | ~75 GB on /workspace (was 55): vLLM venv + uv cache ~15 GB, furniture ~1 GB, compile cache ~1 GB, outputs ~0.5 GB/job |
| + `AGENT_MODEL=Qwen/Qwen3.6-27B-FP8` | +30 GB |
| + `GEN3D=1` | +40 GB (venv + cache ~18, TRELLIS.2 + DINOv3 + BiRefNet weights ~18, CUDA toolkit ~5) |

### Licence findings - image-to-3D candidates (read from each repo's LICENSE at its current main)

| Model | Code / weights licence | Territory / commercial limits | Decision |
|---|---|---|---|
| **TRELLIS.2-4B** (microsoft) | MIT | MIT itself has none. **But** its GLB export imports NVIDIA **nvdiffrast** (`o_voxel/postprocess.py`) and it builds **nvdiffrec** - both under the *NVIDIA Source Code License (1-Way Commercial)*: "may be used or intended for use non-commercially" = research/evaluation only. Image encoder: Meta **DINOv3 License** (commercial use allowed, trade-control terms, gated download). BiRefNet, CuMesh, FlexGEMM, utils3d: MIT. | Implemented, **off by default**; generated models are flagged "evaluation only" in LICENSES.txt and the report. |
| TRELLIS (v1, microsoft) | MIT | Same nvdiffrast dependency for textured GLB export. | Not used (TRELLIS.2 is the successor). |
| Hunyuan3D-2.1 (Tencent) | Tencent Hunyuan 3D 2.1 Community License | **Does not apply in the EU, UK and South Korea**; licence from Tencent needed above **1 M monthly active users**; outputs must not be used to improve other AI models. | Documented alternative only. |
| Stable Fast 3D (Stability AI) | Stability AI Community License | Free below **US$1 M annual revenue**; registration for commercial use; "Powered by Stability AI" notice. | Not used. |
| TripoSR | MIT | none found | Not used: older, lower quality. |
| Step1X-3D (StepFun) | Apache-2.0 (code) | weights licence not checked (HF blocked) - **UNVERIFIED** | Not used. |

TRELLIS.2 needs the CUDA 12.4 toolkit to compile its extensions (upstream README); setup uses the pod's `nvcc` if it
is CUDA 12.x, else NVIDIA's `cuda-toolkit` from the `nvidia/label/cuda-12.4.1` conda channel via micromamba
(**UNVERIFIED**: anaconda.org was not reachable to confirm the package).

### UNVERIFIED (everything that needs the pod)

* Anything on a GPU: vLLM start-up and tool calling with Qwen3.5-9B/4B and Qwen3.6-27B-FP8, the gpu-memory-utilization
  values, real VRAM peaks, Cycles OptiX/CUDA with the new scene (tested here with Cycles **CPU** only), SDXL polish,
  TRELLIS.2 install and generation, FP8 on Ampere.
* Downloads blocked here: Hugging Face model files, Poly Haven API responses (fields taken from its published
  `swagger.yml`/`taxonomy.json`), `torch 2.13.0+cu129` on download.pytorch.org, conda `cuda-toolkit` 12.4.1,
  flash-attn 2.7.3 prebuilt wheel availability for torch 2.6.
* Model licences were read from search-engine snippets of the Hugging Face cards (Qwen3.5-9B, Qwen3.6-27B-FP8,
  Qwen3.8-27B), not from the cards themselves - re-check on the pod: `cat /workspace/hf/hub/models--Qwen--*/snapshots/*/LICENSE`.
* Front axis of Poly Haven and generated models (assumed glTF +Z front); the render critique should catch backwards furniture.
* Blender 5.2.2 binary: every Blender script was run here with the official `bpy==5.2.2` wheel (same Blender version,
  Python module build, CPU) - not with the Linux binary.

---

## 1.0.0 - baseline

The setup.sh supplied on 2026-09-30, kept unchanged as `setup.sh.orig`: floor plan (PDF/DWG/DXF/photo) -> plan.json ->
layout.json -> Blender scene -> Cycles renders (+ optional SDXL polish), 10 install steps, smoke test on generated
test plans.
