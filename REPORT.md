# Final report - setup.sh 2.0.0

`setup.sh` sha256 `e1e0a2a9fbf8703e921019805ecd41c10e83b130fdd70bd939a842ed87844983` (8 313 lines, still one file).
`setup.sh.orig` sha256 `f32e3a304dfd0edb9af4bbf48662e8961a930522a4d0ac6722d81910daf4ed7d` (untouched 1.0.0).
Details and evidence per change: `CHANGELOG.md`. Commands for the pod: `POD_TEST_PLAN.md`.

Nothing here ran on a GPU. Every claim below says how it was checked; everything that needs the pod is listed
under UNVERIFIED.

## 1. Changed files

| File | What |
|---|---|
| `setup.sh` | v2.0.0: 4 new switches, 3 new install steps (13 total), new heredocs, pins, header, shellcheck fixes |
| `setup.sh.orig` | the file you supplied, unchanged |
| `CHANGELOG.md`, `POD_TEST_PLAN.md`, `REPORT.md` | new |

Files `setup.sh --code-only` now writes:

| Generated file | Status |
|---|---|
| `app/agent.py` | new - tool-calling loop, 12 tools, budgets, finish guard, replay, llm/rules/mock backends, report |
| `app/patches.py` | new - 9 patch operations, JSON schema (Draft 2020-12), invariants, `plan_raw.json` + `patches.json` |
| `app/llm_server.py` | new - vLLM lifecycle (own process group, health wait, stop + wait for free VRAM), VRAM planner |
| `app/vision.py` | new - critiques and text reading through the served Qwen3.5 (reuses `vlm.py` prompt/tiles/parser) |
| `app/furniture.py` | new - Poly Haven fetch, user drop-in, glTF inspection, catalog, LICENSES.txt, fit/style choice, coverage |
| `app/deliver.py` | new - export + validation + `final/` files |
| `app/blender_export.py`, `app/validate_export.py` | new - Blender deliverable and its headless validation |
| `app/gen3d.py`, `app/gen3d_worker.py` | new - optional TRELLIS.2 image-to-3D (off by default) |
| `app/requirements-llm.lock`, `app/requirements-gen3d.lock` | new - 198 and 57 exact pins, wheels only |
| `agent.sh` | new - agentic run / replay, always stops the LLM server on exit |
| `app/blender_scene.py` | watertight walls with real openings, library model import + normalisation + QA, one object per item, collections, flush slabs |
| `app/main.py` | `export` stage, furniture model choice before the scene, VRAM peak per stage, `blender()` moved to common |
| `app/report.py` | VRAM column, deliverable validation and furniture sources, copy to `final/report.md` |
| `app/common.py` | additions only: config defaults (preview profile, pod tier, VRAM estimates, llm/agent/furniture/gen3d/export), `blender()`, GPU helpers |
| `app/testplans.py` | `unit` mode (CPU), `final/` checks in the smoke test, one agent run |
| `app/requirements.lock` | +5 pins (jsonschema and its dependencies), nothing else changed |
| `app/env.sh`, `run.sh`, `start.sh` | vLLM telemetry off + cache dir; usage text; vLLM check + stale PID cleanup |
| `app/parse.py` (2.0.2) | DWG conversion fixed (LibreDWG 0.14 empty r2018 output) and UTF-8 text repair for DWG input |
| 11 other app files | byte-identical to the 1.0.0 output |

## 2. Offline test results (this machine: 4 CPUs, no GPU, Python 3.11/3.12/3.13)

Temporary venvs only: Python 3.12 with the lock's exact versions of numpy, opencv, scipy, shapely, scikit-image,
ezdxf, pdfplumber, reportlab, pypdfium2, matplotlib, pillow + jsonschema; Python 3.13 with `bpy==5.2.2` from PyPI
(the official Blender 5.2.2 Python module = same Blender version as the pinned binary) standing in for Blender.
Final run, from a fresh `WS=<tmp> bash setup.sh --code-only` of the final file:

| Check | Result |
|---|---|
| `bash -n setup.sh` | PASS |
| shellcheck 0.11.0 on setup.sh | 0 findings (original: 4, all fixed) |
| shellcheck on generated run.sh, start.sh, agent.sh, env.sh and the embedded GEN3D build script | 0 findings |
| `py_compile` all 27 generated .py (Python 3.12) + Blender scripts (3.13) | PASS |
| Regression: 12 files not meant to change vs 1.0.0 output | 12/12 byte-identical; `requirements.lock` and `common.py` only additions |
| `python -m app.testplans unit` with bpy | 8/8 PASS: textnorm; plan+layout DXF and vector PDF (all 6 rooms within tolerance, layout checks 0); patches; agent mock + replay; catalog + licences; LLM server command + VRAM planner; Blender scene + export + 20 validation checks |
| same without Blender | 7 PASS, Blender check SKIP (exit 0) |
| Rules agent end to end on the vector PDF, with 6 test models in `models_user/` | exit 0, export validation passed, status PARTIAL (no GPU -> no previews, reported as open issue) |
| LLM backend over real HTTP against a local OpenAI-compatible stub | exit 0, export passed; requests carried 12 tools, `tool_choice=auto`, `enable_thinking=false`, `tool_call_id` on every tool reply, 2 images for the overlay critique; a qwen3_coder-style text tool call was parsed and accepted |
| Replay of that run | 5/5 artifacts identical (plan, patches, layout, furniture choice, Blender furniture QA) |
| `app.main` DXF, profile quick, incl. Cycles **CPU** render (6 views) + export | exit 0, validation OK, `testplans.check()` PASS |
| Lock resolution (`uv pip compile`, wheels only, manylinux_2_35) | main (92), vLLM (198), GEN3D (57) all resolve |

Also checked by hand during development: imported models face the right way (sofa backrest at local -Y, +X-facing
TV unit turned to +Y), decimation 81 920 -> 60 000 triangles, centimetre model scaled x0.01, texture packed after
import, unlicensed and GPL files skipped with reasons, the stale-PID guard does not kill an unrelated process, and
one render inspected by eye (real door/window holes, imported furniture in place).

## 3. What could NOT be tested here (UNVERIFIED)

* GPU anything: vLLM start-up and serving of Qwen3.5-9B/4B, tool-calling quality in a 40-step loop, the
  `gpu-memory-utilization` values (0.57 / 0.65 / 0.80), real VRAM peaks, server stop/restart timing, OptiX/CUDA
  Cycles with the new scene (only Cycles CPU ran), SDXL polish, `read_text_vlm` through the server on real scans.
* The official Blender 5.2.2 Linux binary (scripts ran in the `bpy` 5.2.2 module, CPU).
* Downloads blocked by this environment's network policy: Hugging Face (all model files), api.polyhaven.com (the
  real model list; code follows the published swagger/taxonomy), download.pytorch.org (`torch 2.13.0+cu129` for the
  driver < 580 fallback), anaconda.org (`cuda-toolkit` 12.4.1 for GEN3D), download.blender.org.
* The vLLM `+cu129` fallback path for drivers < 580 (wheel existence confirmed on the release page, not installed).
* Qwen/Qwen3.6-27B-FP8 on Ampere (Marlin weight-only FP8 path in vLLM's code, not run).
* GEN3D: the whole TRELLIS.2 install (CUDA extension builds, flash-attn 2.7.3 wheel availability for torch 2.6),
  its weights, generation, and the TRELLIS.2 output orientation/scale.
* Poly Haven furniture in practice: which types it really covers, whether their front axis matches the glTF
  convention, how many pass the +-15 % fit.
* Model licences were read from search-result snippets of the Hugging Face cards, not the cards themselves.

## 4. Licence findings

| Item | Licence | Restrictions found | Used? |
|---|---|---|---|
| Qwen/Qwen3.5-9B (agent + VLM, default on 48 GB) | Apache-2.0 (HF card via search result) | none beyond Apache-2.0 notice | yes |
| Qwen/Qwen3.5-4B (default on 24 GB) | assumed Apache-2.0 like the rest of the series - **not checked**, re-check the LICENSE file on the pod | - | yes |
| Qwen/Qwen3.6-27B-FP8 (48 GB option) | Apache-2.0 (README front matter via search result) | none beyond Apache-2.0 notice | optional |
| Qwen/Qwen3.8-27B(-FP8) | Apache-2.0 (search results) | - | not chosen (vLLM text-only report) |
| vLLM 0.30.0 | Apache-2.0 (PyPI metadata) | - | yes |
| Poly Haven models / HDRIs | CC0 | API ToS: unique User-Agent, "Powered by Poly Haven" credit when surfacing live-API content; commercial use allowed | yes |
| ambientCG textures | CC0 | - | yes (unchanged) |
| TRELLIS.2-4B code + weights | MIT | GLB export needs **nvdiffrast** and builds **nvdiffrec**: NVIDIA Source Code License, **non-commercial (research/evaluation) only** | GEN3D=1 only, flagged "evaluation only" |
| DINOv3 (TRELLIS.2 image encoder) | Meta DINOv3 License | commercial allowed; trade-control terms; gated (HF_TOKEN + acceptance) | GEN3D=1 only |
| BiRefNet, CuMesh, FlexGEMM, utils3d | MIT | - | GEN3D=1 only |
| Hunyuan3D-2.1 | Tencent Hunyuan 3D 2.1 Community License | **not licensed in the EU, UK, South Korea**; separate licence above 1 M MAU; outputs may not train other models | no (documented alternative) |
| Stable Fast 3D | Stability AI Community License | free below US$1 M annual revenue, registration for commercial use | no |
| TripoSR | MIT | - | no |
| SDXL base (existing polish model, also GEN3D input images) | not re-checked here (HF blocked); unchanged from 1.0.0 | - | unchanged |

## 5. Known limitations

* The agent can only act through the 9 patch operations: it cannot add a missing wall, split a room or add a door
  it did not detect - it reports those as uncertain.
* `render_preview` renders every camera; there is no per-room re-render.
* The rules backend fixes only one thing deterministically (a consistent scale error across all labelled rooms).
* Library model choice ranks style words above small fit differences; style tags from Poly Haven are coarse.
* Plants are skipped when no plant model exists (unchanged 1.0.0 behaviour); rugs and kitchen runs are always
  parametric (their sizes are per plan).
* Front axis of downloaded/generated models is assumed (glTF +Z); a wrong one shows as backwards furniture in
  renders - fix per file with `front_axis`.
* The VRAM table is an estimate until step 5/6 of the pod plan is done; the planner only uses those numbers.
* Validation tolerances: 1-2 cm on the wall box, 5 cm for door frames/handrail outside the origin.
* GEN3D is evaluation-only by licence (nvdiffrast) and needs a gated HF download.

## 6. First three things most likely to break on the pod

1. **vLLM start-up for Qwen3.5 with the chosen memory settings** - e.g. "not enough KV cache for max_model_len",
   a start-up longer than `llm.startup_timeout_s` (first run compiles), or an out-of-memory next to Blender.
   Symptom: `WARNING: the agent LLM check failed` in step 13 (setup then switches `agent.backend` to `rules`, so
   everything else still works). Look at `logs/llm_server.log`; fix in `/workspace/config.json`:
   `"llm": {"gpu_memory_utilization": 0.65, "max_model_len": 16384, "extra_args": ["--enforce-eager"]}`.
2. **Tool-calling quality of the small model over many steps** - wrong argument shapes, repeated rejected patches,
   or stopping without `finish`. The harness bounds all of it (schema errors go back to the model, step/time/cost
   and patch budgets, deterministic safe finish that still exports), so the likely outcome is a PARTIAL status with
   few useful repairs rather than a crash. If so, compare with `AGENT_MODEL=Qwen/Qwen3.6-27B-FP8` (pod plan step 9).
3. **Real Poly Haven models in the slot fit / import** - many may fail the +-15 % non-uniform or 35 % size limits
   (-> parametric fallback, listed in `final/coverage.md`), face the wrong way, or carry glTF features the checks do
   not expect. The job does not fail on this (QA falls back), but furniture may stay mostly parametric; widen
   `furniture.max_nonuniform` / `max_uniform_change` in config.json or add your own models.
