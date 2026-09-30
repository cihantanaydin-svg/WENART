# Pod test plan - setup.sh 2.0.0

Everything below needs the GPU pod. Run the steps in order. For each step: the command, what you should see, and
what to send back if it does not look like that. Times and VRAM numbers in this file are **ESTIMATES** - the point
of this plan is to replace them with measurements.

**Pod**: one GPU with 48 GB (A40, RTX A6000, L40S, RTX 6000 Ada) - optional second run on a 24 GB GPU (step 11).
Template with CUDA 13 (Runpod filter "CUDA Versions: 13.0", driver >= 580), Ubuntu 22.04 or newer, container disk
>= 20 GB, volume 150-200 GB at `/workspace`, environment variable `NVIDIA_DRIVER_CAPABILITIES=all`.
`HF_TOKEN` only for GEN3D (step 10) - set it as a Runpod secret/env var, never in a file.

Keep a terminal log: `tmux new -s t` and work inside it. Most commands assume `cd /workspace`.

---

## 0. Record the machine (1 min)

```bash
nvidia-smi
nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv
ldd --version | head -1          # glibc: the vLLM lock needs >= 2.31 (resolved for 2.35 = Ubuntu 22.04)
head -3 /etc/os-release; nproc; free -g; df -h /workspace /
command -v nvcc && nvcc --version | tail -2 || echo "no nvcc"
```
**Send**: the whole output (once). Watch for: driver < 580 (then the untested cu128/cu129 fallbacks are used),
glibc < 2.31 (vLLM install will fail), `/workspace` < 90 GB free.

## 1. Copy setup.sh and run the offline checks on the pod (1 min)

Upload `setup.sh` to `/workspace/setup.sh` (JupyterLab upload or `runpodctl send/receive`).
```bash
sha256sum /workspace/setup.sh            # compare with the value in the final report / commit message
bash -n /workspace/setup.sh && echo SYNTAX_OK
rm -rf /tmp/ws_co && WS=/tmp/ws_co bash /workspace/setup.sh --code-only | tail -1
ls /tmp/ws_co/app/*.py | wc -l           # expected: 27
```
Expected: `SYNTAX_OK`, `Pipeline code written to /tmp/ws_co/app (27 Python files), plus .../run.sh, .../start.sh and .../agent.sh`.
**Send if different**: the output.

## 2. Full install with default switches (60-100 min, ESTIMATE)

Log GPU memory every 2 s in a second tmux window (`Ctrl-b c`), then run setup in the first:
```bash
mkdir -p /workspace/logs
nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv -l 2 > /workspace/logs/vram_setup.csv
```
```bash
cd /workspace && time bash /workspace/setup.sh
```
Lines to look for (in this order):

| Step | Expected |
|---|---|
| 1/13 | `GPU: <name> \| VRAM: ~46068 MB \| driver: 58x...`, `Text-reading model: Qwen/Qwen3.5-9B`, `Pod tier: 48gb \| agent LLM: Qwen/Qwen3.5-9B (vLLM 0.30.0) \| GEN3D: 0` |
| 5/13 | `PyTorch 2.14.0 \| CUDA 13.0 \| <GPU name>` (after installing 5 new packages: jsonschema & co.) |
| 6/13 | `BLENDER_OK 5.2.2 LTS` |
| 8/13 | `settings file: /workspace/config.json {... 'pod_tier': '48gb', 'llm': {'model': 'Qwen/Qwen3.5-9B'}, 'agent': {'backend': 'llm'}, 'gen3d': {'enabled': False}}` |
| 10/13 | `vLLM 0.30.0 \| torch 2.13.0... \| CUDA 13.0` |
| 11/13 | `Poly Haven <type>: <slug> (1k)` lines, then a table `type polyhaven user generated` and `models: N, skipped files: M; types without any model (parametric boxes are used): ...` |
| 12/13 | `Skipped (GEN3D=0): nothing downloaded or installed for image-to-3D.` |
| 13/13 | `DEVICE_CHECK {"device": "OPTIX", ...}`, `Blender render device: OPTIX`, then `Agent LLM check ...` + JSON with `"tool_call_ok": true`, `AGENT LLM OK`, then the smoke test (step 4 below) and `Setup 2.0.0 finished.` |

Stop the VRAM logger (`Ctrl-C`).

**If a step fails, send**: the `=== [hh:mm:ss] N/13 ...` header of the failing step and
`tail -n 120 $(ls -t /workspace/logs/setup_*.log | head -1)`, plus:
* 5/13 or 10/13 (package install): the uv error text; for 10/13 also `ls /workspace/venv-llm/lib/python3.12/site-packages | grep -iE "^(vllm|torch)"`.
* 11/13 (Poly Haven): the warning line (the pipeline still works with parametric furniture).
* 13/13 LLM check: `tail -n 200 /workspace/logs/llm_server.log` and `nvidia-smi` output.
* 13/13 smoke test: see step 4.

## 3. LLM server on its own (3-6 min)

```bash
cd /workspace && source app/env.sh
python -m app.llm_server status          # model, gpu-memory-utilization, full vllm serve command
time python -m app.llm_server test       # start, one tool call, stop
grep "llm: ready in" -r /workspace/logs/ | tail -3
```
Expected: `"tool_call_ok": true`, a `tool_calls` entry calling `get_room_count`, `vram_used_mb` around 26 000 on
48 GB with Qwen3.5-9B (ESTIMATE; `util` in `status` should be 0.57). First start can take several minutes
(compile + CUDA graphs, cached in `/workspace/cache/vllm`); the second start should be faster.
**Send**: the JSON output, the start-up seconds, and on failure `tail -n 200 /workspace/logs/llm_server.log`.

## 4. Smoke test (part of step 13; re-run any time, 15-35 min)

```bash
cd /workspace && source app/env.sh
time python -m app.testplans unit        # CPU-only checks, ~1 min
time python -m app.testplans quick       # unit + every test input + one agent run
```
Expected, `unit`: 8 `PASS` lines (textnorm, plan+layout dxf, plan+layout pdf_vector, patches, agent mock + replay,
catalog + licences, llm server + planner, blender scene/export/validate (CPU)) and `UNIT CHECKS: PASS`.
Expected, `quick` summary: `unit checks PASS`, then `dxf PASS`, `pdf_vector PASS`, `pdf_scan PASS`, `photo PASS`
(and `dwg PASS` if LibreDWG built) each with room lines ending `OK`, then
`agent (llm) PASS (... s, status complete|partial, N tool calls, VRAM peak ... MB)` and `SMOKE TEST: PASS`.
**Send on FAIL**: the whole summary block, and for the failing job:
`cat /workspace/outputs/smoke_<kind>/report.md`, `cat /workspace/outputs/smoke_<kind>/final/validation.json`,
`ls /workspace/outputs/smoke_<kind>/logs/` + the tail of the log of the failing stage; for the agent:
`/workspace/outputs/smoke_agent/final/report.md` and `/workspace/outputs/smoke_agent/agent_log.jsonl`.

## 5. One full run on a real plan (both flows)

Copy a real plan (PDF / DWG / DXF / photo) to `/workspace/inputs/`. Log VRAM in the second window:
```bash
nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv -l 1 > /workspace/logs/vram_run.csv
```
```bash
cd /workspace
time ./run.sh /workspace/inputs/<plan file> --profile full      # classic pipeline + final/
time ./agent.sh /workspace/inputs/<plan file>                   # agentic run (LLM inspects and repairs)
```
Expected: run.sh ends with `DONE: /workspace/outputs/<plan>_<time>` and `report: .../report.md`; agent.sh ends
with `AGENT COMPLETE|PARTIAL: /workspace/outputs/<plan>_agent_<time>`, `export validation: passed` and the
uncertain points. Both jobs contain `final/apartment.blend`, `apartment.glb`, `apartment.usdc`, `previews/`,
`overlay.png`, `report.md`, `validation.json`, `coverage.md`.

Per-tool time and VRAM of the agent run:
```bash
J=$(ls -dt /workspace/outputs/*_agent_* | head -1)
python -c "import json,sys; [print(r['step'], r['tool'], r.get('seconds'), 'VRAM', r.get('vram_peak_mb'), 'server_up', r.get('server_up'), r.get('vram_decision') or '') for r in map(json.loads, open(sys.argv[1])) if r['type']=='tool']" $J/agent_log.jsonl
grep -E "llm: (starting|ready|server stopped)" /workspace/logs/agent_*.log | tail -20
```
**Send**: both `final/report.md`, `$J/agent_log.jsonl`, both `final/validation.json` and `final/coverage.md`,
`final/overlay.png` + 2-3 `final/previews/*.png`, and `logs/vram_run.csv`.

## 6. Measurements to record (fill in and send)

| Measurement | Where it comes from | Value |
|---|---|---|
| Setup total time (first run) | `time bash setup.sh` | |
| Time per setup step | timestamps of the `=== [hh:mm:ss] N/13` lines | |
| Disk used on /workspace after setup | `du -sh /workspace/{venv,venv-llm,hf,cache,assets,opt}` | |
| vLLM start-up time (first / second) | `llm: ready in Ns` lines | |
| VRAM of the running server | `python -m app.llm_server test` -> `vram_used_mb` | |
| VRAM peak per agent tool | step 5 one-liner (`vram_peak_mb`) | |
| Did the planner stop the server for a stage? | `vram_decision` in the same output / "Scheduling decisions" in report.md | |
| Agent: tool calls, minutes, cost, patches, status | top of `final/report.md` | |
| Preview render time per view | `05_render/render.json` of the agent job | |
| Smoke test total time and per input | `time python -m app.testplans quick` summary | |
| Peak VRAM during setup and during the real run | max of column 2 in `vram_setup.csv` / `vram_run.csv` | |

## 7. Open the .blend and check it by eye

Automatic check again on the pod (must print `VALIDATION_OK`):
```bash
J=$(ls -dt /workspace/outputs/*_agent_* | head -1)
/workspace/opt/blender-5.2.2/blender -b --factory-startup --python-exit-code 1 -P /workspace/app/validate_export.py -- $J
```
Then download `final/apartment.blend` (JupyterLab: right click > Download, or `runpodctl send $J/final/apartment.blend`)
and open it in **Blender 5.2** on your computer:

1. *Scene properties > Units*: Metric, Unit Scale 1.000, Length Meters.
2. *Outliner*: collections `Walls`, `Floors_Ceilings`, `Openings`, `Furniture.<room name>` (one per room with
   furniture), `Lights`, `Cameras`. Each furniture item is one object named like `r5_sofa_0`.
3. Top view (numpad 7): the 3D cursor / world origin sits at the lower-left corner of the outer walls (or of the
   balcony if it sticks out further). Ceilings are hidden in the viewport (eye icon in `Floors_Ceilings`).
4. Walls: every door and window is a real hole. Select a wall, Tab (edit mode), *Select > Select All by Trait >
   Non Manifold* - nothing may be selected.
5. Furniture: click items - the orange origin dot is on the floor under the item's centre; nothing floats or sinks;
   fronts face into the room (sofa back to the wall, wardrobe doors to the room). *Object properties > Custom
   properties* show `asset_id`, `licence`, `source` for library models.
6. *File > External Data > Report Missing Files*: nothing missing. Textures are packed (*Unpack Resources* lists them).
7. Measure 2-3 rooms (floor object, N panel > Dimensions) against `final/overlay.png` and the room table in `report.md`.
8. Cameras: select a camera, numpad 0, F12 renders one view in Cycles.
9. *File > Import > glTF* `apartment.glb` into an empty file and *File > Import > USD* `apartment.usdc`: same walls,
   same furniture names, same size.

**Send**: screenshots of the Outliner, the top view and one furniture close-up, and anything that looks wrong.

## 8. Replay (5-15 min)

```bash
J=$(ls -dt /workspace/outputs/*_agent_* | head -1)
./agent.sh --replay $J/agent_log.jsonl --job-dir /workspace/outputs/replay_check
for f in 02_plan/patches.json 02_plan/plan.json 03_layout/layout.json 03_layout/assets.json; do
  python -c "import json,sys; a,b=[json.load(open(p)) for p in sys.argv[1:]]; strip=lambda x:[{k:v for k,v in e.items() if k!='ts'} for e in x] if isinstance(x,list) else x; print(sys.argv[1].split('/')[-1], 'identical' if strip(a)==strip(b) else 'DIFFERENT')" $J/$f /workspace/outputs/replay_check/$f
done
```
Expected: 4x `identical`. **Send**: the output if anything differs.

## 9. Optional: the larger 48 GB agent model (Qwen/Qwen3.6-27B-FP8, +30 GB disk)

```bash
AGENT_MODEL=Qwen/Qwen3.6-27B-FP8 SKIP_SMOKE_TEST=1 bash /workspace/setup.sh
cd /workspace && source app/env.sh && time python -m app.llm_server test
time ./agent.sh /workspace/inputs/<same plan as step 5>
```
Expected: `util` 0.8, `tool_call_ok: true`; in the agent log the planner stops the server before
`render_preview` (`vram_decision: cycles_preview: server stopped ...`). On A40/A6000 (Ampere, no FP8 units) this
uses vLLM's Marlin weight-only FP8 path - **UNVERIFIED**, watch the server log for errors.
Compare with step 5: tool calls, patches, repair quality, total minutes. Back to the default: `AGENT_MODEL=auto SKIP_SMOKE_TEST=1 bash /workspace/setup.sh`.
**Send**: `llm_server test` JSON, both `final/report.md`, `logs/llm_server.log` on failure.

## 10. Optional: GEN3D image-to-3D (+40 GB disk, 30-90 min; licence limits in CHANGELOG.md)

Accept the Meta DINOv3 licence on Hugging Face with the account of your `HF_TOKEN` (the exact repo id is read from
TRELLIS.2's `pipeline.json` and printed as `downloading <repo> @ <hash>`; if that download fails with 401/403,
open that repo page and accept the licence). Then:
```bash
GEN3D=1 SKIP_SMOKE_TEST=1 bash /workspace/setup.sh
cd /workspace && source app/env.sh
python -m app.gen3d plan                        # types without any library model
python -m app.llm_server stop                   # GEN3D must be alone on the GPU
time python -m app.gen3d run --types sofa --max 1
python -m app.furniture report                  # the sofa row should show generated 1
```
Expected: `GEN3D imports OK` in `logs/gen3d_build.log`, `GEN3D weights OK: [...]`, `gen3d sofa: ok (N triangles, S s)`.
**Send**: `tail -n 200 /workspace/logs/gen3d_build.log`, the `gen3d run` output, `nvidia-smi` during the run.

## 11. Optional: degrade test on a 24 GB GPU

New pod with a 24 GB GPU, same volume layout: `bash /workspace/setup.sh`.
Expected in 1/13: `Pod tier: 24gb | agent LLM: Qwen/Qwen3.5-4B (vLLM 0.30.0) | GEN3D: 0`; `llm_server status`
shows util 0.65; the smoke test passes. Run step 5 once and send the same files.

## 12. Optional: no LLM

```bash
SKIP_AGENT_LLM=1 SKIP_SMOKE_TEST=1 bash /workspace/setup.sh
./agent.sh /workspace/inputs/<plan file>          # uses the rules backend
```
Expected: `Skipped (SKIP_AGENT_LLM=1)` in 10/13 and an agent report with `backend rules`.

## 13. Your own furniture models

Put a CC0 model (for example from a Kenney or Quaternius pack) in `/workspace/assets/models_user/<type>/`, e.g.
`/workspace/assets/models_user/armchair/chair.glb`, and a licence file next to it:
```bash
echo '{"licence": "CC0-1.0", "source_url": "https://kenney.nl/assets/furniture-kit", "author": "Kenney"}' \
  > /workspace/assets/models_user/armchair/licence.json
cd /workspace && source app/env.sh && python -m app.furniture catalog
```
Type folder names: sofa, armchair, chair, coffee_table, dining_table, desk, nightstand, bistro_table, wardrobe,
bookshelf, shoe_cabinet, tv_unit, bed (or bed_double / bed_single), floor_lamp, plant, fridge, washer, toilet,
bathtub, vanity, basin_small, shower. If a model faces the wrong way in the renders, add `"front_axis": "+Y"`
(or `+X`, `-X`; default `-Y`) to its licence/sidecar JSON and run the catalog again.

## What to send back (checklist)

1. Step 0 output.
2. `time` of setup and the 13 step timestamps; `du -sh` of step 6.
3. `llm_server test` JSON (step 3).
4. The smoke test summary block (step 4).
5. From each real run (step 5): `final/report.md`, `final/validation.json`, `final/coverage.md`, `agent_log.jsonl`,
   `overlay.png`, 2-3 previews, `vram_run.csv`.
6. The filled measurement table (step 6) and the Blender screenshots (step 7).
7. For every failure: the files listed in that step.
