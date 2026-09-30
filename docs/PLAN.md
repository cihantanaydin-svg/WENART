# Plan - from pipeline to a flexible, agentic, AI-driven system

Starting point: `setup.sh` 2.0.3, described in `docs/ARCHITECTURE.md`. Progress is tracked in
`docs/PROGRESS.md`. Work happens on branch `claude/clever-euler-gcd4qr`, one phase at a time, in small
commits. After each phase: tests, a `PROGRESS.md` update, and a list of what was verified and what was
not. Then I stop for your go-ahead.

This machine has no GPU (`nvidia-smi` is not present). Every phase is built so that it can be tested on
the CPU with mock, scripted or rules backends and Blender on the CPU (`bpy` 5.2.2). Each phase also
lists the checks that need the GPU pod. Those stay **UNVERIFIED** until you have run them.

---

## 0. Where the brief and the code disagree (decisions needed)

Each item has my recommendation. Section 7 repeats them as questions. If you answer "ok to all", I
will follow the recommendations.

| # | Brief says | Code / facts | Recommendation |
|---|---|---|---|
| C1 | "If the folder isn't a git repo, initialise one and commit the original `setup.sh` first." | It already is a repo on branch `claude/clever-euler-gcd4qr`, with 2.0.0 → 2.0.3 committed. `setup.sh.orig` is the 1.0.0 file. | Treat commit `3881587` (2.0.3) as "the original". Tag it `v2.0.3` locally for the byte-identity check. |
| C2 | Phase 1: "setup.sh installs from [real files]" + "bundler regenerates a single-file `dist/setup.sh`" | Your pod workflow is "upload one `setup.sh` to /workspace and run it". The installer in source form needs the repo tree next to it. | Pod entry point = **`dist/setup.sh`** (single file, committed, workflow unchanged). The root `setup.sh` + `src/` are for development, or for a `git clone` on the pod. |
| C3 | Phase 1: "no behaviour change" + "byte-identical check" | ruff (or any formatter) would change bytes. | Phase 1 runs ruff as **lint only**, with a recorded baseline, and never reformats. Fixes for real findings go to Phase 2. `SETUP_VERSION` stays 2.0.3 in Phase 1, so `dist/setup.sh` can equal the 2.0.3 file byte for byte. |
| C4 | Phase 3 acceptance: "a plan edit invalidates exactly layout/scene/render/export" | The code has an `assets` stage (furniture choice, which depends on `layout.json`) between layout and scene. `polish` and `report` also sit downstream. | "Exactly" = layout, **assets**, scene, render, **polish**, export, **report**. Never parse, vlm or plan. The test asserts this exact set. |
| C5 | Principle 4: "specialists share the served model" vs Phase 3: "each role with its own backend/model" | Only one vLLM server fits on the GPU next to Blender and SDXL (the VRAM planner assumes one). | Config allows a model per role, but every role on the **local** backend must resolve to the one served model; validation rejects anything else. Per-role differences are prompt, tool subset, temperature, thinking and max tokens. A different model per role is possible only through an explicitly enabled external endpoint (C6). |
| C6 | "optional hosted provider (opt-in)" vs the earlier brief: "no paid keys", "never put secrets in files", "plans can be confidential" | - | A generic OpenAI-compatible client only (no vendor SDK, no new dependency). It is off by default. A non-loopback URL needs `allow_remote: true`. The key is read from an environment variable **named** in config (`api_key_env`), never stored. Every remote call is logged in `agent_log.jsonl`, and the report says so in a banner. Which provider: your choice (Q5). |
| C7 | Phase 2 generator: "a diagonal wall" | The plan stage handles only h/v walls (L1). Angled walls are Phase 4 work. | Phase 2 generates diagonal-wall seeds. Their tests check for **no crash + a warning**; geometry accuracy is `xfail` until Phase 4. |
| C8 | Phase 2 item 7: "HF `revision=` from a shipped `models.lock.json`, Poly Haven slugs + `files_hash` pinned, ambientCG checksums" | From this sandbox huggingface.co, api.polyhaven.com and ambientcg.com cannot be reached, so I cannot read the revisions or checksums here. **Rule: do not invent them.** | I build the mechanism with the pins left empty; an empty pin means an "UNPINNED" warning and today's behaviour. You run the snippet in section 8 on the pod and I fill in the lock from its output. Files already in the HF cache are not downloaded again. `snapshot_download(revision=<the commit already cached>)` reuses the cached blobs. The step-8 marker files also still skip the step on re-runs. |
| C9 | Phase 5: "evaluate Infinigen Indoors (verify licence first)" | Infinigen installs its own Blender-as-module environment and builds from source. That conflicts with "prebuilt wheels only" and the pinned Blender 5.2.2. | Evaluation only: licence, install footprint, determinism, a CPU trial in a scratch venv (not shipped). Written up in `docs/INFINIGEN.md`. It is not integrated unless you decide otherwise after reading that. |
| C10 | Phase 7: "MCP server (stdio)" | The MCP SDK would be a new dependency. | A stdlib JSON-RPC 2.0 stdio server generated from the tool registry (no new package). It is off by default, and every mutating tool still goes through validated proposals. |

---

## 1. Target architecture

### 1.1 One rule over everything: proposal → validation → deterministic apply

```
 LLM / VLM (role)                 harness (deterministic)                                  job state
 ----------------     JSON      -------------------------------------------------      -----------------
 reader               ───────▶  1 schema (Draft 2020-12, generated per role)  ──x──▶ rejected: errors go
 designer             proposal  2 references (ids exist, types match)                   back to the model,
 director                        3 invariants ("no new violation" + per-domain)         nothing changes
 orchestrator                    4 budgets (patches, repairs/issue, steps, min, $)
                                 5 apply (pure function) ─────────────────────────────▶ new job version vN+1
                                 6 manifest: hashes, staleness, log record              + agent_log.jsonl
```

* Models never produce geometry, coordinates beyond the scalars in a schema, or code.
* Every decision point also has a **rules policy** that emits the same proposal format.
* `backend: rules | llm | scripted` can be set **per specialist**, which makes A/B tests a config
  change.

### 1.2 Components

```
                       ┌──────────────────── Orchestrator (role "orchestrator") ────────────────────┐
  input + brief.json ─▶│ compact state summary (≤ ~1.2k tokens) · budgets · routes goals to         │
                       │ specialists · decides finish                                              │
                       └───────┬─────────────────────────┬──────────────────────────┬──────────────┘
                               ▼                         ▼                          ▼
                     Plan Reader (reader)      Interior Designer (designer)   Render Director (director)
                     page class, per-room      brief → DesignSpec: room        cameras, exposure, VLM
                     VLM crop checks, north    programs, style/palette,        critique 1–5, polish
                     arrow, opening kinds,     materials, furniture, layout    ladder, final render
                     plan patches              intents
                               │                         │                          │
             deterministic ────┼─────────────────────────┼──────────────────────────┼──────────────
             engines           ▼                         ▼                          ▼
                     parse → plan (+patches)   intent solver + circulation     Blender scene/render
                     invariants incl.          check → layout.json →           (EXR passes, persistent
                     reachability              asset choice                    data), polish, export
                                                                               + validation
             shared: tool registry · job manifest (hashes, versions) · VRAM planner · one vLLM server
```

* **Specialists are not separate models.** Each one is a role prompt (≤ 350 words), a tool subset
  (≤ 8 tools) and a rules policy, all running on the one served model. The orchestrator calls a
  specialist as a tool (`consult_reader(goal)`, …). The specialist then runs a bounded sub-loop, and
  its step budget is carved out of the global budget.
* **Context budget for a 4B–9B model with a 32k window:**
  * the state summary has a length test (≤ 4 800 characters)
  * tool results for specialists are capped at 4 000 characters (6 000 today for the single agent)
  * at most 2 images per request (already a server flag)
  * details are fetched through tools (`get_plan_summary(detail)`, `get_room(id)`, `get_issue(key)`)
* **Thinking** stays off by default (as today). It can be switched on per role (`thinking: true`)
  for an A/B run.

### 1.3 Roles and backends (config)

```jsonc
"llm": { ...today's keys stay valid (they become backends.local)... },
"roles": {
  "orchestrator": {"backend": "local", "temperature": 0.2, "thinking": false, "max_tokens": 1024},
  "reader":       {"backend": "local", "temperature": 0.0, "vision": true},
  "designer":     {"backend": "local", "temperature": 0.3},
  "critic":       {"backend": "local", "temperature": 0.0, "vision": true}
},
"backends": {
  "local":  {"type": "vllm_managed"},                                   // today's llm.* settings
  "endpoint": {"type": "openai_compatible", "base_url": "http://127.0.0.1:8012/v1", "enabled": false},
  "hosted": {"type": "openai_compatible", "base_url": "https://…", "api_key_env": "MY_PROVIDER_KEY",
             "enabled": false, "allow_remote": false}
},
"agent": {"policy": {"reader": "llm", "designer": "llm", "director": "llm"}, ...today's budgets...}
```

* `agent.backend: rules|llm|mock` keeps working and maps onto `agent.policy`.
* **Structured output:** vLLM 0.30.0 is verified from its source to support
  `response_format: json_schema`, a `structured_outputs` field, and named `tool_choice`. The named
  choice forces the tool's parameter schema; the `qwen3_coder` parser supports it (ARCHITECTURE §6).
  * Specialists call proposal tools with a **named** `tool_choice`.
  * Free-form turns keep `"auto"`.
  * **Retries are bounded:** `roles.*.max_retries`, default 2. The validator's errors go back as the
    tool result.

### 1.4 Job manifest and versions (replaces `Ctx.fresh`)

* **Where it is declared:** the stage graph is declared once in `app/stages.py`. Each stage lists its
  input files, the config keys it depends on, and its outputs.
* **`job.json` (the manifest):** for each stage it stores the sha256 of every input and output. JSON
  is hashed in canonical form, without volatile keys (`ts`, `seconds`, `started`, `finished`).
* **Freshness:** a stage is fresh when its recorded input hashes equal the current ones.
* **Consequences:**
  * Freshness now survives process restarts.
  * An edit that does not change a stage's inputs does not make that stage stale.
* **Versions:**
  * Each accepted change set creates `vN+1`.
  * **Version state** is the small set of user- and AI-editable files: `brief.json`,
    `design_spec.json`, `patches.json`, `asset_overrides.json`, `render_spec.json`. It is copied to
    `versions/vN/`.
  * Large outputs are rebuilt, not versioned.
  * `revert vK` restores that state and lets the manifest mark what is stale.
* **Compatibility:**
  * `Ctx.fresh` stays as a thin facade, so tool code barely changes.
  * Old jobs without `job.json` are bootstrapped from the files on disk, with a warning.

### 1.5 Tool registry (one source, several outputs)

* **Where:** `app/registry.py`.
* **What each tool declares:** `name`, `desc`, `params` (JSON schema), `fn`, `reads`, `writes`,
  `gpu` (planner component), `cost` (estimated seconds), `cacheable`, `roles`, `mutates`.
* **What it generates:**
  * the OpenAI tool list per role (today's 12 schemas unchanged, byte-equal)
  * `docs/TOOLS.md`
  * the names the rules policies may call (checked at import)
  * the MCP tool list
* **Cacheable tools** (critiques, summaries) are keyed by input hashes. Replay uses the same key.

### 1.6 Replay for every agent and tool

* **Log records:**
  * Every `agent_log.jsonl` record gains `agent` (orchestrator / reader / designer / director) and
    `call_id`.
  * Proposals made through `response_format` are logged as `type: "proposal"` records, with the raw
    text.
* **Replay mechanics:**
  * `replay_turns()` groups records by `(agent, turn)`.
  * `ScriptedVision` becomes `ScriptedModel`, which answers vision and proposal requests from the log
    **by call_id**, not only in order.
  * Rules-policy decisions depend on tool results, so a replay still recomputes tools and compares
    results.
* **Replay test:** it compares the canonical hashes of the version-state files and of `plan.json` /
  `layout.json` / `assets.json`. A Blender `.blend` is not byte-deterministic, so the scene is compared
  by a scene summary: object counts, bounding box, materials.

### 1.7 Rules default for every decision point

| Decision | Owner | Rules policy (same proposal format) | Phase |
|---|---|---|---|
| global scale repair | reader | consistent-factor `scale_from_label` (today) + unit-confusion switch | 2/4 |
| unlabelled / mistyped room | reader | shape rule (today) | 4 |
| opening kind | reader | width + swing-arc rule (today's plan.py logic, exposed as `set_opening_kind`) | 4 |
| unreachable room | reader | `add_opening` 0.80 m on the longest wall shared with a hall/living room, placed in the middle of the free span | 4 |
| room programs | designer | `data/room_programs.json` by type, area and flags | 5 |
| style, palette, materials | designer | brief words matched to `data/styles/*.json`, else the default style (= today's look) | 5 |
| layout intents | designer | today's `furnish()` templates rewritten as intents | 5 |
| furniture model | designer | `select_assets` ranking (today) + asset-aware sizing | 5 |
| cameras | director | today's `build_cameras` scoring | 6 |
| exposure | director | auto-exposure from the preview luminance histogram | 6 |
| polish | director | ladder 0.25 → 0.15 → raw | 6 |
| continue / finish | orchestrator | fixed order + issue budgets (today's `RulesBackend`) | 3 |

### 1.8 Repository layout after Phase 1

```
setup.sh                    source-form installer (same steps; write_code copies from src/)
dist/setup.sh               GENERATED single file for the pod (== 2.0.3 byte for byte in Phase 1)
src/app/                    27 .py + 3 .lock + env.sh (template)      ← today's heredocs, byte-identical
src/scripts/                run.sh start.sh agent.sh (templates with __WS__/__BLV__/__APT__)
src/installer/              dl_models.py write_config.py dl_agent_model.py gen3d_build.sh dl_gen3d.py
tools/bundle.py             source form → dist/setup.sh; --check (byte identity vs v2.0.3 + generated tree)
tests/                      pytest wrappers (+ later phases' tests), conftest with a temp PIPE_WS
requirements-dev.lock       pytest, ruff (dev only, never installed on the pod)
Makefile                    test · lint · bundle · check-identical · shellcheck · (docker)
Dockerfile                  optional CPU dev image (Python 3.12 + the CPU part of the lock + bpy)
docs/                       ARCHITECTURE.md PLAN.md PROGRESS.md (+ TOOLS.md later)
data/ (from Phase 2)        models.lock.json assets.lock.json, later styles/ room_programs.json ...
```

**How the byte-exact bundle works.**

* In the source form every heredoc becomes a stdin redirect with a tag:
  `cat > "$APP/common.py" < "$SRC/app/common.py"  # bundle:heredoc __END_OF_COMMON_PY__`
* The bundler turns each tagged line back into
  `cat > "$APP/common.py" <<'__END_OF_COMMON_PY__'` + the file + the delimiter.
* The inline installer scripts use the same trick (`"$PY" - args < "$SRC/installer/dl_models.py"  # bundle:heredoc __END_OF_DL__`).
* Lines tagged `# bundle:source-only` (the `SRC=` line) are dropped.
* So `bundle(setup.sh) == 2.0.3 setup.sh` can be checked byte for byte.

---

## 2. Phases

Every phase ends with the following, and the version goes up per phase (2.1.0 … 3.0.0):

* `make test` and `make lint` green
* a `dist/setup.sh` regenerated and checked
* `PROGRESS.md` updated
* a summary of what was and was not verified, plus the GPU checklist for the pod

### Phase 1 - make the code editable (no behaviour change)

Work:

* `tools/extract.py` (one-off): writes the 39 heredoc bodies to `src/` and rewrites `setup.sh` into
  source form.
* `tools/bundle.py`: builds `dist/setup.sh`.
* `--code-only`, paths, env, outputs and the substitution stay identical.
* pytest wrappers:
  * one session fixture runs the `testplans.unit()` checks once in a temporary `PIPE_WS`
  * one parametrised test per named check
  * Blender checks run when a bpy Python is found, otherwise they are skipped
* `ruff check` (lint, baseline) + shellcheck 0.11.0 on every shell file, including the generated
  scripts and the GEN3D build script.
* Makefile, optional Dockerfile.

Acceptance:

1. Each of the 39 files in `src/` equals its heredoc body in `v2.0.3:setup.sh`, byte for byte.
2. `dist/setup.sh` == `v2.0.3:setup.sh`, byte for byte.
3. `WS=X bash setup.sh --code-only`, `WS=X bash dist/setup.sh --code-only` and the same run of the
   2.0.3 file produce identical trees (`diff -r`).
4. `make test` shows the same results as `testplans unit` does today: 8/8 PASS with bpy, 7 PASS +
   1 SKIP without.
5. shellcheck finds 0 issues; ruff passes against the baseline, and its findings are listed for
   Phase 2.

GPU: none. The pod gets a byte-identical file.

### Phase 2 - verified bugs + robustness

**Generator.**

* `app/synthplan.py`, seeded: `synthplan(seed) → (DXF, vector PDF 1/50 and 1/100, truth.json)`.
* Room layout: recursive subdivision; room types by area; doors from a spanning tree over room
  adjacency plus an entrance; windows on exterior walls.
* Walls: thickness 0.07–0.35 m, hatched, double-line or polyline-width walls, an optional diagonal
  wall.
* Openings: door widths 0.7–1.0 m, double (1.4–1.8) and sliding doors.
* Labels: TEXT, MTEXT and block ATTRIB labels, all 9 attachment points.
* `$INSUNITS`: correct, missing, or mislabelled (e.g. says mm while drawn in cm).
* Determinism: fixed DXF header variables, so a seed gives the same bytes. `truth.json` holds exact
  polygons, areas and openings.

**Fixes.** Each one first gets a test that fails on 2.0.3. The failing output is recorded in
PROGRESS; then comes the fix.

1. **Unit guess.** A `unit_candidates()` vote over dimension texts (majority, not the last match).
   Plausibility checks: median wall thickness 0.07–0.5 m, extent 4–60 m. It also runs when
   `$INSUNITS` is set. `solve_scale` cross-checks area labels for DXF and switches the unit on a clear
   ×10/×100/×1000/×25.4 disagreement, with a warning. Adds `dxf_unit_m` to the `parse_plan` tool,
   `--dxf-unit-m` to `app.main`, and a rules repair.
2. **Block attributes.** `walk()` reads `insert.attribs`, skipping invisible ones, including nested
   virtual inserts.
3. **Label position.** Use `Text.get_placement()` (align + `align_point`) and MTEXT `attachment_point`
   to put the label centre in the right place.
4. **Scale notes.** `parse_scale` gets `(?<!\d)`, and PDF scale notes prefer phrases with ÖLÇEK/SCALE.
5. **`room_type`.** `0→O` only inside letter tokens, **before** digits are stripped.
6. **Robustness.** Atomic `jsave` (temp file + `os.replace`). `common.run(timeout=…)` with
   `timeouts_s` per command kind in config; the defaults are generous, so only runs that hang for
   hours change. `config.json` is validated against a schema generated from `DEFAULTS` plus enums:
   unknown keys give a warning, wrong types give a clear error.
7. **Reproducibility (C8).**
   * `data/models.lock.json` (HF id → revision) is used by `snapshot_download`, `from_pretrained` and
     vLLM `--revision`.
   * `data/assets.lock.json` holds Poly Haven slugs per type with `files_hash`, HDRI slugs + sha256,
     and ambientCG sha256. Files are verified after download or extraction.
   * An empty pin gives an "UNPINNED" warning and today's behaviour.

Acceptance:

* Every fix has a failing → passing test.
* `testplans unit` and the new tests pass.
* Generator seeds:
  * the pass rate before and after is reported
  * no seed regresses
  * every orthogonal seed covered by fixes 1–5 is within tolerance (±5 cm room size for vector)
  * diagonal seeds: no crash + a warning (C7)
* `setup.sh` 2.1.0 bundles and `--code-only` works.

GPU checklist: re-run `testplans quick` (the pipeline is unchanged on the GPU side); check that
pinned downloads hit the cache.

### Phase 3 - agent foundation

Work:

* Job manifest + versions (§1.4).
* Tool registry + generated outputs (§1.5), including the MCP tool list; the MCP server itself is
  Phase 7.
* LLM layer (`app/llm/`):
  * `Backend` classes: `vllm_managed`, `openai_compatible`, `scripted`
  * roles config and validation (C5, C6)
  * structured proposals (named `tool_choice` / `response_format`), bounded retries
  * a remote-use log and report banner
* Orchestrator on a compact state summary; specialists get a stub interface. Their real prompts and
  tools come in Phases 4–6; until then the orchestrator drives today's 12 tools with the same
  behaviour.
* CPU check of schema compatibility: compile every role schema with xgrammar 0.2.8, the version in
  the vLLM lock, in a scratch venv. If xgrammar cannot be installed without CUDA, this becomes a pod
  check.

Acceptance:

1. The existing agent test (mock + replay) and `agent_smoke` with rules pass unchanged.
2. A plan edit (`patch_plan retype_room`) marks exactly {layout, assets, scene, render, polish,
   export, report} stale (C4). A no-op edit marks nothing stale. Freshness survives a process restart.
3. The registry reproduces today's 12 OpenAI tool schemas byte-equal.
4. Stub-server tests (HTTP OpenAI stub, as in 2.0.0) cover:
   * named `tool_choice` / `response_format` are sent
   * invalid JSON N times leads to a rejection after `max_retries`, with no state change
   * a remote URL without `allow_remote` is refused
   * a local role with a second model is refused
5. The state summary is at most 4 800 characters on the smoke plan.

GPU checklist:

* Structured-output requests on Qwen3.5-9B/4B: latency, and whether xgrammar accepts the schemas.
* Named tool choice with `qwen3_coder`.
* A rerun of the smoke agent with the llm backend.

### Phase 4 - Plan Reader

Work:

* **Specialist:**
  * role prompt
  * tools: `get_plan_summary`, `get_room`, `view_crop` (VLM on a zoomed crop of page + overlay per
    room), `classify_page` (floor plan / section / elevation / site / title; multi-page PDFs),
    `find_north` (VLM + a symbol template), `patch_plan`
  * rules policy
* **New ops:**
  * `add_opening(wall_id, s_m, width, kind, sill?, head?)`: scalars along a wall, no coordinates
  * `merge_rooms(room_ids, reason)`: only rooms that share a detection seam
  * `set_opening_kind(id, kind: hinged|double|sliding|opening|window|balcony_door)`
* **Invariants:** **reachability**. Every non-balcony room is reachable from the entrance through
  doors or openings; balconies through a balcony door or window. It is enforced as "no new
  violation".
* **Evidence:** wallseg classes 2/3 are exported as candidates. A **wall-mask coverage** issue fires
  when too many mask pixels are not explained by plan walls, or plan walls are not supported by the
  mask.
* **Angled walls for vector inputs (DXF/DWG, vector PDF):**
  * extract walls from the wall polygons (hatch or polyline outlines)
  * store walls with any `a`/`b`
  * keep true diagonals in `orthogonalize`
  * check Blender walls and validation for angled walls

Acceptance:

* Scripted tests on generator plans with injected errors, where both the rules policy and a scripted
  reader repair the error and replay is identical:
  * removed door → `add_opening`
  * false split → `merge_rooms`
  * wrong kind → `set_opening_kind`
  * mislabelled unit → unit switch
* Diagonal-wall DXF seeds within ±3 % room area.
* No regression on Phase 2 seeds or the smoke plan.

GPU checklist:

* VLM crop checks on real scans and photos: accuracy and time.
* Page classification on multi-page PDFs.
* North arrow on real plans.

### Phase 5 - Interior Designer

Work:

* **Inputs:** `--brief "…"` or `--brief brief.json` for `app.main` and `agent.sh`, stored as
  `00_input/brief.json`. Schema: household, style words, must-haves, room-use overrides.
* **`DesignSpec` (schema):**
  * room programs; style id; palette
  * materials from a pinned manifest: ambientCG ids + sha256, extending `data/assets.lock.json`,
    recorded in LICENSES.txt
  * furniture list; layout intents
* **Data files:**
  * `data/styles/*.json`: palette, material ids, style tags, polish prompt; the default style
    reproduces today's look
  * `data/room_programs.json`
  * `data/furniture_types.json`: moves `layout.SIZES` + `furniture.TYPES` + clearances out of code
* **Intent solver:** a generalisation of `against_wall` / `free_spot`, plus a **circulation check**:
  * a rasterised free-space BFS from each door to each item's access side, at least 0.6 m wide
  * unsatisfied-constraint feedback: intent, reason, the nearest alternative
* **Rules intent generator:** today's `furnish()` rewritten as intents.
* **Asset-aware sizing:** slot sizes snap to catalog model dimensions when that lowers fit error
  within the room's limits.
* **Blender:** materials and colours follow the DesignSpec; `SPEC` becomes the default style data.
* **Infinigen:** the evaluation write-up (C9). Optional scene enrichment (decor from CC0 sources) is
  off by default.

Acceptance:

* Schema tests (brief, DesignSpec, data files).
* Solver tests: satisfied, unsatisfied with a reason, and a circulation check that catches a blocked
  path.
* **Rules intents reproduce layout checks = 0 and no unfurnished main rooms** on every orthogonal
  generator seed and on the smoke plan.
* A Blender CPU test shows the palette reaching the materials.
* Every new material is licence-checked and listed.

GPU checklist:

* The designer on Qwen3.5 with real briefs (EN/TR).
* The render look per style.

### Phase 6 - Render Director

Work:

* **`blender_render.py`:**
  * multilayer EXR (`media_type = MULTI_LAYER_IMAGE`) with Z, normal and denoising passes
  * `render.use_persistent_data = True`
  * PNG previews are still written
* **Depth:** full-resolution depth from the Z pass (`<cam>_depth.npy` at render resolution) replaces
  the 320×180 ray-cast. The polish interface is unchanged.
* **Render profiles** move to `data/render_profiles.json`; the values stay the same.
* **Tools:**
  * `set_camera` (per room: position intent corner/door/window, height, lens) and `set_exposure`
  * auto-exposure (histogram → EV) as the rules default
  * `render_preview(rooms=…)` renders only the chosen cameras
* **VLM critique:** a structured score of 1–5 each for composition, lighting, realism, furniture
  plausibility and artifacts, plus issues. The critique loop is bounded by budget and a score
  threshold.
* **Polish:**
  * retry ladder 0.25 → 0.15 → raw
  * a pluggable `polish.model` interface; the only allowed value is today's pinned SDXL + ControlNet
    depth until you approve another, permissively licensed model
* **After export:** the harness runs the full-profile final render + polish. The VRAM planner stops
  the LLM server first.

Acceptance, all on the CPU:

* A tiny Blender CPU render writes the EXR with Z/normal, a full-resolution depth `.npy` and the
  persistent-data flag.
* Auto-exposure unit tests on synthetic histograms.
* Critique schema + a scripted critique loop.
* The ladder, with a fake polish pipeline, produces the strength sequence and falls back to raw.
* The harness final-render path runs after export (mocked GPU).

GPU checklist:

* OptiX EXR render time vs PNG; the persistent-data speed-up.
* SDXL with full-resolution depth: edge-gate pass rate, before and after.
* Real VLM scores and loop cost within the 45-minute budget on 48 GB.
* 24 GB tier VRAM with the server stopped and started.

### Phase 7 - flexibility

Work:

* **Editing:**
  * `./agent.sh edit <job> "…"`: the orchestrator in edit mode proposes changes; the diff is shown
    (plan / spec / overrides / render spec, plus before/after overlay)
  * high-impact ops (scale, remove or merge, deleting furniture in bulk) ask for confirmation unless
    `--yes`
  * `./agent.sh diff <job> vA vB` and `./agent.sh revert <job> vK`
* **Batch mode:** `./agent.sh batch <files…>`. The server stays up between jobs; a summary table is
  written.
* **vLLM sleep mode (verified in the 0.30.0 source, UNVERIFIED on a GPU):**
  * off by default
  * `--enable-sleep-mode` plus `/sleep?level=1` and `/wake_up`, which need `VLLM_SERVER_DEV_MODE=1`
  * used by the VRAM planner instead of stop/start when host RAM ≥ the model weights
* **MCP stdio server (C10), off by default.**
* **Experience memory, off by default:**
  * stores issue → accepted fix per job
  * retrieval is deterministic (keyword match)
  * at most 5 lines of hints per role prompt, recorded for replay

Acceptance:

* **Edit round trip:** edit → diff → revert. The version state after the revert equals v1 byte for
  byte, and a rebuild reproduces the v1 plan/layout/assets hashes.
* An identical replay of an edit session.
* A batch run over 3 generator plans.
* MCP: a CPU test with a JSON-RPC client over stdio (`initialize`, `tools/list`, one `tools/call`).
* Sleep-mode command and planner logic tested against the stub.

GPU checklist:

* Sleep and wake time vs the ~300 s cold start.
* Host RAM needed.

### Phase 8 - evaluation, docs, 3.0.0

Work:

* **`app.evaluate` metrics:**
  * room count/type accuracy; opening recall and precision (generator truth)
  * reachability; layout checks and circulation
  * furniture library share
  * VLM render scores; human review (existing CSV)
  * time, cost and VRAM
  * steps, patches, rejections and retries
* **A rules vs agentic comparison:**
  * CPU: rules vs scripted runs on generator seeds (geometry)
  * GPU: real runs on the pod
  * written to `tests/results.md`
* **Static HTML run viewer:** `final/index.html`, self-contained, no network. It shows the timeline
  from `agent_log.jsonl`, images, diffs and versions.
* **Docs and version:** README, CHANGELOG, `SETUP_VERSION` 3.0.0.

Acceptance: the Definition of done in the brief.

* All CPU tests pass.
* Replay is deterministic for every agent.
* Agentic ≥ rules on geometry, and better on design/render within today's budgets on the 48 GB tier.
  The design/render half **needs pod runs**; until they exist it is UNVERIFIED.
* Every GPU path is verified or marked UNVERIFIED.
* Docs are updated.

---

## 3. Test strategy

* **CPU (this machine, and CI-able):**
  * pytest over `testplans unit`
  * generator seeds (fast subset per commit, full set per phase)
  * scripted and replay tests per specialist
  * an HTTP OpenAI stub for the LLM layer
  * Blender CPU through bpy 5.2.2
  * shellcheck, ruff
* **Golden replays:** each phase stores 1–2 small `agent_log.jsonl` fixtures and checks that replay
  gives identical state hashes.
* **GPU (pod):** one `POD_TEST_PLAN.md` section per phase, with exact commands and what to send back.

## 4. Risks

| # | Risk | Mitigation |
|---|---|---|
| R1 | A 4B–9B model makes worse decisions than the rules | rules default for every decision, per-specialist policy switch, structured output, small tool sets, bounded retries; the agentic path is enabled only where evaluation shows ≥ rules |
| R2 | xgrammar/llguidance rejects or slows our schemas (`oneOf`, `pattern`, `anyOf`) | CPU compile check (Phase 3); flatten to one proposal tool per op where needed; fall back to `auto` + validation |
| R3 | 32k context with images: image token cost of Qwen3.5 at 1280 px is unknown | ≤ 2 images per request, crops instead of full pages, summaries instead of history; measured on the pod (Phase 3 checklist) |
| R4 | VRAM on 24 GB: server (≈0.65·24 GB) + Blender + SDXL; 300 s cold start per restart | planner stays in charge; sleep mode (Phase 7); the final render/polish runs once, after the loop |
| R5 | Replay drift: GPU vs CPU machines give different results (e.g. render skipped on CPU) | replay compares state hashes, not wall-clock results; record `gpu` in the start record and warn on mismatch |
| R6 | Angled walls touch plan, layout, Blender and validation | vector inputs only, behind `plan.angled_walls` (on after the Phase 4 tests pass) |
| R7 | Phase 1 byte identity blocks cleanup | cleanup deliberately moved to Phase 2 |
| R8 | Pins need network access this sandbox lacks | pod snippet (section 8); empty pins = today's behaviour + warning |
| R9 | Licence creep (styles, materials, polish models, Infinigen) | every new asset or model pinned + LICENSES.txt + your approval; permissive licences only |
| R10 | Plan confidentiality with remote backends | off by default, loopback check, `allow_remote`, key from env only, logged + banner |
| R11 | Scope: 8 phases | strict gating; each phase ships a working `dist/setup.sh` |

## 5. What stays exactly as it is

* The job folder names.
* `run.sh`, `start.sh` and `agent.sh` usage (new sub-commands are only added).
* `testplans unit|quick`, `evaluate`.
* Offline runtime; the pins, checksums and `--only-binary`.
* The export validation gate.
* The licence manifest.
* `textnorm` behaviour, apart from the D4/D5 fixes, which have tests.

## 6. Still pending from before

* Your full setup rerun on 2.0.3: `agent (llm)` smoke line and the second vLLM start-up time. This
  does not block Phase 1.

## 7. Open questions (recommendation first)

1. **Pod entry point (C2):** keep uploading a single file (`dist/setup.sh`), or git-clone the repo on
   the pod? → *single file*
2. **Pins (C8):** please run the snippet in section 8 on the pod and send me the JSON. Otherwise the
   pins stay empty and warn.
3. **Stale set (C4):** is layout + assets + scene + render + polish + export + report the "exactly"
   you mean? → *yes*
4. **Per-role models (C5):** do local roles share the one served model, with other models only
   through an explicitly enabled endpoint? → *yes*
5. **Hosted provider (C6):** do you want one at all, and which? The plan is a generic
   OpenAI-compatible client, off by default. → *none enabled; generic client only*
6. **Diagonal walls in Phase 2 (C7):** no crash + warning until Phase 4? → *yes*
7. **Infinigen (C9):** evaluation write-up only? → *yes*
8. **ruff (C3):** lint with a baseline in Phase 1, fixes in Phase 2? → *yes*
9. **New dev-only tools** (pytest, ruff pinned in `requirements-dev.lock`, never installed on the
   pod) - OK? → *yes*

## 8. Pod snippet for the pins (read-only, prints JSON, downloads nothing)

```bash
cd /workspace && source app/env.sh && python - <<'EOF' > /workspace/pins_from_pod.json
import hashlib, json, pathlib
W = pathlib.Path("/workspace")
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
lock = W / "app/models.lock.json"
metas = [json.loads(p.read_text()) for p in sorted((W / "assets/furniture/polyhaven").glob("*/meta.json"))]
out = {
  "models_lock": json.loads(lock.read_text()) if lock.exists() else None,
  "polyhaven_furniture": [{k: m.get(k) for k in ("slug", "type", "file", "files_hash")} for m in metas],
  "hdri": {p.name: sha(p) for p in sorted((W / "assets/hdri").glob("*.hdr"))},
  "ambientcg": {str(p.relative_to(W / "assets/materials")): sha(p)
                for p in sorted((W / "assets/materials").rglob("*")) if p.is_file()},
  "legacy_models": {str(p.relative_to(W / "assets/models")): sha(p)
                    for p in sorted((W / "assets/models").rglob("*")) if p.is_file()},
}
print(json.dumps(out, indent=1))
EOF
wc -c /workspace/pins_from_pod.json
```
