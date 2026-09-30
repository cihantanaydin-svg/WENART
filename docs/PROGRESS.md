# Progress

Branch `claude/clever-euler-gcd4qr`. Baseline: `setup.sh` 2.0.3 (commit `3881587`). Plan: `docs/PLAN.md`.
Legend: `[x]` done and verified here (CPU) · `[ ]` open · **UNVERIFIED** = needs the GPU pod.

## Phase 0 - reading and planning (done, waiting for go-ahead)

- [x] `nvidia-smi`: not present on this machine, so no GPU. All work is CPU-tested with
      mock/scripted/rules backends and bpy 5.2.2.
- [x] Mapped the heredocs: 39 in total. 34 are in `write_code()`, 5 are inline installer scripts.
      Line table in ARCHITECTURE §1.2.
- [x] Read every module. The 34 `write_code()` bodies were extracted from `setup.sh` 2.0.3 and are
      byte-identical to the copies read.
- [x] Checked the pinned-package APIs from the packages themselves:
  - [x] ezdxf 1.4.4: probed.
  - [x] vLLM 0.30.0: sdist sha256 = PyPI, source read.
  - [x] Blender 5.2.2: bpy probed.
  - [x] diffusers 0.40.0: wheel checked.
  - Details in ARCHITECTURE §6.
- [x] Verified defects D1–D8 and limitations L1–L8 against the code and the packages
      (ARCHITECTURE §5).
- [x] `docs/ARCHITECTURE.md`, `docs/PLAN.md`, `docs/PROGRESS.md` written.
- [x] Brief/code conflicts C1–C10 and open questions Q1–Q9 listed (PLAN §0, §7).
- [ ] Your answers to Q1–Q9 (or "ok to all").
- [ ] Pins JSON from the pod (PLAN §8). Needed for Phase 2 item 7; everything else can go ahead
      without it.

Not verified in Phase 0: anything on a GPU; the image token cost of Qwen3.5; whether xgrammar
accepts our schemas (planned CPU check in Phase 3).

## Phase 1 - make the code editable (no behaviour change)

- [ ] tag `v2.0.3` → `3881587`
- [ ] `tools/extract.py`: 39 heredoc bodies → `src/`; `setup.sh` in source form with `# bundle:heredoc` tags
- [ ] `tools/bundle.py` → `dist/setup.sh`; `--check`
- [ ] Byte identity:
  - [ ] each `src/` file == its heredoc body
  - [ ] `dist/setup.sh` == `v2.0.3:setup.sh`
  - [ ] `--code-only` trees identical (source form, dist, 2.0.3)
- [ ] pytest wrappers around `testplans unit` (temporary `PIPE_WS`, bpy auto-detected)
- [ ] ruff (lint only, baseline) + shellcheck 0.11.0 on all shell files; `requirements-dev.lock`
- [ ] Makefile (test, lint, bundle, check-identical, shellcheck); optional Dockerfile
- [ ] PROGRESS update + summary → stop

## Phase 2 - verified bugs + robustness

- [ ] `app/synthplan.py` seeded generator + truth (all variants in PLAN §2 Phase 2)
- [ ] Baseline pass rate of the generator seeds on 2.0.3 recorded
- [ ] D1 unit vote + plausibility + label cross-check + `dxf_unit_m` (failing → passing test)
- [ ] D2 `insert.attribs` (failing → passing)
- [ ] D3 label placement from `get_placement` / `attachment_point` (failing → passing)
- [ ] D4 `parse_scale` lookbehind + ÖLÇEK/SCALE preference (failing → passing)
- [ ] D5 `room_type` 0→O order (failing → passing)
- [ ] D6 atomic `jsave`, `run()` timeouts; D7 config schema + unknown-key warnings
- [ ] D8 `data/models.lock.json` + `data/assets.lock.json` mechanism; pins from the pod (C8)
- [ ] ruff findings from the Phase 1 baseline fixed
- [ ] SETUP_VERSION 2.1.0, CHANGELOG, `dist/setup.sh`, GPU checklist

## Phase 3 - agent foundation

- [ ] `app/stages.py` graph + `job.json` manifest (hashes, versions, bootstrap of old jobs)
- [ ] `Ctx.fresh` → manifest facade; plan edit → exactly {layout, assets, scene, render, polish, export, report}
- [ ] `app/registry.py`; today's 12 tool schemas reproduced byte-equal; `docs/TOOLS.md`
- [ ] `app/llm/`: backends, roles, validation (C5/C6), structured proposals, bounded retries, remote log/banner
- [ ] Orchestrator on a compact state summary (≤ 4 800 chars); per-specialist policy switch
- [ ] Replay: `agent` + `call_id` in the log, `ScriptedModel`
- [ ] xgrammar CPU schema compile check (or pod check)
- [ ] SETUP_VERSION 2.2.0, GPU checklist

## Phase 4 - Plan Reader

- [ ] Reader role, tools (`get_room`, `view_crop`, `classify_page`, `find_north`), rules policy
- [ ] Ops `add_opening`, `merge_rooms`, `set_opening_kind`; reachability invariant
- [ ] wallseg door/window evidence; wall-mask coverage issue
- [ ] Angled walls for vector inputs (flag `plan.angled_walls`)
- [ ] Injected-error tests (rules + scripted), replay identical
- [ ] SETUP_VERSION 2.3.0, GPU checklist

## Phase 5 - Interior Designer

- [ ] `--brief` / `brief.json`; DesignSpec schema
- [ ] `data/styles/`, `data/room_programs.json`, `data/furniture_types.json`; pinned materials + LICENSES
- [ ] Intent solver + circulation check + unsatisfied feedback; rules intents from today's templates
- [ ] Asset-aware sizing; DesignSpec → Blender materials
- [ ] Infinigen evaluation write-up (C9); optional enrichment (off)
- [ ] Acceptance: layout checks = 0 on generator seeds + smoke plan
- [ ] SETUP_VERSION 2.4.0, GPU checklist

## Phase 6 - Render Director

- [ ] Multilayer EXR (`media_type`), passes, `use_persistent_data`; full-resolution depth
- [ ] `data/render_profiles.json`; `set_camera`, `set_exposure`, auto-exposure; per-room previews
- [ ] VLM critique 1–5 loop; polish ladder; pluggable polish interface
- [ ] Harness final render + polish after export
- [ ] SETUP_VERSION 2.5.0, GPU checklist

## Phase 7 - flexibility

- [ ] `agent.sh edit/diff/revert`, confirmations; batch mode
- [ ] Sleep mode (off by default); MCP stdio (off); experience memory (off)
- [ ] Edit → diff → revert round trip + identical replay
- [ ] SETUP_VERSION 2.6.0, GPU checklist

## Phase 8 - evaluation

- [ ] `app.evaluate` metrics; rules vs agentic → `tests/results.md`
- [ ] Static HTML run viewer
- [ ] README, CHANGELOG, SETUP_VERSION 3.0.0; Definition of done checklist
