# Progress

Branch `claude/clever-euler-gcd4qr`. Baseline: `setup.sh` 2.0.3 (commit `3881587`). Plan: `docs/PLAN.md`.
Legend: `[x]` done and verified here (CPU) · `[ ]` open · **UNVERIFIED** = needs the GPU pod.

## Phase 0 - reading and planning (done; go-ahead given 2026-09-30)

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
- [x] Answers to Q1–Q9: "all others ok". Q5 is changed to **free open-weight models only**, with no
      hosted provider. The deliverable is **one setup.sh** for the pod at the end (PLAN §0).
- [ ] Pins JSON from the pod (PLAN §8). Needed for Phase 2 item 7; until it arrives the pins stay
      empty and only warn.

Not verified in Phase 0: anything on a GPU; the image token cost of Qwen3.5; whether xgrammar
accepts our schemas (planned CPU check in Phase 3).

## Phase 1 - make the code editable (done 2026-09-30, waiting for go-ahead)

- [x] Reference: commit `3881587`:setup.sh, sha256 `75431821…c043051`. The check verifies this hash; no git
      tag is needed.
- [x] `tools/extract.py`:
  - [x] the 39 heredoc bodies went to `src/app` (27 .py, 3 .lock, env.sh), `src/scripts` (3) and
        `src/installer` (5)
  - [x] `setup.sh` is now in source form, with `# bundle:heredoc` tags and a source-only `SRC=` line + guard
- [x] `tools/bundle.py` builds `dist/setup.sh` (`--check` fails when it is stale). It refuses untagged `$SRC`
      references, content that contains its own delimiter, and files without a final newline.
- [x] **Byte identity** (`make check-identical` = `tools/check_identical.py --strict`):
  - [x] A: 39/39 `src/` files identical to their heredoc bodies in 2.0.3
  - [x] B: `dist/setup.sh` == 2.0.3 byte for byte (sha256 `75431821…`)
  - [x] C: `--code-only` into the same WS: 34 files identical across 2.0.3, the source form and dist
        (content + executable bit)
  - [x] Negative test: adding one newline to `src/app/textnorm.py` makes A, C and `bundle --check` fail
- [x] pytest (`make test`, 18 tests, 42 s):
  - the 8 `testplans unit` checks, run in a temporary `PIPE_WS` against `src/app`
  - bundle consistency and failure modes; `bash -n`
  - Result: 8/8 PASS with bpy 5.2.2 (Blender check "scene + export + 20 validation checks passed"),
    7 PASS + 1 SKIP without Blender, the same as before the split.
- [x] `requirements-dev.lock` (Python 3.12: the CPU part of the pipeline at the exact versions of
      `requirements.lock`, plus pytest 9.1.1 and ruff 0.16.9) and `requirements-bpy.lock` (bpy 5.2.2, Python 3.13)
- [x] ruff 0.16.9, lint only:
  - explicit rules E4/E7/E9/F/B (ruff 0.16's built-in default enables about 413 rules)
  - `tools/` and `tests/` are clean
  - the 2.0.3 code has a per-file baseline of 99 findings, for Phase 2:
    - 26 E401 multiple imports on one line
    - 16 F401 unused import
    - 13 B905 `zip()` without `strict`
    - 12 E731 lambda assignment
    - 11 B007 unused loop variable
    - 10 E741 ambiguous name
    - 7 B023 closure over a loop variable (plan.py; to review)
    - 2 F841 unused variable
    - 1 B008 function call in a default argument
    - 1 B904 `raise` without `from` inside `except`
- [x] shellcheck 0.11.0 (`make shellcheck`): 0 findings on the source-form and dist installers, the GEN3D build
      script, and the generated env/run/start/agent scripts (+ `bash -n`)
- [x] Makefile (dev, dev-bpy, bundle, check-bundle, check-identical, test, ruff, shellcheck, lint, check),
      `.gitignore`, `docs/DEVELOPMENT.md`; POD_TEST_PLAN now says to upload `dist/setup.sh`; CHANGELOG
      "Unreleased" entry
- [ ] Optional Dockerfile - **not added**. There is no Docker daemon here, so it could not be built or tested,
      and its base image could not be pinned to a verified digest.

Not verified: nothing new needs the GPU. The pod file is byte-identical to 2.0.3, so the 2.0.3 pod results
still apply, and your 2.0.3 rerun is still pending.

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
