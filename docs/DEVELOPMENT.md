# Development

The pod still gets **one file**: `dist/setup.sh`. Upload it to the pod as `/workspace/setup.sh` and run
it exactly as before. That file is **generated**. Edit the sources, then rebuild it.

## Layout

| Path | What |
|---|---|
| `setup.sh` | the installer in **source form**: same steps, switches and pins as the pod file, but every embedded file is a tagged stdin redirect from `src/` (needs `src/` next to it) |
| `src/app/` | the pipeline package (27 modules, 3 lock files, `env.sh` template) - written to `/workspace/app` |
| `src/scripts/` | `run.sh`, `start.sh`, `agent.sh` templates (`__WS__`, `__BLV__`, `__APT__` are filled in by setup.sh) |
| `src/installer/` | scripts setup.sh runs during install (model download, config writer, agent model download, GEN3D build + download) |
| `dist/setup.sh` | **generated** single file for the pod (`make bundle`); committed so it can be downloaded directly |
| `tools/bundle.py` | source form → `dist/setup.sh`; `--check` fails when `dist/` is stale |
| `tools/check_identical.py` | compares `src/`, `dist/setup.sh` and the `--code-only` output with setup.sh 2.0.3 |
| `tools/extract.py` | the one-off Phase 1 split (kept for provenance; `bundle` is its exact inverse) |
| `tools/shellcheck_all.sh` | shellcheck on both installers, the GEN3D build script and the generated scripts |
| `tests/` | pytest (CPU): the `app.testplans unit` checks, bundle consistency |
| `requirements-dev.lock`, `requirements-bpy.lock` | dev-only pins (never installed on the pod) |
| `ruff.toml` | lint rules; a per-file baseline for the 2.0.3 code, which gets fixed in Phase 2 |
| `docs/` | `ARCHITECTURE.md` (2.0.3 as found), `PLAN.md`, `PROGRESS.md`, this file |

## How the bundle works

In `setup.sh` (source form):

```bash
cat > "$APP/common.py" < "$SRC/app/common.py"  # bundle:heredoc __END_OF_COMMON_PY__
```

In `dist/setup.sh`, built by `tools/bundle.py`:

```bash
cat > "$APP/common.py" <<'__END_OF_COMMON_PY__'
<src/app/common.py byte for byte>
__END_OF_COMMON_PY__
```

* Lines ending in `# bundle:source-only` are left out of the bundle. These are the `SRC=` line, its
  guard and a header note.
* The bundler refuses three kinds of input: a `$SRC` reference without a tag, a file that contains its
  own delimiter line, and a file that does not end with a newline.

## Commands

```bash
make dev dev-bpy       # .venv-dev (Python 3.12 + CPU pipeline deps + pytest + ruff), .venv-bpy (bpy 5.2.2, Python 3.13)
make bundle            # after every change in setup.sh or src/
make check             # bundle up to date + ruff + shellcheck + pytest
make check-identical   # byte identity with 2.0.3 (Phase 1 acceptance; later phases list what changed)
```

* `make test` runs the Blender checks in `.venv-bpy`, or in `PIPE_BLENDER` if that is set. If neither
  exists, the checks are skipped.
* shellcheck 0.11.0 is expected on `PATH`, or set `SHELLCHECK=/path/to/shellcheck`.
* Everything runs on the CPU with `PIPE_WS` set to a temporary folder. Nothing touches `/workspace`.

## Rules

* Never reformat `src/` automatically. The pod file embeds it byte for byte, and diffs must stay
  reviewable.
* Every change to `setup.sh` or `src/` is followed by `make bundle`. `make check` fails when
  `dist/setup.sh` is out of date.
