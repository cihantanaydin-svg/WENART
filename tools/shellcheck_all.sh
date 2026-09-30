#!/usr/bin/env bash
# Lint every shell file with shellcheck: both installers, the GEN3D build script, the env/run/start/agent templates
# as setup.sh writes them (after the __WS__ substitution, from a --code-only run into a temp folder).
set -Eeuo pipefail
SC="${SHELLCHECK:-shellcheck}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
"$SC" --version | sed -n 2p
T="$(mktemp -d)"
trap 'rm -rf "$T"' EXIT
WS="$T/ws" bash "$ROOT/dist/setup.sh" --code-only > /dev/null
fail=0
check() { local what="$1 ${*: -1}"; if "$@"; then echo "ok    ${what#"$SC "}"; else echo "FAIL  ${what#"$SC "}"; fail=1; fi; }
check "$SC" -f gcc "$ROOT/setup.sh"
check "$SC" -f gcc "$ROOT/dist/setup.sh"
check "$SC" -s bash -f gcc "$ROOT/src/installer/gen3d_build.sh"
check "$SC" -s bash -f gcc "$T/ws/app/env.sh"
for f in run.sh start.sh agent.sh; do
  check bash -n "$T/ws/$f"
  check "$SC" -f gcc -e SC1091 "$T/ws/$f"
done
check "$SC" -f gcc "$ROOT/tools/shellcheck_all.sh"
exit "$fail"
