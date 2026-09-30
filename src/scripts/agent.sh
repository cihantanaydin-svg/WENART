#!/usr/bin/env bash
# Agentic run: a local open-weights LLM (vLLM, started and stopped automatically) drives the pipeline, checks the
# results, repairs problems with validated patches and delivers __WS__/outputs/<job>/final/
# (apartment.blend primary, apartment.glb, apartment.usdc, previews/, overlay.png, report.md).
#   ./agent.sh __WS__/inputs/plan.pdf                        LLM agent (settings: config.json llm.* and agent.*)
#   ./agent.sh __WS__/inputs/plan.pdf --backend rules        no LLM: fixed order + deterministic repairs
#   ./agent.sh --replay __WS__/outputs/<job>/agent_log.jsonl  replay the same tool calls into a new job folder
set -Eeuo pipefail
trap 'echo "ERROR: agent.sh stopped at line $LINENO: $BASH_COMMAND"' ERR
source __WS__/app/env.sh
if [[ $# -lt 1 ]]; then sed -n '2,7p' "$0"; exit 1; fi
mkdir -p __WS__/logs
cd __WS__
cleanup() { python -m app.llm_server stop >/dev/null 2>&1 || true; }   # never leave a server holding the GPU
trap cleanup EXIT
LOGF="__WS__/logs/agent_$(date +%Y%m%d_%H%M%S).log"
if [[ "$1" == "--replay" ]]; then
  shift
  python -m app.agent replay "$@" 2>&1 | tee -a "$LOGF"
else
  python -m app.agent run "$@" 2>&1 | tee -a "$LOGF"
fi
