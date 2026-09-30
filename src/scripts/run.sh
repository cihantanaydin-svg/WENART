#!/usr/bin/env bash
# Run the whole pipeline on one plan (PDF, DWG, DXF, JPG/PNG photo). Outputs: __WS__/outputs/<job>/
#   ./run.sh __WS__/inputs/plan.pdf                  full quality (1920x1080, 3 views per main room)
#   ./run.sh __WS__/inputs/plan.pdf --profile quick  fast check (640x360, 1 view per room)
#   ./run.sh __WS__/inputs/plan.pdf --job-dir __WS__/outputs/<job> --from-stage render   resume a job
#   Stages: parse vlm plan layout scene render polish export report.  --no-polish skips the AI polish.
#   Deliverable: __WS__/outputs/<job>/final/ (apartment.blend/.glb/.usdc). Agentic run with repairs: ./agent.sh
set -Eeuo pipefail
trap 'echo "ERROR: run.sh stopped at line $LINENO: $BASH_COMMAND"' ERR
source __WS__/app/env.sh
if [[ $# -lt 1 ]]; then sed -n '2,7p' "$0"; exit 1; fi
mkdir -p __WS__/logs
cd __WS__
python -m app.main "$@" 2>&1 | tee -a "__WS__/logs/run_$(date +%Y%m%d_%H%M%S).log"
