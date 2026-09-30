#!/usr/bin/env bash
# After a pod stop/restart the container disk is new (apt packages gone) but __WS__ is kept.
# This puts the small system packages back and checks GPU, PyTorch and Blender. Time: ~1-2 min.
set -Eeuo pipefail
trap 'echo "ERROR: start.sh stopped at line $LINENO: $BASH_COMMAND"' ERR
MISSING=""
for p in __APT__; do dpkg -s "$p" >/dev/null 2>&1 || MISSING="$MISSING $p"; done
# shellcheck disable=SC2086  # MISSING is a space-separated package list
if [[ -n "$MISSING" ]]; then apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends $MISSING; fi
source __WS__/app/env.sh
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
python -c "import torch; assert torch.cuda.is_available(), 'no GPU for PyTorch'; print('PyTorch OK', torch.__version__)"
"$PIPE_BLENDER" -b --factory-startup --python-expr "import bpy; print('Blender OK', bpy.app.version_string)" 2>/dev/null | grep "Blender OK"
rm -f __WS__/cache/llm_server.pid   # an LLM server from before the restart is gone; its PID may be reused now
if [[ -x __WS__/venv-llm/bin/vllm ]]; then
  __WS__/venv-llm/bin/python -c "import vllm, torch; print('vLLM OK', vllm.__version__, '| torch', torch.__version__, '| GPU', torch.cuda.is_available())"
else
  echo "vLLM not installed (SKIP_AGENT_LLM=1): agent.sh runs with --backend rules"
fi
echo "Ready. Put plans in __WS__/inputs and run:  cd __WS__ && ./run.sh __WS__/inputs/<plan file>"
echo "Agentic run (LLM checks and repairs):         cd __WS__ && ./agent.sh __WS__/inputs/<plan file>"
