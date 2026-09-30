# Environment for the pipeline. run.sh and start.sh load it: source __WS__/app/env.sh
export WS=__WS__ PIPE_WS=__WS__
export HF_HOME=__WS__/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export UV_CACHE_DIR=__WS__/cache/uv UV_PYTHON_INSTALL_DIR=__WS__/opt/python PIP_CACHE_DIR=__WS__/cache/pip XDG_CACHE_HOME=__WS__/cache
export PIPE_BLENDER=__WS__/opt/blender-__BLV__/blender
export LD_LIBRARY_PATH=__WS__/opt/libredwg/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export PATH=__WS__/venv/bin:__WS__/opt/libredwg/bin:$PATH
export PYTHONPATH=__WS__${PYTHONPATH:+:$PYTHONPATH}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1 VLLM_CACHE_ROOT=__WS__/cache/vllm
