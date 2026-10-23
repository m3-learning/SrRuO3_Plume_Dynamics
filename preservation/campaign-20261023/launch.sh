#!/usr/bin/env bash
set -euo pipefail
export PATH="/home/srruo3-venv/bin:/home/tools:$PATH"
export JUPYTER_DATA_DIR=/home/srruo3-runtime/jupyter
export JUPYTER_CONFIG_DIR=/home/srruo3-runtime/jupyter-config
export JUPYTER_RUNTIME_DIR=/home/srruo3-runtime/jupyter-runtime
export IPYTHONDIR=/home/srruo3-runtime/ipython
export MPLCONFIGDIR=/home/srruo3-runtime/matplotlib
export MPLBACKEND=module://matplotlib_inline.backend_inline
export SRRUO3_SHARED_CACHE=/home/srruo3-server-rerun/cache
mkdir -p /home/srruo3-server-rerun
cd /home/SrRuO3_Plume_Dynamics
python /home/execute-srruo3-server-only.py "$@" 2>&1 | tee -a /home/srruo3-server-rerun/execution.log
