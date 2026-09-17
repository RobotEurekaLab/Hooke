#!/usr/bin/env bash
# Native Isaac 4.5 diagnostic. Completion never implies functional qualification.
set -euo pipefail
HOOKE_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
HOOKE_ISAAC_PATH=$(python3 - "$HOOKE_ROOT" <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, str(Path(sys.argv[1]) / 'Hooke'))
from backends.config import isaac_installation
print(isaac_installation())
PY
)
HOOKE_GPU=${HOOKE_ISAAC_RENDER_GPU:-6}
if (( $# < 2 )); then
    echo 'Usage: scripts/isaac_scene_probe.sh SOURCE_ARCHIVE OUTPUT_DIR [--mode preview|controls] [--max-steps N]' >&2
    exit 2
fi
HOOKE_SOURCE=$(realpath -- "$1")
mkdir -p -- "$2"
HOOKE_OUTPUT=$(realpath -- "$2")
shift 2
HOOKE_UUID=$(nvidia-smi -i "$HOOKE_GPU" --query-gpu=uuid --format=csv,noheader)
HOOKE_MEMORY=$(nvidia-smi -i "$HOOKE_GPU" --query-gpu=memory.used --format=csv,noheader,nounits)
(( HOOKE_MEMORY < 2048 )) || { echo 'Selected GPU is busy; select an idle GPU with HOOKE_ISAAC_RENDER_GPU.' >&2; exit 2; }
python3 -c 'import json,sys; json.dump({"status":"RUNNING","parity_qualified":False},open(sys.argv[1],"w"))' "$HOOKE_OUTPUT/result.json"
env -u CONDA_PREFIX -u PYTHONPATH -u PYTHONHOME -u PYTHONEXE \
    CUDA_VISIBLE_DEVICES="$HOOKE_UUID" HOOKE_ISAAC_RENDER_GPU="$HOOKE_GPU" \
    OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1 \
    "$HOOKE_ISAAC_PATH/python.sh" "$HOOKE_ROOT/Hooke/backends/isaac_probe.py" \
    --source "$HOOKE_SOURCE" --output "$HOOKE_OUTPUT" "$@" > "$HOOKE_OUTPUT/run.log" 2>&1
# Kit fast shutdown can hide Python's intended process exit code.
python3 - "$HOOKE_OUTPUT/result.json" <<'PY'
import json, sys
result = json.load(open(sys.argv[1]))
print(json.dumps(result, indent=2))
sys.exit(0 if result.get('status') == 'PROBE_COMPLETED_UNQUALIFIED' else 1)
PY
