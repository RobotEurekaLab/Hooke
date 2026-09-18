#!/usr/bin/env bash
# Run from any directory while preserving the source plugins' working directory.
set -euo pipefail

hooke_repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
hooke_source_python="${HOOKE_SOURCE_PYTHON:-$hooke_repo_root/.venv/bin/python}"
if [[ ! -x "$hooke_source_python" ]]; then
  printf 'Source Python is unavailable: %s\nSet HOOKE_SOURCE_PYTHON to your configured Python executable.\n' "$hooke_source_python" >&2
  exit 2
fi

export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
cd -- "$hooke_repo_root/Hooke"
if [[ "${1:-}" == "--doctor" ]]; then
  shift
  exec "$hooke_source_python" -m backends.doctor "$@"
fi
if [[ "${1:-}" == "--configure-isaac" ]]; then
  shift
  exec "$hooke_source_python" -m backends.config "$@"
fi
exec "$hooke_source_python" -m webui.server "$@"
