#!/usr/bin/env bash
# Run with bash oct.sh; no chmod or environment activation is required.
set -euo pipefail
cd "$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
env_python="$PWD/.venv-dense/bin/python"
if [[ "${1:-}" == "setup" ]]; then
  bootstrap_python="${PYTHON:-python3}"
  "$bootstrap_python" -c 'import sys; assert (3,10) <= sys.version_info[:2] <= (3,14), "Use Python 3.10-3.14 (3.12 recommended)"'
  "$bootstrap_python" -m venv .venv-dense
  "$env_python" -m pip install --upgrade pip
  "$env_python" -m pip install -r requirements.txt
  "$env_python" -m pip check
  echo 'Setup complete. Next: bash oct.sh test'
  exit 0
fi
if [[ ! -x "$env_python" ]]; then
  echo 'Run this once first: bash oct.sh setup' >&2
  exit 1
fi
export OCT_COMMAND_PREFIX='bash oct.sh'
exec "$env_python" oct.py "$@"
