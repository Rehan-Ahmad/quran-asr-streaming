#!/usr/bin/env bash
# Activate the project's .venv
set -euo pipefail
if [ -d ".venv" ] && [ -x ".venv/bin/activate" ] || [ -f ".venv/bin/activate" ]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
  echo "Activated .venv -> $(which python) ($(python -V 2>&1))"
else
  echo ".venv not found or missing activate script. Create the venv first." >&2
  exit 1
fi
