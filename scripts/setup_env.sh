#!/usr/bin/env bash
# Create a reproducible Python virtual environment using uv.
set -euo pipefail

VENV_DIR=${VENV_DIR:-.venv}

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required but not installed. Install it first: https://docs.astral.sh/uv/"
  exit 1
fi

echo "Creating uv-managed virtual environment at ${VENV_DIR}"
uv venv "${VENV_DIR}"

if [ -f pyproject.toml ]; then
  echo "Installing project dependencies from pyproject.toml"
  uv pip install -e . --python "${VENV_DIR}/bin/python"
else
  echo "pyproject.toml not found; installing minimal runtime dependencies"
  uv pip install sentencepiece huggingface-hub pyarrow pyyaml tqdm --python "${VENV_DIR}/bin/python"
fi

echo "Virtual environment ready. Activate with: source ${VENV_DIR}/bin/activate"
