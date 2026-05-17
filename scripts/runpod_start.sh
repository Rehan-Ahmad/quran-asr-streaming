#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export UV_LINK_MODE="${UV_LINK_MODE:-copy}"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required but not found in PATH" >&2
  exit 1
fi

echo "Syncing the environment from uv.lock..."
uv sync --frozen --no-install-project

echo "Verifying the core runtime imports..."
.venv/bin/python - <<'PY'
import importlib
for name in [
    'nemo',
    'lightning.pytorch',
    'omegaconf',
    'hydra.core.utils',
    'lhotse',
    'librosa',
    'pyannote.core',
    'braceexpand',
    'nv_one_logger.api.config',
]:
    importlib.import_module(name)
    print(f'{name}: OK')
PY

if [[ "${1:-}" == "--train" ]]; then
  echo "Launching the NeMo finetune job..."
  exec .venv/bin/python scripts/run_nemo_finetune_local.py
fi

echo "Environment is ready. Re-run with --train to start the finetune job."