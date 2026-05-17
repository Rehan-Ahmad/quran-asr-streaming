#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${ROOT_DIR}/.venv/bin/python"
TRAIN_CONFIG="${ROOT_DIR}/configs/train_config.yaml"
RUNNER="${ROOT_DIR}/scripts/run_nemo_finetune_local.py"
PRETRAINED="nvidia/nemotron-speech-streaming-en-0.6b"
SMOKE_ONLY=0
SMOKE_BATCH_SIZE=2
SMOKE_MAX_EPOCHS=1
SMOKE_ACCUM=1

usage() {
  cat <<EOF
Usage: $0 [--train-config PATH] [--pretrained NAME] [--smoke-batch-size N] [--smoke-max-epochs N] [--accumulate-grad-batches N] [--smoke-only]

Runs a preflight check, creates a temporary smoke-test config copy, and then launches the full NeMo finetune workflow.
The tracked YAML files are never rewritten by this wrapper.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --train-config)
      TRAIN_CONFIG="$(realpath "$2")"
      shift 2
      ;;
    --pretrained)
      PRETRAINED="$2"
      shift 2
      ;;
    --smoke-batch-size)
      SMOKE_BATCH_SIZE="$2"
      shift 2
      ;;
    --smoke-max-epochs)
      SMOKE_MAX_EPOCHS="$2"
      shift 2
      ;;
    --accumulate-grad-batches)
      SMOKE_ACCUM="$2"
      shift 2
      ;;
    --smoke-only)
      SMOKE_ONLY=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ ! -x "$PYTHON" ]]; then
  echo "Missing virtualenv interpreter: $PYTHON" >&2
  exit 1
fi

TMP_ROOT="$(mktemp -d)"
cleanup() {
  rm -rf "$TMP_ROOT"
}
trap cleanup EXIT

SMOKE_TRAIN_CONFIG="$TMP_ROOT/train_config_smoke.yaml"
SMOKE_NEMO_CONFIG="$TMP_ROOT/conf/asr_finetune/speech_to_text_finetune.yaml"
mkdir -p "$(dirname "$SMOKE_NEMO_CONFIG")"

cat <<EOF
Preflight: checking GPU availability, manifests, and tokenizer paths.
EOF

"$PYTHON" - "$TRAIN_CONFIG" <<'PY'
import sys
from pathlib import Path
import yaml
import torch

train_config = Path(sys.argv[1])
with train_config.open('r', encoding='utf-8') as handle:
    cfg = yaml.safe_load(handle)

device_count = torch.cuda.device_count()
print(f'GPUs detected: {device_count}')
if device_count <= 0:
    raise SystemExit('No CUDA GPUs detected. Abort per workflow requirements.')

model = cfg.get('model', {})
for key in ('train_ds', 'validation_ds', 'test_ds'):
    ds = model.get(key, {})
    manifest = ds.get('manifest_filepath')
    if manifest and not Path(manifest).exists():
        raise SystemExit(f'Missing manifest: {manifest}')

tokenizer_dir = Path(model.get('tokenizer', {}).get('dir', ''))
if not tokenizer_dir.exists():
    raise SystemExit(f'Missing tokenizer directory: {tokenizer_dir}')

base_nemo = cfg.get('init_from_nemo_model')
if base_nemo:
    if not Path(base_nemo).exists():
        raise SystemExit(f'Missing base .nemo artifact: {base_nemo}')
else:
    pretrained = cfg.get('init_from_pretrained_model')
    if not pretrained:
        raise SystemExit('Neither init_from_nemo_model nor init_from_pretrained_model is set.')
    print(f'Using pretrained model: {pretrained}')
PY

if [[ "$SMOKE_ONLY" -eq 1 ]]; then
  echo "Smoke-only mode requested; full training will be skipped."
fi

echo "Smoke test runner:"
current_batch="$SMOKE_BATCH_SIZE"
current_accum="$SMOKE_ACCUM"
smoke_status=1
for attempt in 1 2 3; do
  "$PYTHON" - "$TRAIN_CONFIG" "$SMOKE_TRAIN_CONFIG" "$SMOKE_MAX_EPOCHS" "$current_batch" "$current_accum" <<'PY'
import sys
from pathlib import Path
import yaml

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
smoke_max_epochs = int(sys.argv[3])
smoke_batch_size = int(sys.argv[4])
smoke_accum = int(sys.argv[5])

with src.open('r', encoding='utf-8') as handle:
    cfg = yaml.safe_load(handle)

cfg.setdefault('trainer', {})
cfg['trainer']['max_epochs'] = smoke_max_epochs
cfg['trainer']['max_steps'] = -1
cfg['trainer']['accumulate_grad_batches'] = smoke_accum
cfg['trainer']['val_check_interval'] = 1.0

model = cfg.setdefault('model', {})
for split in ('train_ds', 'validation_ds', 'test_ds'):
    ds = model.get(split)
    if isinstance(ds, dict) and ds.get('batch_size'):
        ds['batch_size'] = smoke_batch_size if split != 'test_ds' else max(1, smoke_batch_size)

dst.parent.mkdir(parents=True, exist_ok=True)
with dst.open('w', encoding='utf-8') as handle:
    yaml.safe_dump(cfg, handle, sort_keys=False)
PY
  if "$PYTHON" "$RUNNER" --train-config "$SMOKE_TRAIN_CONFIG" --nemo-config "$SMOKE_NEMO_CONFIG" --pretrained "$PRETRAINED"; then
    smoke_status=0
    break
  fi

  echo "Smoke run failed on attempt ${attempt}."
  if [[ "$current_batch" -gt 1 ]]; then
    current_batch=$(( (current_batch + 1) / 2 ))
    echo "Retrying smoke with batch size ${current_batch}."
  elif [[ "$current_accum" -lt 2 ]]; then
    current_accum=2
    echo "Retrying smoke with accumulate_grad_batches=${current_accum}."
  else
    echo "Smoke test failed after retries." >&2
    break
  fi
done

if [[ "$smoke_status" -ne 0 ]]; then
  exit 1
fi

if [[ "$SMOKE_ONLY" -eq 1 ]]; then
  exit 0
fi

echo "Launching full training using the tracked config template."
"$PYTHON" "$RUNNER" --train-config "$TRAIN_CONFIG" --nemo-config "$SMOKE_NEMO_CONFIG" --pretrained "$PRETRAINED"