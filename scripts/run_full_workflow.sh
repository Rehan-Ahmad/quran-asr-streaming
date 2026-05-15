#!/usr/bin/env bash
# Orchestrate reproducible workflow: uv env -> download -> tokenizer -> training template
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)
VENV="${ROOT_DIR}/.venv"
PYTHON="${VENV}/bin/python"

usage(){
  cat <<EOF
Usage: $0 --output-dir RUN_DIR [--percent PERCENT] [--vocab-size N] [--text-column COL] [--dry-run]

Creates a uv venv (if missing), downloads requested percent of Tadabur, runs tokenizer prep, and prints a Nemotron training template.
EOF
}

PERCENT=1
RUN_DIR="runs/workflow_out"
VOCAB=8000
TEXT_COLUMN="text_ar_uthmani"
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --output-dir) RUN_DIR=$(realpath "$2"); shift 2;;
    --percent) PERCENT="$2"; shift 2;;
    --vocab-size) VOCAB="$2"; shift 2;;
    --text-column) TEXT_COLUMN="$2"; shift 2;;
    --dry-run) DRY_RUN=1; shift;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown arg: $1"; usage; exit 1;;
  esac
done

DATA_DIR="$RUN_DIR/data"
TOKENIZER_DIR="$RUN_DIR/tokenizer"
TRAIN_DIR="$RUN_DIR/training"

mkdir -p "$DATA_DIR" "$TOKENIZER_DIR" "$TRAIN_DIR"

if [ ! -d "$VENV" ]; then
  echo "Creating virtualenv at $VENV"
  bash "$ROOT_DIR/scripts/setup_env.sh"
fi

if [ "$DRY_RUN" -eq 1 ]; then
  echo "DRY RUN: will not execute training. Will run download and tokenizer prep in dry-run mode."
fi

echo "Step 1: Downloading dataset (percent=${PERCENT}) into ${DATA_DIR}"
"$PYTHON" "$ROOT_DIR/scripts/download_dataset.py" --percent "$PERCENT" --output-dir "$DATA_DIR" || exit 1

# Determine how many files were downloaded
MANIFEST="$DATA_DIR/download_manifest.json"
NUM_FILES=$($PYTHON - <<PY
import json
with open('$MANIFEST') as f:
    m=json.load(f)
print(len(m['files']))
PY
)

echo "Step 2: Prepare tokenizer (vocab=${VOCAB}, text_column=${TEXT_COLUMN}, num_files=${NUM_FILES})"
CMD=("$PYTHON" "$ROOT_DIR/scripts/run_tokenizer_repro.py" --output-dir "$TOKENIZER_DIR" --vocab-size "$VOCAB" --text-column "$TEXT_COLUMN" --num-files "$NUM_FILES" --local-data-dir "$DATA_DIR")
if [ "$DRY_RUN" -eq 1 ]; then
  echo "Would run: ${CMD[*]}"
else
  "${CMD[@]}"
fi

echo "Step 3: Nemotron training (template)"
TRAIN_CFG="$ROOT_DIR/configs/nemotron_streaming.yaml"
cat <<EOF
To run Nemotron streaming training follow the model's recommended environment.

Example (edit $TRAIN_CFG first):

1) Activate environment:
   source $VENV/bin/activate

2) Run distributed training (example, edit for your cluster):
   torchrun --nproc_per_node=1 ${ROOT_DIR}/scripts/train_nemotron.py --config $TRAIN_CFG

Or use the official Nemotron training launcher as documented at:
  https://huggingface.co/nvidia/nemotron-speech-streaming-en-0.6b

This script wrote the dataset manifest into: $DATA_DIR
This script wrote the tokenizer and a reproduce manifest into: $TOKENIZER_DIR
EOF
