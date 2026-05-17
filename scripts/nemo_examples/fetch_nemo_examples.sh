#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
OUT_DIR="$ROOT_DIR/scripts/nemo_examples"
mkdir -p "$OUT_DIR"

declare -A files
files[
  speech_to_text_finetune.py]=https://raw.githubusercontent.com/NVIDIA-NeMo/NeMo/r2.6.0/examples/asr/speech_to_text_finetune.py
files[
  nemo_run_helper.py]=https://raw.githubusercontent.com/NVIDIA-NeMo/NeMo/r2.6.0/examples/asr/nemo_run_helper.py
files[
  slurm_example.sh]=https://raw.githubusercontent.com/NVIDIA-NeMo/NeMo/r2.6.0/examples/asr/slurm_example.sh

for name in "${!files[@]}"; do
  url="${files[$name]}"
  out="$OUT_DIR/$name"
  echo "Fetching $name from $url"
  curl -fsSL "$url" -o "$out"
  chmod +x "$out" || true
done

echo "Fetched ${#files[@]} files into $OUT_DIR"
