#!/usr/bin/env bash
set -euo pipefail

PY=python3
if ! command -v $PY &> /dev/null; then
  echo "python3 not found"
  exit 1
fi

if [ -d "venv" ]; then
  echo "Using existing venv/; activate it with: source venv/bin/activate"
  exit 0
fi

$PY -m venv venv
source venv/bin/activate
pip install -U pip
pip install -r requirements.txt

echo "Virtualenv created and requirements installed. Activate with: source venv/bin/activate"
