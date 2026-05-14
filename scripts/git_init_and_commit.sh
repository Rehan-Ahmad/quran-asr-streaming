#!/usr/bin/env bash
set -euo pipefail

if [ ! -d .git ]; then
  git init
  git add .
  git commit -m "Initial scaffold for Quran ASR project"
  echo "Repository initialized and initial commit created."
else
  echo "Git repo already initialized."
fi
