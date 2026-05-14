#!/usr/bin/env python3
"""
Wrapper to prepare a subword tokenizer using NeMo's process_asr_text_tokenizer.py
It will clone NeMo into `tools/nemo` if not present, and run the tokenizer script.
"""
import argparse
import subprocess
import os
import sys

ROOT = os.path.dirname(os.path.dirname(__file__))
NEMO_DIR = os.path.join(ROOT, "tools", "nemo")

def ensure_nemo():
    if not os.path.isdir(NEMO_DIR):
        print("Cloning NeMo (this may take a while)...")
        subprocess.check_call(["git", "clone", "https://github.com/NVIDIA/NeMo", NEMO_DIR])

def run_tokenizer(input_manifest, out_dir, vocab_size=8000, model_type="bpe"):
    ensure_nemo()
    script = os.path.join(NEMO_DIR, "scripts", "tokenizers", "process_asr_text_tokenizer.py")
    if not os.path.isfile(script):
        print(f"Tokenizer script not found at {script}")
        sys.exit(1)

    cmd = [sys.executable, script,
           "--input_manifest", input_manifest,
           "--output_dir", out_dir,
           "--vocab_size", str(vocab_size),
           "--tokenizer_model_type", model_type]
    print("Running:", " ".join(cmd))
    subprocess.check_call(cmd)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input-manifest", required=True, dest="input_manifest")
    p.add_argument("--out-dir", required=True, dest="out_dir")
    p.add_argument("--vocab-size", type=int, default=8000)
    p.add_argument("--model-type", default="bpe")
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    run_tokenizer(args.input_manifest, args.out_dir, args.vocab_size, args.model_type)

if __name__ == '__main__':
    main()
