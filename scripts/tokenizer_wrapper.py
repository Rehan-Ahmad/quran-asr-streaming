#!/usr/bin/env python3
"""
Wrapper to build a subword tokenizer using NeMo's process_asr_text_tokenizer.py.
Uses the local copy of the NeMo tokenizer script.
"""
import argparse
import subprocess
import os
import sys
from pathlib import Path


def run_tokenizer(
    input_manifest: str,
    out_dir: str,
    vocab_size: int = 8000,
    model_type: str = "bpe"
):
    """Run the local NeMo tokenizer script."""
    
    # Get the script path relative to this wrapper
    script_dir = os.path.dirname(os.path.abspath(__file__))
    script = os.path.join(script_dir, "process_asr_text_tokenizer.py")
    
    if not os.path.isfile(script):
        print(f"ERROR: Tokenizer script not found at {script}")
        sys.exit(1)
    
    os.makedirs(out_dir, exist_ok=True)
    
    cmd = [
        sys.executable, script,
        "--data_file", input_manifest,
        "--data_root", out_dir,
        "--vocab_size", str(vocab_size),
        "--tokenizer", "spe",
        "--spe_type", model_type
    ]
    
    print(f"Running tokenizer with command:")
    print(f"  {' '.join(cmd)}")
    
    try:
        subprocess.check_call(cmd)
        print(f"Tokenizer completed successfully. Output in: {out_dir}")
    except subprocess.CalledProcessError as e:
        print(f"Tokenizer failed with error: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Build a subword tokenizer for Quran ASR using NeMo's local tokenizer script."
    )
    parser.add_argument(
        "--input-manifest", required=True,
        help="Path to input manifest or text file (with 'text' column for manifests)"
    )
    parser.add_argument(
        "--out-dir", required=True,
        help="Output directory for tokenizer artifacts"
    )
    parser.add_argument(
        "--vocab-size", type=int, default=8000,
        help="Vocabulary size for tokenizer. Default: 8000"
    )
    parser.add_argument(
        "--model-type", default="bpe",
        choices=["bpe", "unigram", "char", "word"],
        help="Tokenizer type. Default: bpe"
    )
    
    args = parser.parse_args()
    
    # Validate input manifest exists
    if not os.path.isfile(args.input_manifest):
        print(f"ERROR: Input manifest/file not found: {args.input_manifest}")
        sys.exit(1)
    
    run_tokenizer(
        args.input_manifest,
        args.out_dir,
        args.vocab_size,
        args.model_type
    )


if __name__ == "__main__":
    main()
