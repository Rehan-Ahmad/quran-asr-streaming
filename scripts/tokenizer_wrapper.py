#!/usr/bin/env python3
"""
Wrapper to build a subword tokenizer using NeMo's process_asr_text_tokenizer.py.
Clones NeMo repo if not present, then runs the tokenizer script.
"""
import argparse
import subprocess
import os
import sys
from pathlib import Path


def ensure_nemo(nemo_dir: str):
    """Clone NeMo repo if not already present."""
    if os.path.isdir(nemo_dir):
        print(f"NeMo already present at {nemo_dir}")
        return
    
    print(f"Cloning NeMo into {nemo_dir} (this may take a while)...")
    try:
        subprocess.check_call([
            "git", "clone", "--depth", "1",
            "https://github.com/NVIDIA/NeMo",
            nemo_dir
        ])
        print(f"NeMo cloned successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error cloning NeMo: {e}")
        sys.exit(1)


def run_tokenizer(
    input_manifest: str,
    out_dir: str,
    vocab_size: int = 8000,
    model_type: str = "bpe",
    nemo_dir: str = "tools/nemo"
):
    """Run NeMo's tokenizer script."""
    ensure_nemo(nemo_dir)
    
    script = os.path.join(nemo_dir, "scripts", "tokenizers", "process_asr_text_tokenizer.py")
    if not os.path.isfile(script):
        print(f"ERROR: Tokenizer script not found at {script}")
        sys.exit(1)
    
    os.makedirs(out_dir, exist_ok=True)
    
    cmd = [
        sys.executable, script,
        "--input_manifest", input_manifest,
        "--output_dir", out_dir,
        "--vocab_size", str(vocab_size),
        "--tokenizer_model_type", model_type
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
        description="Build a subword tokenizer for Quran ASR using NeMo."
    )
    parser.add_argument(
        "--input-manifest", required=True,
        help="Path to input manifest CSV (with 'text' column)"
    )
    parser.add_argument(
        "--out-dir", required=True,
        help="Output directory for tokenizer artifacts"
    )
    parser.add_argument(
        "--vocab-size", type=int, default=8000,
        help="Vocabulary size for BPE tokenizer. Default: 8000"
    )
    parser.add_argument(
        "--model-type", default="bpe",
        choices=["bpe", "unigram"],
        help="Tokenizer type (bpe or unigram). Default: bpe"
    )
    parser.add_argument(
        "--nemo-dir", default="tools/nemo",
        help="Path to NeMo repository. Default: tools/nemo"
    )
    
    args = parser.parse_args()
    
    # Validate input manifest exists
    if not os.path.isfile(args.input_manifest):
        print(f"ERROR: Input manifest not found: {args.input_manifest}")
        sys.exit(1)
    
    run_tokenizer(
        args.input_manifest,
        args.out_dir,
        args.vocab_size,
        args.model_type,
        args.nemo_dir
    )


if __name__ == "__main__":
    main()
