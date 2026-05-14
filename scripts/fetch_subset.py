#!/usr/bin/env python3
"""
Download a small subset of the Tadabur dataset and generate a manifest CSV.
Uses HuggingFace datasets library to stream and fetch audio + text pairs.
"""
import argparse
import os
import csv
from datasets import load_dataset
from pathlib import Path


def fetch_subset(out_dir: str, percent: float = 1.0, split: str = "train"):
    """
    Fetch a subset of the Tadabur dataset.
    
    Args:
        out_dir: Output directory for manifest and audio files.
        percent: Percentage of dataset to fetch (1-100).
        split: Dataset split to fetch ("train", "validation", "test").
    """
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"Loading Tadabur dataset split '{split}'...")
    ds = load_dataset("FaisaI/tadabur", split=split, trust_remote_code=True)
    
    total = len(ds)
    if percent < 100:
        take = max(1, int(total * (percent / 100.0)))
    else:
        take = total
    
    print(f"Dataset size: {total}. Taking {take} samples ({percent}%).")
    
    manifest_path = os.path.join(out_dir, "manifest.csv")
    print(f"Writing manifest to: {manifest_path}")
    
    with open(manifest_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["audio_filepath", "duration", "text"])
        writer.writeheader()
        
        for i, item in enumerate(ds):
            if i >= take:
                break
            
            # Extract audio and text from the dataset item
            # Tadabur structure: { "audio": {...}, "text": "..." }
            audio_data = item.get("audio")
            text = item.get("text", "")
            
            if not audio_data:
                print(f"Skipping sample {i}: no audio")
                continue
            
            # audio_data is typically { "path": "...", "array": [...], "sampling_rate": ... }
            audio_path = audio_data.get("path", "")
            duration = audio_data.get("duration", "")
            
            if not audio_path:
                print(f"Skipping sample {i}: no audio path")
                continue
            
            writer.writerow({
                "audio_filepath": audio_path,
                "duration": duration,
                "text": text
            })
            
            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1} samples...")
    
    print(f"\nDone! Manifest saved to: {manifest_path}")
    print(f"Total samples in manifest: {i + 1}")


def main():
    parser = argparse.ArgumentParser(
        description="Download a subset of the Tadabur dataset."
    )
    parser.add_argument(
        "--out-dir", required=True,
        help="Output directory for manifest.csv and audio files."
    )
    parser.add_argument(
        "--percent", type=float, default=1.0,
        help="Percentage of dataset to fetch (1-100). Default: 1.0"
    )
    parser.add_argument(
        "--split", default="train",
        help="Dataset split to fetch (train/validation/test). Default: train"
    )
    
    args = parser.parse_args()
    fetch_subset(args.out_dir, args.percent, args.split)


if __name__ == "__main__":
    main()
