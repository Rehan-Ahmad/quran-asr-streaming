#!/usr/bin/env python3
"""
Download a small subset of the Tadabur dataset (audio + text) and generate a manifest CSV.
Uses `datasets` to stream metadata and selectively download audio files.
"""
import argparse
import os
import csv
from datasets import load_dataset


def fetch(out_dir: str, percent: float = 1.0, split="train"):
    os.makedirs(out_dir, exist_ok=True)
    ds = load_dataset("FaisaI/tadabur", split=split)
    total = len(ds)
    take = max(1, int(total * (percent / 100.0))) if percent < 100 else total
    print(f"Dataset size: {total}. Taking {take} ({percent}%).")

    manifest_path = os.path.join(out_dir, "manifest.csv")
    with open(manifest_path, "w", newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=["audio_filepath", "duration", "text"])
        writer.writeheader()
        for i, item in enumerate(ds):
            if i >= take:
                break
            audio = item.get("audio") or item.get("wav") or item.get("speech")
            if audio is None:
                continue
            # audio could be dict with 'path' or 'array' depending on dataset
            path = audio.get("path") if isinstance(audio, dict) else None
            if path is None:
                # attempt to download the file locally via dataset's features
                path = ds._download_and_extract(audio["path"]) if isinstance(audio, dict) and "path" in audio else None
            text = item.get("text", "")
            writer.writerow({"audio_filepath": path or "", "duration": audio.get("duration", "" ) if isinstance(audio, dict) else "", "text": text})

    print("Wrote manifest:", manifest_path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", required=True)
    p.add_argument("--percent", type=float, default=1.0, help="Percent of dataset to fetch (1-100)")
    p.add_argument("--split", default="train")
    args = p.parse_args()
    fetch(args.out_dir, args.percent, args.split)

if __name__ == '__main__':
    main()
