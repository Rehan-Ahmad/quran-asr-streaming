#!/usr/bin/env python3
"""
Download a small subset of the Tadabur dataset and generate a NeMo-style manifest.
Uses HuggingFace datasets library to stream and fetch audio + text pairs.
"""
import argparse
import os
import json
import glob
import hashlib
import pyarrow.parquet as pq
from datasets import load_dataset
from pathlib import Path


def fetch_subset(out_dir: str, percent: float = 1.0, split: str = "train", force_download: bool = False, max_samples_arg: int | None = None, cache_dir: str | None = None):
    """
    Fetch a subset of the Tadabur dataset.
    
    Args:
        out_dir: Output directory for manifest and audio files.
        percent: Percentage of dataset to fetch (1-100).
        split: Dataset split to fetch ("train", "validation", "test").
    """
    os.makedirs(out_dir, exist_ok=True)
    
    # Prefer local parquet shards if available to avoid heavy HF downloads.
    # Candidates for local parquet search
    if cache_dir:
        search_paths = [cache_dir]
    else:
        search_paths = []
        search_paths.append(os.path.join("data", "tadabur", "data"))
        search_paths.append(os.path.join(".cache", "huggingface"))
        # search_paths.append(os.path.expanduser(os.path.join("~", ".cache", "huggingface")))
        # Respect HF env vars when present
        hf_home = os.environ.get("HF_HOME")
        if hf_home:
            search_paths.append(hf_home)
        hf_datasets_cache = os.environ.get("HF_DATASETS_CACHE")
        if hf_datasets_cache:
            search_paths.append(hf_datasets_cache)

    parquet_files = []
    for base in search_paths:
        if not base:
            continue
        pattern = os.path.join(base, "**", "train-*.parquet")
        found = glob.glob(pattern, recursive=True)
        for f in found:
            parquet_files.append(os.path.abspath(f))
    # deduplicate likely-duplicate files: use (size, partial-sha1) signature
    seen_sigs = set()
    unique_files = []
    def partial_sha1(path, n=1048576):
        try:
            h = hashlib.sha1()
            with open(path, 'rb') as fh:
                chunk = fh.read(n)
                h.update(chunk)
            return h.hexdigest()
        except Exception:
            return None

    for p in sorted(set(parquet_files)):
        try:
            real = os.path.realpath(p)
            size = os.path.getsize(p)
        except Exception:
            continue
        sig = (size, partial_sha1(p))
        if sig in seen_sigs:
            continue
        seen_sigs.add(sig)
        unique_files.append(p)
    parquet_files = unique_files

    manifest_path = os.path.join(out_dir, f"{split}_manifest.json")
    if parquet_files and not force_download:
        print(f"Found {len(parquet_files)} local parquet shards under cache paths; using local files.")
        written = 0
        total_est = 0
        # estimate total rows by summing first file metadata
        try:
            pf0 = pq.ParquetFile(parquet_files[0])
            total_est = pf0.metadata.num_rows * len(parquet_files)
        except Exception:
            total_est = 0
        max_samples = None
        if max_samples_arg is not None:
            max_samples = int(max_samples_arg)
        elif percent < 100 and total_est > 0:
            max_samples = max(1, int(total_est * (percent / 100.0)))

        with open(manifest_path, "w", encoding="utf-8") as fh:
            for pf_path in parquet_files:
                try:
                    pf = pq.ParquetFile(pf_path)
                except Exception:
                    continue
                # iterate row groups to avoid chunked nested arrays
                for rg in range(pf.num_row_groups):
                    try:
                        table = pf.read_row_group(rg)
                    except Exception:
                        continue
                    d = table.to_pydict()
                    audios = d.get('audio', [])
                    durs = d.get('ayah_duration_s', [])
                    texts = d.get('text_ar_uthmani', []) or d.get('text_ar_simple', []) or []
                    n = max(len(audios), len(texts))
                    for i in range(n):
                        if max_samples is not None and written >= max_samples:
                            print(f"Reached target of {max_samples} samples")
                            return
                        audio_item = audios[i] if i < len(audios) else None
                        audio_path = ''
                        if isinstance(audio_item, dict):
                            bytes_blob = audio_item.get('bytes')
                            fname = audio_item.get('path') or f"sample_{written:06d}.wav"
                            audio_dir = os.path.join(out_dir, 'audio')
                            os.makedirs(audio_dir, exist_ok=True)
                            if bytes_blob:
                                target = os.path.join(audio_dir, fname)
                                try:
                                    with open(target, 'wb') as bf:
                                        bf.write(bytes_blob)
                                    audio_path = target
                                except Exception:
                                    audio_path = fname
                            else:
                                audio_path = fname
                        # Duration – taken from the separate `durs` column (ayah_duration_s)
                        raw_duration = durs[i] if i < len(durs) else ""
                        try:
                            duration = float(raw_duration)
                        except (TypeError, ValueError):
                            duration = 0.0   # or skip the entry if you prefer
                        fh.write(
                            json.dumps(
                                {"audio_filepath": audio_path, "duration": duration, "text": texts[i]},
                                ensure_ascii=False,
                            )
                            + "\n",
                        )
                        written += 1
            print(f"Wrote {written} samples to {manifest_path}")
        return

    # Fallback to HF dataset if local not present or --force-download used
    print(f"Loading Tadabur dataset split '{split}' from HuggingFace (force_download={force_download})...")
    ds = load_dataset("FaisaI/tadabur", split=split)
    total = len(ds)
    if percent < 100:
        take = max(1, int(total * (percent / 100.0)))
    else:
        take = total
    print(f"Dataset size: {total}. Taking {take} samples ({percent}%).")
    print(f"Writing manifest to: {manifest_path}")
    with open(manifest_path, "w", encoding="utf-8") as fh:
        for i, item in enumerate(ds):
            if i >= take:
                break
            audio_data = item.get("audio")
            text = item.get("text", "")
            if not audio_data:
                continue
            audio_path = audio_data.get("path", "")
            raw_duration = audio_data.get("duration", "")
            try:
                duration = float(raw_duration)
            except (TypeError, ValueError):
                duration = 0.0
            if not audio_path:
                continue
            fh.write(json.dumps({"audio_filepath": audio_path, "duration": duration, "text": text}, ensure_ascii=False) + "\n")
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
    parser.add_argument(
        "--force-download", action="store_true",
        help="Force fetching from HuggingFace even if local shards exist."
    )
    parser.add_argument(
        "--cache-dir", default=None,
        help="Restrict local parquet search to this cache directory (e.g., './.cache')."
    )
    parser.add_argument(
        "--max-samples", type=int, default=None,
        help="Absolute cap on number of samples to write to manifest (overrides --percent)."
    )
    
    args = parser.parse_args()
    fetch_subset(args.out_dir, args.percent, args.split, force_download=args.force_download, max_samples_arg=args.max_samples, cache_dir=args.cache_dir)


if __name__ == "__main__":
    main()
