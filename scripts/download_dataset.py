#!/usr/bin/env python3
"""
Deterministic downloader for Tadabur dataset shards (percent-based).

Produces a JSON manifest listing exact shard filenames downloaded so runs are reproducible.
"""
import argparse
import json
import math
import os
import time

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

TOTAL_SHARDS = {
    "train": 771,
    "validation": 28,
    "test": 13,
}
REPO_ID = "FaisaI/tadabur"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--percent", type=float, default=1.0, help="Percent of train split to download (1-100)")
    p.add_argument("--output-dir", required=True, help="Directory to place downloaded shards and manifest")
    p.add_argument("--cache-dir", default=".cache/huggingface", help="HF cache dir")
    p.add_argument("--split", default="train", choices=["train","validation","test"])
    p.add_argument("--local-data-dir", default=None, help="Prefer local parquet files in this dir (optional)")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    total_shards = TOTAL_SHARDS[args.split]
    take = max(1, int(math.ceil(total_shards * (max(0.0, min(100.0, args.percent)) / 100.0))))
    # Deterministic selection: first N shards of requested split
    filenames = [f"data/{args.split}-{i:05d}.parquet" for i in range(take)]

    downloaded = []
    for fn in filenames:
        try:
            if args.local_data_dir:
                local_path = os.path.join(args.local_data_dir, os.path.basename(fn))
                if os.path.exists(local_path):
                    print(f"Using local file for {fn}: {local_path}")
                    target = os.path.join(args.output_dir, os.path.basename(fn))
                    if not os.path.exists(target):
                        os.symlink(os.path.abspath(local_path), target)
                    downloaded.append({"shard": os.path.basename(fn), "path": target, "rows": None, "source": "local"})
                    continue
            print(f"Downloading {fn} ...")
            local = hf_hub_download(repo_id=REPO_ID, filename=fn, repo_type="dataset", cache_dir=args.cache_dir)
            # copy (or symlink) into output dir to make manifest self-contained
            target = os.path.join(args.output_dir, os.path.basename(local))
            if not os.path.exists(target):
                try:
                    os.symlink(os.path.abspath(local), target)
                except Exception:
                    import shutil
                    shutil.copy(local, target)
            # quick sanity: read metadata
            try:
                pf = pq.ParquetFile(local)
                n = pf.metadata.num_rows
            except Exception:
                n = None
            downloaded.append({"shard": os.path.basename(local), "path": target, "rows": n, "source": "huggingface"})
        except Exception as e:
            print(f"Failed to download {fn}: {e}")

    manifest = {
        "repo_id": REPO_ID,
        "split": args.split,
        "percent": args.percent,
        "total_shards_for_split": total_shards,
        "requested_shards": take,
        "downloaded_files": len(downloaded),
        "downloaded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "files": downloaded,
    }

    manifest_path = os.path.join(args.output_dir, "download_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print("Manifest written to", manifest_path)


if __name__ == "__main__":
    main()
