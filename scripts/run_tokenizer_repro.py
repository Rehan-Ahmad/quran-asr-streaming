#!/usr/bin/env python3
"""
Reproducible tokenizer runner for local subset experiments and future RunPod runs.

This wrapper keeps the actual SentencePiece extraction/training logic inside
`scripts/build_tokenizer.py` and adds two things that are useful for repeatable
experiments:

- opinionated presets for the current local subset workflow vs. the future
  RunPod full-dataset workflow
- a manifest plus a copyable shell script so the exact invocation can be
  replayed later
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass
class RunManifest:
    profile: str
    timestamp_utc: str
    git_commit: str | None
    repo_root: str
    build_script: str
    build_command: list[str]
    output_dir: str
    vocab_size: int
    model_type: str
    character_coverage: float
    split: str
    percent: float
    max_samples: int | None
    num_files: int | None
    cache_dir: str
    local_data_dir: str | None
    text_column: str
    num_threads: int


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def git_commit(root: Path) -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    except Exception:
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run tokenizer training with reproducible presets for local subset or RunPod full-dataset runs."
    )
    parser.add_argument(
        "--profile",
        choices=["local-subset", "runpod-full"],
        default="local-subset",
        help="Preset to use. local-subset matches the current lightweight workflow; runpod-full targets the full Tadabur train split.",
    )
    parser.add_argument("--output-dir", required=True, help="Tokenizer output directory")
    parser.add_argument("--vocab-size", type=int, default=8000, help="SentencePiece vocab size")
    parser.add_argument(
        "--model-type",
        choices=["bpe", "unigram", "char", "word"],
        default="bpe",
        help="SentencePiece model type",
    )
    parser.add_argument(
        "--character-coverage",
        type=float,
        default=0.9995,
        help="SentencePiece character coverage",
    )
    parser.add_argument("--split", choices=["train", "validation", "test"], default="train")
    parser.add_argument("--percent", type=float, default=100.0)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--num-files", type=int, default=None)
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=".cache/huggingface",
        help="Repo-local Hugging Face cache directory",
    )
    parser.add_argument(
        "--local-data-dir",
        type=str,
        default=None,
        help="Optional local parquet directory to prefer before downloading",
    )
    parser.add_argument(
        "--text-column",
        choices=["text_ar_simple", "text_ar_uthmani"],
        default="text_ar_uthmani",
        help="Which Tadabur text field to tokenize",
    )
    parser.add_argument("--num-threads", type=int, default=8)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved command and manifest without running training",
    )
    return parser.parse_args()


def apply_profile_defaults(args: argparse.Namespace) -> None:
    if args.profile == "local-subset":
        if args.num_files is None:
            args.num_files = 10
        args.percent = 100.0 if args.percent is None else args.percent
    elif args.profile == "runpod-full":
        if args.num_files is None:
            args.num_files = 771
        args.percent = 100.0


def build_command(args: argparse.Namespace) -> list[str]:
    script = repo_root() / "scripts" / "build_tokenizer.py"
    command = [
        sys.executable,
        str(script),
        "--output-dir",
        args.output_dir,
        "--vocab-size",
        str(args.vocab_size),
        "--model-type",
        args.model_type,
        "--character-coverage",
        str(args.character_coverage),
        "--split",
        args.split,
        "--percent",
        str(args.percent),
        "--num-threads",
        str(args.num_threads),
        "--cache-dir",
        args.cache_dir,
        "--text-column",
        args.text_column,
    ]
    if args.max_samples is not None:
        command.extend(["--max-samples", str(args.max_samples)])
    if args.num_files is not None:
        command.extend(["--num-files", str(args.num_files)])
    if args.local_data_dir:
        command.extend(["--local-data-dir", args.local_data_dir])
    return command


def write_reproducer(output_dir: Path, command: list[str]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    script_path = output_dir / "reproduce_tokenizer.sh"
    quoted = " \\\n+  ".join(shlex.quote(part) for part in command)
    script_path.write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n\n" + quoted + "\n",
        encoding="utf-8",
    )
    script_path.chmod(0o755)
    return script_path


def main() -> int:
    args = parse_args()
    apply_profile_defaults(args)

    root = repo_root()
    command = build_command(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = RunManifest(
        profile=args.profile,
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        git_commit=git_commit(root),
        repo_root=str(root),
        build_script=str(root / "scripts" / "build_tokenizer.py"),
        build_command=command,
        output_dir=str(output_dir.resolve()),
        vocab_size=args.vocab_size,
        model_type=args.model_type,
        character_coverage=args.character_coverage,
        split=args.split,
        percent=args.percent,
        max_samples=args.max_samples,
        num_files=args.num_files,
        cache_dir=args.cache_dir,
        local_data_dir=args.local_data_dir,
        text_column=args.text_column,
        num_threads=args.num_threads,
    )

    manifest_path = output_dir / "run_manifest.json"
    manifest_path.write_text(json.dumps(asdict(manifest), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    reproducer_path = write_reproducer(output_dir, command)

    print(f"Profile: {args.profile}")
    print(f"Manifest: {manifest_path}")
    print(f"Reproducer: {reproducer_path}")
    print("Command:")
    print(" ".join(command))

    if args.dry_run:
        return 0

    result = subprocess.run(command, cwd=root)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())