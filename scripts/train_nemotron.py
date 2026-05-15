#!/usr/bin/env python3
"""
Template trainer wrapper for Nemotron-like streaming training.

This script is intentionally minimal: it loads the config and
prints the training command that you should run with your NeMo/Nemotron
environment. Running full Nemotron training requires heavy GPU,
PyTorch and NeMo dependencies which are environment-specific.
"""
import argparse
import yaml
import sys

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    print("Loaded config:")
    print(yaml.dump(cfg, sort_keys=False))

    print("\nThis is a template. To run Nemotron training, activate your NeMo environment and run the vendor training launcher with the above config.")
    print("Example:\n  source .venv/bin/activate\n  # then follow your Nemotron training launcher, e.g. torchrun ...")

if __name__ == '__main__':
    main()
