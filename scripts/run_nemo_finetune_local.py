#!/usr/bin/env python3
"""Create a NeMo-compatible config from our train_config.yaml and launch finetune.

This script writes a caller-provided NeMo config path and runs the upstream
`speech_to_text_finetune.py` example via `torchrun --nproc_per_node=1`.
"""
import argparse
from pathlib import Path
import yaml
import os
import subprocess
import shlex


def build_nemo_config(train_cfg_path: str, out_path: str, pretrained: str):
    with open(train_cfg_path) as f:
        cfg = yaml.safe_load(f)

    nemo_cfg = {}
    if 'name' in cfg:
        nemo_cfg['name'] = cfg['name']
    # copy trainer and exp_manager if present
    if 'trainer' in cfg:
        nemo_cfg['trainer'] = cfg['trainer']
    if 'exp_manager' in cfg:
        nemo_cfg['exp_manager'] = cfg['exp_manager']

    # Build model block: start from existing model and embed dataset/optim/tokenizer
    model_block = cfg.get('model', {}).copy()
    # move train/validation/test/optim/tokenizer under model
    for k in ('train_ds', 'validation_ds', 'test_ds', 'optim', 'tokenizer'):
        if k in cfg:
            model_block[k] = cfg[k]

    nemo_cfg['model'] = model_block

    # set pretrained init
    nemo_cfg['init_from_pretrained_model'] = pretrained

    # If train/validation/test manifest filenames are generic 'manifest.json',
    # rename them to descriptive names and copy existing files if present.
    for ds_name, new_suffix in (('train_ds', 'train_manifest.json'), ('validation_ds', 'validation_manifest.json'), ('test_ds', 'test_manifest.json')):
        ds = model_block.get(ds_name)
        if isinstance(ds, dict):
            mf = ds.get('manifest_filepath')
            if mf and os.path.basename(mf) == 'manifest.json':
                new_mf = os.path.join(os.path.dirname(mf), new_suffix)
                # copy existing manifest if present
                try:
                    if os.path.exists(mf) and not os.path.exists(new_mf):
                        from shutil import copyfile
                        copyfile(mf, new_mf)
                except Exception:
                    pass
                ds['manifest_filepath'] = new_mf

    # sanitize LR scheduler: avoid conflicting warmup params
    try:
        sched = nemo_cfg.get('model', {}).get('optim', {}).get('sched')
        if isinstance(sched, dict):
            # If both warmup_steps and warmup_ratio present, prefer ratio and drop explicit steps
            if 'warmup_steps' in sched and 'warmup_ratio' in sched:
                sched.pop('warmup_steps', None)

            # If max_steps is not set, try to compute it from the train manifest and trainer.max_epochs
            if 'max_steps' not in sched or sched.get('max_steps') in (None, -1):
                train_ds = nemo_cfg.get('model', {}).get('train_ds', {}) or {}
                mf = train_ds.get('manifest_filepath')
                batch_size = int(train_ds.get('batch_size', 1))
                max_epochs = int(nemo_cfg.get('trainer', {}).get('max_epochs', 1))
                if mf and os.path.exists(mf) and batch_size > 0 and max_epochs > 0:
                    try:
                        with open(mf, 'r', encoding='utf-8') as fh:
                            num_samples = sum(1 for _ in fh)
                        import math
                        steps_per_epoch = max(1, math.ceil(num_samples / batch_size))
                        sched['max_steps'] = steps_per_epoch * max_epochs
                    except Exception:
                        pass
    except Exception:
        pass

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as fh:
        yaml.safe_dump(nemo_cfg, fh, sort_keys=False)

    return out_path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--train-config', default='configs/train_config.yaml')
    p.add_argument('--nemo-config', default='configs/nemo_finetune.yaml')
    p.add_argument('--pretrained', default='nvidia/nemotron-speech-streaming-en-0.6b')
    p.add_argument('--torchrun', default='.venv/bin/torchrun')
    p.add_argument('--script', default='scripts/nemo_examples/run_finetune_wrapper.py')
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()

    out = build_nemo_config(args.train_config, args.nemo_config, args.pretrained)
    print('Wrote NeMo config:', out)

    # Hydra expects --config-path to point to the directory that contains
    # the config group (i.e. the parent of 'asr_finetune') and --config-name
    # to be the filename without extension used by the example.
    out_p = Path(out).resolve()
    config_path = str(out_p.parents[1])  # conf
    config_name = out_p.stem  # speech_to_text_finetune
    cmd = f"{args.torchrun} --nproc_per_node=1 {shlex.quote(args.script)} --config-path={shlex.quote(config_path)} --config-name={shlex.quote(config_name)}"
    print('Running:', cmd)
    if args.dry_run:
        return

    # launch process (stream output)
    proc = subprocess.Popen(cmd, shell=True)
    proc.wait()
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


if __name__ == '__main__':
    main()
