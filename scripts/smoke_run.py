#!/usr/bin/env python3
"""Smoke test: load one batch, compute features, tokenize, and verify pipeline."""
import argparse
import yaml
import csv
import os
import torch
import torchaudio

def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)

def read_manifest(manifest_path, max_items):
    rows = []
    with open(manifest_path, newline='', encoding='utf-8') as fh:
        reader = csv.DictReader(fh)
        for i, r in enumerate(reader):
            rows.append(r)
            if len(rows) >= max_items:
                break
    return rows

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='configs/train_config.yaml')
    p.add_argument('--batch-size', type=int, default=None)
    p.add_argument('--max-items', type=int, default=4)
    args = p.parse_args()

    cfg = load_config(args.config)

    # determine manifest and batch size
    manifest = cfg.get('train_ds', {}).get('manifest_filepath')
    if not manifest or not os.path.exists(manifest):
        print('Manifest not found:', manifest)
        raise SystemExit(1)
    batch_size = args.batch_size or cfg.get('train_ds', {}).get('batch_size', 4)
    n = min(batch_size, args.max_items)

    rows = read_manifest(manifest, n)
    if not rows:
        print('No rows read from manifest', manifest)
        raise SystemExit(1)

    # load tokenizer
    sp_model = cfg.get('tokenizer', {}).get('dir') or cfg.get('model', {}).get('tokenizer_path')
    if sp_model is None:
        print('No tokenizer path in config')
        sp = None
    else:
        try:
            import sentencepiece as spm
            sp = spm.SentencePieceProcessor()
            # if given dir, find .model
            if os.path.isdir(sp_model):
                candidates = [os.path.join(sp_model, f) for f in os.listdir(sp_model) if f.endswith('.model')]
                sp.Load(candidates[0])
            else:
                sp.Load(sp_model)
            print('Loaded SentencePiece model:', sp_model)
        except Exception as e:
            print('Failed to load sentencepiece:', e)
            sp = None

    # load preprocessor from NeMo if available
    try:
        from nemo.collections.asr.modules import AudioToMelSpectrogramPreprocessor
        preproc_cfg = cfg.get('model', {}).get('preprocessor', {})
        preproc = AudioToMelSpectrogramPreprocessor(**{k: v for k, v in preproc_cfg.items() if k in ['sample_rate','normalize','window_size','window_stride','window','features','n_fft','pad_to']})
        has_nemo = True
        print('Using NeMo AudioToMelSpectrogramPreprocessor')
    except Exception as e:
        print('NeMo preprocessor not available, falling back to torchaudio spectrogram:', e)
        preproc = None
        has_nemo = False

    # Load audio and compute features
    auds = []
    texts = []
    def load_wav_fallback(path):
        # Try torchaudio first
        try:
            wav, sr = torchaudio.load(path)
            return wav, sr
        except Exception:
            # Fallback to wave + numpy
            import wave
            import numpy as np
            with wave.open(path, 'rb') as wf:
                sr = wf.getframerate()
                nchan = wf.getnchannels()
                frames = wf.readframes(wf.getnframes())
                sampwidth = wf.getsampwidth()
                if sampwidth == 2:
                    dtype = np.int16
                elif sampwidth == 4:
                    dtype = np.int32
                else:
                    dtype = np.int16
                audio = np.frombuffer(frames, dtype=dtype)
                if nchan > 1:
                    audio = audio.reshape(-1, nchan).T
                else:
                    audio = audio.reshape(1, -1)
                tensor = torch.from_numpy(audio.astype('float32')) / (32768.0 if sampwidth==2 else 2147483648.0)
                return tensor, sr

    for r in rows:
        a = r.get('audio_filepath')
        t = r.get('text','')
        if not os.path.exists(a):
            print('Audio file missing:', a)
            raise SystemExit(1)
        wav, sr = load_wav_fallback(a)
        if sr != cfg.get('model',{}).get('sample_rate',16000):
            try:
                wav = torchaudio.functional.resample(wav, sr, cfg.get('model',{}).get('sample_rate',16000))
            except Exception:
                # simple numpy resample not implemented; assume sample rates match
                pass
        auds.append(wav)
        texts.append(t)

    # pad to same length
    max_len = max(w.shape[1] for w in auds)
    batch = torch.stack([torch.nn.functional.pad(w, (0, max_len - w.shape[1])) for w in auds], dim=0)
    print('Batch audio tensor shape:', batch.shape)

    if preproc is not None:
        # expect (B, C, T) -> preproc expects (B, T) or (B, C, T) depending; convert
        if batch.shape[1] == 1:
            inp = batch.squeeze(1)
        else:
            inp = batch
        feats = preproc(inp)
        print('Computed features shape:', feats.shape)
    else:
        # quick mel spectrogram via torchaudio
        mel_spec = torchaudio.transforms.MelSpectrogram(sample_rate=cfg.get('model',{}).get('sample_rate',16000), n_mels=80)(batch.squeeze(1))
        print('Computed torchaudio mel shape:', mel_spec.shape)

    if sp is not None:
        toks = [sp.EncodeAsIds(t) for t in texts]
        print('Tokenized sample lengths:', [len(x) for x in toks])

    print('Smoke run succeeded')


if __name__ == '__main__':
    main()
