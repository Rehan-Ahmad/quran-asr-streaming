# Quran ASR Streaming with NeMo Toolkit

Fine-tune NVIDIA Nemotron-like streaming ASR models on Quranic Arabic using the NeMo toolkit and Tadabur dataset.

## Project Overview

This project provides a complete pipeline for:
- **Streaming tokenizer**: Build SentencePiece tokenizers directly from Tadabur dataset without downloading everything
- **Patched NeMo scripts**: Exact NeMo tokenizer scripts with minimal dependencies (no heavy Lightning stack)
- **Data streaming**: Efficient on-the-fly text extraction from HuggingFace Tadabur dataset
- **Quranic optimization**: Tokenizer configured for Arabic diacritics, extended character sets, and Quranic context

## Quick Start

### 1. Environment Setup

The project uses `uv` for dependency management with Python 3.12:

```bash
# Activate environment
source .venv/bin/activate

# Or use uv directly
uv sync
```

### 2. Build Tokenizer from Tadabur Dataset (Streaming Mode)

**Quick test with 500 samples:**
```bash
python scripts/build_tokenizer.py \
  --output-dir tokenizers/quran \
  --vocab-size 5000 \
  --model-type bpe \
  --max-samples 500
```

**Reproducible presets for the current subset and the future RunPod full run:**
```bash
# Current local subset workflow used in this repo
scripts/run_tokenizer_repro.sh \
  --profile local-subset \
  --output-dir tokenizers/quran_uthmani_10 \
  --local-data-dir data/tadabur/data \
  --text-column text_ar_uthmani \
  --vocab-size 8000

# RunPod/full-dataset target when you are ready to scale up
scripts/run_tokenizer_repro.sh \
  --profile runpod-full \
  --output-dir tokenizers/quran_runpod_full \
  --text-column text_ar_uthmani \
  --cache-dir .cache/huggingface \
  --vocab-size 8000
```

Each run writes `run_manifest.json` and `reproduce_tokenizer.sh` inside the output directory so the exact invocation is preserved.

**Use 10% of training data:**
```bash
python scripts/build_tokenizer.py \
  --output-dir tokenizers/quran \
  --vocab-size 8000 \
  --model-type bpe \
  --percent 10 \
  --num-threads 8
```

If you want the same behavior with a saved manifest, use `scripts/run_tokenizer_repro.sh` and choose the `local-subset` or `runpod-full` profile.

**Use validation split:**
```bash
python scripts/build_tokenizer.py \
  --output-dir tokenizers/quran_val \
  --split validation \
  --vocab-size 8000 \
  --max-samples 1000
```

### 3. Use Trained Tokenizer

```python
import sentencepiece as spm

sp = spm.SentencePieceProcessor()
sp.Load('tokenizers/quran/quran_tokenizer.model')

text = "بسم الله الرحمن الرحيم"
tokens = sp.EncodeAsPieces(text)
ids = sp.EncodeAsIds(text)
```

### 4. Configure Training

Edit `configs/train_config.yaml` with your dataset paths, batch size, learning rate, etc.

### 5. Train/Finetune

Use NeMo training script or your custom script in `src/`. See references for details.

For the streaming NeMo finetune workflow in this repo, use the launcher below instead of editing the tracked YAML in place:

```bash
bash scripts/train_streaming.sh --smoke-only
bash scripts/train_streaming.sh
```

The launcher creates a temporary config copy for the smoke test, applies the smaller batch/epoch overrides there, and leaves the committed YAML templates untouched.

### RunPod Reproducible Startup

To recreate the exact environment from the lockfile and optionally launch the finetune job on RunPod:

```bash
bash scripts/runpod_start.sh

# or bootstrap and start training in one step
bash scripts/runpod_start.sh --train
```

The script runs `uv sync --frozen --no-install-project` from the repo root, verifies the critical NeMo imports, and then starts the local finetune launcher when `--train` is passed.

See Project Structure below.

## Tokenizer Parameters

**`--output-dir`** (required)
- Directory where tokenizer model files will be saved
- Creates: `quran_tokenizer.model`, `quran_tokenizer.vocab`

**`--vocab-size`** (default: 8000)
- Number of subword units in vocabulary
- Recommended: 5000-16000 for Quranic Arabic

**`--model-type`** (default: bpe)
- `bpe`: Byte-pair encoding (recommended)
- `unigram`: Unigram language model
- `char`: Character-level
- `word`: Word-level

**`--character-coverage`** (default: 0.9995)
- Coverage for Unicode characters
- 0.9995: Extended coverage for Arabic diacritics

**`--split`** (default: train)
- Dataset split: `train`, `validation`, `test`

**`--percent`** (default: 10)
- Percentage of split to use (1-100)

**`--max-samples`** (optional)
- Absolute maximum samples (overrides `--percent`)

**`--num-threads`** (default: 4)
- Number of training threads

**`--keep-text-file`**
- Keep temporary text extraction file

**`--log`**
- Enable verbose logging

## Key Features

### Streaming Mode
- Stream from HuggingFace on-the-fly
- No need to download entire 1TB dataset
- Memory-efficient processing

### Quranic Arabic Optimized
- High character coverage (0.9995) for extended Arabic
- Identity normalization preserves diacritics/haraka
- Special tokens: `<unk>`, `<s>`, `</s>`, `<pad>`
- Tested with authentic Quranic phrases

### NeMo Integration
- Exact NVIDIA NeMo tokenizer scripts
- Patched to avoid heavy Lightning stack
- All NeMo parameters supported

---

## Project Structure

```
quran-asr-streaming/
├── .venv/                          # Virtual environment (uv)
├── scripts/
│   ├── build_tokenizer.py          # ⭐ NEW: Streaming tokenizer builder
│   ├── process_asr_text_tokenizer.py  # Exact NeMo script (patched)
│   ├── tokenizer_wrapper.py        # Tokenizer wrapper
│   └── fetch_subset.py             # Download Tadabur subset
├── tokenizers/                     # Output: trained tokenizers
│   └── quran/                      # Example output directory
├── configs/
│   └── train_config.yaml           # Training configuration
├── src/ (removed)                  # Previously held reusable package modules; now empty and removed
└── README.md                       # This file
```

## Dataset: Tadabur

- **Source**: FaisaI/tadabur on HuggingFace
- **Language**: Classical Quranic Arabic
- **Size**: ~1TB (full) / <100MB (10% subset)
- **Splits**: train, validation, test
- **Features**: Quranic recitation audio + transcription

Streaming mode allows efficient processing of this large dataset.

## Dependencies

Core packages (managed by uv):
- `torch>=2.12.0` (CUDA 13.0)
- `torchaudio>=2.11.0`
- `nemo-toolkit>=2.7.3`
- `sentencepiece>=0.2.1`
- `datasets>=4.8.5` (HuggingFace)
- `transformers>=5.8.1`

Full list in `pyproject.toml`.

## Performance

- Training time: 500 samples → 2-5 minutes
- Memory: ~2-4 GB (streaming mode)
- Output: 8k vocab tokenizer → 100-150 MB

---

## Troubleshooting

**HuggingFace rate limits:**
```bash
export HF_TOKEN="hf_..."
```

**Slow dataset loading:**
- Use smaller `--max-samples` for testing
- Run multiple times to cache files

**Memory issues:**
- Reduce `--vocab-size`
- Use `--max-samples` instead of `--percent`
- Reduce `--num-threads`

---

## References

- NeMo Toolkit: https://github.com/NVIDIA/NeMo
- ASR with Subword Tokenization: https://github.com/NVIDIA-NeMo/NeMo/blob/main/tutorials/asr/ASR_with_Subword_Tokenization.ipynb
- Nemotron-0.6b: https://huggingface.co/nvidia/nemotron-speech-streaming-en-0.6b
- SentencePiece: https://github.com/google/sentencepiece
- Tadabur Dataset: https://huggingface.co/datasets/FaisaI/tadabur

---

## Recent Updates

**NEW: Streaming Tokenizer** (`build_tokenizer.py`)
- Stream Tadabur dataset on-the-fly without downloading
- Build SentencePiece tokenizers optimized for Quranic Arabic
- Configurable vocabulary, model type, character coverage
- Extensive logging for monitoring training

**FIXED: NeMo Tokenizer** (`process_asr_text_tokenizer.py`)
- Patched to avoid heavy Lightning import stack
- All NeMo functionality preserved with minimal dependencies
- Drop-in replacement for original NeMo script

---

## License

- **NeMo**: Apache License 2.0
- **SentencePiece**: Apache License 2.0
- **Tadabur Dataset**: Check repository for license


## Reproducible Workflow

This repository includes a deterministic workflow to: create a `uv` virtualenv, download a reproducible subset of the Tadabur dataset (1-100%), prepare a SentencePiece tokenizer, and provide a template to run Nemotron streaming training.

Files:
- `scripts/setup_env.sh`: Create `.venv` with `uv` and install project dependencies.
- `scripts/download_dataset.py`: Deterministic downloader that selects the first N shards from the requested split and writes `download_manifest.json`.
- `scripts/run_full_workflow.sh`: Orchestrates env creation, dataset download (`--percent`), tokenizer preparation and prints a Nemotron training template (dry-run support).
- `configs/nemotron_streaming.yaml`: Nemotron-card-aligned streaming template — edit before running training.
- `scripts/train_nemotron.py`: Lightweight trainer wrapper that prints the loaded config and a short reminder of how to launch Nemotron training.

Quick example (1% subset, 8k vocab, dry-run):

```bash
bash scripts/run_full_workflow.sh --output-dir runs/my_run --percent 1 --vocab-size 8000 --dry-run
```

To actually run training, edit `configs/nemotron_streaming.yaml` and then use your Nemotron/NeMo launcher. The workflow preserves reproducibility by writing `download_manifest.json` and the tokenizer reproducer inside the `--output-dir`.

Run layout:
- `runs/<name>/data/`: downloaded shards and `download_manifest.json`
- `runs/<name>/tokenizer/`: tokenizer output, `run_manifest.json`, and `reproduce_tokenizer.sh`
- `runs/<name>/training/`: place training logs, checkpoints, and export artifacts here

Note: existing reproducer artifacts (for previous runs) were moved to [scripts/reproducers](scripts/reproducers) for a cleaner repository layout. The default write location for new reproducer artifacts remains the run output directory (for example `runs/<name>/tokenizer/` or `tokenizers/<run-name>/`) so the exact invocation is still preserved with each run unless you explicitly change the target directory.
