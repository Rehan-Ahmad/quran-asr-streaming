# Quran ASR — Nemotron-style Streaming Speech Recognition

Fine-tuning NVIDIA Nemotron-like streaming ASR models on the Quran Tadabur dataset (~1TB) using NeMo toolkit and `uv` package manager.

## Quick Start

### 1. Environment Setup (Already Done with `uv`)

The environment is pre-configured with Python 3.12 and all dependencies via `uv`:

```bash
source .venv/bin/activate
```

Verify installation:
```bash
python -c "import nemo; print(nemo.__version__)"
```

### 2. Download a Small Subset (Optional, for Testing)

To test locally without downloading the full ~1TB dataset, fetch a small subset:

```bash
python scripts/fetch_subset.py \
  --out-dir data/subset \
  --percent 1.0 \
  --split train
```

This creates `data/subset/manifest.csv` with 1% of the data.

### 3. Build Tokenizer

Create a subword tokenizer (using NeMo's built-in script):

```bash
python scripts/tokenizer_wrapper.py \
  --input-manifest data/subset/manifest.csv \
  --out-dir tokenizers \
  --vocab-size 8000 \
  --model-type bpe
```

Output: Tokenizer artifacts in `tokenizers/`.

### 4. Configure Training

Edit `configs/train_config.yaml` with your dataset paths, batch size, learning rate, etc. Example provided.

### 5. Train/Finetune

Use the NeMo training script or your custom script in `src/` to train. For details, see NeMo documentation:
- [ASR with Subword Tokenization](https://github.com/NVIDIA-NeMo/NeMo/blob/main/tutorials/asr/ASR_with_Subword_Tokenization.ipynb)

---

## Project Structure

```
quran-asr-streaming/
├── .venv/                     # Virtual environment (created by uv)
├── pyproject.toml             # Project config & dependencies (uv)
├── uv.lock                    # Locked dependencies (reproducible)
├── README.md                  # This file
├── .gitignore                 # Git ignore patterns
│
├── configs/
│   └── train_config.yaml      # Training configuration template
│
├── scripts/
│   ├── fetch_subset.py        # Download a subset of Tadabur dataset
│   └── tokenizer_wrapper.py   # Wrapper for NeMo tokenizer script
│
├── src/
│   ├── __init__.py
│   ├── data/                  # Data utilities
│   │   └── __init__.py
│   └── tokenizer/             # Tokenizer utilities
│       └── __init__.py
│
├── data/                      # Local data directory (ignored by git)
│   └── subset/                # Downloaded subset manifest & audio
│
├── tokenizers/                # Built tokenizer outputs (ignored by git)
├── logs/                      # Training logs (ignored by git)
└── checkpoints/               # Model checkpoints (ignored by git)
```

---

## Dataset Notes

- **Tadabur Dataset**: https://huggingface.co/datasets/FaisaI/tadabur
- **Size**: ~1TB (full dataset)
- **Recommendation**: Start with `--percent 1` to `--percent 10` for testing on limited systems.

---

## References

- NeMo Toolkit: https://github.com/NVIDIA/NeMo
- ASR Tutorial: https://github.com/NVIDIA-NeMo/NeMo/blob/main/tutorials/asr/ASR_with_Subword_Tokenization.ipynb
- Nemotron-0.6b Model: https://huggingface.co/nvidia/nemotron-speech-streaming-en-0.6b
- Tadabur Dataset: https://huggingface.co/datasets/FaisaI/tadabur

---

## Troubleshooting

**Missing packages?**
```bash
uv sync
```

**GPU issues?**
- Ensure CUDA/cuDNN are available: `nvidia-smi`
- PyTorch is pre-configured for CUDA 12.x via `uv`.

**Out of memory?**
- Reduce `--batch-size` in `configs/train_config.yaml`
- Use `--percent` in `fetch_subset.py` to work with smaller data subsets.
