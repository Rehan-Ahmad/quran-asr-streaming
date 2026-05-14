# Quran ASR — Nemotron-style fine-tuning

This project scaffolds training/finetuning a NeMo/NVIDIA Nemotron-like streaming ASR model on the Quran Tadabur dataset (https://huggingface.co/datasets/FaisaI/tadabur).

Quick steps
- Create and activate a Python venv: `bash scripts/setup_venv.sh`
- (Optional) download a small subset (1-10%) for testing: `python scripts/fetch_subset.py --percent 1 --out-dir data/subset`
- Build tokenizer: `python scripts/tokenizer_wrapper.py --input-manifest data/subset/manifest.csv --out-dir tokenizers --vocab-size 8000`
- Prepare training config in `configs/train_config.yaml` and run NeMo training / finetuning.

Notes
- The Tadabur dataset is large (~1TB). Use `--percent` in `fetch_subset.py` to test with a small subset.
- The tokenizer script wraps NeMo's `process_asr_text_tokenizer.py`; this scaffold will clone NeMo if not present.
- `requirements.txt` lists minimal packages; for GPU setups follow PyTorch/NeMo official recommended install.

References
- NeMo ASR tutorial: https://github.com/NVIDIA-NeMo/NeMo/blob/main/tutorials/asr/ASR_with_Subword_Tokenization.ipynb
- NeMo tokenizer script: https://github.com/NVIDIA-NeMo/NeMo/blob/main/scripts/tokenizers/process_asr_text_tokenizer.py
- Tadabur dataset: https://huggingface.co/datasets/FaisaI/tadabur
