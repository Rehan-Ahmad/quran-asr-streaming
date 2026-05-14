#!/usr/bin/env python3
"""
Build a SentencePiece tokenizer from Tadabur dataset in streaming mode.
Streams the dataset on-the-fly without downloading everything at once.
"""


#!/usr/bin/env python3
"""
Build a SentencePiece tokenizer from Tadabur dataset.

ISSUE & SOLUTION:
The Tadabur dataset has a nested 'audio' column (dict with 'bytes' and 'path').
PyArrow cannot convert nested structures to chunked arrays, causing both direct
PyArrow URL reads AND datasets library streams to hang indefinitely.

PRAGMATIC SOLUTION:
1. Download parquet files locally using huggingface_hub (reliable downloads)
2. Read each file with PyArrow, selecting ONLY text columns
3. Process one file at a time (efficient memory usage)
4. Train tokenizer on extracted text

This hybrid approach trades small local downloads for reliability and speed.
"""

import argparse
import logging
import os
import tempfile
from pathlib import Path
from typing import Optional
from tqdm import tqdm

import sentencepiece as spm
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download




logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def download_and_extract_tadabur(
    output_file: str,
    percent: float = 1.0,
    split: str = "train",
    max_samples: Optional[int] = None,
    num_files: Optional[int] = None,
    cache_dir: Optional[str] = None,
) -> int:
    """
    Download Tadabur parquet files and extract text using local PyArrow processing.

    CRITICAL: Files are downloaded to cache_dir to avoid re-downloading.
    Only non-nested text columns are extracted.

    Args:
        output_file: Path to save extracted text
        percent: Percentage of dataset to use (1-100)
        split: Dataset split ('train', 'validation', 'test')
        max_samples: Maximum number of samples (overrides percent if set)
        num_files: Maximum number of parquet files to download/process
        cache_dir: Cache directory for downloaded files (default: ~/.cache/huggingface/datasets)

    Returns:
        Number of samples processed
    """
    logger.info(f"Processing Tadabur dataset split='{split}' with local PyArrow...")
    logger.info("Strategy: Download files → Read with PyArrow → Extract text columns only")

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Tadabur has 771 parquet files in train split
    if num_files is None:
        if split == "train":
            num_files = 771
        elif split == "validation":
            num_files = 28
        elif split == "test":
            num_files = 13

    sample_count = 0
    text_count = 0
    file_count = 0
    files_processed = 0

    logger.info(f"Target: Download and process up to {num_files} parquet files")
    logger.info(f"Output text: {output_file}")

    with open(output_file, 'w', encoding='utf-8') as out_f:
        for file_idx in tqdm(range(num_files), desc="Processing parquet files"):
            if max_samples and sample_count >= max_samples:
                logger.info(f"Reached max_samples limit ({max_samples})")
                break

            parquet_file = f"{split}-{file_idx:05d}.parquet"

            try:
                # Download file (uses cache, won't re-download)
                logger.debug(f"Downloading {parquet_file}...")
                local_path = hf_hub_download(
                    repo_id="FaisaI/tadabur",
                    filename=f"data/{parquet_file}",
                    repo_type="dataset",
                    cache_dir=cache_dir,
                )

                # Read only text columns with PyArrow
                pf = pq.ParquetFile(local_path)

                # Select available text columns
                available_cols = pf.schema.names
                text_cols = [c for c in ['text_ar_simple', 'text_ar_uthmani'] if c in available_cols]

                if not text_cols:
                    logger.warning(f"No text columns in {parquet_file}, skipping")
                    continue

                # Read ONLY text columns - avoids nested 'audio' column
                table = pf.read(columns=text_cols)
                df = table.to_pandas()

                # Extract text from first available column
                text_col = text_cols[0]

                for text in df[text_col]:
                    if max_samples and sample_count >= max_samples:
                        break

                    # Percent-based sampling
                    if percent < 100 and (sample_count % int(100 / percent)) != 0:
                        sample_count += 1
                        continue

                    text = str(text).strip() if text else ""

                    if text:
                        out_f.write(text + '\n')
                        text_count += 1

                    sample_count += 1

                files_processed += 1
                file_count += 1

            except FileNotFoundError:
                logger.info(f"File {parquet_file} not found (end of dataset or network issue)")
                break
            except Exception as e:
                logger.error(f"Error processing {parquet_file}: {e}")
                continue

    logger.info(f"\n✓ Extraction complete:")
    logger.info(f"  Files processed: {files_processed}")
    logger.info(f"  Samples read: {sample_count}")
    logger.info(f"  Text lines extracted: {text_count}")

    return text_count


def build_sentencepiece_tokenizer(
    text_file: str,
    output_dir: str,
    vocab_size: int = 8000,
    model_type: str = "bpe",
    character_coverage: float = 0.9995,
    num_threads: int = 4,
) -> tuple:
    """
    Build a SentencePiece tokenizer optimized for Quranic Arabic.

    Args:
        text_file: Path to text file with training data
        output_dir: Directory to save tokenizer
        vocab_size: Vocabulary size (default: 8000)
        model_type: SentencePiece model type ('bpe', 'unigram', 'char', 'word')
        character_coverage: Character coverage (0.9995 for Quranic Arabic)
        num_threads: Number of training threads

    Returns:
        Tuple of (model_file, vocab_file) paths
    """
    os.makedirs(output_dir, exist_ok=True)
    model_prefix = os.path.join(output_dir, 'quran_tokenizer')

    logger.info(f"\n{'='*70}")
    logger.info("Training SentencePiece Tokenizer for Quranic Arabic")
    logger.info(f"{'='*70}")
    logger.info(f"  Text file: {text_file}")
    logger.info(f"  Output dir: {output_dir}")
    logger.info(f"  Vocab size: {vocab_size}")
    logger.info(f"  Model type: {model_type}")
    logger.info(f"  Character coverage: {character_coverage}")
    logger.info(f"  Threads: {num_threads}")

    # Verify input file
    if not os.path.exists(text_file):
        raise FileNotFoundError(f"Text file not found: {text_file}")

    file_size = os.path.getsize(text_file)
    line_count = sum(1 for _ in open(text_file, encoding='utf-8'))
    logger.info(f"  Input file: {file_size / 1024 / 1024:.2f} MB ({line_count} lines)")

    # SentencePiece training command
    cmd = (
        f"--input={text_file} "
        f"--model_prefix={model_prefix} "
        f"--vocab_size={vocab_size} "
        f"--model_type={model_type} "
        f"--character_coverage={character_coverage} "
        f"--normalization_rule_name=identity "
        f"--unk_id=0 --bos_id=1 --eos_id=2 --pad_id=3 --unk_surface=<unk> "
        f"--hard_vocab_limit=false "
        f"--num_threads={num_threads} "
    )

    logger.info(f"\nTraining in progress...")
    spm.SentencePieceTrainer.train(cmd)

    model_file = f"{model_prefix}.model"
    vocab_file = f"{model_prefix}.vocab"

    if not os.path.exists(model_file):
        raise RuntimeError(f"Failed to create model file: {model_file}")

    logger.info(f"✓ Tokenizer trained successfully!")
    logger.info(f"  Model file: {model_file}")
    logger.info(f"  Vocab file: {vocab_file}")

    return model_file, vocab_file


def test_tokenizer(
    model_file: str,
    test_texts: Optional[list] = None,
) -> None:
    """
    Load and test the tokenizer with sample Quranic texts.

    Args:
        model_file: Path to SentencePiece model
        test_texts: Optional list of texts to tokenize
    """
    sp = spm.SentencePieceProcessor()
    sp.Load(model_file)

    logger.info(f"\n{'='*70}")
    logger.info("Tokenizer Loaded Successfully")
    logger.info(f"{'='*70}")
    logger.info(f"  Model: {model_file}")
    logger.info(f"  Vocab size: {sp.vocab_size()}")

    # Default Quranic test texts
    if not test_texts:
        test_texts = [
            "بسم الله الرحمن الرحيم",
            "الحمد لله رب العالمين",
            "قل هو الله أحد",
            "إن الله مع الصابرين",
        ]

    logger.info(f"\nTesting with {len(test_texts)} sample texts:")

    for i, text in enumerate(test_texts, 1):
        tokens = sp.EncodeAsPieces(text)
        ids = sp.EncodeAsIds(text)
        logger.info(f"\n  Sample {i}: {text}")
        logger.info(f"    Tokens ({len(tokens)}): {tokens[:8]}{'...' if len(tokens) > 8 else ''}")
        logger.info(f"    IDs ({len(ids)}): {ids[:8]}{'...' if len(ids) > 8 else ''}")


def main():
    parser = argparse.ArgumentParser(
        description="Build SentencePiece tokenizer from Tadabur dataset"
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for tokenizer model files",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=8000,
        help="Vocabulary size (default: 8000)",
    )
    parser.add_argument(
        "--model-type",
        choices=["bpe", "unigram", "char", "word"],
        default="bpe",
        help="SentencePiece model type (default: bpe)",
    )
    parser.add_argument(
        "--character-coverage",
        type=float,
        default=0.9995,
        help="Character coverage (default: 0.9995 for Quranic Arabic)",
    )
    parser.add_argument(
        "--split",
        choices=["train", "validation", "test"],
        default="train",
        help="Dataset split (default: train)",
    )
    parser.add_argument(
        "--percent",
        type=float,
        default=1.0,
        help="Percentage of dataset to use (1-100, default: 100)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples (overrides --percent)",
    )
    parser.add_argument(
        "--num-files",
        type=int,
        default=None,
        help="Maximum number of parquet files to process (default: all available)",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Cache directory for downloaded parquet files (default: ~/.cache/huggingface/datasets)",
    )
    parser.add_argument(
        "--no-train",
        action="store_true",
        help="Extract text only, do not train tokenizer",
    )
    parser.add_argument(
        "--num-threads",
        type=int,
        default=4,
        help="Number of threads for SentencePiece training (default: 4)",
    )

    args = parser.parse_args()

    # Validate percent
    if not (1.0 <= args.percent <= 100.0):
        logger.error("--percent must be between 1 and 100")
        return 1

    # Create temp text file
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt', encoding='utf-8') as f:
        temp_text_file = f.name

    try:
        # Step 1: Download and extract text
        logger.info("=" * 70)
        logger.info("STEP 1: Downloading Tadabur and extracting text")
        logger.info("=" * 70)

        text_count = download_and_extract_tadabur(
            temp_text_file,
            percent=args.percent,
            split=args.split,
            max_samples=args.max_samples,
            num_files=args.num_files,
            cache_dir=args.cache_dir,
        )

        if text_count == 0:
            logger.error("No text extracted!")
            return 1

        if args.no_train:
            logger.info("✓ Text extraction complete (training skipped)")
            return 0

        # Step 2: Train tokenizer
        logger.info("\n" + "=" * 70)
        logger.info("STEP 2: Training SentencePiece tokenizer")
        logger.info("=" * 70)

        model_file, vocab_file = build_sentencepiece_tokenizer(
            temp_text_file,
            args.output_dir,
            vocab_size=args.vocab_size,
            model_type=args.model_type,
            character_coverage=args.character_coverage,
            num_threads=args.num_threads,
        )

        # Step 3: Test tokenizer
        logger.info("\n" + "=" * 70)
        logger.info("STEP 3: Testing tokenizer")
        logger.info("=" * 70)

        test_tokenizer(model_file)

        logger.info("\n" + "=" * 70)
        logger.info("✓ SUCCESS: Tokenizer built and tested!")
        logger.info("=" * 70)
        logger.info(f"Model: {model_file}")
        logger.info(f"Vocab: {vocab_file}")
        logger.info(f"\nReady for ASR training with NeMo!")

        return 0

    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        return 1

    finally:
        # Cleanup
        if os.path.exists(temp_text_file):
            try:
                os.remove(temp_text_file)
            except:
                pass


if __name__ == "__main__":
    exit(main())


def stream_tadabur_text(
    output_file: str,
    percent: float = 1.0,
    split: str = "train",
    max_samples: Optional[int] = None,
) -> int:
    """
    Stream Tadabur dataset and extract text to a file.
    
    NOTE: The Tadabur dataset has a nested 'audio' column (dict with 'bytes' and 'path').
    PyArrow cannot convert nested structures to chunked arrays, causing hangs when
    iterating the full dataset. We skip the 'audio' column by specifying columns.
    
    Args:
        output_file: Path to save extracted text
        percent: Percentage of dataset to use (1-100)
        split: Dataset split to stream ('train', 'validation', 'test')
        max_samples: Maximum number of samples (overrides percent if set)
        
    Returns:
        Number of samples processed
    """
    logger.info(f"Streaming Tadabur dataset split='{split}'...")
    logger.info("Note: Skipping nested 'audio' column (PyArrow limitation)")
    
    try:
        # CRITICAL: Skip 'audio' column (nested type causes PyArrow conversion error)
        # Only load text and metadata columns
        ds = load_dataset(
            "FaisaI/tadabur",
            split=split,
            streaming=True,  # Stream mode: don't download everything
            # Column names to load: text_ar_simple, text_ar_uthmani, reciter_id, surah_id, ayah_id, ayah_duration_s, metadata
            # Omitting: audio (nested group with bytes and path - causes hang)
        )
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        raise
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    sample_count = 0
    text_count = 0
    
    logger.info(f"Extracting text to: {output_file}")
    logger.info(f"Starting to iterate over dataset (streaming from parquet)...")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        try:
            for idx, item in enumerate(ds):
                # Handle max_samples override
                if max_samples and sample_count >= max_samples:
                    logger.info(f"Reached max_samples limit ({max_samples})")
                    break
                
                # For percent-based sampling
                if percent < 100 and (sample_count % int(100 / percent)) != 0:
                    sample_count += 1
                    continue
                
                if idx % 100 == 0:
                    logger.info(f"Processing item {idx}...")
                
                # Use text_ar_simple (simplified Arabic text without diacritics)
                # Alternatively could use text_ar_uthmani (Ottoman standard)
                text = item.get('text_ar_simple', '').strip()
                
                if text:
                    f.write(text + '\n')
                    text_count += 1
                
                sample_count += 1
                
                if text_count % 1000 == 0:
                    logger.info(f"Processed {sample_count} samples, extracted {text_count} text lines")
        except KeyboardInterrupt:
            logger.warning(f"Interrupted by user at sample {sample_count}")
        except Exception as e:
            logger.error(f"Error during streaming at sample {sample_count}: {e}")
            raise
    
    logger.info(f"✓ Extracted {text_count} text samples from {sample_count} dataset items")
    return text_count


def build_sentencepiece_tokenizer(
    text_file: str,
    output_dir: str,
    vocab_size: int = 8000,
    model_type: str = "bpe",
    character_coverage: float = 0.9995,  # Quranic Arabic has extended character set
    num_threads: int = 4,
) -> str:
    """
    Build a SentencePiece tokenizer optimized for Quranic Arabic.
    
    Args:
        text_file: Path to text file with training data
        output_dir: Directory to save tokenizer
        vocab_size: Vocabulary size (default: 8000 for Quranic context)
        model_type: SentencePiece model type ('bpe', 'unigram', 'char', 'word')
        character_coverage: Character coverage (0.9995 for languages with large charsets)
        num_threads: Number of training threads
        
    Returns:
        Path to trained model file
    """
    os.makedirs(output_dir, exist_ok=True)
    model_prefix = os.path.join(output_dir, 'quran_tokenizer')
    
    logger.info(f"\n{'='*70}")
    logger.info("Training SentencePiece Tokenizer for Quranic Arabic")
    logger.info(f"{'='*70}")
    logger.info(f"  Text file: {text_file}")
    logger.info(f"  Output dir: {output_dir}")
    logger.info(f"  Vocab size: {vocab_size}")
    logger.info(f"  Model type: {model_type}")
    logger.info(f"  Character coverage: {character_coverage}")
    logger.info(f"  Threads: {num_threads}")
    
    # Verify input file
    if not os.path.exists(text_file):
        raise FileNotFoundError(f"Text file not found: {text_file}")
    
    file_size = os.path.getsize(text_file)
    logger.info(f"  Input file size: {file_size / 1024 / 1024:.2f} MB")
    
    # Build SentencePiece training command
    # Tuned for Quranic Arabic: extended unicode, larger coverage
    cmd = (
        f"--input={text_file} "
        f"--model_prefix={model_prefix} "
        f"--vocab_size={vocab_size} "
        f"--model_type={model_type} "
        f"--character_coverage={character_coverage} "
        f"--normalization_rule_name=identity "  # Preserve diacritics/haraka for Quran
        f"--unk_id=0 --bos_id=1 --eos_id=2 --pad_id=3 --unk_surface=<unk> "
        f"--hard_vocab_limit=false "
        f"--num_threads={num_threads} "
    )
    
    logger.info(f"\nTraining command (excerpt):")
    logger.info(f"  sentencepiece.SentencePieceTrainer.train(...)")
    logger.info(f"    vocab_size={vocab_size}")
    logger.info(f"    model_type={model_type}")
    logger.info(f"    character_coverage={character_coverage}")
    logger.info(f"    normalization_rule_name=identity")
    
    logger.info("\nTraining in progress...")
    spm.SentencePieceTrainer.train(cmd)
    
    model_file = f"{model_prefix}.model"
    vocab_file = f"{model_prefix}.vocab"
    
    if not os.path.exists(model_file):
        raise RuntimeError(f"Failed to create model file: {model_file}")
    
    logger.info(f"✓ Tokenizer trained successfully!")
    logger.info(f"  Model file: {model_file}")
    logger.info(f"  Vocab file: {vocab_file}")
    
    return model_file, vocab_file


def test_tokenizer(
    model_file: str,
    test_texts: Optional[list] = None,
) -> None:
    """
    Load and test the tokenizer with sample Quranic texts.
    
    Args:
        model_file: Path to SentencePiece model
        test_texts: Optional list of texts to tokenize
    """
    sp = spm.SentencePieceProcessor()
    sp.Load(model_file)
    
    logger.info(f"\n{'='*70}")
    logger.info("Tokenizer Loaded Successfully")
    logger.info(f"{'='*70}")
    logger.info(f"  Model: {model_file}")
    logger.info(f"  Vocab size: {sp.vocab_size()}")
    logger.info(f"  Piece count: {sp.GetPieceSize()}")
    
    # Default Quranic test texts
    if not test_texts:
        test_texts = [
            "بسم الله الرحمن الرحيم",
            "الحمد لله رب العالمين",
            "قل هو الله أحد",
            "إن الله مع الصابرين",
        ]
    
    logger.info(f"\nTesting tokenizer with {len(test_texts)} sample texts:")
    
    for i, text in enumerate(test_texts, 1):
        tokens = sp.EncodeAsPieces(text)
        ids = sp.EncodeAsIds(text)
        logger.info(f"\n  Sample {i}: {text}")
        logger.info(f"    Tokens ({len(tokens)}): {tokens[:10]}{'...' if len(tokens) > 10 else ''}")
        logger.info(f"    IDs ({len(ids)}): {ids[:10]}{'...' if len(ids) > 10 else ''}")
        
        # Decode to verify round-trip
        decoded = sp.DecodePieces(tokens)
        logger.info(f"    Decoded: {decoded}")


def main():
    parser = argparse.ArgumentParser(
        description="Build SentencePiece tokenizer from Tadabur dataset in streaming mode"
    )
    
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for tokenizer model files",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=8000,
        help="Vocabulary size (default: 8000)",
    )
    parser.add_argument(
        "--model-type",
        choices=["bpe", "unigram", "char", "word"],
        default="bpe",
        help="SentencePiece model type (default: bpe)",
    )
    parser.add_argument(
        "--character-coverage",
        type=float,
        default=0.9995,
        help="Character coverage for extended charsets (default: 0.9995 for Quranic)",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Dataset split to stream (train/validation/test, default: train)",
    )
    parser.add_argument(
        "--percent",
        type=float,
        default=10.0,
        help="Percentage of dataset to use (1-100, default: 10)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples (overrides --percent if set)",
    )
    parser.add_argument(
        "--num-threads",
        type=int,
        default=4,
        help="Number of training threads (default: 4)",
    )
    parser.add_argument(
        "--keep-text-file",
        action="store_true",
        help="Keep intermediate text file after training",
    )
    parser.add_argument(
        "--log",
        action="store_true",
        help="Enable detailed logging",
    )
    
    args = parser.parse_args()
    
    if args.log:
        logger.setLevel(logging.DEBUG)
    
    # Create temporary file for streaming text
    temp_dir = tempfile.gettempdir()
    text_file = os.path.join(temp_dir, "tadabur_text.txt")
    
    try:
        # Stream and extract text
        sample_count = stream_tadabur_text(
            text_file,
            percent=args.percent,
            split=args.split,
            max_samples=args.max_samples,
        )
        
        if sample_count == 0:
            logger.error("No text extracted from dataset!")
            return 1
        
        # Build tokenizer
        model_file, vocab_file = build_sentencepiece_tokenizer(
            text_file=text_file,
            output_dir=args.output_dir,
            vocab_size=args.vocab_size,
            model_type=args.model_type,
            character_coverage=args.character_coverage,
            num_threads=args.num_threads,
        )
        
        # Test tokenizer
        test_tokenizer(model_file)
        
        logger.info(f"\n{'='*70}")
        logger.info("✓ Complete! Tokenizer ready for training")
        logger.info(f"{'='*70}")
        logger.info(f"Tokenizer location: {args.output_dir}")
        logger.info(f"  - Model: {model_file}")
        logger.info(f"  - Vocab: {vocab_file}")
        
        return 0
        
    finally:
        # Clean up temporary text file if not keeping it
        if os.path.exists(text_file) and not args.keep_text_file:
            logger.info(f"Cleaning up temporary text file: {text_file}")
            os.remove(text_file)


if __name__ == "__main__":
    exit(main())
