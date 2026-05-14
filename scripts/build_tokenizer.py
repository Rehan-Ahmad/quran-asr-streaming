#!/usr/bin/env python3
"""
Build a SentencePiece tokenizer from Tadabur dataset in streaming mode.
Streams the dataset on-the-fly without downloading everything at once.
"""

import argparse
import logging
import os
import tempfile
from pathlib import Path
from typing import Optional

import sentencepiece as spm
from datasets import load_dataset


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def stream_tadabur_text(
    output_file: str,
    percent: float = 1.0,
    split: str = "train",
    max_samples: Optional[int] = None,
) -> int:
    """
    Stream Tadabur dataset and extract text to a file.
    
    Args:
        output_file: Path to save extracted text
        percent: Percentage of dataset to use (1-100)
        split: Dataset split to stream ('train', 'validation', 'test')
        max_samples: Maximum number of samples (overrides percent if set)
        
    Returns:
        Number of samples processed
    """
    logger.info(f"Streaming Tadabur dataset split='{split}'...")
    
    try:
        ds = load_dataset(
            "FaisaI/tadabur",
            split=split,
            streaming=True,  # Stream mode: don't download everything
        )
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        raise
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    sample_count = 0
    text_count = 0
    
    logger.info(f"Extracting text to: {output_file}")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in ds:
            # Handle max_samples override
            if max_samples and sample_count >= max_samples:
                break
            
            # For percent-based sampling
            if percent < 100 and (sample_count % int(100 / percent)) != 0:
                sample_count += 1
                continue
            
            text = item.get('text', '').strip()
            
            if text:
                f.write(text + '\n')
                text_count += 1
            
            sample_count += 1
            
            if text_count % 1000 == 0:
                logger.info(f"Processed {sample_count} samples, extracted {text_count} text lines")
    
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
