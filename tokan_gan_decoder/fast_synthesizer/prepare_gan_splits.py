#!/usr/bin/env python3
"""
prepare_gan_splits.py: Prepare and verify train/val/test data splits for GAN decoder.

Split Strategy:
- Test speakers (EBVS, SKA): completely held out for test_speaker evaluation
- Training speakers (22 L2 speakers):
  - 50 audio files → test_sentence split (unseen sentences)
  - 50 audio files → validation split
  - Remaining → training split

Usage:
    python prepare_gan_splits.py --data_dir /path/to/gan_targets
"""
import json
import argparse
import logging
from pathlib import Path
from collections import defaultdict

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Test speakers (completely held out for evaluation)
TEST_SPEAKERS = ["EBVS", "SKA"]

# All L2 speakers by accent
L2_SPEAKERS = {
    "arabic": ["ABA", "SKA", "YBAA", "ZHAA"],
    "chinese": ["BWC", "LXC", "NCC", "TXHC"],
    "hindi": ["ASI", "RRBI", "SVBI", "TNI"],
    "korean": ["HJK", "HKK", "YDCK", "YKWK"],
    "spanish": ["EBVS", "ERMS", "MBMPS", "NJS"],
    "vietnamese": ["HQTV", "PNV", "THV", "TLV"],
}

# Per-speaker split configuration
DEFAULT_SENTENCES_PER_SPEAKER_TEST = 50
DEFAULT_SENTENCES_PER_SPEAKER_VAL = 50


def analyze_manifest(manifest_path: Path) -> dict:
    """Load and analyze the manifest."""
    with open(manifest_path) as f:
        samples = json.load(f)
    
    by_speaker = defaultdict(list)
    by_accent = defaultdict(list)
    
    for sample in samples:
        speaker = sample["speaker_id"]
        accent = sample.get("accent", "unknown")
        by_speaker[speaker].append(sample)
        by_accent[accent].append(sample)
    
    return {
        "total_samples": len(samples),
        "samples": samples,
        "by_speaker": dict(by_speaker),
        "by_accent": dict(by_accent),
    }


def create_splits(
    data: dict,
    test_speakers: list,
    sentences_per_speaker_test: int = DEFAULT_SENTENCES_PER_SPEAKER_TEST,
    sentences_per_speaker_val: int = DEFAULT_SENTENCES_PER_SPEAKER_VAL,
) -> dict:
    """
    Create train/val/test splits.

    Groups samples by audio_path to ensure all chunks from the same
    audio file stay together in the same split (prevents data leakage).
    """
    np.random.seed(42)
    
    # Group all samples by audio_path
    by_audio = defaultdict(list)
    for sample in data["samples"]:
        by_audio[sample["audio_path"]].append(sample)
    
    # Group audio files by speaker
    audios_by_speaker = defaultdict(list)
    for audio_path, chunks in by_audio.items():
        speaker = chunks[0]["speaker_id"]
        audios_by_speaker[speaker].append((audio_path, chunks))
    
    # Initialize split containers
    train_samples = []
    val_samples = []
    test_speaker_samples = []
    test_sentence_samples = []
    
    for speaker, audio_list in audios_by_speaker.items():
        # Shuffle audio files for this speaker
        shuffled_indices = np.random.permutation(len(audio_list))
        shuffled_audios = [audio_list[i] for i in shuffled_indices]
        
        if speaker in test_speakers:
            # Test speaker - all samples go to test_speaker split
            for audio_path, chunks in shuffled_audios:
                test_speaker_samples.extend(chunks)
        else:
            # Training speaker - split into train/val/test_sentence
            n_audios = len(shuffled_audios)
            
            # Determine split boundaries
            test_end = min(sentences_per_speaker_test, n_audios)
            val_end = min(test_end + sentences_per_speaker_val, n_audios)
            
            for i, (audio_path, chunks) in enumerate(shuffled_audios):
                if i < test_end:
                    test_sentence_samples.extend(chunks)
                elif i < val_end:
                    val_samples.extend(chunks)
                else:
                    train_samples.extend(chunks)
    
    return {
        "train": train_samples,
        "val": val_samples,
        "test_speaker": test_speaker_samples,
        "test_sentence": test_sentence_samples,
    }


def verify_sample(data_dir: Path, sample: dict) -> dict:
    """Verify a single sample's files."""
    try:
        h_y = np.load(data_dir / sample["h_y_path"])
        mel = np.load(data_dir / sample["mel_path"])
        spk_embed = np.load(data_dir / sample["spk_embed_path"])
        
        return {
            "valid": True,
            "h_y_shape": h_y.shape,
            "mel_shape": mel.shape,
            "spk_shape": spk_embed.shape,
        }
    except Exception as e:
        return {"valid": False, "error": str(e)}


def main():
    parser = argparse.ArgumentParser(description="Prepare and verify GAN decoder data splits")
    parser.add_argument("--data_dir", required=True, help="Path to gan_targets directory")
    parser.add_argument("--sentences_test", type=int, default=50, 
                        help="Sentences per speaker for test_sentence split")
    parser.add_argument("--sentences_val", type=int, default=50,
                        help="Sentences per speaker for validation split")
    parser.add_argument("--verify_samples", type=int, default=10,
                        help="Number of samples to verify")
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    manifest_path = data_dir / "manifest.json"
    
    if not manifest_path.exists():
        logger.error(f"Manifest not found: {manifest_path}")
        return
    
    # Analyze manifest
    logger.info("=" * 60)
    logger.info("ANALYZING GAN DECODER TARGETS")
    logger.info("=" * 60)
    
    data = analyze_manifest(manifest_path)
    logger.info(f"Total samples: {data['total_samples']}")
    
    # Print by accent
    logger.info("\nSamples by accent:")
    for accent, samples in sorted(data["by_accent"].items()):
        logger.info(f"  {accent}: {len(samples)}")
    
    # Print by speaker
    logger.info("\nSamples by speaker:")
    for speaker, samples in sorted(data["by_speaker"].items()):
        marker = " [TEST SPEAKER]" if speaker in TEST_SPEAKERS else ""
        logger.info(f"  {speaker}: {len(samples)}{marker}")
    
    # Create splits
    logger.info("\n" + "=" * 60)
    logger.info("CREATING SPLITS")
    logger.info("=" * 60)
    logger.info(f"Test speakers (completely held out): {TEST_SPEAKERS}")
    logger.info(f"Sentences per training speaker for test_sentence: {args.sentences_test}")
    logger.info(f"Sentences per training speaker for validation: {args.sentences_val}")
    
    splits = create_splits(
        data, TEST_SPEAKERS, 
        sentences_per_speaker_test=args.sentences_test,
        sentences_per_speaker_val=args.sentences_val
    )
    
    logger.info(f"\nTrain samples: {len(splits['train'])}")
    logger.info(f"Val samples: {len(splits['val'])}")
    logger.info(f"Test (held-out speakers): {len(splits['test_speaker'])}")
    logger.info(f"Test (unseen sentences): {len(splits['test_sentence'])}")
    
    # Count samples by accent for each split
    logger.info("\n--- Train Split ---")
    train_by_accent = defaultdict(int)
    train_by_speaker = defaultdict(int)
    for sample in splits["train"]:
        train_by_accent[sample.get("accent", "unknown")] += 1
        train_by_speaker[sample["speaker_id"]] += 1
    logger.info(f"By accent: {dict(train_by_accent)}")
    logger.info(f"Speakers: {list(train_by_speaker.keys())}")
    
    logger.info("\n--- Val Split ---")
    val_by_speaker = defaultdict(int)
    for sample in splits["val"]:
        val_by_speaker[sample["speaker_id"]] += 1
    logger.info(f"By speaker: {dict(val_by_speaker)}")
    
    logger.info("\n--- Test Speaker Split ---")
    test_speaker_by_speaker = defaultdict(int)
    for sample in splits["test_speaker"]:
        test_speaker_by_speaker[sample["speaker_id"]] += 1
    logger.info(f"By speaker: {dict(test_speaker_by_speaker)}")
    
    logger.info("\n--- Test Sentence Split ---")
    test_sentence_by_speaker = defaultdict(int)
    for sample in splits["test_sentence"]:
        test_sentence_by_speaker[sample["speaker_id"]] += 1
    logger.info(f"By speaker: {dict(test_sentence_by_speaker)}")
    
    # Verify samples
    logger.info("\n" + "=" * 60)
    logger.info("VERIFYING SAMPLES")
    logger.info("=" * 60)
    
    samples_to_verify = splits["train"][:args.verify_samples]
    valid_count = 0
    
    for sample in samples_to_verify:
        result = verify_sample(data_dir, sample)
        
        if result["valid"]:
            valid_count += 1
            logger.info(f"  ✓ {sample['id']}")
            logger.info(f"    h_y: {result['h_y_shape']}, mel: {result['mel_shape']}")
        else:
            logger.error(f"  ✗ {sample['id']}: {result['error']}")
    
    logger.info(f"\nVerified {valid_count}/{len(samples_to_verify)} samples successfully")
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Data directory: {data_dir}")
    logger.info(f"Total samples: {data['total_samples']}")
    logger.info(f"Train/Val/Test_Speaker/Test_Sentence: "
                f"{len(splits['train'])}/{len(splits['val'])}/"
                f"{len(splits['test_speaker'])}/{len(splits['test_sentence'])}")
    logger.info(f"Test speakers (held out): {TEST_SPEAKERS}")
    logger.info(f"Training speakers: {len(train_by_speaker)} speakers")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
