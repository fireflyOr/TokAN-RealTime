"""
gan_decoder_datamodule.py: LightningDataModule for GAN decoder training.

Split Strategy:
1. Test speakers (EBVS, SKA) → completely held out for test_speaker evaluation
2. Training speakers (22 L2 speakers):
   - 50 audio files → test_sentence split (unseen sentences)
   - 50 audio files → validation split
   - Remaining → training split

This enables evaluation on:
- Unseen speakers (speaker generalization)
- Unseen sentences (content generalization)
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from lightning import LightningDataModule

logger = logging.getLogger(__name__)

# =============================================================================
# Speaker Configuration
# =============================================================================

# Test speakers - completely held out for evaluation
TEST_SPEAKERS = ["EBVS", "SKA"]

# All L2 speakers by accent (24 total)
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


# =============================================================================
# Dataset
# =============================================================================

class GANDecoderDataset(Dataset):
    """
    Dataset for GAN decoder training.
    
    Expected directory structure:
        gan_targets/
        ├── manifest.json
        ├── h_y/
        │   └── SPEAKER/filename.npy
        ├── mels/
        │   └── SPEAKER/filename.npy
        └── spk_embeds/
            └── SPEAKER/filename.npy
    """
    
    def __init__(
        self,
        data_dir: str,
        samples: List[Dict],
        max_mel_length: int = 1000,
        training: bool = False,
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.samples = samples
        self.max_mel_length = max_mel_length
        self.training = training
        
        logger.info(f"GANDecoderDataset initialized with {len(self.samples)} samples")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        
        # Load from separate files
        h_y = np.load(self.data_dir / sample["h_y_path"])
        target_mel = np.load(self.data_dir / sample["mel_path"])
        spk_emb = np.load(self.data_dir / sample["spk_embed_path"])
        
        # Convert to tensors
        h_y = torch.from_numpy(h_y).float()
        target_mel = torch.from_numpy(target_mel).float()
        spk_emb = torch.from_numpy(spk_emb).float().squeeze()
        
        # Ensure correct dimensions: (channels, time)
        if h_y.dim() == 3:
            h_y = h_y.squeeze(0)
        if target_mel.dim() == 3:
            target_mel = target_mel.squeeze(0)
        
        # Truncate if too long
        if target_mel.shape[-1] > self.max_mel_length:
            h_y = h_y[..., :self.max_mel_length]
            target_mel = target_mel[..., :self.max_mel_length]
        
        # Ensure same length
        min_len = min(h_y.shape[-1], target_mel.shape[-1])
        h_y = h_y[..., :min_len]
        target_mel = target_mel[..., :min_len]
        
        return {
            "h_y": h_y,
            "target_mel": target_mel,
            "spk_emb": spk_emb,
            "length": min_len,
            "sample_id": sample["id"],
            "speaker_id": sample["speaker_id"],
            "accent": sample.get("accent", "unknown"),
        }


# =============================================================================
# Collator
# =============================================================================

class GANDecoderCollator:
    """Collate function for batching GAN decoder samples."""
    
    def __init__(self, h_y_dim: int = 256, n_mels: int = 80):
        self.h_y_dim = h_y_dim
        self.n_mels = n_mels
    
    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        max_len = max(x["length"] for x in batch)
        batch_size = len(batch)
        
        # Initialize tensors
        h_y = torch.zeros(batch_size, self.h_y_dim, max_len)
        target_mel = torch.zeros(batch_size, self.n_mels, max_len)
        spk_emb = torch.stack([x["spk_emb"] for x in batch])
        lengths = torch.zeros(batch_size, dtype=torch.long)
        
        # Fill tensors
        for i, sample in enumerate(batch):
            length = sample["length"]
            h_y[i, :, :length] = sample["h_y"]
            target_mel[i, :, :length] = sample["target_mel"]
            lengths[i] = length
        
        # Create mask
        mask = torch.arange(max_len).unsqueeze(0) < lengths.unsqueeze(1)
        mask = mask.unsqueeze(1).float()
        
        return {
            "h_y": h_y,
            "mel": target_mel,
            "spk_embed": spk_emb,
            "lengths": lengths,
            "mask": mask,
        }


# =============================================================================
# Split Creation
# =============================================================================

def create_splits(
    samples: List[Dict],
    test_speakers: List[str] = None,
    sentences_per_speaker_test: int = DEFAULT_SENTENCES_PER_SPEAKER_TEST,
    sentences_per_speaker_val: int = DEFAULT_SENTENCES_PER_SPEAKER_VAL,
    seed: int = 42,
) -> Dict[str, List[Dict]]:
    """
    Create train/val/test splits with comprehensive holdout strategy.

    Strategy:
    1. Test speakers (EBVS, SKA) → all samples go to 'test_speaker' split
    2. Training speakers (22 L2 speakers):
       - First 50 audio files → 'test_sentence' split (unseen sentences)
       - Next 50 audio files → 'val' split
       - Remaining → 'train' split

    Groups samples by audio_path to ensure all chunks from the same
    audio file stay together (prevents data leakage).
    """
    if test_speakers is None:
        test_speakers = TEST_SPEAKERS
    
    np.random.seed(seed)
    
    # Group all samples by audio_path
    by_audio = defaultdict(list)
    for sample in samples:
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
    
    logger.info(f"Split created:")
    logger.info(f"  train: {len(train_samples)}")
    logger.info(f"  val: {len(val_samples)}")
    logger.info(f"  test_speaker: {len(test_speaker_samples)}")
    logger.info(f"  test_sentence: {len(test_sentence_samples)}")
    
    return {
        "train": train_samples,
        "val": val_samples,
        "test_speaker": test_speaker_samples,
        "test_sentence": test_sentence_samples,
    }


# =============================================================================
# LightningDataModule
# =============================================================================

class GANDecoderDataModule(LightningDataModule):
    """
    LightningDataModule for GAN decoder training with CFM distillation targets.
    
    Implements comprehensive split strategy:
    - Test speakers (EBVS, SKA) are completely held out
    - Training speakers have 50 test + 50 val sentences reserved per speaker
    
    This enables evaluation on:
    - Unseen speakers (speaker generalization)
    - Unseen sentences (content generalization)
    """
    
    def __init__(
        self,
        data_dir: str,
        batch_size: int = 16,
        num_workers: int = 4,
        pin_memory: bool = True,
        test_speakers: List[str] = None,
        sentences_per_speaker_test: int = DEFAULT_SENTENCES_PER_SPEAKER_TEST,
        sentences_per_speaker_val: int = DEFAULT_SENTENCES_PER_SPEAKER_VAL,
        max_mel_length: int = 1000,
        h_y_dim: int = 256,
        n_mels: int = 80,
        seed: int = 42,
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.test_speakers = test_speakers or TEST_SPEAKERS
        self.sentences_per_speaker_test = sentences_per_speaker_test
        self.sentences_per_speaker_val = sentences_per_speaker_val
        self.max_mel_length = max_mel_length
        self.h_y_dim = h_y_dim
        self.n_mels = n_mels
        self.seed = seed
        
        self.collator = GANDecoderCollator(h_y_dim=h_y_dim, n_mels=n_mels)
        
        # Will be populated in setup()
        self.splits = None
    
    def setup(self, stage: Optional[str] = None):
        """Load manifest and create splits."""
        # Load manifest
        manifest_path = self.data_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")
        
        with open(manifest_path) as f:
            all_samples = json.load(f)
        
        logger.info(f"Loaded {len(all_samples)} samples from manifest")
        
        # Create splits
        self.splits = create_splits(
            samples=all_samples,
            test_speakers=self.test_speakers,
            sentences_per_speaker_test=self.sentences_per_speaker_test,
            sentences_per_speaker_val=self.sentences_per_speaker_val,
            seed=self.seed,
        )
        
        # Create datasets based on stage
        if stage == "fit" or stage is None:
            self.train_dataset = GANDecoderDataset(
                data_dir=self.data_dir,
                samples=self.splits["train"],
                max_mel_length=self.max_mel_length,
                training=True,
            )
            self.val_dataset = GANDecoderDataset(
                data_dir=self.data_dir,
                samples=self.splits["val"],
                max_mel_length=self.max_mel_length,
                training=False,
            )
        
        if stage == "test" or stage is None:
            self.test_speaker_dataset = GANDecoderDataset(
                data_dir=self.data_dir,
                samples=self.splits["test_speaker"],
                max_mel_length=self.max_mel_length,
                training=False,
            )
            self.test_sentence_dataset = GANDecoderDataset(
                data_dir=self.data_dir,
                samples=self.splits["test_sentence"],
                max_mel_length=self.max_mel_length,
                training=False,
            )
    
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self.collator,
            drop_last=True,
        )
    
    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self.collator,
        )
    
    def test_dataloader(self) -> List[DataLoader]:
        """
        Returns two test dataloaders:
        1. test_speaker: samples from completely held-out speakers (EBVS, SKA)
        2. test_sentence: unseen sentences from training speakers
        """
        return [
            DataLoader(
                self.test_speaker_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                collate_fn=self.collator,
            ),
            DataLoader(
                self.test_sentence_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                collate_fn=self.collator,
            ),
        ]
    
    def get_test_speaker_dataloader(self) -> DataLoader:
        """Get dataloader for held-out test speakers only."""
        return DataLoader(
            self.test_speaker_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self.collator,
        )
    
    def get_test_sentence_dataloader(self) -> DataLoader:
        """Get dataloader for unseen sentences from training speakers."""
        return DataLoader(
            self.test_sentence_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self.collator,
        )
    
    def get_split_info(self) -> Dict:
        """Return information about the splits."""
        if self.splits is None:
            return {}
        
        def count_by_speaker(samples):
            counts = defaultdict(int)
            for s in samples:
                counts[s["speaker_id"]] += 1
            return dict(counts)
        
        def count_by_accent(samples):
            counts = defaultdict(int)
            for s in samples:
                counts[s.get("accent", "unknown")] += 1
            return dict(counts)
        
        return {
            "train": {
                "total": len(self.splits["train"]),
                "by_speaker": count_by_speaker(self.splits["train"]),
                "by_accent": count_by_accent(self.splits["train"]),
            },
            "val": {
                "total": len(self.splits["val"]),
                "by_speaker": count_by_speaker(self.splits["val"]),
                "by_accent": count_by_accent(self.splits["val"]),
            },
            "test_speaker": {
                "total": len(self.splits["test_speaker"]),
                "by_speaker": count_by_speaker(self.splits["test_speaker"]),
                "by_accent": count_by_accent(self.splits["test_speaker"]),
            },
            "test_sentence": {
                "total": len(self.splits["test_sentence"]),
                "by_speaker": count_by_speaker(self.splits["test_sentence"]),
                "by_accent": count_by_accent(self.splits["test_sentence"]),
            },
        }
    
    def print_split_summary(self):
        """Print a summary of the data splits."""
        info = self.get_split_info()
        
        print("=" * 60)
        print("GAN DECODER DATA SPLITS SUMMARY")
        print("=" * 60)
        print(f"Test speakers (held out): {self.test_speakers}")
        print(f"Sentences per speaker for test_sentence: {self.sentences_per_speaker_test}")
        print(f"Sentences per speaker for validation: {self.sentences_per_speaker_val}")
        print()
        
        for split_name, split_info in info.items():
            print(f"{split_name.upper()}: {split_info['total']} samples")
            print(f"  By accent: {split_info['by_accent']}")
            print(f"  Speakers: {list(split_info['by_speaker'].keys())}")
            print()
        
        print("=" * 60)


# =============================================================================
# Utility Functions
# =============================================================================

def verify_preprocessed_data(data_dir: str, n_samples: int = 10) -> bool:
    """Verify preprocessed data integrity."""
    data_path = Path(data_dir)
    manifest_path = data_path / "manifest.json"
    
    if not manifest_path.exists():
        logger.error(f"Manifest not found: {manifest_path}")
        return False
    
    with open(manifest_path) as f:
        samples = json.load(f)
    
    logger.info(f"Verifying {min(n_samples, len(samples))} samples...")
    
    valid_count = 0
    for sample in samples[:n_samples]:
        try:
            h_y = np.load(data_path / sample["h_y_path"])
            mel = np.load(data_path / sample["mel_path"])
            spk_emb = np.load(data_path / sample["spk_embed_path"])
            
            logger.info(f"  ✓ {sample['id']}: h_y={h_y.shape}, mel={mel.shape}")
            valid_count += 1
        except Exception as e:
            logger.error(f"  ✗ {sample['id']}: {e}")
    
    success = valid_count == min(n_samples, len(samples))
    logger.info(f"\nVerified {valid_count}/{min(n_samples, len(samples))} samples")
    return success


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test GAN decoder data module")
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    if args.verify:
        verify_preprocessed_data(args.data_dir)
    else:
        # Test the data module
        dm = GANDecoderDataModule(
            data_dir=args.data_dir,
            batch_size=4,
            num_workers=0,
        )
        dm.setup()
        dm.print_split_summary()
        
        # Test loading a batch
        train_loader = dm.train_dataloader()
        batch = next(iter(train_loader))
        print("\nSample batch:")
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                print(f"  {k}: {v.shape}")
