"""Data module for TokAN GAN Decoder."""

from .dataset import (
    PreprocessedDataset,
    ArcticDataset,
    MelExtractor,
    TokANFeatureExtractor,
    collate_fn,
    preprocess_dataset,
)

# v2: CFM Distillation approach
from .dataset_v2 import (
    TokANDistillationExtractor,
    preprocess_dataset_distillation,
    preprocess_dataset_mixed,
)

# v3: Proper splits with LightningDataModule (RECOMMENDED)
from .dataset import (
    TEST_SPEAKERS,
    TRAIN_SPEAKERS,
    L2_SPEAKERS,
    GANDecoderDataset,
    GANDecoderCollator,
    create_splits,
    preprocess_gan_targets,
)

from .gan_decoder_datamodule import (
    GANDecoderDataModule,
    verify_preprocessed_data,
)

__all__ = [
    # Original (same-speaker targets)
    "PreprocessedDataset",
    "ArcticDataset",
    "MelExtractor",
    "TokANFeatureExtractor",
    "collate_fn",
    "preprocess_dataset",
    # v2: Distillation approach
    "TokANDistillationExtractor",
    "preprocess_dataset_distillation",
    "preprocess_dataset_mixed",
    # v3: Proper splits (RECOMMENDED)
    "TEST_SPEAKERS",
    "TRAIN_SPEAKERS",
    "L2_SPEAKERS",
    "GANDecoderDataset",
    "GANDecoderCollator",
    "create_splits",
    "preprocess_gan_targets",
    "GANDecoderDataModule",
    "verify_preprocessed_data",
]
