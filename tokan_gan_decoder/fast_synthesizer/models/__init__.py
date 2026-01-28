"""
TokAN GAN Decoder Package

Single-pass GAN-based decoder for fast TokAN inference.
"""

from .gan_decoder import GANMelDecoder, ConformerMelDecoder, get_gan_decoder
from .discriminators import CombinedDiscriminator, LightweightDiscriminator
from .losses import GANLoss, SimpleLoss, DistillationLoss

__all__ = [
    "GANMelDecoder",
    "ConformerMelDecoder", 
    "get_gan_decoder",
    "CombinedDiscriminator",
    "LightweightDiscriminator",
    "GANLoss",
    "SimpleLoss",
    "DistillationLoss",
]
