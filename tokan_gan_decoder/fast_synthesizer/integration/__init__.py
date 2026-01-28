"""Integration module for TokAN GAN Decoder."""

from .fast_synthesizer import (
    FastMelSynthesizer,
    FastAccentConverter,
    benchmark_speed,
)

__all__ = [
    "FastMelSynthesizer",
    "FastAccentConverter",
    "benchmark_speed",
]
