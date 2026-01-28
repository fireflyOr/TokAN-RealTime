"""
fast_synthesizer.py: Integration of GAN decoder with TokAN pipeline.

This module provides a drop-in replacement for the CFM-based synthesizer
using the trained GAN decoder for fast inference.
"""

import torch
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class FastMelSynthesizer:
    """
    Fast mel synthesizer using GAN-based decoder.
    
    Drop-in replacement for MelSynthesizer in components.py
    """
    
    def __init__(
        self,
        gan_decoder_path: str,
        tokan_encoder_path: str,
        device: torch.device,
    ):
        """
        Initialize the fast synthesizer.
        
        Args:
            gan_decoder_path: Path to trained GAN decoder checkpoint
            tokan_encoder_path: Path to TokAN encoder checkpoint
            device: Torch device
        """
        self.gan_decoder_path = gan_decoder_path
        self.tokan_encoder_path = tokan_encoder_path
        self.device = device
        
        self._gan_decoder = None
        self._tokan_encoder = None
        self._duration_predictor = None
        self._spk_embedder = None

    def setup(self):
        """Load models."""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))
        
        from models.gan_decoder import get_gan_decoder
        from tokan.yirga.models.yirga_token_to_mel import YirgaTokenToMel
        
        logger.info("Loading TokAN encoder...")
        tokan_model = YirgaTokenToMel.load_from_checkpoint(
            self.tokan_encoder_path,
            map_location=self.device
        )
        
        # Extract encoder components
        self._tokan_encoder = tokan_model.encoder.to(self.device).eval()
        self._duration_predictor = tokan_model.duration_predictor
        if self._duration_predictor is not None:
            self._duration_predictor = self._duration_predictor.to(self.device).eval()
        self._spk_embedder = tokan_model.spk_embedder.to(self.device).eval()
        self._upsample_rate = tokan_model.upsample_rate
        self._predict_duration = tokan_model.predict_duration
        
        # Freeze encoder components
        for module in [self._tokan_encoder, self._spk_embedder]:
            for param in module.parameters():
                param.requires_grad = False
        
        if self._duration_predictor is not None:
            for param in self._duration_predictor.parameters():
                param.requires_grad = False
        
        logger.info("Loading GAN decoder...")
        # Load GAN decoder
        checkpoint = torch.load(self.gan_decoder_path, map_location=self.device)
        
        # Get decoder config
        config = checkpoint.get("config", {})
        model_config = config.get("model", {})
        
        self._gan_decoder = get_gan_decoder(
            decoder_type=model_config.get("decoder_type", "conv"),
            in_channels=model_config.get("in_channels", 256),
            out_channels=model_config.get("out_channels", 80),
            hidden_channels=model_config.get("hidden_channels", 512),
            kernel_sizes=tuple(model_config.get("kernel_sizes", [3, 7, 11])),
            dilations=tuple(tuple(d) for d in model_config.get("dilations", [[1,3,5]]*3)),
            n_res_blocks=model_config.get("n_res_blocks", 3),
            spk_emb_dim=model_config.get("spk_emb_dim", 256),
            dropout=0.0,  # No dropout during inference
        ).to(self.device)
        
        self._gan_decoder.load_state_dict(checkpoint["generator_state_dict"])
        self._gan_decoder.eval()
        
        for param in self._gan_decoder.parameters():
            param.requires_grad = False
        
        logger.info("Fast synthesizer ready!")

    @torch.inference_mode()
    def synthesize(
        self,
        tokens_list: List[torch.Tensor],
        spk_embed: torch.Tensor,
        durations: Optional[List[float]] = None,
        preserve_duration: bool = False,
        hop: int = 256,
        sr: int = 22050,
        **kwargs  # Ignore extra args like n_timesteps
    ) -> List[torch.Tensor]:
        """
        Synthesize mel spectrograms from tokens.
        
        Args:
            tokens_list: List of token tensors
            spk_embed: Speaker embedding (1, spk_dim)
            durations: Optional list of target durations
            preserve_duration: Whether to preserve original durations
            hop: Vocoder hop size
            sr: Sample rate
            
        Returns:
            List of mel spectrogram tensors
        """
        from tokan.matcha.utils.model import (
            sequence_mask, fix_len_compatibility, generate_path
        )
        
        mels = []
        
        for i, tokens in enumerate(tokens_list):
            # Prepare inputs
            x = tokens.unsqueeze(0).to(self.device)
            x_lengths = torch.tensor([len(tokens)]).to(self.device)
            spks = spk_embed.to(self.device)
            
            # Encode
            x_enc, x_mask = self._tokan_encoder(x, x_lengths, spks)
            
            # Add speaker embedding
            spk_emb = self._spk_embedder(spks)
            x_enc = x_enc + spk_emb.unsqueeze(-1) * x_mask
            
            # Predict/compute durations and alignment
            if self._predict_duration and self._duration_predictor is not None:
                # Use duration predictor
                total_dur = None
                if preserve_duration and durations and i < len(durations):
                    total_dur = torch.tensor(durations[i] * sr // hop).long().to(self.device)
                
                if total_dur is not None and hasattr(self._duration_predictor, 'support_total_duration') and self._duration_predictor.support_total_duration:
                    w = self._duration_predictor(x_enc, x_mask, total_duration=total_dur)
                else:
                    w = self._duration_predictor(x_enc, x_mask)
                
                from tokan.yirga.utils.model import scale_to_total_duration
                if total_dur is not None:
                    w_ceil = scale_to_total_duration(w, total_dur)
                else:
                    w_ceil = self._duration_predictor.round_duration(w)
                
                y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
                y_max_length = y_lengths.max()
                y_max_length_ = fix_len_compatibility(y_max_length)
                
                y_mask = sequence_mask(y_lengths, y_max_length_).unsqueeze(1).to(x_mask.dtype)
                attn_mask = x_mask.unsqueeze(-1) * y_mask.unsqueeze(2)
                attn = generate_path(w_ceil.squeeze(1), attn_mask.squeeze(1)).unsqueeze(1)
            else:
                # Fixed upsampling
                from tokan.yirga.utils.model import generate_even_path
                
                y_lengths = torch.floor(x_lengths * self._upsample_rate).long()
                y_max_length = y_lengths.max()
                y_max_length_ = fix_len_compatibility(y_max_length)
                y_mask = sequence_mask(y_lengths, y_max_length_).unsqueeze(1).to(x_mask.dtype)
                attn_mask = x_mask.unsqueeze(-1) * y_mask.unsqueeze(2)
                attn = generate_even_path(self._upsample_rate, attn_mask.squeeze(1))
            
            # Expand encoder output
            h_y = torch.matmul(attn.squeeze(1).transpose(1, 2), x_enc.transpose(1, 2))
            h_y = h_y.transpose(1, 2)
            
            # Generate mel with GAN decoder (SINGLE FORWARD PASS!)
            y_max_length = y_lengths.max()
            y_mask = sequence_mask(y_lengths, y_max_length).unsqueeze(1).to(h_y.dtype)
            
            mel_pred, _ = self._gan_decoder(h_y[:, :, :y_max_length], y_mask, spks)
            
            # Trim to actual length
            actual_len = y_lengths[0].item()
            mel = mel_pred[:, :, :actual_len]
            
            mels.append(mel)
        
        return mels


class FastAccentConverter:
    """
    Fast accent converter using GAN-based decoder.
    
    Drop-in replacement for AccentConverter.
    """
    
    def __init__(self, config: Dict, gan_decoder_path: str, device: str = None):
        """
        Initialize fast accent converter.
        
        Args:
            config: Configuration dictionary (same as AccentConverter)
            gan_decoder_path: Path to trained GAN decoder
            device: Device string
        """
        self.cfg = config
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.gan_decoder_path = gan_decoder_path
        
        # Import original components
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        
        from components import AudioProcessor, FeatureExtractor, TokenConverter
        
        self.audio = AudioProcessor(self.device)
        self.features = FeatureExtractor(self.device)
        self.converter = TokenConverter(config["conversion_model"], self.device)
        
        # Use fast synthesizer instead of original
        self.synth = FastMelSynthesizer(
            gan_decoder_path=gan_decoder_path,
            tokan_encoder_path=config["synthesis_model"],
            device=self.device,
        )
        
        # Vocoder (same as original)
        from components import Vocoder
        self.vocoder = Vocoder(config, self.device)

    def setup(self):
        """Initialize all models."""
        logger.info(f"Initializing models on {self.device}...")
        
        self.features.setup_content_encoder(
            self.cfg["hubert_model"],
            self.cfg["hubert_layer"],
            self.cfg["hubert_km_path"]
        )
        self.converter.setup(beam_size=self.cfg["beam_size"])
        self.synth.setup()
        self.vocoder.setup()
        
        logger.info("Fast accent converter ready!")

    def convert(
        self,
        input_path: str,
        output_path: str = None,
    ) -> torch.Tensor:
        """
        Convert accent of input audio.
        
        Args:
            input_path: Path to input audio file
            output_path: Optional path to save output
            
        Returns:
            Converted waveform tensor
        """
        import torchaudio
        import numpy as np
        import time
        
        t0 = time.perf_counter()
        
        # Load and segment audio
        wav = self.audio.load_audio(input_path)
        ts = self.audio.segment_audio(
            wav,
            self.cfg["vad_threshold"],
            self.cfg["max_duration"],
            self.cfg["min_duration"]
        )
        
        if not ts:
            logger.warning("No speech detected in audio")
            return None
        
        chunks = self.audio.split_waveform(wav, ts)
        
        # Extract embeddings from longest chunk
        durations = [c['end'] - c['start'] for c in ts]
        embed_idx = np.argmax(durations)
        embed_wav = chunks[embed_idx]
        
        # Get speaker and accent embeddings
        spk_embed = self.features.extract_speaker_embedding(embed_wav.numpy())
        accent_embed, _ = self.features.extract_accent_embedding(embed_wav)
        
        # Extract and convert tokens
        src_tokens = self.features.extract_content_tokens(chunks)
        tgt_tokens = self.converter.convert_tokens(src_tokens, accent_embed)
        
        # Synthesize mel (FAST - single forward pass!)
        mels = self.synth.synthesize(
            tgt_tokens,
            spk_embed,
            durations,
            self.cfg["preserve_duration"],
            self.vocoder.hop_size,
            self.vocoder.sampling_rate,
        )
        
        # Vocode
        wavs = self.vocoder.generate_waveforms(mels)
        
        # Concatenate
        final_wav = self.audio.concatenate_waveforms(wavs, ts, self.vocoder.sampling_rate)
        
        total_time = time.perf_counter() - t0
        audio_duration = len(wav) / 16000
        rtf = total_time / audio_duration
        
        logger.info(f"Conversion complete: RTF = {rtf:.3f}")
        
        # Save if output path provided
        if output_path:
            import os
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            torchaudio.save(output_path, final_wav.cpu(), self.vocoder.sampling_rate)
            logger.info(f"Saved to {output_path}")
        
        return final_wav


# =============================================================================
# Benchmarking
# =============================================================================

def benchmark_speed(
    gan_decoder_path: str,
    tokan_checkpoint: str,
    test_lengths: List[int] = [50, 100, 200, 500],
    n_runs: int = 10,
    device: str = "cuda",
):
    """
    Benchmark GAN decoder speed vs CFM decoder.
    
    Args:
        gan_decoder_path: Path to trained GAN decoder
        tokan_checkpoint: Path to TokAN checkpoint
        test_lengths: Sequence lengths to test
        n_runs: Number of runs per length
        device: Device to benchmark on
    """
    import time
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    from models.gan_decoder import get_gan_decoder
    from tokan.yirga.models.yirga_token_to_mel import YirgaTokenToMel
    
    device = torch.device(device)
    
    # Load models
    logger.info("Loading models...")
    
    # GAN decoder
    checkpoint = torch.load(gan_decoder_path, map_location=device)
    config = checkpoint.get("config", {})
    model_config = config.get("model", {})
    
    gan_decoder = get_gan_decoder(
        decoder_type=model_config.get("decoder_type", "conv"),
        in_channels=model_config.get("in_channels", 256),
        out_channels=model_config.get("out_channels", 80),
        hidden_channels=model_config.get("hidden_channels", 512),
        dropout=0.0,
    ).to(device).eval()
    gan_decoder.load_state_dict(checkpoint["generator_state_dict"])
    
    # CFM decoder (from TokAN)
    tokan_model = YirgaTokenToMel.load_from_checkpoint(
        tokan_checkpoint, map_location=device
    ).to(device).eval()
    
    print("\n" + "="*60)
    print("SPEED BENCHMARK: GAN vs CFM Decoder")
    print("="*60)
    print(f"{'Length':<10} {'GAN (ms)':<15} {'CFM-4 (ms)':<15} {'CFM-32 (ms)':<15} {'Speedup':<10}")
    print("-"*60)
    
    for seq_len in test_lengths:
        # Create dummy inputs
        h_y = torch.randn(1, 256, seq_len).to(device)
        y_mask = torch.ones(1, 1, seq_len).to(device)
        spks = torch.randn(1, 256).to(device)
        
        # Warmup
        with torch.inference_mode():
            for _ in range(3):
                _ = gan_decoder(h_y, y_mask, spks)
                _ = tokan_model.decoder(h_y, y_mask, 4, 1.0, spks)
        
        if device.type == "cuda":
            torch.cuda.synchronize()
        
        # Benchmark GAN
        gan_times = []
        with torch.inference_mode():
            for _ in range(n_runs):
                start = time.perf_counter()
                _ = gan_decoder(h_y, y_mask, spks)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                gan_times.append((time.perf_counter() - start) * 1000)
        
        # Benchmark CFM (4 steps)
        cfm4_times = []
        with torch.inference_mode():
            for _ in range(n_runs):
                start = time.perf_counter()
                _ = tokan_model.decoder(h_y, y_mask, 4, 1.0, spks)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                cfm4_times.append((time.perf_counter() - start) * 1000)
        
        # Benchmark CFM (32 steps)
        cfm32_times = []
        with torch.inference_mode():
            for _ in range(n_runs):
                start = time.perf_counter()
                _ = tokan_model.decoder(h_y, y_mask, 32, 1.0, spks)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                cfm32_times.append((time.perf_counter() - start) * 1000)
        
        gan_avg = sum(gan_times) / len(gan_times)
        cfm4_avg = sum(cfm4_times) / len(cfm4_times)
        cfm32_avg = sum(cfm32_times) / len(cfm32_times)
        speedup = cfm32_avg / gan_avg
        
        print(f"{seq_len:<10} {gan_avg:<15.2f} {cfm4_avg:<15.2f} {cfm32_avg:<15.2f} {speedup:<10.1f}x")
    
    print("="*60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["convert", "benchmark"], required=True)
    parser.add_argument("--gan_decoder", required=True, help="Path to GAN decoder checkpoint")
    parser.add_argument("--tokan_checkpoint", required=True, help="Path to TokAN checkpoint")
    parser.add_argument("--input", help="Input audio file (for convert mode)")
    parser.add_argument("--output", help="Output audio file (for convert mode)")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    if args.mode == "benchmark":
        benchmark_speed(
            gan_decoder_path=args.gan_decoder,
            tokan_checkpoint=args.tokan_checkpoint,
            device=args.device,
        )
    else:
        # Convert mode requires additional config
        logger.error("Convert mode requires full config. Use FastAccentConverter class directly.")
