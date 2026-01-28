"""
components.py: Core processing components for the TokAN pipeline.
"""
import sys
import argparse
import logging
import torch
import torchaudio
import torch.nn.functional as F
import numpy as np
from torch.quantization import quantize_dynamic
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
from einops import rearrange
from torch import Tensor
from pathlib import Path

logger = logging.getLogger(__name__)

# ==============================================================================
# Audio Processor
# ==============================================================================
class AudioProcessor:
    def __init__(self, device: torch.device):
        self.device = device
        self._vad_model = None

    @property
    def vad_model(self):
        if self._vad_model is None:
            from silero_vad import load_silero_vad
            self._vad_model = load_silero_vad()
        return self._vad_model

    def load_audio(self, path: str, target_sr: int = 16000) -> Tensor:
        from silero_vad import read_audio
        return read_audio(path, sampling_rate=target_sr)

    def segment_audio(self, wav: Tensor, threshold: float = 0.5, 
                     max_duration: float = 10.0, min_duration: float = 3.0) -> List[Dict]:
        from silero_vad import get_speech_timestamps
        timestamps = get_speech_timestamps(wav, self.vad_model, return_seconds=True, threshold=threshold)
        return self._reorganize_chunks(timestamps, max_duration, min_duration)

    def _reorganize_chunks(self, timestamps: List[Dict], max_d: float, min_d: float) -> List[Dict]:
        if not timestamps: return []
        reorganized, i = [], 0
        while i < len(timestamps):
            chunk = timestamps[i]
            dur = chunk["end"] - chunk["start"]
            
            # Case 1: Too long (keep as is, warn user)
            if dur > max_d:
                reorganized.append(chunk)
                i += 1
                continue
            
            # Case 2: Too short (try to merge with next)
            if dur < min_d and i < len(timestamps) - 1:
                next_chunk = timestamps[i+1]
                if (next_chunk["end"] - chunk["start"]) <= max_d:
                    reorganized.append({"start": chunk["start"], "end": next_chunk["end"]})
                    i += 2
                    continue
            
            # Case 3: Just right
            reorganized.append(chunk)
            i += 1
        return reorganized

    def split_waveform(self, wav: Tensor, timestamps: List[Dict], sr: int = 16000) -> List[Tensor]:
        if len(timestamps) <= 1: return [wav]
        # Extract segments based on timestamps
        return [wav[int(c["start"]*sr):int(c["end"]*sr)] for c in timestamps]

    def concatenate_waveforms(self, wav_list: List[Tensor], timestamps: List[Dict], sr: int) -> Tensor:
        concat_list = []
        prev_end = 0.0
        for wav, chunk in zip(wav_list, timestamps):
            # Insert silence if needed
            silence_dur = chunk["start"] - prev_end
            if silence_dur > 0:
                concat_list.append(torch.zeros(int(silence_dur * sr)).to(self.device))
            concat_list.append(wav.squeeze().to(self.device))
            prev_end = chunk["end"]
        return torch.cat(concat_list, dim=-1).unsqueeze(0)

# ==============================================================================
# Feature Extractor
# ==============================================================================
class FeatureExtractor:
    def __init__(self, device: torch.device):
        self.device = device
        self._speaker_encoder = None
        self._accent_model = None
        self._speech_encoder = None

    def setup_content_encoder(self, hubert_path: str, layer: int, kmeans_path: str):
        from tokan.textless.hubert_feature_reader import HubertFeatureReader
        from tokan.textless.kmeans_quantizer import KMeansQuantizer
        from tokan.textless.speech_encoder import SpeechEncoder
        
        hubert = HubertFeatureReader(hubert_path, layer=layer)
        kmeans = KMeansQuantizer(kmeans_path)
        self._speech_encoder = SpeechEncoder(hubert, kmeans, need_f0=False, deduplicate=True, padding=True).to(self.device)

    def extract_speaker_embedding(self, wav: np.ndarray) -> Tensor:
        if self._speaker_encoder is None:
            from resemblyzer import VoiceEncoder
            self._speaker_encoder = VoiceEncoder(device=self.device)
        embed = self._speaker_encoder.embed_utterance(wav)
        return torch.FloatTensor(embed).unsqueeze(0).to(self.device)

    def extract_accent_embedding(self, wav: Tensor) -> Tuple[Tensor, Tensor]:
        if self._accent_model is None:
            from speechbrain.pretrained.interfaces import foreign_class
            self._accent_model = foreign_class(
                source="Jzuluaga/accent-id-commonaccent_xlsr-en-english",
                pymodule_file="custom_interface.py",
                classname="CustomEncoderWav2vec2Classifier",
                run_opts={"device": self.device},
            )
        
        batch = wav.unsqueeze(0).to(self._accent_model.device)
        embeds = self._accent_model.encode_batch(batch, torch.tensor([1.0]))
        embeds = embeds.squeeze(dim=1)
        # Note: We return normalized embedding and probs (probs calculation omitted for speed if not used)
        return F.layer_norm(embeds, embeds.shape), None 

    def extract_content_tokens(self, wav_list: List[Tensor]) -> List[Tensor]:
        if self._speech_encoder is None: raise RuntimeError("Content encoder not initialized.")
        return [self._speech_encoder(wav)["units"].squeeze(0) for wav in tqdm(wav_list, desc="Extracting tokens", leave=False)]

# ==============================================================================
# Token Converter
# ==============================================================================
class TokenConverter:
    def __init__(self, model_path: str, device: torch.device):
        self.path = model_path
        self.device = device
        self._generator = None
        self._task = None
        self._models = None

    def setup(self, beam_size: int = 5):
        from fairseq import checkpoint_utils, utils as fairseq_utils
        import os

        # Get absolute path to fairseq_modules
        module_dir = Path(__file__).parent / "tokan" / "fairseq_modules"
        if not module_dir.exists():
            # Try parent directory (in case we're in a subdirectory)
            module_dir = Path(__file__).parent.parent / "tokan" / "fairseq_modules"

        try:
            fairseq_utils.import_user_module(argparse.Namespace(user_dir=str(module_dir)))
        except Exception as e:
            logger.warning(f"Failed to import fairseq user module: {e}")
            # Try relative path as fallback
            try:
                fairseq_utils.import_user_module(argparse.Namespace(user_dir="tokan/fairseq_modules"))
            except Exception as e2:
                logger.error(f"Failed to import fairseq user module with relative path: {e2}")

        self._models, cfg, self._task = checkpoint_utils.load_model_ensemble_and_task([self.path])
        self._models[0].to(self.device).eval()
        cfg.generation.beam = beam_size
        self._generator = self._task.build_generator(self._models, cfg.generation)

    def convert_tokens(self, token_list: List[Tensor], accent_embed: Tensor) -> List[Tensor]:
        if not self._generator: raise RuntimeError("Converter not setup.")
        results = []
        for tokens in tqdm(token_list, desc="Converting", leave=False):
            src_str = " ".join(map(str, tokens.tolist()))
            src_tokens = self._task.source_dictionary.encode_line(src_str, add_if_not_exist=False, append_eos=False).long()
            src_batch = src_tokens.unsqueeze(0).to(self.device)
            
            sample = {"net_input": {
                "src_tokens": src_batch, 
                "src_lengths": torch.tensor([src_batch.shape[1]]).to(self.device),
                "condition_embeds": accent_embed
            }}
            
            hypos = self._task.inference_step(self._generator, self._models, sample)
            hypo_str = self._task.target_dictionary.string(hypos[0][0]["tokens"].int().cpu())
            results.append(torch.LongTensor([int(t) for t in hypo_str.split()]).to(self.device))
        return results

# ==============================================================================
# Synthesizer & Vocoder
# ==============================================================================
class MelSynthesizer:
    def __init__(self, path: str, device: torch.device):
        self.path = path
        self.device = device
        self._model = None

    def setup(self):
        from tokan.yirga.models.yirga_token_to_mel import YirgaTokenToMel
        self._model = YirgaTokenToMel.load_from_checkpoint(self.path, map_location=self.device).to(self.device).eval()

    def synthesize(self, tokens_list: List[Tensor], spk_embed: Tensor, durations: List[float], 
                   preserve: bool, hop: int, sr: int, n_timesteps: int = 32) -> List[Tensor]:
        mels = []
        for i, tokens in enumerate(tqdm(tokens_list, desc="Synthesizing", leave=False)):
            dur_tensor = None
            if preserve and durations:
                dur_tensor = torch.tensor(durations[i] * sr // hop).long().to(self.device)
            
            out = self._model.synthesise(
                x=tokens.unsqueeze(0), 
                x_lengths=torch.tensor([len(tokens)]).to(self.device),
                spks=spk_embed, 
                cond=None, 
                total_duration=dur_tensor, 
                n_timesteps=n_timesteps
            )
            mels.append(out["mel"])
        return mels

# Fast Mel Synthesizer (single forward pass GAN-based decoder replace CFM iterative decoder)
class FastMelSynthesizer:
    def __init__(self, gan_path: str, tokan_path: str, device: torch.device):
        self.gan_path = gan_path
        self.tokan_path = tokan_path
        self.device = device
        self._gan_decoder = None
        self._encoder = None
        self._spk_embedder = None
        self._duration_predictor = None

    def setup(self):
        from tokan.yirga.models.yirga_token_to_mel import YirgaTokenToMel
        
        # Load TokAN encoder components
        tokan = YirgaTokenToMel.load_from_checkpoint(self.tokan_path, map_location=self.device)
        self._encoder = tokan.encoder.to(self.device).eval()
        self._spk_embedder = tokan.spk_embedder.to(self.device).eval()
        self._duration_predictor = tokan.duration_predictor
        if self._duration_predictor:
            self._duration_predictor = self._duration_predictor.to(self.device).eval()
        self._upsample_rate = tokan.upsample_rate
        self._predict_duration = tokan.predict_duration
        
        # Freeze encoder
        for p in self._encoder.parameters(): p.requires_grad = False
        for p in self._spk_embedder.parameters(): p.requires_grad = False
        
        # Load GAN decoder
        ckpt = torch.load(self.gan_path, map_location=self.device)
        cfg = ckpt.get("config", {}).get("model", {})
        
        # Import locally to avoid circular imports
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent / "fast_synthesizer" / "models"))
        from gan_decoder import get_gan_decoder
        
        self._gan_decoder = get_gan_decoder(
            decoder_type=cfg.get("decoder_type", "conv"),
            in_channels=cfg.get("in_channels", 512),
            out_channels=cfg.get("out_channels", 80),
            hidden_channels=cfg.get("hidden_channels", 512),
            kernel_sizes=tuple(cfg.get("kernel_sizes", [3, 7, 11])),
            dilations=tuple(tuple(d) for d in cfg.get("dilations", [[1,3,5]]*3)),
            n_res_blocks=cfg.get("n_res_blocks", 3),
            spk_emb_dim=cfg.get("spk_emb_dim", 256),
            dropout=0.0,
        ).to(self.device).eval()
        
        # Load weights
        self._gan_decoder.load_state_dict(ckpt["generator_state_dict"])
        
        for p in self._gan_decoder.parameters(): p.requires_grad = False
        logger.info("FastMelSynthesizer ready (GAN decoder loaded)")
        # Clear CUDA cache after loading all models
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

    @torch.no_grad()
    def synthesize(self, tokens_list: List[Tensor], spk_embed: Tensor, durations: List[float],
                   preserve: bool, hop: int, sr: int, **kwargs) -> List[Tensor]:
        from tokan.matcha.utils.model import sequence_mask, fix_len_compatibility, generate_path
        
        mels = []
        for i, tokens in enumerate(tokens_list):
            x = tokens.unsqueeze(0).to(self.device)
            x_lengths = torch.tensor([len(tokens)]).to(self.device)
            spks = spk_embed.to(self.device)
            
            # Encode
            x_enc, x_mask = self._encoder(x, x_lengths, spks)
            spk_emb = self._spk_embedder(spks)
            x_enc = x_enc + spk_emb.unsqueeze(-1) * x_mask
            
            # Duration & alignment
            if self._predict_duration and self._duration_predictor:
                total_dur = None
                if preserve and durations and i < len(durations):
                    total_dur = torch.tensor(durations[i] * sr // hop).long().to(self.device)
                
                w = self._duration_predictor(x_enc, x_mask)
                if total_dur is not None:
                    from tokan.yirga.utils.model import scale_to_total_duration
                    w_ceil = scale_to_total_duration(w, total_dur)
                else:
                    w_ceil = self._duration_predictor.round_duration(w)
                
                y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
                y_max_length = fix_len_compatibility(y_lengths.max())
                y_mask = sequence_mask(y_lengths, y_max_length).unsqueeze(1).to(x_mask.dtype)
                attn_mask = x_mask.unsqueeze(-1) * y_mask.unsqueeze(2)
                attn = generate_path(w_ceil.squeeze(1), attn_mask.squeeze(1)).unsqueeze(1)
            else:
                from tokan.yirga.utils.model import generate_even_path
                y_lengths = torch.floor(x_lengths * self._upsample_rate).long()
                y_max_length = fix_len_compatibility(y_lengths.max())
                y_mask = sequence_mask(y_lengths, y_max_length).unsqueeze(1).to(x_mask.dtype)
                attn_mask = x_mask.unsqueeze(-1) * y_mask.unsqueeze(2)
                attn = generate_even_path(self._upsample_rate, attn_mask.squeeze(1))
            
            # Expand & generate (SINGLE FORWARD PASS)
            h_y = torch.matmul(attn.squeeze(1).transpose(1, 2), x_enc.transpose(1, 2)).transpose(1, 2)
            y_max = y_lengths.max()
            y_mask = sequence_mask(y_lengths, y_max).unsqueeze(1).to(h_y.dtype)
            
            mel_pred, _ = self._gan_decoder(h_y[:, :, :y_max], y_mask, spks)
            mels.append(mel_pred[:, :, :y_lengths[0].item()])
        
        return mels

# class Vocoder:
#     def __init__(self, tag_or_path: str, device: torch.device):
#         self.tag = tag_or_path
#         self.device = device
#         self._model = None

#     def setup(self):
#         from tokan.utils.model_utils import load_bigvgan
#         self._model = load_bigvgan(self.tag, device=self.device)

#     @property
#     def hop_size(self): return self._model.h["hop_size"]
    
#     @property
#     def sampling_rate(self): return self._model.h["sampling_rate"]

#     def generate_waveforms(self, mels: List[Tensor]) -> List[Tensor]:
#         # returns [1, T] tensors
#         return [rearrange(self._model(mel), "1 1 t -> 1 t") for mel in tqdm(mels, desc="Vocoding", leave=False)]

class Vocoder:
    """Converts mel spectrograms to waveforms using BigVGAN (PyTorch) or HiFi-GAN (ONNX)."""

    def __init__(self, model_path_or_tag: str, device: torch.device):
        self.model_path = model_path_or_tag
        self.device = device
        self._vocoder = None
        # Check if the file is an ONNX model
        self._is_onnx = str(model_path_or_tag).strip().lower().endswith(".onnx")
        
        # Default parameters (HiFi-GAN & BigVGAN typically use these)
        self._hop_size = 256
        self._sampling_rate = 22050

    def hop_size(self) -> int:
        # For ONNX, we use the default hardcoded value
        if not self._is_onnx and hasattr(self._vocoder, "h"):
             return self._vocoder.h["hop_size"]
        return self._hop_size

    def sampling_rate(self) -> int:
        if not self._is_onnx and hasattr(self._vocoder, "h"):
             return self._vocoder.h["sampling_rate"]
        return self._sampling_rate

    def setup(self) -> None:
        """Load the vocoder model."""
        if self._is_onnx:
            logger.info(f"⚡ Loading ONNX Vocoder from: {self.model_path}")
            
            # Configure providers: Try CUDA if available, else CPU
            providers = ['CPUExecutionProvider']
            if self.device.type == 'cuda':
                # Note: Requires onnxruntime-gpu installed for CUDA support
                providers.insert(0, 'CUDAExecutionProvider')
            
            try:
                self._vocoder = ort.InferenceSession(self.model_path, providers=providers)
                logger.info(f"ONNX Vocoder loaded on {providers[0]}")
            except Exception as e:
                logger.warning(f"Failed to load on requested device, falling back to CPU. Error: {e}")
                self._vocoder = ort.InferenceSession(self.model_path, providers=['CPUExecutionProvider'])
                
        else:
            logger.info(f"Loading PyTorch BigVGAN from: {self.model_path}")
            # Fallback to original BigVGAN loader logic
            from tokan.utils.model_utils import load_bigvgan
            self._vocoder = load_bigvgan(self.model_path, device=self.device)

    def generate_waveforms(self, mel_list: List[Tensor]) -> List[Tensor]:
        """Generate waveforms from mel spectrograms."""
        if self._vocoder is None:
            raise RuntimeError("Vocoder not initialized. Call setup first.")

        wav_list = []
        # No grad needed for inference
        with torch.no_grad():
            for mel in tqdm(mel_list, desc="Vocoding"):
                if self._is_onnx:
                    # --- ONNX INFERENCE ---
                    # 1. Convert Tensor to Numpy
                    # Mel shape coming in is typically [1, 80, T]
                    mel_np = mel.cpu().numpy().astype(np.float32)
                    
                    # 2. Get input name (usually 'mel' or 'input')
                    input_name = self._vocoder.get_inputs()[0].name
                    
                    # 3. Run Inference
                    # ONNX Runtime output is a list of numpy arrays
                    audio_np = self._vocoder.run(None, {input_name: mel_np})[0]
                    
                    # 4. Convert back to Tensor
                    # Output shape is typically [1, 1, T] or [1, T]
                    wav = torch.from_numpy(audio_np)
                    
                    # Ensure we have [1, T] shape
                    if wav.dim() == 3:
                        wav = wav.squeeze(1)
                    
                    if self.device.type == 'cuda':
                        wav = wav.to(self.device)
                        
                else:
                    # --- PYTORCH INFERENCE ---
                    if self.device.type == 'cuda':
                        mel = mel.to(self.device)
                    wav = self._vocoder(mel)
                    wav = rearrange(wav, "1 1 t -> 1 t")
                
                wav_list.append(wav)

        return wav_list
    
    def get_vocoder_name(self) -> str:
        """Extract a short name for the vocoder."""
        tag = self.model_path
        if self._is_onnx:
            return "HiFiGAN-ONNX"
        if "/" in tag:
            return tag.split("/")[-1]
        return Path(tag).stem if tag else "unknown_vocoder"