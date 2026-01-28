"""
tokan_profiler.py: Unified Batch Profiler & Converter
"""
import os
import sys
import time
import json
import yaml
import logging
import argparse
import numpy as np
import torch
import torchaudio
from pathlib import Path
from typing import Dict, List, Optional
from collections import defaultdict
from functools import wraps
import onnxruntime as ort
from tqdm import tqdm
from einops import rearrange

# Import components
from components import AudioProcessor, FeatureExtractor, TokenConverter, MelSynthesizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("TokAN")

# ==============================================================================
# 1. Configuration & Model Preparation
# ==============================================================================
class ModelPrep:
    @staticmethod
    def ensure_config(cfg: Dict):
        required_keys = ["hubert_model", "hubert_km_path", "conversion_model", "synthesis_model"]
        missing_models = [k for k in required_keys if not cfg.get(k)]
        
        if missing_models:
            if not cfg.get("download_models", False):
                raise ValueError(f"Missing models {missing_models} and download_models=False")
            
            logger.info(f"Downloading missing models: {missing_models}...")
            from tokan.utils.model_utils import ModelManager
            key_map = {
                "hubert_model": "hubert",
                "hubert_km_path": "hubert_km",
                "conversion_model": "token_to_token",
                "synthesis_model": "token_to_mel_v2" if cfg.get("preserve_duration") else "token_to_mel_v1"
            }
            for k in missing_models:
                cfg[k] = ModelManager.ensure_model_available(key_map[k])
                
        defaults = {
            "vad_threshold": 0.5, "max_duration": 10.0, "min_duration": 3.0,
            "min_embed_duration": 5.0, "beam_size": 1, "preserve_duration": False,
            "bigvgan_tag_or_ckpt": "nvidia/bigvgan_22khz_80band",
            "hubert_layer": 17, "use_cache": False, "save_audio": True, "n_timesteps": 32 
        }
        for k, v in defaults.items():
            if k not in cfg or cfg[k] is None:
                cfg[k] = v     
        return cfg

# ==============================================================================
# 2. Vocoder Class (Supports BigVGAN & HiFi-GAN ONNX)
# ==============================================================================
class Vocoder:
    """Converts mel spectrograms to waveforms using BigVGAN (PyTorch) or HiFi-GAN (ONNX)."""

    def __init__(self, config: Dict, device: torch.device):
        self.device = device
        self.config = config
        self._vocoder = None
        
        # Determine type and path from config
        self.vocoder_type = config.get("vocoder_type", "bigvgan").lower()
        
        if self.vocoder_type == "hifigan":
            self.model_path = config.get("hifigan_onnx")
            self._is_onnx = True
            if not self.model_path or not os.path.exists(self.model_path):
                raise ValueError(f"HiFi-GAN ONNX path not found: {self.model_path}")
        else:
            self.model_path = config.get("bigvgan_tag_or_ckpt", "nvidia/bigvgan_22khz_80band")
            self._is_onnx = False
        
        # Default Parameters
        self._default_hop = config.get("hop_size", 256)
        self._default_sr = config.get("sampling_rate", 22050)

    @property
    def hop_size(self) -> int:
        if not self._is_onnx and hasattr(self._vocoder, "h"):
             return self._vocoder.h["hop_size"]
        return self._default_hop

    @property
    def sampling_rate(self) -> int:
        if not self._is_onnx and hasattr(self._vocoder, "h"):
             return self._vocoder.h["sampling_rate"]
        return self._default_sr


    def setup(self) -> None:
        """Load the vocoder model."""
        if self._is_onnx:
            logger.info(f"⚡ Loading HiFi-GAN (ONNX) from: {self.model_path}")
            
            # Configure providers (GPU support)
            providers = ['CPUExecutionProvider']
            if self.device.type == 'cuda':
                if 'CUDAExecutionProvider' in ort.get_available_providers():
                    providers.insert(0, 'CUDAExecutionProvider')
                else:
                    logger.warning("CUDA requested but onnxruntime-gpu not found/compatible. Using CPU.")
            
            try:
                self._vocoder = ort.InferenceSession(self.model_path, providers=providers)
                logger.info(f"Vocoder loaded on {self._vocoder.get_providers()[0]}")
            except Exception as e:
                logger.warning(f"Failed to load ONNX on requested device, falling back to CPU. Error: {e}")
                self._vocoder = ort.InferenceSession(self.model_path, providers=['CPUExecutionProvider'])
                
        else:
            logger.info(f"Loading BigVGAN (PyTorch) from: {self.model_path}")
            from tokan.utils.model_utils import load_bigvgan
            self._vocoder = load_bigvgan(self.model_path, device=self.device)

    def generate_waveforms(self, mel_list: List[torch.Tensor]) -> List[torch.Tensor]:
        """Generate waveforms from mel spectrograms."""
        if self._vocoder is None:
            raise RuntimeError("Vocoder not initialized. Call setup first.")

        wav_list = []
        with torch.no_grad():
            for mel in tqdm(mel_list, desc="Vocoding"):
                if self._is_onnx:
                    # --- ONNX INFERENCE ---
                    # 1. Prepare Input
                    mel_np = mel.cpu().numpy().astype(np.float32)
                    input_name = self._vocoder.get_inputs()[0].name
                    
                    # 2. Run Inference
                    audio_np = self._vocoder.run(None, {input_name: mel_np})[0]
                    
                    # 3. Convert back to Tensor
                    wav = torch.from_numpy(audio_np)
                    
                    # 4. Move to device & Fix Shape (Batch, Time)
                    if self.device.type == 'cuda':
                        wav = wav.to(self.device)
                        
                    # Squeeze (1, 1, T) -> (1, T)
                    if wav.dim() == 3: 
                        wav = wav.squeeze(1)
                else:
                    # --- PYTORCH INFERENCE ---
                    if self.device.type == 'cuda':
                        mel = mel.to(self.device)
                    wav = self._vocoder(mel)
                    
                    # Ensure shape is (1, T)
                    if wav.dim() == 3:
                        wav = wav.squeeze(1)
                
                wav_list.append(wav)

        return wav_list

# ==============================================================================
# 3. Latency Profiler
# ==============================================================================
class LatencyProfiler:
    def __init__(self):
        self.timings = defaultdict(list)
        self.enabled = True

    def reset(self):
        self.timings.clear()

    def get_summary(self):
        return {k: {"mean": float(np.mean(v)), "total": float(np.sum(v)), "count": len(v)} 
                for k, v in self.timings.items()}

def profile_step(name):
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            if not getattr(self, 'profiler', None) or not self.profiler.enabled:
                return func(self, *args, **kwargs)
            start = time.perf_counter()
            try:
                return func(self, *args, **kwargs)
            finally:
                self.profiler.timings[name].append(time.perf_counter() - start)
        return wrapper
    return decorator

# ==============================================================================
# 3. AccentConverter (Atomic File Processor)
# ==============================================================================
class AccentConverter:
    def __init__(self, config: Dict, device: str = None):
        self.cfg = config
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.profiler = LatencyProfiler()
        
        self.audio = AudioProcessor(self.device)
        self.features = FeatureExtractor(self.device)
        self.converter = TokenConverter(config["conversion_model"], self.device)
        #self.synth = MelSynthesizer(config["synthesis_model"], self.device)
        #self.vocoder = Vocoder(config["bigvgan_tag_or_ckpt"], self.device)
        self.vocoder = Vocoder(config, self.device)
        # Choose synthesizer based on config
        if config.get("use_gan_decoder") and config.get("gan_decoder_path"):
            from components import FastMelSynthesizer
            self.synth = FastMelSynthesizer(
                config["gan_decoder_path"], 
                config["synthesis_model"], 
                self.device
            )
            self.use_gan = True
        else:
            self.synth = MelSynthesizer(config["synthesis_model"], self.device)
            self.use_gan = False
            
        self.use_cache = config.get("use_cache", False)
        self._cached_spk, self._cached_act = None, None
        self._cache_valid = False

    def setup(self):
        logger.info(f"Initializing models on {self.device}...")
        self.features.setup_content_encoder(
            self.cfg["hubert_model"], self.cfg["hubert_layer"], self.cfg["hubert_km_path"]
        )
        self.converter.setup(beam_size=self.cfg["beam_size"])  ### use_compile=True for Token Converter optimization
        self.synth.setup()
        self.vocoder.setup()

    def clear_cache(self):
        self._cache_valid = False

    # --- Steps ---
    @profile_step("1_audio_loading")
    def step_load(self, path): return self.audio.load_audio(path)

    @profile_step("2_vad_segmentation")
    def step_segment(self, wav): 
        return self.audio.segment_audio(wav, self.cfg["vad_threshold"], 
                                      self.cfg["max_duration"], self.cfg["min_duration"])

    @profile_step("3_waveform_splitting")
    def step_split(self, wav, ts): return self.audio.split_waveform(wav, ts)

    def _get_embedding_wav(self, chunks, ts):
        if self.use_cache and self._cache_valid: return None
        
        # Logic to find best chunks for embedding
        durations = [c['end']-c['start'] for c in ts]
        sorted_idx = np.argsort(durations)[::-1]
        selected = []
        curr_dur = 0.0
        for i in sorted_idx:
            if curr_dur > self.cfg["min_embed_duration"]: break
            selected.append(chunks[i])
            curr_dur += durations[i]
        return torch.cat(selected, dim=0) if selected else chunks[0]

    @profile_step("4_speaker_embedding")
    def step_spk(self, wav): return self.features.extract_speaker_embedding(wav)

    @profile_step("5_accent_embedding")
    def step_act(self, wav): return self.features.extract_accent_embedding(wav)

    @profile_step("6_content_token_extraction")
    def step_content(self, chunks): return self.features.extract_content_tokens(chunks)

    @profile_step("7_token_conversion")
    def step_convert(self, toks, act): return self.converter.convert_tokens(toks, act)

    @profile_step("8_mel_synthesis")
    def step_synth(self, toks, spk, dur): 
        return self.synth.synthesize(toks, spk, dur, 
                                     self.cfg["preserve_duration"], 
                                     self.vocoder.hop_size, 
                                     self.vocoder.sampling_rate,
                                     n_timesteps=self.cfg["n_timesteps"])

    @profile_step("9_vocoding")
    def step_vocode(self, mels): return self.vocoder.generate_waveforms(mels)

    @profile_step("10_waveform_concatenation")
    def step_concat(self, wavs, ts): 
        return self.audio.concatenate_waveforms(wavs, ts, self.vocoder.sampling_rate)

    def run(self, input_path: str, output_path: str = None):
        self.profiler.reset()
        t0 = time.perf_counter()
        
        wav = self.step_load(input_path)
        ts = self.step_segment(wav)
        if not ts: return None
        
        chunks = self.step_split(wav, ts)
        
        # Embedding Logic with Caching
        if self.use_cache and self._cache_valid:
            spk, act = self._cached_spk, self._cached_act
        else:
            embed_wav = self._get_embedding_wav(chunks, ts)
            spk = self.step_spk(embed_wav.numpy())
            act, _ = self.step_act(embed_wav)
            if self.use_cache:
                self._cached_spk, self._cached_act = spk, act
                self._cache_valid = True

        src = self.step_content(chunks)
        tgt = self.step_convert(src, act)
        
        durs = [t['end'] - t['start'] for t in ts]
        mels = self.step_synth(tgt, spk, durs)
        resyn = self.step_vocode(mels)
        final = self.step_concat(resyn, ts)

        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            torchaudio.save(output_path, final.cpu(), self.vocoder.sampling_rate)

        total_t = time.perf_counter() - t0
        dur = len(wav) / 16000 
        
        return {
            "file": str(input_path), "duration": dur, "total_time": total_t,
            "rtf": total_t/dur if dur > 0 else 0, "timings": self.profiler.get_summary()
        }


# ==============================================================================
# 5. Batch Profiler & Statistics
# ==============================================================================
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)): return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)): return float(obj)
        elif isinstance(obj, (np.ndarray,)): return obj.tolist()
        return super().default(obj)

def compute_statistics(results: List[Dict]) -> Dict:
    """Computes detailed stats: totals, mean/std RTF, and component latency."""
    if not results: return {}

    rtfs = np.array([r["rtf"] for r in results])
    durs = np.array([r["duration"] for r in results])
    times = np.array([r["total_time"] for r in results])

    stats = {
        "count": len(results),
        "total_audio_duration": float(np.sum(durs)),
        "total_processing_time": float(np.sum(times)),
        "rtf": {
            "mean": float(np.mean(rtfs)),
            "std": float(np.std(rtfs)),
            "min": float(np.min(rtfs)),
            "max": float(np.max(rtfs)),
            "median": float(np.median(rtfs))
        },
        "components": {}
    }

    # Component Latency Analysis
    all_comps = set().union(*(r["timings"].keys() for r in results))
    for c in all_comps:
        # Get total time for this component across all files
        vals = [r["timings"][c]["total"] for r in results if c in r["timings"]]
        if vals:
            stats["components"][c] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "total": float(np.sum(vals))
            }
    return stats

class BatchLatencyProfiler:
    def __init__(self, config: Dict, pipeline: AccentConverter):
        self.cfg = config
        self.pipeline = pipeline
        self.results_by_speaker = defaultdict(list)
        self.results_by_accent = defaultdict(list)

    def profile_dataset(self, files_by_speaker: Dict[str, List[str]]):
        """Main loop: Process dataset and organize results by speaker/accent."""
        logger.info("Starting batch profiling...")
        
        with torch.inference_mode():
            for i, (spk, files) in enumerate(files_by_speaker.items(), 1):
                self._process_speaker(spk, files, i, len(files_by_speaker))
        
        return self.compute_grouped_statistics()

    def _process_speaker(self, spk: str, files: List[str], idx: int, total: int):
        print(f"\n{'='*70}\nSPEAKER {idx}/{total}: {spk}\n{'='*70}")
        
        if self.pipeline.use_cache:
            self.pipeline.clear_cache()
            print("Cache cleared. Populating from first file...")

        accent = SPK_TO_ACCENT.get(spk, "unknown")

        for j, fpath in enumerate(files, 1):
            print(f"[{j}/{len(files)}] {Path(fpath).name}", end=" ", flush=True)
            
            # Output path logic
            out_path = None
            if self.cfg.get("save_audio"):
                d = Path(self.cfg["output_audio_dir"]) / spk
                d.mkdir(parents=True, exist_ok=True)
                out_path = str(d / f"{Path(fpath).stem}.wav")

            try:
                # Run Inference
                res = self.pipeline.run(fpath, out_path)
                
                if res:
                    # Enrich and Store
                    res.update({"speaker": spk, "accent": accent})
                    if self.pipeline.use_cache: res["cache_hit"] = (j > 1)
                    
                    self.results_by_speaker[spk].append(res)
                    self.results_by_accent[accent].append(res)
                    
                    print(f"-> RTF: {res['rtf']:.3f} | Time: {res['total_time']:.2f}s")
            except Exception as e:
                print(f"\nFAILED: {e}")
                logger.error(f"Error processing {fpath}", exc_info=True)

    def compute_grouped_statistics(self) -> Dict:
        """Compute statistics grouped by speaker and accent."""
        all_results = [r for res_list in self.results_by_speaker.values() for r in res_list]
        
        return {
            "overall": compute_statistics(all_results),
            "by_speaker": {k: compute_statistics(v) for k, v in self.results_by_speaker.items()},
            "by_accent": {k: compute_statistics(v) for k, v in self.results_by_accent.items()}
        }

    def print_grouped_summary(self, stats: Dict):
        """Print comprehensive summary to console."""
        print("\n" + "="*70)
        print("OVERALL DATASET SUMMARY")
        print("="*70)
        if stats.get("overall"):
            s = stats["overall"]
            print(f" Files: {s['count']} | Duration: {s['total_audio_duration']:.2f}s | Proc Time: {s['total_processing_time']:.2f}s")
            print(f" RTF Mean: {s['rtf']['mean']:.3f} ± {s['rtf']['std']:.3f}")

        print("\n" + "="*70)
        print("PER-ACCENT BREAKDOWN")
        print("="*70)
        for accent, acc_stats in sorted(stats.get("by_accent", {}).items()):
            print(f"\n[{accent.upper()}]")
            print(f"  Files: {acc_stats['count']} | Total Dur: {acc_stats['total_audio_duration']:.2f}s")
            print(f"  Mean RTF: {acc_stats['rtf']['mean']:.3f} | Median RTF: {acc_stats['rtf']['median']:.3f}")

        print("\n" + "="*70)
        print("PER-SPEAKER SUMMARY")
        print("="*70)
        print(f"{'Speaker':<15} {'Accent':<12} {'Files':<8} {'Mean RTF':<12} {'Total Dur (s)':<15}")
        print("-" * 70)
        
        for spk, spk_stats in sorted(stats.get("by_speaker", {}).items()):
            accent = SPK_TO_ACCENT.get(spk, "unknown")
            print(f"{spk:<15} {accent:<12} {spk_stats['count']:<8} {spk_stats['rtf']['mean']:<12.3f} {spk_stats['total_audio_duration']:<15.2f}")

    def save_analysis(self, path: str):
        """Computes, prints, and saves statistics."""
        stats = self.compute_grouped_statistics()
        self.print_grouped_summary(stats)

        output = {
            "device": str(self.pipeline.device),
            "config": self.cfg,
            "statistics": stats,
            "raw_results": dict(self.results_by_speaker) # Save raw data grouped by speaker
        }
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(output, f, indent=2, cls=NumpyEncoder)
        print(f"\nAnalysis saved to: {path}")

# ==============================================================================
# 6. Helpers
# ==============================================================================
L2_SPEAKERS = {
    "arabic": ["ABA", "SKA", "YBAA", "ZHAA"], "chinese": ["BWC", "LXC", "NCC", "TXHC"],
    "hindi": ["ASI", "RRBI", "SVBI", "TNI"], "korean": ["HJK", "HKK", "YDCK", "YKWK"],
    "spanish": ["EBVS", "ERMS", "MBMPS", "NJS"], "vietnamese": ["HQTV", "PNV", "THV", "TLV"],
    "us": ["BDL", "RMS", "SLT", "CLB"]
}
SPK_TO_ACCENT = {spk: acc for acc, spks in L2_SPEAKERS.items() for spk in spks}

def get_dataset_files(root: str, cfg: Dict) -> Dict[str, List[str]]:
    files = defaultdict(list)
    root_path = Path(root)
    
    # Determine target speakers
    if cfg.get("speakers"):
        targets = cfg["speakers"]
    elif cfg.get("accents"):
        targets = [s for acc in cfg["accents"] for s in L2_SPEAKERS.get(acc, [])]
    else:
        targets = list(SPK_TO_ACCENT.keys())

    for spk in targets:
        # Search in standard structure: root/speaker/wav/*.wav
        wav_dir = root_path / spk / "wav"
        if not wav_dir.exists(): 
            wav_dir = root_path / spk # Fallback to direct folder
            
        wavs = sorted(list(wav_dir.glob("*.wav")))
        if cfg.get("max_files_per_speaker"): 
            wavs = wavs[:cfg["max_files_per_speaker"]]
            
        if wavs: files[spk] = [str(w) for w in wavs]
            
    return files

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    with open(args.config) as f: cfg = yaml.safe_load(f)
    if args.device: cfg["device"] = args.device
    cfg = ModelPrep.ensure_config(cfg)
    
    # Paths
    if cfg.get("auto_output_paths"):
        name = f"{cfg['device']}_gan_decoder_file{cfg['max_files_per_speaker']}_beam{cfg['beam_size']}_{cfg['vocoder_type']}"
        cfg["output_audio_dir"] = str(Path(cfg["base_output_dir"]) / name)
        json_path = str(Path(cfg["base_json_dir"]) / f"{name}.json")
    else:
        json_path = "results.json"

    # Run
    pipe = AccentConverter(cfg, device=cfg["device"])
    pipe.setup()
    
    dataset = get_dataset_files(cfg["dataset_root"], cfg)
    if not dataset: return
    
    profiler = BatchLatencyProfiler(cfg, pipe)
    profiler.profile_dataset(dataset)
    profiler.save_analysis(json_path)

if __name__ == "__main__":
    main()