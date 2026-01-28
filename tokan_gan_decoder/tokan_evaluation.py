"""
TokAN Evaluation Script (Following Original Paper Methodology)

This script evaluates accent conversion performance using the metrics from the paper:
1. WER (Word Error Rate): Measures intelligibility with native-only ASR
2. Speaker Similarity: Measures speaker identity preservation  
3. PPG Distance: Measures phonetic content preservation vs synthetic targets

Based on the original TokAN evaluation scripts.
"""

import os
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

import torch
import torchaudio
import numpy as np
from tqdm import tqdm

# For WER computation
import jiwer
from transformers import pipeline
from whisper.normalizers.english import EnglishTextNormalizer

# For Speaker Similarity
from resemblyzer import VoiceEncoder, preprocess_wav

# For PPG Distance
import ppgs
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# L2ARCTIC speaker configuration (from TokAN)
L2ARCTIC_SPEAKERS = {
    "arabic": ["ABA", "YBAA", "ZHAA", "SKA"],
    "chinese": ["BWC", "LXC", "NCC", "TXHC"],
    "hindi": ["ASI", "RRBI", "SVBI", "TNI"],
    "korean": ["HJK", "YDCK", "YKWK", "HKK"],
    "spanish": ["EBVS", "ERMS", "NJS", "MBMPS"],
    "vietnamese": ["HQTV", "PNV", "THV", "TLV"],
    "us": ["BDL", "RMS", "SLT", "CLB"]
}

# Create reverse mapping
SPEAKER_TO_ACCENT = {}
for accent, speakers in L2ARCTIC_SPEAKERS.items():
    for speaker in speakers:
        SPEAKER_TO_ACCENT[speaker] = accent


class AudioPairLoader:
    """Loads and pairs original and converted audio files."""
    
    def __init__(self, original_root: str, converted_root: str):
        self.original_root = Path(original_root)
        self.converted_root = Path(converted_root)
        
    def load_audio_pairs_with_text(
        self,
        speakers: Optional[List[str]] = None,
        max_files_per_speaker: Optional[int] = None
    ) -> List[Dict]:
        """
        Load pairs of (original, converted) audio files with text transcripts.
        
        Returns:
            List of dictionaries with keys: id, speaker, accent, src_audio, gen_audio, text
        """
        samples = []
        
        # Determine speakers to process
        if speakers is None:
            speakers = []
            for accent_speakers in L2ARCTIC_SPEAKERS.values():
                speakers.extend(accent_speakers)
        
        for speaker in speakers:
            # Check directories
            orig_speaker_dir = self.original_root / speaker / "wav"
            conv_speaker_dir = self.converted_root / speaker
            transcript_dir = self.original_root / speaker / "transcript"

            if not orig_speaker_dir.exists():
                logger.warning(f"Original directory not found: {orig_speaker_dir}")
                continue

            if not conv_speaker_dir.exists():
                logger.warning(f"Converted directory not found: {conv_speaker_dir}")
                continue

            # Load transcripts from individual files
            transcripts = {}
            if transcript_dir.exists():
                # Read all individual transcript files (e.g., arctic_a0001.txt)
                for transcript_file in transcript_dir.glob("arctic_*.txt"):
                    utt_id = transcript_file.stem  # e.g., "arctic_a0001"
                    with open(transcript_file, 'r') as f:
                        text = f.read().strip()
                        if text:
                            transcripts[utt_id] = text
                logger.info(f"Loaded {len(transcripts)} transcripts for speaker {speaker}")
            else:
                logger.warning(f"Transcript directory not found: {transcript_dir}")
            
            # Find matching pairs - iterate over converted files (fewer files)
            conv_files = sorted(conv_speaker_dir.glob("*.wav"))

            if max_files_per_speaker:
                conv_files = conv_files[:max_files_per_speaker]

            for conv_file in conv_files:
                # Extract utterance ID from converted filename
                # e.g., "arctic_a0001.wav" -> "arctic_a0001"
                utt_id = conv_file.stem  # e.g., "arctic_a0001"

                # Check if original file exists
                orig_file = orig_speaker_dir / f"{utt_id}.wav"

                if orig_file.exists():
                    # Get transcript
                    text = transcripts.get(utt_id, "")

                    if not text:
                        logger.warning(f"No transcript found for {utt_id}")
                        continue
                    
                    sample = {
                        "id": f"{speaker}_{utt_id}",
                        "speaker": speaker,
                        "accent": SPEAKER_TO_ACCENT.get(speaker, "unknown"),
                        "src_audio": str(orig_file),
                        "gen_audio": str(conv_file),
                        "text": text
                    }
                    samples.append(sample)
                else:
                    logger.warning(f"Converted file not found: {conv_file}")
            
            if samples:
                logger.info(f"Found {len([s for s in samples if s['speaker'] == speaker])} valid samples for speaker {speaker}")
        
        return samples


class WERComputer:
    """Computes Word Error Rate using ASR."""
    
    def __init__(self, model_tag: str = "facebook/s2t-medium-librispeech-asr"):
        self.model_tag = model_tag
        self._model = None
        self._normalizer = None
        
    def setup(self, device: str = "cuda"):
        """Load ASR model."""
        logger.info(f"Loading ASR model: {self.model_tag}")
        self._model = pipeline("automatic-speech-recognition", model=self.model_tag, device=device)
        self._normalizer = EnglishTextNormalizer()
        logger.info("ASR model loaded successfully")
    
    def recognize(self, audio_path: str) -> str:
        """Transcribe audio file."""
        if self._model is None:
            raise RuntimeError("Model not loaded. Call setup() first.")
        
        transcription = self._model(audio_path)["text"]
        return transcription
    
    def compute_wer(self, reference_texts: List[str], hypothesis_texts: List[str]) -> float:
        """Compute WER between reference and hypothesis texts."""
        # Normalize texts
        ref_normalized = [self._normalizer(t) for t in reference_texts]
        hyp_normalized = [self._normalizer(t) for t in hypothesis_texts]
        
        # Compute WER
        wer = jiwer.wer(ref_normalized, hyp_normalized)
        return wer


class SpeakerSimilarityComputer:
    """Computes speaker embedding similarity."""
    
    def __init__(self):
        self._encoder = None
        
    def setup(self, device: str = "cuda"):
        """Load speaker encoder."""
        logger.info("Loading speaker encoder (Resemblyzer)...")
        self._encoder = VoiceEncoder(device=device)
        logger.info("Speaker encoder loaded successfully")
    
    def compute_similarity(self, audio_path_1: str, audio_path_2: str) -> float:
        """Compute cosine similarity between two audio files."""
        if self._encoder is None:
            raise RuntimeError("Encoder not loaded. Call setup() first.")
        
        # Load and preprocess audio
        audio_1 = preprocess_wav(audio_path_1)
        audio_2 = preprocess_wav(audio_path_2)
        
        # Compute embeddings
        emb_1 = self._encoder.embed_utterance(audio_1)
        emb_2 = self._encoder.embed_utterance(audio_2)
        
        # Normalize embeddings
        emb_1 = emb_1 / np.linalg.norm(emb_1)
        emb_2 = emb_2 / np.linalg.norm(emb_2)
        
        # Compute similarity
        similarity = np.dot(emb_1, emb_2)
        
        return float(similarity)


class PPGDistanceComputer:
    """Computes Phonetic PosteriorGram distance."""
    
    def __init__(self):
        self.device_id = None
        
    def setup(self, device_id: Optional[int] = None):
        """Setup PPG extractor."""
        logger.info("Setting up PPG distance computer...")
        self.device_id = device_id
        logger.info("PPG distance computer ready")
    
    @torch.inference_mode()
    def compute_distance(self, target_audio_path: str, reference_audio_path: str) -> float:
        """
        Compute PPG distance between target and reference audio.
        Uses DTW alignment followed by normalized distance.
        """
        # Load audio
        ref_audio = ppgs.load.audio(reference_audio_path)
        tgt_audio = ppgs.load.audio(target_audio_path)
        
        # Extract PPGs
        ref_ppgs = ppgs.from_audio(ref_audio, ppgs.SAMPLE_RATE, gpu=self.device_id)[0].cpu().numpy().T  # (T1, dim)
        tgt_ppgs = ppgs.from_audio(tgt_audio, ppgs.SAMPLE_RATE, gpu=self.device_id)[0].cpu().numpy().T  # (T2, dim)
        
        # Align sequences using DTW
        distance, path = fastdtw(ref_ppgs, tgt_ppgs, dist=euclidean)
        
        # Warp sequences to align them
        aligned_ref = ref_ppgs[np.array([i for i, j in path])]
        aligned_tgt = tgt_ppgs[np.array([j for i, j in path])]
        
        # Convert back to tensors
        aligned_ref_tensor = torch.tensor(aligned_ref).to(self.device_id if self.device_id is not None else 'cpu')
        aligned_tgt_tensor = torch.tensor(aligned_tgt).to(self.device_id if self.device_id is not None else 'cpu')
        
        # Compute distance (transposed back to original shape)
        ppg_dist = ppgs.distance(aligned_tgt_tensor.T, aligned_ref_tensor.T, reduction="mean", normalize=True)
        
        return ppg_dist.item()


class EvaluationPipeline:
    """Main evaluation pipeline following TokAN paper methodology."""
    
    def __init__(self, device: str = "cuda", compute_ppg: bool = True):
        self.device = device
        self.compute_ppg_flag = compute_ppg
        
        # Initialize components
        self.wer_computer = WERComputer()
        self.similarity_computer = SpeakerSimilarityComputer()
        if compute_ppg:
            self.ppg_computer = PPGDistanceComputer()
        
    def setup(self):
        """Set up all evaluation components."""
        logger.info("Setting up evaluation pipeline...")
        self.wer_computer.setup(self.device)
        self.similarity_computer.setup(self.device)
        if self.compute_ppg_flag:
            device_id = 0 if self.device == "cuda" and torch.cuda.is_available() else None
            self.ppg_computer.setup(device_id)
        logger.info("Evaluation pipeline ready!")
    
    def evaluate_sample(self, sample: Dict) -> Dict:
        """
        Evaluate a single sample.
        
        Args:
            sample: Dict with keys: id, speaker, accent, src_audio, gen_audio, text
            
        Returns:
            Dict with evaluation metrics
        """
        result = {
            "id": sample["id"],
            "speaker": sample["speaker"],
            "accent": sample["accent"],
            "src_audio": sample["src_audio"],
            "gen_audio": sample["gen_audio"],
            "text": sample["text"],
        }
        
        try:
            # 1. Recognize converted speech for WER
            recognized_text = self.wer_computer.recognize(sample["gen_audio"])
            result["recognized_text"] = recognized_text
            
            # 2. Compute speaker similarity
            similarity = self.similarity_computer.compute_similarity(
                sample["src_audio"], 
                sample["gen_audio"]
            )
            result["speaker_similarity"] = similarity
            
            # 3. Compute PPG distance (if enabled and target audio provided)
            if self.compute_ppg_flag and "tgt_audio" in sample:
                ppg_dist = self.ppg_computer.compute_distance(
                    sample["gen_audio"],
                    sample["tgt_audio"]
                )
                result["ppg_distance"] = ppg_dist
            
            result["success"] = True
            
        except Exception as e:
            logger.error(f"Error evaluating sample {sample['id']}: {e}")
            result["success"] = False
            result["error"] = str(e)
        
        return result
    
    def evaluate_dataset(self, samples: List[Dict]) -> List[Dict]:
        """
        Evaluate entire dataset.
        
        Args:
            samples: List of sample dictionaries
            
        Returns:
            List of result dictionaries
        """
        results = []
        
        for sample in tqdm(samples, desc="Evaluating samples"):
            result = self.evaluate_sample(sample)
            results.append(result)
        
        return results


class ResultsAnalyzer:
    """Analyzes and summarizes evaluation results."""
    
    @staticmethod
    def compute_metrics(results: List[Dict]) -> Dict:
        """Compute aggregate metrics from results."""
        successful_results = [r for r in results if r.get("success", False)]
        
        if not successful_results:
            return {}
        
        metrics = {
            "n_samples": len(successful_results),
        }
        
        # 1. Word Error Rate (WER)
        reference_texts = [r["text"] for r in successful_results]
        recognized_texts = [r["recognized_text"] for r in successful_results]
        
        normalizer = EnglishTextNormalizer()
        ref_normalized = [normalizer(t) for t in reference_texts]
        hyp_normalized = [normalizer(t) for t in recognized_texts]
        
        overall_wer = jiwer.wer(ref_normalized, hyp_normalized)
        metrics["wer"] = overall_wer
        metrics["wer_pct"] = overall_wer * 100
        
        # 2. Speaker Similarity
        similarities = [r["speaker_similarity"] for r in successful_results]
        metrics["speaker_similarity_mean"] = np.mean(similarities)
        metrics["speaker_similarity_std"] = np.std(similarities)
        metrics["speaker_similarity_min"] = np.min(similarities)
        metrics["speaker_similarity_max"] = np.max(similarities)
        
        # 3. PPG Distance (if available)
        ppg_distances = [r["ppg_distance"] for r in successful_results if "ppg_distance" in r]
        if ppg_distances:
            metrics["ppg_distance_mean"] = np.mean(ppg_distances)
            metrics["ppg_distance_std"] = np.std(ppg_distances)
            metrics["ppg_distance_min"] = np.min(ppg_distances)
            metrics["ppg_distance_max"] = np.max(ppg_distances)
        
        return metrics
    
    @staticmethod
    def compute_grouped_metrics(results: List[Dict]) -> Dict:
        """Compute metrics grouped by speaker and accent."""
        successful_results = [r for r in results if r.get("success", False)]
        
        # Group by speaker
        by_speaker = defaultdict(list)
        by_accent = defaultdict(list)
        
        for r in successful_results:
            by_speaker[r["speaker"]].append(r)
            by_accent[r["accent"]].append(r)
        
        # Compute metrics for each group
        speaker_metrics = {
            speaker: ResultsAnalyzer.compute_metrics(results)
            for speaker, results in by_speaker.items()
        }
        
        accent_metrics = {
            accent: ResultsAnalyzer.compute_metrics(results)
            for accent, results in by_accent.items()
        }
        
        overall_metrics = ResultsAnalyzer.compute_metrics(successful_results)
        
        return {
            "overall": overall_metrics,
            "by_speaker": speaker_metrics,
            "by_accent": accent_metrics
        }
    
    @staticmethod
    def print_summary(metrics_dict: Dict):
        """Print formatted summary of results."""
        print("\n" + "="*80)
        print("TOKAN EVALUATION RESULTS (Following Paper Methodology)")
        print("="*80)
        
        overall = metrics_dict.get("overall", {})
        
        # Overall metrics
        print("\n📊 OVERALL METRICS")
        print("-" * 80)
        print(f"Total samples evaluated: {overall.get('n_samples', 0)}")
        
        if "wer_pct" in overall:
            print(f"\n✓ Word Error Rate (WER):")
            print(f"  WER: {overall['wer_pct']:.2f}%")
            print(f"  (Lower is better - measures intelligibility)")
        
        if "speaker_similarity_mean" in overall:
            print(f"\n✓ Speaker Similarity:")
            print(f"  Mean:   {overall['speaker_similarity_mean']:.4f}")
            print(f"  Std:    {overall['speaker_similarity_std']:.4f}")
            print(f"  Range:  [{overall['speaker_similarity_min']:.4f}, {overall['speaker_similarity_max']:.4f}]")
            print(f"  (Higher is better - measures speaker preservation)")
        
        if "ppg_distance_mean" in overall:
            print(f"\n✓ PPG Distance:")
            print(f"  Mean: {overall['ppg_distance_mean']:.4f}")
            print(f"  Std:  {overall['ppg_distance_std']:.4f}")
            print(f"  Range: [{overall['ppg_distance_min']:.4f}, {overall['ppg_distance_max']:.4f}]")
            print(f"  (Lower is better - measures phonetic content preservation)")
        
        # By accent breakdown
        by_accent = metrics_dict.get("by_accent", {})
        if by_accent:
            print("\n" + "="*80)
            print("📊 RESULTS BY ACCENT")
            print("="*80)
            
            for accent in sorted(by_accent.keys()):
                metrics = by_accent[accent]
                print(f"\n{accent.upper()}: {metrics['n_samples']} samples")
                if "wer_pct" in metrics:
                    print(f"  WER:  {metrics['wer_pct']:6.2f}%")
                if "speaker_similarity_mean" in metrics:
                    print(f"  SIM:  {metrics['speaker_similarity_mean']:6.4f} ± {metrics['speaker_similarity_std']:.4f}")
                if "ppg_distance_mean" in metrics:
                    print(f"  PPG:  {metrics['ppg_distance_mean']:6.4f} ± {metrics['ppg_distance_std']:.4f}")
        
        # By speaker summary (top 10)
        by_speaker = metrics_dict.get("by_speaker", {})
        if by_speaker:
            print("\n" + "="*80)
            print("📊 RESULTS BY SPEAKER (Top 10 by Sample Count)")
            print("="*80)
            
            sorted_speakers = sorted(
                by_speaker.items(),
                key=lambda x: x[1]['n_samples'],
                reverse=True
            )[:10]
            
            for speaker, metrics in sorted_speakers:
                accent = SPEAKER_TO_ACCENT.get(speaker, "unknown")
                print(f"\n{speaker} ({accent}): {metrics['n_samples']} samples")
                if "wer_pct" in metrics:
                    print(f"  WER:  {metrics['wer_pct']:6.2f}%")
                if "speaker_similarity_mean" in metrics:
                    print(f"  SIM:  {metrics['speaker_similarity_mean']:6.4f} ± {metrics['speaker_similarity_std']:.4f}")
                if "ppg_distance_mean" in metrics:
                    print(f"  PPG:  {metrics['ppg_distance_mean']:6.4f} ± {metrics['ppg_distance_std']:.4f}")
        
        print("\n" + "="*80)
    
    @staticmethod
    def save_results(results: List[Dict], metrics_dict: Dict, output_path: str):
        """Save detailed results to JSON file."""
        output = {
            "overall_metrics": metrics_dict.get("overall", {}),
            "by_accent_metrics": metrics_dict.get("by_accent", {}),
            "by_speaker_metrics": metrics_dict.get("by_speaker", {}),
            "detailed_results": results,
        }
        
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)
        
        logger.info(f"Results saved to {output_path}")


def get_parser():
    """Create argument parser."""
    parser = argparse.ArgumentParser(
        description="Evaluate TokAN accent conversion (following paper methodology)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data paths
    parser.add_argument(
        "--original_root",
        type=str,
        required=True,
        help="Root directory of original L2ARCTIC data"
    )
    parser.add_argument(
        "--converted_root",
        type=str,
        required=True,
        help="Root directory of converted audio"
    )
    parser.add_argument(
        "--synthetic_target_root",
        type=str,
        help="Root directory of synthetic target audio for PPG distance computation (optional)"
    )
    
    # Data selection
    parser.add_argument(
        "--speakers",
        type=str,
        nargs="+",
        help="Specific speakers to evaluate"
    )
    parser.add_argument(
        "--accents",
        type=str,
        nargs="+",
        choices=list(L2ARCTIC_SPEAKERS.keys()),
        help="Specific accents to evaluate"
    )
    parser.add_argument(
        "--max_files_per_speaker",
        type=int,
        help="Maximum number of files to evaluate per speaker"
    )
    
    # Evaluation options
    parser.add_argument(
        "--skip_ppg",
        action="store_true",
        help="Skip PPG distance computation (requires synthetic targets)"
    )
    
    # Output
    parser.add_argument(
        "--output_json",
        type=str,
        default="tokan_evaluation_results.json",
        help="Path to save evaluation results"
    )
    
    # Device
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to run evaluation on"
    )
    
    return parser


def main():
    """Main evaluation function."""
    parser = get_parser()
    args = parser.parse_args()
    
    # Set up device
    device = args.device if (args.device == "cuda" and torch.cuda.is_available()) else "cpu"
    logger.info(f"Using device: {device}")
    
    # Load audio pairs
    logger.info("Loading audio file pairs...")
    loader = AudioPairLoader(args.original_root, args.converted_root)
    
    # Determine speakers to evaluate
    speakers = args.speakers
    if not speakers and args.accents:
        speakers = []
        for accent in args.accents:
            speakers.extend(L2ARCTIC_SPEAKERS[accent])
    
    samples = loader.load_audio_pairs_with_text(
        speakers=speakers,
        max_files_per_speaker=args.max_files_per_speaker
    )
    
    if not samples:
        logger.error("No valid audio pairs found!")
        return
    
    # Add synthetic target paths if provided
    if args.synthetic_target_root and not args.skip_ppg:
        synthetic_root = Path(args.synthetic_target_root)
        for sample in samples:
            speaker = sample["speaker"]
            utt_id = sample["id"].split("_", 1)[1]  # Remove speaker prefix
            synthetic_path = synthetic_root / speaker / f"{utt_id}.wav"
            if synthetic_path.exists():
                sample["tgt_audio"] = str(synthetic_path)
            else:
                logger.warning(f"Synthetic target not found: {synthetic_path}")
    
    logger.info(f"Found {len(samples)} valid samples")
    
    # Set up evaluation pipeline
    compute_ppg = not args.skip_ppg and args.synthetic_target_root is not None
    pipeline = EvaluationPipeline(device=device, compute_ppg=compute_ppg)
    pipeline.setup()
    
    # Run evaluation
    logger.info("Starting evaluation...")
    results = pipeline.evaluate_dataset(samples)
    
    # Compute metrics
    logger.info("Computing metrics...")
    metrics_dict = ResultsAnalyzer.compute_grouped_metrics(results)
    
    # Print summary
    ResultsAnalyzer.print_summary(metrics_dict)
    
    # Save results
    ResultsAnalyzer.save_results(results, metrics_dict, args.output_json)
    
    logger.info("Evaluation complete!")


if __name__ == "__main__":
    main()