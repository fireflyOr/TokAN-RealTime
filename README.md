# Accelerating TokAN for Real-Time Accent Conversion on CPU

**Authors:** Or Davidovich & Chenxi Liu  
**Date:** January 28, 2026

This repository contains the official implementation of the final project: **"Accelerating TokAN for Real-Time Accent Conversion on CPU"**.

We present a redesigned inference pipeline for TokAN that achieves **real-time performance** on standard CPUs while maintaining high speech quality. This is achieved by replacing the iterative CFM synthesizer with a single-pass GAN decoder and integrating a lightweight HiFi-GAN vocoder.


## 🚀 Key Improvements

Based on our bottleneck analysis (see `scripts/tokan_profiler.py`), we introduced three major modifications (Mod. 4):

1.  **Single-Pass GAN Decoder:** Replaced the iterative Diffusion/CFM decoder (32 steps) with a feed-forward GAN decoder (1 step), trained via distillation from the original model.
2.  **HiFi-GAN Vocoder:** Replaced the heavy BigVGAN with HiFi-GAN, reducing vocoding time by **50x** (16s $\to$ 0.3s).
3.  **Greedy Decoding:** Optimized token conversion by switching from Beam Search ($k=10$) to Greedy Decoding ($k=1$), yielding a **3.2x** speedup in that module.

---

## 📊 Performance Results

Tested on the **L2-ARCTIC** dataset (28 speakers) using a standard CPU.

| Configuration | RTF (CPU) $\downarrow$ | Speedup | WER (%) | Speaker Similarity $\uparrow$ |
| :--- | :--- | :--- | :--- | :--- |
| **Original TokAN** | $11.43 \pm 1.71$ | 1.0x | 15.88% | $0.874$ |
| **Our Model (Mod. 4)** | **$1.76 \pm 0.16$** | **~6.5x** | **16.10%** | **$0.838$** |

*Results show that our GAN-based distillation effectively compresses the model while maintaining perceptual quality.*

---

## 🏗️ Architecture

The pipeline consists of a frozen Token-to-Mel encoder and a trainable GAN Decoder.

### The GAN Decoder
* **Input:** Aligned encoder features ($h_y$) + Speaker Embeddings.
* **Generator:** 9-layer Residual Convolutional Stack (ResStack) with dilated convolutions to capture prosodic context.
* **Discriminator:** Unified Multi-Period (MPD) and Multi-Scale (MSD) discriminators.
* **Losses:** L1 Mel Reconstruction, Multi-Resolution Loss, Adversarial Loss (LSGAN), and Feature Matching.

---

## 🛠️ Installation

### Prerequisites
* Python 3.8+
* `espeak` (System dependency for phonemization)

```bash
# Ubuntu/Debian
sudo apt-get install espeak espeak-data

```

### 1. Setup Environment

```bash
git clone --recurse-submodules [https://github.com/YourRepo/TokAN-RealTime.git](https://github.com/YourRepo/TokAN-RealTime.git)
cd TokAN-RealTime
pip install -r requirements.txt

```

### 2. Install Fairseq (Crucial)

You must install the specific fairseq version provided in the submodule.

```bash
cd third_party/fairseq
pip install -e .
cd ../..

```

---

## 🏃 Usage

### 1. Data Preparation (Distillation)

We use the original TokAN model as a "Teacher" to generate training targets (Mel spectrograms and hidden features).

```bash
python tokan_gan_decoder/data/dataset.py \
    --data_dir /path/to/L2Arctic \
    --output_dir ./gan_targets \
    --tokan_checkpoint ./pretrained_models/tokan.ckpt \
    --cfm_timesteps 32

```

### 2. Verify Splits

Ensure held-out speakers (EBVS, SKA) are isolated.

```bash
python tokan_gan_decoder/prepare_gan_splits.py --data_dir ./gan_targets

```

### 3. Training

Train the GAN decoder using PyTorch Lightning.

```bash
# Run on GPU
CUDA_VISIBLE_DEVICES="0" python tokan_gan_decoder/training/trainer.py \
    --config tokan_gan_decoder/training/config.yaml

```

### 4. Real-Time Inference

You can run inference using the main script which integrates the GAN decoder.

```bash
python inference.py \
    --input_path input.wav \
    --output_path output.wav \
    --use_gan \
    --gan_checkpoint checkpoints/best_gan.pt

```

Alternatively, use the `FastMelSynthesizer` wrapper in Python:

```python
from tokan_gan_decoder.integration.fast_synthesizer import FastMelSynthesizer

# Load optimized components
synth = FastMelSynthesizer(
    gan_decoder_path="checkpoints/best_gan.pt",
    vocoder_type="hifigan", 
    device="cpu"
)

# Synthesize (Single forward pass)
audio = synth.synthesize(tokens, speaker_embedding)

```

---

## 📁 File Structure

```text
TokAN-RealTime/
├── components/                 # Original TokAN modules
├── third_party/                # Fairseq submodule
├── scripts/                    # Profiling and evaluation tools
│   ├── tokan_profiler.py       # Bottleneck analysis script
│   └── tokan_evaluation.py     # WER/Similarity metrics
├── tokan_gan_decoder/          # === New GAN Implementation ===
│   ├── prepare_gan_splits.py   # Split verification script
│   ├── data/
│   │   ├── dataset.py          # Distillation & Preprocessing
│   │   └── gan_decoder_datamodule.py
│   ├── models/                 # Generator & Discriminator
│   ├── training/               # Trainer & Config
│   └── integration/            # FastSynthesizer wrapper
├── inference.py                # Main inference entry point
├── requirements.txt
└── README.md

```

## Acknowledgements

This project builds upon the official [TokAN implementation](https://github.com/P1ping/TokAN).

```

```