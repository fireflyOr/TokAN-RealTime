# Accelerating TokAN for Real-Time Accent Conversion on CPU

**Authors:** Or Davidovich & Chenxi Liu  
**Date:** January 28, 2026

This repository contains the official implementation of the final project: **"Accelerating TokAN for Real-Time Accent Conversion on CPU"**.

We present a redesigned inference pipeline for **TokAN** (originally presented at **Interspeech 2025**) that achieves **real-time performance** on standard CPUs while maintaining high speech quality. This is achieved by replacing the iterative CFM synthesizer with a single-pass GAN decoder and integrating a lightweight HiFi-GAN vocoder.

---

## 🚀 Key Improvements

Based on our bottleneck analysis (see `scripts/tokan_profiler.py`), we introduced three major modifications (Mod. 4):

1.  **Single-Pass GAN Decoder:** Replaced the iterative Diffusion/CFM decoder (32 steps) with a feed-forward GAN decoder (1 step), trained via distillation.
2.  **HiFi-GAN Vocoder:** Replaced the heavy BigVGAN with HiFi-GAN, reducing vocoding time by **50x** (16s $\to$ 0.3s).
3.  **Greedy Decoding:** Optimized token conversion by switching from Beam Search ($k=10$) to Greedy Decoding ($k=1$), yielding a **3.2x** speedup.

---

## 📊 Performance Results

Tested on the **L2-ARCTIC** dataset (28 speakers) using a standard CPU.

| Configuration | RTF (CPU) $\downarrow$ | Speedup | WER (%) | Speaker Similarity $\uparrow$ |
| :--- | :--- | :--- | :--- | :--- |
| **Original TokAN** | $11.43 \pm 1.71$ | 1.0x | 15.88% | $0.874$ |
| **Our Model (Mod. 4)** | **$1.76 \pm 0.16$** | **~6.5x** | **16.10%** | **$0.838$** |

---

## 🛠️ Installation

### Prerequisites
* Python 3.8+
* `espeak` (System dependency)

```bash
# Ubuntu/Debian
sudo apt-get install espeak espeak-data
```
1. Setup Environment
```Bash
git clone --recurse-submodules [https://github.com/YourRepo/TokAN-RealTime.git](https://github.com/YourRepo/TokAN-RealTime.git)
cd TokAN-RealTime
pip install -r requirements.txt
```
2. Install Fairseq (Crucial)
You must install the specific fairseq version provided in the submodule.

```Bash
cd third_party/fairseq
pip install -e .
cd ../..
```
3. Download Pre-trained Models
Due to file size limits, the models are hosted on Hugging Face.

Step A: Automatic Download Run the following script to download the base models (HuBERT, etc.):

```Bash
python tokan/utils/model_utils.py
```
Step B: Manual Download from Hugging Face Download the specific checkpoints from our Model Hub:

Link: https://huggingface.co/OrDavidovich/TokAN-RealTime

Required Files to Download:

model.pt

dict.src.txt

dict.tgt.txt

dict.aux.txt

best_gan.pt (Our trained GAN checkpoint)

Placement Instructions: Place the files exactly as shown below:

```Plaintext
pretrained_models/
├── token_to_token/
│   └── tokan-t2t-base-paper/   <-- (Create this folder manually)
│       ├── model.pt
│       ├── dict.src.txt
│       ├── dict.tgt.txt
│       └── dict.aux.txt
└── checkpoints/
    └── best_gan.pt             <-- (Place GAN model here)
```
📂 Data Preparation
Note: The dataset audio files (.wav) are not included in this repository.

1. Download Datasets
L2ARCTIC contains non-native English speech, while ARCTIC provides native English speakers for comparison. The combined dataset should include:

**L2ARCTIC Speakers (by accent):**
- Arabic (`<ar>`): ABA, YBAA, ZHAA, SKA
- Chinese (`<zh>`): BWC, LXC, NCC, TXHC  
- Hindi (`<hi>`): ASI, RRBI, SVBI, TNI
- Korean (`<ko>`): HJK, YDCK, YKWK, HKK
- Spanish (`<es>`): EBVS, ERMS, NJS, MBMPS
- Vietnamese (`<vi>`): HQTV, PNV, THV, TLV

**ARCTIC Native Speakers (`<us>`):** BDL, RMS, SLT, CLB

**Expected Directory Structure:**
```
l2arctic/
├ # L2ARCTIC speakers (direct extraction)
├── YBAA/
├── BWC/
├── ...
├ # ARCTIC native speakers
├── BDL/
├── RMS/
├── SLT/
└── CLB/
```

**Download Instructions:**

*For L2ARCTIC:*
1. Visit: https://psi.engr.tamu.edu/l2-arctic-corpus/
2. Download and extract to your L2ARCTIC root directory

*For ARCTIC Native Speakers:*
1. Visit: http://festvox.org/cmu_arctic/
2. Download: `cmu_us_bdl_arctic.tar.bz2`, `cmu_us_rms_arctic.tar.bz2`, `cmu_us_slt_arctic.tar.bz2`, `cmu_us_clb_arctic.tar.bz2`
3. Extract to L2ARCTIC's directory with the same meaning pattern (speaker's tag in the upper case)


2. Prepare Targets (Distillation)
We use the original TokAN model as a "Teacher" to generate training targets.

```Bash
python tokan_gan_decoder/data/dataset.py \
    --data_dir /path/to/L2Arctic \
    --output_dir ./gan_targets \
    --tokan_checkpoint ./pretrained_models/token_to_mel/tokan-t2m-v1-paper/model.ckpt \
    --cfm_timesteps 32
```
3. Verify Splits
Ensure held-out speakers (EBVS, SKA) are isolated.

```Bash
python tokan_gan_decoder/prepare_gan_splits.py --data_dir ./gan_targets
```
🏃 Usage
Training
Train the GAN decoder using PyTorch Lightning.

```Bash
# Run on GPU
CUDA_VISIBLE_DEVICES="0" python tokan_gan_decoder/training/trainer.py \
    --config tokan_gan_decoder/training/config.yaml
```
Real-Time Inference
You can run inference using the main script which integrates the GAN decoder.

```Bash
python inference.py \
    --input_path input.wav \
    --output_path output.wav \
    --use_gan \
    --gan_checkpoint checkpoints/best_gan.pt
```
Alternatively, use the FastMelSynthesizer wrapper in Python:

```Python
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
📁 File Structure
```Plaintext
TokAN-RealTime/
├── components/                 # Original TokAN modules
├── third_party/                # Fairseq submodule
├── scripts/                    # Profiling and evaluation tools
├── pretrained_models/          # Model checkpoints (Download required)
│   └── token_to_token/
│       └── tokan-t2t-base-paper/  # Place manual downloads here
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
Acknowledgements
This project builds upon the official TokAN implementation (Interspeech 2025).
