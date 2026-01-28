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
