# NSTL Underwater Acoustic Denoising Pipeline

A two-stage ML pipeline for underwater acoustic signal processing, built as part of an NSTL/DRDO internship project using PyTorch.

---

## Project Structure

```
nstl_acoustic/
├── config.py              # Central configuration (edit paths here)
├── utils.py               # Shared utilities (.dat loader, normalisation, etc.)
├── dataset.py             # Dataset classes for all stages
├── detector.py            # Stage 1: SignalDetectorCNN
├── detector_train.py      # Stage 1: Training pipeline
├── unet.py                # Stage 2: Configurable1DUNet
├── denoiser_pretrain.py   # Stage 2A: Entry point for synthetic pretraining
├── denoiser_finetune.py   # Stage 2A+2B: Canonical training classes
├── inference.py           # Inference with plot output
├── run_pretrain.py        # IDLE-friendly: open and press F5 to pretrain
├── run_finetune.py        # IDLE-friendly: open and press F5 to fine-tune
├── run_inference.py       # IDLE-friendly: open and press F5 to run inference
└── requirements.txt       # Required Python libraries
```

---

## Pipeline Overview

```
Stage 1 — Signal Detection (SignalDetectorCNN)
        ↓
Stage 2A — Synthetic Pretraining (Configurable1DUNet)
        ↓
  Manual Verification
        ↓
Stage 2B — Trial Fine-Tuning (unsupervised)
        ↓
Final Inference (with plot output)
```

---

## Requirements

```
torch
numpy
scipy
matplotlib   # optional, for inference plots
tqdm         # optional
```

---

## Setup

**1. Clone the repo**
```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd nstl_acoustic
```

**2. Edit `config.py`** — set your data paths:
```python
SYNTHETIC_CLEAN_SUBDIR: str = r"D:\your\path\to\clean"
SYNTHETIC_NOISY_SUBDIR: str = r"D:\your\path\to\noisy"
TRIAL_SIGNAL_SUBDIR:    str = r"D:\your\path\to\Signal"
TRIAL_NOISE_SUBDIR:     str = r"D:\your\path\to\Noise"
```

---

## Running (Terminal)

```bash
# Stage 2A — pretrain on synthetic data
python denoiser_pretrain.py

# Verify the model
python inference.py --source synthetic

# Stage 2B — fine-tune on trial data (only after manual verification)
python denoiser_finetune.py --stage finetune

# Final inference
python inference.py --source trial --pri pri01 --channel 0
```

## Running (Python IDLE)

| Task | File to open | Action |
|---|---|---|
| Pretrain | `run_pretrain.py` | Press F5 |
| Fine-tune | `run_finetune.py` | Press F5 |
| Inference | `run_inference.py` | Press F5 |

---

## Data Layout

### Synthetic data
```
<SYNTHETIC_DATA_DIR>/
    clean/   sig_001.dat  sig_002.dat  ...
    noisy/   sig_001.dat  sig_002.dat  ...   (matched by filename)
```

### Trial data
```
<TRIAL_DATA_DIR>/
    Signal/
        PRI_01/   channel001.dat  channel002.dat  ...
        PRI_02/   ...
    Noise/
        PRI_11/   channel001.dat  ...
```

---

*NSTL/DRDO Internship Project*
