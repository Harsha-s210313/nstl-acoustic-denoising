# NSTL Underwater Acoustic Signal Detection & Denoising Pipeline

A robust, two-stage PyTorch pipeline for detecting and denoising 1-D hydrophone recordings of Linear Frequency Modulated (LFM / chirp) acoustic pulses buried in ambient ocean noise.

Designed for air-gapped laboratory / shipboard deployment and edge porting to NVIDIA Jetson AGX Xavier.

---

## 1. Pipeline Architecture

```
Raw 1-D Hydrophone Recording (.dat)
           │
           ▼
   [Stage 1: Detector]
   1-D CNN (7-tap Conv1d, BatchNorm, MaxPool)
   Output: Pulse Present (Conf %) vs Ambient Noise
           │
           ▼
   [Active Region Extraction & Normalization]
   O(N) Cumulative Sum RMS Envelope -> Noise Floor & Pulse Bounds
   Shared Mean/Std Normalization for Signal and Noise Reference
           │
           ▼
   [Stage 2: 1-D U-Net Denoiser]
   - Fully Convolutional (Preserves arbitrary lengths: 25k, 5k, 4999)
   - Stage 2A: Supervised Pretraining (MSE on matched synthetic pairs)
   - [MANUAL VERIFICATION GATE]
   - Stage 2B: Unsupervised Fine-Tuning (Physical Energy Matching Loss on Real Trial Data)
           │
           ▼
   Outputs: Denoised .dat file + Time Series / PSD Diagnostic Plot
```

### Models
1. **SignalDetectorCNN** (`detector.py`):
   - Configurable channel widths (`[16, 32, 64]`), 7-sample kernels, batch normalization, ReLU, and 4x max-pooling per stage.
   - Global adaptive pooling to dense layer (128 units), 30% dropout, and 2-class logits.
2. **Configurable1DUNet** (`unet.py`):
   - Fully-convolutional encoder-decoder with double-convolution blocks (`Conv1d -> BN -> ReLU` x 2).
   - Linear interpolation upsampling with 1x1 projection, channel concatenation, and skip connections.
   - Exact length guard ensuring output length strictly matches input length regardless of pooling downsampling.

---

## 2. Directory Layout

```
nstl_acoustic/
├── config.py                 # Single source of truth (Config dataclass)
├── utils.py                  # Stateless I/O, O(N) RMS envelope, normalization
├── dataset.py                # PyTorch Datasets (Detector, Pretrain, NoiseAdaptation)
├── detector.py               # SignalDetectorCNN architecture
├── detector_train.py         # Supervised detector training script
├── unet.py                   # Configurable1DUNet architecture
├── denoiser_finetune.py      # Stage 2A & 2B trainers and UnsupervisedNoiseSuppressionLoss
├── denoiser_pretrain.py      # Thin entry point for Stage 2A pretraining
├── generate_trial_data.py    # Synthetic chirp and trial-like .dat generator
├── inference.py              # End-to-end detector + denoiser inference & plot generator
├── run_pretrain.py           # Python IDLE F5 runner for Stage 2A
├── run_finetune.py           # Python IDLE F5 runner for Stage 2B
├── run_inference.py          # Python IDLE F5 runner for inference
├── test_inference_headless.py# Headless verification of inference pipeline
├── requirements.txt          # Pinned dependencies & offline install notes
├── README.md                 # Complete documentation
├── .gitignore                # Ignores data/, models/, inference_output/
└── tests/
    ├── conftest.py           # Pytest sys.path setup
    └── test_pipeline.py      # Comprehensive unit tests
```

### Data Layout
All acoustic data is stored as plain-text `.dat` files with **one float per line**. Blank lines and `#` comments are ignored; CRLF and LF are supported.

- **Synthetic Data Directory** (`SYNTHETIC_DATA_DIR`):
  `C:\Users\HARSHA\.gemini\antigravity\scratch\nstl_acoustic\data`
  ```
  data/
  ├── detector/
  │   ├── signal/               # Noisy chirp files (label 1)
  │   └── noise/                # Pure ambient noise files (label 0)
  └── synthetic/
      ├── clean/                # Clean LFM chirps (ground truth)
      └── noisy/                # Clean chirps + Gaussian/colored noise
  ```

- **Trial Data Directory** (`TRIAL_DATA_DIR`):
  `C:\Users\HARSHA\Documents\pri_generated`
  Contains 100 hydrophone recording files directly:
  `channel0.dat`, `channel001.dat`, `channel10.dat`, ..., `channel99.dat`.

---

## 3. Offline / Air-Gapped Installation & Jetson Xavier Porting

The project has minimal dependencies: `torch`, `numpy`, `scipy` (with `matplotlib` and `tqdm` optional and loaded lazily).

### Step 1: Build the Offline Wheel Bundle (Internet-Connected Machine)
On a machine with internet access matching the target architecture:
```bash
pip download -r requirements.txt -d ./wheelhouse
```
Copy the `nstl_acoustic/` directory and the `wheelhouse/` folder onto CD/DVD or approved USB media.

### Step 2: Install Offline (Air-Gapped Machine / Jetson Xavier)
On the target air-gapped machine:
```bash
pip install --no-index --find-links=./wheelhouse -r requirements.txt
```
No external network connections or runtime downloads are made by any script.

---

## 4. Execution Workflow (Strict Run Order)

### Phase 1: Generate Data
Generate synthetic 100 kHz pairs and trial 17.8 kHz bandpass recordings with realistic artefacts (clipping, CRLF, comments):
```bash
python generate_trial_data.py --artefacts
```

### Phase 2: Train Stage 1 Detector
```bash
python detector_train.py --epochs 20 --batch-size 8
```
Saves best model to `models/detector_best.pth`.

### Phase 3: Stage 2A Supervised Pretraining
Pretrain the 1-D U-Net on matched clean/noisy synthetic pairs:
```bash
python denoiser_pretrain.py --epochs 25 --batch-size 4
```
*Note:* Displays a mandatory manual verification notice upon completion. Checkpoint saved to `models/unet_pretrained_best.pth`.

### Phase 4: MANUAL VERIFICATION GATE
Before proceeding to unsupervised trial fine-tuning, evaluate pretraining quality on synthetic data:
```bash
python inference.py --source synthetic --model pretrained
```
Inspect:
1. Ground-truth SNR improvement in console (typically `+8 dB` to `+14 dB`).
2. Time-series and Welch PSD plot saved in `inference_output/`.
3. Confirm residual noise reduction without pulse waveform distortion.

### Phase 5: Stage 2B Unsupervised Fine-Tuning
Fine-tune the model directly on real hydrophone trial recordings without clean labels:
```bash
python denoiser_finetune.py --stage finetune --epochs 10 --batch-size 4
```
- Verifies `models/unet_pretrained_best.pth` exists before running (raises `FileNotFoundError` otherwise).
- Displays a 3-second safety delay and reminder.
- Uses `UnsupervisedNoiseSuppressionLoss` against quiet recording segments.
- Saves to a separate checkpoint `models/unet_finetuned_best.pth` without overwriting the pretrained model.

### Phase 6: Run Inference on Trial Data
```bash
python inference.py --source trial --pri PRI_01 --channel channel001.dat --model finetuned
```
Generates:
- Denoised 1-D plain-text `.dat` file in `inference_output/channel001_denoised.dat`.
- Headless diagnostic plot in `inference_output/channel001_analysis.png`.

---

## 5. Python IDLE F5 Workflow (Windows)

For operators running directly inside Python IDLE:
1. Open `run_pretrain.py` in IDLE and press **F5** to pretrain.
2. Open `run_inference.py` in IDLE and press **F5** to verify results.
3. Open `run_finetune.py` in IDLE and press **F5** to run trial fine-tuning.
4. Open `run_inference.py` in IDLE and press **F5** to inspect trial outputs.

No command-line arguments are required.

---

## 6. Verification and Testing

Run the full pytest suite:
```bash
pytest -v
```

Verified test coverage includes:
- `test_load_dat_file_edge_cases`: Robust handling of comments, CRLF, truncation, and padding.
- `test_rms_envelope_vs_brute_force`: Mathematical equality between $O(N)$ cumulative sum envelope and brute force.
- `test_active_region_detection_accuracy`: Known pulse location recovery within tolerance bounds.
- `test_unet_output_shapes`: Strict shape preservation for $L \in \{5000, 25000, 4999\}$.
- `test_unsupervised_loss_not_minimized_by_identity_or_zeros`: Proof that loss penalizes both identity and all-zeros silence.
- `test_config_key_coverage`: Grep-based validation confirming all `cfg.*` references exist in `Config`.
- `test_finetuner_refuses_without_pretrained_checkpoint`: Safety gate verification.
- `test_finetuner_never_overwrites_pretrained_checkpoint`: Non-destructive fine-tuning verification.
- `test_headless_inference_end_to_end`: Display-free execution check.
