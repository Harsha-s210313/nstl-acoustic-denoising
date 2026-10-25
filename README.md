# NSTL Underwater Acoustic Signal Denoising Pipeline

A robust, high-performance PyTorch pipeline for denoising 1-D hydrophone recordings of Linear Frequency Modulated (LFM / chirp) acoustic pulses buried in ambient ocean noise.

Designed for air-gapped laboratory / shipboard deployment and edge porting to NVIDIA Jetson AGX Xavier.

---

## 1. Pipeline Architecture

```
Raw 1-D Hydrophone Recording (.dat)
           │
           ▼
   [Active Region Extraction & Normalization]
   O(N) Cumulative Sum RMS Envelope -> Noise Floor & Pulse Bounds
   Shared Mean/Std Normalization for Signal and Noise Reference
           │
           ▼
   [1-D U-Net Denoiser]
   - Fully Convolutional (Preserves arbitrary lengths: 25k, 5k, 4999)
   - Stage 2A: Supervised Pretraining (MSE on matched synthetic clean/noisy pairs)
   - [MANUAL VERIFICATION GATE]
   - Stage 2B: Unsupervised Fine-Tuning (Physical Energy Matching Loss on Real Trial Data)
           │
           ▼
   Outputs: Denoised .dat file + Time Series / PSD Diagnostic Plot
```

### Model
- **Configurable1DUNet** (`unet.py`):
  - Fully-convolutional encoder-decoder with double-convolution blocks (`Conv1d -> BN -> ReLU` x 2).
  - Linear interpolation upsampling with 1x1 projection, channel concatenation, and skip connections.
  - Exact length guard ensuring output length strictly matches input length regardless of pooling downsampling.

---

## 2. Directory Layout

```
nstl_acoustic/
├── config.py                 # Single source of truth (Config dataclass)
├── utils.py                  # Stateless I/O, O(N) RMS envelope, normalization
├── dataset.py                # PyTorch Datasets (Pretrain, NoiseAdaptation)
├── unet.py                   # Configurable1DUNet architecture
├── denoiser_finetune.py      # Stage 2A & 2B trainers and UnsupervisedNoiseSuppressionLoss
├── denoiser_pretrain.py      # Thin entry point for Stage 2A pretraining
├── generate_trial_data.py    # Synthetic chirp and physics-based trial .dat generator
├── inference.py              # Denoiser inference & diagnostic plot generator
├── run_pretrain.py           # Python IDLE F5 runner for Stage 2A
├── run_finetune.py           # Python IDLE F5 runner for Stage 2B
├── run_inference.py          # Python IDLE F5 runner for inference
├── test_pri_batch.py         # Batch runner across all trial recording channels
├── test_inference_headless.py# Headless verification of inference pipeline
├── verify_fresh_weights.py   # Quantitative matched-filter verification script
├── requirements.txt          # Pinned dependencies & offline install notes
├── README.md                 # Documentation
├── .gitignore                # Ignores data/, models/, inference_output/
└── tests/
    ├── conftest.py           # Pytest sys.path setup
    └── test_pipeline.py      # Comprehensive unit tests
```

### Data Layout
All acoustic data is stored as plain-text `.dat` files with **one float per line**. Blank lines and `#` comments are ignored; CRLF and LF line endings are supported.

- **Synthetic Data Directory** (`SYNTHETIC_DATA_DIR`):
  `C:\Users\HARSHA\.gemini\antigravity\scratch\nstl_acoustic\data`
  ```
  data/
  └── synthetic/
      ├── clean/                # Clean LFM chirps (ground truth)
      └── noisy/                # Clean chirps + multipath reverberation + colored noise
  ```

- **Trial Data Directory** (`TRIAL_DATA_DIR`):
  `C:\Users\HARSHA\Documents\pri`
  Contains hydrophone recordings directly:
  `channel0.dat`, `channel001.dat`, `channel2.dat`, ..., `channel10.dat`.

---

## 3. Offline / Air-Gapped Installation & Jetson Xavier Porting

The project requires minimal dependencies: `torch`, `numpy`, `scipy` (with `matplotlib` optional and loaded lazily).

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

### Phase 1: Generate Synthetic and Trial Data
Generate physics-based synthetic pairs and trial recordings:
```bash
python generate_trial_data.py --num-synthetic 32 --num-trial-pri 3 --channels-per-pri 4
```

### Phase 2: Stage 2A Supervised Pretraining
Pretrain the 1-D U-Net on matched clean/noisy synthetic pairs:
```bash
python denoiser_pretrain.py --epochs 60 --device cuda
```
*Note:* Displays a mandatory manual verification notice upon completion. Checkpoint saved to `models/unet_pretrained_best.pth`.

### Phase 3: MANUAL VERIFICATION GATE
Before proceeding to unsupervised trial fine-tuning, evaluate pretraining quality on synthetic data:
```bash
python inference.py --source synthetic --model pretrained
```

### Phase 4: Stage 2B Unsupervised Fine-Tuning
Fine-tune the model directly on real hydrophone trial recordings without clean labels:
```bash
python denoiser_finetune.py --stage finetune --epochs 25 --device cuda
```
- Verifies `models/unet_pretrained_best.pth` exists before running (raises `FileNotFoundError` otherwise).
- Displays a 3-second safety delay and reminder.
- Uses `UnsupervisedNoiseSuppressionLoss` against quiet recording segments.
- Saves to a separate checkpoint `models/unet_finetuned_best.pth` without overwriting the pretrained model.

### Phase 5: Run Inference on Trial Data
```bash
# Run on a specific channel and auto-open diagnostic plot
python inference.py --source trial --channel channel001.dat --show-plot

# Or run batch inference across all channels
python test_pri_batch.py
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
4. Open `run_inference.py` in IDLE and press **F5** to inspect trial outputs (the plot automatically pops up in the Windows Photos app).

No command-line arguments are required.

---

## 6. Verification and Testing

Run the full pytest suite:
```bash
pytest -v
```
