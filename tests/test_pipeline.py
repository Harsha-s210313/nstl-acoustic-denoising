"""
Comprehensive Unit Test Suite for NSTL Acoustic Detection and Denoising Pipeline.
Verifies all Section 11 acceptance requirements.
"""

import os
import re
import glob
import tempfile
import pytest
import numpy as np
import torch

try:
    from nstl_acoustic.config import Config
    from nstl_acoustic.utils import (
        load_dat_file,
        save_dat_file,
        compute_rms_envelope,
        detect_active_region,
        extract_active_region,
        normalize_signal,
        denormalize_signal,
        to_tensor,
    )
    from nstl_acoustic.unet import Configurable1DUNet
    from nstl_acoustic.detector import SignalDetectorCNN
    from nstl_acoustic.denoiser_finetune import (
        UnsupervisedNoiseSuppressionLoss,
        TrialDataFineTuner,
    )
except ModuleNotFoundError:
    from config import Config
    from utils import (
        load_dat_file,
        save_dat_file,
        compute_rms_envelope,
        detect_active_region,
        extract_active_region,
        normalize_signal,
        denormalize_signal,
        to_tensor,
    )
    from unet import Configurable1DUNet
    from detector import SignalDetectorCNN
    from denoiser_finetune import (
        UnsupervisedNoiseSuppressionLoss,
        TrialDataFineTuner,
    )


def test_load_dat_file_edge_cases():
    """
    Test load_dat_file robustness:
    - blank lines and comments (#)
    - CRLF (\\r\\n) and LF (\\n) line endings
    - parse errors / non-numeric tokens
    - zero-padding on short files
    - truncation on long files
    - exception on empty / invalid files
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        # 1. Normal file with CRLF, comments, and blank lines
        test_file_1 = os.path.join(tmp_dir, "test1.dat")
        with open(test_file_1, "wb") as f:
            content = (
                b"# Header comment\r\n"
                b"\r\n"
                b"1.5\r\n"
                b"-2.25\r\n"
                b"# Inline comment\r\n"
                b"3.125\r\n"
                b"INVALID_FLOAT_TOKEN\r\n"
                b"4.0\n"
            )
            f.write(content)

        arr = load_dat_file(test_file_1)
        assert len(arr) == 4
        assert np.allclose(arr, [1.5, -2.25, 3.125, 4.0])

        # 2. Expected length truncation
        arr_trunc = load_dat_file(test_file_1, expected_length=2)
        assert len(arr_trunc) == 2
        assert np.allclose(arr_trunc, [1.5, -2.25])

        # 3. Expected length zero-padding
        arr_pad = load_dat_file(test_file_1, expected_length=6)
        assert len(arr_pad) == 6
        assert np.allclose(arr_pad, [1.5, -2.25, 3.125, 4.0, 0.0, 0.0])

        # 4. Empty / invalid file raises ValueError
        empty_file = os.path.join(tmp_dir, "empty.dat")
        with open(empty_file, "w") as f:
            f.write("# Only comments\n\n")

        with pytest.raises(ValueError):
            load_dat_file(empty_file)


def test_rms_envelope_vs_brute_force():
    """
    Verify O(N) cumulative sum RMS envelope matches brute force O(N*W) calculation.
    """
    np.random.seed(42)
    n = 2000
    window = 101  # Odd window
    signal = np.random.randn(n).astype(np.float64)

    fast_rms = compute_rms_envelope(signal, window=window)
    assert len(fast_rms) == n

    # Brute force moving RMS with edge padding
    pad_left = window // 2
    pad_right = window - 1 - pad_left
    sq_padded = np.pad(signal ** 2, (pad_left, pad_right), mode="edge")

    brute_rms = np.zeros(n, dtype=np.float64)
    for i in range(n):
        brute_rms[i] = np.sqrt(np.mean(sq_padded[i : i + window]))

    assert np.allclose(fast_rms, brute_rms, atol=1e-5)


def test_active_region_detection_accuracy():
    """
    Verify that detect_active_region identifies a known pulse injected into noise.
    """
    np.random.seed(42)
    sig_len = 19299
    noise = np.random.normal(0.0, 0.05, size=sig_len).astype(np.float32)

    pulse_start = 7000
    pulse_end = 9011
    pulse = 1.5 * np.sin(np.linspace(0, 50 * np.pi, pulse_end - pulse_start))
    signal = noise.copy()
    signal[pulse_start:pulse_end] += pulse

    cfg = Config()
    rms = compute_rms_envelope(signal, cfg.TRIAL_RMS_WINDOW)
    noise_floor = float(np.percentile(rms, cfg.EXTRACTION_NOISE_PERCENTILE))

    start, end = detect_active_region(
        rms=rms,
        noise_floor=noise_floor,
        threshold_factor=cfg.EXTRACTION_THRESHOLD_FACTOR,
        pad_samples=cfg.TRIAL_PAD_SAMPLES,
        signal_length=sig_len,
    )

    # Active bounds must encompass pulse within pad tolerance
    assert start <= pulse_start
    assert end >= pulse_end
    # Ensure bounds did not blindly return the entire signal
    assert (end - start) < (pulse_end - pulse_start + 2 * cfg.TRIAL_PAD_SAMPLES + 500)


def test_unet_output_shapes():
    """
    Verify U-Net preserves length for L in {5000, 25000, 4999}.
    """
    cfg = Config()
    model = Configurable1DUNet(cfg=cfg)
    model.eval()

    test_lengths = [5000, 25000, 4999]
    with torch.no_grad():
        for length in test_lengths:
            x = torch.randn(2, 1, length)
            out = model(x)
            assert out.shape == (2, 1, length), f"Shape mismatch for L={length}: got {out.shape}"


def test_unsupervised_loss_not_minimized_by_identity_or_zeros():
    """
    Verify UnsupervisedNoiseSuppressionLoss properties:
    - Loss is NOT minimized by identity (denoised == noisy)
    - Loss is NOT minimized by all-zeros silence (denoised == 0)
    - Clean signal achieves lower loss than identity and silence
    """
    loss_fn = UnsupervisedNoiseSuppressionLoss(lam=0.7)

    # Synthetic signal and noise
    t = torch.linspace(0, 1, 5000)
    clean = torch.sin(2 * np.pi * 10 * t).unsqueeze(0).unsqueeze(0)  # (1, 1, 5000)
    noise = (0.3 * torch.randn(1, 1, 5000)).float()
    noise_ref = (0.3 * torch.randn(1, 1, 356)).float()
    noisy = clean + noise

    loss_clean = loss_fn(clean, noisy, noise_ref).item()
    loss_identity = loss_fn(noisy, noisy, noise_ref).item()
    loss_silence = loss_fn(torch.zeros_like(noisy), noisy, noise_ref).item()

    # Neither identity nor silence should achieve optimal loss
    assert loss_identity > 0.0, "Identity loss must be non-zero"
    assert loss_silence > 0.0, "Silence loss must be non-zero"
    assert loss_clean < loss_identity, f"Clean loss ({loss_clean}) should be lower than identity ({loss_identity})"
    assert loss_clean < loss_silence, f"Clean loss ({loss_clean}) should be lower than silence ({loss_silence})"


def test_config_key_coverage():
    """
    Grep-based unit test ensuring every cfg.<ATTR> or Config.<ATTR>
    referenced across the codebase exists in Config.
    """
    pkg_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    py_files = glob.glob(os.path.join(pkg_dir, "*.py"))

    config_instance = Config()
    config_keys = set(dir(config_instance))

    pattern = re.compile(r"(?:cfg|Config)\.([A-Z0-9_]+)")
    referenced_keys = set()

    for file_path in py_files:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                matches = pattern.findall(line)
                for m in matches:
                    referenced_keys.add(m)

    missing_keys = referenced_keys - config_keys
    assert not missing_keys, f"Found attributes referenced in code but missing in Config: {missing_keys}"


def test_finetuner_refuses_without_pretrained_checkpoint():
    """
    Verify Stage 2B TrialDataFineTuner refuses to run if pretrained checkpoint is missing.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        non_existent_ckpt = os.path.join(tmp_dir, "missing_pretrained.pth")
        cfg = Config(UNET_PRETRAINED_CHECKPOINT=non_existent_ckpt)

        with pytest.raises(FileNotFoundError) as exc_info:
            TrialDataFineTuner(cfg=cfg)

        assert "Stage 2A pretrained checkpoint not found" in str(exc_info.value)


def test_finetuner_never_overwrites_pretrained_checkpoint():
    """
    Verify Stage 2B never overwrites the Stage 2A checkpoint.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        pre_ckpt = os.path.join(tmp_dir, "unet_pre.pth")
        fine_ckpt = os.path.join(tmp_dir, "unet_fine.pth")

        cfg = Config(
            UNET_PRETRAINED_CHECKPOINT=pre_ckpt,
            UNET_FINETUNED_CHECKPOINT=fine_ckpt,
            TRIAL_DATA_DIR=os.path.join(tmp_dir, "trial"),
        )

        # Create dummy pretrained checkpoint
        model = Configurable1DUNet(cfg=cfg)
        torch.save({
            "tag": "pretrained",
            "epoch": 1,
            "val_loss": 0.1234,
            "model_state_dict": model.state_dict(),
            "unet_config": cfg.UNET_CONFIG,
            "signal_length": cfg.SIGNAL_LENGTH,
        }, pre_ckpt)

        original_bytes = open(pre_ckpt, "rb").read()

        # Instantiate FineTuner
        finetuner = TrialDataFineTuner(cfg=cfg)
        assert finetuner.cfg.UNET_FINETUNED_CHECKPOINT != finetuner.cfg.UNET_PRETRAINED_CHECKPOINT

        # Check that original pretrained checkpoint is completely intact
        after_bytes = open(pre_ckpt, "rb").read()
        assert original_bytes == after_bytes, "Pretrained checkpoint was modified!"


def test_correlation_zero_shift():
    """
    Verify that extract_active_region preserves exact time alignment (0 sample shift)
    when correlated against a clean LFM replica.
    If a trained checkpoint is available, verifies end-to-end U-Net zero shift as well.
    """
    fs = 17800.0
    dur = 0.113
    pulse_len = int(round(dur * fs))  # 2011 samples
    t = np.arange(pulse_len, dtype=np.float32) / fs
    f1, f2 = 3200.0, 5700.0
    k = (f2 - f1) / (pulse_len / fs)
    replica = (np.sin(2.0 * np.pi * (f1 * t + 0.5 * k * (t ** 2))) * np.hanning(pulse_len)).astype(np.float32)

    cfg = Config()

    # 1. 5000-sample segment with pulse at sample 750
    np.random.seed(42)
    sig_5k = (0.05 * np.random.randn(5000)).astype(np.float32)
    sig_5k[750 : 750 + pulse_len] += replica

    ext = extract_active_region(sig_5k, cfg, target_length=5000)
    assert ext["start"] == 0, f"Expected start=0 for pre-segmented signal, got {ext['start']}"
    assert ext["end"] == 5000

    c_raw = np.correlate(sig_5k, replica, mode="full")
    c_ext = np.correlate(ext["active"], replica, mode="full")
    shift_ext = int(np.argmax(c_raw)) - int(np.argmax(c_ext))
    assert shift_ext == 0, f"Expected 0 shift in extracted active signal, got {shift_ext}"

    # 2. If trained checkpoint exists, verify end-to-end model has 0 shift
    ckpt_path = cfg.UNET_FINETUNED_CHECKPOINT if os.path.isfile(cfg.UNET_FINETUNED_CHECKPOINT) else cfg.UNET_PRETRAINED_CHECKPOINT
    if os.path.isfile(ckpt_path):
        model = Configurable1DUNet(cfg=cfg)
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        norm_in, mean_val, std_val = normalize_signal(ext["active"])
        x = to_tensor(norm_in)
        with torch.no_grad():
            den_norm = model(x).squeeze().numpy()
        denoised = denormalize_signal(den_norm, mean_val, std_val)

        c_den = np.correlate(denoised, replica, mode="full")
        shift_model = int(np.argmax(c_raw)) - int(np.argmax(c_den))
        assert shift_model == 0, f"Expected 0 shift with trained model, got {shift_model}"


