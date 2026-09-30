import os
import glob
from config import Config
from inference import run_pipeline

folder = r"C:\Users\HARSHA\Documents\pri"
files = sorted(glob.glob(os.path.join(folder, "*.dat")))
cfg = Config(TRIAL_DATA_DIR=folder)

print("=" * 90)
print(r"TESTING ACOUSTIC PIPELINE ON ALL CHANNELS IN C:\Users\HARSHA\Documents\pri")
print("=" * 90)
header = f"{'File':<16} | {'Stage 1 Detector':<32} | {'SNR Improvement':<16} | {'Output Plot'}"
print(header)
print("-" * 90)

for f in files:
    fname = os.path.basename(f)
    res = run_pipeline(source=f, cfg=cfg, save_plot=True, show_plot=False)
    gain_str = f"{res['snr_improvement']:+.2f} dB"
    plot_name = os.path.basename(res['output_plot']) if res.get('output_plot') else 'N/A'
    row = f"{fname:<16} | {res['detector_verdict']:<32} | {gain_str:<16} | {plot_name}"
    print(row)

print("=" * 90)
