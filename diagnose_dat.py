# diagnose_dat.py
# ----------------
# Run this in IDLE (F5) to inspect what's actually inside your .dat files.
# It will show the first 20 lines AND check if the file is binary.

import os

# *** POINT THIS AT ONE OF YOUR ACTUAL .dat FILES ***
FILE_PATH = r"C:\Users\HARSHA\Documents\nstl\signal_generation\Project\Noise\PRI_01\channel001.dat"

print("=" * 60)
print(f"File: {FILE_PATH}")
print(f"Size: {os.path.getsize(FILE_PATH)} bytes")
print("=" * 60)

# --- Check if binary ---
with open(FILE_PATH, "rb") as f:
    raw = f.read(512)

null_bytes = raw.count(b'\x00')
print(f"\nFirst 512 bytes contain {null_bytes} null bytes.")
if null_bytes > 0:
    print(">>> FILE APPEARS TO BE BINARY (not plain text)")
else:
    print(">>> File appears to be text-based")

# --- Show first 20 lines as text ---
print("\n--- First 20 lines (text mode) ---")
try:
    with open(FILE_PATH, "r", errors="replace") as f:
        for i, line in enumerate(f):
            if i >= 20:
                break
            print(f"  line {i+1:3d}: {repr(line.rstrip())}")
except Exception as e:
    print(f"Could not open as text: {e}")

# --- Show first 64 raw bytes as hex ---
print("\n--- First 64 bytes (hex) ---")
print(" ".join(f"{b:02x}" for b in raw[:64]))

print("\n--- First 64 bytes (repr) ---")
print(repr(raw[:64]))
