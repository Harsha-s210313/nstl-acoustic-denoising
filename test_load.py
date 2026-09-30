# test_load.py
# -------------
# Open in IDLE and press F5.
# This directly tests load_dat_file on your actual trial file
# and tells us exactly what's failing.

import sys
sys.path.insert(0, r"C:\Users\HARSHA\.gemini\antigravity\scratch\nstl_acoustic")

# *** POINT THIS AT THE FILE THAT FAILED ***
FILE_PATH = r"C:\Users\HARSHA\Documents\nstl\signal_generation\Project\Noise\PRI_01\channel001.dat"

print("=" * 60)
print("TEST 1: Direct float() parsing on first 5 lines")
print("=" * 60)
with open(FILE_PATH, "r", encoding="utf-8", errors="replace") as f:
    for i, line in enumerate(f):
        if i >= 5:
            break
        stripped = line.strip()
        try:
            val = float(stripped)
            print(f"  line {i+1}: {repr(stripped)} → float OK: {val}")
        except ValueError as e:
            print(f"  line {i+1}: {repr(stripped)} → FAILED: {e}")

print()
print("=" * 60)
print("TEST 2: Calling load_dat_file with verbose=True")
print("=" * 60)
from utils import load_dat_file
try:
    arr = load_dat_file(FILE_PATH, expected_length=14113, verbose=True)
    print(f"\nSUCCESS: loaded {len(arr)} samples")
    print(f"First 5 values: {arr[:5]}")
except Exception as e:
    print(f"\nFAILED: {e}")

print()
print("=" * 60)
print("TEST 3: Check utils.py is the right version")
print("=" * 60)
import utils, inspect
src = inspect.getsource(utils.load_dat_file)
if "No parseable values found" in src:
    print("  utils.py is the CORRECT version (has the right error message)")
else:
    print("  *** utils.py might be an OLD version — check __pycache__ ***")
