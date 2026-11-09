"""
One-Click IDLE (F5) & GUI-Friendly Launcher for Batch PRI Correlation Analysis.

Run directly in Python IDLE by pressing F5, or from the terminal:
    python run_batch_analysis.py

Prompts for the PRI folder directory if not specified, with intelligent auto-detection
of office laptop / NSTL folder paths.
"""

import os
import sys

# Add script directory to sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from batch_pri_correlation_analysis import main

if __name__ == "__main__":
    main()
