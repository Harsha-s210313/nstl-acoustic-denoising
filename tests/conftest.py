import os
import sys

# Ensure both the package root and its parent directory are on sys.path for pytest
tests_dir = os.path.dirname(os.path.abspath(__file__))
pkg_dir = os.path.dirname(tests_dir)
parent_dir = os.path.dirname(pkg_dir)

for path in [pkg_dir, parent_dir]:
    if path not in sys.path:
        sys.path.insert(0, path)
