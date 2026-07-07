#!/usr/bin/env python3
"""Entry point. Run from the repo root: `python run.py`."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from main import main  # noqa: E402

if __name__ == "__main__":
    main()
