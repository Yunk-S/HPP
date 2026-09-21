#!/usr/bin/env python3
"""Run from any working directory: python scripts/hyperseg_h.py --help."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hyperseg_h.cli import main

if __name__ == '__main__':
    main()
