#!/usr/bin/env python3
"""Evaluate Track A1/A2/C from an audited checkpoint."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hyperseg_h.cli import main

if __name__ == '__main__':
    main(['eval', *sys.argv[1:]])
