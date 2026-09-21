#!/usr/bin/env python3
"""Write excluded_overlap_ids.txt and train_deoverlapped.json."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hyperseg_h.cli import main

if __name__ == '__main__':
    main(['audit-ids', *sys.argv[1:]])
