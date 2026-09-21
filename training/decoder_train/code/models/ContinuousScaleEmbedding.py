"""Shared implementation used by training and inference."""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[4]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from decoder.models.ContinuousScaleEmbedding import *  # noqa: F401,F403
