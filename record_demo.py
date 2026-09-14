"""Keyboard demonstration entry point; run with the Isaac Sim Python environment."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "source"))

from pp_scripts.teleop_collect_rgb_npz import main

if __name__ == "__main__":
    main()
