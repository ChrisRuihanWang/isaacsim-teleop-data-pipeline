"""Inspect and validate NPZ demonstrations without Isaac Sim or LeRobot.

Usage: python scripts/inspect_dataset.py data/teleop_fruit_v1
Old files: add --legacy-fps 30 only when their capture rate is known.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pp_scripts.dataset_schema import validate_episode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    parser.add_argument("--legacy-fps", type=float)
    args = parser.parse_args()
    files = [args.path] if args.path.is_file() else sorted(args.path.glob("ep_*.npz"))
    if not files:
        parser.error(f"No episodes found in {args.path}")
    summaries = []
    signature = None
    for path in files:
        try:
            with np.load(path, allow_pickle=False) as data:
                summary = validate_episode(data, legacy_fps=args.legacy_fps)
            current = (summary["state_dimension"], summary["action_dimension"], summary["cameras"])
            if signature is not None and signature != current:
                raise ValueError("Inconsistent dimensions across episodes")
            signature = current
            summaries.append({"file": path.name, **summary})
        except (ValueError, KeyError, OSError) as exc:
            parser.exit(1, f"Invalid episode {path}: {exc}\n")
    print(json.dumps({"episodes": len(summaries), "frames": sum(s["frames"] for s in summaries),
                      "fps": sorted({s["fps"] for s in summaries if s["fps"] is not None}),
                      "episodes_with_unknown_fps": sum(s["fps"] is None for s in summaries),
                      "state_dimension": signature[0], "action_dimension": signature[1],
                      "camera_shape": signature[2], "details": summaries}, indent=2))


if __name__ == "__main__":
    main()
