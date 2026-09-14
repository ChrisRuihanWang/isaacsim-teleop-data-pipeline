"""Replay recorded RGB observations and commands as MP4, without a simulator.

This is observation playback, not action re-execution in physics. Requires byte RGB
from current recordings; normalized float images from older datasets need conversion.
"""
import argparse
from pathlib import Path
import sys

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pp_scripts.dataset_schema import CAMERA_KEYS, validate_episode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--legacy-fps", type=float)
    args = parser.parse_args()
    if args.stride < 1:
        parser.error("--stride must be positive")
    if args.output.exists():
        parser.error(f"Output already exists: {args.output}")
    with np.load(args.episode, allow_pickle=False) as data:
        summary = validate_episode(data, legacy_fps=args.legacy_fps)
        if summary["fps"] is None:
            parser.error("Unknown capture rate; supply --legacy-fps")
        if summary["legacy_float_images"]:
            parser.error("Replay requires uint8 RGB; legacy normalized images need explicit conversion")
        cameras = {key: data[key] for key in CAMERA_KEYS}
        actions, states = data["action"], data["state"]
        phases = data["phase"] if "phase" in data else None
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(str(args.output), fps=summary["fps"] / args.stride, codec="libx264") as writer:
        for i in range(0, summary["frames"], args.stride):
            canvas = Image.new("RGB", (768, 320), (17, 24, 39))
            draw = ImageDraw.Draw(canvas)
            for column, key in enumerate(CAMERA_KEYS):
                tile = ImageOps.contain(Image.fromarray(cameras[key][i]), (256, 256))
                canvas.paste(tile, (column * 256 + (256 - tile.width) // 2, 24 + (256 - tile.height) // 2))
                draw.text((column * 256 + 8, 6), key, fill="white")
            phase = str(phases[i]) if phases is not None else "recorded observation"
            draw.text((8, 283), f"{i / summary['fps']:.2f}s | frame {i} | {phase} | gripper width {states[i, -1]:.3f} m", fill="white")
            draw.text((8, 301), "action: " + np.array2string(actions[i], precision=3, suppress_small=True), fill="white")
            writer.append_data(np.asarray(canvas))
    print(f"Saved {args.output}: {summary['frames']} recorded frames, {summary['fps']:g} Hz simulation clock")


if __name__ == "__main__":
    main()
