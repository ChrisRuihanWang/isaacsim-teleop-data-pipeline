"""Encode a README GIF from recorded simulator video using FFmpeg's palette filter."""
import argparse
from pathlib import Path
import subprocess

import imageio_ffmpeg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--speed", type=float, default=3.0)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--width", type=int, default=560)
    args = parser.parse_args()
    if min(args.speed, args.fps, args.width) <= 0:
        parser.error("Speed, FPS and width must be positive")
    if args.output.exists():
        parser.error(f"Output already exists: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    filters = (
        f"[0:v]setpts=PTS/{args.speed},fps={args.fps},scale={args.width}:-1:flags=lanczos,split[a][b];"
        "[a]palettegen=stats_mode=diff:max_colors=128[p];[b][p]paletteuse=dither=bayer:bayer_scale=3"
    )
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-n", "-loglevel", "error", "-i", str(args.video),
                    "-filter_complex", filters, "-loop", "0", str(args.output)], check=True)
    print(f"Saved {args.output} ({args.output.stat().st_size / 1024**2:.1f} MiB, {args.speed:g}x playback)")


if __name__ == "__main__":
    main()
