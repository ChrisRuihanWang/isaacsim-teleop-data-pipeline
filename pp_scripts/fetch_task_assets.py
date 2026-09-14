"""Fetch the pinned fruit/bin assets, verify hashes, then build fruit physics layers.

Run with the Isaac Sim Python environment: python pp_scripts/fetch_task_assets.py
Use --check for an offline integrity check (no writes or USD dependencies).
External files remain local; their source URLs and terms are in assets/task_assets.json.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]


def matches(path, entry):
    if not path.is_file() or path.stat().st_size != entry["size"]:
        return False
    algorithm = "sha256" if "sha256" in entry else "md5"
    return hashlib.new(algorithm, path.read_bytes()).hexdigest() == entry[algorithm]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    manifest = json.loads((ROOT / "assets/task_assets.json").read_text())
    for entry in manifest["files"]:
        path = ROOT / entry["path"]
        if not matches(path, entry):
            if args.check:
                parser.exit(1, f"Missing or changed asset: {entry['path']}\nRun this script without --check to fetch it.\n")
            if path.exists():
                raise FileExistsError(f"Asset differs from manifest; move it aside before fetching: {path}")
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(path.suffix + ".download.tmp")
            request = Request(entry["url"], headers={"User-Agent": "PickAndPlace-AssetSetup/1.0"})
            with urlopen(request, timeout=60) as response, temporary.open("wb") as stream:
                while chunk := response.read(1024 * 1024):
                    stream.write(chunk)
            if not matches(temporary, entry):
                raise ValueError(f"Asset checksum mismatch: {entry['path']}")
            temporary.replace(path)
        print(f"Verified: {entry['path']}")
    if not args.check:
        subprocess.run([sys.executable, str(ROOT / "pp_scripts/prepare_fruit_assets.py")], check=True)
    else:
        for name in ("banana", "apple"):
            if not (ROOT / "assets/fruits" / name / "asset.usda").is_file():
                parser.exit(1, "Fruit physics layers missing; run this script without --check.\n")


if __name__ == "__main__":
    main()
