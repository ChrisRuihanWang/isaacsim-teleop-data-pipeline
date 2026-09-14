#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
isaac_python="${ISAAC_PYTHON:-python}"
export PYTHONPATH="$project_dir/source${PYTHONPATH:+:$PYTHONPATH}"
cd "$project_dir"
mode="${1:-validate}"
if [ "$#" -gt 0 ]; then shift; fi
case "$mode" in
  validate)
    # Kit may return zero on early shutdown; also require a fresh passing report.
    mkdir -p "$project_dir/data"
    output_dir="$(mktemp -d "$project_dir/data/local_validation_XXXXXX")"
    "$isaac_python" pp_scripts/validate_local.py --headless "$@" --output "$output_dir"
    "$isaac_python" -c 'import json,sys; from pathlib import Path; p=Path(sys.argv[1]); r=json.loads(p.read_text()); assert r["passed"], r; print("PASS:", p)' "$output_dir/report.json"
    ;;
  assets)
    "$isaac_python" pp_scripts/fetch_task_assets.py "$@"
    ;;
  collect)
    "$isaac_python" record_demo.py "$@"
    ;;
  grasp)
    mkdir -p "$project_dir/data"
    output_dir="$(mktemp -d "$project_dir/data/grasp_validation_XXXXXX")"
    "$isaac_python" pp_scripts/check_fruit_grasp.py --headless "$@" --output "$output_dir"
    "$isaac_python" -c 'import json,sys; from pathlib import Path; p=Path(sys.argv[1]); r=json.loads(p.read_text()); assert r["passed"], r; print("PASS:", p)' "$output_dir/report.json"
    ;;
  *) echo "Usage: bash pp_scripts/local.sh {assets|validate|grasp|collect} [arguments]" >&2; exit 2 ;;
esac
