#!/usr/bin/env bash
# Run run_minimal.py for each immediate subdirectory of DATA_ROOT, matching
# .vscode/launch.json "Single Video:Depth Anything V2" (vits, frame_seq, metric, save_npz_separate).
#
# Usage:
#   ./run_minimal_batch.sh [DATA_ROOT]
# Default DATA_ROOT is "data" (relative to this repo).

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
export PYTHONUNBUFFERED=1

DATA_ROOT="${1:-data}"

if [[ ! -d "$DATA_ROOT" ]]; then
  echo "error: not a directory: $DATA_ROOT" >&2
  exit 1
fi

failed=0
shopt -s nullglob
for dir in "$DATA_ROOT"/*/; do
  [[ -d "$dir" ]] || continue
  name="${dir%/}"
  name="${name##*/}"

  in="${DATA_ROOT}/${name}/images"
  out="${DATA_ROOT}/${name}/depth_vda2"

  if [[ ! -d "$in" ]]; then
    echo "skip ${name}: missing ${in}" >&2
    continue
  fi

  echo "=== ${name} ==="
  if ! python -m run_minimal \
    --input_video "$in" \
    --output_dir "$out" \
    --encoder vits \
    --frame_seq \
    --metric \
    --save_npz_separate
  then
    echo "failed: ${name}" >&2
    failed=$((failed + 1))
  fi
done

if (( failed > 0 )); then
  echo "done with ${failed} failure(s)" >&2
  exit 1
fi
echo "all runs finished"
