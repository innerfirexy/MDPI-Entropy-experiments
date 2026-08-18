#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 || $# -gt 5 ]]; then
  echo "Usage: $0 PREPARED_DIR MODEL MODEL_LABEL OUTPUT_ROOT [BATCH_SIZE]" >&2
  exit 2
fi

prepared_dir=$1
model=$2
model_label=$3
output_root=$4
batch_size=${5:-4}
launcher=/home/xy/projects/MDPI-Entropy-experiments/gutenberg_benchmark/run_8gpu_score.sh

shopt -s nullglob
prepared_chunks=("$prepared_dir"/chunk-*.npz)
if [[ ${#prepared_chunks[@]} -eq 0 ]]; then
  echo "No prepared chunks found in $prepared_dir" >&2
  exit 1
fi

mkdir -p "$output_root"
for prepared in "${prepared_chunks[@]}"; do
  chunk_name=$(basename "$prepared" .npz)
  output_dir="$output_root/$chunk_name"
  echo "[$model_label] scoring $chunk_name"
  "$launcher" "$prepared" "$model" "$model_label" "$output_dir" "$batch_size"
done

echo "[$model_label] all ${#prepared_chunks[@]} chunks complete"
