#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 || $# -gt 5 ]]; then
  echo "Usage: $0 PREPARED MODEL MODEL_LABEL OUTPUT_DIR [BATCH_SIZE]" >&2
  exit 2
fi

prepared=$1
model=$2
model_label=$3
output_dir=$4
batch_size=${5:-4}
python_bin=/home/xy/miniconda3/envs/cs310/bin/python
script=/home/xy/projects/MDPI-Entropy-experiments/gutenberg_benchmark/score_gutenberg_windows.py
validator=/home/xy/projects/MDPI-Entropy-experiments/gutenberg_benchmark/validate_score_chunk.py

mkdir -p "$output_dir/logs"
if "$python_bin" "$validator" \
  --prepared "$prepared" \
  --model "$model" \
  --model-label "$model_label" \
  --output-dir "$output_dir" \
  --world-size 8 \
  --require-success \
  >"$output_dir/checkpoint-validation.log" 2>&1; then
  echo "complete checkpoint: $output_dir"
  exit 0
fi

pids=()
for rank in $(seq 0 7); do
  CUDA_VISIBLE_DEVICES=$rank "$python_bin" "$script" \
    --prepared "$prepared" \
    --model "$model" \
    --model-label "$model_label" \
    --output-dir "$output_dir" \
    --rank "$rank" \
    --world-size 8 \
    --batch-size "$batch_size" \
    --device cuda:0 \
    --resume \
    >"$output_dir/logs/rank${rank}.log" 2>&1 &
  pids+=("$!")
done

status=0
for rank in $(seq 0 7); do
  if ! wait "${pids[$rank]}"; then
    echo "rank $rank failed; see $output_dir/logs/rank${rank}.log" >&2
    status=1
  fi
done
if [[ "$status" -ne 0 ]]; then
  exit "$status"
fi

"$python_bin" "$validator" \
  --prepared "$prepared" \
  --model "$model" \
  --model-label "$model_label" \
  --output-dir "$output_dir" \
  --world-size 8 \
  --write-success \
  >"$output_dir/checkpoint-validation.log" 2>&1
echo "sealed checkpoint: $output_dir/_SUCCESS.json"
