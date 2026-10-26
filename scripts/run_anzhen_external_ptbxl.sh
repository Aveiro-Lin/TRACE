#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

python_bin="${TRACE_PYTHON:-python3}"
data_root="${ANZHEN_DATA_ROOT:-data/anzhen_external}"
output_dir="${ANZHEN_OUTPUT_DIR:-results/anzhen_external_ptbxl}"
gpu="${TRACE_GPU:-0}"

exec env CUDA_VISIBLE_DEVICES="$gpu" "$python_bin" evaluate_anzhen_external.py \
  --data-root "$data_root" \
  --base-checkpoint ckpt/ptbxl/downstream_base/best.pt \
  --trace-checkpoint ckpt/ptbxl/downstream_refined/best.pt \
  --tokenizer-checkpoint ckpt/ptbxl/tokenizer/ecg_tokenizer_level_2_code_256_len_2250_ratio_0.1_st2.pth \
  --threshold-config data/ptbxl_validation_thresholds.json \
  --output-dir "$output_dir" \
  --models base trace \
  --batch-size "${BATCH_SIZE:-64}" \
  --num-workers "${NUM_WORKERS:-4}" \
  --bootstrap-replicates "${BOOTSTRAP_REPLICATES:-2000}" \
  --bootstrap-seed "${BOOTSTRAP_SEED:-42}" \
  "$@"
