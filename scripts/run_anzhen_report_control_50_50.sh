#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${TRACE_PYTHON:-python3}"
gpu_id="${TRACE_GPU:-0}"
data_root="${ANZHEN_DATA_ROOT:-data/anzhen_external}"
split_manifest="${ANZHEN_SPLIT_MANIFEST:-$data_root/trace_compat/paper_report_control_split_seed42.csv}"

cd "$repo_root"
export CUDA_VISIBLE_DEVICES="$gpu_id"

exec "$python_bin" finetune_anzhen_report_control.py \
  --data-root "$data_root" \
  --source-checkpoint "${PTBXL_REPORT_CONTROL_CHECKPOINT:-ckpt/ptbxl/report_control/best.pt}" \
  --tokenizer-checkpoint "${PTBXL_TOKENIZER_CHECKPOINT:-ckpt/ptbxl/tokenizer/ecg_tokenizer_level_2_code_256_len_2250_ratio_0.1_st2.pth}" \
  --locked-split-manifest "$split_manifest" \
  --output-dir "${ANZHEN_REPORT_CONTROL_OUTPUT_DIR:-results/anzhen_report_control/split_50_50_seed42}" \
  --seed "${SPLIT_SEED:-42}" \
  --split-trials "${SPLIT_TRIALS:-512}" \
  --epochs "${EPOCHS:-30}" \
  --patience "${PATIENCE:-7}" \
  --min-epochs "${MIN_EPOCHS:-5}" \
  --batch-size "${BATCH_SIZE:-32}" \
  --num-workers "${NUM_WORKERS:-4}" \
  --backbone-lr "${BACKBONE_LR:-1e-5}" \
  --head-lr "${HEAD_LR:-1e-4}" \
  --weight-decay "${WEIGHT_DECAY:-1e-4}" \
  --pos-weight-cap "${POS_WEIGHT_CAP:-10}" \
  --diagnostic-loss-weight "${DIAGNOSTIC_LOSS_WEIGHT:-0.25}" \
  --form-loss-weight "${FORM_LOSS_WEIGHT:-1.0}" \
  --rhythm-loss-weight "${RHYTHM_LOSS_WEIGHT:-1.0}" \
  --target-metric "${TARGET_METRIC:-prompt_macro_auroc}" \
  --bootstrap-replicates "${BOOTSTRAP_REPLICATES:-1000}" \
  "$@"
